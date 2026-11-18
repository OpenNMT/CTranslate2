#include "ctranslate2/layers/moe.h"
#include "ctranslate2/ops/moe_topk.h"
#include "ctranslate2/types.h"
#include "ops/storage_view_overlap.h"
#include <climits>
#ifdef CT2_WITH_MPS
#include "mps/kernels.h"
#include "mps/moe.h"
#endif

namespace ctranslate2 {
  namespace layers {
    struct MixtureOfExperts::Expert {
      Dense gate, up, down;
      Expert(const models::Model& model, const std::string& scope,
             const ops::ActivationType* activation)
          : gate(model, scope + "/linear_0", activation),
            up(model, scope + "/linear_0_noact"),
            down(model, scope + "/linear_1") {}
    };
    MixtureOfExperts::MixtureOfExperts(const models::Model& model,
                                       const std::string& scope)
        : _dtype(get_default_float_type(model.effective_compute_type())),
          _hidden(model.get_variable(scope + "/router/weight").dim(1)),
          _topk(model.get_attribute<int32_t>(scope + "/num_experts_per_token")),
          _normalize_topk(
              model.get_flag_with_default(scope + "/normalize_topk", true)),
          _router(model.get_variable(scope + "/router/weight")
                      .to(DataType::FLOAT32)) {
      if (model.device() != Device::CPU && model.device() != Device::MPS)
        throw std::invalid_argument(
            "Decoder MoE currently supports CPU and MPS devices");
      if (model.tensor_parallel())
        throw std::invalid_argument(
            "Decoder MoE does not support tensor parallelism");
      if (_router.rank() != 2 || _hidden <= 0 || _topk <= 0 ||
          _topk > _router.dim(0))
        throw std::invalid_argument("Invalid decoder MoE router shape or TopK");
      if (model.get_variable_if_exists(scope + "/router/bias"))
        throw std::invalid_argument(
            "Decoder MoE does not support biased routers");
      if (model.device() == Device::CPU && _dtype != DataType::FLOAT32)
        throw std::invalid_argument(
            "CPU decoder MoE requires float32 activations");
      if (model.layer_exists(scope + "/shared_expert")) {
        _shared_expert = std::make_unique<Expert>(
            model, scope + "/shared_expert", &_activation);
        _shared_gate = std::make_unique<Dense>(
            model, scope + "/shared_expert_gate", &_sigmoid);
        if (_shared_expert->down.output_size() != _hidden ||
            _shared_expert->gate.output_size() !=
                _shared_expert->up.output_size() ||
            _shared_gate->output_size() != 1)
          throw std::invalid_argument("Invalid shared MoE expert dimensions");
      } else if (model.layer_exists(scope + "/shared_expert_gate"))
        throw std::invalid_argument("MoE shared gate requires a shared expert");
      dim_t intermediate = 0;
      for (dim_t e = 0; e < _router.dim(0); ++e) {
        const auto expert_scope = scope + "/experts_" + std::to_string(e);
        auto expert =
            std::make_unique<Expert>(model, expert_scope, &_activation);
        if (expert->down.output_size() != _hidden ||
            expert->gate.output_size() != expert->up.output_size() ||
            (intermediate && intermediate != expert->gate.output_size()))
          throw std::invalid_argument(
              "Decoder MoE experts have incompatible dimensions");
        intermediate = expert->gate.output_size();
        const char* projections[] = {
            "/linear_0/weight", "/linear_0_noact/weight", "/linear_1/weight"};
        for (size_t i = 0; i < 3; ++i) {
          // CPU packing can remove the unpacked matrix; Dense retains its own
          // metadata/checks for that case. Validate available raw weights here.
          const auto* raw =
              model.get_variable_if_exists(expert_scope + projections[i]);
          const Shape expected = i == 2 ? Shape{_hidden, intermediate}
                                        : Shape{intermediate, _hidden};
          if (raw && raw->shape() != expected)
            throw std::invalid_argument(
                "Invalid decoder MoE projection dimensions");
        }
        for (const auto* suffix :
             {"/linear_0/bias", "/linear_0_noact/bias", "/linear_1/bias"})
          if (model.get_variable_if_exists(expert_scope + suffix))
            throw std::invalid_argument(
                "Decoder MoE does not support biased expert projections");
        if (model.device() == Device::MPS) {
          std::array<const StorageView*, 3> matrices{}, scales{};
          for (size_t i = 0; i < 3; ++i) {
            const std::string base = expert_scope + projections[i];
            scales[i] = model.get_variable_if_exists(base + "_int4_scale");
            matrices[i] =
                &model.get_variable(base + (scales[i] ? "_int4" : ""));
            const dim_t n = i == 2 ? _hidden : intermediate;
            const dim_t k = i == 2 ? intermediate : _hidden;
            if (matrices[i]->rank() != 2 ||
                matrices[i]->shape() != Shape{n, scales[i] ? (k + 1) / 2 : k} ||
                matrices[i]->dtype() != (scales[i] ? DataType::INT8 : _dtype) ||
                (scales[i] &&
                 (_dtype != DataType::FLOAT16 ||
                  model.get_attribute<int32_t>(base.substr(0, base.size() - 7) +
                                               "/int4_depth") != k)))
              throw std::invalid_argument(
                  "Invalid MPS decoder MoE projection dimensions/format");
          }
          if (bool(scales[0]) != bool(scales[1]))
            throw std::invalid_argument(
                "Paired MoE gate/up weights must use the same format");
          _metal_weights.push_back(matrices);
          _metal_scales.push_back(scales);
        }
        _experts.emplace_back(std::move(expert));
      }
      if (model.layer_exists(scope + "/experts_" +
                             std::to_string(_experts.size())))
        throw std::invalid_argument("MoE router/expert count mismatch");
    }
    MixtureOfExperts::~MixtureOfExperts() = default;

    void MixtureOfExperts::operator()(const StorageView& input,
                                      StorageView& output) const {
      PROFILE("MixtureOfExperts");
      if (input.dtype() != _dtype || input.device() != _router.device() ||
          input.rank() < 2 || input.dim(-1) != _hidden ||
          output.dtype() != _dtype || output.device() != input.device())
        throw std::invalid_argument("Invalid decoder MoE input/output");
      if (&input == &output ||
          ops::storage_views_overlap(input, input.size() * input.item_size(),
                                     output, output.reserved_memory()))
        throw std::invalid_argument("Decoder MoE output must not alias input");
      const dim_t rows = input.size() / _hidden;
      if (rows > INT32_MAX)
        throw std::invalid_argument(
            "Decoder MoE token count exceeds INT32 indices");
      output.resize_as(input);
      if (!rows)
        return;
      StorageView x(input.dtype(), input.device());
      x.view(const_cast<void*>(input.buffer()), {rows, _hidden});
      StorageView logits(DataType::FLOAT32, input.device()),
          ids(DataType::INT32, input.device()),
          weights(DataType::FLOAT32, input.device());
#ifdef CT2_WITH_MPS
      if (input.device() == Device::MPS) {
        logits.resize({rows, _router.dim(0)});
        mps::moe_router(input.dtype(), input.buffer(), _router.data<float>(),
                        logits.data<float>(), rows, _router.dim(0), _hidden);
      } else
#endif
        ops::Gemm(1, 0, false, true)(x, _router, logits);
      ops::MoeTopK{_topk, _normalize_topk}(logits, ids, weights);
#ifdef CT2_WITH_MPS
      if (input.device() == Device::MPS) {
        mps::moe_forward(x, ids, weights, _metal_weights, output, false, false,
                         &_metal_scales);
        output.reshape(input.shape());
        add_shared(input, output);
        return;
      }
#endif
      output.zero();
      for (size_t e = 0; e < _experts.size(); ++e) {
        std::vector<int32_t> tokens, slots;
        for (dim_t r = 0; r < rows; ++r)
          for (dim_t j = 0; j < _topk; ++j)
            if (ids.data<int32_t>()[r * _topk + j] == int32_t(e)) {
              tokens.push_back(r);
              slots.push_back(j);
            }
        if (tokens.empty())
          continue;
        StorageView indices({dim_t(tokens.size())}, tokens), grouped, gate, up,
            result;
        ops::Gather()(x, indices, grouped);
        const auto& expert = *_experts[e];
        expert.gate(grouped, gate);
        expert.up(grouped, up);
        ops::Mul()(gate, up, gate);
        expert.down(gate, result);
        for (size_t i = 0; i < tokens.size(); ++i)
          for (dim_t d = 0; d < _hidden; ++d)
            output.data<float>()[tokens[i] * _hidden + d] +=
                weights.data<float>()[tokens[i] * _topk + slots[i]] *
                result.data<float>()[i * _hidden + d];
      }
      add_shared(input, output);
    }

    void MixtureOfExperts::add_shared(const StorageView& input,
                                      StorageView& output) const {
      if (!_shared_expert)
        return;
      StorageView gate(_dtype, input.device()), up(_dtype, input.device()),
          shared(_dtype, input.device()), weight(_dtype, input.device());
      _shared_expert->gate(input, gate);
      _shared_expert->up(input, up);
      ops::Mul()(gate, up, gate);
      _shared_expert->down(gate, shared);
      (*_shared_gate)(input, weight);
#ifdef CT2_WITH_MPS
      if (input.device() == Device::MPS) {
        mps::moe_add_shared(_dtype, shared.buffer(), weight.buffer(),
                            output.buffer(), input.size() / _hidden, _hidden);
        return;
      }
#endif
      for (dim_t i = 0; i < output.size(); ++i)
        output.data<float>()[i] +=
            shared.data<float>()[i] * weight.data<float>()[i / _hidden];
    }
  } // namespace layers
} // namespace ctranslate2
