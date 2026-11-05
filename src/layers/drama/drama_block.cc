#include "layers/drama/drama_block.h"
#include "ctranslate2/ops/rms_norm.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        if (!a.buffer() || !b.buffer() || !a.reserved_memory() || !b.reserved_memory())
          return false;
        const auto x = reinterpret_cast<uintptr_t>(a.buffer());
        const auto y = reinterpret_cast<uintptr_t>(b.buffer());
        return x <= y ? y - x < static_cast<uintptr_t>(a.reserved_memory())
                      : x - y < static_cast<uintptr_t>(b.reserved_memory());
      }

    }

    DramaBlock::DramaBlock(const models::Model& model, const std::string& scope,
                           DramaBlockConfig config, const DramaExecutionOptions& options)
      : _config(config)
      , _input_norm(model.get_variable(scope + "/input_layernorm/weight"))
      , _post_norm(model.get_variable(scope + "/post_attention_layernorm/weight"))
      , _moe(model, scope + "/mlp", config.num_experts, config.top_k, options) {
      if (!std::isfinite(config.norm_epsilon) || config.norm_epsilon <= 0)
        throw std::invalid_argument("DramaBlock requires positive finite RMS epsilon");
      for (const auto* norm : {&_input_norm, &_post_norm}) {
        if (norm->device() != Device::CPU || norm->dtype() != DataType::FLOAT32
            || norm->shape() != Shape{output_size()})
          throw std::invalid_argument("DramaBlock norms require CPU float32 [hidden] weights");
        for (dim_t i = 0; i < norm->size(); ++i) {
          if (!std::isfinite(norm->data<float>()[i]))
            throw std::invalid_argument("DramaBlock norm weights must be finite");
        }
      }
      if (config.kind == DramaMixerKind::LinearAttention)
        _linear = std::make_unique<GatedDeltaNet>(model, scope + "/linear_attn", config.key_heads,
                                                  config.value_heads, config.key_dim,
                                                  config.value_dim, config.norm_epsilon, options);
      else if (config.kind == DramaMixerKind::FullAttention)
        _attention =
            std::make_unique<GatedSelfAttention>(model, scope + "/self_attn", config.attention, options);
      else
        throw std::invalid_argument("DramaBlock mixer kind is invalid");
      if ((_linear && _linear->output_size() != output_size())
          || (_attention && _attention->output_size() != output_size()))
        throw std::invalid_argument("DramaBlock mixer and MoE hidden dimensions must match");
      _resident_gpu = options.resident_gpu();
      if (_resident_gpu) {
        _gpu_device_index = options.device_index;
        const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
        _gpu_input_norm = upload_drama_weight(_input_norm, options, _shared_gpu_parameters);
        _gpu_post_norm = upload_drama_weight(_post_norm, options, _shared_gpu_parameters);
      }
    }

    DataType DramaBlock::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t DramaBlock::output_size() const {
      return _moe.output_size();
    }

    void DramaBlock::initialize_state(DramaBlockState& state, dim_t max_tokens) const {
      if (max_tokens < 0 || (_attention && max_tokens == 0))
        throw std::invalid_argument("DramaBlock attention requires a positive cache capacity");
      const ScopedDeviceSetter setter(_resident_gpu ? Device::CUDA : Device::CPU,
                                      _gpu_device_index);
      DramaBlockState initialized;
      if (_linear)
        _linear->initialize_state(initialized.recurrent, initialized.history);
      else {
        const auto& cfg = _config.attention;
        initialized.attention.keys =
            StorageView({max_tokens, cfg.num_kv_heads, cfg.head_dim}, DataType::FLOAT32);
        initialized.attention.values =
            StorageView({max_tokens, cfg.num_kv_heads, cfg.head_dim}, DataType::FLOAT32);
      }
      if (_resident_gpu && _attention) {
        initialized.attention.keys = initialized.attention.keys.to(Device::CUDA);
        initialized.attention.values = initialized.attention.values.to(Device::CUDA);
      }
      state = std::move(initialized);
    }

    void DramaBlock::forward_gpu(const StorageView& input, DramaBlockState& state,
                                 StorageView& output) const {
      if (input.dtype() != DataType::FLOAT32 || output.device() != Device::CUDA
          || output.dtype() != DataType::FLOAT32 || input.device_index() != _gpu_device_index
          || output.device_index() != _gpu_device_index || input.rank() != 2
          || input.dim(1) != output_size() || &input == &output || overlaps(input, output))
        throw std::invalid_argument("Resident DramaBlock requires compatible distinct GPU tensors");
      for (const auto* cache :
           {&state.recurrent, &state.history, &state.attention.keys, &state.attention.values})
        if (overlaps(input, *cache) || overlaps(output, *cache) || &input == cache
            || &output == cache)
          throw std::invalid_argument("Resident DramaBlock caches must not alias inputs/outputs");
      const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
      if (!input.dim(0)) {
        output.resize_as(input);
        return;
      }
      StorageView normalized(DataType::FLOAT32, Device::CUDA),
          mixed(DataType::FLOAT32, Device::CUDA);
      ops::RMSNorm(_config.norm_epsilon, true)(_gpu_input_norm, input, normalized);
      if (_linear)
        (*_linear)(normalized, state.recurrent, state.history, mixed);
      else
        (*_attention)(normalized, state.attention, mixed);
      ops::Add()(mixed, input, mixed);
      ops::RMSNorm(_config.norm_epsilon, true)(_gpu_post_norm, mixed, normalized);
      StorageView experts(DataType::FLOAT32, Device::CUDA);
      _moe(normalized, experts);
      ops::Add()(experts, mixed, output);
    }

    void DramaBlock::validate_input(const StorageView& input, const DramaBlockState& state,
                                    const StorageView& output) const {
      if (input.device() != Device::CPU || input.dtype() != DataType::FLOAT32
          || output.device() != Device::CPU || output.dtype() != DataType::FLOAT32
          || input.rank() != 2 || input.dim(1) != output_size())
        throw std::invalid_argument("DramaBlock expects CPU float32 [time,hidden] input/output");
      if (&input == &output || overlaps(input, output))
        throw std::invalid_argument("DramaBlock input/output must not alias");
      const StorageView* caches[] = {&state.recurrent, &state.history, &state.attention.keys,
                                     &state.attention.values};
      for (const auto* cache : caches) {
        if (&input == cache || &output == cache || overlaps(input, *cache)
            || overlaps(output, *cache) || overlaps(_input_norm, *cache)
            || overlaps(_post_norm, *cache))
          throw std::invalid_argument(
              "DramaBlock caches must not alias input/output or norm weights");
      }
      if (overlaps(_input_norm, output) || overlaps(_post_norm, output))
        throw std::invalid_argument("DramaBlock output must not alias norm weights");
      for (dim_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input.data<float>()[i]))
          throw std::invalid_argument("DramaBlock input must be finite");
      }
    }

    void DramaBlock::operator()(const StorageView& input, DramaBlockState& state,
                                StorageView& output) const {
      if (_resident_gpu && input.device() == Device::CUDA) {
        forward_gpu(input, state, output);
        return;
      }
      validate_input(input, state, output);
      StorageView normalized, mixed;
      ops::RMSNorm(_config.norm_epsilon, true)(_input_norm, input, normalized);
      if (_linear)
        (*_linear)(normalized, state.recurrent, state.history, mixed);
      else
        (*_attention)(normalized, state.attention, mixed);
      ops::Add()(mixed, input, mixed);
      ops::RMSNorm(_config.norm_epsilon, true)(_post_norm, mixed, normalized);
      StorageView expert_output;
      _moe(normalized, expert_output);
      ops::Add()(expert_output, mixed, expert_output);
      output = std::move(expert_output);
    }

  }
}
