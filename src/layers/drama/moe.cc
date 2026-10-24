#include "layers/drama/moe.h"
#include "ctranslate2/ops/moe_topk.h"

#include <cmath>
#include <cstring>

#include "env.h"
#include "ops/drama/grouped_moe.h"
#include <cstdint>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps_output(const StorageView& input, const StorageView& output) {
        if (input.empty() || !output.buffer() || output.reserved_memory() == 0)
          return false;
        const auto a = reinterpret_cast<uintptr_t>(input.buffer());
        const auto b = reinterpret_cast<uintptr_t>(output.buffer());
        return a <= b ? b - a < static_cast<uintptr_t>(input.size() * input.item_size())
                      : a - b < static_cast<uintptr_t>(output.reserved_memory());
      }

    }

    SparseMoE::Expert::Expert(const models::Model& model, const std::string& scope,
                              const DramaExecutionOptions& options)
      : gate(model, scope + "/gate_proj", options)
      , up(model, scope + "/up_proj", options)
      , down(model, scope + "/down_proj", options)
      , gpu_device_index(options.device == Device::CUDA ? options.device_index : -1) {
    }

    void SparseMoE::Expert::operator()(const StorageView& input, StorageView& output) const {
      if (gpu_device_index >= 0) {
        const ScopedDeviceSetter setter(Device::CUDA, gpu_device_index);
        StorageView uploaded_input;
        const StorageView* gpu_input = &input;
        if (input.device() != Device::CUDA) {
          uploaded_input = input.to(Device::CUDA);
          gpu_input = &uploaded_input;
        }
        StorageView gated(DataType::FLOAT32, Device::CUDA);
        StorageView values(DataType::FLOAT32, Device::CUDA);
        StorageView activated(DataType::FLOAT32, Device::CUDA);
        StorageView result(DataType::FLOAT32, Device::CUDA);
        gate(*gpu_input, gated);
        up(*gpu_input, values);
        ops::get_activation_op(ops::ActivationType::Swish)(gated, activated);
        ops::Mul()(activated, values, activated);
        down(activated, result);
        if (output.device() == Device::CUDA)
          output = std::move(result);
        else {
          output.copy_from(result);
          synchronize_stream(Device::CUDA);
        }
        return;
      }
      StorageView gated, values, activated;
      gate(input, gated);
      up(input, values);
      ops::get_activation_op(ops::ActivationType::Swish)(gated, activated);
      ops::Mul()(activated, values, activated);
      down(activated, output);
    }

    SparseMoE::SparseMoE(const models::Model& model, const std::string& scope, dim_t num_experts,
                         dim_t k, const DramaExecutionOptions& options)
      : _router(model, scope + "/router", options)
      , _shared_gate(model, scope + "/shared_expert_gate", options)
      , _shared(model, scope + "/shared_expert", options)
      , _topk(k)
      , _input_size(0)
      , _k(k) {
      if (model.device() != Device::CPU || _router.output_type() != DataType::FLOAT32
          || num_experts <= 0 || k > num_experts)
        throw std::invalid_argument(
            "SparseMoE requires CPU float32 activations and 0 < k <= num_experts");

      const auto cache_bytes = options.expert_cache_bytes;
      _resident_gpu = options.resident_gpu();
      _grouped_gpu = _shared.gpu_device_index >= 0
                     && options.compute_type == ComputeType::INT8_FLOAT32
                     && (_resident_gpu || cache_bytes);
      // Requires unpacked /weight variables; the model must disable packing.
      // Dense still handles quantized matrices and their scale metadata.
      auto matrix = [&](const std::string& name) -> const StorageView& {
        const auto& weight = model.get_variable(name + "/weight");
        if (weight.rank() != 2 || weight.device() != Device::CPU)
          throw std::invalid_argument("SparseMoE weights must be rank-2 matrices");
        if (model.get_variable_if_exists(name + "/weight_packed"))
          throw std::invalid_argument("SparseMoE requires model weight prepacking to be disabled");
        const auto* bias = model.get_variable_if_exists(name + "/bias");
        if (bias && bias->shape() != Shape{weight.dim(0)})
          throw std::invalid_argument("SparseMoE projection bias dimension is incompatible");
        _parameters.push_back(&weight);
        for (const auto* suffix : {"/bias", "/weight_scale", "/weight_scale_inv", "/weight_zero",
                                   "/weight_packed", "/weight_compensation"}) {
          if (const auto* value = model.get_variable_if_exists(name + suffix))
            _parameters.push_back(value);
        }
        return weight;
      };
      const auto& router = matrix(scope + "/router");
      _input_size = router.dim(1);
      if (_input_size <= 0 || router.dim(0) != num_experts)
        throw std::invalid_argument("SparseMoE router dimensions must be [num_experts, hidden]");
      const auto& shared_gate = matrix(scope + "/shared_expert_gate");
      if (shared_gate.shape() != Shape{1, _input_size})
        throw std::invalid_argument("SparseMoE shared gate dimensions must be [1, hidden]");
      auto validate_expert = [&](const std::string& name) {
        const auto& gate = matrix(name + "/gate_proj");
        const auto& up = matrix(name + "/up_proj");
        const auto& down = matrix(name + "/down_proj");
        if (gate.dim(0) <= 0 || gate.dim(1) != _input_size || up.shape() != gate.shape()
            || down.shape() != Shape{_input_size, gate.dim(0)})
          throw std::invalid_argument("SparseMoE expert projection dimensions are incompatible");
      };
      validate_expert(scope + "/shared_expert");
      // The fused kernels currently implement the bias-free Drama3 expert layout.
      const auto shared_width =
          model.get_variable(scope + "/shared_expert/gate_proj/weight").dim(0);
      auto supports_grouped = [&](const std::string& name) {
        if (model.get_variable(name + "/gate_proj/weight").dim(0) != shared_width)
          _grouped_gpu = false;
        for (const auto* projection : {"/gate_proj", "/up_proj", "/down_proj"})
          if (model.get_variable_if_exists(name + projection + "/bias"))
            _grouped_gpu = false;
      };
      supports_grouped(scope + "/shared_expert");
      for (dim_t e = 0; e < num_experts; ++e) {
        const auto name = scope + "/experts_" + std::to_string(e);
        validate_expert(name);
        supports_grouped(name);
        _experts.emplace_back(std::make_unique<Expert>(model, name, options));
      }
      if (cache_bytes) {
        if (_shared.gpu_device_index < 0)
          throw std::invalid_argument("Drama3 expert cache requires GPU execution");
        // Reserve for the largest expert, including retained FP8 block scales.
        for (size_t id = 0; id <= _experts.size(); ++id) {
          size_t bytes = 0;
          const auto& expert = cached_expert(id);
          for (const auto* dense : {&expert.gate, &expert.up, &expert.down}) {
            bytes += dense->estimated_device_weight_bytes();
          }
          _cache_expert_bytes = std::max(_cache_expert_bytes, bytes);
        }
        _cache_capacity = std::min<uint64_t>(num_experts + 1, cache_bytes / _cache_expert_bytes);
        if (_cache_capacity < static_cast<size_t>(_k + 1))
          throw std::invalid_argument(
              "CT2_DRAMA3_GPU_EXPERT_CACHE_GB is too small: each layer must fit "
              "the selected experts plus its shared expert");
      }
    }

    const SparseMoE::Expert& SparseMoE::cached_expert(dim_t id) const {
      return id == static_cast<dim_t>(_experts.size()) ? _shared : *_experts[id];
    }

    size_t SparseMoE::gpu_expert_cache_bytes() const {
      size_t bytes = 0;
      for (const auto id : _cache_lru) {
        const auto& expert = cached_expert(id);
        for (const auto* dense : {&expert.gate, &expert.up, &expert.down})
          bytes += dense->device_weight_bytes();
      }
      return bytes;
    }

    void SparseMoE::cache_experts(const int32_t* selected) const {
      // All descriptors from the previous row must finish before eviction.
      // This uses CT2's shared CUDA/HIP stream API on the configured device.
      synchronize_stream(Device::CUDA);
      auto pinned = [&](dim_t id) {
        return id == static_cast<dim_t>(_experts.size())
               || std::find(selected, selected + _k, id) != selected + _k;
      };
      for (dim_t i = 0; i <= _k; ++i) {
        const dim_t id = i == _k ? _experts.size() : selected[i];
        auto entry = std::find(_cache_lru.begin(), _cache_lru.end(), id);
        if (entry != _cache_lru.end()) {
          _cache_lru.splice(_cache_lru.end(), _cache_lru, entry);
          continue;
        }
        if (_cache_lru.size() == _cache_capacity) {
          auto victim = std::find_if(_cache_lru.begin(), _cache_lru.end(), [&](dim_t old) {
            return !pinned(old);
          });
          if (victim == _cache_lru.end())
            throw std::runtime_error("Drama3 expert cache has no evictable entry");
          const auto& expert = cached_expert(*victim);
          for (const auto* dense : {&expert.gate, &expert.up, &expert.down}) {
            dense->release_device_weights();
          }
          _cache_lru.erase(victim);
        }
        const auto& expert = cached_expert(id);
        try {
          for (const auto* dense : {&expert.gate, &expert.up, &expert.down})
            dense->device_weights();
        } catch (...) {
          synchronize_stream(Device::CUDA);
          for (const auto* dense : {&expert.gate, &expert.up, &expert.down}) {
            dense->release_device_weights();
          }
          throw;
        }
        _cache_lru.push_back(id);
        ++_cache_misses;
      }
    }

    void SparseMoE::forward_grouped_gpu(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_CUDA
      const ScopedDeviceSetter setter(Device::CUDA, _shared.gpu_device_index);
      StorageView uploaded_input;
      const StorageView* gpu_input = &input;
      if (input.device() != Device::CUDA) {
        uploaded_input = input.to(Device::CUDA);
        gpu_input = &uploaded_input;
      }
      StorageView logits;
      _router(*gpu_input, logits);
      StorageView indices(DataType::INT32), probabilities;
      _topk(logits, indices, probabilities);
      StorageView gpu_probabilities = probabilities.to(Device::CUDA);
      StorageView result(input.shape(), DataType::FLOAT32, Device::CUDA);
      const dim_t width = _shared.gate.output_size();
      for (dim_t r = 0; r < input.dim(0); ++r) {
        if (_cache_capacity)
          cache_experts(indices.data<int32_t>() + r * _k);
        StorageView row({1, _input_size},
                        const_cast<float*>(gpu_input->data<float>()) + r * _input_size,
                        Device::CUDA);
        StorageView weights({_k}, gpu_probabilities.data<float>() + r * _k, Device::CUDA);
        std::vector<ops::MoeExpertWeights> pointers;
        for (dim_t e = 0; e <= _k; ++e) {
          const auto& expert = e == _k ? _shared : *_experts[indices.data<int32_t>()[r * _k + e]];
          for (const auto* dense : {&expert.gate, &expert.up, &expert.down})
            dense->device_weights();
          pointers.push_back(
              {expert.gate.device_weights().weight.data<int8_t>(), expert.up.device_weights().weight.data<int8_t>(),
               expert.down.device_weights().weight.data<int8_t>(), expert.gate.device_weights().scale.data<float>(),
               expert.up.device_weights().scale.data<float>(), expert.down.device_weights().scale.data<float>()});
        }
        StorageView host_descriptors({dim_t(pointers.size() * sizeof(ops::MoeExpertWeights))},
                                     DataType::INT8);
        std::memcpy(host_descriptors.buffer(), pointers.data(), host_descriptors.size());
        auto descriptors = host_descriptors.to(Device::CUDA);
        StorageView shared_logit(DataType::FLOAT32, Device::CUDA);
        _shared_gate(row, shared_logit);
        StorageView row_result({1, _input_size}, result.data<float>() + r * _input_size,
                               Device::CUDA);
        ops::grouped_moe_gpu(row, descriptors, weights, shared_logit, _input_size, width, _k,
                             row_result);
      }
      output.copy_from(result);
      synchronize_stream(Device::CUDA);
#else
      throw std::runtime_error("Grouped MoE requires the GPU backend");
#endif
    }

    void SparseMoE::forward_individual_gpu(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_CUDA
      // Keep expert activations and reduction on-device for every compute type.
      // Routing stays on the existing CPU top-k path. Shared gating and
      // weighted accumulation stay on-device without a per-row host wait.
      StorageView logits, indices(DataType::INT32), probabilities;
      _router(input, logits);
      _topk(logits, indices, probabilities);
      primitives<Device::CUDA>::fill(output.data<float>(), 0.f, output.size());
      StorageView expert_output(DataType::FLOAT32, Device::CUDA);
      StorageView shared_gate(DataType::FLOAT32, Device::CUDA);
      _shared_gate(input, shared_gate);
      for (dim_t r = 0; r < input.dim(0); ++r) {
        if (_cache_capacity)
          cache_experts(indices.data<int32_t>() + r * _k);
        StorageView row({1, _input_size}, const_cast<float*>(input.data<float>()) + r * _input_size,
                        Device::CUDA);
        StorageView result({1, _input_size}, output.data<float>() + r * _input_size, Device::CUDA);
        for (dim_t i = 0; i < _k; ++i) {
          (*_experts[indices.data<int32_t>()[r * _k + i]])(row, expert_output);
          ops::moe_accumulate_gpu(expert_output, probabilities.data<float>()[r * _k + i], nullptr,
                                  result);
        }
        _shared(row, expert_output);
        ops::moe_accumulate_gpu(expert_output, 1.f, shared_gate.data<float>() + r, result);
      }
#else
      throw std::runtime_error("Resident MoE requires GPU support");
#endif
    }

    void SparseMoE::forward_resident_gpu(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_CUDA
      if (input.dtype() != DataType::FLOAT32 || input.rank() != 2 || input.dim(1) != _input_size
          || output.device() != Device::CUDA || output.dtype() != DataType::FLOAT32
          || input.device_index() != _shared.gpu_device_index
          || output.device_index() != _shared.gpu_device_index || &input == &output
          || overlaps_output(input, output))
        throw std::invalid_argument(
            "Resident MoE requires compatible, distinct GPU float32 tensors");
      const ScopedDeviceSetter setter(Device::CUDA, _shared.gpu_device_index);
      output.resize_as(input);
      if (!input.dim(0))
        return;
      if (_cache_capacity && _grouped_gpu) {
        forward_grouped_gpu(input, output);
        return;
      }
      if (!_grouped_gpu) {
        forward_individual_gpu(input, output);
        return;
      }
      if (_gpu_experts.empty()) {
        // Cache all expert descriptors once so routing never needs host IDs.
        std::vector<ops::MoeExpertWeights> pointers;
        for (size_t e = 0; e <= _experts.size(); ++e) {
          const auto& expert = e == _experts.size() ? _shared : *_experts[e];
          for (const auto* dense : {&expert.gate, &expert.up, &expert.down})
            dense->device_weights();
          pointers.push_back(
              {expert.gate.device_weights().weight.data<int8_t>(), expert.up.device_weights().weight.data<int8_t>(),
               expert.down.device_weights().weight.data<int8_t>(), expert.gate.device_weights().scale.data<float>(),
               expert.up.device_weights().scale.data<float>(), expert.down.device_weights().scale.data<float>()});
        }
        StorageView host({dim_t(pointers.size() * sizeof(ops::MoeExpertWeights))}, DataType::INT8);
        std::memcpy(host.buffer(), pointers.data(), host.size());
        _gpu_experts = host.to(Device::CUDA);
      }
      StorageView logits(DataType::FLOAT32, Device::CUDA),
          shared_logits(DataType::FLOAT32, Device::CUDA);
      _router(input, logits);
      _shared_gate(input, shared_logits);
      StorageView ids(DataType::INT32, Device::CUDA),
          probabilities(DataType::FLOAT32, Device::CUDA);
      _topk(logits, ids, probabilities);
      // Batch prefill without allowing expert activations to grow with context.
      constexpr dim_t max_rows = 32;
      for (dim_t start = 0; start < input.dim(0); start += max_rows) {
        const dim_t count = std::min(max_rows, input.dim(0) - start);
        StorageView rows({count, _input_size},
                         const_cast<float*>(input.data<float>()) + start * _input_size,
                         Device::CUDA);
        StorageView weights({count, _k}, probabilities.data<float>() + start * _k, Device::CUDA);
        StorageView gates({count, 1}, shared_logits.data<float>() + start, Device::CUDA);
        StorageView result({count, _input_size}, output.data<float>() + start * _input_size,
                           Device::CUDA);
        ops::grouped_moe_gpu(rows, _gpu_experts, weights, gates, _input_size,
                             _shared.gate.output_size(), _k, result,
                             ids.data<int32_t>() + start * _k, _experts.size());
      }
#else
      throw std::runtime_error("Resident MoE requires GPU support");
#endif
    }

    DataType SparseMoE::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t SparseMoE::output_size() const {
      return _input_size;
    }

    void SparseMoE::validate_input(const StorageView& input, const StorageView& output) const {
      if (input.device() != Device::CPU || output.device() != Device::CPU
          || input.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32
          || input.rank() != 2 || input.dim(1) != _input_size)
        throw std::invalid_argument("SparseMoE expects CPU float32 [rows, hidden] input/output");
      if (&input == &output || overlaps_output(input, output))
        throw std::invalid_argument("SparseMoE output must not alias input");
      for (const auto* parameter : _parameters) {
        if (overlaps_output(*parameter, output))
          throw std::invalid_argument("SparseMoE output must not alias model parameters");
      }
      for (dim_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input.data<float>()[i]))
          throw std::invalid_argument("SparseMoE input must be finite");
      }
    }

    void SparseMoE::operator()(const StorageView& input, StorageView& output) const {
      if (_resident_gpu && input.device() == Device::CUDA) {
        forward_resident_gpu(input, output);
        return;
      }
      validate_input(input, output);
      const dim_t rows = input.dim(0);
      if (rows == 0) {
        output.resize_as(input);
        return;
      }
      if (_grouped_gpu) {
        forward_grouped_gpu(input, output);
        return;
      }
      StorageView logits, indices(DataType::INT32), weights;
      _router(input, logits);
      _topk(logits, indices, weights);
      StorageView result(input.shape(), 0.f);
      StorageView row({1, _input_size}, DataType::FLOAT32);
      StorageView expert_output, shared_output, shared_gate;
      for (dim_t r = 0; r < rows; ++r) {
        if (_cache_capacity) {
          const ScopedDeviceSetter setter(Device::CUDA, _shared.gpu_device_index);
          cache_experts(indices.data<int32_t>() + r * _k);
        }
        for (dim_t d = 0; d < _input_size; ++d)
          row.data<float>()[d] = input.data<float>()[r * _input_size + d];
        auto* destination = result.data<float>() + r * _input_size;
        for (dim_t i = 0; i < _k; ++i) {
          const auto expert = indices.data<int32_t>()[r * _k + i];
          (*_experts[expert])(row, expert_output);
          const float probability = weights.data<float>()[r * _k + i];
          for (dim_t d = 0; d < _input_size; ++d)
            destination[d] += probability * expert_output.data<float>()[d];
        }
        _shared(row, shared_output);
        _shared_gate(row, shared_gate);
        const float gate = 1.f / (1.f + std::exp(-shared_gate.data<float>()[0]));
        for (dim_t d = 0; d < _input_size; ++d)
          destination[d] += gate * shared_output.data<float>()[d];
      }
      output = std::move(result);
    }

  }
}
