#include "gated_self_attention.h"
#include "layers/attention_utils.h"
#include "ops/drama/drama_gpu.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/utils.h"

#include <algorithm>
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

      void require_float_cpu(const StorageView& tensor) {
        if (tensor.device() != Device::CPU || tensor.dtype() != DataType::FLOAT32)
          throw std::invalid_argument(
              "GatedSelfAttention requires CPU float32 activations and cache");
      }

    }

    GatedSelfAttention::GatedSelfAttention(const models::Model& model, const std::string& scope,
                                           GatedSelfAttentionConfig config, const DramaExecutionOptions& options)
      : _config(config)
      , _query(model, scope + "/q_proj", options)
      , _key(model, scope + "/k_proj", options)
      , _value(model, scope + "/v_proj", options)
      , _output(model, scope + "/o_proj", options)
      , _query_norm(model.get_variable(scope + "/q_norm/weight"))
      , _key_norm(model.get_variable(scope + "/k_norm/weight"))
      , _hidden_size(_output.output_size())
      , _rotary(config.rotary_dim, config.rotary_interleaved, RotaryScalingType::None, 1.f,
                config.rope_base, 64) {
      const auto& cfg = _config;
      if (model.device() != Device::CPU || _query.output_type() != DataType::FLOAT32
          || cfg.num_heads <= 0 || cfg.num_kv_heads <= 0 || cfg.head_dim <= 0
          || cfg.num_heads % cfg.num_kv_heads || cfg.rotary_dim < 0 || cfg.rotary_dim > cfg.head_dim
          || cfg.rotary_dim % 2 || !std::isfinite(cfg.rope_base) || cfg.rope_base <= 0
          || !std::isfinite(cfg.norm_epsilon) || cfg.norm_epsilon <= 0 || _hidden_size <= 0)
        throw std::invalid_argument("GatedSelfAttention configuration is incompatible");
      const dim_t widths[] = {cfg.num_heads * cfg.head_dim * 2, cfg.num_kv_heads * cfg.head_dim,
                              cfg.num_kv_heads * cfg.head_dim, _hidden_size};
      size_t i = 0;
      for (const auto* name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        const std::string path = scope + "/" + name;
        const auto& weight = model.get_variable(path + "/weight");
        const dim_t in_width = i == 3 ? cfg.num_heads * cfg.head_dim : _hidden_size;
        if (weight.device() != Device::CPU || weight.shape() != Shape{widths[i], in_width}
            || model.get_variable_if_exists(path + "/weight_packed"))
          throw std::invalid_argument(
              "GatedSelfAttention requires compatible unpacked projection matrices");
        _parameters.push_back(&weight);
        for (const auto* suffix :
             {"/bias", "/weight_scale", "/weight_compensation", "/weight_zero"}) {
          if (const auto* parameter = model.get_variable_if_exists(path + suffix)) {
            if (std::string(suffix) == "/bias" && parameter->shape() != Shape{widths[i]})
              throw std::invalid_argument("GatedSelfAttention bias dimensions are incompatible");
            _parameters.push_back(parameter);
          }
        }
        ++i;
      }
      for (const auto* norm : {&_query_norm, &_key_norm}) {
        require_float_cpu(*norm);
        if (norm->shape() != Shape{cfg.head_dim})
          throw std::invalid_argument(
              "GatedSelfAttention QK norm weights must have shape [head_dim]");
        for (dim_t d = 0; d < cfg.head_dim; ++d) {
          if (!std::isfinite(norm->data<float>()[d]))
            throw std::invalid_argument("GatedSelfAttention QK norm weights must be finite");
        }
        _parameters.push_back(norm);
      }
      if (options.resident_gpu()) {
        const ScopedDeviceSetter setter(Device::CUDA,
                                        options.device_index);
        _gpu_query_norm = upload_drama_weight(_query_norm, options, _shared_gpu_parameters);
        _gpu_key_norm = upload_drama_weight(_key_norm, options, _shared_gpu_parameters);
      }
    }

    DataType GatedSelfAttention::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t GatedSelfAttention::output_size() const {
      return _hidden_size;
    }

    void GatedSelfAttention::forward_gpu(const StorageView& input, GatedSelfAttentionCache& cache,
                                         StorageView& output) const {
#ifdef CT2_WITH_CUDA
      const auto& cfg = _config;
      for (const StorageView* tensor :
           std::initializer_list<const StorageView*>{&input, &cache.keys, &cache.values, &output})
        if (tensor->device() != Device::CUDA || tensor->dtype() != DataType::FLOAT32
            || tensor->device_index() != input.device_index())
          throw std::invalid_argument("Resident attention requires matching GPU float32 tensors");
      if (_gpu_query_norm.empty() || input.rank() != 2 || input.dim(1) != _hidden_size
          || cache.keys.rank() != 3 || cache.keys.dim(1) != cfg.num_kv_heads
          || cache.keys.dim(2) != cfg.head_dim || cache.values.shape() != cache.keys.shape()
          || cache.length < 0 || cache.length > cache.keys.dim(0)
          || input.dim(0) > cache.keys.dim(0) - cache.length || &input == &output
          || overlaps(input, output) || overlaps(input, cache.keys) || overlaps(input, cache.values)
          || overlaps(cache.keys, cache.values) || overlaps(output, cache.keys)
          || overlaps(output, cache.values))
        throw std::invalid_argument("Resident attention shapes/aliasing are incompatible");
      const ScopedDeviceSetter setter(Device::CUDA, input.device_index());
      const dim_t steps = input.dim(0);
      if (!steps) {
        output.resize_as(input);
        return;
      }
      StorageView queries(DataType::FLOAT32, Device::CUDA), keys(DataType::FLOAT32, Device::CUDA),
          values(DataType::FLOAT32, Device::CUDA);
      _query(input, queries);
      _key(input, keys);
      _value(input, values);
      StorageView normalized(DataType::FLOAT32, Device::CUDA);
      ops::drama_extract_queries_gpu(queries, cfg.num_heads, cfg.head_dim, normalized);
      normalized.reshape({1, steps, cfg.num_heads, cfg.head_dim});
      keys.reshape({1, steps, cfg.num_kv_heads, cfg.head_dim});
      ops::RMSNorm norm(cfg.norm_epsilon, cfg.zero_centered_qk_norm);
      norm(_gpu_query_norm, normalized, normalized);
      norm(_gpu_key_norm, keys, keys);
      if (cfg.rotary_dim) {
        StorageView tq(DataType::FLOAT32, Device::CUDA), tk(DataType::FLOAT32, Device::CUDA);
        ops::Transpose({0, 2, 1, 3})(normalized, tq);
        ops::Transpose({0, 2, 1, 3})(keys, tk);
        _rotary.apply(tq, cache.length);
        _rotary.apply(tk, cache.length);
        ops::Transpose({0, 2, 1, 3})(tq, normalized);
        ops::Transpose({0, 2, 1, 3})(tk, keys);
      }
      normalized.reshape({steps, cfg.num_heads, cfg.head_dim});
      keys.reshape({steps, cfg.num_kv_heads * cfg.head_dim});
      StorageView pq(DataType::FLOAT32, Device::CUDA), pk(DataType::FLOAT32, Device::CUDA),
          pv(DataType::FLOAT32, Device::CUDA);
      StorageView lengths(DataType::INT32, Device::CUDA);
      ops::drama_pack_attention_gpu(normalized, keys, values, cache.keys, cache.values,
                                    cache.length, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, pq,
                                    pk, pv, lengths);
      StorageView context(DataType::FLOAT32, Device::CUDA);
      dot_product_attention(pq, pk, pv, &lengths, nullptr, nullptr, nullptr, nullptr, 0, 0, 0,
                            context, nullptr, true, 1.f / std::sqrt(float(cfg.head_dim)));
      StorageView attended(DataType::FLOAT32, Device::CUDA);
      ops::drama_attention_gate_gpu(context, queries, cfg.num_heads, cfg.head_dim, attended);
      _output(attended, output);
      cache.length += steps;
#else
      throw std::runtime_error("Resident attention requires GPU support");
#endif
    }

    void GatedSelfAttention::operator()(const StorageView& input, GatedSelfAttentionCache& cache,
                                        StorageView& output) const {
      if (input.device() == Device::CUDA && !_gpu_query_norm.empty()) {
        forward_gpu(input, cache, output);
        return;
      }
      for (const auto* tensor : {&input, static_cast<const StorageView*>(&cache.keys),
                                 static_cast<const StorageView*>(&cache.values),
                                 static_cast<const StorageView*>(&output)})
        require_float_cpu(*tensor);
      const auto& cfg = _config;
      if (input.rank() != 2 || input.dim(1) != _hidden_size || cache.keys.rank() != 3
          || cache.keys.dim(1) != cfg.num_kv_heads || cache.keys.dim(2) != cfg.head_dim
          || cache.values.shape() != cache.keys.shape() || cache.length < 0
          || cache.length > cache.keys.dim(0) || input.dim(0) > cache.keys.dim(0) - cache.length)
        throw std::invalid_argument(
            "GatedSelfAttention input/cache shape or capacity is incompatible");
      for (const auto* a : {&cache.keys, &cache.values, &output}) {
        if (a == &input || overlaps(input, *a))
          throw std::invalid_argument("GatedSelfAttention input must not alias cache/output");
        for (const auto* parameter : _parameters) {
          if (overlaps(*parameter, *a))
            throw std::invalid_argument(
                "GatedSelfAttention cache/output must not alias model parameters");
        }
      }
      if (&cache.keys == &output || &cache.values == &output || overlaps(cache.keys, cache.values)
          || overlaps(cache.keys, output) || overlaps(cache.values, output))
        throw std::invalid_argument("GatedSelfAttention cache/output tensors must not alias");
      const dim_t steps = input.dim(0);
      const dim_t kv_width = cfg.num_kv_heads * cfg.head_dim;
      for (const auto* tensor : {&input, static_cast<const StorageView*>(&cache.keys),
                                 static_cast<const StorageView*>(&cache.values)}) {
        const dim_t count = tensor == &input ? input.size() : cache.length * kv_width;
        for (dim_t i = 0; i < count; ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::invalid_argument(
                "GatedSelfAttention input and initialized cache prefix must be finite");
        }
      }
      if (steps == 0) {
        output.resize_as(input);
        return;
      }
      StorageView queries, keys, values;
      _query(input, queries);
      _key(input, keys);
      _value(input, values);
      for (const auto* tensor : {&queries, &keys, &values})
        for (dim_t i = 0; i < tensor->size(); ++i)
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::runtime_error(
                "GatedSelfAttention projection overflow; cache was not updated");
      const dim_t q_width = cfg.num_heads * cfg.head_dim;
      StorageView normalized_queries({steps, q_width}, DataType::FLOAT32);
      for (dim_t t = 0; t < steps; ++t) {
        for (dim_t h = 0; h < cfg.num_heads; ++h)
          std::copy_n(queries.data<float>() + (t * cfg.num_heads + h) * 2 * cfg.head_dim,
                      cfg.head_dim,
                      normalized_queries.data<float>() + t * q_width + h * cfg.head_dim);
      }
      normalized_queries.reshape({1, steps, cfg.num_heads, cfg.head_dim});
      keys.reshape({1, steps, cfg.num_kv_heads, cfg.head_dim});
      const ops::RMSNorm norm(cfg.norm_epsilon, cfg.zero_centered_qk_norm);
      norm(_query_norm, normalized_queries, normalized_queries);
      norm(_key_norm, keys, keys);
      if (cfg.rotary_dim) {
        StorageView transposed_queries, transposed_keys;
        ops::Transpose({0, 2, 1, 3})(normalized_queries, transposed_queries);
        ops::Transpose({0, 2, 1, 3})(keys, transposed_keys);
        _rotary.apply(transposed_queries, cache.length);
        _rotary.apply(transposed_keys, cache.length);
        ops::Transpose({0, 2, 1, 3})(transposed_queries, normalized_queries);
        ops::Transpose({0, 2, 1, 3})(transposed_keys, keys);
      }
      keys.reshape({steps, kv_width});
      // Use the same batched MatMul/SoftMax attention implementation as MHA.
      // Pack grouped query heads as additional query rows, avoiding KV replication.
      const dim_t groups = cfg.num_heads / cfg.num_kv_heads;
      const dim_t length = cache.length + steps;
      StorageView packed_queries({1, cfg.num_kv_heads, groups * steps, cfg.head_dim});
      StorageView packed_keys({1, cfg.num_kv_heads, length, cfg.head_dim});
      StorageView packed_values(packed_keys.shape());
      StorageView lengths({cfg.num_kv_heads * groups * steps}, DataType::INT32);
      for (dim_t h = 0; h < cfg.num_kv_heads; ++h) {
        for (dim_t s = 0; s < length; ++s) {
          const auto* key = s < cache.length ? cache.keys.data<float>() + s * kv_width
                                             : keys.data<float>() + (s - cache.length) * kv_width;
          const auto* value = s < cache.length
                                  ? cache.values.data<float>() + s * kv_width
                                  : values.data<float>() + (s - cache.length) * kv_width;
          const dim_t offset = (h * length + s) * cfg.head_dim;
          std::copy_n(key + h * cfg.head_dim, cfg.head_dim, packed_keys.data<float>() + offset);
          std::copy_n(value + h * cfg.head_dim, cfg.head_dim, packed_values.data<float>() + offset);
        }
        for (dim_t g = 0; g < groups; ++g) {
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t row = (h * groups + g) * steps + t;
            std::copy_n(normalized_queries.data<float>() + t * q_width
                            + (h * groups + g) * cfg.head_dim,
                        cfg.head_dim, packed_queries.data<float>() + row * cfg.head_dim);
            lengths.data<int32_t>()[row] = cache.length + t + 1;
          }
        }
      }
      StorageView context;
      dot_product_attention(packed_queries, packed_keys, packed_values, &lengths, nullptr, nullptr,
                            nullptr, nullptr, 0, 0, 0, context, nullptr, true,
                            1.f / std::sqrt(static_cast<float>(cfg.head_dim)));
      StorageView attended({steps, q_width});
      for (dim_t h = 0; h < cfg.num_heads; ++h) {
        for (dim_t t = 0; t < steps; ++t) {
          for (dim_t d = 0; d < cfg.head_dim; ++d) {
            const float gate =
                queries
                    .data<float>()[(t * cfg.num_heads + h) * 2 * cfg.head_dim + cfg.head_dim + d];
            attended.data<float>()[t * q_width + h * cfg.head_dim + d] =
                context.data<float>()[(h * steps + t) * cfg.head_dim + d] / (1.f + std::exp(-gate));
          }
        }
      }
      StorageView result;
      _output(attended, result);
      for (const auto* tensor : {&result, &keys, &values}) {
        for (dim_t i = 0; i < tensor->size(); ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::runtime_error(
                "GatedSelfAttention produced non-finite values; cache was not updated");
        }
      }
      std::copy_n(keys.data<float>(), keys.size(),
                  cache.keys.data<float>() + cache.length * kv_width);
      std::copy_n(values.data<float>(), values.size(),
                  cache.values.data<float>() + cache.length * kv_width);
      cache.length += steps;
      output = std::move(result);
    }

  }
}
