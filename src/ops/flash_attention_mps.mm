#include "ctranslate2/ops/flash_attention.h"
#include "ctranslate2/ops/ops.h"
#include "mps/flash_attention.h"
#include "mps/kernels.h"
#include "mps/utils.h"
#include <cmath>

namespace ctranslate2 {
  namespace mps {
    void flash_attention(const StorageView& queries, const StorageView& keys,
                         const StorageView& values, const StorageView* lengths,
                         StorageView& output, StorageView* cached_keys,
                         StorageView* cached_values, StorageView* attention,
                         bool normalized_attention, float scale, dim_t offset,
                         dim_t window, bool causal, bool interleaved_lengths,
                         unsigned kv_cache_bits) {
      const auto dtype = queries.dtype();
      if (queries.device() != Device::MPS || keys.device() != Device::MPS ||
          values.device() != Device::MPS || output.device() != Device::MPS ||
          output.dtype() != dtype ||
          (dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) ||
          keys.dtype() != dtype || values.dtype() != dtype ||
          queries.rank() != 4 || keys.rank() != 4 ||
          keys.shape() != values.shape() || queries.dim(0) != keys.dim(0) ||
          queries.dim(1) != keys.dim(1) || queries.dim(3) != keys.dim(3) ||
          queries.dim(2) <= 0 || keys.dim(2) <= 0 ||
          queries.dim(2) % keys.dim(2) || queries.dim(0) <= 0 ||
          queries.dim(1) <= 0 || queries.dim(3) <= 0 || offset < 0 ||
          offset > INT32_MAX - queries.dim(1) || window < 0 ||
          (window && !causal) || !std::isfinite(scale) ||
          bool(cached_keys) != bool(cached_values) ||
          (cached_keys && (cached_keys == cached_values ||
                           cached_keys == &output || cached_values == &output)))
        throw std::invalid_argument("invalid MPS FlashAttention inputs");
      const dim_t batch = queries.dim(0), qlen = queries.dim(1),
                  heads = queries.dim(2);
      const dim_t kvheads = keys.dim(2), depth = queries.dim(3),
                  valid = offset + qlen;
      if (lengths && (lengths->device() != Device::MPS ||
                      lengths->dtype() != DataType::INT32 ||
                      lengths->size() != batch * heads * qlen))
        throw std::invalid_argument("invalid MPS FlashAttention length mask");
      if (!cached_keys && offset)
        throw std::invalid_argument(
            "MPS FlashAttention offset requires a KV cache");
      const dim_t cache_width =
          kv_cache_bits ? packed_kv_row_bytes(depth, kv_cache_bits) : depth;
      if (kv_cache_bits && dtype != DataType::FLOAT16)
        throw std::invalid_argument(
            "Quantized KV caches require float16 inputs");
      const auto cache_dtype = kv_cache_bits ? DataType::INT8 : dtype;
      // Forward-only calls still quantize K/V, matching incremental semantics.
      StorageView local_k(cache_dtype, Device::MPS),
          local_v(cache_dtype, Device::MPS);
      if (kv_cache_bits && !cached_keys) {
        cached_keys = &local_k;
        cached_values = &local_v;
      }
      auto append = [&](const void* oldk, const void* oldv, StorageView& ck,
                        StorageView& cv, dim_t old_capacity, dim_t capacity,
                        bool copy) {
        if (kv_cache_bits)
          append_quantized_kv(keys.buffer(), values.buffer(), oldk, oldv,
                              ck.buffer(), cv.buffer(), batch, qlen, kvheads,
                              depth, offset, old_capacity, capacity, copy,
                              kv_cache_bits);
        else
          append_compact_kv(keys.buffer(), values.buffer(), oldk, oldv,
                            ck.buffer(), cv.buffer(), batch, qlen, kvheads,
                            depth, offset, old_capacity, capacity, copy);
      };
      const StorageView* k = &keys;
      const StorageView* v = &values;
      dim_t capacity = qlen;
      if (cached_keys) {
        if (cached_keys->empty() != cached_values->empty())
          throw std::invalid_argument("inconsistent MPS KV caches");
        const bool empty = cached_keys->empty();
        if (!empty &&
            (cached_keys->rank() != 4 ||
             cached_keys->shape() != cached_values->shape() ||
             cached_keys->device() != Device::MPS ||
             cached_values->device() != Device::MPS ||
             cached_keys->dtype() != cache_dtype ||
             cached_values->dtype() != cache_dtype ||
             cached_keys->dim(0) != batch || cached_keys->dim(2) != kvheads ||
             cached_keys->dim(3) != cache_width))
          throw std::invalid_argument("incompatible MPS KV cache layout");
        const dim_t old_capacity = empty ? 0 : cached_keys->dim(1);
        if (offset > old_capacity)
          throw std::invalid_argument(
              "MPS KV cache does not contain the requested prefix");
        capacity = old_capacity;
        if (capacity < valid) {
          capacity = ((valid + 511) / 512) * 512;
          StorageView new_k({batch, capacity, kvheads, cache_width},
                            cache_dtype, Device::MPS);
          StorageView new_v(new_k.shape(), cache_dtype, Device::MPS);
          append(empty ? nullptr : cached_keys->buffer(),
                 empty ? nullptr : cached_values->buffer(), new_k, new_v,
                 old_capacity, capacity, true);
          *cached_keys = std::move(new_k);
          *cached_values = std::move(new_v);
        } else {
          append(nullptr, nullptr, *cached_keys, *cached_values, old_capacity,
                 capacity, false);
        }
        k = cached_keys;
        v = cached_values;
      }
      output.resize_as(queries);
      // Small prefills do not amortize a 32-row tile. Preserve the established
      // path there, including its rounding behavior for short cached prompts.
      // Quantized caches use FP32 attention in both prefill and decode: an
      // intermediate FP16 probability rounding can amplify later quantization.
      if ((qlen >= 64 || (kv_cache_bits && qlen > 1)) && !attention &&
          flash_prefill(dtype, queries.buffer(), k->buffer(), v->buffer(),
                        lengths ? lengths->data<int32_t>() : nullptr,
                        output.buffer(), batch, heads, kvheads, qlen, valid,
                        capacity, depth, scale, offset, window, causal,
                        interleaved_lengths, kv_cache_bits))
        return;
      if (qlen == 1 && !attention && depth <= 256) {
        const dim_t first = causal && window && offset + 1 > window
                                ? (offset + 1 - window) / 256 * 256
                                : 0;
        const dim_t splits = (valid - first + 255) / 256;
        StorageView partials(DataType::FLOAT32, Device::MPS);
        if (splits > 1)
          partials.resize({batch, heads, splits, depth + 2});
        if (flash_decode(
                dtype, queries.buffer(), k->buffer(), v->buffer(),
                lengths ? lengths->data<int32_t>() : nullptr, output.buffer(),
                splits > 1 ? partials.data<float>() : nullptr, partials.size(),
                batch, heads, kvheads, valid, capacity, depth, scale, offset,
                window, causal, 1, true, true, kv_cache_bits))
          return;
      }
      // Reference GPU fallback preserves the established FP16/BF16 arithmetic.
      // No replicated KV cache is retained and no full weight dequantization
      // occurs.
      StorageView compact_k(dtype, Device::MPS), compact_v(dtype, Device::MPS);
      if (kv_cache_bits) {
        compact_k.resize({batch, valid, kvheads, depth});
        compact_v.resize(compact_k.shape());
        unpack_quantized_kv(k->buffer(), compact_k.buffer(), batch, valid,
                            kvheads, depth, capacity, kv_cache_bits);
        unpack_quantized_kv(v->buffer(), compact_v.buffer(), batch, valid,
                            kvheads, depth, capacity, kv_cache_bits);
      } else if (capacity == valid) {
        compact_k.view(const_cast<void*>(k->buffer()), k->shape());
        compact_v.view(const_cast<void*>(v->buffer()), v->shape());
      } else {
        ops::Slide(1, 0, valid)(*k, compact_k);
        ops::Slide(1, 0, valid)(*v, compact_v);
      }
      StorageView qt(dtype, Device::MPS), kt(dtype, Device::MPS),
          vt(dtype, Device::MPS);
      ops::Transpose({0, 2, 1, 3})(queries, qt);
      ops::Transpose({0, 2, 1, 3})(compact_k, kt);
      ops::Transpose({0, 2, 1, 3})(compact_v, vt);
      if (heads != kvheads) {
        for (auto* tensor : {&kt, &vt}) {
          tensor->expand_dims(2);
          ops::Tile(2, heads / kvheads)(*tensor);
          tensor->reshape({batch, heads, valid, depth});
        }
      }
      StorageView scores(dtype, Device::MPS),
          bounds({batch, heads, qlen}, DataType::INT32, Device::MPS);
      ops::MatMul(false, true, scale)(qt, kt, scores);
      mask_flash_scores(
          dtype, scores.buffer(), lengths ? lengths->data<int32_t>() : nullptr,
          bounds.data<int32_t>(), batch * heads * qlen, heads, qlen, valid,
          offset, window, causal, interleaved_lengths);
      if (attention && !normalized_attention)
        *attention = scores;
      ops::SoftMax()(scores, bounds, scores);
      if (attention && normalized_attention)
        *attention = scores;
      StorageView context(dtype, Device::MPS);
      ops::MatMul()(scores, vt, context);
      ops::Transpose({0, 2, 1, 3})(context, output);
      record_profile_detail("flash_attention_materialized_fallback");
    }
  } // namespace mps
} // namespace ctranslate2

namespace ctranslate2 {
  namespace ops {
    template <>
    void FlashAttention::compute<Device::MPS>(
        StorageView& queries, StorageView& keys, StorageView& values,
        StorageView& output, StorageView* cached_keys,
        StorageView* cached_values, StorageView* attention, bool normalized,
        StorageView* rotary_cos, StorageView* rotary_sin, bool,
        StorageView* alibi, dim_t offset) const {
      if (rotary_cos || rotary_sin || alibi)
        throw std::invalid_argument(
            "MPS FlashAttention requires pre-rotated Q/K and no ALiBi");
      mps::flash_attention(queries, keys, values, nullptr, output, cached_keys,
                           cached_values, attention, normalized, _queries_scale,
                           offset, _sliding_window, _is_causal);
    }
  } // namespace ops
} // namespace ctranslate2
