#include "ctranslate2/layers/flash_attention.h"
#include "mps/flash_attention.h"

namespace ctranslate2 {
  namespace layers {
    void FlashMultiHeadAttention::compute_mps(
        const StorageView& queries, const StorageView* lengths,
        StorageView& output, StorageView* cached_keys,
        StorageView* cached_values, StorageView* attention,
        const Padder* padder, bool normalized_attention, dim_t offset) const {
      if (_tensor_parallel)
        throw std::invalid_argument(
            "MPS FlashAttention does not support tensor parallel execution");
      const auto dtype = queries.dtype();
      const auto device = queries.device();
      StorageView normalized(dtype, device), fused(dtype, device);
      const StorageView* input = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, normalized);
        input = &normalized;
      }
      _linear.front()(*input, fused);
      if (padder)
        padder->add_padding(fused);
      const dim_t batch = fused.dim(0), time = fused.dim(1);
      StorageView q(dtype, device), k(dtype, device), v(dtype, device);
      // Qwen3's Q width can differ from the residual/model width.
      ops::Split(2, {_num_heads * _d_head, _num_heads_kv * _d_head,
                     _num_heads_kv * _d_head})(fused, q, k, v);
      q.reshape({batch, time, _num_heads, _d_head});
      k.reshape({batch, time, _num_heads_kv, _d_head});
      v.reshape(k.shape());
      auto norm = [](const std::unique_ptr<const LayerNorm>& layer,
                     StorageView& tensor) {
        if (layer) {
          StorageView result(tensor.dtype(), tensor.device());
          (*layer)(tensor, result);
          tensor = std::move(result);
        }
      };
      norm(_mps_q_norm, q);
      norm(_mps_k_norm, k);
      norm(_mps_v_norm, v);
      if (_rotary_embeddings) {
        // Fully rotate the new rows on the GPU, including incremental decoding.
        // The CUDA-specific deferred-rotation convention is intentionally
        // absent.
        _rotary_embeddings->apply(q, offset);
        _rotary_embeddings->apply(k, offset);
      }
      StorageView context(dtype, device);
      mps::flash_attention(q, k, v, lengths, context, cached_keys,
                           cached_values, attention, normalized_attention,
                           _queries_scale, offset, _sliding_window, _is_decoder,
                           _multi_query, _mps_kv_cache_bits);
      context.reshape({batch, time, _num_heads * _d_head});
      if (padder)
        padder->remove_padding(context);
      _linear.back()(context, output, _layer_norm ? &queries : nullptr);
      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }
  } // namespace layers
} // namespace ctranslate2
