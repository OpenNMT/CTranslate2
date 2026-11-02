#pragma once

#include "drama_projection.h"
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    // Configures attention heads, rotary positions, and query/key normalization.
    struct GatedSelfAttentionConfig {
      dim_t num_heads;
      dim_t num_kv_heads;
      dim_t head_dim;
      dim_t rotary_dim;
      float rope_base = 10000.f;
      float norm_epsilon = 1e-6f;
      bool rotary_interleaved = false;
      bool zero_centered_qk_norm = true;
    };

    // Retains keys and values from earlier tokens for causal decoding.
    // Caller allocates CPU float32 [capacity, kv_heads, head_dim] keys/values.
    // Only [0,length) is initialized/read; reset length=0 to start a sequence.
    struct GatedSelfAttentionCache {
      StorageView keys;
      StorageView values;
      dim_t length = 0;
    };

    // Applies causal self-attention with a learned sigmoid gate on each output feature.
    // Batch-one causal attention with CPU float32 [time, hidden] activations.
    // Model must outlive this layer and keep unpacked matrices unchanged.
    // q_proj rows are per-head [query_d | gate_d]; k/v/o use ordinary Dense
    // layouts. Fused checkpoints must be split/reordered during conversion.
    // q_norm/weight and k_norm/weight each have shape [head_dim].
    class GatedSelfAttention : public Layer {
    public:
      GatedSelfAttention(const models::Model& model, const std::string& scope,
                         GatedSelfAttentionConfig config, const DramaExecutionOptions& options = {});
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, GatedSelfAttentionCache& cache,
                      StorageView& output) const;

    private:
      std::vector<std::shared_ptr<const DramaDeviceWeights>> _shared_gpu_parameters;
      void forward_gpu(const StorageView&, GatedSelfAttentionCache&, StorageView&) const;
      StorageView _gpu_query_norm, _gpu_key_norm;
      GatedSelfAttentionConfig _config;
      DramaProjection _query;
      DramaProjection _key;
      DramaProjection _value;
      DramaProjection _output;
      const StorageView& _query_norm;
      const StorageView& _key_norm;
      dim_t _hidden_size;
      mutable RotaryEmbeddings _rotary;
      std::vector<const StorageView*> _parameters;
    };

  }
}
