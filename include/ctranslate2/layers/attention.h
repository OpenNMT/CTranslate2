#pragma once

#include "ctranslate2/layers/attention_layer.h"
#include "ctranslate2/padder.h"
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position);

    StorageView make_asymmetric_relative_positions(dim_t queries_length,
                                                   dim_t keys_length,
                                                   dim_t left_max_position,
                                                   dim_t right_max_position);

    class RotaryEmbeddings;
    class Alibi;

    class MultiHeadAttention : public AttentionLayer
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false,
                         Alibi* alibi = nullptr);
      DataType output_type() const override;
      dim_t output_size() const override;
      virtual void operator()(const StorageView& queries,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr,
                      const Padder* queries_padder = nullptr,
                      const Padder* values_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0) const override;

      virtual bool has_positional_embeddings() const override {
            return _relative_position_keys || _relative_attention_bias || _rotary_embeddings || _alibi;
      }

    protected:
      void process_cross_attention(const StorageView& queries,
                                const StorageView& values,
                                StorageView& fused_proj,
                                StorageView& queries_proj,
                                StorageView& keys_proj,
                                StorageView& values_proj,
                                StorageView* cached_keys,
                                StorageView* cached_values,
                                const Padder* queries_padder,
                                const Padder* values_padder,
                                dim_t& beam_size) const;

    private:
      static void split_heads(StorageView& x,
                               dim_t num_heads,
                               const Padder* padder = nullptr,
                               dim_t beam_size = 1);

      static void combine_heads(StorageView& x,
                                 dim_t num_heads,
                                 const Padder* padder = nullptr,
                                 dim_t beam_size = 1);

      void apply_k_norm(StorageView& keys_proj) const;

      void apply_qk_norm(StorageView& queries_proj,
                          StorageView& keys_proj) const;

      const StorageView* _relative_attention_bias;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_asymmetric_position_keys;
      const StorageView* _relative_position_values;
      dim_t _maximum_relative_position;
      dim_t _relative_left_max_position;
      dim_t _relative_right_max_position;
      const bool _merge_time_and_head_dims;
      const dim_t _cache_time_dim;
      std::unique_ptr<const LayerNorm> _q_norm;  // Query normalization
      std::unique_ptr<const LayerNorm> _k_norm;  // Key normalization
    };
  }
}
