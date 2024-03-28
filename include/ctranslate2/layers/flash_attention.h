#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/padder.h"
#include "ctranslate2/layers/flash-attention/flash.h"

namespace ctranslate2 {
  namespace layers {

    class RotaryEmbeddings;
    class Alibi;

    class FlashMultiHeadAttention : public Layer
    {
    public:
      FlashMultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false,
                         Alibi* alibi = nullptr);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& queries,
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
                      dim_t offset = 0) const;

      bool has_positional_embeddings() const {
        return _relative_position_keys || _relative_attention_bias;
      }

      bool multi_query() const {
        return _multi_query;
      }

      static StorageView prepare_length_mask(const StorageView& lengths,
                                             const dim_t num_heads,
                                             const dim_t num_queries,
                                             const bool mask_future = false,
                                             const bool multi_query = false);

    private:
      const bool _tensor_parallel;
      const dim_t _num_heads;
      const bool _self_attention;
      const bool _is_decoder;
      const std::vector<Dense> _linear;
      const dim_t _d_model;
      const dim_t _d_head;
      const bool _pre_norm;
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const std::unique_ptr<RotaryEmbeddings> _rotary_embeddings;
      Alibi* _alibi;
      const StorageView* _relative_attention_bias;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_position_values;
      dim_t _maximum_relative_position;
      const float _queries_scale;
      const bool _multi_query;
      const dim_t _num_heads_kv;
      const bool _merge_time_and_head_dims;
      const dim_t _cache_time_dim;
      const dim_t _sliding_window;
    };
  }
}
