#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t length,
                                        dim_t max_position,
                                        bool with_cache = false);

    StorageView build_alibi(dim_t batch_size,
                            dim_t num_heads,
                            dim_t query_max_length,
                            dim_t key_max_length,
                            const StorageView* key_lengths = nullptr);

    class RotaryEmbeddings;

    class MultiHeadAttention : public Layer
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false);
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
                      const StorageView* alibi = nullptr) const;

      bool has_positional_embeddings() const {
        return _relative_position_keys || _relative_attention_bias || _rotary_embeddings;
      }

      static StorageView prepare_length_mask(const StorageView& lengths,
                                             const dim_t num_heads,
                                             const dim_t num_queries,
                                             const bool mask_future = false);

    private:
      const dim_t _num_heads;
      const bool _self_attention;
      const bool _is_decoder;
      const std::vector<Dense> _linear;
      const dim_t _d_model;
      const bool _pre_norm;
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const std::unique_ptr<RotaryEmbeddings> _rotary_embeddings;
      const StorageView* _relative_attention_bias;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_position_values;
      dim_t _maximum_relative_position;
      const float _queries_scale;
    };

    class RotaryEmbeddings {
    public:
      RotaryEmbeddings(const dim_t dim = 0,
                       const dim_t num_initial_positions = 2048,
                       const float base = 10000);

      void apply(StorageView& x, const dim_t offset = 0);

    private:
      void apply_impl(StorageView& x, const dim_t offset);
      void rotate_every_two(StorageView& x, StorageView& y) const;
      void initialize(const dim_t num_positions,
                      const dim_t dim,
                      const Device device,
                      const DataType dtype);

      const dim_t _dim;
      const dim_t _num_initial_positions;
      const float _base;

      StorageView _sin;
      StorageView _cos;
    };

  }
}
