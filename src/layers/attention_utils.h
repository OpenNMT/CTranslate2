#pragma once

#include "ctranslate2/layers/attention.h"

namespace ctranslate2 {
  namespace layers {

    void dot_product_attention(const StorageView& queries,
                               const StorageView& keys,
                               const StorageView& values,
                               const StorageView* values_lengths,
                               const StorageView* relative_position_keys,
                               const StorageView* relative_asymmetric_position_keys,
                               const StorageView* relative_position_values,
                               const StorageView* relative_attention_bias,
                               dim_t relative_left_max_position,
                               dim_t relative_right_max_position,
                               dim_t maximum_relative_position,
                               StorageView& output,
                               StorageView* attention = nullptr,
                               bool return_normalized_attention = true,
                               float queries_scale = 1,
                               bool is_decoder = false,
                               bool with_cache = false,
                               dim_t beam_size = 1,
                               Alibi* alibi = nullptr,
                               StorageView* position_bias = nullptr);

  }
}
