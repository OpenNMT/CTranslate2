#pragma once
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace mps {
    // BTHD queries and compact GQA/MQA caches. lengths is the precomputed BHQ
    // exclusive-key mask used by the generic decoder. Prefill/unsupported
    // shapes use existing GPU matmuls; incremental decoding uses native split-K
    // attention.
    void flash_attention(const StorageView& queries, const StorageView& keys,
                         const StorageView& values, const StorageView* lengths,
                         StorageView& output, StorageView* cached_keys,
                         StorageView* cached_values, StorageView* attention,
                         bool normalized_attention, float scale, dim_t offset,
                         dim_t window, bool causal = true,
                         bool interleaved_lengths = false,
                         unsigned kv_cache_bits = 0);
  } // namespace mps
} // namespace ctranslate2
