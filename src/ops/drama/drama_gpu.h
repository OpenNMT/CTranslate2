#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_CUDA
    void drama_conv_gpu(const StorageView&, const StorageView&, const StorageView*, StorageView&,
                        StorageView&);
    void drama_delta_gpu(StorageView& mixed, const StorageView& a, const StorageView& b,
                         const StorageView& a_log, const StorageView& dt_bias, StorageView& state,
                         dim_t key_heads, dim_t heads, dim_t key_dim, dim_t value_dim,
                         StorageView& core);
    void drama_swigate_gpu(StorageView&, const StorageView&);
    void drama_extract_queries_gpu(const StorageView&, dim_t heads, dim_t depth, StorageView&);
    void drama_pack_attention_gpu(const StorageView& query, const StorageView& keys,
                                  const StorageView& values, StorageView& cache_keys,
                                  StorageView& cache_values, dim_t past, dim_t heads,
                                  dim_t kv_heads, dim_t depth, StorageView& packed_query,
                                  StorageView& packed_keys, StorageView& packed_values,
                                  StorageView& lengths);
    void drama_attention_gate_gpu(const StorageView& context, const StorageView& query, dim_t heads,
                                  dim_t depth, StorageView& attended);
#endif

  }
}
