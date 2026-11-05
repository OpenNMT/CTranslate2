#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_CUDA
    void fish_causal_gpu(const StorageView&, const StorageView&, const StorageView&, dim_t dilation,
                         StorageView&);
    // Includes Fish's right crop: output length = input length * stride.
    void fish_transpose_gpu(const StorageView&, const StorageView&, const StorageView&,
                            dim_t stride, StorageView&);
    void fish_tanh_gpu(StorageView&);
#endif

  }
}
