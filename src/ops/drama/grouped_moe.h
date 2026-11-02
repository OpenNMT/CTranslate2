#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Device pointers to Dense's cached, row-quantized INT8 projections.
    struct MoeExpertWeights {
      const int8_t* gate;
      const int8_t* up;
      const int8_t* down;
      const float* gate_scale;
      const float* up_scale;
      const float* down_scale;
    };
#ifdef CT2_WITH_CUDA
    // Accumulate an expert row, optionally using an on-device shared gate logit.
    void moe_accumulate_gpu(const StorageView& expert, float probability, const float* shared_logit,
                            StorageView& output);
    // Batch of rows, k selected experts plus the shared expert. No host synchronization.
    void grouped_moe_gpu(const StorageView& input, const StorageView& descriptors,
                         const StorageView& routing_weights, const StorageView& shared_logit,
                         dim_t hidden, dim_t intermediate, dim_t k, StorageView& output,
                         const int32_t* indices = nullptr, dim_t shared_index = 0);
#endif

  }
}
