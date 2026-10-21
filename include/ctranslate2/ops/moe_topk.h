#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Selects the top-k experts and applies softmax to their logits.
    // Ties favor the lowest index; NaNs rank below all numbers and have zero weight.
    // Selected +inf logits share all weight equally. Rows with only -inf/NaN get zero weights.
    class MoeTopK : public Op {
    public:
      explicit MoeTopK(dim_t k);

      void operator()(const StorageView& router_logits, StorageView& expert_indices,
                      StorageView& routing_weights) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& router_logits, StorageView& expert_indices,
                   StorageView& routing_weights) const;

      dim_t _k;
    };

  }
}
