#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Selects top-k experts. By default softmax normalizes the selected logits;
    // normalize_selected=false retains probabilities from the full expert softmax.
    // Ties favor the lowest index; NaNs rank below all numbers and have zero weight.
    // Selected +inf logits share all weight equally. Rows with only -inf/NaN get zero weights.
    class MoeTopK : public Op {
    public:
      explicit MoeTopK(dim_t k, bool normalize_selected = true);

      void operator()(const StorageView& router_logits, StorageView& expert_indices,
                      StorageView& routing_weights) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& router_logits, StorageView& expert_indices,
                   StorageView& routing_weights) const;

      dim_t _k;
      bool _normalize_selected;
    };

  }
}
