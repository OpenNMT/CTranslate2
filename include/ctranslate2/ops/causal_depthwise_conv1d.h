#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Applies causal depthwise convolution, updating history in place.
    class CausalDepthwiseConv1D : public Op {
    public:
      explicit CausalDepthwiseConv1D(bool apply_silu = true);

      void operator()(const StorageView& input, const StorageView& weight, const StorageView* bias,
                      StorageView& history, StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView& weight, const StorageView* bias,
                   StorageView& history, StorageView& output) const;

      void validate_input(const StorageView& input, const StorageView& weight,
                          const StorageView* bias, const StorageView& history,
                          const StorageView& output) const;

      bool _apply_silu;
    };

  }
}
