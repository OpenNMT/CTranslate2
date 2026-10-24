#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Dequantizes E4M3FN bytes using one scale per 128x128 block.
    class DequantizeFP8 : public Op {
    public:
      void operator()(const StorageView& input, const StorageView& scale_inv,
                      StorageView& output) const;

    private:
      template <Device D>
      void compute(const StorageView& input, const StorageView& scale_inv,
                   StorageView& output) const;
    };

  }
}
