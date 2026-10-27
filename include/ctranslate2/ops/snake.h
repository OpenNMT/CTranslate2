#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Applies x + sin(alpha*x)^2 / (alpha + 1e-9), with alpha per channel.
    class Snake : public Op {
    public:
      void operator()(const StorageView& x, const StorageView& alpha, StorageView& y) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x, const StorageView& alpha, StorageView& y) const;
    };

  }
}
