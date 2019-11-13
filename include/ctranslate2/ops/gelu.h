#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class GELU : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives<D>::gelu(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
