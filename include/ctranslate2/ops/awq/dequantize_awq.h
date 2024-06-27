#pragma once

#include "../op.h"

namespace ctranslate2 {
  namespace ops {

    class DequantizeAwq : public Op {
    public:
      DequantizeAwq();

      void operator()(const StorageView& input,
                      const StorageView& scale,
                      const StorageView& zeros,
                      StorageView& output) const;

    private:
      template <Device D, typename InT, typename OutT>
      void dequantize(const StorageView& input,
                      const StorageView& scale,
                      const StorageView& zeros,
                      StorageView& output) const;
    };

  }
}
