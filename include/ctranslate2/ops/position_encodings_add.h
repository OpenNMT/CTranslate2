#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class PositionEncodingsAdd : public Op {
    public:
      void operator()(const StorageView& input,
                      const StorageView& encodings,
                      StorageView& output,
                      const StorageView* offsets = nullptr,
                      const dim_t step = 0) const;

    private:
      template <Device D, typename T>
      void compute(const dim_t step,
                   const StorageView* offsets,
                   const StorageView& input,
                   const StorageView& encodings,
                   StorageView& output) const;
    };

  }
}
