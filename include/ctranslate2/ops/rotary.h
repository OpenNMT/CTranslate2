#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Rotary : public Op {
    public:
      Rotary(const dim_t ndims, const bool interleave);

      void operator()(const StorageView& input,
                      const StorageView& sin,
                      const StorageView& cos,
                      StorageView& output,
                      const StorageView* offsets = nullptr,
                      const dim_t step = 0) const;

    private:
      const dim_t _ndims;
      const bool _interleave;

      template <Device D, typename T>
      void compute(const dim_t step,
                   const StorageView* offsets,
                   const StorageView& input,
                   const StorageView& sin,
                   const StorageView& cos,
                   StorageView& output) const;
    };

  }
}
