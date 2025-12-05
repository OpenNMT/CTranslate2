#pragma once
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MedianFilter : public Op {
    public:
      explicit MedianFilter(dim_t width);
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      const dim_t _width;
      template <Device D, typename T>
      void compute(const StorageView& input, const dim_t axis_size, StorageView& output) const;
    };

  }
}
