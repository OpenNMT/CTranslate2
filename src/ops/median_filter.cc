#include "ctranslate2/ops/median_filter.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    MedianFilter::MedianFilter(dim_t width)
      : _width(width)
      {
      }

    void MedianFilter::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("MedianFilter");

      const dim_t axis = input.rank() - 1;
      const dim_t axis_size = input.dim(axis);

      output.resize_as(input);

      DEVICE_AND_FLOAT_DISPATCH("MedianFilter", input.device(), input.dtype(),
                                (compute<D, T>(input, axis_size, output)));
    }

  }
}
