#include "ctranslate2/ops/median_filter.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                               dim_t axis_size,
                               StorageView& output) const {
      const dim_t rank = _width / 2;
      if (_width <= 1 || axis_size <= rank) {
        if (&input != &output)
          output.copy_from(input);
        return;
      }
      if ((_width & 1) == 0)
        throw std::invalid_argument("MedianFilter width must be odd");
      if (_width > 129)
        throw std::invalid_argument("MPS MedianFilter width exceeds 129");

      mps::median_filter(DataTypeToEnum<T>::value,
                         input.data<T>(),
                         output.data<T>(),
                         input.size() / axis_size,
                         axis_size,
                         _width);
    }

#define DECLARE_MEDIAN_FILTER(T) \
    template void MedianFilter::compute<Device::MPS, T>( \
      const StorageView&, dim_t, StorageView&) const;
    DECLARE_MEDIAN_FILTER(float)
    DECLARE_MEDIAN_FILTER(float16_t)
    DECLARE_MEDIAN_FILTER(bfloat16_t)
#undef DECLARE_MEDIAN_FILTER

  }
}
