#include "ctranslate2/ops/slide.h"

#include "ctranslate2/primitives.h"
#include "mps/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Slide::compute(const StorageView& input,
                        StorageView& output,
                        const dim_t& index) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t stride_axis = input.stride(axis) == 0 ? 1 : input.stride(axis);
      const dim_t step_size = input.dim(axis) * stride_axis;
      const T* in = input.data<T>() + index * stride_axis;
      T* out = output.data<T>();

      const dim_t copy_size = mps::copy_size(output, axis);
      if (copy_size == 0)
        return;
      const dim_t iter_size = mps::iteration_size(output, axis);
      for (dim_t i = 0; i < iter_size; ++i)
        primitives<Device::MPS>::copy(in + i * step_size,
                                      out + i * copy_size,
                                      copy_size);
    }

#define DECLARE_SLIDE(T) \
    template void Slide::compute<Device::MPS, T>( \
      const StorageView&, StorageView&, const dim_t&) const;
    DECLARE_ALL_TYPES(DECLARE_SLIDE)
#undef DECLARE_SLIDE

  }
}
