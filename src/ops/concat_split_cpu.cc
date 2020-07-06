#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    static dim_t compute_copy_size(const StorageView& x, dim_t axis) {
      dim_t copy_size = 1;
      for (dim_t i = axis; i < x.rank(); ++i)
        copy_size *= x.dim(i);
      return copy_size;
    }

    static dim_t compute_iter_size(const StorageView& x, dim_t axis) {
      dim_t iter_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        iter_size *= x.dim(i);
      return iter_size;
    }

    template <Device D, typename T>
    void Concat::compute(const std::vector<StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t step_size = output.dim(axis) * output.stride(axis);
      T* output_data = output.data<T>();

      for (const StorageView* input : inputs) {
        const StorageView& x = *input;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        const T* x_data = x.data<T>();

        #pragma omp parallel for
        for (dim_t i = 0; i < iter_size; ++i)
          primitives<D>::copy(x_data + i * copy_size, output_data + i * step_size, copy_size);

        output_data += copy_size;  // Copy next input with an offset.
      }
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t step_size = input.dim(axis) * input.stride(axis);
      const T* input_data = input.data<T>();

      for (StorageView* output : outputs) {
        StorageView& x = *output;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        T* x_data = x.data<T>();

        #pragma omp parallel for
        for (dim_t i = 0; i < iter_size; ++i)
          primitives<D>::copy(input_data + i * step_size, x_data + i * copy_size, copy_size);

        input_data += copy_size;  // Read next with an offset.
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Concat::compute<Device::CPU, T>(const std::vector<StorageView*>& inputs, \
                                    StorageView& output) const;         \
    template void                                                       \
    Split::compute<Device::CPU, T>(const StorageView& input,            \
                                   std::vector<StorageView*>& outputs) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
