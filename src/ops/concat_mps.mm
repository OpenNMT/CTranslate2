#include "ctranslate2/ops/concat.h"

#include "ctranslate2/primitives.h"
#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t step_size = output.dim(axis) * output.stride(axis);
      T* out = output.data<T>();

      static const bool use_concat2 = mps::env_flag("CT2_MPS_USE_CONCAT2", true);
      if (use_concat2 && inputs.size() == 2) {
        const dim_t a_block_size = mps::copy_size(*inputs[0], axis);
        const dim_t b_block_size = mps::copy_size(*inputs[1], axis);
        const dim_t outer_size = mps::iteration_size(output, axis);
        const size_t output_bytes = static_cast<size_t>(output.size()) * sizeof(T);
        const size_t a_bytes = static_cast<size_t>(inputs[0]->size()) * sizeof(T);
        const size_t b_bytes = static_cast<size_t>(inputs[1]->size()) * sizeof(T);
        if (a_block_size > 0 && b_block_size > 0 && outer_size > 0
            && !mps::ranges_overlap(inputs[0]->data<T>(), a_bytes, out, output_bytes)
            && !mps::ranges_overlap(inputs[1]->data<T>(), b_bytes, out, output_bytes)) {
          mps::concat2(DataTypeToEnum<T>::value,
                       inputs[0]->data<T>(),
                       a_block_size,
                       inputs[1]->data<T>(),
                       b_block_size,
                       out,
                       outer_size);
          return;
        }
      }

      for (const StorageView* input : inputs) {
        const dim_t copy_size = mps::copy_size(*input, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = mps::iteration_size(*input, axis);
        const T* x = input->data<T>();
        for (dim_t i = 0; i < iter_size; ++i)
          primitives<Device::MPS>::copy(x + i * copy_size,
                                        out + i * step_size,
                                        copy_size);
        out += copy_size;
      }
    }

#define DECLARE_CONCAT(T) \
    template void Concat::compute<Device::MPS, T>( \
      const std::vector<const StorageView*>&, StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_CONCAT)
#undef DECLARE_CONCAT

  }
}
