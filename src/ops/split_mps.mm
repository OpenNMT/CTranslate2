#include "ctranslate2/ops/split.h"

#include <numeric>

#include "ctranslate2/primitives.h"
#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t step_size = input.dim(axis) * input.stride(axis);
      const T* in = input.data<T>();

      static const bool use_fused_split = mps::env_flag("CT2_MPS_USE_FUSED_SPLIT", true);
      if (use_fused_split && (outputs.size() == 2 || outputs.size() == 3)) {
        const dim_t outer_size = mps::iteration_size(input, axis);
        std::vector<dim_t> block_sizes;
        block_sizes.reserve(outputs.size());
        bool supported = outer_size > 0;
        const size_t input_bytes = static_cast<size_t>(input.size()) * sizeof(T);
        for (size_t i = 0; i < outputs.size(); ++i) {
          const dim_t block_size = mps::copy_size(*outputs[i], axis);
          block_sizes.emplace_back(block_size);
          const size_t output_bytes = static_cast<size_t>(outputs[i]->size()) * sizeof(T);
          supported = supported
                      && block_size > 0
                      && !mps::ranges_overlap(in,
                                              input_bytes,
                                              outputs[i]->data<T>(),
                                              output_bytes);
          for (size_t j = 0; supported && j < i; ++j) {
            const size_t other_bytes = static_cast<size_t>(outputs[j]->size()) * sizeof(T);
            supported = !mps::ranges_overlap(outputs[j]->data<T>(),
                                             other_bytes,
                                             outputs[i]->data<T>(),
                                             output_bytes);
          }
        }
        supported = supported
                    && std::accumulate(block_sizes.begin(), block_sizes.end(), dim_t(0))
                       == step_size;
        if (supported && outputs.size() == 2) {
          mps::split2(DataTypeToEnum<T>::value,
                      in,
                      outputs[0]->data<T>(), block_sizes[0],
                      outputs[1]->data<T>(), block_sizes[1],
                      outer_size);
          return;
        }
        if (supported && outputs.size() == 3) {
          mps::split3(DataTypeToEnum<T>::value,
                      in,
                      outputs[0]->data<T>(), block_sizes[0],
                      outputs[1]->data<T>(), block_sizes[1],
                      outputs[2]->data<T>(), block_sizes[2],
                      outer_size);
          return;
        }
      }

      for (StorageView* output : outputs) {
        const dim_t copy_size = mps::copy_size(*output, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = mps::iteration_size(*output, axis);
        T* x = output->data<T>();
        for (dim_t i = 0; i < iter_size; ++i)
          primitives<Device::MPS>::copy(in + i * step_size,
                                        x + i * copy_size,
                                        copy_size);
        in += copy_size;
      }
    }

#define DECLARE_SPLIT(T) \
    template void Split::compute<Device::MPS, T>( \
      const StorageView&, std::vector<StorageView*>&) const;
    DECLARE_ALL_TYPES(DECLARE_SPLIT)
#undef DECLARE_SPLIT

  }
}
