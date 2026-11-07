#include "ctranslate2/ops/gather.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      if (axis != batch_dims)
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");

      const dim_t copy_size = data.stride(axis);
      const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
      const dim_t batch_size = data.size() / batch_stride;
      const dim_t num_indices = input.size();
      const dim_t num_indices_per_batch = num_indices / batch_size;
      mps::gather(DataTypeToEnum<T>::value,
                  data.data<T>(),
                  input.data<int32_t>(),
                  output.data<T>(),
                  copy_size,
                  batch_stride,
                  num_indices,
                  num_indices_per_batch);
    }

#define DECLARE_GATHER(T) \
    template void Gather::compute<Device::MPS, T>( \
      const StorageView&, const StorageView&, dim_t, dim_t, StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_GATHER)
#undef DECLARE_GATHER

  }
}
