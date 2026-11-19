#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "mps/kernels.h"
#include "mps/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      if (mps::supports_topk(x.dtype(), _k)) {
        mps::topk(x.dtype(),
                  x.data<DataType>(),
                  values.data<DataType>(),
                  indices.data<int32_t>(),
                  batch_size,
                  depth,
                  _k);
        return;
      }

      mps::record_profile_event(mps::ProfileEvent::TopKCpu);
      mps::record_profile_event(mps::ProfileEvent::CpuFallback);
      mps::synchronize();

      const DataType* x_data = x.data<DataType>();
      DataType* values_data = values.data<DataType>();
      IndexType* indices_data = indices.data<IndexType>();

      if (_k == 1) {
        for (dim_t i = 0; i < batch_size; ++i) {
          const DataType* row = x_data + i * depth;
          const DataType* best = std::max_element(row, row + depth);
          values_data[i] = *best;
          indices_data[i] = static_cast<IndexType>(std::distance(row, best));
        }
      } else {
        std::vector<IndexType> ids(depth);
        for (dim_t i = 0; i < batch_size; ++i) {
          const DataType* row = x_data + i * depth;
          DataType* row_values = values_data + i * _k;
          IndexType* row_indices = indices_data + i * _k;

          std::iota(ids.begin(), ids.end(), IndexType(0));
          std::partial_sort(ids.begin(),
                            ids.begin() + _k,
                            ids.end(),
                            [row](IndexType a, IndexType b) {
                              return row[a] > row[b];
                            });
          for (dim_t j = 0; j < _k; ++j) {
            row_indices[j] = ids[j];
            row_values[j] = row[row_indices[j]];
          }
        }
      }
    }

    template void TopK::compute<Device::MPS, float, int32_t>(
      const StorageView&, StorageView&, StorageView&) const;
    template void TopK::compute<Device::MPS, float16_t, int32_t>(
      const StorageView&, StorageView&, StorageView&) const;
    template void TopK::compute<Device::MPS, bfloat16_t, int32_t>(
      const StorageView&, StorageView&, StorageView&) const;

  }
}
