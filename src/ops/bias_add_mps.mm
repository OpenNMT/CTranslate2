#include "ctranslate2/ops/bias_add.h"

#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output,
                          const StorageView* residual) const {
      dim_t block = 0;
      if (_axis != -1 && _axis != value.rank() - 1) {
        const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
        for (dim_t i = axis + 1; i < value.rank(); ++i)
          block = (block == 0 ? 1 : block) * value.dim(i);
      }

      mps::bias_add(DataTypeToEnum<T>::value,
                    bias.data<T>(),
                    value.data<T>(),
                    residual ? residual->data<T>() : nullptr,
                    output.data<T>(),
                    bias.size(),
                    value.size(),
                    block,
                    mps::activation_code(_activation_type));
    }

    template void BiasAdd::compute<Device::MPS, float>(
      const StorageView&, const StorageView&, StorageView&, const StorageView*) const;
    template void BiasAdd::compute<Device::MPS, float16_t>(
      const StorageView&, const StorageView&, StorageView&, const StorageView*) const;
    template void BiasAdd::compute<Device::MPS, bfloat16_t>(
      const StorageView&, const StorageView&, StorageView&, const StorageView*) const;

  }
}
