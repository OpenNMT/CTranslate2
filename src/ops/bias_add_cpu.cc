#include "ctranslate2/ops/bias_add.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output,
                          const StorageView* residual) const {
      if (_axis == -1 || _axis == value.rank() - 1) {
        primitives<D>::add_batch_broadcast(bias.data<T>(),
                                          value.data<T>(),
                                          output.data<T>(),
                                          bias.size(),
                                          value.size());
      } else {
        const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
        dim_t width = 1;
        for (dim_t i = axis + 1; i < value.rank(); ++i)
          width *= value.dim(i);

        primitives<D>::add_block_broadcast(bias.data<T>(),
                                          value.data<T>(),
                                          output.data<T>(),
                                          width,
                                          bias.size(),
                                          value.size());
      }
      if (residual)
        Add()(*residual, output, output);
      if (_activation_type)
        get_activation_op(*_activation_type)(output, output);
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    BiasAdd::compute<Device::CPU, T>(const StorageView& value,  \
                                     const StorageView& bias,   \
                                     StorageView& output,       \
                                     const StorageView* residual) const;

    DECLARE_IMPL(float)

  }
}
