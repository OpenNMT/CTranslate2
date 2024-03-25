#include "ctranslate2/ops/bias_add.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output) const {
      primitives<D>::add_batch_broadcast(bias.data<T>(),
                                         value.data<T>(),
                                         output.data<T>(),
                                         bias.size(),
                                         value.size(),
                                         _activation_type == nullptr); // if no activation, then synchronize stream here
      if (_activation_type)
        get_activation_op(*_activation_type)(output, output);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    BiasAdd::compute<Device::CANN, T>(const StorageView& value,         \
                                      const StorageView& bias,          \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
