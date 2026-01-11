#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    BiasAdd::BiasAdd(const ActivationType* activation_type, const dim_t axis)
      : _activation_type(activation_type)
      , _axis(axis)
    {
    }

    void BiasAdd::operator()(const StorageView& value,
                             const StorageView& bias,
                             StorageView& output,
                             const StorageView* residual) const {
      PROFILE("BiasAdd");
      output.resize_as(value);

      DEVICE_AND_FLOAT_DISPATCH("BiasAdd", value.device(), value.dtype(),
                                (compute<D, T>(value, bias, output, residual)));
    }

  }
}
