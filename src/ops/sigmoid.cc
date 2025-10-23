#include "ctranslate2/ops/sigmoid.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Sigmoid::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Sigmoid");
      DEVICE_AND_FLOAT_DISPATCH("Sigmoid", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
