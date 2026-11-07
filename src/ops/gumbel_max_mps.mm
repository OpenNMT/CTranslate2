#include "ctranslate2/ops/gumbel_max.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& input, StorageView& output) const {
      mps::gumbel_noise(DataTypeToEnum<T>::value,
                        input.data<T>(),
                        output.data<T>(),
                        input.size());
    }

#define DECLARE_GUMBEL_MAX(T) \
    template void GumbelMax::add_gumbel_noise<Device::MPS, T>( \
      const StorageView&, StorageView&) const;
    DECLARE_GUMBEL_MAX(float)
    DECLARE_GUMBEL_MAX(float16_t)
    DECLARE_GUMBEL_MAX(bfloat16_t)
#undef DECLARE_GUMBEL_MAX

  }
}
