#include "ctranslate2/ops/gumbel_max.h"

#include "type_dispatch.h" 

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GumbelMax::add_gumbel_noise<Device::CANN, T>(const StorageView& x,  \
                                                 StorageView& y) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
