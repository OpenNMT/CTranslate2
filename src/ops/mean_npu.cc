#include "ctranslate2/ops/mean.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Mean::compute<Device::CANN, T>(const StorageView& input,    \
                                   const dim_t outer_size,      \
                                   const dim_t axis_size,       \
                                   const dim_t inner_size,      \
                                   StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
