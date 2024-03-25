#include "ctranslate2/ops/rotary.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::CANN, T>(const StorageView&,                \
                                     const StorageView&,                \
                                     const StorageView&,                \
                                     StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
