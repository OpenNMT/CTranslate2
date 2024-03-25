#include "ctranslate2/ops/alibi_add.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           const dim_t alibi_offset,
                           StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    AlibiAdd::compute<Device::CANN, T>(const StorageView& input,        \
                                       const StorageView& alibi,        \
                                       const dim_t alibi_offset,        \
                                       StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
