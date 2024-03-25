#include "ctranslate2/ops/rms_norm.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");

    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::CANN, T>(const StorageView&, \
                                                    const StorageView&, \
                                                    StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
