#include "ctranslate2/ops/multinomial.h"

namespace ctranslate2 {
  namespace ops {


    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::CANN, T>(const StorageView& input,     \
                                          StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
