#include "ctranslate2/ops/topp_mask.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

    template<>
    dim_t TopPMask::max_num_classes<Device::CANN>() {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopPMask::compute<Device::CANN, T>(const StorageView&, \
                                                     const StorageView&, \
                                                     StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
