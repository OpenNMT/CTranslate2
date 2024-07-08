#include <ctranslate2/ops/gemm.h>

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       const StorageView& scaleAndZero,
                       StorageView& c) const {
    }

    template <>
    void Gemm::convert_weight_to_int4pack<Device::CPU>(const StorageView& a,
                                          StorageView& b,
                                          int32_t innerKTiles) {
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gemm::compute<Device::CPU, int32_t, T>(const StorageView& a,       \
                       const StorageView& b,                            \
                       const StorageView& scaleAndZero,                 \
                       StorageView& c) const;

    DECLARE_IMPL(bfloat16_t)
  }
}