#include <ctranslate2/ops/gemm.h>

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       const StorageView& scaleAndZero,
                       StorageView& c) const {
      // todo
      throw std::runtime_error("int4mm is not supported for CPU");
    }

    template <>
    void Gemm::convert_weight_to_int4pack<Device::CPU>(const StorageView& a,
                                          StorageView& b,
                                          int32_t innerKTiles) {
      // todo
      throw std::runtime_error("convert_weight_to_int4pack is not supported for CPU");
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