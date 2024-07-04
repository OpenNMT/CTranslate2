#include <ctranslate2/ops/awq/gemv.h>

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv(const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               StorageView&) const {
      throw std::runtime_error("AWQ gemv is not applied for the cpu");
    }
    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv2(const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                StorageView&) const {
      throw std::runtime_error("AWQ gemv2 is not applied for the cpu");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemvAwq::compute_gemv2<Device::CPU, T, int>(                       \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    GemvAwq::compute_gemv<Device::CPU, T, int>(                        \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
