#include <ctranslate2/ops/awq/gemm.h>

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename In, typename Out>
    void GemmAwq::compute(const StorageView&,
                          const StorageView&,
                          const StorageView&,
                          const StorageView&,
                          StorageView&) const {
      throw std::runtime_error("AWQ gemm is not applied for the cpu");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemmAwq::compute<Device::CPU, T, int>(                             \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
