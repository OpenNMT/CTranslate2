#include "ctranslate2/ops/topp_mask.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
      // TODO: CUDA kernel.
      StorageView output_host;
      operator()(input.to_float32().to(Device::CPU), output_host);
      output = output_host.to(output.dtype()).to(Device::CUDA);
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopPMask::compute<Device::CUDA, T>(const StorageView&, \
                                                     const StorageView&, \
                                                     StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
