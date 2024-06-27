#include <ctranslate2/ops/awq/gemv.h>
#include <ctranslate2/ops/sum.h>
#include <iostream>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void GemvAwq::operator()(const StorageView& a,
                          const StorageView& b,
                          const StorageView& scale,
                          const StorageView& zero,
                          StorageView& c,
                          const StorageView* bias) const {
      PROFILE("Gemv Awq");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      throw std::runtime_error("AWQ Gemv does not support for cuda arch < 7.5");
#else
      if (a.dtype() != DataType::FLOAT16 && b.dtype() != DataType::INT32)
        throw std::invalid_argument("Awq gemm is only supported for float16 input and int32 weight");
      if (a.device() == Device::CPU)
        throw std::invalid_argument("Awq gemm is only supported on GPU");

      if (a.dim(0) > 8) {
        DEVICE_DISPATCH(a.device(), (compute_gemv2<D, float16_t, int>(a, b, scale, zero, c)));
        StorageView tmp(c.dtype(), c.device());
        ops::Sum(0)(c, tmp);
        tmp.squeeze(0);
        c = std::move(tmp);
      }
      else
        DEVICE_DISPATCH(a.device(), (compute_gemv<D, float16_t, int>(a, b, scale, zero, c)));

      apply_bias_and_activation(c, bias, _activation_type);
#endif
    }
  }
}
