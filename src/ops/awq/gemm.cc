#include <ctranslate2/ops/awq/gemm.h>
#include <ctranslate2/ops/sum.h>
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void GemmAwq::operator()(const StorageView& a,
                          const StorageView& b,
                          const StorageView& scale,
                          const StorageView& zero,
                          StorageView& c,
                          const StorageView* bias) const {
      PROFILE("Gemm Awq");

      if (a.dtype() != DataType::FLOAT16 && b.dtype() != DataType::INT32)
        throw std::invalid_argument("Awq gemm is only supported for float16 input and int32 weight");
      if (a.device() == Device::CPU)
        throw std::invalid_argument("Awq gemm is only supported on GPU");

      compute<Device::CUDA, float16_t, int>(a, b, scale, zero, c);

      StorageView tmp(c.dtype(), c.device());
      ops::Sum(0)(c, tmp);
      tmp.squeeze(0);
      c = std::move(tmp);

      apply_bias_and_activation(c, bias, _activation_type);
    }
  }
}
