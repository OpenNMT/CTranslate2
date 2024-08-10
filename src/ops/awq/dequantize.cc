#include <ctranslate2/ops/awq/dequantize_awq.h>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    DequantizeAwq::DequantizeAwq() = default;

    void DequantizeAwq::operator()(const StorageView& input,
                    const StorageView& scale,
                    const StorageView& zeros,
                    StorageView& output) const{
      PROFILE("Dequantize Awq");

      if (input.dtype() != DataType::INT32 && output.dtype() != DataType::FLOAT16)
        throw std::invalid_argument("Awq dequantization is only supported for int32 input and float16 output");
      if (input.device() == Device::CPU)
        throw std::invalid_argument("Awq dequantization is only supported on GPU");

      DEVICE_DISPATCH(input.device(), (dequantize<D, int, float16_t>(input, scale, zeros, output)));
    }
  }
}
