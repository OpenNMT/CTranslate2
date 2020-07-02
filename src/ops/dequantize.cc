#include "ctranslate2/ops/dequantize.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Dequantize::operator()(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      PROFILE("Dequantize");
      output.resize_as(input);

      switch (input.dtype()) {
      case DataType::INT16: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("INT16 dequantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");
        dequantize<Device::CPU, int16_t>(input, scale, output);
        break;
      }

      case DataType::INT8: {
        const dim_t batch_size = input.size() / input.dim(-1);
        if (scale.size() != batch_size)
          throw std::invalid_argument("INT8 dequantization expects per-batch scales");
        DEVICE_DISPATCH(input.device(), (dequantize<D, int8_t>(input, scale, output)));
        break;
      }

      default:
        throw std::invalid_argument("Dequantize: invalid quantized type " + dtype_name(input.dtype())
                                    + ", expected int8 or int16");
      }
    }

    void Dequantize::operator()(const StorageView& c,
                                const StorageView& a_scale,
                                const StorageView& b_scale,
                                const bool transpose_a,
                                const bool transpose_b,
                                StorageView& y) const {
      PROFILE("DequantizeGemmOutput");
      y.resize_as(c);
      DEVICE_DISPATCH(c.device(), dequantize_gemm_output<D>(c,
                                                            a_scale,
                                                            b_scale,
                                                            transpose_a,
                                                            transpose_b,
                                                            y));
    }

  }
}
