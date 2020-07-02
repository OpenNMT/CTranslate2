#include "ctranslate2/ops/quantize.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    const StorageView Quantize::default_int16_scale(static_cast<float>(1000));

    Quantize::Quantize(ScaleType int16_scale_type)
      : _int16_scale_type(int16_scale_type) {
      if (int16_scale_type != ScaleType::GLOBAL && int16_scale_type != ScaleType::PER_LAYER)
        throw std::invalid_argument("INT16 quantization only supports GLOBAL and PER_LAYER scales");
    }

    void Quantize::operator()(const StorageView& input,
                              StorageView& output,
                              StorageView& scale,
                              float shift) const {
      PROFILE("Quantize");
      output.resize_as(input);

      switch (output.dtype()) {
      case DataType::INT16: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("INT16 quantization is only supported on CPU");
        quantize<Device::CPU, int16_t>(input, output, scale, shift);
        break;
      }

      case DataType::INT8: {
        const dim_t depth = input.dim(-1);
        const dim_t batch_size = input.size() / depth;
        scale.resize({batch_size});
        DEVICE_DISPATCH(input.device(), (quantize<D, int8_t>(input, output, scale, shift)));
        break;
      }

      default:
        throw std::invalid_argument("Quantize: invalid quantized type " + dtype_name(output.dtype())
                                    + ", expected int8 or int16");
      }
    }

  }
}
