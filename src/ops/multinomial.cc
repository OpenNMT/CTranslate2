#include "ctranslate2/ops/multinomial.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Multinomial::Multinomial(dim_t sample_size)
      : _sample_size(sample_size) {
    }

    void Multinomial::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Multinomial");

      Shape output_shape = input.shape();
      output_shape.back() = _sample_size;
      output.resize(output_shape);

      switch (input.dtype()) {
      case DataType::FLOAT:
        DEVICE_DISPATCH(input.device(), (compute<D, float>(input, output)));
        break;
      case DataType::FLOAT16:
        DEVICE_DISPATCH(input.device(), (compute<D, float16_t>(input, output)));
        break;
      default:
        throw std::invalid_argument("Multinomial only supports float types");
      }
    }

  }
}
