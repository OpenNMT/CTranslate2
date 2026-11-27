#include "ctranslate2/ops/pad.h"

#include <stdexcept>
#include <string>

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Pad1D::Pad1D(dim_t left, dim_t right, Mode mode)
      : _left(left)
      , _right(right)
      , _mode(mode) {
    }

    void Pad1D::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Pad1D");

      if (!(input.rank() > 1))
        throw std::invalid_argument("Pad1D only supports ranks > 1");

      const dim_t length = input.shape().back();
      if (!(_right < length))
        throw std::invalid_argument("Pad1D: right padding (" + std::to_string(_right)
                                    + ") must be smaller than the input length ("
                                    + std::to_string(length) + ")");

      if (!(_left < length))
        throw std::invalid_argument("Pad1D: left padding (" + std::to_string(_left)
                                    + ") must be smaller than the input length ("
                                    + std::to_string(length) + ")");


      if (output.device() != input.device() || output.device_index() != input.device_index())
        throw std::invalid_argument("Pad1D requires input and output on the same device");
      if (input.dtype() != DataType::FLOAT32)
        throw std::invalid_argument("Pad1D only supports float32");

      const ScopedDeviceSetter setter(input.device(), input.device_index());
      DEVICE_DISPATCH(input.device(), (compute<D, float>(input, output)));
    }

  }
}
