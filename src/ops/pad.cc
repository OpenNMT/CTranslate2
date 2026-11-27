#include "ctranslate2/ops/pad.h"

#include <stdexcept>

namespace ctranslate2 {
  namespace ops {

    Pad1D::Pad1D(dim_t left, dim_t right, Mode mode)
      : _left(left)
      , _right(right)
      , _mode(mode) {
      // TODO: validate left >= 0 and right >= 0.
    }

    void Pad1D::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Pad1D");
      // TODO: validate rank/length (reflect requires left, right < length),
      // and resize output.
      if (input.device() != Device::CPU || output.device() != Device::CPU)
        throw std::invalid_argument("Pad1D only supports CPU");
      if (input.dtype() != DataType::FLOAT32)
        throw std::invalid_argument("Pad1D only supports float32");

      compute<Device::CPU, float>(input, output);
    }

  }
}
