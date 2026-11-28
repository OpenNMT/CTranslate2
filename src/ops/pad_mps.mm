#include "ctranslate2/ops/pad.h"

#include "mps/kernels.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Pad1D::compute(const StorageView& input, StorageView& output) const {
      Shape out_shape = input.shape();
      out_shape.back() += _left + _right;
      output.resize(std::move(out_shape));

      if (output.empty())
        return;

      const dim_t length = input.dim(-1);
      mps::pad1d(input.data<T>(),
                 output.data<T>(),
                 input.size() / length,
                 length,
                 _left,
                 _right);
    }

    template void Pad1D::compute<Device::MPS, float>(const StorageView&,
                                                    StorageView&) const;

  }
}
