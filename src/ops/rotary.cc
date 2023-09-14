#include "ctranslate2/ops/rotary.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Rotary::Rotary(const dim_t ndims, const bool interleave)
      : _ndims(ndims)
      , _interleave(interleave)
    {
    }

    void Rotary::operator()(const StorageView& input,
                            const StorageView& sin,
                            const StorageView& cos,
                            StorageView& output,
                            const StorageView* offsets,
                            const dim_t step) const {
      PROFILE("Rotary");

      if (offsets) {
        const dim_t batch_size = input.size() / (input.dim(-1) * input.dim(-2));
        if (offsets->size() != batch_size)
          throw std::invalid_argument("Offsets has size "
                                      + std::to_string(offsets->size())
                                      + " which is different than the current batch size "
                                      + std::to_string(batch_size));
      }

      output.resize_as(input);

      DEVICE_AND_FLOAT_DISPATCH("Rotary", input.device(), input.dtype(),
                                (compute<D, T>(step, offsets, input, sin, cos, output)));
    }

  }
}
