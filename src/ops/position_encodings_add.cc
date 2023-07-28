#include "ctranslate2/ops/position_encodings_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void PositionEncodingsAdd::operator()(const StorageView& input,
                                          const StorageView& encodings,
                                          StorageView& output,
                                          const StorageView* offsets,
                                          const dim_t step) const {
      PROFILE("PositionEncodingsAdd");

      const dim_t time = input.dim(1);
      const dim_t depth = input.dim(2);
      const dim_t max_time = time + step;

      if (max_time > encodings.dim(0))
        throw std::runtime_error("No position encodings are defined for positions >= "
                                 + std::to_string(encodings.dim(0))
                                 + ", but got position "
                                 + std::to_string(max_time - 1));

      if (depth != encodings.dim(1))
        throw std::invalid_argument("Shape mismatch: position encodings have depth "
                                    + std::to_string(encodings.dim(1))
                                   + ", but the input has depth "
                                    + std::to_string(depth));

      output.resize_as(input);

      DEVICE_AND_FLOAT_DISPATCH(
        "PositionEncodingsAdd", input.device(), input.dtype(),
        ({
          if (offsets)
            compute<D, T>(step, offsets, input, encodings, output);
          else
            primitives<D>::add_batch_broadcast(encodings.data<T>() + step * depth,
                                               input.data<T>(),
                                               output.data<T>(),
                                               time * depth,
                                               input.size());
        }));
    }

  }
}
