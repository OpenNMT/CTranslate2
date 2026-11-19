#include "ctranslate2/ops/dequantize_fp8.h"

#include <stdexcept>

#include "device_dispatch.h"
#include "storage_view_overlap.h"

namespace ctranslate2 {
  namespace ops {

    void DequantizeFP8::operator()(const StorageView& input, const StorageView& scale_inv,
                                   StorageView& output) const {
      PROFILE("DequantizeFP8");
      if (scale_inv.device() != input.device() || output.device() != input.device()
          || input.device_index() != scale_inv.device_index()
          || input.device_index() != output.device_index() || input.dtype() != DataType::INT8
          || scale_inv.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32)
        throw std::invalid_argument(
            "DequantizeFP8 requires INT8 bytes and F32 scales/output on the same device");
      if (input.rank() != 2 || input.dim(0) <= 0 || input.dim(1) <= 0 || scale_inv.rank() != 2
          || scale_inv.shape()
                 != Shape{input.dim(0) / 128 + (input.dim(0) % 128 != 0),
                          input.dim(1) / 128 + (input.dim(1) % 128 != 0)})
        throw std::invalid_argument("DequantizeFP8 requires a matrix and 128x128 block scales");
      for (const auto* source : {&input, &scale_inv}) {
        if (source == &output
            || storage_views_overlap(*source, source->size() * source->item_size(),
                                     output, output.reserved_memory()))
          throw std::invalid_argument("DequantizeFP8 output must not alias inputs");
      }
      const ScopedDeviceSetter device_setter(input.device(), input.device_index());
      DEVICE_DISPATCH(input.device(), (compute<D>(input, scale_inv, output)));
    }

  }
}
