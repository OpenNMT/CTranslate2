#include "ctranslate2/ops/quantize.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T, typename Q>
    void Quantize::quantize(const StorageView& input,
                            StorageView& output,
                            StorageView& scale) const {
      if (_shift_to_uint8)
        throw std::invalid_argument("Shift to uint8_t is not defined on MPS");

      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      mps::quantize(DataTypeToEnum<T>::value,
                    input.data<T>(),
                    output.data<int8_t>(),
                    scale.data<float>(),
                    batch_size,
                    depth,
                    _round_before_cast);
    }

#define DECLARE_QUANTIZE(T) \
    template void Quantize::quantize<Device::MPS, T, int8_t>( \
      const StorageView&, StorageView&, StorageView&) const;
    DECLARE_QUANTIZE(float)
    DECLARE_QUANTIZE(float16_t)
    DECLARE_QUANTIZE(bfloat16_t)
#undef DECLARE_QUANTIZE

  }
}
