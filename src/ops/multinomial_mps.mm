#include "ctranslate2/ops/multinomial.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      mps::multinomial(DataTypeToEnum<T>::value,
                       input.data<T>(),
                       output.data<int32_t>(),
                       batch_size,
                       depth,
                       _sample_size);
    }

#define DECLARE_MULTINOMIAL(T) \
    template void Multinomial::compute<Device::MPS, T>( \
      const StorageView&, StorageView&) const;
    DECLARE_MULTINOMIAL(float)
    DECLARE_MULTINOMIAL(float16_t)
    DECLARE_MULTINOMIAL(bfloat16_t)
#undef DECLARE_MULTINOMIAL

  }
}
