#include "ctranslate2/ops/topp_mask.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probabilities,
                           StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      mps::top_p_mask(DataTypeToEnum<T>::value,
                      input.data<T>(),
                      probabilities.data<T>(),
                      output.data<T>(),
                      batch_size,
                      depth,
                      _p,
                      _mask_value);
    }

#define DECLARE_TOPP_MASK(T) \
    template void TopPMask::compute<Device::MPS, T>( \
      const StorageView&, const StorageView&, StorageView&) const;
    DECLARE_TOPP_MASK(float)
    DECLARE_TOPP_MASK(float16_t)
    DECLARE_TOPP_MASK(bfloat16_t)
#undef DECLARE_TOPP_MASK

    template<>
    dim_t TopPMask::max_num_classes<Device::MPS>() {
      return mps::max_top_p_classes();
    }

  }
}
