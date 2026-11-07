#include "ctranslate2/ops/alibi_add.h"

#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           dim_t alibi_offset,
                           StorageView& output) const {
      mps::alibi_add(DataTypeToEnum<T>::value,
                     input.data<T>(),
                     alibi.data<T>(),
                     output.data<T>(),
                     input.dim(0),
                     input.dim(1),
                     input.dim(2),
                     input.dim(3),
                     alibi.dim(-1),
                     alibi_offset);
    }

#define DECLARE_ALIBI_ADD(T) \
    template void AlibiAdd::compute<Device::MPS, T>( \
      const StorageView&, const StorageView&, dim_t, StorageView&) const;
    DECLARE_ALIBI_ADD(float)
    DECLARE_ALIBI_ADD(float16_t)
    DECLARE_ALIBI_ADD(bfloat16_t)
#undef DECLARE_ALIBI_ADD

  }
}
