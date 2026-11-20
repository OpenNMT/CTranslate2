#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include "ctranslate2/ops/mean.h"
#include "ctranslate2/primitives.h"
#include "mps/kernels.h"

namespace ctranslate2 {
namespace ops {

#define DECLARE_IMPL(T, DTYPE)                                          \
template <>                                                             \
void Mean::compute<Device::MPS, T>(const StorageView& input,            \
                                   const dim_t outer_size,              \
                                   const dim_t axis_size,               \
                                   const dim_t inner_size,              \
                                   const bool get_sum,                  \
                                   StorageView& output) const {         \
  mps::mean(DTYPE,                                                       \
            input.data<T>(),                                             \
            output.data<T>(),                                            \
            outer_size,                                                  \
            axis_size,                                                   \
            inner_size,                                                  \
            get_sum);                                                    \
}

DECLARE_IMPL(float, DataType::FLOAT32)
DECLARE_IMPL(float16_t, DataType::FLOAT16)
DECLARE_IMPL(bfloat16_t, DataType::BFLOAT16)

#undef DECLARE_IMPL

}  // namespace ops
}  // namespace ctranslate2

#endif
#endif
