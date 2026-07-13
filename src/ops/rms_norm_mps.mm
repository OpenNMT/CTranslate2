#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/primitives.h"
#include "mps/kernels.h"

namespace ctranslate2 {
namespace ops {

#define DECLARE_IMPL(T, DTYPE)                                      \
template <>                                                         \
void RMSNorm::compute<Device::MPS, T>(const StorageView& gamma,     \
                                      const StorageView& input,     \
                                      StorageView& output) const {  \
  const dim_t depth = input.dim(-1);                                \
  const dim_t batch_size = input.size() / depth;                    \
  mps::rms_norm(DTYPE,                                              \
                input.data<T>(),                                    \
                gamma.data<T>(),                                    \
                output.data<T>(),                                   \
                batch_size,                                         \
                depth,                                              \
                _epsilon,                                           \
                _use_residual);                                     \
}

DECLARE_IMPL(float, DataType::FLOAT32)
DECLARE_IMPL(float16_t, DataType::FLOAT16)
DECLARE_IMPL(bfloat16_t, DataType::BFLOAT16)

#undef DECLARE_IMPL

}  // namespace ops
}  // namespace ctranslate2

#endif
#endif
