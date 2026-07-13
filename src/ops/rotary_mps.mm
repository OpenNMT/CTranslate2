#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include "ctranslate2/ops/rotary.h"
#include "mps/kernels.h"

namespace ctranslate2 {
namespace ops {

#define DECLARE_IMPL(T, DTYPE)                                              \
template <>                                                                 \
void Rotary::compute<Device::MPS, T>(const StorageView& input,              \
                                     const StorageView& sin,                \
                                     const StorageView& cos,                \
                                     StorageView& output,                   \
                                     bool is_transposed) const {            \
  const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);     \
  const dim_t depth = input.dim(-1);                                        \
  const dim_t batch_size = input.size() / (max_time * depth);               \
  const dim_t ndims = _ndims == 0 ? depth : _ndims;                         \
  mps::rotary(DTYPE,                                                        \
              input.data<T>(),                                              \
              sin.data<T>(),                                                \
              cos.data<T>(),                                                \
              output.data<T>(),                                             \
              batch_size,                                                   \
              max_time,                                                     \
              ndims,                                                        \
              depth,                                                        \
              _interleave);                                                 \
}

DECLARE_IMPL(float, DataType::FLOAT32)
DECLARE_IMPL(float16_t, DataType::FLOAT16)
DECLARE_IMPL(bfloat16_t, DataType::BFLOAT16)

#undef DECLARE_IMPL

}  // namespace ops
}  // namespace ctranslate2

#endif
#endif
