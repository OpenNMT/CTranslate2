#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include "ctranslate2/ops/layer_norm.h"
#include "ctranslate2/primitives.h"
#include "mps/kernels.h"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <memory>

namespace ctranslate2 {
namespace ops {

static bool use_reference_mps_norms() {
  const char* all = std::getenv("CT2_MPS_REFERENCE_NORMS");
  const char* layer_norm = std::getenv("CT2_MPS_REFERENCE_LAYER_NORM");
  return (all && all[0] != '\0' && all[0] != '0')
         || (layer_norm && layer_norm[0] != '\0' && layer_norm[0] != '0');
}

template <typename T>
static void reference_layer_norm(const StorageView* beta,
                                 const StorageView* gamma,
                                 const StorageView& input,
                                 dim_t axis,
                                 dim_t outer_size,
                                 dim_t axis_size,
                                 dim_t inner_size,
                                 StorageView& output,
                                 float epsilon) {
  StorageView input_cpu = input.to(Device::CPU);
  std::unique_ptr<StorageView> beta_cpu;
  std::unique_ptr<StorageView> gamma_cpu;
  if (beta)
    beta_cpu = std::make_unique<StorageView>(beta->to(Device::CPU));
  if (gamma)
    gamma_cpu = std::make_unique<StorageView>(gamma->to(Device::CPU));
  StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);
  const T* x = input_cpu.data<T>();
  const T* b = beta_cpu ? beta_cpu->data<T>() : nullptr;
  const T* g = gamma_cpu ? gamma_cpu->data<T>() : nullptr;
  T* y = output_cpu.data<T>();
  for (dim_t outer = 0; outer < outer_size; ++outer) {
    for (dim_t inner = 0; inner < inner_size; ++inner) {
      const dim_t base = outer * axis_size * inner_size + inner;
      float sum = 0;
      float sumsq = 0;
      for (dim_t k = 0; k < axis_size; ++k) {
        const float value = static_cast<float>(x[base + k * inner_size]);
        sum += value;
        sumsq += value * value;
      }
      const float mean = sum / axis_size;
      const float variance = std::max(sumsq / axis_size - mean * mean, 0.0f);
      const float scale = 1.0f / std::sqrt(variance + epsilon);
      for (dim_t k = 0; k < axis_size; ++k) {
        float value = (static_cast<float>(x[base + k * inner_size]) - mean) * scale;
        if (g)
          value *= static_cast<float>(g[k]);
        if (b)
          value += static_cast<float>(b[k]);
        y[base + k * inner_size] = static_cast<T>(value);
      }
    }
  }
  output.copy_from(output_cpu);
}

#define DECLARE_IMPL(T, DTYPE)                                                 \
template <>                                                                    \
void LayerNorm::compute<Device::MPS, T>(const StorageView* beta,               \
                                        const StorageView* gamma,              \
                                        const StorageView& input,              \
                                        const dim_t axis,                      \
                                        const dim_t outer_size,                \
                                        const dim_t axis_size,                 \
                                        const dim_t inner_size,                \
                                        StorageView& output) const {           \
  if (use_reference_mps_norms()) {                                             \
    reference_layer_norm<T>(beta, gamma, input, axis, outer_size,              \
                            axis_size, inner_size, output, _epsilon);           \
    return;                                                                    \
  }                                                                            \
  mps::layer_norm(DTYPE,                                                       \
                  input.data<T>(),                                             \
                  gamma ? gamma->data<T>() : nullptr,                          \
                  beta ? beta->data<T>() : nullptr,                            \
                  output.data<T>(),                                            \
                  outer_size,                                                  \
                  axis_size,                                                   \
                  inner_size,                                                  \
                  _epsilon);                                                   \
}

DECLARE_IMPL(float, DataType::FLOAT32)
DECLARE_IMPL(float16_t, DataType::FLOAT16)
DECLARE_IMPL(bfloat16_t, DataType::BFLOAT16)

#undef DECLARE_IMPL

}  // namespace ops
}  // namespace ctranslate2

#endif
#endif
