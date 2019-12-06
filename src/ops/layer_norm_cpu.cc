#include "ctranslate2/ops/layer_norm.h"

#include <cmath>

#define EPSILON 0.000001f

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView& beta,
                            const StorageView& gamma,
                            const StorageView& input,
                            StorageView& output) const {
      const auto* gamma_data = gamma.data<T>();
      const auto* beta_data = beta.data<T>();
      size_t depth = input.dim(-1);
      size_t batch_size = input.size() / depth;
      #pragma omp parallel for
      for (long long i = 0; i < static_cast<long long>(batch_size); ++i) {
        const auto* x = input.data<T>() + i * depth;
        auto* y = output.data<T>() + i * depth;
        T mean = 0;  // sum(x)/n
        T rstd = 0;  // 1/sqrt(var(x)) where var(x) = sum((x-mean)^2)/n = sum(x^2)/n - mean^2
        for (size_t j = 0; j < depth; ++j) {
          mean += x[j];
          rstd += x[j] * x[j];
        }
        mean /= depth;
        rstd = std::max(rstd / depth - mean * mean, static_cast<T>(0));
        rstd = static_cast<T>(1) / std::sqrt(rstd + EPSILON);
        for (size_t j = 0; j < depth; ++j) {
          y[j] = (x[j] - mean) * rstd * gamma_data[j] + beta_data[j];
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CPU, T>(const StorageView& beta,         \
                                       const StorageView& gamma,        \
                                       const StorageView& input,        \
                                       StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
