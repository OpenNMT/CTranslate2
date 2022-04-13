#include "ctranslate2/ops/gumbel_max.h"

#include <cmath>
#include <limits>

#include "ctranslate2/random.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
      auto& generator = get_random_generator();

      const T* src = x.data<T>();
      T* dst = y.data<T>();

      std::uniform_real_distribution<float> distribution(std::numeric_limits<float>::min(), 1.f);
      for (dim_t i = 0; i < x.size(); ++i)
        dst[i] = src[i] - static_cast<T>(std::log(distribution(generator)));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GumbelMax::add_gumbel_noise<Device::CPU, T>(const StorageView& x,   \
                                                StorageView& y) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
