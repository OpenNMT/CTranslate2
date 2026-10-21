#include "ctranslate2/ops/snake.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Snake::compute(const StorageView& x, const StorageView& alpha, StorageView& y) const {
      const dim_t channels = x.dim(1);
      const dim_t time = x.dim(2);
      const auto* data = x.data<T>();
      const auto* coefficients = alpha.data<T>();
      constexpr float epsilon = 1e-9f;
      for (dim_t c = 0; c < channels; ++c) {
        if (!std::isfinite(coefficients[c]) || coefficients[c] + epsilon == 0.f)
          throw std::invalid_argument("Snake alpha must be finite with nonzero alpha + 1e-9");
      }
      std::atomic<bool> invalid{false};
      cpu::parallel_for(0, x.size(), cpu::GRAIN_SIZE, [&](dim_t begin, dim_t end) {
        bool found_invalid = false;
        for (dim_t start = begin; start < end;) {
          const dim_t row = start / time;
          const dim_t stop = std::min(end, (row + 1) * time);
          const float a = coefficients[row % channels];
          for (dim_t i = start; i < stop; ++i)
            found_invalid |= !std::isfinite(data[i]) | !std::isfinite(data[i] * a);
          start = stop;
        }
        if (found_invalid)
          invalid.store(true, std::memory_order_relaxed);
      });
      if (invalid.load(std::memory_order_relaxed))
        throw std::invalid_argument("Snake input and alpha * input must be finite");
      y.resize_as(x);
      auto* output = y.data<T>();
      cpu::parallel_for(0, x.size(), cpu::GRAIN_SIZE, [&](dim_t begin, dim_t end) {
        for (dim_t start = begin; start < end;) {
          const dim_t row = start / time;
          const dim_t stop = std::min(end, (row + 1) * time);
          const float a = coefficients[row % channels];
          const float inverse = 1.f / (a + epsilon);
          for (dim_t i = start; i < stop; ++i) {
            const float sine = std::sin(a * data[i]);
            output[i] = data[i] + inverse * (sine * sine);
          }
          start = stop;
        }
      });
    }

    template void Snake::compute<Device::CPU, float>(const StorageView&,
                                                    const StorageView&,
                                                    StorageView&) const;

  }
}
