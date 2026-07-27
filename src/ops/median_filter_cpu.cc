#include "ctranslate2/ops/median_filter.h"

#include <iostream>

#include <algorithm>
#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                       const dim_t axis_size,
                       StorageView& output) const {
      const dim_t depth = axis_size;
      const dim_t rank = _width / 2;

      // Guard before dividing by depth: a zero-size axis (e.g. Whisper align()
      // with num_frames < 2, halved to 0 by the encoder stride) would make
      // input.size() / depth an integer division by zero — a native crash,
      // not a catchable exception. Pass the input through like the GPU path.
      if (depth <= rank) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }

      const auto* src = input.data<T>();
      auto* dst = output.data<T>();

      const dim_t batch_size = input.size() / depth;

      cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        StorageView window_storage({_width}, DataType::FLOAT32);
        auto* window = window_storage.data<float>();

        for (dim_t i = begin; i < end; ++i) {
          const dim_t offset = i * depth;
          const auto* in = src + offset;
          auto* out = dst + offset;

          for (dim_t j = 0; j < depth; ++j) {
            for (dim_t k = -rank; k <= rank; ++k) {
              dim_t read = std::abs(j + k);
              if (read >= depth)
                read = depth - (read - depth) - 2;
              window[k + rank] = in[read];
            }

            std::nth_element(window, window + rank, window + _width);
            out[j] = window[rank];
          }
        }
      });
    }

#define DECLARE_IMPL(T)                                             \
    template void                                                   \
    MedianFilter::compute<Device::CPU, T>(const StorageView& input, \
                                          const dim_t axis_size,    \
                                          StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
