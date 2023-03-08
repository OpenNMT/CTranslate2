#include "ctranslate2/ops/whisper.h"

#include <algorithm>

#include "cpu/kernels.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    void NormalizeAttentionWeights::operator()(StorageView& input) const {
      operator()(input, input);
    }

    void NormalizeAttentionWeights::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("NormalizeAttentionWeights");

      if (input.device() != Device::CPU)
        throw std::invalid_argument("NormalizeAttentionWeights currently only supports CPU execution");

      output.resize_as(input);

      const dim_t axis = input.rank() - 2;
      const dim_t axis_size = input.dim(axis);

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis + 1; i < input.rank(); ++i)
        inner_size *= input.dim(i);

      CPU_ISA_DISPATCH((cpu::layer_norm_axis<ISA>(input.data<float>(),
                                                  output.data<float>(),
                                                  outer_size,
                                                  axis_size,
                                                  inner_size,
                                                  /*epsilon=*/0)));
    }


    MedianFilter::MedianFilter(const dim_t width)
      : _width(width)
    {
    }

    void MedianFilter::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("MedianFilter");

      if (input.device() != Device::CPU)
        throw std::invalid_argument("MedianFilter currently only supports CPU execution");

      output.resize_as(input);

      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      const dim_t rank = _width / 2;

      const auto* src = input.data<float>();
      auto* dst = output.data<float>();

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
  }
}
