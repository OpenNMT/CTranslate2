#include "ctranslate2/ops/whisper.h"

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

  }
}
