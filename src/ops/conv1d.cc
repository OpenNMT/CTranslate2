#include "ctranslate2/ops/conv1d.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Conv1D::Conv1D(dim_t stride, dim_t padding, dim_t dilation)
      : _stride(stride)
      , _padding(padding)
      , _dilation(dilation)
    {
    }

    void Conv1D::operator()(const StorageView& input,
                            const StorageView& weight,
                            const StorageView& bias,
                            StorageView& output) const {
      operator()(input, weight, &bias, output);
    }

    void Conv1D::operator()(const StorageView& input,
                            const StorageView& weight,
                            StorageView& output) const {
      operator()(input, weight, nullptr, output);
    }

    void Conv1D::operator()(const StorageView& input,
                            const StorageView& weight,
                            const StorageView* bias,
                            StorageView& output) const {
      PROFILE("Conv1D");

      if (input.dtype() != weight.dtype())
        throw std::invalid_argument("Conv1D: input dtype is "
                                    + dtype_name(input.dtype())
                                    + " but expected dtype "
                                    + dtype_name(weight.dtype()));

      const dim_t batch_size = input.dim(0);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = (
        input_length + (2 * _padding) - (_dilation * (kernel_size - 1) + 1)) / _stride + 1;

      output.resize({batch_size, out_channels, output_length});

      DEVICE_AND_FLOAT_DISPATCH("Conv1D", input.device(), input.dtype(),
                                (compute<D, T>(input, weight, bias, output)));
    }

  }
}
