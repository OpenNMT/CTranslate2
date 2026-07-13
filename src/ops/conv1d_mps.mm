#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"
#include "mps/kernels.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

template <typename T>
static void im2col_transposed_mps(const T* input,
                                  T* output,
                                  dim_t batch_size,
                                  dim_t in_channels,
                                  dim_t input_length,
                                  dim_t kernel_size,
                                  dim_t stride,
                                  dim_t padding,
                                  dim_t dilation,
                                  dim_t groups,
                                  dim_t output_length,
                                  dim_t k,
                                  dim_t in_batch_stride,
                                  dim_t in_group_stride) {
  for (dim_t b = 0; b < batch_size; ++b) {
    for (dim_t g = 0; g < groups; ++g) {
      for (dim_t t = 0; t < output_length; ++t) {
        const dim_t base_t = t * stride - padding;
        for (dim_t c = 0; c < k; ++c) {
          const dim_t c_offset = c / kernel_size;
          const dim_t k_offset = c % kernel_size;
          const dim_t in_t = base_t + dilation * k_offset;

          const dim_t input_idx =
              b * in_batch_stride +
              g * in_group_stride +
              c_offset * input_length +
              in_t;

          const dim_t out_idx =
              ((b * groups + g) * output_length + t) * k + c;

          output[out_idx] =
              (in_t >= 0 && in_t < input_length) ? input[input_idx] : T(0);
        }
      }
    }
  }
}

template <Device D, typename T>
void Conv1D::compute(const StorageView& input,
                     const StorageView& weight,
                     const StorageView* bias,
                     StorageView& output,
                     const StorageView* qscale) const {
  if (qscale)
    throw std::runtime_error("Quantized Conv1D not supported on MPS");

  const dim_t batch_size = input.dim(0);
  const dim_t in_channels = input.dim(1);
  const dim_t input_length = input.dim(2);
  const dim_t out_channels = weight.dim(0);
  const dim_t kernel_size = weight.dim(2);
  const dim_t output_length = output.dim(2);

  const dim_t in_channels_per_group = in_channels / _groups;
  const dim_t out_channels_per_group = out_channels / _groups;
  const dim_t k = in_channels_per_group * kernel_size;

  StorageView col_buffer(
      {batch_size, _groups, output_length, k},
      input.dtype(),
      Device::MPS);

  const T* x = input.data<T>();
  const T* w = weight.data<T>();
  T* o = output.data<T>();
  T* col = col_buffer.data<T>();

  const dim_t in_batch_stride = in_channels * input_length;
  const dim_t in_group_stride = in_batch_stride / _groups;

  mps::im2col_conv1d(input.dtype(),
                     x,
                     col,
                     batch_size,
                     _groups,
                     input_length,
                     kernel_size,
                     _stride,
                     _padding,
                     _dilation,
                     output_length,
                     k,
                     in_batch_stride,
                     in_group_stride);

  const dim_t stridew = out_channels_per_group * k;
  const dim_t stridec = output_length * out_channels_per_group;
  const dim_t stridecol = output_length * k;

  for (dim_t g = 0; g < _groups; ++g) {
    primitives<Device::MPS>::gemm_batch_strided(
        false, true,
        out_channels_per_group, output_length, k,
        1.0f,
        w + g * stridew, k, 0,
        col + g * stridecol, k, _groups * stridecol,
        0.0f,
        o + g * stridec, output_length, _groups * stridec,
        batch_size);
  }

  apply_bias_and_activation(output, bias, _activation_type,
                            /*residual=*/nullptr,
                            /*axis=*/-2);
}

#define DECLARE_IMPL(T)                                             \
template void                                                       \
Conv1D::compute<Device::MPS, T>(const StorageView&,                \
                                const StorageView&,                \
                                const StorageView*,                \
                                StorageView&,                      \
                                const StorageView*) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2
