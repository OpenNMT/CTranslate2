#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/dequantize.h"
#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/ops/quantize.h"
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

  if (qscale) {
    if (weight.dtype() != DataType::INT8)
      throw std::invalid_argument("Quantized MPS Conv1D expects INT8 weights");
    if (qscale->dtype() != DataType::FLOAT32)
      throw std::invalid_argument("Quantized MPS Conv1D expects FLOAT32 scales");
    if (!qscale->is_scalar() && qscale->size() != out_channels)
      throw std::invalid_argument(
          "Quantized MPS Conv1D expects one scale per output channel");

    // Quantization scales loaded from a model are typically CPU-resident.  Keep a
    // device copy alive for the duration of the asynchronously encoded kernels
    // instead of presenting a host pointer as if it belonged to an MTLBuffer.
    StorageView device_qscale;
    if (qscale->device() != Device::MPS)
      device_qscale = qscale->to(Device::MPS);
    const StorageView& mps_qscale = device_qscale ? device_qscale : *qscale;

    StorageView quantized_col(col_buffer.shape(), DataType::INT8, Device::MPS);
    StorageView col_scale(DataType::FLOAT32, Device::MPS);
    Quantize()(col_buffer, quantized_col, col_scale);
    StorageView quantized_output(output.shape(), DataType::INT32, Device::MPS);

    const int8_t* w = weight.data<int8_t>();
    const int8_t* qcol = quantized_col.data<int8_t>();
    int32_t* qout = quantized_output.data<int32_t>();

    for (dim_t g = 0; g < _groups; ++g) {
      primitives<Device::MPS>::gemm_batch_strided<int8_t, int32_t>(
          false, true,
          out_channels_per_group, output_length, k,
          1.0f,
          w + g * stridew, k, 0,
          qcol + g * stridecol, k, _groups * stridecol,
          0.0f,
          qout + g * stridec, output_length, _groups * stridec,
          batch_size);
    }

    const dim_t batch_output_stride = _groups * stridec;
    const dim_t qscale_stride = mps_qscale.is_scalar()
                                ? 0
                                : mps_qscale.size() / _groups;
    for (dim_t b = 0; b < batch_size; ++b) {
      for (dim_t g = 0; g < _groups; ++g) {
        StorageView c_view(DataType::INT32, Device::MPS);
        c_view.view(qout + b * batch_output_stride + g * stridec,
                    {out_channels_per_group, output_length});

        StorageView weight_scale(DataType::FLOAT32, Device::MPS);
        weight_scale.view(const_cast<float*>(mps_qscale.data<float>())
                            + (qscale_stride == 0 ? 0 : g * qscale_stride),
                          mps_qscale.is_scalar() ? Shape{} : Shape{out_channels_per_group});

        StorageView input_scale(DataType::FLOAT32, Device::MPS);
        input_scale.view(col_scale.data<float>()
                           + (b * _groups + g) * output_length,
                         {output_length});

        StorageView output_view(input.dtype(), Device::MPS);
        output_view.view(o + b * batch_output_stride + g * stridec,
                         {out_channels_per_group, output_length});

        Dequantize()(c_view,
                     weight_scale,
                     input_scale,
                     false,
                     true,
                     output_view);
      }
    }

    apply_bias_and_activation(output, bias, _activation_type,
                              /*residual=*/nullptr,
                              /*axis=*/-2);
    return;
  }

  const T* w = weight.data<T>();

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
