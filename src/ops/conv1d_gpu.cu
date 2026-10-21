#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"

#include "cuda/helpers.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // CUDA kernel for im2col transformation (transposed version)
    template<typename T>
    __global__ void im2col_transposed_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        const int batch_size,
        const int in_channels,
        const int input_length,
        const int kernel_size,
        const int stride,
        const int padding,
        const int dilation,
        const int groups,
        const int output_length,
        const int k,
        const int in_batch_stride,
        const int in_group_stride) {

      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= k)
        return;

      // Decompose the linear index
      const int c_offset = idx / kernel_size;
      const int k_offset = idx - c_offset * kernel_size;
      const int ti_idx = blockIdx.y;
      const int batch_group = blockIdx.z;
      const int batch_idx = batch_group / groups;
      const int group_idx = batch_group - batch_idx * groups;

      // Calculate input position
      const int ti = ti_idx * stride - padding;
      const int window_i = dilation * k_offset + ti;

      // Calculate input offset
      const int batch_offset = batch_idx * in_batch_stride;
      const int group_offset = group_idx * in_group_stride;
      const int channel_offset = c_offset * input_length;

      // Fill output
      const int input_idx = batch_offset + group_offset + channel_offset + window_i;
      const int output_idx = (batch_group * output_length + ti_idx) * k + idx;
      output[output_idx] = window_i >= 0 && window_i < input_length ? input[input_idx] : T(0);
    }

    // Generic template implementation
    template <Device D, typename T>
    void Conv1D::compute(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView*) const {

      using DevT = cuda::device_type<T>;

      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = output.dim(2);
      const dim_t in_channels_per_group = in_channels / _groups;
      const dim_t out_channels_per_group = out_channels / _groups;
      const dim_t k = in_channels_per_group * kernel_size;

      StorageView buffer({batch_size, _groups, output_length, k}, DataTypeToEnum<T>::value, D);
      const T* x = input.data<T>();
      const T* w = weight.data<T>();
      T* o = output.data<T>();
      T* p = buffer.data<T>();

      const dim_t in_batch_stride = in_channels * input_length;
      const dim_t in_group_stride = in_batch_stride / _groups;
      const int threads = 256;
      const dim3 grid((k + threads - 1) / threads,
                      output_length,
                      batch_size * _groups);

      im2col_transposed_kernel<<<grid, threads, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast(x), cuda::device_cast(p),
          batch_size, in_channels, input_length, kernel_size,
          _stride, _padding, _dilation, _groups, output_length,
          k, in_batch_stride, in_group_stride);

      const dim_t stridew = out_channels_per_group * in_channels_per_group * kernel_size;
      const dim_t stridep = k * output_length;
      const dim_t strideo = out_channels_per_group * output_length;

      for (dim_t g = 0; g < _groups; ++g) {
        const T* w_g = w + g * stridew;
        const T* p_g = p + g * stridep;
        T* o_g = o + g * strideo;

        primitives<Device::CUDA>::gemm_batch_strided(false, true, // transpose
                                                     out_channels_per_group, output_length, k,
                                                     1.0f, // alpha
                                                     w_g, k, 0, // stridea
                                                     p_g, k, _groups * stridep,
                                                     0.0f, // beta
                                                     o_g, output_length, _groups * strideo,
                                                     batch_size);
      }

      apply_bias_and_activation(output, bias, _activation_type, /*residual=*/nullptr, /*axis=*/-2);
    }

    // Template instantiations
    #define DECLARE_IMPL(T)                                                 \
    template void                                                           \
    Conv1D::compute<Device::CUDA, T>(const StorageView& input,              \
                                     const StorageView& weight,             \
                                     const StorageView* bias,               \
                                     StorageView& output,                   \
                                     const StorageView* qscale) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
