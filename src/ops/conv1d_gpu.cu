#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"
#include "cuda/utils.h"
#include <cuda/helpers.h>

namespace ctranslate2 {
  namespace ops {

    // CUDA kernel for im2col transformation (transposed version)
    template<typename T>
    __global__ void im2col_transposed_kernel(
        const T* input,
        T* output,
        const int batch_size,
        const int in_channels,
        const int input_length,
        const int kernel_size,
        const int stride,
        const int padding,
        const int groups,
        const int output_length) {
      
      const int in_channels_per_group = in_channels / groups;
      const int in_batch_stride = in_channels * input_length;
      const int in_group_stride = in_channels_per_group * input_length;
      
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total_outputs = batch_size * groups * output_length * in_channels_per_group * kernel_size;
      
      if (idx >= total_outputs)
        return;
      
      // Decompose the linear index
      int temp = idx;
      const int k = temp % kernel_size;
      temp /= kernel_size;
      const int c_offset = temp % in_channels_per_group;
      temp /= in_channels_per_group;
      const int ti_idx = temp % output_length;
      temp /= output_length;
      const int group_idx = temp % groups;
      const int batch_idx = temp / groups;
      
      // Calculate input position
      const int ti = ti_idx * stride - padding;
      const int window_i = k + ti;
      
      // Calculate input offset
      const int batch_offset = batch_idx * in_batch_stride;
      const int group_offset = group_idx * in_group_stride;
      const int channel_offset = c_offset * input_length;
      
      // Fill output
      if (window_i >= 0 && window_i < input_length) {
        output[idx] = input[batch_offset + group_offset + channel_offset + window_i];
      } else {
        output[idx] = T(0);
      }
    }

    // Simple 1D bias kernel
    template<typename T>
    __global__ void add_bias_kernel_simple(
        T* __restrict__ output,
        const T* __restrict__ bias,
        const int batch_size,
        const int out_channels,
        const int output_length) {
      
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total_elements = batch_size * out_channels * output_length;
      
      if (idx < total_elements) {
        const int channel_idx = (idx / output_length) % out_channels;
        cuda::plus<T> add_op;
        output[idx] = add_op(output[idx], bias[channel_idx]);
      }
    }

    // Generic template implementation
    template <Device D, typename T>
    void Conv1D::compute(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {

      if (_dilation != 1)
        throw std::runtime_error("Dilation is not supported in this Conv1D implementation");

      using DevT = cuda::device_type<T>;

      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = output.dim(2);
      const dim_t in_channels_per_group = in_channels / _groups;

      // Create im2col_output tensor
      StorageView im2col_output(
          {batch_size, _groups, output_length, in_channels_per_group * kernel_size},
          T(0),
          Device::CUDA);

      // Launch im2col kernel
      const int total_outputs = batch_size * _groups * output_length * in_channels_per_group * kernel_size;
      const int block_size = 256;
      const int grid_size = (total_outputs + block_size - 1) / block_size;
      
      im2col_transposed_kernel<<<grid_size, block_size, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast(input.data<T>()),
          cuda::device_cast(im2col_output.data<T>()),
          batch_size,
          in_channels,
          input_length,
          kernel_size,
          _stride,
          _padding,
          _groups,
          output_length);

      // Perform GEMM operations
      const dim_t m = out_channels / _groups;
      const dim_t n = output_length;
      const dim_t k = in_channels_per_group * kernel_size;
      const dim_t stridew = (out_channels / _groups) * in_channels_per_group * kernel_size * weight.item_size();
      const dim_t strideb = k * output_length;
      const dim_t stridec = m * output_length;

      auto* w = static_cast<int8_t*>(const_cast<void*>(weight.buffer()));
      auto* b = im2col_output.data<T>();
      auto* c = output.data<T>();

      const Gemm gemm(1.0, 0.0, false, true);

      // Process each batch and group
      for (dim_t i = 0; i < batch_size * _groups; ++i) {
        auto group_index = i % _groups;
        void* w_i = w + (group_index * stridew);
        T* b_i = b + (i * strideb);
        T* c_i = c + (i * stridec);

        StorageView aa(weight.dtype(), Device::CUDA);
        aa.view(w_i, {m, k});
        StorageView bb({n, k}, b_i);
        StorageView cc({m, n}, c_i);

        if (qscale) {
          throw std::runtime_error("Scale is not supported");
        } else {
          gemm(aa, bb, cc);
        }
      }

      if (bias) {
        cudaStream_t stream = cuda::get_cuda_stream();
        DevT* output_ptr = cuda::device_cast(output.data<T>());
        const DevT* bias_ptr = cuda::device_cast(bias->data<T>());
        
        const int total_elements = batch_size * out_channels * output_length;
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        
        add_bias_kernel_simple<<<grid_size, block_size, 0, stream>>>(
            output_ptr, bias_ptr, batch_size, out_channels, output_length);
      }
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
