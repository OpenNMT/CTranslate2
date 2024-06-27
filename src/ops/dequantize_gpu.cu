#include "ctranslate2/ops/dequantize.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename InT, typename OutT>
    struct dequantize_func {
      __device__ __forceinline__
      OutT operator()(float scale, InT x, float zero) const {
        return __fdividef(__fsub_rn(static_cast<float>(x), zero) , scale);
      }
      __device__ __forceinline__
      OutT operator()(float scale, InT x) const {
        return __fdividef(static_cast<float>(x), scale);
      }
    };

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      const dim_t depth = input.dim(-1);
      cuda::binary_transform(scale.data<float>(),
                             input.data<InT>(),
                             output.data<OutT>(),
                             input.size(),
                             dequantize_func<InT, cuda::device_type<OutT>>(),
                             cuda::repeat_vec_depth<dim_t>(depth));
    }

    template <typename T>
    __global__ void dequantize_i4_kernel(const float* a,
                                         const float* z,
                                         const uint8_t* b,
                                         T* c,
                                         cuda::index_t depth) {
      const int32_t block_size = 32;
      const auto rescale_func = dequantize_func<uint8_t, T>();
      const cuda::index_t i = blockIdx.x;
      for (cuda::index_t j = threadIdx.x; j < depth; j += blockDim.x) {
        const cuda::index_t index = i * depth + j;
        const cuda::index_t m = index / (block_size / 2);
        const cuda::index_t n = index % (block_size / 2);
        const float scale = a[m];
        const float zero = z[m];
        uint8_t b1 = (b[index] & 0xF0) >> 4;
        uint8_t b2 = (b[index] & 0x0F);
        c[n + m * block_size] = rescale_func(scale, b1, zero);
        c[n + m * block_size + block_size / 2]  = rescale_func(scale, b2, zero);
      }
    }

    template <Device D, typename OutT>
    void Dequantize::dequantize_i4(const StorageView& input,
                                const StorageView& scale,
                                const StorageView& zero,
                                StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);
      dequantize_i4_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
        scale.data<float>(), zero.data<float>(), input.data<uint8_t>(), output.data<float>(), depth);
    }


    template <typename Epilogue, typename T>
    __global__ void dequantize_gemm_output_kernel(const int32_t* c,
                                                  const float* a_scales,
                                                  const float* b_scales,
                                                  const bool transpose_a,
                                                  const bool transpose_b,
                                                  const T* bias,
                                                  const Epilogue& epilogue,
                                                  T* y,
                                                  cuda::index_t depth) {
      // y = c / (expand_dims(a_scales, trans_a ? 0 : 1) * expand_dims(b_scales, trans_b ? 0 : 1)
      // if bias: y += expand_dims(bias, 0)
      // y = epilogue(y)
      const auto add_func = cuda::plus<T>();
      const auto rescale_func = dequantize_func<int32_t, T>();
      const cuda::index_t i = blockIdx.x;
      for (cuda::index_t j = threadIdx.x; j < depth; j += blockDim.x) {
        const cuda::index_t index = i * depth + j;
        const float scale = a_scales[transpose_a ? j : i] * b_scales[transpose_b ? j : i];
        T v = rescale_func(scale, c[index]);
        if (bias)
          v = add_func(v, bias[j]);
        y[index] = epilogue(v);
      }
    }

    template <typename T>
    static void dequantize_gemm_output_kernel_wrapper(const int32_t* c,
                                                      const float* a_scales,
                                                      const float* b_scales,
                                                      const bool transpose_a,
                                                      const bool transpose_b,
                                                      const T* bias,
                                                      const ActivationType* activation_type,
                                                      T* y,
                                                      dim_t batch_size,
                                                      dim_t depth) {
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      if (!activation_type) {
        dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          c, a_scales, b_scales, transpose_a, transpose_b, bias, thrust::identity<T>(), y, depth);

      } else {
        switch (*activation_type) {

        case ActivationType::ReLU: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::relu_func<T>(), y, depth);
          break;
        }

        case ActivationType::GELU: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::gelu_func<T>(), y, depth);
          break;
        }

        case ActivationType::GELUTanh: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::gelu_tanh_func<T>(), y, depth);
          break;
        }

        case ActivationType::GELUSigmoid: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::gelu_sigmoid_func<T>(), y, depth);
          break;
        }

        case ActivationType::Swish: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::swish_func<T>(), y, depth);
          break;
        }

        case ActivationType::Tanh: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::tanh_func<T>(), y, depth);
          break;
        }

        }
      }
    }

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& c,
                                            const StorageView& a_scale,
                                            const StorageView& b_scale,
                                            const bool transpose_a,
                                            const bool transpose_b,
                                            const StorageView* bias,
                                            StorageView& y) const {
      const dim_t batch_size = a_scale.size();
      const dim_t depth = c.dim(-1);
      dequantize_gemm_output_kernel_wrapper(
        c.data<int32_t>(),
        a_scale.data<float>(),
        b_scale.data<float>(),
        transpose_a,
        transpose_b,
        bias ? cuda::device_cast<T>(bias->data<T>()) : nullptr,
        _activation_type,
        cuda::device_cast<T>(y.data<T>()),
        batch_size,
        depth);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Dequantize::dequantize<Device::CUDA, int8_t, T>(                    \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    Dequantize::dequantize_i4<Device::CUDA, T>(                    \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    Dequantize::dequantize_gemm_output<Device::CUDA, T>(                \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const bool,                                                       \
      const bool,                                                       \
      const StorageView*,                                               \
      StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
