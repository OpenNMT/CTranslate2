#include "ctranslate2/ops/median_filter.h"

#include <cuda_fp16.h>
#ifdef CUDA_BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

#include "type_dispatch.h"
#include "cuda/helpers.h"
#include <type_traits>

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 256;

    // Conversion helpers
    __device__ __forceinline__ float to_float(float v) { return v; }
    __device__ __forceinline__ float to_float(const half v) { return __half2float(v); }
#ifdef CUDA_BF16_AVAILABLE
    __device__ __forceinline__ float to_float(const __nv_bfloat16 v) { return __bfloat162float(v); }
#endif

    __device__ __forceinline__ float from_float(float v) { return v; }
    __device__ __forceinline__ half from_float_half(float v) { return __float2half(v); }
#ifdef CUDA_BF16_AVAILABLE
    __device__ __forceinline__ __nv_bfloat16 from_float_bf16(float v) { return __float2bfloat16(v); }
#endif

    namespace {
      constexpr int kMaxWindow = 129; // supports window widths up to 129 (rank 64)
    }

    template <typename DeviceT, int kMax>
    __global__ void sliding_median_lastdim_kernel(const DeviceT* input,
                                                  DeviceT* output,
                                                  int rows,
                                                  int depth,
                                                  int width) {
      const int tid = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = rows * depth;
      if (tid >= total) return;

      int row = tid / depth;
      int col = tid % depth;
      const int rank = width / 2;

      if (depth <= rank) {
        output[tid] = input[tid];
        return;
      }
      if (width > kMax) {
        output[tid] = input[tid];
        return;
      }

      float window[kMax];

      const int row_offset = row * depth;
      // Reflection gather.
      for (int k = -rank; k <= rank; ++k) {
        int read = col + k;
        if (read < 0) read = -read;
        if (read >= depth) read = 2 * depth - read - 2;
        window[k + rank] = to_float(input[row_offset + read]);
      }

      // Insertion sort (width is small: <= kMax, typically < 129).
      for (int i = 1; i < width; ++i) {
        float key = window[i];
        int j = i - 1;
        while (j >= 0 && window[j] > key) {
          window[j + 1] = window[j];
          --j;
        }
        window[j + 1] = key;
      }
      float median = window[rank];

      if constexpr (std::is_same<DeviceT, float>::value) {
        output[tid] = median;
      } else if constexpr (std::is_same<DeviceT, half>::value) {
        output[tid] = from_float_half(median);
#ifdef CUDA_BF16_AVAILABLE
      } else if constexpr (std::is_same<DeviceT, __nv_bfloat16>::value) {
        output[tid] = from_float_bf16(median);
#endif
      }
    }

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                              const dim_t axis_size,
                              StorageView& output) const {
      output.resize_as(input);
      const int depth = static_cast<int>(axis_size);
      const int rows = static_cast<int>(input.size() / depth);
      const int width = static_cast<int>(_width);
      const int rank = width / 2;

      // Host-side guards and fallbacks.
      if (width <= 1) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }
      if ((width & 1) == 0)
        throw std::invalid_argument("MedianFilter width must be odd");
      if (width > kMaxWindow)
        throw std::invalid_argument("MedianFilter width exceeds supported GPU max (" + std::to_string(kMaxWindow) + ")");
      if (depth <= rank) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }

      // Grid configuration
      const int total = rows * depth;
      int blocks = (total + num_threads - 1) / num_threads;
      if (blocks > cuda::max_blocks) {
        blocks = cuda::max_blocks;
      }

      using device_t = cuda::device_type<T>;
      const device_t* in_ptr = cuda::device_cast(input.data<T>());
      device_t* out_ptr = cuda::device_cast(output.data<T>());
      sliding_median_lastdim_kernel<device_t, kMaxWindow><<<blocks, num_threads, 0, cuda::get_cuda_stream()>>>(
        in_ptr,
        out_ptr,
        rows,
        depth,
        width);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    MedianFilter::compute<Device::CUDA, T>(const StorageView& input,    \
                                           const dim_t axis_size,       \
                                           StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
