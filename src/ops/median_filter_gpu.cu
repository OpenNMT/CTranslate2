#include "ctranslate2/ops/median_filter.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 256;
    constexpr int kMaxWindow = 129; // supports window widths up to 129 (rank 64)

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

      float window[kMax];

      const int row_offset = row * depth;
      // Reflection gather.
      for (int k = -rank; k <= rank; ++k) {
        int read = col + k;
        if (read < 0) read = -read;
        if (read >= depth) read = 2 * depth - read - 2;
        window[k + rank] = float(input[row_offset + read]);
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
      output[tid] = DeviceT(window[rank]);
    }

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                              const dim_t axis_size,
                              StorageView& output) const {
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
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    MedianFilter::compute<Device::CUDA, T>(const StorageView& input,    \
                                           const dim_t axis_size,       \
                                           StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
