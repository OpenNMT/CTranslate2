#include "ctranslate2/ops/pad.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      // One thread per output element (grid-stride loop). Each row of the input has
      // `length` samples and each row of the output has `out_len = left + length + right`.
      __global__ void pad_reflect_1d(const float* in, float* out,
                                     dim_t total, dim_t length, dim_t out_len, dim_t left) {
        for (dim_t i = dim_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
             i += dim_t(blockDim.x) * gridDim.x) {
          const dim_t row = i / out_len;
          const dim_t p = i % out_len;
          const dim_t x = p - left;

          dim_t src;
          if (x < 0)
            src = -x;                    // Left border: mirror around index 0.
          else if (x >= length)
            src = 2 * (length - 1) - x;  // Right border: mirror around index length - 1.
          else
            src = x;                     // Middle: copy.

          out[i] = in[row * length + src];
        }
      }

    }

    template <Device D, typename T>
    void Pad1D::compute(const StorageView& input, StorageView& output) const {
      const dim_t length = input.dim(-1);
      const dim_t out_len = length + _left + _right;

      Shape out_shape = input.shape();
      out_shape.back() = out_len;
      output.resize(std::move(out_shape));

      if (output.empty())
        return;

      const dim_t threads = 256;
      const dim_t blocks = std::min((output.size() + threads - 1) / threads, dim_t(65535));
      pad_reflect_1d<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
        input.data<T>(), output.data<T>(), output.size(), length, out_len, _left);
      CUDA_CHECK(cudaGetLastError());
    }

    template void Pad1D::compute<Device::CUDA, float>(const StorageView&,
                                                     StorageView&) const;

  }
}
