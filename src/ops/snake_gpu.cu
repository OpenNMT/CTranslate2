#include "ctranslate2/ops/snake.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void snake(const float* x, const float* alpha, float* out, dim_t n, dim_t time,
                            dim_t channels) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
          const float a = alpha[(i / time) % channels], s = sinf(a * x[i]);
          out[i] = x[i] + (1.f / (a + 1e-9f)) * (s * s);
        }
      }

    }

    template <Device D, typename T>
    void Snake::compute(const StorageView& x, const StorageView& alpha, StorageView& out) const {
      out.resize_as(x);
      if (x.empty())
        return;
      snake<<<(x.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          x.data<T>(), alpha.data<T>(), out.data<T>(), x.size(), x.dim(2), x.dim(1));
      CUDA_CHECK(cudaGetLastError());
    }

    template void Snake::compute<Device::CUDA, float>(const StorageView&,
                                                     const StorageView&,
                                                     StorageView&) const;

  }
}
