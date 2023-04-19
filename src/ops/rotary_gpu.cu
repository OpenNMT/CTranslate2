#include "ctranslate2/ops/rotary.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T, bool interleave>
    __global__ void rotary_kernel(const T* x,
                                  const T* sin,
                                  const T* cos,
                                  T* y,
                                  const cuda::index_t max_time,
                                  const cuda::index_t ndims,
                                  const cuda::index_t depth) {
      const auto time = blockIdx.x % max_time;
      const auto middle = ndims / 2;

      x += blockIdx.x * depth;
      y += blockIdx.x * depth;

      sin += time * ndims;
      cos += time * ndims;

      for (cuda::index_t i = threadIdx.x; i < ndims; i += blockDim.x) {
        if (interleave)
          y[i] = x[i] * cos[i] + (i % 2 == 0 ? -x[i + 1] : x[i - 1]) * sin[i];
        else
          y[i] = x[i] * cos[i] + (i < middle ? -x[i + middle] : x[i - middle]) * sin[i];
      }

      for (cuda::index_t i = ndims + threadIdx.x; i < depth; i += blockDim.x) {
        y[i] = x[i];
      }
    }

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output) const {
      const dim_t max_time = input.dim(-2);
      const dim_t depth = input.dim(-1);
      const dim_t ndims = _ndims == 0 ? depth : _ndims;

      const dim_t blocks = std::min(input.size() / (max_time * depth), cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      const auto* x = cuda::device_cast(input.data<T>());
      const auto* s = cuda::device_cast(sin.data<T>());
      const auto* c = cuda::device_cast(cos.data<T>());
      auto* y = cuda::device_cast(output.data<T>());

      if (_interleave)
        rotary_kernel<cuda::device_type<T>, true><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, s, c, y, max_time, ndims, depth);
      else
        rotary_kernel<cuda::device_type<T>, false><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, s, c, y, max_time, ndims, depth);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::CUDA, T>(const StorageView&,                \
                                     const StorageView&,                \
                                     const StorageView&,                \
                                     StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
