// src/ops/layer_norm_gpu.cu

#include "ctranslate2/ops/layer_norm.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace ctranslate2 {
  namespace ops {

#define CUDA_NUM_THREADS 256
#define CUDA_WARP_SIZE 32

    struct WelfordData {
      float mean;
      float m2;
      float count;
    };

    __device__ __forceinline__ WelfordData welford_init() {
      return WelfordData{0.f, 0.f, 0.f};
    }

    __device__ __forceinline__ void welford_update(WelfordData& a, float x) {
      a.count += 1.f;
      const float delta1 = x - a.mean;
      a.mean += delta1 / a.count;
      const float delta2 = x - a.mean;
      a.m2 += delta1 * delta2;
    }

    __device__ __forceinline__ WelfordData welford_combine(const WelfordData& a,
                                                           const WelfordData& b) {
      if (b.count == 0.f)
        return a;
      if (a.count == 0.f)
        return b;

      const float count = a.count + b.count;
      const float delta = b.mean - a.mean;
      const float mean = a.mean + delta * (b.count / count);
      const float m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / count);
      return WelfordData{mean, m2, count};
    }

    struct WelfordReduceOp {
      __device__ __forceinline__ WelfordData operator()(const WelfordData& a,
                                                        const WelfordData& b) const {
        return welford_combine(a, b);
      }
    };

    template <typename T>
    __device__ __forceinline__ float to_float(T x) {
      return float(x);
    }

    template <>
    __device__ __forceinline__ float to_float<half>(half x) {
      return __half2float(x);
    }

#if CUDA_VERSION >= 11000
    template <>
    __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 x) {
      return __bfloat162float(x);
    }
#endif

    template <typename T>
    __device__ __forceinline__ T from_float(float x) {
      return T(x);
    }

    template <>
    __device__ __forceinline__ half from_float<half>(float x) {
      return __float2half_rn(x);
    }

#if CUDA_VERSION >= 11000
    template <>
    __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float x) {
      return __float2bfloat16_rn(x);
    }
#endif

    template <typename T, typename SizeT, bool HasAffine>
    __global__ void LayerNormLastAxisWarpKernel(SizeT cols,
                                                float eps,
                                                const T* __restrict__ x,
                                                const T* __restrict__ gamma,
                                                const T* __restrict__ beta,
                                                T* __restrict__ y) {
      const int tid = threadIdx.x;
      const int warp_id = tid / CUDA_WARP_SIZE;
      const int lane = tid % CUDA_WARP_SIZE;
      const int warps_per_block = blockDim.x / CUDA_WARP_SIZE;

      const SizeT row = SizeT(blockIdx.x) * SizeT(warps_per_block) + SizeT(warp_id);
      const SizeT rows = SizeT(gridDim.x) * SizeT(warps_per_block);
      for (SizeT r = row; r < rows; r += rows) {
        WelfordData wd = welford_init();
        const SizeT base = r * cols;

        for (SizeT j = SizeT(lane); j < cols; j += SizeT(CUDA_WARP_SIZE)) {
          const float v = to_float(x[base + j]);
          welford_update(wd, v);
        }

        for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
          WelfordData b;
          b.mean = __shfl_down_sync(0xffffffff, wd.mean, offset);
          b.m2 = __shfl_down_sync(0xffffffff, wd.m2, offset);
          b.count = __shfl_down_sync(0xffffffff, wd.count, offset);
          wd = welford_combine(wd, b);
        }

        const float mean = __shfl_sync(0xffffffff, wd.mean, 0);
        const float m2 = __shfl_sync(0xffffffff, wd.m2, 0);
        const float count = __shfl_sync(0xffffffff, wd.count, 0);
        const float var = fmaxf(m2 / count, 0.f);
        const float inv_std = rsqrtf(var + eps);

        for (SizeT j = SizeT(lane); j < cols; j += SizeT(CUDA_WARP_SIZE)) {
          const SizeT idx = base + j;
          float out = (to_float(x[idx]) - mean) * inv_std;
          if constexpr (HasAffine) {
            out = out * to_float(gamma[j]) + to_float(beta[j]);
          }
          y[idx] = from_float<T>(out);
        }
      }
    }

    template <typename T, typename SizeT, bool HasAffine>
    __global__ void LayerNormLastAxisBlockSMemKernel(SizeT cols,
                                                     float eps,
                                                     const T* __restrict__ x,
                                                     const T* __restrict__ gamma,
                                                     const T* __restrict__ beta,
                                                     T* __restrict__ y) {
      extern __shared__ unsigned char smem_raw[];
      T* smem = reinterpret_cast<T*>(smem_raw);

      const SizeT row = SizeT(blockIdx.x);
      const SizeT base = row * cols;

      WelfordData wd = welford_init();
      for (SizeT j = SizeT(threadIdx.x); j < cols; j += SizeT(blockDim.x)) {
        const T v = x[base + j];
        smem[j] = v;
        welford_update(wd, to_float(v));
      }

      using BlockReduce = cub::BlockReduce<WelfordData, CUDA_NUM_THREADS>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      const WelfordData reduced = BlockReduce(temp_storage).Reduce(wd, WelfordReduceOp{});

      __shared__ float s_mean;
      __shared__ float s_inv_std;
      if (threadIdx.x == 0) {
        const float var = fmaxf(reduced.m2 / reduced.count, 0.f);
        s_mean = reduced.mean;
        s_inv_std = rsqrtf(var + eps);
      }
      __syncthreads();

      const float mean = s_mean;
      const float inv_std = s_inv_std;

      for (SizeT j = SizeT(threadIdx.x); j < cols; j += SizeT(blockDim.x)) {
        float out = (to_float(smem[j]) - mean) * inv_std;
        if constexpr (HasAffine) {
          out = out * to_float(gamma[j]) + to_float(beta[j]);
        }
        y[base + j] = from_float<T>(out);
      }
    }

    template <typename T, typename SizeT, bool HasAffine>
    __global__ void LayerNormLastAxisBlockUncachedKernel(SizeT cols,
                                                         float eps,
                                                         const T* __restrict__ x,
                                                         const T* __restrict__ gamma,
                                                         const T* __restrict__ beta,
                                                         T* __restrict__ y) {
      const SizeT row = SizeT(blockIdx.x);
      const SizeT base = row * cols;

      WelfordData wd = welford_init();
      for (SizeT j = SizeT(threadIdx.x); j < cols; j += SizeT(blockDim.x)) {
        welford_update(wd, to_float(x[base + j]));
      }

      using BlockReduce = cub::BlockReduce<WelfordData, CUDA_NUM_THREADS>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      const WelfordData reduced = BlockReduce(temp_storage).Reduce(wd, WelfordReduceOp{});

      __shared__ float s_mean;
      __shared__ float s_inv_std;
      if (threadIdx.x == 0) {
        const float var = fmaxf(reduced.m2 / reduced.count, 0.f);
        s_mean = reduced.mean;
        s_inv_std = rsqrtf(var + eps);
      }
      __syncthreads();

      const float mean = s_mean;
      const float inv_std = s_inv_std;

      for (SizeT j = SizeT(threadIdx.x); j < cols; j += SizeT(blockDim.x)) {
        const SizeT idx = base + j;
        float out = (to_float(x[idx]) - mean) * inv_std;
        if constexpr (HasAffine) {
          out = out * to_float(gamma[j]) + to_float(beta[j]);
        }
        y[idx] = from_float<T>(out);
      }
    }

    template <typename T, typename SizeT, bool HasAffine>
    __global__ void LayerNormGeneralStridedKernel(SizeT axis_size,
                                                  SizeT inner_size,
                                                  float eps,
                                                  const T* __restrict__ x,
                                                  const T* __restrict__ gamma,
                                                  const T* __restrict__ beta,
                                                  T* __restrict__ y) {
      const SizeT group = SizeT(blockIdx.x);
      const SizeT outer_idx = group / inner_size;
      const SizeT inner_idx = group - outer_idx * inner_size;
      const SizeT base = (outer_idx * axis_size * inner_size) + inner_idx;

      WelfordData wd = welford_init();
      for (SizeT j = SizeT(threadIdx.x); j < axis_size; j += SizeT(blockDim.x)) {
        const float v = to_float(x[base + j * inner_size]);
        welford_update(wd, v);
      }

      using BlockReduce = cub::BlockReduce<WelfordData, CUDA_NUM_THREADS>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      const WelfordData reduced = BlockReduce(temp_storage).Reduce(wd, WelfordReduceOp{});

      __shared__ float s_mean;
      __shared__ float s_inv_std;
      if (threadIdx.x == 0) {
        const float var = fmaxf(reduced.m2 / reduced.count, 0.f);
        s_mean = reduced.mean;
        s_inv_std = rsqrtf(var + eps);
      }
      __syncthreads();

      const float mean = s_mean;
      const float inv_std = s_inv_std;

      for (SizeT j = SizeT(threadIdx.x); j < axis_size; j += SizeT(blockDim.x)) {
        const SizeT idx = base + j * inner_size;
        float out = (to_float(x[idx]) - mean) * inv_std;
        if constexpr (HasAffine) {
          out = out * to_float(gamma[j]) + to_float(beta[j]);
        }
        y[idx] = from_float<T>(out);
      }
    }

    template <typename T, typename SizeT, bool HasAffine>
    static void dispatch_last_axis(const SizeT rows,
                                   const SizeT cols,
                                   const float eps,
                                   const T* x,
                                   const T* gamma,
                                   const T* beta,
                                   T* y,
                                   cudaStream_t stream) {
      if (cols <= 1024) {
        const int block = 256;
        const int warps_per_block = block / CUDA_WARP_SIZE;
        const int grid = int((rows + warps_per_block - 1) / warps_per_block);
        LayerNormLastAxisWarpKernel<T, SizeT, HasAffine>
          <<<grid, block, 0, stream>>>(cols, eps, x, gamma, beta, y);
        return;
      }

      int dev = 0;
      cudaGetDevice(&dev);
      int max_smem_optin = 0;
      cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

      const size_t smem_bytes = size_t(cols) * sizeof(T);
      if (smem_bytes > 0 && smem_bytes <= size_t(max_smem_optin)) {
        cudaFuncSetAttribute(LayerNormLastAxisBlockSMemKernel<T, SizeT, HasAffine>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_smem_optin);
        LayerNormLastAxisBlockSMemKernel<T, SizeT, HasAffine>
          <<<int(rows), CUDA_NUM_THREADS, smem_bytes, stream>>>(cols, eps, x, gamma, beta, y);
      } else {
        LayerNormLastAxisBlockUncachedKernel<T, SizeT, HasAffine>
          <<<int(rows), CUDA_NUM_THREADS, 0, stream>>>(cols, eps, x, gamma, beta, y);
      }
    }

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output) const {
      const bool is_last_axis = (axis == input.rank() - 1);
      const bool has_affine = (beta != nullptr && gamma != nullptr);

      const cudaStream_t stream = cuda::get_cuda_stream();
      using CudaT = cuda::device_type<T>;
      using SizeT = cuda::index_t;

      const CudaT* x = cuda::device_cast(input.data<T>());
      CudaT* y = cuda::device_cast(output.data<T>());

      if (is_last_axis) {
        const SizeT rows = SizeT(outer_size);
        const SizeT cols = SizeT(axis_size);

        if (has_affine) {
          const CudaT* g = cuda::device_cast(gamma->data<T>());
          const CudaT* b = cuda::device_cast(beta->data<T>());
          dispatch_last_axis<CudaT, SizeT, true>(rows, cols, _epsilon, x, g, b, y, stream);
        } else {
          dispatch_last_axis<CudaT, SizeT, false>(rows, cols, _epsilon, x, nullptr, nullptr, y, stream);
        }
        return;
      }

      const SizeT groups = SizeT(outer_size) * SizeT(inner_size);
      const SizeT a = SizeT(axis_size);
      const SizeT in = SizeT(inner_size);

      if (has_affine) {
        const CudaT* g = cuda::device_cast(gamma->data<T>());
        const CudaT* b = cuda::device_cast(beta->data<T>());
        LayerNormGeneralStridedKernel<CudaT, SizeT, true>
          <<<dim3(unsigned(groups)), CUDA_NUM_THREADS, 0, stream>>>(
            a, in, _epsilon, x, g, b, y);
      } else {
        LayerNormGeneralStridedKernel<CudaT, SizeT, false>
          <<<dim3(unsigned(groups)), CUDA_NUM_THREADS, 0, stream>>>(
            a, in, _epsilon, x, nullptr, nullptr, y);
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CUDA, T>(const StorageView* beta,        \
                                        const StorageView* gamma,       \
                                        const StorageView& input,       \
                                        const dim_t axis,               \
                                        const dim_t outer_size,         \
                                        const dim_t axis_size,          \
                                        const dim_t inner_size,         \
                                        StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
