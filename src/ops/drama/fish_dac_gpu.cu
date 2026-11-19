#include "ops/drama/fish_dac_gpu.h"
#include "ctranslate2/ops/gemm.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void causal_columns(const float* x, float* columns, dim_t n, dim_t time,
                                     dim_t channels, dim_t kernel, dim_t dilation, dim_t start) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
          return;
        const dim_t t = start + i / (channels * kernel), c = (i / kernel) % channels,
                    k = i % kernel;
        const dim_t source = t - (kernel - 1 - k) * dilation;
        columns[i] = source < 0 ? 0.f : x[c * time + source];
      }

      __global__ void bias(float* y, const float* b, dim_t n, dim_t time) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          y[i] += b[i / time];
      }

      __global__ void scatter_tile(const float* tile, const float* b, float* y, dim_t n,
                                   dim_t count, dim_t time, dim_t start) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          y[(i / count) * time + start + i % count] = tile[i] + b[i / count];
      }

      __global__ void col2im(const float* contributions, const float* b, float* y, dim_t n,
                             dim_t time, dim_t length, dim_t kernel, dim_t stride) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
          return;
        const dim_t c = i / length, pos = i % length;
        float value = b[c];
        for (dim_t k = 0; k < kernel; ++k) {
          const dim_t source = pos - k;
          if (source >= 0 && source % stride == 0 && source / stride < time)
            value += contributions[(c * kernel + k) * time + source / stride];
        }
        y[i] = value;
      }

      __global__ void tanh_output(float* x, dim_t n) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          x[i] = tanhf(x[i]);
      }

    }

    void fish_causal_gpu(const StorageView& x, const StorageView& weight, const StorageView& b,
                         dim_t dilation, StorageView& out) {
      const dim_t time = x.dim(2), cin = x.dim(1), cout = weight.dim(0), kernel = weight.dim(2);
      StorageView w({cout, cin * kernel}, const_cast<float*>(weight.data<float>()), Device::CUDA);
      if (kernel == 1) {
        StorageView input({cin, time}, const_cast<float*>(x.data<float>()), Device::CUDA);
        Gemm(1, 0, false, false)(w, input, out);
        out.reshape({1, cout, time});
        bias<<<(out.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
            out.data<float>(), b.data<float>(), out.size(), time);
      } else {
        // Bound im2col workspace to 16 MiB rather than scaling it with audio length.
        constexpr dim_t workspace = 16 * 1024 * 1024 / sizeof(float);
        const dim_t tile_size = std::min(time, std::max(dim_t(1), workspace / (cin * kernel)));
        out.resize({1, cout, time});
        StorageView columns(DataType::FLOAT32, Device::CUDA), tile(DataType::FLOAT32, Device::CUDA);
        for (dim_t start = 0; start < time; start += tile_size) {
          const dim_t count = std::min(tile_size, time - start);
          columns.resize({count, cin * kernel});
          causal_columns<<<(columns.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
              x.data<float>(), columns.data<float>(), columns.size(), time, cin, kernel, dilation,
              start);
          Gemm(1, 0, false, true)(w, columns, tile);
          scatter_tile<<<(tile.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
              tile.data<float>(), b.data<float>(), out.data<float>(), tile.size(), count, time,
              start);
        }
      }
      CUDA_CHECK(cudaGetLastError());
    }

    void fish_transpose_gpu(const StorageView& x, const StorageView& weight, const StorageView& b,
                            dim_t stride, StorageView& out) {
      const dim_t time = x.dim(2), cin = x.dim(1), cout = weight.dim(1), kernel = weight.dim(2),
                  length = time * stride;
      StorageView w({cin, cout * kernel}, const_cast<float*>(weight.data<float>()), Device::CUDA);
      StorageView input({cin, time}, const_cast<float*>(x.data<float>()), Device::CUDA);
      StorageView contributions(DataType::FLOAT32, Device::CUDA);
      Gemm(1, 0, true, false)(w, input, contributions);
      out.resize({1, cout, length});
      col2im<<<(out.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          contributions.data<float>(), b.data<float>(), out.data<float>(), out.size(), time, length,
          kernel, stride);
      CUDA_CHECK(cudaGetLastError());
    }

    void fish_tanh_gpu(StorageView& x) {
      tanh_output<<<(x.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(x.data<float>(),
                                                                               x.size());
      CUDA_CHECK(cudaGetLastError());
    }

  }
}
