#include "ctranslate2/ops/conv_transpose1d.h"
#include "ctranslate2/ops/gemm.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void pack_tile(const float* x, float* tile, dim_t size, dim_t time,
                                dim_t start, dim_t count) {
        for (dim_t i = dim_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
             i += dim_t(blockDim.x) * gridDim.x)
          tile[i] = x[(i / count) * time + start + i % count];
      }

      // FishDAC's GEMM + col2im, generalized to padding/dilation and time tiles.
      // Each thread owns one output value; no atomics or kernel reversal.
      __global__ void col2im(const float* columns, const float* bias, float* output,
                             dim_t size, dim_t length, dim_t kernel, dim_t stride,
                             dim_t padding, dim_t dilation, dim_t start, dim_t count) {
        for (dim_t i = dim_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
             i += dim_t(blockDim.x) * gridDim.x) {
          const dim_t oc = i / length, pos = i % length;
          float sum = start ? output[i] : (bias ? bias[oc] : 0.f);
          for (dim_t k = 0; k < kernel; ++k) {
            const dim_t source = pos + padding - k * dilation;
            if (source < 0 || source % stride)
              continue;
            const dim_t t = source / stride - start;
            if (t >= 0 && t < count)
              sum += columns[(oc * kernel + k) * count + t];
          }
          output[i] = sum;
        }
      }

    }

    template <Device D, typename T>
    void ConvTranspose1D::compute(const StorageView& input, const StorageView& weight,
                                 const StorageView* bias, StorageView& output) const {
      const dim_t batch = input.dim(0), cin = input.dim(1), time = input.dim(2);
      const dim_t cin_group = cin / _groups, cout_group = weight.dim(1), kernel = weight.dim(2);
      const dim_t cout = output.dim(1), length = output.dim(2);
      const dim_t contribution_rows = cout_group * kernel;
      // Bound temporary storage to 16 MiB, except when a single time step is larger.
      constexpr dim_t workspace = 16 * 1024 * 1024 / sizeof(T);
      const dim_t tile_size = std::min(time, std::max(dim_t(1), workspace / (cin_group + contribution_rows)));
      StorageView columns(DataType::FLOAT32, D), packed(DataType::FLOAT32, D);
      const Gemm gemm(1.f, 0.f, true, false);
      for (dim_t n = 0; n < batch; ++n) {
        for (dim_t group = 0; group < _groups; ++group) {
          StorageView w({cin_group, contribution_rows},
                        const_cast<T*>(weight.data<T>() + group * cin_group * contribution_rows), D);
          const auto* x = input.data<T>() + (n * cin + group * cin_group) * time;
          for (dim_t start = 0; start < time; start += tile_size) {
            const dim_t count = std::min(tile_size, time - start);
            StorageView tile(DataType::FLOAT32, D);
            if (count == time) {
              tile.view(const_cast<T*>(x), {cin_group, count});
            } else {
              packed.resize({cin_group, count});
              const dim_t blocks = std::min((packed.size() + 255) / 256, dim_t(65535));
              pack_tile<<<blocks, 256, 0, cuda::get_cuda_stream()>>>(
                  x, packed.data<T>(), packed.size(), time, start, count);
              CUDA_CHECK(cudaGetLastError());
              tile.view(packed.data<T>(), packed.shape());
            }
            gemm(w, tile, columns);
            const dim_t size = cout_group * length;
            const dim_t blocks = std::min((size + 255) / 256, dim_t(65535));
            col2im<<<blocks, 256, 0, cuda::get_cuda_stream()>>>(
                columns.data<T>(), bias ? bias->data<T>() + group * cout_group : nullptr,
                output.data<T>() + (n * cout + group * cout_group) * length, size, length,
                kernel, _stride, _padding, _dilation, start, count);
            CUDA_CHECK(cudaGetLastError());
          }
        }
      }
    }

    template void ConvTranspose1D::compute<Device::CUDA, float>(
        const StorageView&, const StorageView&, const StorageView*, StorageView&) const;

  }
}
