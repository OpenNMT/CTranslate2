#include "ctranslate2/ops/causal_depthwise_conv1d.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void conv(const float* x, const float* weight, const float* bias,
                           float* history, float* output, dim_t tasks, dim_t steps,
                           dim_t channels, dim_t kernel, bool apply_silu) {
        for (dim_t bc = dim_t(blockIdx.x) * blockDim.x + threadIdx.x; bc < tasks;
             bc += dim_t(blockDim.x) * gridDim.x) {
          const dim_t n = bc / channels, c = bc % channels, past = kernel - 1;
          float* ring = past ? history + bc * past : nullptr;
          dim_t cursor = 0;
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t offset = (n * steps + t) * channels + c;
            float value = bias ? bias[c] : 0.f;
            for (dim_t k = 0; k < past; ++k)
              value += ring[(cursor + k) % past] * weight[c * kernel + k];
            value += x[offset] * weight[c * kernel + past];
            output[offset] = apply_silu ? value / (1.f + expf(-value)) : value;
            if (past) {
              ring[cursor] = x[offset];
              cursor = (cursor + 1) % past;
            }
          }
          // Rotate the ring back into oldest-to-newest order, without scratch.
          for (dim_t i = 0; i < cursor / 2; ++i) {
            const float tmp = ring[i];
            ring[i] = ring[cursor - 1 - i];
            ring[cursor - 1 - i] = tmp;
          }
          for (dim_t i = 0; i < (past - cursor) / 2; ++i) {
            const float tmp = ring[cursor + i];
            ring[cursor + i] = ring[past - 1 - i];
            ring[past - 1 - i] = tmp;
          }
          for (dim_t i = 0; i < past / 2; ++i) {
            const float tmp = ring[i];
            ring[i] = ring[past - 1 - i];
            ring[past - 1 - i] = tmp;
          }
        }
      }

    }

    template <Device D, typename T>
    void CausalDepthwiseConv1D::compute(const StorageView& input, const StorageView& weight,
                                      const StorageView* bias, StorageView& history,
                                      StorageView& output) const {
      output.resize_as(input);
      if (input.empty())
        return;
      const dim_t tasks = input.dim(0) * input.dim(2);
      const dim_t blocks = std::min((tasks + 255) / 256, dim_t(65535));
      conv<<<blocks, 256, 0, cuda::get_cuda_stream()>>>(
          input.data<T>(), weight.data<T>(), bias ? bias->data<T>() : nullptr,
          history.data<T>(), output.data<T>(), tasks, input.dim(1), input.dim(2), weight.dim(1),
          _apply_silu);
      CUDA_CHECK(cudaGetLastError());
    }

    template void CausalDepthwiseConv1D::compute<Device::CUDA, float>(
        const StorageView&, const StorageView&, const StorageView*, StorageView&, StorageView&) const;

  }
}
