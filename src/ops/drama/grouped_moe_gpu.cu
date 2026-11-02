#include "ops/drama/grouped_moe.h"
#include "ctranslate2/ops/quantize.h"
#include "cuda/helpers.h"


namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void accumulate(const float* expert, float probability, const float* shared_logit,
                                 float* output, dim_t size) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size)
          return;
        const float weight = shared_logit ? 1.f / (1.f + expf(-*shared_logit)) : probability;
        output[i] = __fadd_rn(output[i], __fmul_rn(weight, expert[i]));
      }

      // One warp per output feature; eight independent dots per block. Integer
      // accumulation matches GEMM exactly (no floating reduction reassociation).
      __device__ int dot(const int8_t* x, const int8_t* w, dim_t n) {
        int value = 0;
        if (n % 4 == 0) {
          for (dim_t i = threadIdx.x % 32; i < n / 4; i += 32) {
#ifdef CT2_USE_HIP
            value = amd_mixed_dot(reinterpret_cast<const char4*>(x)[i],
                                  reinterpret_cast<const char4*>(w)[i], value, false);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
            value = __dp4a(reinterpret_cast<const int*>(x)[i], reinterpret_cast<const int*>(w)[i],
                           value);
#else
            for (int j = 0; j < 4; ++j)
              value += int(x[4 * i + j]) * int(w[4 * i + j]);
#endif
          }
        } else {
          for (dim_t i = threadIdx.x % 32; i < n; i += 32)
            value += int(x[i]) * int(w[i]);
        }
        for (int offset = 16; offset; offset /= 2) {
#ifdef CT2_USE_HIP
          value += __shfl_down(value, offset, 32);
#else
          value += __shfl_down_sync(0xffffffff, value, offset, 32);
#endif
        }
        return value;
      }

      __global__ void gate_up(const int8_t* input, const float* input_scale,
                              const MoeExpertWeights* experts, float* intermediate, dim_t hidden,
                              dim_t width, const int32_t* indices, dim_t k, dim_t shared_index) {
        const dim_t feature = blockIdx.x * 8 + threadIdx.x / 32;
        if (feature >= width)
          return;
        const auto& e = experts[indices ? (blockIdx.y == k ? shared_index
                                                           : indices[blockIdx.z * k + blockIdx.y])
                                        : blockIdx.y];
        const int gate_sum = dot(input + blockIdx.z * hidden, e.gate + feature * hidden, hidden);
        const int up_sum = dot(input + blockIdx.z * hidden, e.up + feature * hidden, hidden);
        if (threadIdx.x % 32 == 0) {
          const float gate =
              __fdividef(float(gate_sum), input_scale[blockIdx.z] * e.gate_scale[feature]);
          const float up = __fdividef(float(up_sum), input_scale[blockIdx.z] * e.up_scale[feature]);
          intermediate[(blockIdx.z * (k + 1) + blockIdx.y) * width + feature] =
              (gate / (1.f + expf(-gate))) * up;
        }
      }

      __global__ void down(const int8_t* input, const float* scales,
                           const MoeExpertWeights* experts, float* results, dim_t hidden,
                           dim_t width, const int32_t* indices, dim_t k, dim_t shared_index) {
        const dim_t feature = blockIdx.x * 8 + threadIdx.x / 32;
        if (feature >= hidden)
          return;
        const auto& e = experts[indices ? (blockIdx.y == k ? shared_index
                                                           : indices[blockIdx.z * k + blockIdx.y])
                                        : blockIdx.y];
        const dim_t row = blockIdx.z * (k + 1) + blockIdx.y;
        const int sum = dot(input + row * width, e.down + feature * width, width);
        if (threadIdx.x % 32 == 0)
          results[row * hidden + feature] =
              __fdividef(float(sum), scales[row] * e.down_scale[feature]);
      }

      __global__ void combine(const float* results, const float* probabilities,
                              const float* shared_logit, float* output, dim_t hidden, dim_t k,
                              dim_t rows) {
        const dim_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= rows * hidden)
          return;
        const dim_t d = index % hidden, row = index / hidden;
        float sum = 0.f;
        for (dim_t e = 0; e < k; ++e)
          sum = __fadd_rn(sum, __fmul_rn(probabilities[row * k + e],
                                         results[(row * (k + 1) + e) * hidden + d]));
        const float gate = 1.f / (1.f + expf(-shared_logit[row]));
        output[index] = __fadd_rn(sum, __fmul_rn(gate, results[(row * (k + 1) + k) * hidden + d]));
      }

    }

    void moe_accumulate_gpu(const StorageView& expert, float probability, const float* shared_logit,
                            StorageView& output) {
      if (expert.empty())
        return;
      accumulate<<<(expert.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          expert.data<float>(), probability, shared_logit, output.data<float>(), expert.size());
      CUDA_CHECK(cudaGetLastError());
    }

    void grouped_moe_gpu(const StorageView& input, const StorageView& descriptors,
                         const StorageView& weights, const StorageView& shared_logit, dim_t hidden,
                         dim_t width, dim_t k, StorageView& output, const int32_t* indices,
                         dim_t shared_index) {
      StorageView qinput(DataType::INT8, Device::CUDA),
          input_scale(DataType::FLOAT32, Device::CUDA);
      Quantize(Quantize::ScaleType::PER_LAYER, false)(input, qinput, input_scale);
      const dim_t rows = input.dim(0);
      StorageView intermediate({rows * (k + 1), width}, DataType::FLOAT32, Device::CUDA);
      const auto* experts = reinterpret_cast<const MoeExpertWeights*>(descriptors.buffer());
      gate_up<<<dim3((width + 7) / 8, k + 1, rows), 256, 0, cuda::get_cuda_stream()>>>(
          qinput.data<int8_t>(), input_scale.data<float>(), experts, intermediate.data<float>(),
          hidden, width, indices, k, shared_index);
      CUDA_CHECK(cudaGetLastError());
      StorageView qmid(DataType::INT8, Device::CUDA), mid_scale(DataType::FLOAT32, Device::CUDA);
      Quantize(Quantize::ScaleType::PER_LAYER, false)(intermediate, qmid, mid_scale);
      StorageView results({rows * (k + 1), hidden}, DataType::FLOAT32, Device::CUDA);
      down<<<dim3((hidden + 7) / 8, k + 1, rows), 256, 0, cuda::get_cuda_stream()>>>(
          qmid.data<int8_t>(), mid_scale.data<float>(), experts, results.data<float>(), hidden,
          width, indices, k, shared_index);
      CUDA_CHECK(cudaGetLastError());
      output.resize({rows, hidden});
      combine<<<(rows * hidden + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          results.data<float>(), weights.data<float>(), shared_logit.data<float>(),
          output.data<float>(), hidden, k, rows);
      CUDA_CHECK(cudaGetLastError());
    }

  }
}
