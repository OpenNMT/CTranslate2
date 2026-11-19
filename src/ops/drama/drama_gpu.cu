#include "ops/drama/drama_gpu.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      template <typename T>
      __global__ void paired_projection(const float* input, const T* wa, const T* wb,
                                        float* a, float* b, dim_t hidden) {
        const dim_t row = blockIdx.x;
        float sum_a = 0, sum_b = 0;
        for (dim_t i = threadIdx.x; i < hidden; i += blockDim.x) {
          // Match input conversion, then accumulate products in float.
          const float x = float(T(input[i]));
          sum_a += x * float(wa[row * hidden + i]);
          sum_b += x * float(wb[row * hidden + i]);
        }
        __shared__ float partial_a[256], partial_b[256];
        partial_a[threadIdx.x] = sum_a;
        partial_b[threadIdx.x] = sum_b;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
          if (threadIdx.x < stride) {
            partial_a[threadIdx.x] += partial_a[threadIdx.x + stride];
            partial_b[threadIdx.x] += partial_b[threadIdx.x + stride];
          }
          __syncthreads();
        }
        if (!threadIdx.x) {
          // The separate GEMMs also round their outputs to the compute type.
          a[row] = float(T(partial_a[0]));
          b[row] = float(T(partial_b[0]));
        }
      }

      __global__ void conv(const float* x, const float* weight, const float* bias, float* history,
                           float* output, dim_t steps, dim_t channels, dim_t kernel) {
        const dim_t c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= channels)
          return;
        const dim_t past = kernel - 1;
        float* ring = history + c * past;
        dim_t cursor = 0;
        for (dim_t t = 0; t < steps; ++t) {
          float value = bias ? bias[c] : 0.f;
          for (dim_t k = 0; k < past; ++k)
            value += ring[(cursor + k) % past] * weight[c * kernel + k];
          value += x[t * channels + c] * weight[c * kernel + past];
          output[t * channels + c] = value / (1.f + expf(-value));
          if (past) {
            ring[cursor] = x[t * channels + c];
            cursor = (cursor + 1) % past;
          }
        }
        // Rotate in place to the chronological history layout without a scratch array.
        // Three reversals implement a left rotation by cursor.
        for (dim_t i = 0; i < cursor / 2; ++i) {
          float tmp = ring[i];
          ring[i] = ring[cursor - 1 - i];
          ring[cursor - 1 - i] = tmp;
        }
        for (dim_t i = 0; i < (past - cursor) / 2; ++i) {
          float tmp = ring[cursor + i];
          ring[cursor + i] = ring[past - 1 - i];
          ring[past - 1 - i] = tmp;
        }
        for (dim_t i = 0; i < past / 2; ++i) {
          float tmp = ring[i];
          ring[i] = ring[past - 1 - i];
          ring[past - 1 - i] = tmp;
        }
      }

      __global__ void normalize_qk(float* mixed, dim_t steps, dim_t heads, dim_t depth,
                                   dim_t channels) {
        const dim_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= steps * heads * 2)
          return;
        const dim_t t = row / (heads * 2), h = row % (heads * 2);
        float* values = mixed + t * channels + h * depth;
        float sum = 1e-6f;
        for (dim_t d = 0; d < depth; ++d)
          sum += values[d] * values[d];
        const float inverse = 1.f / sqrtf(sum);
        for (dim_t d = 0; d < depth; ++d)
          values[d] *= inverse;
      }

      __device__ float sigmoid(float x) {
        if (x >= 0)
          return 1.f / (1.f + expf(-x));
        const float e = expf(x);
        return e / (1.f + e);
      }

      __global__ void delta(const float* mixed, const float* a, const float* b, const float* a_log,
                            const float* dt_bias, float* state, float* out, dim_t steps,
                            dim_t key_heads, dim_t heads, dim_t kd, dim_t vd) {
        const dim_t h = blockIdx.x, j = threadIdx.x;
        if (j >= vd)
          return;
        const dim_t kw = key_heads * kd, vw = heads * vd, channels = 2 * kw + vw;
        float* memory = state + h * kd * vd + j;
        const float scale = 1.f / sqrtf(float(kd));
        for (dim_t t = 0; t < steps; ++t) {
          const dim_t row = t * heads + h;
          const float av = a[row] + dt_bias[h];
          const float decay = expf(-expf(a_log[h]) * (fmaxf(av, 0.f) + log1pf(expf(-fabsf(av)))));
          const float beta = sigmoid(b[row]);
          const auto* q = mixed + t * channels + (h / (heads / key_heads)) * kd;
          const auto* k = q + kw;
          const auto* v = mixed + t * channels + 2 * kw + h * vd;
          float prediction = 0.f;
          for (dim_t i = 0; i < kd; ++i) {
            memory[i * vd] *= decay;
            prediction += k[i] * memory[i * vd];
          }
          const float residual = beta * (v[j] - prediction);
          float value = 0.f;
          for (dim_t i = 0; i < kd; ++i) {
            memory[i * vd] += k[i] * residual;
            value += q[i] * scale * memory[i * vd];
          }
          out[row * vd + j] = value;
        }
      }

      template <int KeyDim>
      __global__ void delta_cached(const float* mixed, const float* a, const float* b,
                                   const float* a_log, const float* dt_bias, float* state,
                                   float* out, dim_t steps, dim_t key_heads, dim_t heads, dim_t kd,
                                   dim_t vd) {
        const dim_t h = blockIdx.x, j = threadIdx.x;
        if (j >= vd)
          return;
        const dim_t kw = key_heads * kd, vw = heads * vd, channels = 2 * kw + vw;
        float* destination = state + h * kd * vd + j;
        float memory[KeyDim];
#pragma unroll
        for (int i = 0; i < KeyDim; ++i)
          memory[i] = destination[i * vd];
        const float scale = 1.f / sqrtf(float(kd));
        for (dim_t t = 0; t < steps; ++t) {
          const dim_t row = t * heads + h;
          const float av = a[row] + dt_bias[h];
          const float decay = expf(-expf(a_log[h]) * (fmaxf(av, 0.f) + log1pf(expf(-fabsf(av)))));
          const float beta = sigmoid(b[row]);
          const auto* q = mixed + t * channels + (h / (heads / key_heads)) * kd;
          const auto* k = q + kw;
          const auto* v = mixed + t * channels + 2 * kw + h * vd;
          float prediction = 0.f;

#pragma unroll
          for (int i = 0; i < KeyDim; ++i) {
            memory[i] *= decay;
            prediction += k[i] * memory[i];
          }
          const float residual = beta * (v[j] - prediction);
          float value = 0.f;

#pragma unroll
          for (int i = 0; i < KeyDim; ++i) {
            memory[i] += k[i] * residual;
            value += q[i] * scale * memory[i];
          }
          out[row * vd + j] = value;
        }
#pragma unroll
        for (int i = 0; i < KeyDim; ++i)
          destination[i * vd] = memory[i];
      }

      __global__ void swigate(float* core, const float* z, dim_t n) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          core[i] *= z[i] * sigmoid(z[i]);
      }

      __global__ void extract(const float* query, float* out, dim_t n, dim_t depth) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
          out[i] = query[(i / depth) * 2 * depth + i % depth];
      }

      __global__ void append(const float* keys, const float* values, float* ck, float* cv, dim_t n,
                             dim_t offset) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
          ck[offset + i] = keys[i];
          cv[offset + i] = values[i];
        }
      }

      __global__ void pack_q(const float* query, float* out, int32_t* lengths, dim_t steps,
                             dim_t heads, dim_t depth, dim_t past) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= steps * heads * depth)
          return;
        const dim_t d = i % depth, t = (i / depth) % steps, h = i / (depth * steps);
        out[i] = query[(t * heads + h) * depth + d];
        if (d == 0)
          lengths[h * steps + t] = past + t + 1;
      }

      __global__ void pack_kv(const float* ck, const float* cv, float* pk, float* pv, dim_t length,
                              dim_t heads, dim_t depth) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= length * heads * depth)
          return;
        const dim_t d = i % depth, t = (i / depth) % length, h = i / (depth * length);
        const dim_t source = (t * heads + h) * depth + d;
        pk[i] = ck[source];
        pv[i] = cv[source];
      }

      __global__ void attention_gate(const float* context, const float* query, float* out,
                                     dim_t steps, dim_t heads, dim_t depth) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= steps * heads * depth)
          return;
        const dim_t d = i % depth, h = (i / depth) % heads, t = i / (depth * heads);
        out[i] = context[(h * steps + t) * depth + d]
                 / (1.f + expf(-query[(t * heads + h) * 2 * depth + depth + d]));
      }

    }

    bool drama_paired_projection_gpu(const StorageView& input, const StorageView& wa,
                                     const StorageView& wb, StorageView& a, StorageView& b) {
      if (input.rank() != 2 || input.dim(0) != 1 || input.dtype() != DataType::FLOAT32
          || wa.rank() != 2 || wa.shape() != wb.shape() || wa.dtype() != wb.dtype()
          || wa.dim(1) != input.dim(1) || wa.dim(0) <= 0
          || (wa.dtype() != DataType::FLOAT32 && wa.dtype() != DataType::FLOAT16
              && wa.dtype() != DataType::BFLOAT16)
          || (wa.dtype() == DataType::FLOAT16 && cuda::use_true_fp16_gemm()))
        return false;
      a.resize({1, wa.dim(0)});
      b.resize_as(a);
#define LAUNCH_PAIR(TYPE) \
      paired_projection<<<wa.dim(0), 256, 0, cuda::get_cuda_stream()>>>( \
          input.data<float>(), cuda::device_cast(wa.data<TYPE>()), \
          cuda::device_cast(wb.data<TYPE>()), a.data<float>(), b.data<float>(), wa.dim(1))
      switch (wa.dtype()) {
      case DataType::FLOAT32: LAUNCH_PAIR(float); break;
      case DataType::FLOAT16: LAUNCH_PAIR(float16_t); break;
      case DataType::BFLOAT16: LAUNCH_PAIR(bfloat16_t); break;
      default: return false;
      }
#undef LAUNCH_PAIR
      CUDA_CHECK(cudaGetLastError());
      return true;
    }

    void drama_conv_gpu(const StorageView& x, const StorageView& weight, const StorageView* bias,
                        StorageView& history, StorageView& output) {
      const dim_t channels = x.dim(-1);
      output.resize_as(x);
      conv<<<(channels + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          x.data<float>(), weight.data<float>(), bias ? bias->data<float>() : nullptr,
          history.data<float>(), output.data<float>(), x.dim(0), channels, weight.dim(1));
      CUDA_CHECK(cudaGetLastError());
    }

    void drama_delta_gpu(StorageView& mixed, const StorageView& a, const StorageView& b,
                         const StorageView& a_log, const StorageView& dt_bias, StorageView& state,
                         dim_t kh, dim_t heads, dim_t kd, dim_t vd, StorageView& core) {
      const dim_t steps = mixed.dim(0);
      core.resize({steps, heads, vd});
      normalize_qk<<<(steps * kh * 2 + 127) / 128, 128, 0, cuda::get_cuda_stream()>>>(
          mixed.data<float>(), steps, kh, kd, mixed.dim(1));
      if (kd == 128)
        delta_cached<128><<<heads, vd, 0, cuda::get_cuda_stream()>>>(
            mixed.data<float>(), a.data<float>(), b.data<float>(), a_log.data<float>(),
            dt_bias.data<float>(), state.data<float>(), core.data<float>(), steps, kh, heads, kd,
            vd);
      else if (kd == 64)
        delta_cached<64><<<heads, vd, 0, cuda::get_cuda_stream()>>>(
            mixed.data<float>(), a.data<float>(), b.data<float>(), a_log.data<float>(),
            dt_bias.data<float>(), state.data<float>(), core.data<float>(), steps, kh, heads, kd,
            vd);
      else
        delta<<<heads, vd, 0, cuda::get_cuda_stream()>>>(
            mixed.data<float>(), a.data<float>(), b.data<float>(), a_log.data<float>(),
            dt_bias.data<float>(), state.data<float>(), core.data<float>(), steps, kh, heads, kd,
            vd);
      CUDA_CHECK(cudaGetLastError());
    }

    void drama_swigate_gpu(StorageView& core, const StorageView& z) {
      swigate<<<(core.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          core.data<float>(), z.data<float>(), core.size());
      CUDA_CHECK(cudaGetLastError());
    }

    void drama_extract_queries_gpu(const StorageView& query, dim_t heads, dim_t depth,
                                   StorageView& out) {
      out.resize({query.dim(0), heads, depth});
      extract<<<(out.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          query.data<float>(), out.data<float>(), out.size(), depth);
      CUDA_CHECK(cudaGetLastError());
    }

    void drama_pack_attention_gpu(const StorageView& query, const StorageView& keys,
                                  const StorageView& values, StorageView& ck, StorageView& cv,
                                  dim_t past, dim_t heads, dim_t kvh, dim_t depth, StorageView& pq,
                                  StorageView& pk, StorageView& pv, StorageView& lengths) {
      const dim_t steps = query.dim(0), length = past + steps;
      pq.resize({1, kvh, (heads / kvh) * steps, depth});
      pk.resize({1, kvh, length, depth});
      pv.resize_as(pk);
      lengths.resize({heads * steps});
      append<<<(keys.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          keys.data<float>(), values.data<float>(), ck.data<float>(), cv.data<float>(), keys.size(),
          past * kvh * depth);
      pack_q<<<(query.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          query.data<float>(), pq.data<float>(), lengths.data<int32_t>(), steps, heads, depth,
          past);
      pack_kv<<<(length * kvh * depth + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          ck.data<float>(), cv.data<float>(), pk.data<float>(), pv.data<float>(), length, kvh,
          depth);
      CUDA_CHECK(cudaGetLastError());
    }

    void drama_attention_gate_gpu(const StorageView& context, const StorageView& query, dim_t heads,
                                  dim_t depth, StorageView& out) {
      const dim_t steps = query.dim(0);
      out.resize({steps, heads * depth});
      attention_gate<<<(out.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          context.data<float>(), query.data<float>(), out.data<float>(), steps, heads, depth);
      CUDA_CHECK(cudaGetLastError());
    }

  }
}
