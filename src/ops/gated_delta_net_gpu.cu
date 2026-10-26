#include "ctranslate2/ops/gated_delta_net.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      // Drama3's column-wise recurrent update, operating on the public op's
      // separate Q/K/V and precomputed gates instead of fused projections.
      // KeyDim > 0 holds state in a per-thread array for 64/128-wide keys.
      template <int KeyDim>
      __global__ void delta(const float* q, const float* k, const float* v, const float* g,
                            const float* beta, float* state, float* out, dim_t columns,
                            dim_t steps, dim_t key_heads, dim_t heads, dim_t kd, dim_t vd) {
        for (dim_t column = dim_t(blockIdx.x) * blockDim.x + threadIdx.x; column < columns;
             column += dim_t(blockDim.x) * gridDim.x) {
          const dim_t bh = column / vd, j = column % vd, n = bh / heads, h = bh % heads;
          float* destination = state + bh * kd * vd + j;
          float memory[KeyDim ? KeyDim : 1];
          if (KeyDim)
            for (int i = 0; i < KeyDim; ++i)
              memory[i] = destination[i * vd];
          const float scale = 1.f / sqrtf(float(kd));
          const dim_t key_dim = KeyDim ? KeyDim : kd;
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t row = (n * steps + t) * heads + h;
            const dim_t key_row = (n * steps + t) * key_heads + h / (heads / key_heads);
            const auto* qt = q + key_row * kd;
            const auto* kt = k + key_row * kd;
            const float decay = expf(g[row]);
            float prediction = 0.f;
            for (dim_t i = 0; i < key_dim; ++i) {
              float& cell = KeyDim ? memory[i] : destination[i * vd];
              cell *= decay;
              prediction += kt[i] * cell;
            }
            const float residual = beta[row] * (v[row * vd + j] - prediction);
            float value = 0.f;
            for (dim_t i = 0; i < key_dim; ++i) {
              float& cell = KeyDim ? memory[i] : destination[i * vd];
              cell += kt[i] * residual;
              value += qt[i] * scale * cell;
            }
            out[row * vd + j] = value;
          }
          if (KeyDim)
            for (int i = 0; i < KeyDim; ++i)
              destination[i * vd] = memory[i];
        }
      }

    }

    template <Device D, typename T>
    void GatedDeltaNet::compute(const StorageView& query, const StorageView& key,
                               const StorageView& value, const StorageView& log_decay,
                               const StorageView& beta, StorageView& state,
                               StorageView& output) const {
      output.resize_as(value);
      if (value.empty())
        return;
      const dim_t columns = state.dim(0) * state.dim(1) * state.dim(3);
      const int threads = query.dim(3) == 64 || query.dim(3) == 128 ? 32 : 128;
      const dim_t blocks = std::min((columns + threads - 1) / threads, dim_t(65535));
#define LAUNCH_DELTA(KD)                                                          \
      delta<KD><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(                   \
          query.data<T>(), key.data<T>(), value.data<T>(), log_decay.data<T>(),    \
          beta.data<T>(), state.data<T>(), output.data<T>(), columns, query.dim(1), \
          query.dim(2), value.dim(2), query.dim(3), value.dim(3))
      if (query.dim(3) == 128) {
        LAUNCH_DELTA(128);
      } else if (query.dim(3) == 64) {
        LAUNCH_DELTA(64);
      } else {
        LAUNCH_DELTA(0);
      }
#undef LAUNCH_DELTA
      CUDA_CHECK(cudaGetLastError());
    }

    template void GatedDeltaNet::compute<Device::CUDA, float>(
        const StorageView&, const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&, StorageView&) const;

  }
}
