#include "ctranslate2/ops/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GatedDeltaNet::compute(const StorageView& query, const StorageView& key,
                               const StorageView& value, const StorageView& log_decay,
                               const StorageView& beta, StorageView& state,
                               StorageView& output) const {
      const dim_t batch = query.dim(0), steps = query.dim(1);
      const dim_t key_heads = query.dim(2), key_dim = query.dim(3);
      const dim_t heads = value.dim(2), value_dim = value.dim(3);
      output.resize({batch, steps, heads, value_dim});
      const auto* q = query.data<T>();
      const auto* k = key.data<T>();
      const auto* v = value.data<T>();
      const auto* g = log_decay.data<T>();
      const auto* b = beta.data<T>();
      auto* s = state.data<T>();
      auto* out = output.data<T>();
      const float scale = 1.f / std::sqrt(static_cast<float>(key_dim));
      const dim_t heads_per_key = heads / key_heads;

      cpu::parallel_for(0, batch * heads, 1, [&](dim_t begin, dim_t end) {
        std::vector<float> residual(value_dim);
        for (dim_t bh = begin; bh < end; ++bh) {
          const dim_t batch_index = bh / heads;
          const dim_t head = bh % heads;
          auto* memory = s + bh * key_dim * value_dim;
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t row = (batch_index * steps + t) * heads + head;
            const dim_t key_row = (batch_index * steps + t) * key_heads + head / heads_per_key;
            const auto* qt = q + key_row * key_dim;
            const auto* kt = k + key_row * key_dim;
            const auto* vt = v + row * value_dim;
            auto* ot = out + row * value_dim;
            const float decay = std::exp(g[row]);
            for (dim_t i = 0; i < key_dim * value_dim; ++i)
              memory[i] *= decay;
            // Accumulate each column in the original key-dimension order while
            // traversing contiguous state rows, allowing SIMD across values.
            std::fill(residual.begin(), residual.end(), 0.f);
            for (dim_t i = 0; i < key_dim; ++i) {
              const auto* memory_row = memory + i * value_dim;
              for (dim_t j = 0; j < value_dim; ++j)
                residual[j] += kt[i] * memory_row[j];
            }
            for (dim_t j = 0; j < value_dim; ++j) {
              residual[j] = b[row] * (vt[j] - residual[j]);
              ot[j] = 0;
            }
            for (dim_t i = 0; i < key_dim; ++i) {
              for (dim_t j = 0; j < value_dim; ++j) {
                auto& cell = memory[i * value_dim + j];
                cell += kt[i] * residual[j];
                ot[j] += qt[i] * scale * cell;
              }
            }
          }
        }
      });
    }

    template void GatedDeltaNet::compute<Device::CPU, float>(
        const StorageView&, const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&, StorageView&) const;


  }
}
