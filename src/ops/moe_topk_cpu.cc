#include "ctranslate2/ops/moe_topk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cpu/parallel.h"
#include "finite_check.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void MoeTopK::compute(const StorageView& router_logits, StorageView& expert_indices,
                          StorageView& routing_weights) const {
      const auto* logits = router_logits.data<T>();
      if (!all_finite(logits, router_logits.size()))
        throw std::invalid_argument("MoeTopK logits must be finite");

      const dim_t rows = router_logits.dim(0);
      const dim_t experts = router_logits.dim(1);
      expert_indices.resize({rows, _k});
      routing_weights.resize({rows, _k});
      auto* indices = expert_indices.data<int32_t>();
      auto* weights = routing_weights.data<T>();
      cpu::parallel_for(0, rows, 1, [&](dim_t begin, dim_t end) {
        std::vector<int32_t> order(experts);
        std::vector<double> probabilities(_k);
        for (dim_t row = begin; row < end; ++row) {
          const auto* values = logits + row * experts;
          std::iota(order.begin(), order.end(), int32_t(0));
          std::partial_sort(order.begin(), order.begin() + _k, order.end(),
                            [&](int32_t a, int32_t b) {
                              return values[a] > values[b] || (values[a] == values[b] && a < b);
                            });
          const double maximum = values[order[0]];
          double sum = 0;
          for (dim_t i = 0; i < _k; ++i) {
            probabilities[i] = std::exp(static_cast<double>(values[order[i]]) - maximum);
            sum += probabilities[i];
          }
          for (dim_t i = 0; i < _k; ++i) {
            indices[row * _k + i] = order[i];
            weights[row * _k + i] = static_cast<T>(probabilities[i] / sum);
          }
        }
      });
    }

    template void MoeTopK::compute<Device::CPU, float>(const StorageView&,
                                                       StorageView&,
                                                       StorageView&) const;

  }
}
