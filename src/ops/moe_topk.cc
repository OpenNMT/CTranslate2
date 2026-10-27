#include "ctranslate2/ops/moe_topk.h"

#include <limits>
#include <stdexcept>

#include "device_dispatch.h"
#include "storage_view_overlap.h"

namespace ctranslate2 {
  namespace ops {

    MoeTopK::MoeTopK(dim_t k)
      : _k(k) {
      if (k <= 0)
        throw std::invalid_argument("MoeTopK k must be positive");
    }

    void MoeTopK::operator()(const StorageView& router_logits, StorageView& expert_indices,
                             StorageView& routing_weights) const {
      PROFILE("MoeTopK");
      if (expert_indices.device() != router_logits.device()
          || routing_weights.device() != router_logits.device()
          || expert_indices.device_index() != router_logits.device_index()
          || routing_weights.device_index() != router_logits.device_index()
          || router_logits.dtype() != DataType::FLOAT32
          || expert_indices.dtype() != DataType::INT32
          || routing_weights.dtype() != DataType::FLOAT32)
        throw std::invalid_argument(
            "MoeTopK requires float32 logits/weights and int32 indices on the same device");
      if (router_logits.rank() != 2 || router_logits.dim(0) < 0 || router_logits.dim(1) <= 0
          || _k > router_logits.dim(1)
          || router_logits.dim(1) > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument(
            "MoeTopK expects [rows, experts] with 0 < k <= experts <= INT32_MAX");
      const auto input_bytes = router_logits.size() * router_logits.item_size();
      for (const auto* output : {&expert_indices, &routing_weights}) {
        if (&router_logits == output
            || storage_views_overlap(router_logits, input_bytes, *output, output->reserved_memory()))
          throw std::invalid_argument("MoeTopK outputs must not alias logits");
      }
      if (&expert_indices == &routing_weights
          || storage_views_overlap(expert_indices, expert_indices.reserved_memory(), routing_weights,
                                   routing_weights.reserved_memory()))
        throw std::invalid_argument("MoeTopK outputs must not alias each other");
      const ScopedDeviceSetter device_setter(router_logits.device(), router_logits.device_index());
      DEVICE_DISPATCH(router_logits.device(),
                      (compute<D, float>(router_logits, expert_indices, routing_weights)));
    }

  }
}
