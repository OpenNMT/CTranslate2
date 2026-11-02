#include "ctranslate2/ops/gated_delta_net.h"

#include <stdexcept>

#include "device_dispatch.h"
#include "storage_view_overlap.h"

namespace ctranslate2 {
  namespace ops {

    void GatedDeltaNet::validate_input(const StorageView& query, const StorageView& key,
                                       const StorageView& value, const StorageView& log_decay,
                                       const StorageView& beta, const StorageView& state,
                                       const StorageView& output) const {
      const StorageView* inputs[] = {&query, &key, &value, &log_decay, &beta};
      for (const auto* tensor :
           {&query, &key, &value, &log_decay, &beta, static_cast<const StorageView*>(&state),
            static_cast<const StorageView*>(&output)}) {
        if (tensor->device() != query.device()
            || tensor->device_index() != query.device_index()
            || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("GatedDeltaNet requires float32 tensors on the same device");
      }
      if (query.rank() != 4 || key.shape() != query.shape() || value.rank() != 4)
        throw std::invalid_argument(
            "GatedDeltaNet expects matching rank-4 query/key and rank-4 value");
      const dim_t batch = query.dim(0);
      const dim_t steps = query.dim(1);
      const dim_t key_heads = query.dim(2);
      const dim_t key_dim = query.dim(3);
      const dim_t heads = value.dim(2);
      const dim_t value_dim = value.dim(3);
      if (batch <= 0 || steps < 0 || key_heads <= 0 || key_dim <= 0 || heads <= 0 || value_dim <= 0
          || heads % key_heads != 0 || value.dim(0) != batch || value.dim(1) != steps)
        throw std::invalid_argument("GatedDeltaNet has incompatible query/value dimensions");
      if (log_decay.shape() != Shape{batch, steps, heads} || beta.shape() != log_decay.shape())
        throw std::invalid_argument(
            "GatedDeltaNet gates must have shape [batch, time, value_heads]");
      if (state.shape() != Shape{batch, heads, key_dim, value_dim})
        throw std::invalid_argument(
            "GatedDeltaNet state must have shape [batch, value_heads, key_dim, value_dim]");
      if (&state == &output || storage_views_overlap(state, output))
        throw std::invalid_argument("GatedDeltaNet state and output must not alias");
      for (const auto* input : inputs) {
        const dim_t input_bytes = input->size() * input->item_size();
        if (&output == input
            || storage_views_overlap(*input, input_bytes, state, state.reserved_memory())
            || storage_views_overlap(*input, input_bytes, output, output.reserved_memory()))
          throw std::invalid_argument("GatedDeltaNet inputs must not alias state or output");
      }
    }

    void GatedDeltaNet::operator()(const StorageView& query, const StorageView& key,
                                   const StorageView& value, const StorageView& log_decay,
                                   const StorageView& beta, StorageView& state,
                                   StorageView& output) const {
      PROFILE("GatedDeltaNet");
      validate_input(query, key, value, log_decay, beta, state, output);
      const ScopedDeviceSetter setter(query.device(), query.device_index());
      DEVICE_DISPATCH(query.device(),
                      (compute<D, float>(query, key, value, log_decay, beta, state, output)));
    }

  }
}
