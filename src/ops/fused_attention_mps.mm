#include "ctranslate2/ops/fused_attention.h"

#include "mps/kernels.h"
#include "mps/ops_utils.h"

namespace ctranslate2 {
  namespace ops {

    static bool fused_attention_enabled() {
      static const bool enabled = mps::env_flag("CT2_MPS_USE_FUSED_ATTENTION");
      return enabled;
    }

    template<>
    bool FusedAttention::supports_interleaved_queries<Device::MPS>(DataType dtype) const {
      return fused_attention_enabled() && dtype == DataType::FLOAT16;
    }

    template<>
    bool FusedAttention::compute<Device::MPS>(const StorageView& queries,
                                               const StorageView& keys,
                                               const StorageView& values,
                                               const StorageView* values_lengths,
                                               float queries_scale,
                                               bool interleaved_queries,
                                               StorageView& output) const {
      if (!fused_attention_enabled() || queries.dtype() != DataType::FLOAT16)
        return false;

      const bool standard_layout = !interleaved_queries
                                   && queries.rank() == 4
                                   && keys.rank() == 4
                                   && values.rank() == 4
                                   && queries.dim(0) == keys.dim(0)
                                   && queries.dim(0) == values.dim(0)
                                   && queries.dim(1) == keys.dim(1)
                                   && queries.dim(1) == values.dim(1)
                                   && keys.dim(2) == values.dim(2)
                                   && queries.dim(3) == keys.dim(3)
                                   && queries.dim(3) == values.dim(3);
      const bool interleaved_layout = interleaved_queries
                                      && queries.rank() == 4
                                      && keys.rank() == 4
                                      && values.rank() == 4
                                      && queries.dim(0) == keys.dim(0)
                                      && queries.dim(0) == values.dim(0)
                                      && queries.dim(2) == keys.dim(1)
                                      && queries.dim(2) == values.dim(1)
                                      && keys.dim(2) == values.dim(2)
                                      && queries.dim(3) == keys.dim(3)
                                      && queries.dim(3) == values.dim(3);
      if (!standard_layout && !interleaved_layout)
        return false;

      output.resize_as(queries);
      const dim_t num_heads = interleaved_layout ? queries.dim(2) : queries.dim(1);
      const dim_t query_length = interleaved_layout ? queries.dim(1) : queries.dim(2);
      const dim_t matrix_batch = queries.dim(0) * num_heads;
      return mps::fused_attention(DataType::FLOAT16,
                                  queries.data<float16_t>(),
                                  keys.data<float16_t>(),
                                  values.data<float16_t>(),
                                  values_lengths ? values_lengths->data<int32_t>() : nullptr,
                                  output.data<float16_t>(),
                                  matrix_batch,
                                  query_length,
                                  keys.dim(2),
                                  queries.dim(3),
                                  queries_scale,
                                  interleaved_layout ? num_heads : 0);
    }

  }
}
