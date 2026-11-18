#include "ctranslate2/ops/moe_topk.h"
#include "mps/kernels.h"

namespace ctranslate2 {
  namespace ops {
    template <>
    void MoeTopK::compute<Device::MPS, float>(const StorageView& logits,
                                              StorageView& indices,
                                              StorageView& weights) const {
      indices.resize({logits.dim(0), _k});
      weights.resize(indices.shape());
      mps::moe_topk(logits.data<float>(), indices.data<int32_t>(),
                    weights.data<float>(), logits.dim(0), logits.dim(1), _k, _normalize_selected);
    }
  } // namespace ops
} // namespace ctranslate2
