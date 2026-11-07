#include "ctranslate2/ops/beam_search_topk.h"

#include "mps/kernels.h"
#include "mps/ops_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    bool BeamSearchTopK::compute<Device::MPS>(const StorageView& logits,
                                              const StorageView& beam_scores,
                                              dim_t beam_size,
                                              dim_t num_candidates,
                                              StorageView& sampled_ids,
                                              StorageView& sampled_scores) const {
      static const bool enabled = mps::env_flag("CT2_MPS_USE_FUSED_BEAM_SEARCH");
      if (!enabled
          || logits.dtype() != DataType::FLOAT16
          || beam_scores.dtype() != logits.dtype()
          || beam_size <= 0
          || logits.dim(0) % beam_size != 0
          || num_candidates > 16)
        return false;

      const dim_t batch_size = logits.dim(0) / beam_size;
      const dim_t vocabulary_size = logits.dim(-1);
      StorageView beam_scores_device = beam_scores.to(Device::MPS);
      StorageView ids_device({batch_size, num_candidates}, DataType::INT32, Device::MPS);
      StorageView scores_device({batch_size, num_candidates}, logits.dtype(), Device::MPS);

      if (!mps::fused_beam_search_topk(logits.dtype(),
                                       logits.data<float16_t>(),
                                       beam_scores_device.data<float16_t>(),
                                       scores_device.data<float16_t>(),
                                       ids_device.data<int32_t>(),
                                       batch_size,
                                       beam_size,
                                       vocabulary_size,
                                       num_candidates))
        return false;

      sampled_ids.copy_from(ids_device);
      sampled_scores.copy_from(scores_device);
      return true;
    }

  }
}
