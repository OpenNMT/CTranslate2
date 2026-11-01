#include "ctranslate2/ops/beam_search_topk.h"

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    bool BeamSearchTopK::compute<Device::CPU>(const StorageView&,
                                               const StorageView&,
                                               dim_t,
                                               dim_t,
                                               StorageView&,
                                               StorageView&) const {
      return false;
    }

    template<>
    bool BeamSearchTopK::compute<Device::CUDA>(const StorageView&,
                                                const StorageView&,
                                                dim_t,
                                                dim_t,
                                                StorageView&,
                                                StorageView&) const {
      return false;
    }

    bool BeamSearchTopK::operator()(const StorageView& logits,
                                    const StorageView& beam_scores,
                                    dim_t beam_size,
                                    dim_t num_candidates,
                                    StorageView& sampled_ids,
                                    StorageView& sampled_scores) const {
      bool supported = false;
      DEVICE_DISPATCH(logits.device(),
                      supported = compute<D>(logits,
                                             beam_scores,
                                             beam_size,
                                             num_candidates,
                                             sampled_ids,
                                             sampled_scores));
      return supported;
    }

  }
}
