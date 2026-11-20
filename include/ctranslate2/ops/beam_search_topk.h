#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class BeamSearchTopK : public Op {
    public:
      bool operator()(const StorageView& logits,
                      const StorageView& beam_scores,
                      dim_t beam_size,
                      dim_t num_candidates,
                      StorageView& sampled_ids,
                      StorageView& sampled_scores) const;

    private:
      template <Device D>
      bool compute(const StorageView& logits,
                   const StorageView& beam_scores,
                   dim_t beam_size,
                   dim_t num_candidates,
                   StorageView& sampled_ids,
                   StorageView& sampled_scores) const;
    };

  }
}
