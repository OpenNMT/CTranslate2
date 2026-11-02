#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Applies recurrent gated-delta attention, updating state in place.
    class GatedDeltaNet : public Op {
    public:
      void operator()(const StorageView& query, const StorageView& key, const StorageView& value,
                      const StorageView& log_decay, const StorageView& beta, StorageView& state,
                      StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& query, const StorageView& key, const StorageView& value,
                   const StorageView& log_decay, const StorageView& beta, StorageView& state,
                   StorageView& output) const;

      void validate_input(const StorageView& query, const StorageView& key,
                          const StorageView& value, const StorageView& log_decay,
                          const StorageView& beta, const StorageView& state,
                          const StorageView& output) const;
    };

  }
}
