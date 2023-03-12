#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class NormalizeAttentionWeights : public Op {
    public:
      void operator()(StorageView& input) const;
      void operator()(const StorageView& input, StorageView& output) const;
    };

  }
}
