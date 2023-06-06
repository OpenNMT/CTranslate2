#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopPMask : public Op {
    public:
      TopPMask(const float p);

      void operator()(const StorageView& input, StorageView& output) const;

    private:
      const float _p;

      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView& probs, StorageView& output) const;
    };

  }
}
