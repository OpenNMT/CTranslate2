#pragma once

#include "op.h"
#include "mean.h"

namespace ctranslate2 {
  namespace ops {

    class Sum : public Mean {
    public:
      Sum(const dim_t axis);

      void operator()(const StorageView& input, StorageView& output) const override;
    };

  }
}
