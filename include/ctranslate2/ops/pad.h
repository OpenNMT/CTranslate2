#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Scaffold only: 1-D padding along the last axis (reflect mode). Not implemented yet.
    class Pad1D : public Op {
    public:
      enum class Mode {
        Reflect
      };

      Pad1D(dim_t left, dim_t right, Mode mode = Mode::Reflect);

      void operator()(const StorageView& input, StorageView& output) const;

    private:
      dim_t _left;
      dim_t _right;
      Mode _mode;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

  }
}
