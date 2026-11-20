#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class FusedAttention : public Op {
    public:
      bool operator()(const StorageView& queries,
                      const StorageView& keys,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      float queries_scale,
                      bool interleaved_queries,
                      StorageView& output) const;

      bool supports_interleaved_queries(Device device, DataType dtype) const;

    private:
      template <Device D>
      bool compute(const StorageView& queries,
                   const StorageView& keys,
                   const StorageView& values,
                   const StorageView* values_lengths,
                   float queries_scale,
                   bool interleaved_queries,
                   StorageView& output) const;

      template <Device D>
      bool supports_interleaved_queries(DataType dtype) const;
    };

  }
}
