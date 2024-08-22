#pragma once

#include "activation.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Conv1D : public Op {
    public:
      Conv1D(dim_t stride = 1, dim_t padding = 0, dim_t dilation = 1, dim_t groups=1);

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView& bias,
                      StorageView& output,
                      const StorageView* qscale = nullptr) const;

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      StorageView& output,
                      const StorageView* qscale = nullptr) const;

    private:
      dim_t _stride;
      dim_t _padding;
      dim_t _dilation;
      dim_t _groups;

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView* bias,
                      StorageView& output,
                      const StorageView* qscale) const;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   const StorageView& weight,
                   const StorageView* bias,
                   StorageView& output,
                   const StorageView* qscale = nullptr) const;

      void compute_with_gemm(const StorageView& input, const StorageView& weight, StorageView& output,
                             const StorageView* qscale) const;

      void im2col_transposed(const StorageView& input, StorageView& output, dim_t kernel_size) const;
    };

  }
}
