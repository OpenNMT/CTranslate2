#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Applies one-dimensional transposed convolution.
    class ConvTranspose1D : public Op {
    public:
      ConvTranspose1D(dim_t stride = 1, dim_t padding = 0, dim_t output_padding = 0,
                      dim_t dilation = 1, dim_t groups = 1);

      void operator()(const StorageView& input, const StorageView& weight,
                      StorageView& output) const;
      void operator()(const StorageView& input, const StorageView& weight, const StorageView& bias,
                      StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView& weight, const StorageView* bias,
                   StorageView& output) const;

      bool compute_gemm(const StorageView& input, const StorageView& weight,
                        const StorageView* bias, StorageView& output) const;

      Shape validate_input(const StorageView& input, const StorageView& weight,
                           const StorageView* bias, const StorageView& output) const;

      void forward(const StorageView& input, const StorageView& weight, const StorageView* bias,
                   StorageView& output) const;

      dim_t _stride, _padding, _output_padding, _dilation, _groups;
    };

  }
}
