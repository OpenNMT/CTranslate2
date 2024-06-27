#pragma once

#include "../activation.h"
#include "../gemm.h"

namespace ctranslate2 {
  namespace ops {
    class GemvAwq : public Gemm {
    public:
      using Gemm::Gemm;
      void operator()(const StorageView& a,
                      const StorageView& b,
                      const StorageView& scale,
                      const StorageView& zero,
                      StorageView& c,
                      const StorageView* bias = nullptr) const;

    private:
      template <Device D, typename In, typename Out>
      void compute_gemv(const StorageView& a,
                   const StorageView& b,
                   const StorageView& scale,
                   const StorageView& zero,
                   StorageView& c) const;
      template <Device D, typename In, typename Out>
      void compute_gemv2(const StorageView& a,
                        const StorageView& b,
                        const StorageView& scale,
                        const StorageView& zero,
                        StorageView& c) const;
    };
  }
}