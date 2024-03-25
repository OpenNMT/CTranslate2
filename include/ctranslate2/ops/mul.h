#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Mul : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        c.resize_as(a);
        if (b.is_scalar()) {
          const auto scalar = b.data<T>()[0];
          if constexpr (D == Device::CANN) {
              handleCannScalar<T>(scalar, a, c);
          }
          else {
              primitives<D>::mul(scalar, a.data<T>(), c.data<T>(), c.size());
          }
        } else {
          if constexpr (D == Device::CANN) {
              handleCann<T>(a, b, c);
          }
          else {
              primitives<D>::mul(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
          }
        }
      }

      template <typename T>
      void handleCann(const StorageView& a, const StorageView& b, StorageView& c) const;

      template <typename T>
      void handleCannScalar(T scalar, const StorageView& a, StorageView& c) const;
    };

  }
}
