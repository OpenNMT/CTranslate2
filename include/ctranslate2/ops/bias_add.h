#pragma once

#include "activation.h"
#include "add.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class BiasAdd : public Op {
    public:
      BiasAdd(const ActivationType* activation_type = nullptr, const dim_t axis = -1);

      void operator()(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output,
                      const StorageView* residual = nullptr) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& value,
                   const StorageView& bias,
                   StorageView& output,
                   const StorageView* residual) const;

      const ActivationType* _activation_type;
      const dim_t _axis;
    };

  }
}
