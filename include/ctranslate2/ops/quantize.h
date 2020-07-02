#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Quantize : public Op {
    public:
      enum class ScaleType {
        GLOBAL,
        PER_LAYER,
        PER_ROW
      };

      static const float global_int16_scale;

      Quantize(ScaleType int16_scale_type = ScaleType::GLOBAL);
      void operator()(const StorageView& input,
                      StorageView& output,
                      StorageView& scale,
                      float shift = 0) const;

    private:
      template <Device D, typename T>
      void quantize(const StorageView& input,
                    StorageView& output,
                    StorageView& scale,
                    float shift) const;

      ScaleType _int16_scale_type;
    };

  }
}
