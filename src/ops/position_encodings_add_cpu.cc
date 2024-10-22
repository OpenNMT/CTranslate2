#include "ctranslate2/ops/position_encodings_add.h"

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void PositionEncodingsAdd::compute(const dim_t step,
                                       const StorageView* offsets,
                                       const StorageView& input,
                                       const StorageView& encodings,
                                       StorageView& output) const {
      const dim_t batch_size = input.dim(0);
      const dim_t time = input.dim(1);
      const dim_t depth = input.dim(2);

      cpu::parallel_for(0, batch_size * time, 1, [&](const dim_t begin, const dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t b = i / time;
          const dim_t t = i % time;

          const dim_t offset = offsets ? offsets->at<int32_t>(b) : 0;
          const dim_t encoding_offset = t - offset + step;

          if (encoding_offset < 0)
            continue;

          primitives<Device::CPU>::add(encodings.index<float>({encoding_offset, 0}),
                                       input.index<float>({b, t, 0}),
                                       output.index<float>({b, t, 0}),
                                       depth);
        }
      });
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    PositionEncodingsAdd::compute<Device::CPU, T>(const dim_t,          \
                                                  const StorageView*,   \
                                                  const StorageView&,   \
                                                  const StorageView&,   \
                                                  StorageView&) const;

    DECLARE_IMPL(float)

  }
}
