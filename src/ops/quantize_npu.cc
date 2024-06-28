#include "ctranslate2/ops/quantize.h"

namespace ctranslate2 {
    namespace ops {

        template <Device D, typename InT, typename OutT>
        void Quantize::quantize(const StorageView& input,
                                StorageView& output,
                                StorageView& scale) const {
            THROW_RUNTIME_ERROR("not implemented in CANN");
        }

        template void
        Quantize::quantize<Device::CANN, float, int8_t>(const StorageView&,
                                                        StorageView&,
                                                        StorageView&) const;
        template void
        Quantize::quantize<Device::CANN, float16_t, int8_t>(const StorageView&,
                                                            StorageView&,
                                                            StorageView&) const;
        template void
        Quantize::quantize<Device::CANN, bfloat16_t, int8_t>(const StorageView&,
                                                             StorageView&,
                                                             StorageView&) const;

    }
}
