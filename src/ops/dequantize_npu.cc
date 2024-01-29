#include "ctranslate2/ops/dequantize.h" 

namespace ctranslate2 {
  namespace ops { 

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

    template void
    Dequantize::dequantize<Device::CANN, int8_t, float>(const StorageView&,
                                                        const StorageView&,
                                                        StorageView&) const;
    template void
    Dequantize::dequantize<Device::CANN, int8_t, float16_t>(const StorageView&,
                                                            const StorageView&,
                                                            StorageView&) const;
    template void
    Dequantize::dequantize<Device::CANN, int8_t, bfloat16_t>(const StorageView&,
                                                             const StorageView&,
                                                             StorageView&) const;
 
    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& c,
                                            const StorageView& a_scale,
                                            const StorageView& b_scale,
                                            const bool transpose_a,
                                            const bool transpose_b,
                                            const StorageView* bias,
                                            StorageView& y) const {
        THROW_RUNTIME_ERROR("not implemented in CANN");
    }

    template void
    Dequantize::dequantize_gemm_output<Device::CANN, float>(const StorageView&,
                                                            const StorageView&,
                                                            const StorageView&,
                                                            const bool,
                                                            const bool,
                                                            const StorageView*,
                                                            StorageView&) const;
    template void
    Dequantize::dequantize_gemm_output<Device::CANN, float16_t>(const StorageView&,
                                                                const StorageView&,
                                                                const StorageView&,
                                                                const bool,
                                                                const bool,
                                                                const StorageView*,
                                                                StorageView&) const;
    template void
    Dequantize::dequantize_gemm_output<Device::CANN, bfloat16_t>(const StorageView&,
                                                                 const StorageView&,
                                                                 const StorageView&,
                                                                 const bool,
                                                                 const bool,
                                                                 const StorageView*,
                                                                 StorageView&) const;

  }
}
