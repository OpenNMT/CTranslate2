#include "ctranslate2/ops/dequantize.h"

#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      mps::dequantize(DataTypeToEnum<OutT>::value,
                      input.data<InT>(),
                      scale.data<float>(),
                      output.data<OutT>(),
                      batch_size,
                      depth);
    }

#define DECLARE_DEQUANTIZE(T) \
    template void Dequantize::dequantize<Device::MPS, int8_t, T>( \
      const StorageView&, const StorageView&, StorageView&) const;
    DECLARE_DEQUANTIZE(float)
    DECLARE_DEQUANTIZE(float16_t)
    DECLARE_DEQUANTIZE(bfloat16_t)
#undef DECLARE_DEQUANTIZE

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& input,
                                            const StorageView& a_scale,
                                            const StorageView& b_scale,
                                            bool transpose_a,
                                            bool transpose_b,
                                            const StorageView* bias,
                                            StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      mps::dequantize_gemm_output(DataTypeToEnum<T>::value,
                                  input.data<int32_t>(),
                                  a_scale.data<float>(),
                                  a_scale.size(),
                                  b_scale.data<float>(),
                                  b_scale.size(),
                                  transpose_a,
                                  transpose_b,
                                  bias ? bias->data<T>() : nullptr,
                                  output.data<T>(),
                                  batch_size,
                                  depth,
                                  mps::activation_code(_activation_type));
    }

#define DECLARE_DEQUANTIZE_GEMM(T) \
    template void Dequantize::dequantize_gemm_output<Device::MPS, T>( \
      const StorageView&, const StorageView&, const StorageView&, bool, bool, \
      const StorageView*, StorageView&) const;
    DECLARE_DEQUANTIZE_GEMM(float)
    DECLARE_DEQUANTIZE_GEMM(float16_t)
    DECLARE_DEQUANTIZE_GEMM(bfloat16_t)
#undef DECLARE_DEQUANTIZE_GEMM

  }
}
