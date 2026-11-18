#include <stdexcept>

#include "ctranslate2/ops/awq/dequantize_awq.h"
#include "ctranslate2/ops/awq/gemm.h"
#include "ctranslate2/ops/awq/gemv.h"
#include "ctranslate2/ops/causal_depthwise_conv1d.h"
#include "ctranslate2/ops/conv_transpose1d.h"
#include "ctranslate2/ops/dequantize_fp8.h"
#include "ctranslate2/ops/gated_delta_net.h"
#include "ctranslate2/ops/nccl_ops.h"
#include "ctranslate2/ops/snake.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

#define THROW_MPS(NAME) throw std::runtime_error("MPS " #NAME " not implemented");

    template<>
    void DequantizeAwq::dequantize<Device::MPS, int32_t, float16_t>(
      const StorageView&, const StorageView&, const StorageView&, StorageView&) const {
      THROW_MPS(DequantizeAwq)
    }

    template<>
    void GemmAwq::compute<Device::MPS, float16_t, int32_t>(
      const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const {
      THROW_MPS(GemmAwq)
    }

    template<>
    void GemvAwq::compute_gemv<Device::MPS, float16_t, int32_t>(
      const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const {
      THROW_MPS(GemvAwq)
    }

    template<>
    void GemvAwq::compute_gemv2<Device::MPS, float16_t, int32_t>(
      const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const {
      THROW_MPS(GemvAwq2)
    }

    template <Device D, typename T>
    void GatherAll::compute(const StorageView&, StorageView&) const {
      THROW_MPS(GatherAll)
    }

#define DECLARE_GATHER_ALL(T) \
    template void GatherAll::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
    DECLARE_GATHER_ALL(float)
    DECLARE_GATHER_ALL(float16_t)
    DECLARE_GATHER_ALL(bfloat16_t)
    DECLARE_GATHER_ALL(int8_t)
    DECLARE_GATHER_ALL(int32_t)
    DECLARE_GATHER_ALL(int16_t)
#undef DECLARE_GATHER_ALL

    template <Device D, typename T>
    void ReduceAll::compute(const StorageView&, StorageView&) const {
      THROW_MPS(ReduceAll)
    }

#define DECLARE_REDUCE_ALL(T) \
    template void ReduceAll::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
    DECLARE_REDUCE_ALL(float)
    DECLARE_REDUCE_ALL(float16_t)
    DECLARE_REDUCE_ALL(bfloat16_t)
    DECLARE_REDUCE_ALL(int8_t)
    DECLARE_REDUCE_ALL(int32_t)
    DECLARE_REDUCE_ALL(int16_t)
#undef DECLARE_REDUCE_ALL

    // These operations were added after the initial MPS implementation. Keep
    // explicit specializations so shared dispatch links, but reject execution
    // until native Metal implementations are available.
    template<>
    void DequantizeFP8::compute<Device::MPS>(const StorageView&,
                                             const StorageView&,
                                             StorageView&) const {
      THROW_MPS(DequantizeFP8)
    }

    template<>
    void GatedDeltaNet::compute<Device::MPS, float>(const StorageView&,
                                                    const StorageView&,
                                                    const StorageView&,
                                                    const StorageView&,
                                                    const StorageView&,
                                                    StorageView&,
                                                    StorageView&) const {
      THROW_MPS(GatedDeltaNet)
    }

    template<>
    void ConvTranspose1D::compute<Device::MPS, float>(const StorageView&,
                                                      const StorageView&,
                                                      const StorageView*,
                                                      StorageView&) const {
      THROW_MPS(ConvTranspose1D)
    }

    template<>
    void CausalDepthwiseConv1D::compute<Device::MPS, float>(const StorageView&,
                                                            const StorageView&,
                                                            const StorageView*,
                                                            StorageView&,
                                                            StorageView&) const {
      THROW_MPS(CausalDepthwiseConv1D)
    }

    template<>
    void Snake::compute<Device::MPS, float>(const StorageView&,
                                            const StorageView&,
                                            StorageView&) const {
      THROW_MPS(Snake)
    }

#undef THROW_MPS

  }
}
