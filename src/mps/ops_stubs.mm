#ifdef __APPLE__
#include <stdexcept>
#include <vector>
#include "ctranslate2/types.h"
#include "ctranslate2/storage_view.h"
#include "mps/kernels.h"
#include "type_dispatch.h"

// --- Standard Ops ---
#include "ctranslate2/ops/dequantize.h"
#include "ctranslate2/ops/quantize.h"
#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/ops/layer_norm.h"
#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/tile.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/multinomial.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/rotary.h"
#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/alibi_add.h"
#include "ctranslate2/ops/median_filter.h"
#include "ctranslate2/ops/gumbel_max.h"
#include "ctranslate2/ops/mean.h"
#include "ctranslate2/ops/flash_attention.h"
#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/ops/nccl_ops.h"
#include "ctranslate2/ops/topp_mask.h"

// --- AWQ Ops (Corrected Paths) ---
#include "ctranslate2/ops/awq/gemm.h"
#include "ctranslate2/ops/awq/gemv.h"
#include "ctranslate2/ops/awq/dequantize_awq.h"

namespace ctranslate2 {
namespace ops {

#define THROW_MPS(NAME) throw std::runtime_error("MPS " #NAME " not implemented");

namespace {
int activation_code(const ActivationType* activation) {
  if (!activation)
    return -1;
  switch (*activation) {
  case ActivationType::ReLU:
    return static_cast<int>(mps::UnaryOp::RELU);
  case ActivationType::GELUTanh:
    return static_cast<int>(mps::UnaryOp::GELU_TANH);
  case ActivationType::Swish:
    return static_cast<int>(mps::UnaryOp::SWISH);
  case ActivationType::GELU:
    return static_cast<int>(mps::UnaryOp::GELU);
  case ActivationType::GELUSigmoid:
    return static_cast<int>(mps::UnaryOp::GELU_SIGMOID);
  case ActivationType::Tanh:
    return static_cast<int>(mps::UnaryOp::TANH);
  case ActivationType::Sigmoid:
    return static_cast<int>(mps::UnaryOp::SIGMOID);
  }
  return -1;
}
}

// --- Quantization ---
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
#define INST_DEQUANTIZE(T) \
template void Dequantize::dequantize<Device::MPS, int8_t, T>(const StorageView&, const StorageView&, StorageView&) const;
INST_DEQUANTIZE(float)
INST_DEQUANTIZE(float16_t)
INST_DEQUANTIZE(bfloat16_t)

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
                              activation_code(_activation_type));
}
#define INST_DEQ_GEMM(T) \
template void Dequantize::dequantize_gemm_output<Device::MPS, T>(const StorageView&, const StorageView&, const StorageView&, bool, bool, const StorageView*, StorageView&) const;
INST_DEQ_GEMM(float)
INST_DEQ_GEMM(float16_t)
INST_DEQ_GEMM(bfloat16_t)

template <Device D, typename T, typename Q>
void Quantize::quantize(const StorageView& input,
                        StorageView& output,
                        StorageView& scale) const {
  if (_shift_to_uint8)
    throw std::invalid_argument("Shift to uint8_t is not defined on MPS");
  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;
  mps::quantize(DataTypeToEnum<T>::value,
                input.data<T>(),
                output.data<int8_t>(),
                scale.data<float>(),
                batch_size,
                depth,
                _round_before_cast);
}
#define INST_QUANTIZE(T) \
template void Quantize::quantize<Device::MPS, T, int8_t>(const StorageView&, StorageView&, StorageView&) const;
INST_QUANTIZE(float)
INST_QUANTIZE(float16_t)
INST_QUANTIZE(bfloat16_t)

// --- Math & Activation ---
// SoftMax: handled in softmax_mps.cc
template <Device D, typename T>
void SoftMax::compute(const StorageView&, const StorageView*, StorageView&) const { THROW_MPS(SoftMax) }

// RMSNorm: handled in rms_norm_mps.mm
template <Device D, typename T>
void RMSNorm::compute(const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(RMSNorm) }

// LayerNorm: handled in layer_norm_mps.mm
template <Device D, typename T>
void LayerNorm::compute(const StorageView*, const StorageView*, const StorageView&, dim_t, dim_t, dim_t, dim_t, StorageView&) const { THROW_MPS(LayerNorm) }

// BiasAdd: handled in mps/ops_impl.mm
template <Device D, typename T>
void BiasAdd::compute(const StorageView&, const StorageView&, StorageView&, const StorageView*) const { THROW_MPS(BiasAdd) }

// Tile, Split, Concat, Gather, Slide: handled in mps/ops_impl.mm

// --- Advanced ---
template <Device D, typename T>
void Multinomial::compute(const StorageView& input, StorageView& output) const {
  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;
  mps::multinomial(DataTypeToEnum<T>::value,
                   input.data<T>(),
                   output.data<int32_t>(),
                   batch_size,
                   depth,
                   _sample_size);
}
#define INST_MULTINOMIAL(T) template void Multinomial::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_MULTINOMIAL(float)
INST_MULTINOMIAL(float16_t)
INST_MULTINOMIAL(bfloat16_t)

// TopK: handled in mps/ops_impl.mm
template <Device D, typename DataType, typename IndexType>
void TopK::compute(const StorageView&, StorageView&, StorageView&) const { THROW_MPS(TopK) }

// Rotary: handled in rotary_mps.mm
// Conv1D: handled in conv1d_mps.mm - stub removed to avoid ODR conflict

template <Device D, typename T>
void AlibiAdd::compute(const StorageView& input,
                       const StorageView& alibi,
                       dim_t alibi_offset,
                       StorageView& output) const {
  mps::alibi_add(DataTypeToEnum<T>::value,
                 input.data<T>(),
                 alibi.data<T>(),
                 output.data<T>(),
                 input.dim(0),
                 input.dim(1),
                 input.dim(2),
                 input.dim(3),
                 alibi.dim(-1),
                 alibi_offset);
}
#define INST_ALIBI(T) template void AlibiAdd::compute<Device::MPS, T>(const StorageView&, const StorageView&, dim_t, StorageView&) const;
INST_ALIBI(float)
INST_ALIBI(float16_t)
INST_ALIBI(bfloat16_t)

template <Device D, typename T>
void MedianFilter::compute(const StorageView& input,
                           dim_t axis_size,
                           StorageView& output) const {
  const dim_t rank = _width / 2;
  if (_width <= 1 || axis_size <= rank) {
    if (&input != &output)
      output.copy_from(input);
    return;
  }
  if ((_width & 1) == 0)
    throw std::invalid_argument("MedianFilter width must be odd");
  if (_width > 129)
    throw std::invalid_argument("MPS MedianFilter width exceeds 129");
  mps::median_filter(DataTypeToEnum<T>::value,
                     input.data<T>(),
                     output.data<T>(),
                     input.size() / axis_size,
                     axis_size,
                     _width);
}
#define INST_MEDIAN(T) template void MedianFilter::compute<Device::MPS, T>(const StorageView&, dim_t, StorageView&) const;
INST_MEDIAN(float)
INST_MEDIAN(float16_t)
INST_MEDIAN(bfloat16_t)

template <Device D, typename T>
void GumbelMax::add_gumbel_noise(const StorageView& input, StorageView& output) const {
  mps::gumbel_noise(DataTypeToEnum<T>::value,
                    input.data<T>(),
                    output.data<T>(),
                    input.size());
}
#define INST_GUMBEL(T) template void GumbelMax::add_gumbel_noise<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_GUMBEL(float)
INST_GUMBEL(float16_t)
INST_GUMBEL(bfloat16_t)

// Mean: handled in mean_mps.mm
template <Device D, typename T>
void Mean::compute(const StorageView&, dim_t, dim_t, dim_t, bool, StorageView&) const { THROW_MPS(Mean) }

// --- Flash Attention & AWQ ---
template<> void FlashAttention::compute<Device::MPS>(StorageView&, StorageView&, StorageView&, StorageView&, StorageView*, StorageView*, StorageView*, bool, StorageView*, StorageView*, bool, StorageView*, dim_t) const { THROW_MPS(FlashAttention) }

template<> void DequantizeAwq::dequantize<Device::MPS, int32_t, float16_t>(const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(DequantizeAwq) }

template<> void GemmAwq::compute<Device::MPS, float16_t, int32_t>(const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(GemmAwq) }
template<> void GemvAwq::compute_gemv<Device::MPS, float16_t, int32_t>(const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(GemvAwq) }
template<> void GemvAwq::compute_gemv2<Device::MPS, float16_t, int32_t>(const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(GemvAwq2) }

// --- NCCL / Distributed ---
template <Device D, typename T>
void GatherAll::compute(const StorageView&, StorageView&) const { THROW_MPS(GatherAll) }
#define INST_GATHERALL(T) template void GatherAll::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_GATHERALL(float)
INST_GATHERALL(float16_t)
INST_GATHERALL(bfloat16_t)
INST_GATHERALL(int8_t)
INST_GATHERALL(int32_t)
INST_GATHERALL(int16_t)

template <Device D, typename T>
void ReduceAll::compute(const StorageView&, StorageView&) const { THROW_MPS(ReduceAll) }
#define INST_REDUCEALL(T) template void ReduceAll::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_REDUCEALL(float)
INST_REDUCEALL(float16_t)
INST_REDUCEALL(bfloat16_t)
INST_REDUCEALL(int8_t)
INST_REDUCEALL(int32_t)
INST_REDUCEALL(int16_t)

template <Device D, typename T>
void TopPMask::compute(const StorageView& input,
                       const StorageView& probabilities,
                       StorageView& output) const {
  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;
  mps::top_p_mask(DataTypeToEnum<T>::value,
                  input.data<T>(),
                  probabilities.data<T>(),
                  output.data<T>(),
                  batch_size,
                  depth,
                  _p,
                  _mask_value);
}

#define INST_TOPP(T) template void TopPMask::compute<Device::MPS, T>(const StorageView&, const StorageView&, StorageView&) const;
INST_TOPP(float)
INST_TOPP(float16_t)
INST_TOPP(bfloat16_t)

template<> dim_t TopPMask::max_num_classes<Device::MPS>() { return mps::max_top_p_classes(); }

} // namespace ops
} // namespace ctranslate2
#endif
