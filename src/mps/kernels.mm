#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mps/kernels.h"
#include "mps/utils.h"
#include "ctranslate2/random.h"

namespace ctranslate2 {
  namespace mps {
    namespace {

      static constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseArgs {
  ulong size;
  uint op;
  float scalar;
};

struct FillArgs {
  ulong size;
  float float_value;
  int int_value;
};

struct BroadcastArgs {
  ulong a_size;
  ulong b_size;
  ulong block;
  uint op;
  uint mode;
};

struct Transpose2DArgs {
  ulong rows;
  ulong cols;
};

struct TransposeNDArgs {
  ulong d0;
  ulong d1;
  ulong d2;
  ulong d3;
  ulong p0;
  ulong p1;
  ulong p2;
  ulong p3;
};

struct SoftmaxArgs {
  ulong batch_size;
  ulong depth;
  uint has_lengths;
  uint log_output;
};

struct MeanArgs {
  ulong outer_size;
  ulong axis_size;
  ulong inner_size;
  uint get_sum;
};

struct NormArgs {
  ulong outer_size;
  ulong axis_size;
  ulong inner_size;
  float epsilon;
  uint has_gamma;
  uint has_beta;
  uint use_residual;
};

struct RotaryArgs {
  ulong size;
  ulong max_time;
  ulong ndims;
  ulong depth;
  uint interleave;
};

struct Im2Col1DArgs {
  ulong total;
  ulong batch_size;
  ulong groups;
  ulong input_length;
  ulong kernel_size;
  ulong stride;
  ulong padding;
  ulong dilation;
  ulong output_length;
  ulong k;
  ulong in_batch_stride;
  ulong in_group_stride;
};

struct GemvArgs {
  ulong m;
  ulong n;
  ulong k;
  ulong lda;
  ulong ldb;
  ulong ldc;
  ulong stridea;
  ulong strideb;
  ulong stridec;
  ulong batch_size;
  uint transpose_a;
  uint transpose_b;
  float alpha;
  float beta;
  uint has_bias;
  uint has_residual;
  int activation;
};

struct GenericGemmArgs {
  ulong m;
  ulong n;
  ulong k;
  ulong lda;
  ulong ldb;
  ulong ldc;
  ulong stridea;
  ulong strideb;
  ulong stridec;
  ulong batch_size;
  uint transpose_a;
  uint transpose_b;
  float alpha;
  float beta;
};

struct TiledGemmArgs {
  uint m;
  uint n;
  uint k;
  uint lda;
  uint ldb;
  uint ldc;
  uint stridea;
  uint strideb;
  uint stridec;
  uint batch_size;
  uint transpose_a;
  uint transpose_b;
  float alpha;
  float beta;
};

struct SmallTopKArgs {
  uint batch_size;
  uint depth;
  uint k;
};

struct IndexedFillArgs {
  uint size;
  float float_value;
  int int_value;
};

struct RepetitionPenaltyArgs {
  uint total;
  uint length;
  uint vocabulary_size;
  float penalty;
};

struct LengthMaskArgs {
  uint batch_size;
  uint num_heads;
  uint num_queries;
  uint mask_future;
  uint multi_query;
};

struct GatherArgs {
  ulong copy_size;
  ulong batch_stride;
  ulong num_indices;
  ulong num_indices_per_batch;
};

struct Concat2Args {
  ulong outer_size;
  ulong a_block_size;
  ulong b_block_size;
};

struct TileArgs {
  ulong outer_size;
  ulong inner_size;
  ulong num_tiles;
};

struct BiasAddArgs {
  ulong bias_size;
  ulong value_size;
  ulong block;
  uint block_broadcast;
  uint has_residual;
  int activation;
};

struct QuantizeArgs {
  ulong batch_size;
  ulong depth;
  uint round_before_cast;
};

struct DequantizeArgs {
  ulong total;
  ulong depth;
};

struct DequantizeGemmArgs {
  ulong batch_size;
  ulong depth;
  ulong a_scale_size;
  ulong b_scale_size;
  uint transpose_a;
  uint transpose_b;
  uint has_bias;
  int activation;
};

struct MedianFilterArgs {
  ulong total;
  ulong depth;
  ulong width;
};

struct TopPMaskArgs {
  uint batch_size;
  uint depth;
  uint padded_depth;
  float probability;
  float mask_value;
};

struct RandomArgs {
  ulong total;
  ulong depth;
  ulong sample_size;
  ulong seed;
  ulong counter;
};

struct AlibiArgs {
  ulong total;
  ulong num_heads;
  ulong query_length;
  ulong key_length;
  ulong cached_key_length;
  ulong alibi_offset;
};

static inline ushort f32_to_bf16(float x) {
  uint bits = as_type<uint>(x);
  uint rounding_bias = 0x00007fffu + ((bits >> 16) & 1u);
  return ushort((bits + rounding_bias) >> 16);
}

static inline float bf16_to_f32(ushort x) {
  return as_type<float>(uint(x) << 16);
}

static inline float erf_approx(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  const float ax = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * ax);
  const float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                    - 0.284496736f) * t + 0.254829592f) * t;
  return sign * (1.0f - p * exp(-ax * ax));
}

static inline float apply_unary(float x, uint op) {
  switch (op) {
    case 0: return exp(x);
    case 1: return log(x);
    case 2: return cos(x);
    case 3: return sin(x);
    case 4: return tanh(x);
    case 5: return max(x, 0.0f);
    case 6: return 0.5f * x * (1.0f + erf_approx(x * 0.7071067811865475f));
    case 7: return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    case 8: return x / (1.0f + exp(-1.702f * x));
    case 9: return 1.0f / (1.0f + exp(-x));
    case 10: return x / (1.0f + exp(-x));
    default: return x;
  }
}

static inline float apply_binary(float a, float b, uint op) {
  switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return max(a, b);
    case 4: return min(a, b);
    default: return a;
  }
}

kernel void fill_f32(device float* y [[buffer(0)]],
                     constant FillArgs& args [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = args.float_value;
}

kernel void fill_f16(device half* y [[buffer(0)]],
                     constant FillArgs& args [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = half(args.float_value);
}

kernel void fill_bf16(device ushort* y [[buffer(0)]],
                      constant FillArgs& args [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = f32_to_bf16(args.float_value);
}

kernel void fill_i8(device char* y [[buffer(0)]],
                    constant FillArgs& args [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = char(args.int_value);
}

kernel void fill_i16(device short* y [[buffer(0)]],
                     constant FillArgs& args [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = short(args.int_value);
}

kernel void fill_i32(device int* y [[buffer(0)]],
                     constant FillArgs& args [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = args.int_value;
}

#define INDEXED_FILL_KERNEL(NAME, TYPE, VALUE)                                        \
kernel void NAME(device TYPE* y [[buffer(0)]],                                        \
                 device const int* indices [[buffer(1)]],                             \
                 constant IndexedFillArgs& args [[buffer(2)]],                        \
                 uint gid [[thread_position_in_grid]]) {                              \
  if (gid < args.size) y[indices[gid]] = VALUE;                                       \
}

INDEXED_FILL_KERNEL(indexed_fill_f32, float, args.float_value)
INDEXED_FILL_KERNEL(indexed_fill_f16, half, half(args.float_value))
INDEXED_FILL_KERNEL(indexed_fill_bf16, ushort, f32_to_bf16(args.float_value))
INDEXED_FILL_KERNEL(indexed_fill_i8, char, char(args.int_value))
INDEXED_FILL_KERNEL(indexed_fill_i16, short, short(args.int_value))
INDEXED_FILL_KERNEL(indexed_fill_i32, int, args.int_value)

#define REPETITION_PENALTY_KERNEL(NAME, TYPE, LOAD, STORE)                            \
kernel void NAME(device TYPE* scores [[buffer(0)]],                                   \
                 device const TYPE* previous_scores [[buffer(1)]],                   \
                 device const int* previous_ids [[buffer(2)]],                       \
                 constant RepetitionPenaltyArgs& args [[buffer(3)]],                 \
                 uint gid [[thread_position_in_grid]]) {                             \
  if (gid >= args.total) return;                                                      \
  const uint batch = gid / args.length;                                               \
  const int token = previous_ids[gid];                                                \
  if (token < 0 || uint(token) >= args.vocabulary_size) return;                       \
  const ulong write_index = ulong(batch) * ulong(args.vocabulary_size)               \
                            + ulong(token);                                           \
  const float score = LOAD(previous_scores[gid]);                                    \
  scores[write_index] = STORE(score < 0.0f ? score * args.penalty                    \
                                           : score / args.penalty);                  \
}

REPETITION_PENALTY_KERNEL(repetition_penalty_f32, float, float, float)
REPETITION_PENALTY_KERNEL(repetition_penalty_f16, half, float, half)
REPETITION_PENALTY_KERNEL(repetition_penalty_bf16, ushort, bf16_to_f32, f32_to_bf16)

kernel void prepare_length_mask(device const int* lengths [[buffer(0)]],
                                device int* mask [[buffer(1)]],
                                constant LengthMaskArgs& args [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  const uint per_batch = args.num_heads * args.num_queries;
  const uint total = args.batch_size * per_batch;
  if (gid >= total) return;
  const uint batch = gid / per_batch;
  const uint local = gid - batch * per_batch;
  const int length = lengths[batch];
  if (args.mask_future) {
    const uint query = args.multi_query ? local / args.num_heads
                                        : local % args.num_queries;
    mask[gid] = min(length, int(query + 1));
  } else {
    mask[gid] = length;
  }
}

kernel void unary_f32(device const float* x [[buffer(0)]],
                      device float* y [[buffer(1)]],
                      constant ElementwiseArgs& args [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = apply_unary(x[gid], args.op);
}

kernel void unary_f16(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      constant ElementwiseArgs& args [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = half(apply_unary(float(x[gid]), args.op));
}

kernel void unary_bf16(device const ushort* x [[buffer(0)]],
                       device ushort* y [[buffer(1)]],
                       constant ElementwiseArgs& args [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = f32_to_bf16(apply_unary(bf16_to_f32(x[gid]), args.op));
}

kernel void binary_f32(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant ElementwiseArgs& args [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  c[gid] = apply_binary(a[gid], b[gid], args.op);
}

kernel void binary_f16(device const half* a [[buffer(0)]],
                       device const half* b [[buffer(1)]],
                       device half* c [[buffer(2)]],
                       constant ElementwiseArgs& args [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  c[gid] = half(apply_binary(float(a[gid]), float(b[gid]), args.op));
}

kernel void binary_bf16(device const ushort* a [[buffer(0)]],
                        device const ushort* b [[buffer(1)]],
                        device ushort* c [[buffer(2)]],
                        constant ElementwiseArgs& args [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  c[gid] = f32_to_bf16(apply_binary(bf16_to_f32(a[gid]), bf16_to_f32(b[gid]), args.op));
}

kernel void scalar_f32(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       constant ElementwiseArgs& args [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = apply_binary(args.scalar, x[gid], args.op);
}

kernel void scalar_f16(device const half* x [[buffer(0)]],
                       device half* y [[buffer(1)]],
                       constant ElementwiseArgs& args [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = half(apply_binary(args.scalar, float(x[gid]), args.op));
}

kernel void scalar_bf16(device const ushort* x [[buffer(0)]],
                        device ushort* y [[buffer(1)]],
                        constant ElementwiseArgs& args [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if ((ulong)gid >= args.size) return;
  y[gid] = f32_to_bf16(apply_binary(args.scalar, bf16_to_f32(x[gid]), args.op));
}

template <typename T>
static inline float load_value(device const T* x, ulong i) { return float(x[i]); }

static inline float load_value_bf16(device const ushort* x, ulong i) { return bf16_to_f32(x[i]); }

template <typename T>
static inline void store_value(device T* y, ulong i, float v) { y[i] = T(v); }

static inline void store_value_bf16(device ushort* y, ulong i, float v) { y[i] = f32_to_bf16(v); }

#define GEMV_KERNEL(NAME, TYPE, LOAD, STORE)                                            \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                    \
                 device const TYPE* b [[buffer(1)]],                                    \
                 device TYPE* c [[buffer(2)]],                                          \
                 constant GemvArgs& args [[buffer(3)]],                                 \
                 threadgroup float* scratch [[threadgroup(0)]],                         \
                 uint row_id [[threadgroup_position_in_grid]],                          \
                 uint tid [[thread_index_in_threadgroup]],                              \
                 uint nt [[threads_per_threadgroup]],                                   \
                 uint lane [[thread_index_in_simdgroup]],                               \
                 uint simd_id [[simdgroup_index_in_threadgroup]],                       \
                 uint simd_groups [[simdgroups_per_threadgroup]]) {                     \
  ulong out_id = (ulong)row_id;                                                          \
  ulong rows = args.batch_size * args.m;                                                 \
  if (out_id >= rows) return;                                                            \
  ulong batch = out_id / args.m;                                                         \
  ulong row = out_id - batch * args.m;                                                   \
  ulong a_base = args.stridea == 0 ? 0 : batch * args.stridea;                           \
  ulong b_base = args.strideb == 0 ? 0 : batch * args.strideb;                           \
  ulong c_base = args.stridec == 0 ? 0 : batch * args.stridec;                           \
  float sum = 0.0f;                                                                      \
  for (ulong p = (ulong)tid; p < args.k; p += (ulong)nt) {                               \
    ulong ai = args.transpose_a ? (p * args.lda + row) : (row * args.lda + p);           \
    ulong bi = args.transpose_b ? p : (p * args.ldb);                                    \
    sum += LOAD(a, a_base + ai) * LOAD(b, b_base + bi);                                  \
  }                                                                                      \
  float partial = simd_sum(sum);                                                         \
  if (lane == 0) scratch[simd_id] = partial;                                             \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
  if (simd_id == 0) {                                                                    \
    float total = tid < simd_groups ? scratch[tid] : 0.0f;                               \
    total = simd_sum(total);                                                             \
    if (lane != 0) return;                                                               \
    ulong ci = c_base + row * args.ldc;                                                  \
    float old_value = args.beta == 0.0f ? 0.0f : LOAD(c, ci);                            \
    STORE(c, ci, args.alpha * total + args.beta * old_value);                            \
  }                                                                                      \
}

GEMV_KERNEL(gemv_f32, float, load_value<float>, store_value<float>)
GEMV_KERNEL(gemv_f16, half, load_value<half>, store_value<half>)
GEMV_KERNEL(gemv_bf16, ushort, load_value_bf16, store_value_bf16)

#define GEMV_ROW_KERNEL(NAME, TYPE, LOAD, STORE)                                        \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                    \
                 device const TYPE* b [[buffer(1)]],                                    \
                 device TYPE* c [[buffer(2)]],                                          \
                 constant GemvArgs& args [[buffer(3)]],                                 \
                 threadgroup float* scratch [[threadgroup(0)]],                         \
                 uint output_id [[threadgroup_position_in_grid]],                       \
                 uint tid [[thread_index_in_threadgroup]],                              \
                 uint nt [[threads_per_threadgroup]],                                   \
                 uint lane [[thread_index_in_simdgroup]],                               \
                 uint simd_id [[simdgroup_index_in_threadgroup]],                       \
                 uint simd_groups [[simdgroups_per_threadgroup]]) {                     \
  ulong out_id = (ulong)output_id;                                                       \
  ulong outputs = args.batch_size * args.n;                                              \
  if (out_id >= outputs) return;                                                         \
  ulong batch = out_id / args.n;                                                         \
  ulong column = out_id - batch * args.n;                                                \
  ulong a_base = args.stridea == 0 ? 0 : batch * args.stridea;                           \
  ulong b_base = args.strideb == 0 ? 0 : batch * args.strideb;                           \
  ulong c_base = args.stridec == 0 ? 0 : batch * args.stridec;                           \
  float sum = 0.0f;                                                                      \
  for (ulong p = (ulong)tid; p < args.k; p += (ulong)nt) {                               \
    ulong ai = args.transpose_a ? p * args.lda : p;                                      \
    ulong bi = args.transpose_b ? column * args.ldb + p : p * args.ldb + column;         \
    sum += LOAD(a, a_base + ai) * LOAD(b, b_base + bi);                                  \
  }                                                                                      \
  float partial = simd_sum(sum);                                                         \
  if (lane == 0) scratch[simd_id] = partial;                                             \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                       \
  if (simd_id == 0) {                                                                    \
    float total = tid < simd_groups ? scratch[tid] : 0.0f;                               \
    total = simd_sum(total);                                                             \
    if (lane != 0) return;                                                               \
    ulong ci = c_base + column;                                                          \
    float old_value = args.beta == 0.0f ? 0.0f : LOAD(c, ci);                            \
    STORE(c, ci, args.alpha * total + args.beta * old_value);                            \
  }                                                                                      \
}

GEMV_ROW_KERNEL(gemv_row_f32, float, load_value<float>, store_value<float>)
GEMV_ROW_KERNEL(gemv_row_f16, half, load_value<half>, store_value<half>)
GEMV_ROW_KERNEL(gemv_row_bf16, ushort, load_value_bf16, store_value_bf16)

// Output-major FP16 matvec load and dispatch geometry adapted from Apple MLX's
// gemv kernel. Copyright (c) 2023-2024 Apple Inc., Apache License 2.0.
// One SIMD group computes 4 output rows while loading each input-vector block
// only once. This is especially important for the hundreds of small decoder
// projections issued for every generated token.
kernel void gemv_row_output_major_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant GemvArgs& args [[buffer(3)]],
    device const half* bias [[buffer(4)]],
    device const half* residual [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]) {
  const uint batch = group.y;
  const uint first_output = (group.x * simd_groups + simd_id) * 4u;
  const ulong a_base = args.stridea == 0 ? 0 : ulong(batch) * args.stridea;
  const ulong b_base = args.strideb == 0 ? 0 : ulong(batch) * args.strideb;
  const ulong c_base = args.stridec == 0 ? 0 : ulong(batch) * args.stridec;

  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (args.transpose_a == 0 && (args.k & 3ul) == 0) {
    device const half4* av = reinterpret_cast<device const half4*>(a + a_base);
    const uint vectors = uint(args.k >> 2);
    for (uint p = lane; p < vectors; p += 32u) {
      const float4 input = float4(av[p]);
      for (uint output_delta = 0; output_delta < 4; ++output_delta) {
        const uint output = first_output + output_delta;
        if (output < args.n) {
          const ulong weight_row = b_base + ulong(output) * args.ldb;
          device const half4* weights =
              reinterpret_cast<device const half4*>(b + weight_row);
          sums[output_delta] += dot(input, float4(weights[p]));
        }
      }
    }
  } else {
    for (ulong p = lane; p < args.k; p += 32ul) {
      const ulong ai = args.transpose_a ? p * args.lda : p;
      const float input = float(a[a_base + ai]);
      for (uint output_delta = 0; output_delta < 4; ++output_delta) {
        const uint output = first_output + output_delta;
        if (output < args.n) {
          const ulong weight_row = b_base + ulong(output) * args.ldb;
          sums[output_delta] += input * float(b[weight_row + p]);
        }
      }
    }
  }

  for (uint output_delta = 0; output_delta < 4; ++output_delta) {
    const uint output = first_output + output_delta;
    if (output >= args.n)
      continue;
    const float sum = simd_sum(sums[output_delta]);
    if (lane == 0u) {
      const ulong ci = c_base + output;
      const float previous = args.beta == 0.0f ? 0.0f : float(c[ci]);
      // Match the unfused path's FP16 GEMV store before applying its epilogue.
      // This preserves the model's existing numerical behavior while avoiding
      // a second dispatch and an intermediate output read/write.
      float result = float(half(args.alpha * sum + args.beta * previous));
      if (args.has_bias)
        result += float(bias[output]);
      if (args.has_residual)
        result += float(residual[ci]);
      if (args.activation >= 0)
        result = apply_unary(result, uint(args.activation));
      c[ci] = half(result);
    }
  }
}

#define GENERIC_GEMM_KERNEL(NAME, TYPE, LOAD, STORE)                                  \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                  \
                 device const TYPE* b [[buffer(1)]],                                  \
                 device TYPE* c [[buffer(2)]],                                        \
                 constant GenericGemmArgs& args [[buffer(3)]],                        \
                 uint gid [[thread_position_in_grid]]) {                              \
  ulong index = (ulong)gid;                                                            \
  ulong matrix_size = args.m * args.n;                                                 \
  ulong total = args.batch_size * matrix_size;                                         \
  if (index >= total) return;                                                          \
  ulong batch = index / matrix_size;                                                   \
  ulong matrix_index = index - batch * matrix_size;                                    \
  ulong row = matrix_index / args.n;                                                   \
  ulong column = matrix_index - row * args.n;                                          \
  ulong a_base = args.stridea == 0 ? 0 : batch * args.stridea;                         \
  ulong b_base = args.strideb == 0 ? 0 : batch * args.strideb;                         \
  ulong c_base = args.stridec == 0 ? 0 : batch * args.stridec;                         \
  float sum = 0.0f;                                                                    \
  for (ulong p = 0; p < args.k; ++p) {                                                 \
    ulong ai = args.transpose_a ? p * args.lda + row : row * args.lda + p;             \
    ulong bi = args.transpose_b ? column * args.ldb + p : p * args.ldb + column;       \
    sum += LOAD(a, a_base + ai) * LOAD(b, b_base + bi);                                \
  }                                                                                    \
  ulong ci = c_base + row * args.ldc + column;                                         \
  float old_value = args.beta == 0.0f ? 0.0f : LOAD(c, ci);                            \
  STORE(c, ci, args.alpha * sum + args.beta * old_value);                              \
}

GENERIC_GEMM_KERNEL(generic_gemm_f32, float, load_value<float>, store_value<float>)
GENERIC_GEMM_KERNEL(generic_gemm_f16, half, load_value<half>, store_value<half>)
GENERIC_GEMM_KERNEL(generic_gemm_bf16, ushort, load_value_bf16, store_value_bf16)

#define TILED_GEMM_KERNEL(NAME, TYPE, LOAD, STORE)                                   \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                 \
                 device const TYPE* b [[buffer(1)]],                                 \
                 device TYPE* c [[buffer(2)]],                                       \
                 constant TiledGemmArgs& args [[buffer(3)]],                         \
                 threadgroup float* tiles [[threadgroup(0)]],                        \
                 uint3 group [[threadgroup_position_in_grid]],                       \
                 uint3 tid [[thread_position_in_threadgroup]]) {                     \
  const uint row = group.y * 16u + tid.y;                                             \
  const uint column = group.x * 16u + tid.x;                                          \
  const uint batch = group.z;                                                         \
  const uint a_base = args.stridea == 0 ? 0 : batch * args.stridea;                  \
  const uint b_base = args.strideb == 0 ? 0 : batch * args.strideb;                  \
  const uint c_base = args.stridec == 0 ? 0 : batch * args.stridec;                  \
  threadgroup float* tile_a = tiles;                                                  \
  threadgroup float* tile_b = tiles + 256;                                            \
  float sum = 0.0f;                                                                   \
  for (uint base_k = 0; base_k < args.k; base_k += 16u) {                             \
    const uint ak = base_k + tid.x;                                                   \
    const uint bk = base_k + tid.y;                                                   \
    float av = 0.0f;                                                                  \
    float bv = 0.0f;                                                                  \
    if (row < args.m && ak < args.k) {                                                \
      const uint ai = args.transpose_a ? ak * args.lda + row                         \
                                       : row * args.lda + ak;                         \
      av = LOAD(a, a_base + ai);                                                      \
    }                                                                                 \
    if (column < args.n && bk < args.k) {                                             \
      const uint bi = args.transpose_b ? column * args.ldb + bk                      \
                                       : bk * args.ldb + column;                      \
      bv = LOAD(b, b_base + bi);                                                      \
    }                                                                                 \
    tile_a[tid.y * 16u + tid.x] = av;                                                 \
    tile_b[tid.y * 16u + tid.x] = bv;                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
    for (uint p = 0; p < 16u; ++p)                                                   \
      sum += tile_a[tid.y * 16u + p] * tile_b[p * 16u + tid.x];                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
  }                                                                                   \
  if (row < args.m && column < args.n) {                                              \
    const uint ci = c_base + row * args.ldc + column;                                 \
    const float previous = args.beta == 0.0f ? 0.0f : LOAD(c, ci);                   \
    STORE(c, ci, args.alpha * sum + args.beta * previous);                            \
  }                                                                                   \
}

TILED_GEMM_KERNEL(tiled_gemm_f32, float, load_value<float>, store_value<float>)
TILED_GEMM_KERNEL(tiled_gemm_f16, half, load_value<half>, store_value<half>)
TILED_GEMM_KERNEL(tiled_gemm_bf16, ushort, load_value_bf16, store_value_bf16)

kernel void generic_gemm_i8_i32(device const char* a [[buffer(0)]],
                                device const char* b [[buffer(1)]],
                                device int* c [[buffer(2)]],
                                constant GenericGemmArgs& args [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  const ulong index = (ulong)gid;
  const ulong matrix_size = args.m * args.n;
  const ulong total = args.batch_size * matrix_size;
  if (index >= total) return;
  const ulong batch = index / matrix_size;
  const ulong matrix_index = index - batch * matrix_size;
  const ulong row = matrix_index / args.n;
  const ulong column = matrix_index - row * args.n;
  const ulong a_base = args.stridea == 0 ? 0 : batch * args.stridea;
  const ulong b_base = args.strideb == 0 ? 0 : batch * args.strideb;
  const ulong c_base = args.stridec == 0 ? 0 : batch * args.stridec;
  int sum = 0;
  for (ulong p = 0; p < args.k; ++p) {
    const ulong ai = args.transpose_a ? p * args.lda + row : row * args.lda + p;
    const ulong bi = args.transpose_b ? column * args.ldb + p : p * args.ldb + column;
    sum += int(a[a_base + ai]) * int(b[b_base + bi]);
  }
  const ulong ci = c_base + row * args.ldc + column;
  const float previous = args.beta == 0.0f ? 0.0f : float(c[ci]);
  c[ci] = int(rint(args.alpha * float(sum) + args.beta * previous));
}

kernel void tiled_gemm_i8_i32(device const char* a [[buffer(0)]],
                              device const char* b [[buffer(1)]],
                              device int* c [[buffer(2)]],
                              constant TiledGemmArgs& args [[buffer(3)]],
                              threadgroup int* tiles [[threadgroup(0)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 tid [[thread_position_in_threadgroup]]) {
  const uint row = group.y * 16u + tid.y;
  const uint column = group.x * 16u + tid.x;
  const uint batch = group.z;
  const uint a_base = args.stridea == 0 ? 0 : batch * args.stridea;
  const uint b_base = args.strideb == 0 ? 0 : batch * args.strideb;
  const uint c_base = args.stridec == 0 ? 0 : batch * args.stridec;
  threadgroup int* tile_a = tiles;
  threadgroup int* tile_b = tiles + 256;
  int sum = 0;
  for (uint base_k = 0; base_k < args.k; base_k += 16u) {
    const uint ak = base_k + tid.x;
    const uint bk = base_k + tid.y;
    int av = 0;
    int bv = 0;
    if (row < args.m && ak < args.k) {
      const uint ai = args.transpose_a ? ak * args.lda + row : row * args.lda + ak;
      av = int(a[a_base + ai]);
    }
    if (column < args.n && bk < args.k) {
      const uint bi = args.transpose_b ? column * args.ldb + bk : bk * args.ldb + column;
      bv = int(b[b_base + bi]);
    }
    tile_a[tid.y * 16u + tid.x] = av;
    tile_b[tid.y * 16u + tid.x] = bv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < 16u; ++p)
      sum += tile_a[tid.y * 16u + p] * tile_b[p * 16u + tid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < args.m && column < args.n) {
    const uint ci = c_base + row * args.ldc + column;
    const float previous = args.beta == 0.0f ? 0.0f : float(c[ci]);
    c[ci] = int(rint(args.alpha * float(sum) + args.beta * previous));
  }
}

static inline bool topk_better(float av, uint ai, float bv, uint bi) {
  return av > bv || (av == bv && ai < bi);
}

#define TOP_P_MASK_KERNEL(NAME, TYPE, LOAD, STORE)                                     \
kernel void NAME(device const TYPE* input [[buffer(0)]],                               \
                 device const TYPE* probabilities [[buffer(1)]],                       \
                 device TYPE* output [[buffer(2)]],                                    \
                 constant TopPMaskArgs& args [[buffer(3)]],                            \
                 threadgroup float* values [[threadgroup(0)]],                         \
                 threadgroup uint* indices [[threadgroup(1)]],                         \
                 uint row [[threadgroup_position_in_grid]],                            \
                 uint tid [[thread_index_in_threadgroup]],                             \
                 uint nt [[threads_per_threadgroup]]) {                                \
  if (row >= args.batch_size) return;                                                   \
  const ulong offset = (ulong)row * args.depth;                                         \
  for (uint i = tid; i < args.padded_depth; i += nt) {                                  \
    values[i] = i < args.depth ? LOAD(probabilities, offset + i) : -INFINITY;           \
    indices[i] = i;                                                                     \
  }                                                                                     \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
  for (uint width = 2; width <= args.padded_depth; width <<= 1) {                       \
    for (uint stride = width >> 1; stride > 0; stride >>= 1) {                          \
      for (uint i = tid; i < args.padded_depth; i += nt) {                              \
        const uint other = i ^ stride;                                                  \
        if (other > i) {                                                                \
          const bool descending = (i & width) == 0;                                     \
          const bool left_better = topk_better(values[i], indices[i],                   \
                                                values[other], indices[other]);          \
          const bool right_better = topk_better(values[other], indices[other],          \
                                                 values[i], indices[i]);                 \
          if ((descending && right_better) || (!descending && left_better)) {           \
            const float temp_value = values[i];                                         \
            const uint temp_index = indices[i];                                         \
            values[i] = values[other];                                                  \
            indices[i] = indices[other];                                                \
            values[other] = temp_value;                                                 \
            indices[other] = temp_index;                                                \
          }                                                                             \
        }                                                                               \
      }                                                                                 \
      threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
    }                                                                                   \
  }                                                                                     \
  if (tid == 0) {                                                                       \
    float cumulative = 0.0f;                                                           \
    for (uint i = 0; i < args.depth; ++i) {                                            \
      const uint original = indices[i];                                                 \
      const float value = cumulative < args.probability                                \
                          ? LOAD(input, offset + original)                              \
                          : args.mask_value;                                            \
      STORE(output, offset + original, value);                                          \
      cumulative += values[i];                                                         \
    }                                                                                   \
  }                                                                                     \
}

TOP_P_MASK_KERNEL(top_p_mask_f32, float, load_value<float>, store_value<float>)
TOP_P_MASK_KERNEL(top_p_mask_f16, half, load_value<half>, store_value<half>)
TOP_P_MASK_KERNEL(top_p_mask_bf16, ushort, load_value_bf16, store_value_bf16)

#define SMALL_TOPK_KERNEL(NAME, TYPE, LOAD, STORE)                                    \
kernel void NAME(device const TYPE* input [[buffer(0)]],                              \
                 device TYPE* values [[buffer(1)]],                                   \
                 device int* indices [[buffer(2)]],                                   \
                 constant SmallTopKArgs& args [[buffer(3)]],                          \
                 threadgroup float* shared_values [[threadgroup(0)]],                 \
                 threadgroup uint* shared_indices [[threadgroup(1)]],                 \
                 uint row [[threadgroup_position_in_grid]],                           \
                 uint tid [[thread_index_in_threadgroup]],                            \
                 uint nt [[threads_per_threadgroup]]) {                               \
  if (row >= args.batch_size) return;                                                  \
  float local_values[8];                                                              \
  uint local_indices[8];                                                              \
  for (uint rank = 0; rank < 8; ++rank) {                                             \
    local_values[rank] = -INFINITY;                                                   \
    local_indices[rank] = UINT_MAX;                                                   \
  }                                                                                   \
  const uint row_offset = row * args.depth;                                           \
  for (uint column = tid; column < args.depth; column += nt) {                        \
    const float candidate = LOAD(input, row_offset + column);                         \
    uint insert = args.k;                                                             \
    for (uint rank = 0; rank < args.k; ++rank) {                                      \
      if (topk_better(candidate, column, local_values[rank], local_indices[rank])) {   \
        insert = rank;                                                                \
        break;                                                                        \
      }                                                                               \
    }                                                                                 \
    if (insert < args.k) {                                                            \
      for (uint rank = args.k - 1; rank > insert; --rank) {                          \
        local_values[rank] = local_values[rank - 1];                                  \
        local_indices[rank] = local_indices[rank - 1];                                \
      }                                                                               \
      local_values[insert] = candidate;                                               \
      local_indices[insert] = column;                                                 \
    }                                                                                 \
  }                                                                                   \
  for (uint rank = 0; rank < args.k; ++rank) {                                        \
    shared_values[tid] = local_values[0];                                             \
    shared_indices[tid] = local_indices[0];                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
    for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                          \
      if (tid < stride && topk_better(shared_values[tid + stride],                    \
                                      shared_indices[tid + stride],                   \
                                      shared_values[tid],                             \
                                      shared_indices[tid])) {                         \
        shared_values[tid] = shared_values[tid + stride];                             \
        shared_indices[tid] = shared_indices[tid + stride];                           \
      }                                                                               \
      threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    }                                                                                 \
    if (tid == 0) {                                                                   \
      STORE(values, row * args.k + rank, shared_values[0]);                           \
      indices[row * args.k + rank] = int(shared_indices[0]);                         \
    }                                                                                 \
    const uint selected = shared_indices[0];                                          \
    if (local_indices[0] == selected) {                                               \
      for (uint next = 1; next < args.k; ++next) {                                   \
        local_values[next - 1] = local_values[next];                                 \
        local_indices[next - 1] = local_indices[next];                               \
      }                                                                               \
      local_values[args.k - 1] = -INFINITY;                                           \
      local_indices[args.k - 1] = UINT_MAX;                                           \
    }                                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                  \
  }                                                                                   \
}

SMALL_TOPK_KERNEL(small_topk_f32, float, load_value<float>, store_value<float>)
SMALL_TOPK_KERNEL(small_topk_f16, half, load_value<half>, store_value<half>)
SMALL_TOPK_KERNEL(small_topk_bf16, ushort, load_value_bf16, store_value_bf16)

#define GATHER_KERNEL(NAME, TYPE)                                                     \
kernel void NAME(device const TYPE* input [[buffer(0)]],                              \
                 device const int* indices [[buffer(1)]],                             \
                 device TYPE* output [[buffer(2)]],                                   \
                 constant GatherArgs& args [[buffer(3)]],                             \
                 uint gid [[thread_position_in_grid]]) {                              \
  ulong out = (ulong)gid;                                                              \
  ulong total = args.num_indices * args.copy_size;                                     \
  if (out >= total) return;                                                            \
  ulong index_id = out / args.copy_size;                                               \
  ulong within = out - index_id * args.copy_size;                                      \
  ulong batch = index_id / args.num_indices_per_batch;                                 \
  ulong source = batch * args.batch_stride + ulong(indices[index_id]) * args.copy_size;\
  output[out] = input[source + within];                                                 \
}

GATHER_KERNEL(gather_f32, float)
GATHER_KERNEL(gather_f16, half)
GATHER_KERNEL(gather_bf16, ushort)
GATHER_KERNEL(gather_i8, char)
GATHER_KERNEL(gather_i16, short)
GATHER_KERNEL(gather_i32, int)

#define CONCAT2_KERNEL(NAME, TYPE)                                                    \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                 \
                 device const TYPE* b [[buffer(1)]],                                 \
                 device TYPE* output [[buffer(2)]],                                  \
                 constant Concat2Args& args [[buffer(3)]],                           \
                 uint gid [[thread_position_in_grid]]) {                             \
  const ulong out = ulong(gid);                                                      \
  const ulong output_block = args.a_block_size + args.b_block_size;                  \
  const ulong total = args.outer_size * output_block;                                \
  if (out >= total) return;                                                          \
  const ulong outer = out / output_block;                                            \
  const ulong within = out - outer * output_block;                                   \
  output[out] = within < args.a_block_size                                           \
                  ? a[outer * args.a_block_size + within]                            \
                  : b[outer * args.b_block_size + within - args.a_block_size];       \
}

CONCAT2_KERNEL(concat2_f32, float)
CONCAT2_KERNEL(concat2_f16, half)
CONCAT2_KERNEL(concat2_bf16, ushort)
CONCAT2_KERNEL(concat2_i8, char)
CONCAT2_KERNEL(concat2_i16, short)
CONCAT2_KERNEL(concat2_i32, int)

#define TILE_KERNEL(NAME, TYPE)                                                       \
kernel void NAME(device const TYPE* input [[buffer(0)]],                              \
                 device TYPE* output [[buffer(1)]],                                   \
                 constant TileArgs& args [[buffer(2)]],                               \
                 uint gid [[thread_position_in_grid]]) {                              \
  ulong out = (ulong)gid;                                                              \
  ulong group = args.num_tiles * args.inner_size;                                      \
  ulong total = args.outer_size * group;                                               \
  if (out >= total) return;                                                            \
  ulong outer = out / group;                                                           \
  ulong within = out % args.inner_size;                                                \
  output[out] = input[outer * args.inner_size + within];                               \
}

TILE_KERNEL(tile_f32, float)
TILE_KERNEL(tile_f16, half)
TILE_KERNEL(tile_bf16, ushort)
TILE_KERNEL(tile_i8, char)
TILE_KERNEL(tile_i16, short)
TILE_KERNEL(tile_i32, int)

#define BIAS_ADD_KERNEL(NAME, TYPE, LOAD, STORE)                                      \
kernel void NAME(device const TYPE* bias [[buffer(0)]],                               \
                 device const TYPE* value [[buffer(1)]],                              \
                 device const TYPE* residual [[buffer(2)]],                           \
                 device TYPE* output [[buffer(3)]],                                   \
                 constant BiasAddArgs& args [[buffer(4)]],                            \
                 uint gid [[thread_position_in_grid]]) {                              \
  ulong i = (ulong)gid;                                                                \
  if (i >= args.value_size) return;                                                    \
  ulong bias_index = args.block_broadcast                                              \
    ? (i / args.block) % args.bias_size                                                 \
    : i % args.bias_size;                                                               \
  float result = LOAD(value, i) + LOAD(bias, bias_index);                              \
  if (args.has_residual) result += LOAD(residual, i);                                  \
  if (args.activation >= 0) result = apply_unary(result, uint(args.activation));       \
  STORE(output, i, result);                                                            \
}

BIAS_ADD_KERNEL(bias_add_f32, float, load_value<float>, store_value<float>)
BIAS_ADD_KERNEL(bias_add_f16, half, load_value<half>, store_value<half>)
BIAS_ADD_KERNEL(bias_add_bf16, ushort, load_value_bf16, store_value_bf16)

#define BROADCAST_KERNEL(NAME, TYPE, LOAD, STORE)                                      \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                   \
                 device const TYPE* b [[buffer(1)]],                                   \
                 device TYPE* c [[buffer(2)]],                                         \
                 constant BroadcastArgs& args [[buffer(3)]],                           \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong i = (ulong)gid;                                                                 \
  if (i >= args.b_size) return;                                                         \
  ulong ai = 0;                                                                         \
  if (args.mode == 0) {                                                                 \
    ai = i % args.a_size;                                                               \
  } else if (args.mode == 1) {                                                          \
    ulong depth = args.a_size == 0 ? 0 : args.b_size / args.a_size;                     \
    ai = depth == 0 ? 0 : i / depth;                                                    \
  } else {                                                                              \
    ai = (i / args.block) % args.a_size;                                                 \
  }                                                                                     \
  STORE(c, i, apply_binary(LOAD(a, ai), LOAD(b, i), args.op));                          \
}

BROADCAST_KERNEL(broadcast_f32, float, load_value<float>, store_value<float>)
BROADCAST_KERNEL(broadcast_f16, half, load_value<half>, store_value<half>)
BROADCAST_KERNEL(broadcast_bf16, ushort, load_value_bf16, store_value_bf16)

#define QUANTIZE_KERNEL(NAME, TYPE, LOAD)                                                \
kernel void NAME(device const TYPE* input [[buffer(0)]],                                \
                 device char* output [[buffer(1)]],                                     \
                 device float* scales [[buffer(2)]],                                    \
                 constant QuantizeArgs& args [[buffer(3)]],                             \
                 threadgroup float* scratch [[threadgroup(0)]],                         \
                 uint row [[threadgroup_position_in_grid]],                             \
                 uint tid [[thread_index_in_threadgroup]],                              \
                 uint nt [[threads_per_threadgroup]]) {                                 \
  if ((ulong)row >= args.batch_size) return;                                            \
  const ulong offset = (ulong)row * args.depth;                                         \
  float maximum = 0.0f;                                                                 \
  for (ulong i = (ulong)tid; i < args.depth; i += (ulong)nt)                            \
    maximum = max(maximum, fabs(LOAD(input, offset + i)));                              \
  scratch[tid] = maximum;                                                               \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                               \
    if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                    \
  }                                                                                     \
  if (tid == 0) {                                                                       \
    const float scale = scratch[0] == 0.0f ? 1.0f : 127.0f / scratch[0];                \
    scratch[0] = scale;                                                                 \
    scales[row] = scale;                                                                \
  }                                                                                     \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
  const float scale = scratch[0];                                                       \
  for (ulong i = (ulong)tid; i < args.depth; i += (ulong)nt) {                          \
    float value = LOAD(input, offset + i) * scale;                                      \
    if (args.round_before_cast) value = rint(value);                                    \
    value = clamp(value, -127.0f, 127.0f);                                              \
    output[offset + i] = char(value);                                                   \
  }                                                                                     \
}

QUANTIZE_KERNEL(quantize_f32, float, load_value<float>)
QUANTIZE_KERNEL(quantize_f16, half, load_value<half>)
QUANTIZE_KERNEL(quantize_bf16, ushort, load_value_bf16)

#define DEQUANTIZE_KERNEL(NAME, TYPE, STORE)                                            \
kernel void NAME(device const char* input [[buffer(0)]],                               \
                 device const float* scales [[buffer(1)]],                             \
                 device TYPE* output [[buffer(2)]],                                    \
                 constant DequantizeArgs& args [[buffer(3)]],                          \
                 uint gid [[thread_position_in_grid]]) {                               \
  const ulong i = (ulong)gid;                                                          \
  if (i >= args.total) return;                                                         \
  const ulong row = i / args.depth;                                                    \
  STORE(output, i, float(input[i]) / scales[row]);                                     \
}

DEQUANTIZE_KERNEL(dequantize_f32, float, store_value<float>)
DEQUANTIZE_KERNEL(dequantize_f16, half, store_value<half>)
DEQUANTIZE_KERNEL(dequantize_bf16, ushort, store_value_bf16)

#define DEQUANTIZE_GEMM_KERNEL(NAME, TYPE, LOAD, STORE)                                 \
kernel void NAME(device const int* input [[buffer(0)]],                                \
                 device const float* a_scales [[buffer(1)]],                           \
                 device const float* b_scales [[buffer(2)]],                           \
                 device const TYPE* bias [[buffer(3)]],                                \
                 device TYPE* output [[buffer(4)]],                                    \
                 constant DequantizeGemmArgs& args [[buffer(5)]],                      \
                 uint gid [[thread_position_in_grid]]) {                               \
  const ulong i = (ulong)gid;                                                          \
  const ulong total = args.batch_size * args.depth;                                    \
  if (i >= total) return;                                                              \
  const ulong row = i / args.depth;                                                    \
  const ulong column = i - row * args.depth;                                           \
  const ulong ai = args.a_scale_size == 1 ? 0 : (args.transpose_a ? column : row);     \
  const ulong bi = args.b_scale_size == 1 ? 0 : (args.transpose_b ? column : row);     \
  float value = float(input[i]) / (a_scales[ai] * b_scales[bi]);                       \
  if (args.has_bias) value += LOAD(bias, column);                                      \
  if (args.activation >= 0) value = apply_unary(value, uint(args.activation));         \
  STORE(output, i, value);                                                             \
}

DEQUANTIZE_GEMM_KERNEL(dequantize_gemm_f32, float, load_value<float>, store_value<float>)
DEQUANTIZE_GEMM_KERNEL(dequantize_gemm_f16, half, load_value<half>, store_value<half>)
DEQUANTIZE_GEMM_KERNEL(dequantize_gemm_bf16, ushort, load_value_bf16, store_value_bf16)

#define MEDIAN_FILTER_KERNEL(NAME, TYPE, LOAD, STORE)                                  \
kernel void NAME(device const TYPE* input [[buffer(0)]],                               \
                 device TYPE* output [[buffer(1)]],                                    \
                 constant MedianFilterArgs& args [[buffer(2)]],                        \
                 uint gid [[thread_position_in_grid]]) {                               \
  const ulong index = (ulong)gid;                                                      \
  if (index >= args.total) return;                                                     \
  const long rank = long(args.width / 2);                                              \
  const long column = long(index % args.depth);                                        \
  const ulong row_offset = (index / args.depth) * args.depth;                          \
  float window[129];                                                                   \
  for (long k = -rank; k <= rank; ++k) {                                               \
    long read = abs(column + k);                                                       \
    if (read >= long(args.depth)) read = 2 * long(args.depth) - read - 2;              \
    window[k + rank] = LOAD(input, row_offset + ulong(read));                          \
  }                                                                                    \
  for (ulong i = 1; i < args.width; ++i) {                                             \
    const float key = window[i];                                                       \
    long j = long(i) - 1;                                                             \
    while (j >= 0 && window[j] > key) {                                                \
      window[j + 1] = window[j];                                                       \
      --j;                                                                            \
    }                                                                                  \
    window[j + 1] = key;                                                              \
  }                                                                                    \
  STORE(output, index, window[rank]);                                                  \
}

MEDIAN_FILTER_KERNEL(median_filter_f32, float, load_value<float>, store_value<float>)
MEDIAN_FILTER_KERNEL(median_filter_f16, half, load_value<half>, store_value<half>)
MEDIAN_FILTER_KERNEL(median_filter_bf16, ushort, load_value_bf16, store_value_bf16)

#define ALIBI_ADD_KERNEL(NAME, TYPE, LOAD, STORE)                                      \
kernel void NAME(device const TYPE* input [[buffer(0)]],                               \
                 device const TYPE* alibi [[buffer(1)]],                               \
                 device TYPE* output [[buffer(2)]],                                    \
                 constant AlibiArgs& args [[buffer(3)]],                               \
                 uint gid [[thread_position_in_grid]]) {                               \
  const ulong i = (ulong)gid;                                                          \
  if (i >= args.total) return;                                                         \
  const ulong key = i % args.key_length;                                               \
  const ulong query_row = i / args.key_length;                                         \
  const ulong head = (query_row / args.query_length) % args.num_heads;                 \
  const ulong alibi_i = head * args.cached_key_length + args.alibi_offset + key;        \
  STORE(output, i, LOAD(input, i) + LOAD(alibi, alibi_i));                             \
}

ALIBI_ADD_KERNEL(alibi_add_f32, float, load_value<float>, store_value<float>)
ALIBI_ADD_KERNEL(alibi_add_f16, half, load_value<half>, store_value<half>)
ALIBI_ADD_KERNEL(alibi_add_bf16, ushort, load_value_bf16, store_value_bf16)

static inline uint random_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

static inline float random_uniform(ulong seed, ulong counter, ulong index) {
  const uint mixed = random_hash(uint(seed) ^ uint(seed >> 32) ^
                                 uint(counter + index) ^ uint((counter + index) >> 32));
  return (float(mixed >> 8) + 1.0f) * (1.0f / 16777217.0f);
}

#define MULTINOMIAL_KERNEL(NAME, TYPE, LOAD)                                           \
kernel void NAME(device const TYPE* probabilities [[buffer(0)]],                      \
                 device int* output [[buffer(1)]],                                     \
                 constant RandomArgs& args [[buffer(2)]],                              \
                 uint gid [[thread_position_in_grid]]) {                               \
  const ulong sample = (ulong)gid;                                                     \
  if (sample >= args.total) return;                                                    \
  const ulong row = sample / args.sample_size;                                         \
  const ulong offset = row * args.depth;                                               \
  float total = 0.0f;                                                                  \
  for (ulong i = 0; i < args.depth; ++i) total += max(LOAD(probabilities, offset + i), 0.0f); \
  const float target = random_uniform(args.seed, args.counter, sample) * total;         \
  float cumulative = 0.0f;                                                             \
  int selected = int(args.depth - 1);                                                  \
  for (ulong i = 0; i < args.depth; ++i) {                                             \
    cumulative += max(LOAD(probabilities, offset + i), 0.0f);                          \
    if (cumulative >= target) { selected = int(i); break; }                            \
  }                                                                                    \
  output[sample] = selected;                                                           \
}

MULTINOMIAL_KERNEL(multinomial_f32, float, load_value<float>)
MULTINOMIAL_KERNEL(multinomial_f16, half, load_value<half>)
MULTINOMIAL_KERNEL(multinomial_bf16, ushort, load_value_bf16)

#define GUMBEL_NOISE_KERNEL(NAME, TYPE, LOAD, STORE)                                   \
kernel void NAME(device const TYPE* input [[buffer(0)]],                               \
                 device TYPE* output [[buffer(1)]],                                    \
                 constant RandomArgs& args [[buffer(2)]],                              \
                 uint gid [[thread_position_in_grid]]) {                               \
  const ulong i = (ulong)gid;                                                          \
  if (i >= args.total) return;                                                         \
  const float noise = -log(random_uniform(args.seed, args.counter, i));                \
  STORE(output, i, LOAD(input, i) + noise);                                            \
}

GUMBEL_NOISE_KERNEL(gumbel_noise_f32, float, load_value<float>, store_value<float>)
GUMBEL_NOISE_KERNEL(gumbel_noise_f16, half, load_value<half>, store_value<half>)
GUMBEL_NOISE_KERNEL(gumbel_noise_bf16, ushort, load_value_bf16, store_value_bf16)

#define TRANSPOSE_2D_KERNEL(NAME, TYPE)                                                \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                   \
                 device TYPE* b [[buffer(1)]],                                         \
                 constant Transpose2DArgs& args [[buffer(2)]],                         \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong i = (ulong)gid;                                                                 \
  ulong size = args.rows * args.cols;                                                   \
  if (i >= size) return;                                                                \
  ulong r = i / args.cols;                                                              \
  ulong c = i - r * args.cols;                                                         \
  b[c * args.rows + r] = a[i];                                                         \
}

TRANSPOSE_2D_KERNEL(transpose_2d_f32, float)
TRANSPOSE_2D_KERNEL(transpose_2d_f16, half)
TRANSPOSE_2D_KERNEL(transpose_2d_bf16, ushort)

#define TRANSPOSE_3D_KERNEL(NAME, TYPE)                                                \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                   \
                 device TYPE* b [[buffer(1)]],                                         \
                 constant TransposeNDArgs& args [[buffer(2)]],                         \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong bd0 = args.p0 == 0 ? args.d0 : (args.p0 == 1 ? args.d1 : args.d2);             \
  ulong bd1 = args.p1 == 0 ? args.d0 : (args.p1 == 1 ? args.d1 : args.d2);             \
  ulong bd2 = args.p2 == 0 ? args.d0 : (args.p2 == 1 ? args.d1 : args.d2);             \
  ulong size = bd0 * bd1 * bd2;                                                        \
  ulong idx = (ulong)gid;                                                               \
  if (idx >= size) return;                                                              \
  ulong i0 = idx / (bd1 * bd2);                                                        \
  ulong rem = idx - i0 * bd1 * bd2;                                                    \
  ulong i1 = rem / bd2;                                                                \
  ulong i2 = rem - i1 * bd2;                                                           \
  ulong src[3] = {0, 0, 0};                                                            \
  src[args.p0] = i0;                                                                    \
  src[args.p1] = i1;                                                                    \
  src[args.p2] = i2;                                                                    \
  ulong aidx = src[0] * args.d1 * args.d2 + src[1] * args.d2 + src[2];                 \
  b[idx] = a[aidx];                                                                     \
}

TRANSPOSE_3D_KERNEL(transpose_3d_f32, float)
TRANSPOSE_3D_KERNEL(transpose_3d_f16, half)
TRANSPOSE_3D_KERNEL(transpose_3d_bf16, ushort)

#define TRANSPOSE_4D_KERNEL(NAME, TYPE)                                                \
kernel void NAME(device const TYPE* a [[buffer(0)]],                                   \
                 device TYPE* b [[buffer(1)]],                                         \
                 constant TransposeNDArgs& args [[buffer(2)]],                         \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong dims[4] = {args.d0, args.d1, args.d2, args.d3};                                \
  ulong perm[4] = {args.p0, args.p1, args.p2, args.p3};                                \
  ulong bd0 = dims[perm[0]];                                                           \
  ulong bd1 = dims[perm[1]];                                                           \
  ulong bd2 = dims[perm[2]];                                                           \
  ulong bd3 = dims[perm[3]];                                                           \
  ulong size = bd0 * bd1 * bd2 * bd3;                                                  \
  ulong idx = (ulong)gid;                                                               \
  if (idx >= size) return;                                                              \
  ulong i0 = idx / (bd1 * bd2 * bd3);                                                  \
  ulong rem0 = idx - i0 * bd1 * bd2 * bd3;                                             \
  ulong i1 = rem0 / (bd2 * bd3);                                                       \
  ulong rem1 = rem0 - i1 * bd2 * bd3;                                                  \
  ulong i2 = rem1 / bd3;                                                               \
  ulong i3 = rem1 - i2 * bd3;                                                          \
  ulong src[4] = {0, 0, 0, 0};                                                         \
  src[perm[0]] = i0;                                                                    \
  src[perm[1]] = i1;                                                                    \
  src[perm[2]] = i2;                                                                    \
  src[perm[3]] = i3;                                                                    \
  ulong aidx = src[0] * args.d1 * args.d2 * args.d3 + src[1] * args.d2 * args.d3 +     \
               src[2] * args.d3 + src[3];                                             \
  b[idx] = a[aidx];                                                                     \
}

TRANSPOSE_4D_KERNEL(transpose_4d_f32, float)
TRANSPOSE_4D_KERNEL(transpose_4d_f16, half)
TRANSPOSE_4D_KERNEL(transpose_4d_bf16, ushort)

#define SOFTMAX_KERNEL(NAME, TYPE, LOAD, STORE)                                        \
kernel void NAME(device const TYPE* x [[buffer(0)]],                                   \
                 device const int* lengths [[buffer(1)]],                              \
                 device TYPE* y [[buffer(2)]],                                         \
                 constant SoftmaxArgs& args [[buffer(3)]],                             \
                 threadgroup float* scratch [[threadgroup(0)]],                        \
                 uint row [[threadgroup_position_in_grid]],                            \
                 uint tid [[thread_index_in_threadgroup]],                             \
                 uint nt [[threads_per_threadgroup]]) {                                \
  if ((ulong)row >= args.batch_size) return;                                           \
  ulong depth = args.depth;                                                            \
  ulong offset = (ulong)row * depth;                                                   \
  ulong valid = depth;                                                                 \
  if (args.has_lengths) {                                                              \
    int l = lengths[row];                                                              \
    valid = l < 0 ? 0 : min((ulong)l, depth);                                          \
    for (ulong i = (ulong)tid; i < depth; i += (ulong)nt) {                            \
      if (i >= valid) STORE(y, offset + i, 0.0f);                                      \
    }                                                                                  \
  }                                                                                    \
  if (valid == 0) return;                                                              \
  float m = -INFINITY;                                                                 \
  for (ulong i = (ulong)tid; i < valid; i += (ulong)nt) {                              \
    m = max(m, LOAD(x, offset + i));                                                   \
  }                                                                                    \
  scratch[tid] = m;                                                                    \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  m = scratch[0];                                                                      \
  /* All threads must capture the maximum before scratch is reused for sums. */         \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  float s = 0.0f;                                                                      \
  for (ulong i = (ulong)tid; i < valid; i += (ulong)nt) {                              \
    s += exp(LOAD(x, offset + i) - m);                                                  \
  }                                                                                    \
  scratch[tid] = s;                                                                    \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) scratch[tid] += scratch[tid + stride];                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  float log_sum = log(scratch[0]);                                                     \
  for (ulong i = (ulong)tid; i < valid; i += (ulong)nt) {                              \
    float v = LOAD(x, offset + i) - m;                                                 \
    STORE(y, offset + i, args.log_output ? (v - log_sum) : exp(v - log_sum));          \
  }                                                                                    \
}

SOFTMAX_KERNEL(softmax_f32, float, load_value<float>, store_value<float>)
SOFTMAX_KERNEL(softmax_f16, half, load_value<half>, store_value<half>)
SOFTMAX_KERNEL(softmax_bf16, ushort, load_value_bf16, store_value_bf16)

#define MEAN_KERNEL(NAME, TYPE, LOAD, STORE)                                           \
kernel void NAME(device const TYPE* x [[buffer(0)]],                                   \
                 device TYPE* y [[buffer(1)]],                                         \
                 constant MeanArgs& args [[buffer(2)]],                                \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong out_size = args.outer_size * args.inner_size;                                  \
  ulong idx = (ulong)gid;                                                              \
  if (idx >= out_size) return;                                                         \
  ulong outer = idx / args.inner_size;                                                 \
  ulong inner = idx - outer * args.inner_size;                                         \
  float sum = 0.0f;                                                                    \
  ulong base = outer * args.axis_size * args.inner_size + inner;                       \
  for (ulong k = 0; k < args.axis_size; ++k) {                                         \
    sum += LOAD(x, base + k * args.inner_size);                                        \
  }                                                                                    \
  STORE(y, idx, args.get_sum ? sum : sum / float(args.axis_size));                     \
}

MEAN_KERNEL(mean_f32, float, load_value<float>, store_value<float>)
MEAN_KERNEL(mean_f16, half, load_value<half>, store_value<half>)
MEAN_KERNEL(mean_bf16, ushort, load_value_bf16, store_value_bf16)

#define MEAN_ROWS_KERNEL(NAME, TYPE, LOAD, STORE)                                      \
kernel void NAME(device const TYPE* x [[buffer(0)]],                                   \
                 device TYPE* y [[buffer(1)]],                                         \
                 constant MeanArgs& args [[buffer(2)]],                                \
                 threadgroup float* scratch [[threadgroup(0)]],                        \
                 uint row [[threadgroup_position_in_grid]],                            \
                 uint tid [[thread_index_in_threadgroup]],                             \
                 uint nt [[threads_per_threadgroup]]) {                                \
  if ((ulong)row >= args.outer_size) return;                                           \
  ulong base = (ulong)row * args.axis_size;                                            \
  float sum = 0.0f;                                                                    \
  for (ulong k = (ulong)tid; k < args.axis_size; k += (ulong)nt)                       \
    sum += LOAD(x, base + k);                                                          \
  scratch[tid] = sum;                                                                  \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) scratch[tid] += scratch[tid + stride];                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  if (tid == 0)                                                                        \
    STORE(y, row, args.get_sum ? scratch[0] : scratch[0] / float(args.axis_size));     \
}

MEAN_ROWS_KERNEL(mean_rows_f32, float, load_value<float>, store_value<float>)
MEAN_ROWS_KERNEL(mean_rows_f16, half, load_value<half>, store_value<half>)
MEAN_ROWS_KERNEL(mean_rows_bf16, ushort, load_value_bf16, store_value_bf16)

#define ARGMAX_KERNEL(NAME, TYPE, LOAD, STORE)                                         \
kernel void NAME(device const TYPE* x [[buffer(0)]],                                   \
                 device TYPE* values [[buffer(1)]],                                    \
                 device int* indices [[buffer(2)]],                                    \
                 constant SoftmaxArgs& args [[buffer(3)]],                             \
                 threadgroup float* scratch_values [[threadgroup(0)]],                 \
                 threadgroup uint* scratch_indices [[threadgroup(1)]],                 \
                 uint row [[threadgroup_position_in_grid]],                            \
                 uint tid [[thread_index_in_threadgroup]],                             \
                 uint nt [[threads_per_threadgroup]]) {                                \
  if ((ulong)row >= args.batch_size) return;                                           \
  ulong offset = (ulong)row * args.depth;                                              \
  float best_value = -INFINITY;                                                        \
  uint best_index = 0;                                                                 \
  for (ulong i = (ulong)tid; i < args.depth; i += (ulong)nt) {                         \
    float value = LOAD(x, offset + i);                                                 \
    if (value > best_value || (value == best_value && uint(i) < best_index)) {         \
      best_value = value;                                                              \
      best_index = uint(i);                                                            \
    }                                                                                  \
  }                                                                                    \
  scratch_values[tid] = best_value;                                                    \
  scratch_indices[tid] = best_index;                                                   \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) {                                                                \
      float other_value = scratch_values[tid + stride];                                \
      uint other_index = scratch_indices[tid + stride];                                \
      if (other_value > scratch_values[tid] ||                                         \
          (other_value == scratch_values[tid] && other_index < scratch_indices[tid])) {\
        scratch_values[tid] = other_value;                                             \
        scratch_indices[tid] = other_index;                                            \
      }                                                                                \
    }                                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  if (tid == 0) {                                                                      \
    STORE(values, row, scratch_values[0]);                                             \
    indices[row] = int(scratch_indices[0]);                                            \
  }                                                                                    \
}

ARGMAX_KERNEL(argmax_f32, float, load_value<float>, store_value<float>)
ARGMAX_KERNEL(argmax_f16, half, load_value<half>, store_value<half>)
ARGMAX_KERNEL(argmax_bf16, ushort, load_value_bf16, store_value_bf16)

#define LAYER_NORM_KERNEL(NAME, TYPE, LOAD, STORE)                                     \
kernel void NAME(device const TYPE* x [[buffer(0)]],                                   \
                 device const TYPE* gamma [[buffer(1)]],                               \
                 device const TYPE* beta [[buffer(2)]],                                \
                 device TYPE* y [[buffer(3)]],                                         \
                 constant NormArgs& args [[buffer(4)]],                                \
                 threadgroup float* scratch [[threadgroup(0)]],                        \
                 uint row [[threadgroup_position_in_grid]],                            \
                 uint tid [[thread_index_in_threadgroup]],                             \
                 uint nt [[threads_per_threadgroup]]) {                                \
  ulong total = args.outer_size * args.inner_size;                                     \
  if ((ulong)row >= total) return;                                                     \
  ulong outer = (ulong)row / args.inner_size;                                          \
  ulong inner = (ulong)row - outer * args.inner_size;                                  \
  ulong base = outer * args.axis_size * args.inner_size + inner;                       \
  float sum = 0.0f;                                                                    \
  float sumsq = 0.0f;                                                                  \
  for (ulong k = (ulong)tid; k < args.axis_size; k += (ulong)nt) {                     \
    float v = LOAD(x, base + k * args.inner_size);                                     \
    sum += v;                                                                          \
    sumsq += v * v;                                                                    \
  }                                                                                    \
  scratch[tid] = sum;                                                                  \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) scratch[tid] += scratch[tid + stride];                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  float total_sum = scratch[0];                                                        \
  /* Prevent thread 0 from replacing scratch[0] before peers read total_sum. */         \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  scratch[tid] = sumsq;                                                                \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) scratch[tid] += scratch[tid + stride];                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  float mean = total_sum / float(args.axis_size);                                      \
  float variance = max(scratch[0] / float(args.axis_size) - mean * mean, 0.0f);        \
  float rstd = rsqrt(variance + args.epsilon);                                         \
  for (ulong k = (ulong)tid; k < args.axis_size; k += (ulong)nt) {                     \
    ulong index = base + k * args.inner_size;                                          \
    float v = (LOAD(x, index) - mean) * rstd;                                          \
    if (args.has_gamma) v *= LOAD(gamma, k);                                           \
    if (args.has_beta) v += LOAD(beta, k);                                             \
    STORE(y, index, v);                                                                \
  }                                                                                    \
}

LAYER_NORM_KERNEL(layer_norm_f32, float, load_value<float>, store_value<float>)
LAYER_NORM_KERNEL(layer_norm_f16, half, load_value<half>, store_value<half>)
LAYER_NORM_KERNEL(layer_norm_bf16, ushort, load_value_bf16, store_value_bf16)

#define RMS_NORM_KERNEL(NAME, TYPE, LOAD, STORE)                                       \
kernel void NAME(device const TYPE* x [[buffer(0)]],                                   \
                 device const TYPE* gamma [[buffer(1)]],                               \
                 device TYPE* y [[buffer(2)]],                                         \
                 constant NormArgs& args [[buffer(3)]],                                \
                 threadgroup float* scratch [[threadgroup(0)]],                        \
                 uint row [[threadgroup_position_in_grid]],                            \
                 uint tid [[thread_index_in_threadgroup]],                             \
                 uint nt [[threads_per_threadgroup]]) {                                \
  if ((ulong)row >= args.outer_size) return;                                           \
  ulong offset = (ulong)row * args.axis_size;                                          \
  float sumsq = 0.0f;                                                                  \
  for (ulong k = (ulong)tid; k < args.axis_size; k += (ulong)nt) {                     \
    float v = LOAD(x, offset + k);                                                     \
    sumsq += v * v;                                                                    \
  }                                                                                    \
  scratch[tid] = sumsq;                                                                \
  threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
  for (uint stride = nt >> 1; stride > 0; stride >>= 1) {                              \
    if (tid < stride) scratch[tid] += scratch[tid + stride];                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
  }                                                                                    \
  float inv_rms = rsqrt(scratch[0] / float(args.axis_size) + args.epsilon);            \
  for (ulong k = (ulong)tid; k < args.axis_size; k += (ulong)nt) {                     \
    float g = LOAD(gamma, k);                                                          \
    if (args.use_residual) g += 1.0f;                                                  \
    STORE(y, offset + k, LOAD(x, offset + k) * inv_rms * g);                           \
  }                                                                                    \
}

RMS_NORM_KERNEL(rms_norm_f32, float, load_value<float>, store_value<float>)
RMS_NORM_KERNEL(rms_norm_f16, half, load_value<half>, store_value<half>)
RMS_NORM_KERNEL(rms_norm_bf16, ushort, load_value_bf16, store_value_bf16)

#define ROTARY_KERNEL(NAME, TYPE, LOAD, STORE)                                         \
kernel void NAME(device const TYPE* input [[buffer(0)]],                               \
                 device const TYPE* sinv [[buffer(1)]],                                \
                 device const TYPE* cosv [[buffer(2)]],                                \
                 device TYPE* output [[buffer(3)]],                                    \
                 constant RotaryArgs& args [[buffer(4)]],                              \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong idx = (ulong)gid;                                                              \
  if (idx >= args.size) return;                                                        \
  ulong i = idx % args.depth;                                                          \
  ulong t = (idx / args.depth) % args.max_time;                                        \
  if (i >= args.ndims) {                                                               \
    output[idx] = input[idx];                                                          \
    return;                                                                            \
  }                                                                                    \
  ulong pair = 0;                                                                      \
  float rotated = 0.0f;                                                                \
  if (args.interleave) {                                                               \
    bool even = (i & 1ul) == 0ul;                                                      \
    pair = even ? i + 1ul : i - 1ul;                                                   \
    rotated = (even ? -LOAD(input, idx + 1ul) : LOAD(input, idx - 1ul));               \
  } else {                                                                             \
    ulong middle = args.ndims / 2ul;                                                   \
    bool first = i < middle;                                                           \
    pair = first ? i + middle : i - middle;                                            \
    rotated = (first ? -LOAD(input, idx + middle) : LOAD(input, idx - middle));        \
  }                                                                                    \
  (void)pair;                                                                          \
  float s = LOAD(sinv, t * args.ndims + i);                                            \
  float c = LOAD(cosv, t * args.ndims + i);                                            \
  STORE(output, idx, LOAD(input, idx) * c + rotated * s);                              \
}

ROTARY_KERNEL(rotary_f32, float, load_value<float>, store_value<float>)
ROTARY_KERNEL(rotary_f16, half, load_value<half>, store_value<half>)
ROTARY_KERNEL(rotary_bf16, ushort, load_value_bf16, store_value_bf16)

#define IM2COL_KERNEL(NAME, TYPE)                                                      \
kernel void NAME(device const TYPE* input [[buffer(0)]],                               \
                 device TYPE* output [[buffer(1)]],                                    \
                 constant Im2Col1DArgs& args [[buffer(2)]],                            \
                 uint gid [[thread_position_in_grid]]) {                               \
  ulong idx = (ulong)gid;                                                              \
  if (idx >= args.total) return;                                                       \
  ulong c = idx % args.k;                                                              \
  ulong t = (idx / args.k) % args.output_length;                                       \
  ulong g = (idx / (args.k * args.output_length)) % args.groups;                       \
  ulong b = idx / (args.groups * args.output_length * args.k);                         \
  ulong c_offset = c / args.kernel_size;                                               \
  ulong k_offset = c - c_offset * args.kernel_size;                                    \
  long in_t = long(t) * long(args.stride) - long(args.padding) +                       \
              long(args.dilation) * long(k_offset);                                    \
  TYPE v = TYPE(0);                                                                    \
  if (in_t >= 0 && in_t < long(args.input_length)) {                                   \
    ulong input_idx = b * args.in_batch_stride + g * args.in_group_stride +            \
                      c_offset * args.input_length + ulong(in_t);                      \
    v = input[input_idx];                                                              \
  }                                                                                    \
  output[idx] = v;                                                                     \
}

IM2COL_KERNEL(im2col_conv1d_f32, float)
IM2COL_KERNEL(im2col_conv1d_f16, half)
IM2COL_KERNEL(im2col_conv1d_bf16, ushort)

)METAL";

      struct ElementwiseArgs {
        uint64_t size;
        uint32_t op;
        float scalar;
      };

      struct FillArgs {
        uint64_t size;
        float float_value;
        int32_t int_value;
      };

      struct BroadcastArgs {
        uint64_t a_size;
        uint64_t b_size;
        uint64_t block;
        uint32_t op;
        uint32_t mode;
      };

      struct Transpose2DArgs {
        uint64_t rows;
        uint64_t cols;
      };

      struct TransposeNDArgs {
        uint64_t d0;
        uint64_t d1;
        uint64_t d2;
        uint64_t d3;
        uint64_t p0;
        uint64_t p1;
        uint64_t p2;
        uint64_t p3;
      };

      struct SoftmaxArgs {
        uint64_t batch_size;
        uint64_t depth;
        uint32_t has_lengths;
        uint32_t log_output;
      };

      struct MeanArgs {
        uint64_t outer_size;
        uint64_t axis_size;
        uint64_t inner_size;
        uint32_t get_sum;
      };

      struct NormArgs {
        uint64_t outer_size;
        uint64_t axis_size;
        uint64_t inner_size;
        float epsilon;
        uint32_t has_gamma;
        uint32_t has_beta;
        uint32_t use_residual;
      };

      struct RotaryArgs {
        uint64_t size;
        uint64_t max_time;
        uint64_t ndims;
        uint64_t depth;
        uint32_t interleave;
      };

      struct Im2Col1DArgs {
        uint64_t total;
        uint64_t batch_size;
        uint64_t groups;
        uint64_t input_length;
        uint64_t kernel_size;
        uint64_t stride;
        uint64_t padding;
        uint64_t dilation;
        uint64_t output_length;
        uint64_t k;
        uint64_t in_batch_stride;
        uint64_t in_group_stride;
      };

      struct GemvArgs {
        uint64_t m;
        uint64_t n;
        uint64_t k;
        uint64_t lda;
        uint64_t ldb;
        uint64_t ldc;
        uint64_t stridea;
        uint64_t strideb;
        uint64_t stridec;
        uint64_t batch_size;
        uint32_t transpose_a;
        uint32_t transpose_b;
        float alpha;
        float beta;
        uint32_t has_bias;
        uint32_t has_residual;
        int32_t activation;
      };

      struct GenericGemmArgs {
        uint64_t m;
        uint64_t n;
        uint64_t k;
        uint64_t lda;
        uint64_t ldb;
        uint64_t ldc;
        uint64_t stridea;
        uint64_t strideb;
        uint64_t stridec;
        uint64_t batch_size;
        uint32_t transpose_a;
        uint32_t transpose_b;
        float alpha;
        float beta;
      };

      struct TiledGemmArgs {
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t lda;
        uint32_t ldb;
        uint32_t ldc;
        uint32_t stridea;
        uint32_t strideb;
        uint32_t stridec;
        uint32_t batch_size;
        uint32_t transpose_a;
        uint32_t transpose_b;
        float alpha;
        float beta;
      };

      struct SmallTopKArgs {
        uint32_t batch_size;
        uint32_t depth;
        uint32_t k;
      };

      struct IndexedFillArgs {
        uint32_t size;
        float float_value;
        int32_t int_value;
      };

      struct RepetitionPenaltyArgs {
        uint32_t total;
        uint32_t length;
        uint32_t vocabulary_size;
        float penalty;
      };

      struct LengthMaskArgs {
        uint32_t batch_size;
        uint32_t num_heads;
        uint32_t num_queries;
        uint32_t mask_future;
        uint32_t multi_query;
      };

      struct GatherArgs {
        uint64_t copy_size;
        uint64_t batch_stride;
        uint64_t num_indices;
        uint64_t num_indices_per_batch;
      };

      struct Concat2Args {
        uint64_t outer_size;
        uint64_t a_block_size;
        uint64_t b_block_size;
      };

      struct TileArgs {
        uint64_t outer_size;
        uint64_t inner_size;
        uint64_t num_tiles;
      };

      struct BiasAddArgs {
        uint64_t bias_size;
        uint64_t value_size;
        uint64_t block;
        uint32_t block_broadcast;
        uint32_t has_residual;
        int32_t activation;
      };

      struct QuantizeArgs {
        uint64_t batch_size;
        uint64_t depth;
        uint32_t round_before_cast;
      };

      struct DequantizeArgs {
        uint64_t total;
        uint64_t depth;
      };

      struct DequantizeGemmArgs {
        uint64_t batch_size;
        uint64_t depth;
        uint64_t a_scale_size;
        uint64_t b_scale_size;
        uint32_t transpose_a;
        uint32_t transpose_b;
        uint32_t has_bias;
        int32_t activation;
      };

      struct MedianFilterArgs {
        uint64_t total;
        uint64_t depth;
        uint64_t width;
      };

      struct TopPMaskArgs {
        uint32_t batch_size;
        uint32_t depth;
        uint32_t padded_depth;
        float probability;
        float mask_value;
      };

      struct RandomArgs {
        uint64_t total;
        uint64_t depth;
        uint64_t sample_size;
        uint64_t seed;
        uint64_t counter;
      };

      struct AlibiArgs {
        uint64_t total;
        uint64_t num_heads;
        uint64_t query_length;
        uint64_t key_length;
        uint64_t cached_key_length;
        uint64_t alibi_offset;
      };

      static size_t dtype_size(DataType dtype) {
        switch (dtype) {
        case DataType::FLOAT32:
          return sizeof(float);
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
          return 2;
        case DataType::INT8:
          return sizeof(int8_t);
        case DataType::INT16:
          return sizeof(int16_t);
        case DataType::INT32:
          return sizeof(int32_t);
        default:
          throw std::invalid_argument("unsupported MPS dtype");
        }
      }

      static const char* dtype_suffix(DataType dtype) {
        switch (dtype) {
        case DataType::FLOAT32:
          return "f32";
        case DataType::FLOAT16:
          return "f16";
        case DataType::BFLOAT16:
          return "bf16";
        default:
          throw std::invalid_argument("MPS Metal kernels only support floating point tensors");
        }
      }

      static const char* storage_dtype_suffix(DataType dtype) {
        switch (dtype) {
        case DataType::FLOAT32:
          return "f32";
        case DataType::FLOAT16:
          return "f16";
        case DataType::BFLOAT16:
          return "bf16";
        case DataType::INT8:
          return "i8";
        case DataType::INT16:
          return "i16";
        case DataType::INT32:
          return "i32";
        default:
          throw std::invalid_argument("unsupported MPS storage dtype");
        }
      }

      static uint32_t op_code(BinaryOp op) {
        switch (op) {
        case BinaryOp::ADD:
          return 0;
        case BinaryOp::SUB:
          return 1;
        case BinaryOp::MUL:
          return 2;
        case BinaryOp::MAX:
          return 3;
        case BinaryOp::MIN:
          return 4;
        }
        return 0;
      }

      static uint32_t op_code(UnaryOp op) {
        switch (op) {
        case UnaryOp::EXP:
          return 0;
        case UnaryOp::LOG:
          return 1;
        case UnaryOp::COS:
          return 2;
        case UnaryOp::SIN:
          return 3;
        case UnaryOp::TANH:
          return 4;
        case UnaryOp::RELU:
          return 5;
        case UnaryOp::GELU:
          return 6;
        case UnaryOp::GELU_TANH:
          return 7;
        case UnaryOp::GELU_SIGMOID:
          return 8;
        case UnaryOp::SIGMOID:
          return 9;
        case UnaryOp::SWISH:
          return 10;
        }
        return 0;
      }

      static id<MTLLibrary> library() {
        static id<MTLLibrary> lib = nil;
        static std::once_flag library_once;
        std::call_once(library_once, [&]() {
          @autoreleasepool {
            id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
            if (!device)
              throw std::runtime_error("MPS device not available");

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:kMetalSource];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            lib = [device newLibraryWithSource:source options:options error:&error];
#if !__has_feature(objc_arc)
            [options release];
#endif
            if (!lib) {
              std::string message = "failed to compile CTranslate2 MPS kernels";
              if (error && [error localizedDescription])
                message += ": " + std::string([[error localizedDescription] UTF8String]);
              throw std::runtime_error(message);
            }
          }
        });
        return lib;
      }

      static id<MTLComputePipelineState> pipeline(const std::string& name) {
        static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
        static std::mutex cache_mutex;
        static thread_local std::unordered_map<std::string, id<MTLComputePipelineState>> local_cache;

        auto local_it = local_cache.find(name);
        if (local_it != local_cache.end())
          return local_it->second;

        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(name);
        if (it != cache.end()) {
          local_cache.emplace(name, it->second);
          return it->second;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
        NSString* ns_name = [[NSString alloc] initWithUTF8String:name.c_str()];
        id<MTLFunction> function = [library() newFunctionWithName:ns_name];
#if !__has_feature(objc_arc)
        [ns_name release];
#endif
        if (!function)
          throw std::runtime_error("MPS kernel not found: " + name);

        NSError* error = nil;
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
#if !__has_feature(objc_arc)
        [function release];
#endif
        if (!state) {
          std::string message = "failed to build MPS pipeline " + name;
          if (error && [error localizedDescription])
            message += ": " + std::string([[error localizedDescription] UTF8String]);
          throw std::runtime_error(message);
        }

        cache.emplace(name, state);
        local_cache.emplace(name, state);
        return state;
      }

      static size_t dense_matrix_elements(dim_t rows, dim_t cols, dim_t ld) {
        if (rows <= 0 || cols <= 0)
          return 0;
        return static_cast<size_t>((rows - 1) * ld + cols);
      }

      static size_t dense_matrix_bytes(dim_t rows, dim_t cols, dim_t ld, size_t element_size) {
        return dense_matrix_elements(rows, cols, ld) * element_size;
      }

      static size_t strided_matrix_array_bytes(dim_t rows,
                                               dim_t cols,
                                               dim_t ld,
                                               dim_t stride,
                                               dim_t batch_size,
                                               size_t element_size) {
        if (batch_size <= 0)
          return 0;
        return (static_cast<size_t>(batch_size - 1) * static_cast<size_t>(stride)
                + dense_matrix_elements(rows, cols, ld)) * element_size;
      }

      static size_t strided_array_bytes(size_t required_elements,
                                        dim_t stride,
                                        dim_t batch_size,
                                        size_t element_size) {
        if (batch_size <= 0)
          return 0;
        const size_t batch_offset = stride > 0 && batch_size > 1
                                    ? static_cast<size_t>(batch_size - 1) * static_cast<size_t>(stride)
                                    : 0;
        return (batch_offset + required_elements) * element_size;
      }

      static id<MTLBuffer> mtl_buffer(const void* ptr, size_t bytes, NSUInteger& offset) {
        size_t raw_offset = 0;
        void* raw_buffer = get_buffer_for_use(ptr, bytes, &raw_offset);
        if (!raw_buffer)
          throw std::runtime_error("MPS pointer is not backed by a registered Metal buffer");
        offset = static_cast<NSUInteger>(raw_offset);
        return (__bridge id<MTLBuffer>)raw_buffer;
      }

      static void set_buffer(id<MTLComputeCommandEncoder> encoder,
                             const void* ptr,
                             size_t bytes,
                             NSUInteger index) {
        NSUInteger offset = 0;
        id<MTLBuffer> buffer = mtl_buffer(ptr, bytes, offset);
        [encoder setBuffer:buffer offset:offset atIndex:index];
      }

      struct PackedWeightKey {
        const void* address;
        void* device;
        DataType dtype;
        dim_t n;
        dim_t k;
        dim_t ldb;
        bool transpose_b;

        bool operator==(const PackedWeightKey& other) const {
          return address == other.address && device == other.device
                 && dtype == other.dtype && n == other.n && k == other.k
                 && ldb == other.ldb && transpose_b == other.transpose_b;
        }
      };

      struct PackedWeightKeyHash {
        size_t operator()(const PackedWeightKey& key) const {
          size_t result = std::hash<const void*>()(key.address);
          auto mix = [&result](size_t value) {
            result ^= value + 0x9e3779b97f4a7c15ULL + (result << 6) + (result >> 2);
          };
          mix(std::hash<void*>()(key.device));
          mix(static_cast<size_t>(key.dtype));
          mix(static_cast<size_t>(key.n));
          mix(static_cast<size_t>(key.k));
          mix(static_cast<size_t>(key.ldb));
          mix(static_cast<size_t>(key.transpose_b));
          return result;
        }
      };

      struct PackedWeightInfo {
        id<MTLBuffer> buffer;
        NSUInteger offset;
      };

      using PackedWeightMap =
        std::unordered_map<PackedWeightKey, PackedWeightInfo, PackedWeightKeyHash>;

      static PackedWeightMap& packed_weight_cache() {
        // Intentionally leak this process-lifetime cache. Model weights normally
        // outlive worker threads, and avoiding static destruction order hazards
        // is more important than reclaiming a tiny metadata map at process exit.
        static PackedWeightMap* cache = new PackedWeightMap;
        return *cache;
      }

      static std::mutex& packed_weight_mutex() {
        static std::mutex* mutex = new std::mutex;
        return *mutex;
      }

      static PackedWeightInfo output_major_weight(DataType dtype,
                                                   const void* address,
                                                   dim_t n,
                                                   dim_t k,
                                                   dim_t ldb,
                                                   size_t bytes) {
        const PackedWeightKey key{address, get_device(), dtype, n, k, ldb, true};
        std::lock_guard<std::mutex> lock(packed_weight_mutex());
        auto& cache = packed_weight_cache();
        const auto found = cache.find(key);
        if (found != cache.end())
          return found->second;

        NSUInteger offset = 0;
        id<MTLBuffer> buffer = mtl_buffer(address, bytes, offset);
#if !__has_feature(objc_arc)
        [buffer retain];
#endif
        const PackedWeightInfo info{buffer, offset};
        cache.emplace(key, info);
        return info;
      }

      static void run_1d(const std::string& name,
                         uint64_t size,
                         const std::vector<std::pair<const void*, size_t>>& buffers,
                         const void* args,
                         size_t args_size,
                         NSUInteger args_index) {
        if (size == 0)
          return;

        id<MTLComputePipelineState> state = pipeline(name);
        id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
        [encoder setComputePipelineState:state];

        for (NSUInteger i = 0; i < buffers.size(); ++i)
          set_buffer(encoder, buffers[i].first, buffers[i].second, i);
        if (args)
          [encoder setBytes:args length:args_size atIndex:args_index];

        const NSUInteger simd_width = std::max<NSUInteger>(1, state.threadExecutionWidth);
        NSUInteger threads = std::min<NSUInteger>(256, state.maxTotalThreadsPerThreadgroup);
        if (threads >= simd_width)
          threads = std::max<NSUInteger>(simd_width, (threads / simd_width) * simd_width);
        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(size), 1, 1);
        MTLSize group = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        // Dispatches in a reused compute encoder need an explicit visibility
        // boundary when a later primitive consumes a buffer written here.
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        record_compute_dispatch(name.c_str());
      }

      static void run_rows(const std::string& name,
                           uint64_t rows,
                           NSUInteger threads,
                           const std::vector<std::pair<const void*, size_t>>& buffers,
                           const void* args,
                           size_t args_size,
                           NSUInteger args_index) {
        if (rows == 0)
          return;

        id<MTLComputePipelineState> state = pipeline(name);
        id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
        [encoder setComputePipelineState:state];

        for (NSUInteger i = 0; i < buffers.size(); ++i)
          set_buffer(encoder, buffers[i].first, buffers[i].second, i);
        [encoder setBytes:args length:args_size atIndex:args_index];

        while (threads > state.maxTotalThreadsPerThreadgroup)
          threads >>= 1;
        threads = std::max<NSUInteger>(1, threads);
        [encoder setThreadgroupMemoryLength:((threads * sizeof(float) + 15) / 16) * 16
                                    atIndex:0];
        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1);
        MTLSize group = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        record_compute_dispatch(name.c_str());
      }

      static NSUInteger reduction_threads(dim_t size) {
        NSUInteger threads = 1;
        while (threads < static_cast<NSUInteger>(size) && threads < 256)
          threads <<= 1;
        return threads;
      }

      static MPSDataType mps_matrix_data_type(DataType dtype) {
        switch (dtype) {
        case DataType::FLOAT32:
          return MPSDataTypeFloat32;
        case DataType::FLOAT16:
          return MPSDataTypeFloat16;
        default:
          throw std::invalid_argument("MPSMatrix GEMM only supports float32 and float16");
        }
      }

      static std::string kernel_name(const char* base, DataType dtype) {
        return std::string(base) + "_" + dtype_suffix(dtype);
      }

      static std::string storage_kernel_name(const char* base, DataType dtype) {
        return std::string(base) + "_" + storage_dtype_suffix(dtype);
      }

      struct GemmKey {
        DataType dtype;
        bool transpose_a;
        bool transpose_b;
        dim_t m;
        dim_t n;
        dim_t k;
        dim_t batch_size;
        uint32_t alpha_bits;
        uint32_t beta_bits;

        bool operator==(const GemmKey& other) const {
          return dtype == other.dtype
                 && transpose_a == other.transpose_a
                 && transpose_b == other.transpose_b
                 && m == other.m
                 && n == other.n
                 && k == other.k
                 && batch_size == other.batch_size
                 && alpha_bits == other.alpha_bits
                 && beta_bits == other.beta_bits;
        }
      };

      struct GemmKeyHash {
        size_t operator()(const GemmKey& key) const {
          size_t h = static_cast<size_t>(key.dtype);
          auto mix = [&h](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
          };
          mix(static_cast<size_t>(key.transpose_a));
          mix(static_cast<size_t>(key.transpose_b));
          mix(static_cast<size_t>(key.m));
          mix(static_cast<size_t>(key.n));
          mix(static_cast<size_t>(key.k));
          mix(static_cast<size_t>(key.batch_size));
          mix(static_cast<size_t>(key.alpha_bits));
          mix(static_cast<size_t>(key.beta_bits));
          return h;
        }
      };

      static uint32_t float_bits(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
      }

      static MPSMatrixMultiplication* gemm_kernel(id<MTLDevice> device,
                                                  DataType dtype,
                                                  bool transpose_a,
                                                  bool transpose_b,
                                                  dim_t m,
                                                  dim_t n,
                                                  dim_t k,
                                                  dim_t batch_size,
                                                  float alpha,
                                                  float beta) {
        static std::unordered_map<GemmKey, MPSMatrixMultiplication*, GemmKeyHash> cache;
        static std::mutex cache_mutex;
        static thread_local std::unordered_map<GemmKey,
                                               MPSMatrixMultiplication*,
                                               GemmKeyHash> local_cache;

        const GemmKey key{dtype,
                          transpose_a,
                          transpose_b,
                          m,
                          n,
                          k,
                          batch_size,
                          float_bits(alpha),
                          float_bits(beta)};

        auto local_it = local_cache.find(key);
        if (local_it != local_cache.end())
          return local_it->second;

        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
          local_cache.emplace(key, it->second);
          return it->second;
        }

        MPSMatrixMultiplication* matmul =
          [[MPSMatrixMultiplication alloc] initWithDevice:device
                                            transposeLeft:transpose_a
                                           transposeRight:transpose_b
                                               resultRows:static_cast<NSUInteger>(m)
                                            resultColumns:static_cast<NSUInteger>(n)
                                          interiorColumns:static_cast<NSUInteger>(k)
                                                    alpha:alpha
                                                     beta:beta];
        [matmul setBatchSize:static_cast<NSUInteger>(batch_size)];
        cache.emplace(key, matmul);
        local_cache.emplace(key, matmul);
        return matmul;
      }

      static bool log_gemm_enabled() {
        static const bool enabled = []() {
          const char* value = std::getenv("CT2_MPS_LOG_GEMM");
          return value && value[0] != '\0' && std::string(value) != "0";
        }();
        return enabled;
      }

      static bool custom_gemv_enabled() {
        static const bool enabled = []() {
          const char* value = std::getenv("CT2_MPS_USE_GEMV");
          return !value || value[0] == '\0' || std::string(value) != "0";
        }();
        return enabled;
      }

      static bool gpu_topk_enabled() {
        static const bool enabled = []() {
          const char* value = std::getenv("CT2_MPS_USE_TOPK");
          return !value || value[0] == '\0' || std::string(value) != "0";
        }();
        return enabled;
      }

      static void log_gemm(DataType dtype,
                           bool transpose_a,
                           bool transpose_b,
                           dim_t m,
                           dim_t n,
                           dim_t k,
                           dim_t lda,
                           dim_t ldb,
                           dim_t ldc,
                           dim_t batch_size,
                           dim_t stridea,
                           dim_t strideb,
                           dim_t stridec,
                           const char* path) {
        if (!log_gemm_enabled())
          return;
        std::fprintf(stderr,
                     "CT2 MPS GEMM dtype=%s m=%lld n=%lld k=%lld trans_a=%d "
                     "trans_b=%d lda=%lld ldb=%lld ldc=%lld batch=%lld "
                     "stridea=%lld strideb=%lld stridec=%lld path=%s\n",
                     dtype_name(dtype).c_str(),
                     static_cast<long long>(m),
                     static_cast<long long>(n),
                     static_cast<long long>(k),
                     transpose_a,
                     transpose_b,
                     static_cast<long long>(lda),
                     static_cast<long long>(ldb),
                     static_cast<long long>(ldc),
                     static_cast<long long>(batch_size),
                     static_cast<long long>(stridea),
                     static_cast<long long>(strideb),
                     static_cast<long long>(stridec),
                     path);
      }

    }  // namespace

    void invalidate_packed_weight_cache(void* metal_buffer) {
      if (!metal_buffer)
        return;
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metal_buffer;
      std::lock_guard<std::mutex> lock(packed_weight_mutex());
      auto& cache = packed_weight_cache();
      for (auto it = cache.begin(); it != cache.end();) {
        if (it->second.buffer == buffer) {
#if !__has_feature(objc_arc)
          [it->second.buffer release];
#endif
          it = cache.erase(it);
        } else {
          ++it;
        }
      }
    }

    bool supports_gemm_type(DataType dtype) {
      return dtype == DataType::FLOAT32
             || dtype == DataType::FLOAT16
             || dtype == DataType::BFLOAT16;
    }

    static bool mps_matrix_layout_supported(dim_t rows,
                                            dim_t columns,
                                            dim_t leading_dimension,
                                            size_t element_size) {
      if (rows <= 0 || columns <= 0 || leading_dimension < columns)
        return false;
      const size_t row_bytes = static_cast<size_t>(leading_dimension) * element_size;
      return row_bytes % 16 == 0;
    }

    static bool fits_u32(dim_t value) {
      return value >= 0 && static_cast<uint64_t>(value) <= UINT32_MAX;
    }

    static void tiled_gemm(DataType dtype,
                           bool transpose_a,
                           bool transpose_b,
                           dim_t m,
                           dim_t n,
                           dim_t k,
                           float alpha,
                           const void* a,
                           dim_t lda,
                           dim_t stridea,
                           const void* b,
                           dim_t ldb,
                           dim_t strideb,
                           float beta,
                           void* c,
                           dim_t ldc,
                           dim_t stridec,
                           dim_t batch_size,
                           size_t a_bytes,
                           size_t b_bytes,
                           size_t c_bytes) {
      const TiledGemmArgs args{static_cast<uint32_t>(m),
                               static_cast<uint32_t>(n),
                               static_cast<uint32_t>(k),
                               static_cast<uint32_t>(lda),
                               static_cast<uint32_t>(ldb),
                               static_cast<uint32_t>(ldc),
                               static_cast<uint32_t>(std::max<dim_t>(stridea, 0)),
                               static_cast<uint32_t>(std::max<dim_t>(strideb, 0)),
                               static_cast<uint32_t>(std::max<dim_t>(stridec, 0)),
                               static_cast<uint32_t>(batch_size),
                               transpose_a ? 1u : 0u,
                               transpose_b ? 1u : 0u,
                               alpha,
                               beta};
      id<MTLComputePipelineState> state = pipeline(storage_kernel_name("tiled_gemm", dtype));
      id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
      [encoder setComputePipelineState:state];
      set_buffer(encoder, a, a_bytes, 0);
      set_buffer(encoder, b, b_bytes, 1);
      set_buffer(encoder, c, c_bytes, 2);
      [encoder setBytes:&args length:sizeof(args) atIndex:3];
      [encoder setThreadgroupMemoryLength:512 * sizeof(float) atIndex:0];
      [encoder dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(n) + 15) / 16,
                                                (static_cast<NSUInteger>(m) + 15) / 16,
                                                static_cast<NSUInteger>(batch_size))
               threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      record_compute_dispatch(storage_kernel_name("tiled_gemm", dtype).c_str());
    }

    static void generic_gemm(DataType dtype,
                             bool transpose_a,
                             bool transpose_b,
                             dim_t m,
                             dim_t n,
                             dim_t k,
                             float alpha,
                             const void* a,
                             dim_t lda,
                             dim_t stridea,
                             const void* b,
                             dim_t ldb,
                             dim_t strideb,
                             float beta,
                             void* c,
                             dim_t ldc,
                             dim_t stridec,
                             dim_t batch_size) {
      const size_t element_size = dtype_size(dtype);
      const dim_t a_rows = transpose_a ? k : m;
      const dim_t a_cols = transpose_a ? m : k;
      const dim_t b_rows = transpose_b ? n : k;
      const dim_t b_cols = transpose_b ? k : n;
      const size_t a_elements = dense_matrix_elements(a_rows, a_cols, lda);
      const size_t b_elements = dense_matrix_elements(b_rows, b_cols, ldb);
      const size_t c_elements = dense_matrix_elements(m, n, ldc);
      const size_t a_bytes = strided_array_bytes(a_elements, stridea, batch_size, element_size);
      const size_t b_bytes = strided_array_bytes(b_elements, strideb, batch_size, element_size);
      const size_t c_bytes = strided_array_bytes(c_elements, stridec, batch_size, element_size);
      if (fits_u32(m) && fits_u32(n) && fits_u32(k)
          && fits_u32(lda) && fits_u32(ldb) && fits_u32(ldc)
          && fits_u32(std::max<dim_t>(stridea, 0))
          && fits_u32(std::max<dim_t>(strideb, 0))
          && fits_u32(std::max<dim_t>(stridec, 0))
          && fits_u32(batch_size)) {
        tiled_gemm(dtype, transpose_a, transpose_b, m, n, k, alpha,
                   a, lda, stridea, b, ldb, strideb, beta,
                   c, ldc, stridec, batch_size, a_bytes, b_bytes, c_bytes);
        record_profile_event(ProfileEvent::Gemm);
        return;
      }
      const GenericGemmArgs args{static_cast<uint64_t>(m),
                                 static_cast<uint64_t>(n),
                                 static_cast<uint64_t>(k),
                                 static_cast<uint64_t>(lda),
                                 static_cast<uint64_t>(ldb),
                                 static_cast<uint64_t>(ldc),
                                 static_cast<uint64_t>(std::max<dim_t>(stridea, 0)),
                                 static_cast<uint64_t>(std::max<dim_t>(strideb, 0)),
                                 static_cast<uint64_t>(std::max<dim_t>(stridec, 0)),
                                 static_cast<uint64_t>(batch_size),
                                 transpose_a ? 1u : 0u,
                                 transpose_b ? 1u : 0u,
                                 alpha,
                                 beta};
      run_1d(storage_kernel_name("generic_gemm", dtype),
             static_cast<uint64_t>(batch_size * m * n),
             {{a, a_bytes}, {b, b_bytes}, {c, c_bytes}},
             &args,
             sizeof(args),
             3);
      record_profile_event(ProfileEvent::Gemm);
    }

    static void int8_gemm_impl(bool transpose_a,
                               bool transpose_b,
                               dim_t m,
                               dim_t n,
                               dim_t k,
                               float alpha,
                               const int8_t* a,
                               dim_t lda,
                               dim_t stridea,
                               const int8_t* b,
                               dim_t ldb,
                               dim_t strideb,
                               float beta,
                               int32_t* c,
                               dim_t ldc,
                               dim_t stridec,
                               dim_t batch_size) {
      if (batch_size <= 0 || m <= 0 || n <= 0)
        return;
      if (stridea < 0 || strideb < 0 || stridec < 0)
        throw std::invalid_argument("MPS INT8 GEMM received a negative batch stride");

      const dim_t a_rows = transpose_a ? k : m;
      const dim_t a_cols = transpose_a ? m : k;
      const dim_t b_rows = transpose_b ? n : k;
      const dim_t b_cols = transpose_b ? k : n;
      const size_t a_elements = dense_matrix_elements(a_rows, a_cols, lda);
      const size_t b_elements = dense_matrix_elements(b_rows, b_cols, ldb);
      const size_t c_elements = dense_matrix_elements(m, n, ldc);
      const size_t a_bytes = strided_array_bytes(a_elements, stridea, batch_size, sizeof(int8_t));
      const size_t b_bytes = strided_array_bytes(b_elements, strideb, batch_size, sizeof(int8_t));
      const size_t c_bytes = strided_array_bytes(c_elements, stridec, batch_size, sizeof(int32_t));

      if (fits_u32(m) && fits_u32(n) && fits_u32(k)
          && fits_u32(lda) && fits_u32(ldb) && fits_u32(ldc)
          && fits_u32(stridea) && fits_u32(strideb) && fits_u32(stridec)
          && fits_u32(batch_size)) {
        const TiledGemmArgs args{static_cast<uint32_t>(m),
                                 static_cast<uint32_t>(n),
                                 static_cast<uint32_t>(k),
                                 static_cast<uint32_t>(lda),
                                 static_cast<uint32_t>(ldb),
                                 static_cast<uint32_t>(ldc),
                                 static_cast<uint32_t>(stridea),
                                 static_cast<uint32_t>(strideb),
                                 static_cast<uint32_t>(stridec),
                                 static_cast<uint32_t>(batch_size),
                                 transpose_a ? 1u : 0u,
                                 transpose_b ? 1u : 0u,
                                 alpha,
                                 beta};
        id<MTLComputePipelineState> state = pipeline("tiled_gemm_i8_i32");
        id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
        [encoder setComputePipelineState:state];
        set_buffer(encoder, a, a_bytes, 0);
        set_buffer(encoder, b, b_bytes, 1);
        set_buffer(encoder, c, c_bytes, 2);
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder setThreadgroupMemoryLength:512 * sizeof(int32_t) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(n) + 15) / 16,
                                                  (static_cast<NSUInteger>(m) + 15) / 16,
                                                  static_cast<NSUInteger>(batch_size))
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        record_compute_dispatch("tiled_gemm_i8_i32");
      } else {
        const GenericGemmArgs args{static_cast<uint64_t>(m),
                                   static_cast<uint64_t>(n),
                                   static_cast<uint64_t>(k),
                                   static_cast<uint64_t>(lda),
                                   static_cast<uint64_t>(ldb),
                                   static_cast<uint64_t>(ldc),
                                   static_cast<uint64_t>(stridea),
                                   static_cast<uint64_t>(strideb),
                                   static_cast<uint64_t>(stridec),
                                   static_cast<uint64_t>(batch_size),
                                   transpose_a ? 1u : 0u,
                                   transpose_b ? 1u : 0u,
                                   alpha,
                                   beta};
        run_1d("generic_gemm_i8_i32",
               static_cast<uint64_t>(batch_size * m * n),
               {{a, a_bytes}, {b, b_bytes}, {c, c_bytes}},
               &args,
               sizeof(args),
               3);
      }
      record_profile_event(ProfileEvent::Gemm);
    }

    void gemm_int8(bool transpose_a,
                   bool transpose_b,
                   dim_t m,
                   dim_t n,
                   dim_t k,
                   float alpha,
                   const int8_t* a,
                   dim_t lda,
                   const int8_t* b,
                   dim_t ldb,
                   float beta,
                   int32_t* c,
                   dim_t ldc) {
      int8_gemm_impl(transpose_a, transpose_b, m, n, k, alpha,
                     a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1);
    }

    void gemm_int8_batch_strided(bool transpose_a,
                                 bool transpose_b,
                                 dim_t m,
                                 dim_t n,
                                 dim_t k,
                                 float alpha,
                                 const int8_t* a,
                                 dim_t lda,
                                 dim_t stridea,
                                 const int8_t* b,
                                 dim_t ldb,
                                 dim_t strideb,
                                 float beta,
                                 int32_t* c,
                                 dim_t ldc,
                                 dim_t stridec,
                                 dim_t batch_size) {
      int8_gemm_impl(transpose_a, transpose_b, m, n, k, alpha,
                     a, lda, stridea, b, ldb, strideb, beta,
                     c, ldc, stridec, batch_size);
    }

    void gemv(DataType dtype,
              bool transpose_a,
              bool transpose_b,
              dim_t m,
              dim_t k,
              float alpha,
              const void* a,
              dim_t lda,
              dim_t stridea,
              const void* b,
              dim_t ldb,
              dim_t strideb,
              float beta,
              void* c,
              dim_t ldc,
              dim_t stridec,
              dim_t batch_size) {
      if (batch_size <= 0 || m == 0)
        return;
      if (dtype != DataType::FLOAT32 && dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16)
        throw std::invalid_argument("unsupported MPS GEMV dtype");

      const size_t element_size = dtype_size(dtype);
      const dim_t a_rows = transpose_a ? k : m;
      const dim_t a_cols = transpose_a ? m : k;
      const size_t a_elements = dense_matrix_elements(a_rows, a_cols, lda);
      const size_t b_elements = transpose_b ? static_cast<size_t>(k) : dense_matrix_elements(k, 1, ldb);
      const size_t c_elements = dense_matrix_elements(m, 1, ldc);

      const GemvArgs args{static_cast<uint64_t>(m),
                          1,
                          static_cast<uint64_t>(k),
                          static_cast<uint64_t>(lda),
                          static_cast<uint64_t>(ldb),
                          static_cast<uint64_t>(ldc),
                          static_cast<uint64_t>(std::max<dim_t>(stridea, 0)),
                          static_cast<uint64_t>(std::max<dim_t>(strideb, 0)),
                          static_cast<uint64_t>(std::max<dim_t>(stridec, 0)),
                          static_cast<uint64_t>(batch_size),
                          transpose_a ? 1u : 0u,
                          transpose_b ? 1u : 0u,
                          alpha,
                          beta,
                          0,
                          0,
                          -1};

      run_rows(storage_kernel_name("gemv", dtype),
               static_cast<uint64_t>(batch_size * m),
               reduction_threads(k),
               {{a, strided_array_bytes(a_elements, stridea, batch_size, element_size)},
                {b, strided_array_bytes(b_elements, strideb, batch_size, element_size)},
                {c, strided_array_bytes(c_elements, stridec, batch_size, element_size)}},
               &args,
               sizeof(args),
               3);
      record_profile_event(ProfileEvent::Gemv);
    }

    void gemv_row(DataType dtype,
                  bool transpose_a,
                  bool transpose_b,
                  dim_t n,
                  dim_t k,
                  float alpha,
                  const void* a,
                  dim_t lda,
                  dim_t stridea,
                  const void* b,
                  dim_t ldb,
                  dim_t strideb,
                  float beta,
                  void* c,
                  dim_t ldc,
                  dim_t stridec,
                  dim_t batch_size,
                  const void* bias,
                  const void* residual,
                  int activation) {
      if (batch_size <= 0 || n == 0)
        return;
      if (dtype != DataType::FLOAT32 && dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16)
        throw std::invalid_argument("unsupported MPS row GEMV dtype");
      if ((bias || residual || activation >= 0)
          && (dtype != DataType::FLOAT16 || !transpose_b))
        throw std::invalid_argument("fused MPS row GEMV requires FP16 output-major weights");

      const size_t element_size = dtype_size(dtype);
      const size_t a_elements = transpose_a
                                ? dense_matrix_elements(k, 1, lda)
                                : static_cast<size_t>(k);
      const dim_t b_rows = transpose_b ? n : k;
      const dim_t b_cols = transpose_b ? k : n;
      const size_t b_elements = dense_matrix_elements(b_rows, b_cols, ldb);
      const size_t c_elements = dense_matrix_elements(1, n, ldc);

      const GemvArgs args{1,
                          static_cast<uint64_t>(n),
                          static_cast<uint64_t>(k),
                          static_cast<uint64_t>(lda),
                          static_cast<uint64_t>(ldb),
                          static_cast<uint64_t>(ldc),
                          static_cast<uint64_t>(std::max<dim_t>(stridea, 0)),
                          static_cast<uint64_t>(std::max<dim_t>(strideb, 0)),
                          static_cast<uint64_t>(std::max<dim_t>(stridec, 0)),
                          static_cast<uint64_t>(batch_size),
                          transpose_a ? 1u : 0u,
                          transpose_b ? 1u : 0u,
                          alpha,
                          beta,
                          bias ? 1u : 0u,
                          residual ? 1u : 0u,
                          activation};

      // Dense linear weights are stored by CTranslate2 as [N, K] and called
      // with transpose_b=true, which is already the packed output-major layout
      // required by the decode kernel.  Keep dynamic attention matrices on the
      // generic path because their non-transposed layout changes every step.
      if (dtype == DataType::FLOAT16 && transpose_b) {
        id<MTLComputePipelineState> state = pipeline("gemv_row_output_major_f16");
        id<MTLComputeCommandEncoder> encoder =
          (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
        [encoder setComputePipelineState:state];
        set_buffer(encoder,
                   a,
                   strided_array_bytes(a_elements, stridea, batch_size, element_size),
                   0);
        const size_t b_bytes =
          strided_array_bytes(b_elements, strideb, batch_size, element_size);
        if (batch_size == 1 && strideb == 0) {
          const PackedWeightInfo packed = output_major_weight(dtype, b, n, k, ldb, b_bytes);
          record_metal_buffer_use((__bridge void*)packed.buffer);
          [encoder setBuffer:packed.buffer offset:packed.offset atIndex:1];
        } else {
          set_buffer(encoder, b, b_bytes, 1);
        }
        set_buffer(encoder,
                   c,
                   strided_array_bytes(c_elements, stridec, batch_size, element_size),
                   2);
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        set_buffer(encoder,
                   bias ? bias : c,
                   bias ? static_cast<size_t>(n) * element_size : element_size,
                   4);
        set_buffer(encoder,
                   residual ? residual : c,
                   residual
                     ? strided_array_bytes(c_elements, stridec, batch_size, element_size)
                     : element_size,
                   5);
        const NSUInteger simd_groups = n >= 4096 ? 8 : 4;
        const NSUInteger outputs_per_group = simd_groups * 4;
        [encoder dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(n)
                                                   + outputs_per_group - 1)
                                                  / outputs_per_group,
                                                  static_cast<NSUInteger>(batch_size),
                                                  1)
                 threadsPerThreadgroup:MTLSizeMake(simd_groups * 32, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        record_compute_dispatch("gemv_row_output_major_f16");
        record_profile_event(ProfileEvent::Gemv);
        return;
      }

      run_rows(storage_kernel_name("gemv_row", dtype),
               static_cast<uint64_t>(batch_size * n),
               reduction_threads(k),
               {{a, strided_array_bytes(a_elements, stridea, batch_size, element_size)},
                {b, strided_array_bytes(b_elements, strideb, batch_size, element_size)},
                {c, strided_array_bytes(c_elements, stridec, batch_size, element_size)}},
               &args,
               sizeof(args),
               3);
      record_profile_event(ProfileEvent::Gemv);
    }

    void gemm(DataType dtype,
              bool transpose_a,
              bool transpose_b,
              dim_t m,
              dim_t n,
              dim_t k,
              float alpha,
              const void* a,
              dim_t lda,
              const void* b,
              dim_t ldb,
              float beta,
              void* c,
              dim_t ldc) {
      if (!supports_gemm_type(dtype))
        throw std::invalid_argument("unsupported MPS GEMM dtype");
      if (m == 0 || n == 0)
        return;
      if (m == 1 && custom_gemv_enabled()) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 1, 0, 0, 0, "row_gemv");
        gemv_row(dtype,
                 transpose_a,
                 transpose_b,
                 n,
                 k,
                 alpha,
                 a,
                 lda,
                 0,
                 b,
                 ldb,
                 0,
                 beta,
                 c,
                 ldc,
                 0,
                 1);
        return;
      }
      if (n == 1 && custom_gemv_enabled()) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 1, 0, 0, 0, "column_gemv");
        gemv(dtype,
             transpose_a,
             transpose_b,
             m,
             k,
             alpha,
             a,
             lda,
             0,
             b,
             ldb,
             0,
             beta,
             c,
             ldc,
             0,
             1);
        return;
      }

      // MPSMatrix does not expose a portable BF16 matrix type on the minimum
      // deployment target. The custom kernel stores BF16 as ushort and
      // accumulates in FP32, so it works consistently on all supported Macs.
      if (dtype == DataType::BFLOAT16) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 1, 0, 0, 0, "bf16_tiled");
        generic_gemm(dtype, transpose_a, transpose_b, m, n, k, alpha,
                     a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1);
        return;
      }

      const size_t element_size = dtype_size(dtype);
      const dim_t a_rows = transpose_a ? k : m;
      const dim_t a_cols = transpose_a ? m : k;
      const dim_t b_rows = transpose_b ? n : k;
      const dim_t b_cols = transpose_b ? k : n;

      if (!mps_matrix_layout_supported(a_rows, a_cols, lda, element_size)
          || !mps_matrix_layout_supported(b_rows, b_cols, ldb, element_size)
          || !mps_matrix_layout_supported(m, n, ldc, element_size)) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 1, 0, 0, 0, "generic_unaligned");
        generic_gemm(dtype, transpose_a, transpose_b, m, n, k, alpha,
                     a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1);
        return;
      }

      log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
               1, 0, 0, 0, "mps_matrix");

      id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
      if (!device)
        throw std::runtime_error("MPS device/queue not available");

      NSUInteger offset_a = 0;
      NSUInteger offset_b = 0;
      NSUInteger offset_c = 0;
      id<MTLBuffer> buf_a = mtl_buffer(a, dense_matrix_bytes(a_rows, a_cols, lda, element_size), offset_a);
      id<MTLBuffer> buf_b = mtl_buffer(b, dense_matrix_bytes(b_rows, b_cols, ldb, element_size), offset_b);
      id<MTLBuffer> buf_c = mtl_buffer(c, dense_matrix_bytes(m, n, ldc, element_size), offset_c);

      @autoreleasepool {
        const MPSDataType matrix_type = mps_matrix_data_type(dtype);
        MPSMatrixDescriptor* desc_a =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(a_rows)
                                                columns:static_cast<NSUInteger>(a_cols)
                                               rowBytes:static_cast<NSUInteger>(lda * element_size)
                                               dataType:matrix_type];
        MPSMatrixDescriptor* desc_b =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(b_rows)
                                                columns:static_cast<NSUInteger>(b_cols)
                                               rowBytes:static_cast<NSUInteger>(ldb * element_size)
                                               dataType:matrix_type];
        MPSMatrixDescriptor* desc_c =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(m)
                                                columns:static_cast<NSUInteger>(n)
                                               rowBytes:static_cast<NSUInteger>(ldc * element_size)
                                               dataType:matrix_type];
        // MPS descriptor factories return autoreleased objects. Keep them
        // alive until GPU completion while draining the worker-local pool now.
        record_metal_object_use((__bridge void*)desc_a);
        record_metal_object_use((__bridge void*)desc_b);
        record_metal_object_use((__bridge void*)desc_c);

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a offset:offset_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b offset:offset_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c offset:offset_c descriptor:desc_c];
        record_metal_object_use((__bridge void*)mat_a);
        record_metal_object_use((__bridge void*)mat_b);
        record_metal_object_use((__bridge void*)mat_c);

        MPSMatrixMultiplication* matmul = gemm_kernel(device,
                                                      dtype,
                                                      transpose_a,
                                                      transpose_b,
                                                      m,
                                                      n,
                                                      k,
                                                      1,
                                                      alpha,
                                                      beta);

        end_active_encoder();
        id<MTLCommandBuffer> active_command_buffer =
          (__bridge id<MTLCommandBuffer>)command_buffer();
        [matmul encodeToCommandBuffer:active_command_buffer
                           leftMatrix:mat_a
                          rightMatrix:mat_b
                          resultMatrix:mat_c];
        record_compute_dispatch("mps_matrix_gemm");
        record_profile_event(ProfileEvent::Gemm);

#if !__has_feature(objc_arc)
        [mat_a release];
        [mat_b release];
        [mat_c release];
#endif
      }
    }

    void gemm_batch_strided(DataType dtype,
                            bool transpose_a,
                            bool transpose_b,
                            dim_t m,
                            dim_t n,
                            dim_t k,
                            float alpha,
                            const void* a,
                            dim_t lda,
                            dim_t stridea,
                            const void* b,
                            dim_t ldb,
                            dim_t strideb,
                            float beta,
                            void* c,
                            dim_t ldc,
                            dim_t stridec,
                            dim_t batch_size) {
      if (batch_size <= 0 || m == 0 || n == 0)
        return;
      if (m == 1 && custom_gemv_enabled()) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 batch_size, stridea, strideb, stridec, "batched_row_gemv");
        gemv_row(dtype,
                 transpose_a,
                 transpose_b,
                 n,
                 k,
                 alpha,
                 a,
                 lda,
                 stridea,
                 b,
                 ldb,
                 strideb,
                 beta,
                 c,
                 ldc,
                 stridec,
                 batch_size);
        return;
      }
      if (n == 1 && custom_gemv_enabled()) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 batch_size, stridea, strideb, stridec, "batched_column_gemv");
        gemv(dtype,
             transpose_a,
             transpose_b,
             m,
             k,
             alpha,
             a,
             lda,
             stridea,
             b,
             ldb,
             strideb,
             beta,
             c,
             ldc,
             stridec,
             batch_size);
        return;
      }
      if (batch_size == 1) {
        gemm(dtype,
             transpose_a,
             transpose_b,
             m,
             n,
             k,
             alpha,
             a,
             lda,
             b,
             ldb,
             beta,
             c,
             ldc);
        return;
      }
      if (!supports_gemm_type(dtype))
        throw std::invalid_argument("unsupported MPS batched GEMM dtype");

      if (dtype == DataType::BFLOAT16) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 batch_size, stridea, strideb, stridec, "bf16_tiled_batched");
        generic_gemm(dtype, transpose_a, transpose_b, m, n, k, alpha,
                     a, lda, stridea, b, ldb, strideb, beta,
                     c, ldc, stridec, batch_size);
        return;
      }

      const dim_t a_rows = transpose_a ? k : m;
      const dim_t a_cols = transpose_a ? m : k;
      const dim_t b_rows = transpose_b ? n : k;
      const dim_t b_cols = transpose_b ? k : n;
      const size_t a_elements = dense_matrix_elements(a_rows, a_cols, lda);
      const size_t b_elements = dense_matrix_elements(b_rows, b_cols, ldb);
      const size_t c_elements = dense_matrix_elements(m, n, ldc);
      const size_t element_size = dtype_size(dtype);

      if (stridea < 0 || strideb < 0 || stridec <= 0)
        throw std::invalid_argument("MPS batched GEMM received an invalid negative/output stride");
      const bool invalid_dense_stride =
          (stridea > 0 && static_cast<size_t>(stridea) < a_elements)
          || (strideb > 0 && static_cast<size_t>(strideb) < b_elements)
          || static_cast<size_t>(stridec) < c_elements;
      const bool unsupported_mps_layout =
        !mps_matrix_layout_supported(a_rows, a_cols, lda, element_size)
        || !mps_matrix_layout_supported(b_rows, b_cols, ldb, element_size)
        || !mps_matrix_layout_supported(m, n, ldc, element_size)
        || stridea == 0
        || strideb == 0
        || (static_cast<size_t>(stridea) * element_size) % 16 != 0
        || (static_cast<size_t>(strideb) * element_size) % 16 != 0
        || (static_cast<size_t>(stridec) * element_size) % 16 != 0;
      if (invalid_dense_stride || unsupported_mps_layout) {
        log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
                 batch_size, stridea, strideb, stridec, "generic_batched");
        generic_gemm(dtype, transpose_a, transpose_b, m, n, k, alpha,
                     a, lda, stridea, b, ldb, strideb, beta,
                     c, ldc, stridec, batch_size);
        return;
      }

      log_gemm(dtype, transpose_a, transpose_b, m, n, k, lda, ldb, ldc,
               batch_size, stridea, strideb, stridec, "mps_matrix_batched");

      id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
      if (!device)
        throw std::runtime_error("MPS device/queue not available");

      NSUInteger offset_a = 0;
      NSUInteger offset_b = 0;
      NSUInteger offset_c = 0;
      id<MTLBuffer> buf_a =
        mtl_buffer(a, strided_matrix_array_bytes(a_rows, a_cols, lda, stridea, batch_size, element_size), offset_a);
      id<MTLBuffer> buf_b =
        mtl_buffer(b, strided_matrix_array_bytes(b_rows, b_cols, ldb, strideb, batch_size, element_size), offset_b);
      id<MTLBuffer> buf_c =
        mtl_buffer(c, strided_matrix_array_bytes(m, n, ldc, stridec, batch_size, element_size), offset_c);

      @autoreleasepool {
        const MPSDataType matrix_type = mps_matrix_data_type(dtype);
        MPSMatrixDescriptor* desc_a =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(a_rows)
                                                columns:static_cast<NSUInteger>(a_cols)
                                               matrices:static_cast<NSUInteger>(batch_size)
                                               rowBytes:static_cast<NSUInteger>(lda * element_size)
                                            matrixBytes:static_cast<NSUInteger>(stridea * element_size)
                                               dataType:matrix_type];
        MPSMatrixDescriptor* desc_b =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(b_rows)
                                                columns:static_cast<NSUInteger>(b_cols)
                                               matrices:static_cast<NSUInteger>(batch_size)
                                               rowBytes:static_cast<NSUInteger>(ldb * element_size)
                                            matrixBytes:static_cast<NSUInteger>(strideb * element_size)
                                               dataType:matrix_type];
        MPSMatrixDescriptor* desc_c =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(m)
                                                columns:static_cast<NSUInteger>(n)
                                               matrices:static_cast<NSUInteger>(batch_size)
                                               rowBytes:static_cast<NSUInteger>(ldc * element_size)
                                            matrixBytes:static_cast<NSUInteger>(stridec * element_size)
                                               dataType:matrix_type];
        record_metal_object_use((__bridge void*)desc_a);
        record_metal_object_use((__bridge void*)desc_b);
        record_metal_object_use((__bridge void*)desc_c);

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a offset:offset_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b offset:offset_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c offset:offset_c descriptor:desc_c];
        record_metal_object_use((__bridge void*)mat_a);
        record_metal_object_use((__bridge void*)mat_b);
        record_metal_object_use((__bridge void*)mat_c);
        MPSMatrixMultiplication* matmul = gemm_kernel(device,
                                                      dtype,
                                                      transpose_a,
                                                      transpose_b,
                                                      m,
                                                      n,
                                                      k,
                                                      batch_size,
                                                      alpha,
                                                      beta);

        end_active_encoder();
        id<MTLCommandBuffer> active_command_buffer =
          (__bridge id<MTLCommandBuffer>)command_buffer();
        [matmul encodeToCommandBuffer:active_command_buffer
                           leftMatrix:mat_a
                          rightMatrix:mat_b
                          resultMatrix:mat_c];
        record_compute_dispatch("mps_matrix_batched_gemm");
        record_profile_event(ProfileEvent::Gemm);

#if !__has_feature(objc_arc)
        [mat_a release];
        [mat_b release];
        [mat_c release];
#endif
      }
    }

    bool supports_topk(DataType dtype, dim_t k) {
      if (!gpu_topk_enabled()
          || (dtype != DataType::FLOAT32
              && dtype != DataType::FLOAT16
              && dtype != DataType::BFLOAT16)
          || k <= 0
          || k > 16)
        return false;
      // The custom reduction supports the search-critical k values without
      // MPSMatrix's 16-byte result-row restriction. Larger aligned results use
      // MPSMatrixFindTopK.
      if (dtype == DataType::BFLOAT16)
        return k <= 8;
      return k <= 8 || (static_cast<size_t>(k) * dtype_size(dtype)) % 16 == 0;
    }

    static void argmax(DataType dtype,
                       const void* input,
                       void* values,
                       int32_t* indices,
                       dim_t batch_size,
                       dim_t depth) {
      const size_t element_size = dtype_size(dtype);
      const SoftmaxArgs args{static_cast<uint64_t>(batch_size),
                             static_cast<uint64_t>(depth),
                             0,
                             0};
      id<MTLComputePipelineState> state = pipeline(kernel_name("argmax", dtype));
      id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
      [encoder setComputePipelineState:state];
      set_buffer(encoder, input, static_cast<size_t>(batch_size * depth) * element_size, 0);
      set_buffer(encoder, values, static_cast<size_t>(batch_size) * element_size, 1);
      set_buffer(encoder, indices, static_cast<size_t>(batch_size) * sizeof(int32_t), 2);
      [encoder setBytes:&args length:sizeof(args) atIndex:3];
      NSUInteger threads = reduction_threads(depth);
      while (threads > state.maxTotalThreadsPerThreadgroup)
        threads >>= 1;
      const NSUInteger scratch_bytes = ((threads * sizeof(float) + 15) / 16) * 16;
      [encoder setThreadgroupMemoryLength:scratch_bytes atIndex:0];
      [encoder setThreadgroupMemoryLength:scratch_bytes atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      record_compute_dispatch(kernel_name("argmax", dtype).c_str());
      record_profile_event(ProfileEvent::TopKGpu);
    }

    static void small_topk(DataType dtype,
                           const void* input,
                           void* values,
                           int32_t* indices,
                           dim_t batch_size,
                           dim_t depth,
                           dim_t k) {
      if (batch_size > UINT32_MAX || depth > UINT32_MAX || k > 8)
        throw std::invalid_argument("small MPS TopK dimensions are too large");
      const size_t element_size = dtype_size(dtype);
      const SmallTopKArgs args{static_cast<uint32_t>(batch_size),
                               static_cast<uint32_t>(depth),
                               static_cast<uint32_t>(k)};
      id<MTLComputePipelineState> state = pipeline(storage_kernel_name("small_topk", dtype));
      id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
      [encoder setComputePipelineState:state];
      set_buffer(encoder, input, static_cast<size_t>(batch_size * depth) * element_size, 0);
      set_buffer(encoder, values, static_cast<size_t>(batch_size * k) * element_size, 1);
      set_buffer(encoder, indices, static_cast<size_t>(batch_size * k) * sizeof(int32_t), 2);
      [encoder setBytes:&args length:sizeof(args) atIndex:3];
      NSUInteger threads = 256;
      while (threads > state.maxTotalThreadsPerThreadgroup)
        threads >>= 1;
      const NSUInteger scratch_bytes = ((threads * sizeof(float) + 15) / 16) * 16;
      [encoder setThreadgroupMemoryLength:scratch_bytes atIndex:0];
      [encoder setThreadgroupMemoryLength:scratch_bytes atIndex:1];
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      record_compute_dispatch(kernel_name("small_topk", dtype).c_str());
      record_profile_event(ProfileEvent::TopKGpu);
    }

    void topk(DataType dtype,
              const void* input,
              void* values,
              int32_t* indices,
              dim_t batch_size,
              dim_t depth,
              dim_t k) {
      if (!supports_topk(dtype, k))
        throw std::invalid_argument("unsupported MPS TopK configuration");
      if (batch_size == 0 || depth == 0 || k == 0)
        return;
      if (k == 1) {
        argmax(dtype, input, values, indices, batch_size, depth);
        return;
      }
      if (k <= 8) {
        small_topk(dtype, input, values, indices, batch_size, depth, k);
        return;
      }

      id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
      if (!device)
        throw std::runtime_error("MPS device/queue not available");

      const size_t element_size = dtype_size(dtype);
      NSUInteger input_offset = 0;
      NSUInteger values_offset = 0;
      NSUInteger indices_offset = 0;
      id<MTLBuffer> input_buffer =
        mtl_buffer(input, static_cast<size_t>(batch_size * depth) * element_size, input_offset);
      id<MTLBuffer> values_buffer =
        mtl_buffer(values, static_cast<size_t>(batch_size * k) * element_size, values_offset);
      id<MTLBuffer> indices_buffer =
        mtl_buffer(indices, static_cast<size_t>(batch_size * k) * sizeof(uint32_t), indices_offset);

      @autoreleasepool {
        MPSMatrixDescriptor* input_desc =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(batch_size)
                                                columns:static_cast<NSUInteger>(depth)
                                               rowBytes:static_cast<NSUInteger>(depth * element_size)
                                               dataType:mps_matrix_data_type(dtype)];
        MPSMatrixDescriptor* values_desc =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(batch_size)
                                                columns:static_cast<NSUInteger>(k)
                                               rowBytes:static_cast<NSUInteger>(k * element_size)
                                               dataType:mps_matrix_data_type(dtype)];
        MPSMatrixDescriptor* indices_desc =
          [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(batch_size)
                                                columns:static_cast<NSUInteger>(k)
                                               rowBytes:static_cast<NSUInteger>(k * sizeof(uint32_t))
                                               dataType:MPSDataTypeUInt32];
        record_metal_object_use((__bridge void*)input_desc);
        record_metal_object_use((__bridge void*)values_desc);
        record_metal_object_use((__bridge void*)indices_desc);

        MPSMatrix* input_matrix = [[MPSMatrix alloc] initWithBuffer:input_buffer
                                                            offset:input_offset
                                                        descriptor:input_desc];
        MPSMatrix* values_matrix = [[MPSMatrix alloc] initWithBuffer:values_buffer
                                                              offset:values_offset
                                                          descriptor:values_desc];
        MPSMatrix* indices_matrix = [[MPSMatrix alloc] initWithBuffer:indices_buffer
                                                               offset:indices_offset
                                                           descriptor:indices_desc];
        record_metal_object_use((__bridge void*)input_matrix);
        record_metal_object_use((__bridge void*)values_matrix);
        record_metal_object_use((__bridge void*)indices_matrix);

        static thread_local std::unordered_map<dim_t, MPSMatrixFindTopK*> topk_cache;
        MPSMatrixFindTopK* topk_kernel = nil;
        auto cache_it = topk_cache.find(k);
        if (cache_it == topk_cache.end()) {
          topk_kernel = [[MPSMatrixFindTopK alloc]
            initWithDevice:device
            numberOfTopKValues:static_cast<NSUInteger>(k)];
          if (!topk_kernel)
            throw std::runtime_error("failed to create MPS TopK kernel");
          topk_cache.emplace(k, topk_kernel);
        } else {
          topk_kernel = cache_it->second;
        }
        topk_kernel.sourceRows = static_cast<NSUInteger>(batch_size);
        topk_kernel.sourceColumns = static_cast<NSUInteger>(depth);

        end_active_encoder();
        id<MTLCommandBuffer> active_command_buffer =
          (__bridge id<MTLCommandBuffer>)command_buffer();
        [topk_kernel encodeToCommandBuffer:active_command_buffer
                               inputMatrix:input_matrix
                         resultIndexMatrix:indices_matrix
                         resultValueMatrix:values_matrix];
        record_compute_dispatch("mps_matrix_topk");
        record_profile_event(ProfileEvent::TopKGpu);

#if !__has_feature(objc_arc)
        [input_matrix release];
        [values_matrix release];
        [indices_matrix release];
#endif
      }
    }

    void fill(DataType dtype, const void* value, void* y, dim_t size) {
      if (size == 0)
        return;

      FillArgs args{static_cast<uint64_t>(size), 0.0f, 0};
      switch (dtype) {
      case DataType::FLOAT32:
        args.float_value = *static_cast<const float*>(value);
        break;
      case DataType::FLOAT16:
        args.float_value = static_cast<float>(*static_cast<const float16_t*>(value));
        break;
      case DataType::BFLOAT16:
        args.float_value = static_cast<float>(*static_cast<const bfloat16_t*>(value));
        break;
      case DataType::INT8:
        args.int_value = static_cast<int32_t>(*static_cast<const int8_t*>(value));
        break;
      case DataType::INT16:
        args.int_value = static_cast<int32_t>(*static_cast<const int16_t*>(value));
        break;
      case DataType::INT32:
        args.int_value = *static_cast<const int32_t*>(value);
        break;
      default:
        throw std::invalid_argument("unsupported MPS fill dtype");
      }

      run_1d(storage_kernel_name("fill", dtype),
             size,
             {{y, static_cast<size_t>(size) * dtype_size(dtype)}},
             &args,
             sizeof(args),
             1);
    }

    void indexed_fill(DataType dtype,
                      const void* value,
                      void* y,
                      const int32_t* indices,
                      dim_t size) {
      if (size == 0)
        return;
      if (size > UINT32_MAX)
        throw std::invalid_argument("MPS indexed fill is too large");
      IndexedFillArgs args{static_cast<uint32_t>(size), 0.0f, 0};
      switch (dtype) {
      case DataType::FLOAT32:
        args.float_value = *static_cast<const float*>(value);
        break;
      case DataType::FLOAT16:
        args.float_value = static_cast<float>(*static_cast<const float16_t*>(value));
        break;
      case DataType::BFLOAT16:
        args.float_value = static_cast<float>(*static_cast<const bfloat16_t*>(value));
        break;
      case DataType::INT8:
        args.int_value = *static_cast<const int8_t*>(value);
        break;
      case DataType::INT16:
        args.int_value = *static_cast<const int16_t*>(value);
        break;
      case DataType::INT32:
        args.int_value = *static_cast<const int32_t*>(value);
        break;
      default:
        throw std::invalid_argument("unsupported MPS indexed fill dtype");
      }
      run_1d(storage_kernel_name("indexed_fill", dtype),
             size,
             {{y, dtype_size(dtype)},
              {indices, static_cast<size_t>(size) * sizeof(int32_t)}},
             &args,
             sizeof(args),
             2);
    }

    void penalize_previous_tokens(DataType dtype,
                                  void* scores,
                                  const void* previous_scores,
                                  const int32_t* previous_ids,
                                  float penalty,
                                  dim_t batch_size,
                                  dim_t length,
                                  dim_t vocabulary_size) {
      if (batch_size <= 0 || length <= 0 || vocabulary_size <= 0)
        return;
      const uint64_t total = static_cast<uint64_t>(batch_size)
                             * static_cast<uint64_t>(length);
      if (!fits_u32(batch_size) || !fits_u32(length) || !fits_u32(vocabulary_size)
          || total > UINT32_MAX)
        throw std::invalid_argument("MPS repetition penalty dimensions are too large");

      const size_t element_size = dtype_size(dtype);
      const size_t scores_bytes = static_cast<size_t>(batch_size)
                                  * static_cast<size_t>(vocabulary_size)
                                  * element_size;
      const size_t previous_scores_bytes = static_cast<size_t>(total) * element_size;
      const size_t previous_ids_bytes = static_cast<size_t>(total) * sizeof(int32_t);
      const RepetitionPenaltyArgs args{static_cast<uint32_t>(total),
                                       static_cast<uint32_t>(length),
                                       static_cast<uint32_t>(vocabulary_size),
                                       penalty};
      run_1d(kernel_name("repetition_penalty", dtype),
             total,
             {{scores, scores_bytes},
              {previous_scores, previous_scores_bytes},
              {previous_ids, previous_ids_bytes}},
             &args,
             sizeof(args),
             3);
    }

    void prepare_length_mask(const int32_t* lengths,
                             dim_t batch_size,
                             dim_t num_heads,
                             dim_t num_queries,
                             bool mask_future,
                             bool multi_query,
                             int32_t* mask) {
      if (!fits_u32(batch_size) || !fits_u32(num_heads) || !fits_u32(num_queries))
        throw std::invalid_argument("MPS length mask dimensions are too large");
      const uint64_t total = static_cast<uint64_t>(batch_size)
                             * static_cast<uint64_t>(num_heads)
                             * static_cast<uint64_t>(num_queries);
      const LengthMaskArgs args{static_cast<uint32_t>(batch_size),
                                static_cast<uint32_t>(num_heads),
                                static_cast<uint32_t>(num_queries),
                                mask_future ? 1u : 0u,
                                multi_query ? 1u : 0u};
      run_1d("prepare_length_mask",
             total,
             {{lengths, static_cast<size_t>(batch_size) * sizeof(int32_t)},
              {mask, static_cast<size_t>(total) * sizeof(int32_t)}},
             &args,
             sizeof(args),
             2);
    }

    void unary(DataType dtype, UnaryOp op, const void* x, void* y, dim_t size) {
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      const ElementwiseArgs args{static_cast<uint64_t>(size), op_code(op), 0};
      run_1d(kernel_name("unary", dtype), size, {{x, bytes}, {y, bytes}}, &args, sizeof(args), 2);
    }

    void binary(DataType dtype, BinaryOp op, const void* a, const void* b, void* c, dim_t size) {
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      const ElementwiseArgs args{static_cast<uint64_t>(size), op_code(op), 0};
      run_1d(kernel_name("binary", dtype), size, {{a, bytes}, {b, bytes}, {c, bytes}}, &args, sizeof(args), 3);
    }

    void scalar(DataType dtype, BinaryOp op, float a, const void* x, void* y, dim_t size) {
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      const ElementwiseArgs args{static_cast<uint64_t>(size), op_code(op), a};
      run_1d(kernel_name("scalar", dtype), size, {{x, bytes}, {y, bytes}}, &args, sizeof(args), 2);
    }

    void quantize(DataType dtype,
                  const void* input,
                  int8_t* output,
                  float* scales,
                  dim_t batch_size,
                  dim_t depth,
                  bool round_before_cast) {
      if (batch_size <= 0 || depth <= 0)
        return;
      const size_t input_bytes = static_cast<size_t>(batch_size * depth) * dtype_size(dtype);
      const QuantizeArgs args{static_cast<uint64_t>(batch_size),
                              static_cast<uint64_t>(depth),
                              round_before_cast ? 1u : 0u};
      run_rows(kernel_name("quantize", dtype),
               batch_size,
               reduction_threads(depth),
               {{input, input_bytes},
                {output, static_cast<size_t>(batch_size * depth)},
                {scales, static_cast<size_t>(batch_size) * sizeof(float)}},
               &args,
               sizeof(args),
               3);
    }

    void dequantize(DataType dtype,
                    const int8_t* input,
                    const float* scales,
                    void* output,
                    dim_t batch_size,
                    dim_t depth) {
      if (batch_size <= 0 || depth <= 0)
        return;
      const uint64_t total = static_cast<uint64_t>(batch_size * depth);
      const DequantizeArgs args{total, static_cast<uint64_t>(depth)};
      run_1d(kernel_name("dequantize", dtype),
             total,
             {{input, static_cast<size_t>(total)},
              {scales, static_cast<size_t>(batch_size) * sizeof(float)},
              {output, static_cast<size_t>(total) * dtype_size(dtype)}},
             &args,
             sizeof(args),
             3);
    }

    void dequantize_gemm_output(DataType dtype,
                                const int32_t* input,
                                const float* a_scales,
                                dim_t a_scale_size,
                                const float* b_scales,
                                dim_t b_scale_size,
                                bool transpose_a,
                                bool transpose_b,
                                const void* bias,
                                void* output,
                                dim_t batch_size,
                                dim_t depth,
                                int activation) {
      if (batch_size <= 0 || depth <= 0)
        return;
      const uint64_t total = static_cast<uint64_t>(batch_size * depth);
      const size_t element_size = dtype_size(dtype);
      const void* bias_or_dummy = bias ? bias : output;
      const DequantizeGemmArgs args{static_cast<uint64_t>(batch_size),
                                    static_cast<uint64_t>(depth),
                                    static_cast<uint64_t>(a_scale_size),
                                    static_cast<uint64_t>(b_scale_size),
                                    transpose_a ? 1u : 0u,
                                    transpose_b ? 1u : 0u,
                                    bias ? 1u : 0u,
                                    activation};
      run_1d(kernel_name("dequantize_gemm", dtype),
             total,
             {{input, static_cast<size_t>(total) * sizeof(int32_t)},
              {a_scales, static_cast<size_t>(a_scale_size) * sizeof(float)},
              {b_scales, static_cast<size_t>(b_scale_size) * sizeof(float)},
              {bias_or_dummy, bias ? static_cast<size_t>(depth) * element_size : element_size},
              {output, static_cast<size_t>(total) * element_size}},
             &args,
             sizeof(args),
             5);
    }

    void gather(DataType dtype,
                const void* data,
                const int32_t* indices,
                void* output,
                dim_t copy_size,
                dim_t batch_stride,
                dim_t num_indices,
                dim_t num_indices_per_batch) {
      if (num_indices == 0 || copy_size == 0)
        return;
      const dim_t batch_size = num_indices / num_indices_per_batch;
      const size_t element_size = dtype_size(dtype);
      const uint64_t output_size = static_cast<uint64_t>(num_indices * copy_size);
      const GatherArgs args{static_cast<uint64_t>(copy_size),
                            static_cast<uint64_t>(batch_stride),
                            static_cast<uint64_t>(num_indices),
                            static_cast<uint64_t>(num_indices_per_batch)};
      run_1d(storage_kernel_name("gather", dtype),
             output_size,
             {{data, static_cast<size_t>(batch_size * batch_stride) * element_size},
              {indices, static_cast<size_t>(num_indices) * sizeof(int32_t)},
              {output, static_cast<size_t>(output_size) * element_size}},
             &args,
             sizeof(args),
             3);
    }

    void concat2(DataType dtype,
                 const void* a,
                 dim_t a_block_size,
                 const void* b,
                 dim_t b_block_size,
                 void* output,
                 dim_t outer_size) {
      if (outer_size <= 0 || a_block_size <= 0 || b_block_size <= 0)
        return;
      const size_t element_size = dtype_size(dtype);
      const uint64_t output_block = static_cast<uint64_t>(a_block_size + b_block_size);
      const uint64_t output_size = static_cast<uint64_t>(outer_size) * output_block;
      const Concat2Args args{static_cast<uint64_t>(outer_size),
                             static_cast<uint64_t>(a_block_size),
                             static_cast<uint64_t>(b_block_size)};
      run_1d(storage_kernel_name("concat2", dtype),
             output_size,
             {{a, static_cast<size_t>(outer_size * a_block_size) * element_size},
              {b, static_cast<size_t>(outer_size * b_block_size) * element_size},
              {output, static_cast<size_t>(output_size) * element_size}},
             &args,
             sizeof(args),
             3);
    }

    void tile(DataType dtype,
              const void* input,
              void* output,
              dim_t outer_size,
              dim_t inner_size,
              dim_t num_tiles) {
      const uint64_t output_size =
        static_cast<uint64_t>(outer_size * inner_size * num_tiles);
      if (output_size == 0)
        return;
      const size_t element_size = dtype_size(dtype);
      const TileArgs args{static_cast<uint64_t>(outer_size),
                          static_cast<uint64_t>(inner_size),
                          static_cast<uint64_t>(num_tiles)};
      run_1d(storage_kernel_name("tile", dtype),
             output_size,
             {{input, static_cast<size_t>(outer_size * inner_size) * element_size},
              {output, static_cast<size_t>(output_size) * element_size}},
             &args,
             sizeof(args),
             2);
    }

    void bias_add(DataType dtype,
                  const void* bias,
                  const void* value,
                  const void* residual,
                  void* output,
                  dim_t bias_size,
                  dim_t value_size,
                  dim_t block,
                  int activation) {
      if (value_size == 0)
        return;
      const size_t element_size = dtype_size(dtype);
      const void* residual_or_dummy = residual ? residual : value;
      const BiasAddArgs args{static_cast<uint64_t>(bias_size),
                             static_cast<uint64_t>(value_size),
                             static_cast<uint64_t>(block),
                             block > 0 ? 1u : 0u,
                             residual ? 1u : 0u,
                             activation};
      run_1d(kernel_name("bias_add", dtype),
             value_size,
             {{bias, static_cast<size_t>(bias_size) * element_size},
              {value, static_cast<size_t>(value_size) * element_size},
              {residual_or_dummy, residual ? static_cast<size_t>(value_size) * element_size
                                           : element_size},
              {output, static_cast<size_t>(value_size) * element_size}},
             &args,
             sizeof(args),
             4);
    }

    void batch_broadcast(DataType dtype,
                         BinaryOp op,
                         const void* a,
                         const void* b,
                         void* c,
                         dim_t a_size,
                         dim_t b_size) {
      const size_t element_size = dtype_size(dtype);
      const BroadcastArgs args{static_cast<uint64_t>(a_size),
                               static_cast<uint64_t>(b_size),
                               0,
                               op_code(op),
                               0};
      run_1d(kernel_name("broadcast", dtype),
             b_size,
             {{a, static_cast<size_t>(a_size) * element_size},
              {b, static_cast<size_t>(b_size) * element_size},
              {c, static_cast<size_t>(b_size) * element_size}},
             &args,
             sizeof(args),
             3);
    }

    void depth_broadcast(DataType dtype,
                         BinaryOp op,
                         const void* a,
                         const void* b,
                         void* c,
                         dim_t a_size,
                         dim_t b_size) {
      const size_t element_size = dtype_size(dtype);
      const BroadcastArgs args{static_cast<uint64_t>(a_size),
                               static_cast<uint64_t>(b_size),
                               0,
                               op_code(op),
                               1};
      run_1d(kernel_name("broadcast", dtype),
             b_size,
             {{a, static_cast<size_t>(a_size) * element_size},
              {b, static_cast<size_t>(b_size) * element_size},
              {c, static_cast<size_t>(b_size) * element_size}},
             &args,
             sizeof(args),
             3);
    }

    void block_broadcast(DataType dtype,
                         BinaryOp op,
                         const void* a,
                         const void* b,
                         void* c,
                         dim_t block,
                         dim_t a_size,
                         dim_t b_size) {
      const size_t element_size = dtype_size(dtype);
      const BroadcastArgs args{static_cast<uint64_t>(a_size),
                               static_cast<uint64_t>(b_size),
                               static_cast<uint64_t>(block),
                               op_code(op),
                               2};
      run_1d(kernel_name("broadcast", dtype),
             b_size,
             {{a, static_cast<size_t>(a_size) * element_size},
              {b, static_cast<size_t>(b_size) * element_size},
              {c, static_cast<size_t>(b_size) * element_size}},
             &args,
             sizeof(args),
             3);
    }

    void transpose_2d(DataType dtype, const void* a, dim_t rows, dim_t cols, void* b) {
      const uint64_t size = static_cast<uint64_t>(rows * cols);
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      const Transpose2DArgs args{static_cast<uint64_t>(rows), static_cast<uint64_t>(cols)};
      run_1d(kernel_name("transpose_2d", dtype), size, {{a, bytes}, {b, bytes}}, &args, sizeof(args), 2);
    }

    void transpose_3d(DataType dtype, const void* a, const dim_t* dims, const dim_t* perm, void* b) {
      const uint64_t size = static_cast<uint64_t>(dims[0] * dims[1] * dims[2]);
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      const TransposeNDArgs args{static_cast<uint64_t>(dims[0]),
                                 static_cast<uint64_t>(dims[1]),
                                 static_cast<uint64_t>(dims[2]),
                                 1,
                                 static_cast<uint64_t>(perm[0]),
                                 static_cast<uint64_t>(perm[1]),
                                 static_cast<uint64_t>(perm[2]),
                                 3};
      run_1d(kernel_name("transpose_3d", dtype), size, {{a, bytes}, {b, bytes}}, &args, sizeof(args), 2);
    }

    void transpose_4d(DataType dtype, const void* a, const dim_t* dims, const dim_t* perm, void* b) {
      const uint64_t size = static_cast<uint64_t>(dims[0] * dims[1] * dims[2] * dims[3]);
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      const TransposeNDArgs args{static_cast<uint64_t>(dims[0]),
                                 static_cast<uint64_t>(dims[1]),
                                 static_cast<uint64_t>(dims[2]),
                                 static_cast<uint64_t>(dims[3]),
                                 static_cast<uint64_t>(perm[0]),
                                 static_cast<uint64_t>(perm[1]),
                                 static_cast<uint64_t>(perm[2]),
                                 static_cast<uint64_t>(perm[3])};
      run_1d(kernel_name("transpose_4d", dtype), size, {{a, bytes}, {b, bytes}}, &args, sizeof(args), 2);
    }

    void softmax(DataType dtype,
                 const void* input,
                 const int32_t* lengths,
                 void* output,
                 dim_t batch_size,
                 dim_t depth,
                 bool log) {
      const size_t element_size = dtype_size(dtype);
      const size_t bytes = static_cast<size_t>(batch_size * depth) * element_size;
      const SoftmaxArgs args{static_cast<uint64_t>(batch_size),
                             static_cast<uint64_t>(depth),
                             lengths ? 1u : 0u,
                             log ? 1u : 0u};
      const void* lengths_or_dummy = lengths ? static_cast<const void*>(lengths) : input;
      const size_t lengths_bytes = lengths ? static_cast<size_t>(batch_size) * sizeof(int32_t) : element_size;
      run_rows(kernel_name("softmax", dtype),
               batch_size,
               reduction_threads(depth),
               {{input, bytes}, {lengths_or_dummy, lengths_bytes}, {output, bytes}},
               &args,
               sizeof(args),
               3);
    }

    void mean(DataType dtype,
              const void* input,
              void* output,
              dim_t outer_size,
              dim_t axis_size,
              dim_t inner_size,
              bool get_sum) {
      const size_t element_size = dtype_size(dtype);
      const uint64_t output_size = static_cast<uint64_t>(outer_size * inner_size);
      const MeanArgs args{static_cast<uint64_t>(outer_size),
                          static_cast<uint64_t>(axis_size),
                          static_cast<uint64_t>(inner_size),
                          get_sum ? 1u : 0u};
      if (inner_size == 1) {
        run_rows(kernel_name("mean_rows", dtype),
                 outer_size,
                 reduction_threads(axis_size),
                 {{input, static_cast<size_t>(outer_size * axis_size) * element_size},
                  {output, static_cast<size_t>(outer_size) * element_size}},
                 &args,
                 sizeof(args),
                 2);
        return;
      }
      run_1d(kernel_name("mean", dtype),
             output_size,
             {{input, static_cast<size_t>(outer_size * axis_size * inner_size) * element_size},
              {output, static_cast<size_t>(output_size) * element_size}},
             &args,
             sizeof(args),
             2);
    }

    void layer_norm(DataType dtype,
                    const void* input,
                    const void* gamma,
                    const void* beta,
                    void* output,
                    dim_t outer_size,
                    dim_t axis_size,
                    dim_t inner_size,
                    float epsilon) {
      const size_t element_size = dtype_size(dtype);
      const size_t tensor_bytes = static_cast<size_t>(outer_size * axis_size * inner_size) * element_size;
      const size_t param_bytes = static_cast<size_t>(axis_size) * element_size;
      const void* gamma_or_dummy = gamma ? gamma : input;
      const void* beta_or_dummy = beta ? beta : input;
      const NormArgs args{static_cast<uint64_t>(outer_size),
                          static_cast<uint64_t>(axis_size),
                          static_cast<uint64_t>(inner_size),
                          epsilon,
                          gamma ? 1u : 0u,
                          beta ? 1u : 0u,
                          0u};
      run_rows(kernel_name("layer_norm", dtype),
               outer_size * inner_size,
               reduction_threads(axis_size),
               {{input, tensor_bytes},
                {gamma_or_dummy, gamma ? param_bytes : element_size},
                {beta_or_dummy, beta ? param_bytes : element_size},
                {output, tensor_bytes}},
               &args,
               sizeof(args),
               4);
    }

    void rms_norm(DataType dtype,
                  const void* input,
                  const void* gamma,
                  void* output,
                  dim_t batch_size,
                  dim_t depth,
                  float epsilon,
                  bool use_residual) {
      const size_t element_size = dtype_size(dtype);
      const size_t tensor_bytes = static_cast<size_t>(batch_size * depth) * element_size;
      const size_t gamma_bytes = static_cast<size_t>(depth) * element_size;
      const NormArgs args{static_cast<uint64_t>(batch_size),
                          static_cast<uint64_t>(depth),
                          1,
                          epsilon,
                          1u,
                          0u,
                          use_residual ? 1u : 0u};
      run_rows(kernel_name("rms_norm", dtype),
               batch_size,
               reduction_threads(depth),
               {{input, tensor_bytes}, {gamma, gamma_bytes}, {output, tensor_bytes}},
               &args,
               sizeof(args),
               3);
    }

    void rotary(DataType dtype,
                const void* input,
                const void* sin,
                const void* cos,
                void* output,
                dim_t batch_size,
                dim_t max_time,
                dim_t ndims,
                dim_t depth,
                bool interleave) {
      const size_t element_size = dtype_size(dtype);
      const uint64_t size = static_cast<uint64_t>(batch_size * max_time * depth);
      const RotaryArgs args{size,
                            static_cast<uint64_t>(max_time),
                            static_cast<uint64_t>(ndims),
                            static_cast<uint64_t>(depth),
                            interleave ? 1u : 0u};
      run_1d(kernel_name("rotary", dtype),
             size,
             {{input, static_cast<size_t>(size) * element_size},
              {sin, static_cast<size_t>(max_time * ndims) * element_size},
              {cos, static_cast<size_t>(max_time * ndims) * element_size},
              {output, static_cast<size_t>(size) * element_size}},
             &args,
             sizeof(args),
             4);
    }

    void im2col_conv1d(DataType dtype,
                       const void* input,
                       void* output,
                       dim_t batch_size,
                       dim_t groups,
                       dim_t input_length,
                       dim_t kernel_size,
                       dim_t stride,
                       dim_t padding,
                       dim_t dilation,
                       dim_t output_length,
                       dim_t k,
                       dim_t in_batch_stride,
                       dim_t in_group_stride) {
      const size_t element_size = dtype_size(dtype);
      const uint64_t total = static_cast<uint64_t>(batch_size * groups * output_length * k);
      const Im2Col1DArgs args{total,
                              static_cast<uint64_t>(batch_size),
                              static_cast<uint64_t>(groups),
                              static_cast<uint64_t>(input_length),
                              static_cast<uint64_t>(kernel_size),
                              static_cast<uint64_t>(stride),
                              static_cast<uint64_t>(padding),
                              static_cast<uint64_t>(dilation),
                              static_cast<uint64_t>(output_length),
                              static_cast<uint64_t>(k),
                              static_cast<uint64_t>(in_batch_stride),
                              static_cast<uint64_t>(in_group_stride)};
      run_1d(kernel_name("im2col_conv1d", dtype),
             total,
             {{input, static_cast<size_t>(batch_size * in_batch_stride) * element_size},
              {output, static_cast<size_t>(total) * element_size}},
             &args,
             sizeof(args),
             2);
    }

    void median_filter(DataType dtype,
                       const void* input,
                       void* output,
                       dim_t rows,
                       dim_t depth,
                       dim_t width) {
      if (rows <= 0 || depth <= 0)
        return;
      if (width <= 0 || (width & 1) == 0 || width > 129)
        throw std::invalid_argument("MPS MedianFilter requires an odd width no greater than 129");
      const uint64_t total = static_cast<uint64_t>(rows * depth);
      const size_t bytes = static_cast<size_t>(total) * dtype_size(dtype);
      const MedianFilterArgs args{total,
                                  static_cast<uint64_t>(depth),
                                  static_cast<uint64_t>(width)};
      run_1d(kernel_name("median_filter", dtype),
             total,
             {{input, bytes}, {output, bytes}},
             &args,
             sizeof(args),
             2);
    }

    static uint32_t next_power_of_two(uint32_t value) {
      if (value <= 1)
        return 1;
      --value;
      value |= value >> 1;
      value |= value >> 2;
      value |= value >> 4;
      value |= value >> 8;
      value |= value >> 16;
      return value + 1;
    }

    dim_t max_top_p_classes() {
      return 1024;
    }

    void top_p_mask(DataType dtype,
                    const void* input,
                    const void* probabilities,
                    void* output,
                    dim_t batch_size,
                    dim_t depth,
                    float probability,
                    float mask_value) {
      if (batch_size <= 0 || depth <= 0)
        return;
      if (depth > max_top_p_classes())
        throw std::invalid_argument("MPS TopPMask supports at most "
                                    + std::to_string(max_top_p_classes())
                                    + " classes");
      const uint32_t padded_depth = next_power_of_two(static_cast<uint32_t>(depth));
      const TopPMaskArgs args{static_cast<uint32_t>(batch_size),
                              static_cast<uint32_t>(depth),
                              padded_depth,
                              probability,
                              mask_value};
      const size_t bytes = static_cast<size_t>(batch_size * depth) * dtype_size(dtype);
      id<MTLComputePipelineState> state = pipeline(kernel_name("top_p_mask", dtype));
      id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
      [encoder setComputePipelineState:state];
      set_buffer(encoder, input, bytes, 0);
      set_buffer(encoder, probabilities, bytes, 1);
      set_buffer(encoder, output, bytes, 2);
      [encoder setBytes:&args length:sizeof(args) atIndex:3];
      [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(padded_depth) * sizeof(float)
                                  atIndex:0];
      [encoder setThreadgroupMemoryLength:static_cast<NSUInteger>(padded_depth) * sizeof(uint32_t)
                                  atIndex:1];
      NSUInteger threads = std::min<NSUInteger>(256, padded_depth);
      threads = std::min<NSUInteger>(threads, state.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      record_compute_dispatch(kernel_name("top_p_mask", dtype).c_str());
    }

    static std::atomic<uint64_t> g_random_counter{0};

    void multinomial(DataType dtype,
                     const void* probabilities,
                     int32_t* output,
                     dim_t batch_size,
                     dim_t depth,
                     dim_t sample_size) {
      if (batch_size <= 0 || depth <= 0 || sample_size <= 0)
        return;
      const uint64_t total = static_cast<uint64_t>(batch_size * sample_size);
      const uint64_t counter = g_random_counter.fetch_add(total);
      const RandomArgs args{total,
                            static_cast<uint64_t>(depth),
                            static_cast<uint64_t>(sample_size),
                            static_cast<uint64_t>(get_random_seed()),
                            counter};
      run_1d(kernel_name("multinomial", dtype),
             total,
             {{probabilities,
               static_cast<size_t>(batch_size * depth) * dtype_size(dtype)},
              {output, static_cast<size_t>(total) * sizeof(int32_t)}},
             &args,
             sizeof(args),
             2);
    }

    void gumbel_noise(DataType dtype,
                      const void* input,
                      void* output,
                      dim_t size) {
      if (size <= 0)
        return;
      const uint64_t total = static_cast<uint64_t>(size);
      const uint64_t counter = g_random_counter.fetch_add(total);
      const RandomArgs args{total,
                            0,
                            0,
                            static_cast<uint64_t>(get_random_seed()),
                            counter};
      const size_t bytes = static_cast<size_t>(size) * dtype_size(dtype);
      run_1d(kernel_name("gumbel_noise", dtype),
             total,
             {{input, bytes}, {output, bytes}},
             &args,
             sizeof(args),
             2);
    }

    void alibi_add(DataType dtype,
                   const void* input,
                   const void* alibi,
                   void* output,
                   dim_t batch_size,
                   dim_t num_heads,
                   dim_t query_length,
                   dim_t key_length,
                   dim_t cached_key_length,
                   dim_t alibi_offset) {
      const uint64_t total = static_cast<uint64_t>(batch_size)
                             * static_cast<uint64_t>(num_heads)
                             * static_cast<uint64_t>(query_length)
                             * static_cast<uint64_t>(key_length);
      if (total == 0)
        return;
      const size_t element_size = dtype_size(dtype);
      const AlibiArgs args{total,
                           static_cast<uint64_t>(num_heads),
                           static_cast<uint64_t>(query_length),
                           static_cast<uint64_t>(key_length),
                           static_cast<uint64_t>(cached_key_length),
                           static_cast<uint64_t>(alibi_offset)};
      run_1d(kernel_name("alibi_add", dtype),
             total,
             {{input, static_cast<size_t>(total) * element_size},
              {alibi, static_cast<size_t>(num_heads * cached_key_length) * element_size},
              {output, static_cast<size_t>(total) * element_size}},
             &args,
             sizeof(args),
             3);
    }

  }
}

#endif
#endif
