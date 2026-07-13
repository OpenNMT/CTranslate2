// primitives_mps.mm - FIXED version with complete template instantiations
// This eliminates undefined symbol errors by explicitly instantiating all templates

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include "mps/kernels.h"
#include "mps/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {

  // -------------------------
  // Small helpers (host/unified memory)
  // -------------------------

  template <typename T>
  static constexpr bool is_mps_float_type_v =
    std::is_same_v<T, float> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

  template <typename T>
  static inline float to_float(T v) {
    if constexpr (std::is_same_v<T, float>)
      return v;
    else
      return static_cast<float>(v);
  }

  template <>
  inline float to_float<float16_t>(float16_t v) {
    return static_cast<float>(v);
  }

  template <>
  inline float to_float<bfloat16_t>(bfloat16_t v) {
    return static_cast<float>(v);
  }

  template <typename T>
  static inline T from_float(float v) {
    if constexpr (std::is_same_v<T, float>)
      return v;
    else
      return static_cast<T>(v);
  }

  template <>
  inline float16_t from_float<float16_t>(float v) {
    return float16_t(v);
  }

  template <>
  inline bfloat16_t from_float<bfloat16_t>(float v) {
    return bfloat16_t(v);
  }

  template <typename T>
  static inline T abs_val(T v) {
    if constexpr (std::is_floating_point_v<T>)
      return std::fabs(v);
    else
      return v < 0 ? -v : v;
  }

  template <>
  inline float16_t abs_val<float16_t>(float16_t v) {
    return float16_t(std::fabs((float)v));
  }

  template <>
  inline bfloat16_t abs_val<bfloat16_t>(bfloat16_t v) {
    return bfloat16_t(std::fabs((float)v));
  }

  template <typename T>
  static inline T clamp_min(T x, T a) {
    return x < a ? a : x;
  }

  template <typename T>
  static inline T clamp_max(T x, T a) {
    return x > a ? a : x;
  }

  // -------------------------
  // Basic ops
  // -------------------------

  template<>
  template <typename T>
  T primitives<Device::MPS>::at(const T* x, dim_t index) {
    mps::synchronize("host scalar read");
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::fill(T* x, T a, dim_t size) {
    mps::fill(DataTypeToEnum<T>::value, &a, x, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    mps::synchronize("host strided fill");
    for (dim_t i = 0; i < size; ++i)
      x[i * inc_x] = a;
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    mps::indexed_fill(DataTypeToEnum<T>::value, &a, x, indices, num_indices);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::copy(const T* x, T* y, dim_t size) {
    const size_t bytes = static_cast<size_t>(size) * sizeof(T);
    if (bytes == 0 || x == y)
      return;

    size_t src_offset = 0;
    size_t dst_offset = 0;
    void* src_raw = mps::get_buffer(x, bytes, &src_offset);
    void* dst_raw = mps::get_buffer(y, bytes, &dst_offset);

    if (src_raw && dst_raw) {
      const bool same_buffer = src_raw == dst_raw;
      const bool overlaps = same_buffer
                            && src_offset < dst_offset + bytes
                            && dst_offset < src_offset + bytes;
      if (overlaps) {
        // Metal does not define memmove semantics for overlapping blits.
        mps::synchronize("overlapping copy");
        std::memmove(y, x, bytes);
        mps::record_profile_event(mps::ProfileEvent::CpuFallback);
        return;
      }
      // Cache growth and beam reordering create chains of small copies. A
      // compute copy participates in the stream's explicit buffer barriers,
      // unlike deferred blits that can observe a stale source in these chains.
      mps::tile(DataTypeToEnum<T>::value, x, y, 1, size, 1);
      mps::record_copy_bytes(bytes);
      return;
    }

    mps::synchronize("unregistered copy");
    std::memmove(y, x, bytes);
    mps::record_profile_event(mps::ProfileEvent::CpuFallback);
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::MPS>::convert(const U* x, V* y, dim_t size) {
    mps::synchronize("host conversion");
    for (dim_t i = 0; i < size; ++i)
      y[i] = static_cast<V>(x[i]);
  }

  template void primitives<Device::MPS>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::MPS>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::MPS>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const bfloat16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  T primitives<Device::MPS>::sum(const T* array, dim_t size) {
    mps::synchronize("host sum reduction");
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
      float acc = 0.f;
      for (dim_t i = 0; i < size; ++i)
        acc += to_float(array[i]);
      return from_float<T>(acc);
    } else {
      long long acc = 0;
      for (dim_t i = 0; i < size; ++i)
        acc += static_cast<long long>(array[i]);
      return static_cast<T>(acc);
    }
  }

  template<>
  template <typename T>
  dim_t primitives<Device::MPS>::max_element(const T* array, dim_t size) {
    mps::synchronize("host max index reduction");
    if (size <= 0)
      return 0;
    dim_t best = 0;
    for (dim_t i = 1; i < size; ++i) {
      if (array[i] > array[best])
        best = i;
    }
    return best;
  }

  template<>
  template <typename T>
  T primitives<Device::MPS>::max(const T* array, dim_t size) {
    mps::synchronize("host max reduction");
    if (size <= 0)
      return T();
    T best = array[0];
    for (dim_t i = 1; i < size; ++i)
      best = (array[i] > best ? array[i] : best);
    return best;
  }

  template<>
  template <typename T>
  T primitives<Device::MPS>::amax(const T* array, dim_t size) {
    mps::synchronize("host absolute max reduction");
    if (size <= 0)
      return T();
    T best = abs_val(array[0]);
    for (dim_t i = 1; i < size; ++i) {
      T v = abs_val(array[i]);
      best = (v > best ? v : best);
    }
    return best;
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add(T a, const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::scalar(DataTypeToEnum<T>::value, mps::BinaryOp::ADD, to_float(a), x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = x[i] + a;
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add(const T* a, const T* b, T* c, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::binary(DataTypeToEnum<T>::value, mps::BinaryOp::ADD, a, b, c, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      c[i] = a[i] + b[i];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::batch_broadcast(DataTypeToEnum<T>::value, mps::BinaryOp::ADD, a, b, c, a_size, b_size);
      return;
    }
    for (dim_t i = 0; i < b_size; ++i)
      c[i] = a[i % a_size] + b[i];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::depth_broadcast(DataTypeToEnum<T>::value, mps::BinaryOp::ADD, a, b, c, a_size, b_size);
      return;
    }
    const dim_t depth = (a_size == 0 ? 0 : b_size / a_size);
    for (dim_t i = 0; i < b_size; ++i) {
      const dim_t j = (depth == 0 ? 0 : (i / depth));
      c[i] = a[j] + b[i];
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add_block_broadcast(const T* a, const T* b, T* c,
                                                    dim_t block, dim_t a_size, dim_t b_size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::block_broadcast(DataTypeToEnum<T>::value, mps::BinaryOp::ADD, a, b, c, block, a_size, b_size);
      return;
    }
    if (block == 0 || a_size == 0)
      return;

    const dim_t iter_size = b_size / block;
    for (dim_t i = 0; i < iter_size; ++i) {
      const T a_i = a[i % a_size];
      const dim_t offset = i * block;
      for (dim_t j = 0; j < block; ++j)
        c[offset + j] = a_i + b[offset + j];
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::sub(const T* a, const T* b, T* c, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::binary(DataTypeToEnum<T>::value, mps::BinaryOp::SUB, a, b, c, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      c[i] = a[i] - b[i];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::max(T a, const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::scalar(DataTypeToEnum<T>::value, mps::BinaryOp::MAX, to_float(a), x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = (x[i] > a ? x[i] : a);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::max(const T* a, const T* b, T* c, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::binary(DataTypeToEnum<T>::value, mps::BinaryOp::MAX, a, b, c, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      c[i] = (a[i] > b[i] ? a[i] : b[i]);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::min(T a, const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::scalar(DataTypeToEnum<T>::value, mps::BinaryOp::MIN, to_float(a), x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = (x[i] < a ? x[i] : a);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::min(const T* a, const T* b, T* c, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::binary(DataTypeToEnum<T>::value, mps::BinaryOp::MIN, a, b, c, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      c[i] = (a[i] < b[i] ? a[i] : b[i]);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::mul(T a, const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::scalar(DataTypeToEnum<T>::value, mps::BinaryOp::MUL, to_float(a), x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = x[i] * a;
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::mul(const T* a, const T* b, T* c, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::binary(DataTypeToEnum<T>::value, mps::BinaryOp::MUL, a, b, c, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      c[i] = a[i] * b[i];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::batch_broadcast(DataTypeToEnum<T>::value, mps::BinaryOp::MUL, a, b, c, a_size, b_size);
      return;
    }
    for (dim_t i = 0; i < b_size; ++i)
      c[i] = a[i % a_size] * b[i];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::penalize_previous_tokens(T* scores,
                                                         const T* previous_scores,
                                                         const int32_t* previous_ids,
                                                         T penalty,
                                                         dim_t batch_size,
                                                         dim_t length,
                                                         dim_t vocabulary_size) {
    mps::synchronize("host repetition penalty");
    const float p = to_float(penalty);
    for (dim_t i = 0; i < batch_size * length; ++i) {
      const dim_t write_index = (i / length) * vocabulary_size + previous_ids[i];
      const float s = to_float(previous_scores[i]);
      const float out = (s < 0.f ? s * p : s / p);
      scores[write_index] = from_float<T>(out);
    }
  }

  template<>
  void primitives<Device::MPS>::prepare_length_mask(const int32_t* lengths,
                                                    dim_t batch_size,
                                                    dim_t num_heads,
                                                    dim_t num_queries,
                                                    bool mask_future,
                                                    bool multi_query,
                                                    int32_t* mask) {
    // This path is host-visible by design. The mask layout depends on whether
    // queries or heads are the merged dimension, so keep the CPU reference
    // mapping until every attention layout has a dedicated tested GPU kernel.
    mps::synchronize("host attention mask");
    for (dim_t b = 0; b < batch_size; ++b) {
      const int32_t length = lengths[b];
      int32_t* output = mask + b * num_heads * num_queries;
      for (dim_t i = 0; i < num_heads * num_queries; ++i) {
        if (mask_future) {
          const int32_t query = multi_query
                                ? static_cast<int32_t>(i / num_heads)
                                : static_cast<int32_t>(i % num_queries);
          output[i] = std::min<int32_t>(length, query + 1);
        } else {
          output[i] = length;
        }
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::transpose_2d(DataTypeToEnum<T>::value, a, dims[0], dims[1], b);
      return;
    }
    const dim_t rows = dims[0];
    const dim_t cols = dims[1];
    for (dim_t i = 0; i < rows; ++i)
      for (dim_t j = 0; j < cols; ++j)
        b[j * rows + i] = a[i * cols + j];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::transpose_3d(const T* a, const dim_t* dims,
                                            const dim_t* perm, T* b) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::transpose_3d(DataTypeToEnum<T>::value, a, dims, perm, b);
      return;
    }
    const dim_t d0 = dims[0], d1 = dims[1], d2 = dims[2];
    const dim_t a_stride[3] = {d1 * d2, d2, 1};
    const dim_t bd[3] = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
    const dim_t b_stride[3] = {bd[1] * bd[2], bd[2], 1};

    for (dim_t i0 = 0; i0 < bd[0]; ++i0) {
      for (dim_t i1 = 0; i1 < bd[1]; ++i1) {
        for (dim_t i2 = 0; i2 < bd[2]; ++i2) {
          dim_t idx_b = i0 * b_stride[0] + i1 * b_stride[1] + i2;
          dim_t src_idx[3] = {0, 0, 0};
          src_idx[perm[0]] = i0;
          src_idx[perm[1]] = i1;
          src_idx[perm[2]] = i2;
          dim_t idx_a = src_idx[0] * a_stride[0] + src_idx[1] * a_stride[1] + src_idx[2];
          b[idx_b] = a[idx_a];
        }
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::transpose_4d(const T* a, const dim_t* dims,
                                            const dim_t* perm, T* b) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::transpose_4d(DataTypeToEnum<T>::value, a, dims, perm, b);
      return;
    }
    const dim_t d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
    const dim_t a_stride[4] = {d1 * d2 * d3, d2 * d3, d3, 1};

    const dim_t bd[4] = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
    const dim_t b_stride[4] = {bd[1] * bd[2] * bd[3], bd[2] * bd[3], bd[3], 1};

    for (dim_t i0 = 0; i0 < bd[0]; ++i0) {
      for (dim_t i1 = 0; i1 < bd[1]; ++i1) {
        for (dim_t i2 = 0; i2 < bd[2]; ++i2) {
          for (dim_t i3 = 0; i3 < bd[3]; ++i3) {
            dim_t idx_b = i0 * b_stride[0] + i1 * b_stride[1] + i2 * b_stride[2] + i3;
            dim_t src_idx[4] = {0, 0, 0, 0};
            src_idx[perm[0]] = i0;
            src_idx[perm[1]] = i1;
            src_idx[perm[2]] = i2;
            src_idx[perm[3]] = i3;
            dim_t idx_a = src_idx[0] * a_stride[0] + src_idx[1] * a_stride[1] +
                          src_idx[2] * a_stride[2] + src_idx[3];
            b[idx_b] = a[idx_a];
          }
        }
      }
    }
  }

  // -------------------------
  // Float ops (CUDA-style: float/float16/bfloat16 supported)
  // -------------------------

  template<>
  template <typename T>
  float primitives<Device::MPS>::logsumexp(const T* x, dim_t size) {
    mps::synchronize("host logsumexp reduction");
    if (size <= 0)
      return -std::numeric_limits<float>::infinity();
    float maxv = -std::numeric_limits<float>::infinity();
    for (dim_t i = 0; i < size; ++i)
      maxv = std::max(maxv, to_float(x[i]));
    float sum = 0.f;
    for (dim_t i = 0; i < size; ++i)
      sum += std::exp(to_float(x[i]) - maxv);
    return std::log(sum) + maxv;
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::exp(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::EXP, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = from_float<T>(std::exp(to_float(x[i])));
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::log(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::LOG, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = from_float<T>(std::log(to_float(x[i])));
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::cos(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::COS, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = from_float<T>(std::cos(to_float(x[i])));
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::sin(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::SIN, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = from_float<T>(std::sin(to_float(x[i])));
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::tanh(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::TANH, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i)
      y[i] = from_float<T>(std::tanh(to_float(x[i])));
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::relu(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::RELU, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i) {
      const float v = to_float(x[i]);
      y[i] = from_float<T>(v > 0.f ? v : 0.f);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::gelu(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::GELU, x, y, size);
      return;
    }
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    for (dim_t i = 0; i < size; ++i) {
      const float v = to_float(x[i]);
      const float out = 0.5f * v * (1.f + std::erf(v * inv_sqrt2));
      y[i] = from_float<T>(out);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::gelu_tanh(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::GELU_TANH, x, y, size);
      return;
    }
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float k = 0.044715f;
    for (dim_t i = 0; i < size; ++i) {
      const float v = to_float(x[i]);
      const float u = sqrt_2_over_pi * (v + k * v * v * v);
      const float out = 0.5f * v * (1.f + std::tanh(u));
      y[i] = from_float<T>(out);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::gelu_sigmoid(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::GELU_SIGMOID, x, y, size);
      return;
    }
    constexpr float k = 1.702f;
    for (dim_t i = 0; i < size; ++i) {
      const float v = to_float(x[i]);
      const float s = 1.f / (1.f + std::exp(-k * v));
      y[i] = from_float<T>(v * s);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::sigmoid(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::SIGMOID, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i) {
      const float v = to_float(x[i]);
      const float out = 1.f / (1.f + std::exp(-v));
      y[i] = from_float<T>(out);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::swish(const T* x, T* y, dim_t size) {
    if constexpr (is_mps_float_type_v<T>) {
      mps::unary(DataTypeToEnum<T>::value, mps::UnaryOp::SWISH, x, y, size);
      return;
    }
    for (dim_t i = 0; i < size; ++i) {
      const float v = to_float(x[i]);
      const float s = 1.f / (1.f + std::exp(-v));
      y[i] = from_float<T>(v * s);
    }
  }

  // -------------------------
  // Quant helper / packing
  // -------------------------

  template<>
  void primitives<Device::MPS>::compute_u8_compensation(const int8_t* b,
                                                        bool transpose_b,
                                                        dim_t k,
                                                        dim_t n,
                                                        float alpha,
                                                        int32_t* compensation) {
    mps::synchronize("INT8 compensation");
    for (dim_t j = 0; j < n; ++j) {
      int32_t sum = 0;
      for (dim_t i = 0; i < k; ++i) {
        const int8_t v = transpose_b ? b[j * k + i] : b[i * n + j];
        sum += static_cast<int32_t>(v);
      }
      compensation[j] = static_cast<int32_t>(alpha * static_cast<float>(sum));
    }
  }

  template<>
  template <typename T>
  dim_t primitives<Device::MPS>::gemm_pack_b(const T*, const bool, const dim_t, const dim_t,
                                            const float, T*) {
    return 0;  // Not supported, return 0 like CUDA
  }

  // -------------------------
  // GEMM on MPS
  // -------------------------

  static void mps_gemm_float(bool transpose_a,
                            bool transpose_b,
                            dim_t m, dim_t n, dim_t k,
                            float alpha,
                            const float* a, dim_t lda,
                            const float* b, dim_t ldb,
                            float beta,
                            float* c, dim_t ldc) {
    mps::gemm(DataType::FLOAT32,
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
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool /*a_is_packed*/, bool /*b_is_packed*/,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const float* a, dim_t lda,
                                     const float* b, dim_t ldb,
                                     float beta,
                                     float* c, dim_t ldc,
                                     const float* /*a_shift_compensation*/) {
    mps_gemm_float(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool, bool,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const float16_t* a, dim_t lda,
                                     const float16_t* b, dim_t ldb,
                                     float beta,
                                     float16_t* c, dim_t ldc,
                                     const float16_t*) {
    mps::gemm(DataType::FLOAT16,
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
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool, bool,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const bfloat16_t* a, dim_t lda,
                                     const bfloat16_t* b, dim_t ldb,
                                     float beta,
                                     bfloat16_t* c, dim_t ldc,
                                     const bfloat16_t*) {
    (void)transpose_a;
    (void)transpose_b;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
    throw std::invalid_argument(
      "BF16 GEMM is not supported on MPS; load the model with compute_type=float16");
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool, bool,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const int8_t* a, dim_t lda,
                                     const int8_t* b, dim_t ldb,
                                     float beta,
                                     int32_t* c, dim_t ldc,
                                     const int32_t* a_shift_compensation) {
    (void)transpose_a;
    (void)transpose_b;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;
    (void)a_shift_compensation;
    // TODO: implement fused weight-only INT8/INT4 dequantization + GEMV/GEMM.
    throw std::invalid_argument(
      "INT8 GEMM is not supported on MPS; use float16 MPS or route the model to CPU");
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::MPS>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                   dim_t m, dim_t n, dim_t k,
                                                   float alpha,
                                                   const In* a, dim_t lda, dim_t stridea,
                                                   const In* b, dim_t ldb, dim_t strideb,
                                                   float beta,
                                                   Out* c, dim_t ldc, dim_t stridec,
                                                   dim_t batch_size) {
    if constexpr (std::is_same_v<In, Out> && std::is_same_v<In, float>) {
      mps::gemm_batch_strided(DataType::FLOAT32,
                              transpose_a,
                              transpose_b,
                              m,
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
    if constexpr (std::is_same_v<In, Out> && std::is_same_v<In, float16_t>) {
      mps::gemm_batch_strided(DataType::FLOAT16,
                              transpose_a,
                              transpose_b,
                              m,
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
    for (dim_t i = 0; i < batch_size; ++i) {
      primitives<Device::MPS>::gemm(false, false,
                                    transpose_a, transpose_b,
                                    m, n, k,
                                    alpha,
                                    a + i * stridea, lda,
                                    b + i * strideb, ldb,
                                    beta,
                                    c + i * stridec, ldc,
                                    static_cast<const Out*>(nullptr));
    }
  }

  // -------------------------
  // Cross-device copies (unified memory => memcpy)
  // -------------------------

  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::MPS>::copy(const T* x, T* y, dim_t size) {
    const size_t bytes = static_cast<size_t>(size) * sizeof(T);
    if (mps::buffer_in_use(y, bytes))
      mps::synchronize("CPU to MPS copy", bytes);
    std::memcpy(y, x, bytes);
    mps::record_copy_bytes(bytes);
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::MPS, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    const size_t bytes = static_cast<size_t>(size) * sizeof(T);
    if (mps::buffer_in_use(x, bytes))
      mps::synchronize("MPS to CPU copy", bytes);
    std::memcpy(y, x, bytes);
    mps::record_copy_bytes(bytes);
  }

  // -------------------------
  // Integer GEMM Stubs (not implemented on MPS)
  // -------------------------

  #define MPS_STUB_GEMM(T) \
  template<> template<> \
  void primitives<Device::MPS>::gemm(bool, bool, bool, bool, dim_t, dim_t, dim_t, \
                                     float, const T*, dim_t, const T*, dim_t, \
                                     float, T*, dim_t, const T*) { \
    throw std::runtime_error("MPS integer GEMM <" #T "," #T "> not implemented"); \
  }

  MPS_STUB_GEMM(int8_t)
  MPS_STUB_GEMM(int16_t)
  MPS_STUB_GEMM(int32_t)

  // =========================================================================
  // EXPLICIT TEMPLATE INSTANTIATIONS (CRITICAL - avoids linker errors)
  // =========================================================================

#define MPS_DECLARE_IMPL(T) \
  template T primitives<Device::MPS>::at(const T*, dim_t); \
  template void primitives<Device::MPS>::fill(T*, T, dim_t); \
  template void primitives<Device::MPS>::strided_fill(T*, T, dim_t, dim_t); \
  template void primitives<Device::MPS>::indexed_fill(T*, T, const int32_t*, dim_t); \
  template void primitives<Device::MPS>::copy(const T*, T*, dim_t); \
  template T primitives<Device::MPS>::sum(const T*, dim_t); \
  template dim_t primitives<Device::MPS>::max_element(const T*, dim_t); \
  template T primitives<Device::MPS>::max(const T*, dim_t); \
  template T primitives<Device::MPS>::amax(const T*, dim_t); \
  template void primitives<Device::MPS>::add(T, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::add(const T*, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::add_batch_broadcast(const T*, const T*, T*, dim_t, dim_t); \
  template void primitives<Device::MPS>::add_depth_broadcast(const T*, const T*, T*, dim_t, dim_t); \
  template void primitives<Device::MPS>::add_block_broadcast(const T*, const T*, T*, dim_t, dim_t, dim_t); \
  template void primitives<Device::MPS>::sub(const T*, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::min(T, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::min(const T*, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::max(T, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::max(const T*, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::mul(T, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::mul(const T*, const T*, T*, dim_t); \
  template void primitives<Device::MPS>::mul_batch_broadcast(const T*, const T*, T*, dim_t, dim_t); \
  template void primitives<Device::MPS>::penalize_previous_tokens(T*, const T*, const int32_t*, T, dim_t, dim_t, dim_t); \
  template void primitives<Device::MPS>::transpose_2d(const T*, const dim_t*, T*); \
  template void primitives<Device::MPS>::transpose_3d(const T*, const dim_t*, const dim_t*, T*); \
  template void primitives<Device::MPS>::transpose_4d(const T*, const dim_t*, const dim_t*, T*); \
  template dim_t primitives<Device::MPS>::gemm_pack_b(const T*, bool, dim_t, dim_t, float, T*); \
  template void primitives<Device::MPS>::gemm_batch_strided<T, T>(bool, bool, dim_t, dim_t, dim_t, float, const T*, dim_t, dim_t, const T*, dim_t, dim_t, float, T*, dim_t, dim_t, dim_t); \
  template void cross_device_primitives<Device::CPU, Device::MPS>::copy<T>(const T*, T*, dim_t); \
  template void cross_device_primitives<Device::MPS, Device::CPU>::copy<T>(const T*, T*, dim_t);

  DECLARE_ALL_TYPES(MPS_DECLARE_IMPL)

  // Float-math operations (float/float16/bfloat16 only)
#define MPS_DECLARE_FLOAT_IMPL(T) \
  template void primitives<Device::MPS>::relu(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::gelu(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::gelu_tanh(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::gelu_sigmoid(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::sigmoid(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::swish(const T*, T*, dim_t); \
  template float primitives<Device::MPS>::logsumexp(const T*, dim_t); \
  template void primitives<Device::MPS>::sin(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::cos(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::tanh(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::exp(const T*, T*, dim_t); \
  template void primitives<Device::MPS>::log(const T*, T*, dim_t);

  MPS_DECLARE_FLOAT_IMPL(float)
  MPS_DECLARE_FLOAT_IMPL(float16_t)
  MPS_DECLARE_FLOAT_IMPL(bfloat16_t)

}  // namespace ctranslate2
#endif  // __APPLE__
