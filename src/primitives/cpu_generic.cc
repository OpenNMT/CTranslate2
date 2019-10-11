#include "ctranslate2/primitives/primitives_decl.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#ifdef WITH_MKL
#  include "ctranslate2/primitives/cpu_mkl.h"
#endif

#include "ctranslate2/types.h"

namespace ctranslate2 {

  template <typename T1, typename T2, typename Function>
  void unary_transform(const T1* x, T2* y, size_t size, Function func) {
    std::transform(x, x + size, y, func);
  }

  template <typename T1, typename T2, typename Function>
  void binary_transform(const T1* a, const T1* b, T2* c, size_t size, Function func) {
    std::transform(a, a + size, b, c, func);
  }

  template<>
  void primitives<Device::CPU>::set_device(int) {
  }

  template<>
  int primitives<Device::CPU>::get_device() {
    return 0;
  }

#ifndef WITH_MKL
  template<>
  void* primitives<Device::CPU>::alloc_data(size_t size) {
    return malloc(size);
  }

  template<>
  void primitives<Device::CPU>::free_data(void* data) {
    free(data);
  }

  template<>
  void primitives<Device::CPU>::clear_cache() {
  }
#endif

  template<>
  template <typename T>
  T primitives<Device::CPU>::deref(const T* x, size_t index) {
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::fill(T* x, T a, size_t size) {
    std::fill_n(x, size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::strided_fill(T* x, T a, size_t inc_x, size_t size) {
    for (size_t i = 0, j = 0; i < size; i++, j += inc_x) {
      x[j] = a;
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::copy(const T* x, T* y, size_t size) {
    std::copy_n(x, size, y);
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::sum(const T* array, size_t size) {
    return std::accumulate(array, array + size, static_cast<T>(0));
  }

  template<>
  template <typename T>
  size_t primitives<Device::CPU>::max_element(const T* array, size_t size) {
    return std::distance(array, std::max_element(array, array + size));
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::max(const T* array, size_t size) {
    return *std::max_element(array, array + size);
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::amax(const T* array, size_t size) {
    return std::abs(*std::max_element(array, array + size,
                                      [](T a, T b){
                                        return std::abs(a) < std::abs(b);
                                      }));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [&a](T v) { return v + a; });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::plus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    size_t a_size, size_t b_size) {
    size_t iter_size = b_size / a_size;
    for (size_t i = 0; i < iter_size; ++i) {
      size_t offset = i * a_size;
      add(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                    size_t a_size, size_t b_size) {
    size_t iter_size = a_size;
    size_t depth = b_size / a_size;
    for (size_t i = 0; i < iter_size; ++i) {
      size_t offset = i * depth;
      add(a[i], b + offset, c + offset, depth);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sub(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::minus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [&a](T v) { return v * a; });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::multiplies<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    size_t a_size, size_t b_size) {
    size_t iter_size = b_size / a_size;
    for (size_t i = 0; i < iter_size; ++i) {
      size_t offset = i * a_size;
      mul(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::inv(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(1) / v; });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::quantize(const float* x, T* y, size_t size, float scale) {
    unary_transform(x, y, size, [&scale](float v) {
      return static_cast<T>(
        std::max(
          std::min(v * scale, static_cast<float>(std::numeric_limits<T>::max())),
          static_cast<float>(std::numeric_limits<T>::lowest())));
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::unquantize(const T* x, float* y, size_t size, float scale) {
    unary_transform(x, y, size, [&scale](T v) {
      return static_cast<float>(v) / scale;
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::unquantize_batch(const T* x, const float* scale, float* y,
                                                 size_t x_size, size_t scale_size) {
    size_t depth = x_size / scale_size;
    #pragma omp parallel for
    for (size_t i = 0; i < scale_size; ++i) {
      const auto offset = i * depth;
      unquantize(x + offset, y + offset, depth, scale[i]);
    }
  }

  template<>
  void primitives<Device::CPU>::quantize_batch(const float* x, float* scales, int8_t* qx,
                                               size_t batch_size, size_t depth) {
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
      const float* row = x + i * depth;
      int8_t* qrow = qx + i * depth;
      auto scale = static_cast<float>(std::numeric_limits<int8_t>::max()) / amax(row, depth);
      unary_transform(row, qrow, depth, [scale](float v) { return static_cast<int8_t>(v * scale); });
      scales[i] = scale;
    }
  }

  template<>
  void primitives<Device::CPU>::rescale_output(const int32_t* x,
                                               const float* input_scales,
                                               const float* weight_scales,
                                               float* y,
                                               size_t batch_size,
                                               size_t depth) {
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < depth; ++j) {
        const auto index = j + i * depth;
        y[index] = static_cast<float>(x[index]) / (input_scales[i] * weight_scales[j]);
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::relu(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) {
      return v > 0 ? v : static_cast<T>(0);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::pow(const T* x, T* y, T power, size_t size) {
    unary_transform(x, y, size, [&power](T v) {
      return static_cast<T>(std::pow(static_cast<float>(v), static_cast<float>(power)));
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::exp(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::exp(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::log(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::log(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::cos(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::cos(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sin(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::sin(v)); });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::tanh(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::tanh(v)); });
  }

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_2d(const DataType* a, const IndexType* dims, DataType* b) {
    #pragma omp parallel for
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
      }
    }
  }

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_3d(const DataType* a,
                                             const IndexType* dims,
                                             const IndexType* perm,
                                             DataType* b) {
    size_t perm_ind[3];
    for (size_t i = 0; i < 3; ++i)
      perm_ind[perm[i]] = i;
    size_t a_stride[3] = {dims[1] * dims[2], dims[2], 1};
    size_t b_stride[3] = {dims[perm[1]] * dims[perm[2]], dims[perm[2]], 1};
    size_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                               b_stride[perm_ind[2]]};

    #pragma omp parallel for
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        for (size_t i2 = 0; i2 < dims[2]; ++i2) {
          const size_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                              i2 * perm_b_stride[2]);
          const size_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                              i2 * a_stride[2]);
          b[b_i] = a[a_i];
        }
      }
    }
  }

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_4d(const DataType* a,
                                             const IndexType* dims,
                                             const IndexType* perm,
                                             DataType* b) {
    size_t perm_ind[4];
    for (size_t i = 0; i < 4; ++i)
      perm_ind[perm[i]] = i;
    size_t a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    size_t b_stride[4] = {dims[perm[1]] * dims[perm[2]] * dims[perm[3]],
                          dims[perm[2]] * dims[perm[3]], dims[perm[3]], 1};
    size_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                               b_stride[perm_ind[2]], b_stride[perm_ind[3]]};

    #pragma omp parallel for
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        for (size_t i2 = 0; i2 < dims[2]; ++i2) {
          for (size_t i3 = 0; i3 < dims[3]; ++i3) {
            const size_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                                i2 * perm_b_stride[2] + i3 * perm_b_stride[3]);
            const size_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                                i2 * a_stride[2] + i3 * a_stride[3]);
            b[b_i] = a[a_i];
          }
        }
      }
    }
  }


#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::CPU>::deref(const T* x, size_t index);             \
  template void                                                         \
  primitives<Device::CPU>::fill(T* x, T a, size_t size);                \
  template void                                                         \
  primitives<Device::CPU>::strided_fill(T* x, T a, size_t inc_x, size_t size); \
  template void                                                         \
  primitives<Device::CPU>::copy(const T* x, T* y, size_t size);         \
  template T                                                            \
  primitives<Device::CPU>::sum(const T* array, size_t size);            \
  template size_t                                                       \
  primitives<Device::CPU>::max_element(const T* array, size_t size);    \
  template T                                                            \
  primitives<Device::CPU>::max(const T* array, size_t size);            \
  template T                                                            \
  primitives<Device::CPU>::amax(const T* array, size_t size);           \
  template void                                                         \
  primitives<Device::CPU>::add(T a, const T* x, T* y, size_t size);     \
  template void                                                         \
  primitives<Device::CPU>::add(const T* a, const T* b, T* c, size_t size); \
  template void                                                         \
  primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c, \
                                               size_t a_size, size_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c, \
                                               size_t a_size, size_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::sub(const T* a, const T* b, T* c, size_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul(T a, const T* x, T* y, size_t size);     \
  template void                                                         \
  primitives<Device::CPU>::mul(const T* a, const T* b, T* c, size_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c, \
                                               size_t a_size, size_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::inv(const T* x, T* y, size_t size);          \
  template void                                                         \
  primitives<Device::CPU>::relu(const T* x, T* y, size_t size);         \
  template void                                                         \
  primitives<Device::CPU>::pow(const T* x, T* y, T power, size_t size); \
  template void                                                         \
  primitives<Device::CPU>::exp(const T* x, T* y, size_t size);          \
  template void                                                         \
  primitives<Device::CPU>::log(const T* x, T* y, size_t size);          \
  template void                                                         \
  primitives<Device::CPU>::cos(const T* x, T* y, size_t size);          \
  template void                                                         \
  primitives<Device::CPU>::sin(const T* x, T* y, size_t size);          \
  template void                                                         \
  primitives<Device::CPU>::tanh(const T* x, T* y, size_t size);         \
  template void                                                         \
  primitives<Device::CPU>::transpose_2d(const T* a, const size_t* dims, T* b); \
  template void                                                         \
  primitives<Device::CPU>::transpose_3d(const T* a,                     \
                                        const size_t* dims,             \
                                        const size_t* perm,             \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_4d(const T* a,                     \
                                        const size_t* dims,             \
                                        const size_t* perm,             \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::unquantize_batch(const T* x,                 \
                                            const float* scale,         \
                                            float* y,                   \
                                            size_t x_size,              \
                                            size_t scale_size);         \
  template void                                                         \
  primitives<Device::CPU>::quantize(const float* x, T* y, size_t size, float scale); \
  template void                                                         \
  primitives<Device::CPU>::unquantize(const T* x, float* y, size_t size, float scale);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
