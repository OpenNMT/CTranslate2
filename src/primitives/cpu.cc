#include "ctranslate2/primitives/primitives.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef WITH_MKL
#  include <mkl.h>
#endif

#ifdef WITH_MKLDNN
#  include <mkldnn.hpp>
#endif

#include "ctranslate2/types.h"

#define ALIGNMENT 64

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

  template<>
  void* primitives<Device::CPU>::alloc_data(size_t size) {
#ifdef WITH_MKL
    return mkl_malloc(size, ALIGNMENT);
#else
    return malloc(size);
#endif
  }

  template<>
  void primitives<Device::CPU>::free_data(void* data) {
#ifdef WITH_MKL
    mkl_free(data);
#else
    free(data);
#endif
  }

  template<>
  void primitives<Device::CPU>::clear_cache() {
#ifdef WITH_MKL
    mkl_free_buffers();
#endif
  }

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

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::copy(const float* x, float* y, size_t size) {
    cblas_scopy(size, x, 1 /* incx */, y, 1 /* incy */);
  }
#endif

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

#ifdef WITH_MKL
  template<>
  template<>
  float primitives<Device::CPU>::amax(const float* x, size_t size) {
    return std::abs(x[cblas_isamax(size, x, /*incx=*/1)]);
  }
#endif

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

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::add(const float* a, const float* b, float* c, size_t size) {
    vsAdd(size, a, b, c);
  }
#endif

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

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::sub(const float* a, const float* b, float* c, size_t size) {
    vsSub(size, a, b, c);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [&a](T v) { return v * a; });
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::mul(float a, float* y, size_t size) {
    cblas_sscal(size, a, y, 1 /* incx */);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, size_t size) {
    binary_transform(a, b, c, size, std::multiplies<T>());
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::mul(const float* a, const float* b, float* c, size_t size) {
    vsMul(size, a, b, c);
  }
#endif

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

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::inv(const float* x, float* y, size_t size) {
    vsInv(size, x, y);
  }
#endif

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
    for (long long i = 0; i < scale_size; ++i) {
      const auto offset = i * depth;
      unquantize(x + offset, y + offset, depth, scale[i]);
    }
  }

  template<>
  void primitives<Device::CPU>::quantize_batch(const float* x, float* scales, int8_t* qx,
                                               size_t batch_size, size_t depth) {
    #pragma omp parallel for
    for (long long i = 0; i < batch_size; ++i) {
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
    for (long long i = 0; i < batch_size; ++i) {
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

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::pow(const float* x, float *y, float power, size_t size) {
    vsPowx(size, x, power, y);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::exp(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::exp(v)); });
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::exp(const float* x, float* y, size_t size) {
    vmsExp(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::log(const T* x, T* y, size_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(std::log(v)); });
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::log(const float* x, float* y, size_t size) {
    vmsLn(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
  }
#endif

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

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::tanh(const float* x, float* y, size_t size) {
    vsTanh(size, x, y);
  }
#endif

  template<>
  template <typename DataType, typename IndexType>
  void primitives<Device::CPU>::transpose_2d(const DataType* a, const IndexType* dims, DataType* b) {
    #pragma omp parallel for
    for (long long i0 = 0; i0 < dims[0]; ++i0) {
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
      }
    }
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::transpose_2d(const float* a, const size_t* dims, float* b) {
    auto rows = dims[0];
    auto cols = dims[1];
    mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
  }
#endif

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
    for (long long i0 = 0; i0 < dims[0]; ++i0) {
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
    for (long long i0 = 0; i0 < dims[0]; ++i0) {
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


  template<>
  template<>
  void primitives<Device::CPU>::gemm(const float* a, const float* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     float* c) {
#ifdef WITH_MKL
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    cblas_sgemm(CblasRowMajor,
                trans_a, trans_b,
                m_, n_, k_,
                alpha, a, lda,
                b, ldb,
                beta, c, ldc);
#else
    throw std::runtime_error("SGEMM not available for CPU");
#endif
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int16_t* a, const int16_t* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     int32_t* c) {
#ifdef WITH_MKL
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;
    CBLAS_OFFSET offsetc = CblasFixOffset;

    MKL_INT16 oa = 0;
    MKL_INT16 ob = 0;
    MKL_INT32 oc = 0;

    cblas_gemm_s16s16s32(CblasRowMajor,
                         trans_a, trans_b,
                         offsetc, m_, n_, k_,
                         alpha,
                         reinterpret_cast<const MKL_INT16*>(a), lda, oa,
                         reinterpret_cast<const MKL_INT16*>(b), ldb, ob,
                         beta,
                         reinterpret_cast<MKL_INT32*>(c), ldc, &oc);
#else
    throw std::runtime_error("INT16 GEMM not available for CPU");
#endif
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int8_t* a, const int8_t* b,
                                     bool transpose_a, bool transpose_b,
                                     size_t m, size_t n, size_t k,
                                     float alpha, float beta,
                                     int32_t* c) {
#ifdef WITH_MKLDNN
    int lda = transpose_a ? m : k;
    int ldb = transpose_b ? k : n;
    int ldc = n;

    int m_ = m;
    int n_ = n;
    int k_ = k;

    const char* transa = transpose_a ? "T" : "N";
    const char* transb = transpose_b ? "T" : "N";
    const char* offsetc = "F";

    int8_t ao = 0;
    int8_t bo = 0;
    int32_t co = 0;

    // mkldnn assumes column-major storage, so swap a and b accordingly.
    mkldnn::error::wrap_c_api(
      mkldnn_gemm_s8s8s32(transb, transa, offsetc,
                          &n_, &m_, &k_,
                          &alpha,
                          b, &ldb, &bo,
                          a, &lda, &ao,
                          &beta,
                          c, &ldc, &co),
      "mkldnn_gemm_s8s8s32 returned with an error");
#else
    throw std::runtime_error("INT8 GEMM not available for CPU");
#endif
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm_batch(const float* a, const float* b,
                                           bool transpose_a, bool transpose_b,
                                           size_t batch_size,
                                           size_t m, size_t n, size_t k,
                                           float alpha, float beta,
                                           float* c) {
#ifdef WITH_MKL
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    MKL_INT b_ = batch_size;
    MKL_INT m_ = m;
    MKL_INT n_ = n;
    MKL_INT k_ = k;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    std::vector<const float*> a_array(batch_size);
    std::vector<const float*> b_array(batch_size);
    std::vector<float*> c_array(batch_size);
    for (MKL_INT i = 0; i < b_; ++i) {
      a_array[i] = a + (i * m_ * k_);
      b_array[i] = b + (i * k_ * n_);
      c_array[i] = c + (i * m_ * n_);
    }

    cblas_sgemm_batch(CblasRowMajor,
                      &trans_a, &trans_b,
                      &m_, &n_, &k_,
                      &alpha, a_array.data(), &lda,
                      b_array.data(), &ldb,
                      &beta, c_array.data(), &ldc,
                      1 /* group_count */, &b_);
#else
    for (size_t i = 0; i < batch_size; ++i) {
      const float* a_i = a + (i * m * k);
      const float* b_i = b + (i * k * n);
      float* c_i = c + (i * m * n);

      gemm(a_i, b_i, transpose_a, transpose_b, m, n, k, alpha, beta, c_i);
    }
#endif
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
