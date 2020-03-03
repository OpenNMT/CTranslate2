#include "ctranslate2/primitives/primitives.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

#ifdef WITH_MKL
#  include <mkl.h>
#endif

#include "ctranslate2/utils.h"

#include "./parallel.h"

#define ALIGNMENT 64

namespace ctranslate2 {

  // work_size is an estimation of the amount of work per index (for example,
  // 1 for a basic operator + - *, 2 for /, and 4 for exp, log, etc.).

  template <typename T1, typename T2, typename Function>
  void unary_transform(const T1* x,
                       T2* y,
                       dim_t size,
                       const Function& func,
                       dim_t work_size = 1) {
    (void)work_size;
    std::transform(x, x + size, y, func);
  }

  template <typename T1, typename T2, typename T3, typename Function>
  void binary_transform(const T1* a,
                        const T2* b,
                        T3* c,
                        dim_t size,
                        const Function& func,
                        dim_t work_size = 1) {
    (void)work_size;
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
  void* primitives<Device::CPU>::alloc_data(dim_t size) {
    return aligned_alloc(size, ALIGNMENT);
  }

  template<>
  void primitives<Device::CPU>::free_data(void* data) {
    aligned_free(data);
  }

  template<>
  void primitives<Device::CPU>::clear_cache() {
#ifdef WITH_MKL
    mkl_free_buffers();
#endif
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::deref(const T* x, dim_t index) {
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::fill(T* x, T a, dim_t size) {
    std::fill_n(x, size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    for (dim_t i = 0, j = 0; i < size; i++, j += inc_x) {
      x[j] = a;
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::copy(const T* x, T* y, dim_t size) {
    std::copy_n(x, size, y);
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::copy(const float* x, float* y, dim_t size) {
    cblas_scopy(size, x, 1 /* incx */, y, 1 /* incy */);
  }
#endif

  template<>
  template <typename T>
  T primitives<Device::CPU>::sum(const T* array, dim_t size) {
    return std::accumulate(array, array + size, static_cast<T>(0));
  }

  template<>
  template <typename T>
  dim_t primitives<Device::CPU>::max_element(const T* array, dim_t size) {
    return std::distance(array, std::max_element(array, array + size));
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::max(const T* array, dim_t size) {
    return *std::max_element(array, array + size);
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::amax(const T* array, dim_t size) {
    return std::abs(*std::max_element(array, array + size,
                                      [](T a, T b){
                                        return std::abs(a) < std::abs(b);
                                      }));
  }

#ifdef WITH_MKL
  template<>
  template<>
  float primitives<Device::CPU>::amax(const float* x, dim_t size) {
    return std::abs(x[cblas_isamax(size, x, /*incx=*/1)]);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(T a, const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, [&a](T v) { return v + a; });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, std::plus<T>());
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::add(const float* a, const float* b, float* c, dim_t size) {
    vsAdd(size, a, b, c);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / a_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < iter_size; ++i) {
      const dim_t offset = i * a_size;
      add(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = a_size;
    const dim_t depth = b_size / a_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < iter_size; ++i) {
      const dim_t offset = i * depth;
      add(a[i], b + offset, c + offset, depth);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sub(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, std::minus<T>());
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::sub(const float* a, const float* b, float* c, dim_t size) {
    vsSub(size, a, b, c);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, [&a](T v) { return v * a; });
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::mul(float a, const float* x, float* y, dim_t size) {
    cblas_saxpby(size, a, x, 1 /* incx */, 0 /* b */, y, 1 /* incy */);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, std::multiplies<T>());
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::mul(const float* a, const float* b, float* c, dim_t size) {
    vsMul(size, a, b, c);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / a_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < iter_size; ++i) {
      const dim_t offset = i * a_size;
      mul(a, b + offset, c + offset, a_size);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::inv(const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, [](T v) { return static_cast<T>(1) / v; }, /*work_size=*/2);
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::inv(const float* x, float* y, dim_t size) {
    vsInv(size, x, y);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::quantize(const float* x,
                                         T* y,
                                         dim_t size,
                                         float scale,
                                         float shift) {
    unary_transform(x, y, size,
                    [scale, shift](float v) {
                      return static_cast<T>(
                        std::max(std::min(v * scale + shift,
                                          static_cast<float>(std::numeric_limits<T>::max())),
                                 static_cast<float>(std::numeric_limits<T>::lowest())));
                    }, /*work_size=*/5);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::dequantize(const T* x,
                                           float* y,
                                           dim_t size,
                                           float scale,
                                           float shift) {
    unary_transform(x, y, size,
                    [scale, shift](T v) {
                      return (static_cast<float>(v) - shift) / scale;
                    }, /*work_size=*/4);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::dequantize_batch(const T* x, const float* scale, float* y,
                                                 dim_t x_size, dim_t scale_size, float shift) {
    const dim_t depth = x_size / scale_size;
    #pragma omp parallel for
    for (dim_t i = 0; i < scale_size; ++i) {
      const dim_t offset = i * depth;
      dequantize(x + offset, y + offset, depth, scale[i], shift);
    }
  }

  template<>
  void primitives<Device::CPU>::quantize_batch(const float* x,
                                               float* scales,
                                               int8_t* qx,
                                               dim_t batch_size,
                                               dim_t depth,
                                               float shift) {
    #pragma omp parallel for
    for (dim_t i = 0; i < batch_size; ++i) {
      const float* row = x + i * depth;
      int8_t* qrow = qx + i * depth;
      auto scale = static_cast<float>(std::numeric_limits<int8_t>::max()) / amax(row, depth);
      unary_transform(row, qrow, depth,
                      [scale, shift](float v) {
                        return static_cast<int8_t>(v * scale + shift);
                      });
      scales[i] = scale;
    }
  }

  template<>
  void primitives<Device::CPU>::rescale_output(const int32_t* x,
                                               const float* input_scales,
                                               const float* weight_scales,
                                               float* y,
                                               dim_t batch_size,
                                               dim_t depth) {
    #pragma omp parallel for
    for (dim_t i = 0; i < batch_size; ++i) {
      for (dim_t j = 0; j < depth; ++j) {
        const dim_t index = j + i * depth;
        y[index] = static_cast<float>(x[index]) / (input_scales[i] * weight_scales[j]);
      }
    }
  }

  template<>
  void primitives<Device::CPU>::relu(const float* x, float* y, dim_t size) {
    unary_transform(x, y, size, [](float v) { return std::max(v, static_cast<float>(0)); });
  }

  template<>
  void primitives<Device::CPU>::gelu(const float* x, float* y, dim_t size) {
    static const float pi = std::acos(-1.f);
    static const float scale = std::sqrt(2.f / pi);
    unary_transform(x, y, size,
                    [](float v) {
                      return 0.5f * v * (1.f + std::tanh(scale * (v + 0.044715f * std::pow(v, 3.f))));
                    }, /*work_size=*/14);
  }

  template<>
  void primitives<Device::CPU>::pow(const float* x, float *y, float power, dim_t size) {
#ifdef WITH_MKL
    vsPowx(size, x, power, y);
#else
    unary_transform(x, y, size, [power](float v) { return std::pow(v, power); }, /*work_size=*/4);
#endif
  }

  template<>
  void primitives<Device::CPU>::exp(const float* x, float* y, dim_t size) {
#ifdef WITH_MKL
    vmsExp(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    unary_transform(x, y, size, [](float v) { return std::exp(v); }, /*work_size=*/4);
#endif
  }

  template<>
  void primitives<Device::CPU>::log(const float* x, float* y, dim_t size) {
#ifdef WITH_MKL
    vmsLn(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
#else
    unary_transform(x, y, size, [](float v) { return std::log(v); }, /*work_size=*/4);
#endif
  }

  template<>
  void primitives<Device::CPU>::cos(const float* x, float* y, dim_t size) {
#ifdef WITH_MKL
    vsCos(size, x, y);
#else
    unary_transform(x, y, size, [](float v) { return std::cos(v); }, /*work_size=*/4);
#endif
  }

  template<>
  void primitives<Device::CPU>::sin(const float* x, float* y, dim_t size) {
#ifdef WITH_MKL
    vsSin(size, x, y);
#else
    unary_transform(x, y, size, [](float v) { return std::sin(v); }; /*work_size=*/4);
#endif
  }

  template<>
  void primitives<Device::CPU>::tanh(const float* x, float* y, dim_t size) {
#ifdef WITH_MKL
    vsTanh(size, x, y);
#else
    unary_transform(x, y, size, [](float v) { return std::tanh(v); }, /*work_size=*/4);
#endif
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    #pragma omp parallel for
    for (dim_t i0 = 0; i0 < dims[0]; ++i0) {
      for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
        b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
      }
    }
  }

#ifdef WITH_MKL
  template<>
  template<>
  void primitives<Device::CPU>::transpose_2d(const float* a, const dim_t* dims, float* b) {
    const dim_t rows = dims[0];
    const dim_t cols = dims[1];
    mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
  }
#endif

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_3d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    dim_t perm_ind[3];
    for (dim_t i = 0; i < 3; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[3] = {dims[1] * dims[2], dims[2], 1};
    const dim_t b_stride[3] = {dims[perm[1]] * dims[perm[2]], dims[perm[2]], 1};
    const dim_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]]};

    #pragma omp parallel for
    for (dim_t i0 = 0; i0 < dims[0]; ++i0) {
      for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
        for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
          const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                             i2 * perm_b_stride[2]);
          const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                             i2 * a_stride[2]);
          b[b_i] = a[a_i];
        }
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_4d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    dim_t perm_ind[4];
    for (dim_t i = 0; i < 4; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    const dim_t b_stride[4] = {dims[perm[1]] * dims[perm[2]] * dims[perm[3]],
                               dims[perm[2]] * dims[perm[3]], dims[perm[3]], 1};
    const dim_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]], b_stride[perm_ind[3]]};

    #pragma omp parallel for
    for (dim_t i0 = 0; i0 < dims[0]; ++i0) {
      for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
        for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
          for (dim_t i3 = 0; i3 < dims[3]; ++i3) {
            const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                               i2 * perm_b_stride[2] + i3 * perm_b_stride[3]);
            const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
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
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha, float beta,
                                     float* c,
                                     const float*) {
#ifdef WITH_MKL
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

    cblas_sgemm(CblasRowMajor,
                trans_a, trans_b,
                m, n, k,
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
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha, float beta,
                                     int32_t* c,
                                     const int32_t*) {
#ifdef WITH_MKL
    MKL_INT lda = transpose_a ? m : k;
    MKL_INT ldb = transpose_b ? k : n;
    MKL_INT ldc = n;

    CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;
    CBLAS_OFFSET offsetc = CblasFixOffset;

    MKL_INT16 oa = 0;
    MKL_INT16 ob = 0;
    MKL_INT32 oc = 0;

    cblas_gemm_s16s16s32(CblasRowMajor,
                         trans_a, trans_b,
                         offsetc, m, n, k,
                         alpha,
                         reinterpret_cast<const MKL_INT16*>(a), lda, oa,
                         reinterpret_cast<const MKL_INT16*>(b), ldb, ob,
                         beta,
                         reinterpret_cast<MKL_INT32*>(c), ldc, &oc);
#else
    throw std::runtime_error("INT16 GEMM not available for CPU");
#endif
  }

  static void shift_to_u8(const int8_t* x, uint8_t* ux, dim_t size) {
    unary_transform(x, ux, size, [](int8_t v) { return static_cast<uint8_t>(v + 128); });
  }

  template<>
  void primitives<Device::CPU>::compute_u8_compensation(const int8_t* b,
                                                        bool transpose_b,
                                                        dim_t k,
                                                        dim_t n,
                                                        float alpha,
                                                        int32_t* compensation) {
    #pragma omp parallel for
    for (dim_t i = 0; i < n; ++i) {
      int32_t val = 0;

      if (transpose_b) {
        const int8_t* row = b + i * k;
        val = std::accumulate(row, row + k, static_cast<int32_t>(0));
      } else {
        for (dim_t j = 0; j < k; ++j) {
          val += b[j * n + i];
        }
      }

      if (alpha != 1) {
        val = static_cast<int32_t>(static_cast<float>(val) * alpha * -128.0);
      } else {
        val *= -128;
      }

      compensation[i] = val;
    }
  }

  template<>
  bool primitives<Device::CPU>::prefer_u8s8s32_gemm() {
#ifdef WITH_MKL
    return true;
#else
    return false;
#endif
  }


  template<>
  template<>
  void primitives<Device::CPU>::gemm(const int8_t* a, const int8_t* b,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha, float beta,
                                     int32_t* c,
                                     const int32_t* a_shift_compensation) {
#ifdef WITH_MKL
    // We are implementing s8s8s32 GEMM with cblas_gemm_s8u8s32. In row major mode,
    // it expects a to be unsigned and b to be signed. So we need to shift a to the
    // uint8 domain and add a compensation term. For more details, see
    // https://intel.github.io/mkl-dnn/dev_guide_int8_computations.html

    const uint8_t* ua = nullptr;
    uint8_t* tmp_ua = nullptr;
    int32_t* tmp_a_shift_compensation = nullptr;

    if (a_shift_compensation) {
      // If the compensation term is passed as argument, we assume a is already shifted.
      ua = reinterpret_cast<const uint8_t*>(a);
    } else {
      const dim_t a_size = m * k;
      tmp_ua = static_cast<uint8_t*>(alloc_data(a_size));
      shift_to_u8(a, tmp_ua, a_size);
      ua = tmp_ua;

      tmp_a_shift_compensation = static_cast<int32_t*>(alloc_data(n * sizeof (int32_t)));
      compute_u8_compensation(b, transpose_b, k, n, alpha, tmp_a_shift_compensation);
      a_shift_compensation = tmp_a_shift_compensation;
    }

    const MKL_INT lda = transpose_a ? m : k;
    const MKL_INT ldb = transpose_b ? k : n;
    const MKL_INT ldc = n;

    cblas_gemm_s8u8s32(CblasRowMajor,
                       transpose_a ? CblasTrans : CblasNoTrans,
                       transpose_b ? CblasTrans : CblasNoTrans,
                       CblasRowOffset,
                       m, n, k,
                       alpha,
                       ua, lda, 0,
                       b, ldb, 0,
                       beta,
                       c, ldc, a_shift_compensation);

    if (tmp_ua)
      free_data(tmp_ua);
    if (tmp_a_shift_compensation)
      free_data(tmp_a_shift_compensation);
#else
    throw std::runtime_error("INT8 GEMM not available for CPU");
#endif
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm_batch(const float* a, const float* b,
                                           bool transpose_a, bool transpose_b,
                                           dim_t batch_size,
                                           dim_t m, dim_t n, dim_t k,
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

    auto ptr_array = static_cast<float**>(alloc_data(3 * batch_size * sizeof (float*)));
    auto a_array = const_cast<const float**>(ptr_array);
    auto b_array = const_cast<const float**>(ptr_array + batch_size);
    auto c_array = ptr_array + 2 * batch_size;
    for (MKL_INT i = 0; i < b_; ++i) {
      a_array[i] = a + (i * m_ * k_);
      b_array[i] = b + (i * k_ * n_);
      c_array[i] = c + (i * m_ * n_);
    }

    cblas_sgemm_batch(CblasRowMajor,
                      &trans_a, &trans_b,
                      &m_, &n_, &k_,
                      &alpha, a_array, &lda,
                      b_array, &ldb,
                      &beta, c_array, &ldc,
                      1 /* group_count */, &b_);

    free_data(ptr_array);
#else
    #pragma omp parallel for
    for (dim_t i = 0; i < batch_size; ++i) {
      const float* a_i = a + (i * m * k);
      const float* b_i = b + (i * k * n);
      float* c_i = c + (i * m * n);

      gemm(a_i, b_i, transpose_a, transpose_b, m, n, k, alpha, beta, c_i);
    }
#endif
  }


#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::CPU>::deref(const T* x, dim_t index);              \
  template void                                                         \
  primitives<Device::CPU>::fill(T* x, T a, dim_t size);                 \
  template void                                                         \
  primitives<Device::CPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::copy(const T* x, T* y, dim_t size);          \
  template T                                                            \
  primitives<Device::CPU>::sum(const T* array, dim_t size);             \
  template dim_t                                                        \
  primitives<Device::CPU>::max_element(const T* array, dim_t size);     \
  template T                                                            \
  primitives<Device::CPU>::max(const T* array, dim_t size);             \
  template T                                                            \
  primitives<Device::CPU>::amax(const T* array, dim_t size);            \
  template void                                                         \
  primitives<Device::CPU>::add(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::inv(const T* x, T* y, dim_t size);           \
  template void                                                         \
  primitives<Device::CPU>::transpose_2d(const T* a,                     \
                                        const dim_t* dims,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_3d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_4d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::dequantize_batch(const T* x,                 \
                                            const float* scale,         \
                                            float* y,                   \
                                            dim_t x_size,               \
                                            dim_t scale_size,           \
                                            float shift);               \
  template void                                                         \
  primitives<Device::CPU>::quantize(const float* x, T* y,               \
                                    dim_t size,                         \
                                    float scale, float shift);          \
  template void                                                         \
  primitives<Device::CPU>::dequantize(const T* x, float* y,             \
                                      dim_t size,                       \
                                      float scale, float shift);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}
