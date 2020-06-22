#include "ctranslate2/primitives/primitives.h"

#include <cmath>
#include <type_traits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <cub/util_allocator.cuh>

#include "../cuda/utils.h"

namespace ctranslate2 {

  template <typename T1, typename T2, typename UnaryFunction>
  void unary_transform(const T1* x, T2* y, dim_t size, const UnaryFunction& op) {
    THRUST_CALL(thrust::transform, x, x + size, y, op);
  }

  template <typename T1, typename T2, typename T3, typename BinaryFunction>
  void binary_transform(const T1* a, const T2* b, T3* c, dim_t size, const BinaryFunction& op) {
    THRUST_CALL(thrust::transform, a, a + size, b, c, op);
  }

  template <typename T1, typename T2, typename T3, typename BinaryFunction, typename IndexFunction>
  void binary_transform(T1 a, T2 b, T3 c, dim_t size,
                        const BinaryFunction& op, const IndexFunction& index_a) {
    auto index_it = thrust::make_transform_iterator(thrust::counting_iterator<dim_t>(0), index_a);
    auto a_it = thrust::make_permutation_iterator(a, index_it);
    THRUST_CALL(thrust::transform, a_it, a_it + size, b, c, op);
  }

  // perm_fun is a functor that takes the index in the permuted iterator and
  // return the index in the original iterator.
  template <typename T, typename PermFunction>
  void permute(const T* x, T* y, dim_t size, const PermFunction& perm_fun) {
    auto ind_it = thrust::counting_iterator<dim_t>(0);
    auto perm_ind_it = thrust::make_transform_iterator(ind_it, perm_fun);
    auto perm_it = thrust::make_permutation_iterator(x, perm_ind_it);
    THRUST_CALL(thrust::copy_n, perm_it, size, y);
  }

  template <typename T>
  struct repeat_vec : thrust::unary_function<T, T> {
    T _size;
    repeat_vec(T size)
      : _size(size) {
    }
    __host__ __device__
    T operator()(const T i) {
      return i % _size;
    }
  };

  template <typename T>
  struct repeat_vec_depth : thrust::unary_function<T, T> {
    T _size;
    repeat_vec_depth(T size)
      : _size(size) {
    }
    __host__ __device__
    T operator()(const T i) {
      return i / _size;
    }
  };


  static const cuda::CachingAllocatorConfig allocator_config = cuda::get_caching_allocator_config();
  static cub::CachingDeviceAllocator allocator(
    allocator_config.bin_growth,
    allocator_config.min_bin,
    allocator_config.max_bin,
    allocator_config.max_cached_bytes);

  template<>
  void primitives<Device::CUDA>::set_device(int index) {
    CUDA_CHECK(cudaSetDevice(index));
  }

  template<>
  int primitives<Device::CUDA>::get_device() {
    int index;
    CUDA_CHECK(cudaGetDevice(&index));
    return index;
  }

  template<>
  void* primitives<Device::CUDA>::alloc_data(dim_t size, int device_index) {
    if (device_index < 0)
      device_index = cub::CachingDeviceAllocator::INVALID_DEVICE_ORDINAL;
    void* data = nullptr;
    CUDA_CHECK(allocator.DeviceAllocate(device_index, &data, size, cuda::get_cuda_stream()));
    return data;
  }

  template<>
  void primitives<Device::CUDA>::free_data(void* data, int device_index) {
    CUDA_CHECK(allocator.DeviceFree(device_index, data));
  }

  template<>
  void primitives<Device::CUDA>::clear_cache() {
    CUDA_CHECK(allocator.FreeAllCached());
  }

  template<>
  template <typename T>
  T primitives<Device::CUDA>::deref(const T* x, dim_t index) {
    T val = T();
    cross_device_primitives<Device::CUDA, Device::CPU>::copy(x + index, &val, 1);
    return val;
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::fill(T* x, T a, dim_t size) {
    THRUST_CALL(thrust::fill_n, x, size, a);
  }
  template<>
  template <typename T>
  void primitives<Device::CUDA>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    auto it = thrust::make_permutation_iterator(
      x, thrust::make_transform_iterator(thrust::counting_iterator<dim_t>(0),
                                         thrust::placeholders::_1 * inc_x));
    THRUST_CALL(thrust::fill_n, it, size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::copy(const T* x, T* y, dim_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T),
                               cudaMemcpyDeviceToDevice, cuda::get_cuda_stream()));
  }

  template<>
  template <typename T>
  T primitives<Device::CUDA>::sum(const T* array, dim_t size) {
    return THRUST_CALL(thrust::reduce, array, array + size);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::CUDA>::max_element(const T* array, dim_t size) {
    const auto* max = THRUST_CALL(thrust::max_element, array, array + size);
    return static_cast<dim_t>(max - array);
  }

  template<>
  template <typename T>
  T primitives<Device::CUDA>::max(const T* array, dim_t size) {
    const auto* max = THRUST_CALL(thrust::max_element, array, array + size);
    return deref(max, 0);
  }

  template <typename T1, typename T2>
  struct greater_tuple {
    __device__ __host__
    thrust::tuple<T1, T2> operator()(const thrust::tuple<T1, T2>& a,
                                     const thrust::tuple<T1, T2>& b) const {
      if (a > b)
        return a;
      else
        return b;
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::row_max(const T* x,
                                         const dim_t rows,
                                         const dim_t cols,
                                         T* values,
                                         int32_t* indices) {
    auto keys_it = thrust::make_transform_iterator(thrust::counting_iterator<int32_t>(0),
                                                   repeat_vec_depth<int32_t>(cols));
    auto ids_it = thrust::make_transform_iterator(thrust::counting_iterator<int32_t>(0),
                                                  repeat_vec<int32_t>(cols));

    THRUST_CALL(thrust::reduce_by_key,
                keys_it, keys_it + (rows * cols),
                thrust::make_zip_iterator(thrust::make_tuple(x, ids_it)),
                thrust::make_discard_iterator(),
                thrust::make_zip_iterator(thrust::make_tuple(values, indices)),
                thrust::equal_to<int32_t>(),
                greater_tuple<T, int32_t>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(T a, const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, thrust::placeholders::_1 + a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, thrust::plus<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    binary_transform(a, b, c, b_size, thrust::plus<T>(), repeat_vec<dim_t>(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    binary_transform(a, b, c, b_size, thrust::plus<T>(), repeat_vec_depth<dim_t>(b_size / a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::sub(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, thrust::minus<T>());
  }

  template<typename T>
  struct min_func : public thrust::unary_function<T, T> {
    T a_;
    __host__ __device__
    min_func(T a):a_(a){}
    __host__ __device__
    T operator()(T x) {return x > a_ ? a_ : x;}
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::min(T a, const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, min_func<T>(a));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::min(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, thrust::minimum<T>());
  }

  template<typename T>
  struct max_func : public thrust::unary_function<T, T> {
    T a_;
    __host__ __device__
    max_func(T a):a_(a){}
    __host__ __device__
    T operator()(T x) {return x > a_ ? x : a_;}
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::max(T a, const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, max_func<T>(a));
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::max(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, thrust::maximum<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul(T a, const T* x, T* y, dim_t size) {
    unary_transform(x, y, size, thrust::placeholders::_1 * a);
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul(const T* a, const T* b, T* c, dim_t size) {
    binary_transform(a, b, c, size, thrust::multiplies<T>());
  }

  template<>
  template <typename T>
  void primitives<Device::CUDA>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    binary_transform(a, b, c, b_size, thrust::multiplies<T>(), repeat_vec<dim_t>(a_size));
  }

  struct absolute_maximum_func : public thrust::binary_function<float, float, float> {
    __host__ __device__
    float operator()(float a, float b) {
      return fmaxf(fabsf(a), fabsf(b));
    }
  };

  template <typename T>
  class quantize_func : public thrust::binary_function<float, float, T> {
  private:
    float _shift;
  public:
    quantize_func(float shift)
      : _shift(shift) {
    }
    __host__ __device__
    T operator()(float scale, float x) {
      return static_cast<T>(x * scale + _shift);
    }
  };

  template<>
  void primitives<Device::CUDA>::quantize_batch(const float* x,
                                                float* scales,
                                                int8_t* qx,
                                                dim_t batch_size,
                                                dim_t depth,
                                                float shift) {
    const dim_t size = batch_size * depth;

    // Assign 1 key per batch.
    auto keys_it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                   repeat_vec_depth<int>(depth));

    // scales = 127.0 / reduce_max(abs(x), axis=1)
    THRUST_CALL(thrust::reduce_by_key,
                keys_it, keys_it + size,
                x,
                thrust::make_discard_iterator(),
                thrust::make_transform_output_iterator(
                  scales, static_cast<float>(127) / thrust::placeholders::_1),
                thrust::equal_to<int>(),
                absolute_maximum_func());

    // qx = x * expand_dims(scales, 1)
    binary_transform(scales, x, qx, size,
                     quantize_func<int8_t>(shift),
                     repeat_vec_depth<dim_t>(depth));
  }

  template <typename T>
  class dequantize_func : public thrust::binary_function<float, T, float> {
  private:
    float _shift;
  public:
    dequantize_func(float shift)
      : _shift(shift) {
    }
    __device__
    float operator()(float scale, T x) {
      return __fdividef(static_cast<float>(x) - _shift, scale);
    }
  };

  template<>
  template<>
  void primitives<Device::CUDA>::dequantize_batch(const int8_t* x, const float* scale, float* y,
                                                  dim_t x_size, dim_t scale_size, float shift) {
    binary_transform(scale, x, y, x_size,
                     dequantize_func<int8_t>(shift),
                     repeat_vec_depth<dim_t>(x_size / scale_size));
  }

  struct rescale_func : public thrust::binary_function<int32_t, thrust::tuple<float, float>, float> {
    __device__
    float operator()(int32_t x, const thrust::tuple<float, float>& scales) {
      return __fdividef(__int2float_rn(x), (thrust::get<0>(scales) * thrust::get<1>(scales)));
    }
  };

  template <bool transpose_a, bool transpose_b>
  static void rescale_output_impl(const int32_t* c,
                                  const float* a_scales,
                                  const float* b_scales,
                                  float* y,
                                  dim_t batch_size,
                                  dim_t depth) {
#define EXPAND(scales, transpose)                                       \
    thrust::make_permutation_iterator(                                  \
      scales,                                                           \
      thrust::make_transform_iterator(                                  \
        thrust::counting_iterator<int>(0),                              \
        typename std::conditional<transpose, repeat_vec<int>, repeat_vec_depth<int>>::type(depth)))

    // y = c / (expand_dims(a_scales, trans_a ? 0 : 1) * expand_dims(b_scales, trans_b ? 0 : 1)
    auto a_scales_it = EXPAND(a_scales, transpose_a);
    auto b_scales_it = EXPAND(b_scales, transpose_b);
    auto scales_it = thrust::make_zip_iterator(thrust::make_tuple(a_scales_it, b_scales_it));
    const dim_t size = batch_size * depth;
    THRUST_CALL(thrust::transform,
                c, c + size,
                scales_it,
                y,
                rescale_func());

#undef EXPAND
  }

  template<>
  void primitives<Device::CUDA>::rescale_output(const int32_t* c,
                                                const float* a_scales,
                                                const float* b_scales,
                                                const bool transpose_a,
                                                const bool transpose_b,
                                                float* y,
                                                dim_t batch_size,
                                                dim_t depth) {
    if (transpose_a && transpose_b)
      rescale_output_impl<true, true>(c, a_scales, b_scales, y, batch_size, depth);
    else if (transpose_a)
      rescale_output_impl<true, false>(c, a_scales, b_scales, y, batch_size, depth);
    else if (transpose_b)
      rescale_output_impl<false, true>(c, a_scales, b_scales, y, batch_size, depth);
    else
      rescale_output_impl<false, false>(c, a_scales, b_scales, y, batch_size, depth);
  }

  struct relu_func : public thrust::unary_function<float, float> {
    __host__ __device__
    float operator()(float x) { return fmaxf(x, 0); }
  };

  template<>
  void primitives<Device::CUDA>::relu(const float* x, float* y, dim_t size) {
    unary_transform(x, y, size, relu_func());
  }

  struct gelu_func : public thrust::unary_function<float, float> {
    float _scale;
    gelu_func(float scale)
      : _scale(scale) {
    }
    __host__ __device__
    float operator()(float x) {
      return 0.5f * x * (1.f + tanhf(_scale * (x + 0.044715f * powf(x, 3.f))));
    }
  };

  template<>
  void primitives<Device::CUDA>::gelu(const float* x, float* y, dim_t size) {
    static const float pi = std::acos(-1.f);
    static const float scale = std::sqrt(2.f / pi);
    unary_transform(x, y, size, gelu_func(scale));
  }

  template <typename T>
  struct perm_indices_2d : public thrust::unary_function<T, T> {
    T _rows, _cols;
    perm_indices_2d(T rows, T cols)
      : _rows(rows)
      , _cols(cols) {
    }
    __host__ __device__
    T operator()(const T i) const {
      const T i0 = i / _rows;
      const T i1 = i % _rows;
      return i1 * _cols + i0;
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    permute(a, b, dims[0] * dims[1], perm_indices_2d<dim_t>(dims[0], dims[1]));
  }

  template <typename T>
  struct perm_indices_3d : public thrust::unary_function<T, T> {
    T _a_ps0, _a_ps1, _a_ps2; // Permuted strides of the original array.
    T _b_d0, _b_d1, _b_d2;    // Dimension of the permutated array.
    T _b_s0, _b_s1, _b_s2;    // Strides of the permutated array.
    perm_indices_3d(const T* dims, const T* perm) {
      const T a_stride[3] = {dims[1] * dims[2], dims[2], 1};
      _a_ps0 = a_stride[perm[0]];
      _a_ps1 = a_stride[perm[1]];
      _a_ps2 = a_stride[perm[2]];
      _b_d0 = dims[perm[0]];
      _b_d1 = dims[perm[1]];
      _b_d2 = dims[perm[2]];
      _b_s0 = _b_d1 * _b_d2;
      _b_s1 = _b_d2;
      _b_s2 = 1;
    }
    __host__ __device__
    T operator()(const T i) const {
      const T i0 = i / _b_s0;
      const T i1 = i / _b_s1 % _b_d1;
      const T i2 = i % _b_d2;
      return i0 * _a_ps0 + i1 * _a_ps1 + i2 * _a_ps2;
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_3d(const T* a,
                                              const dim_t* dims,
                                              const dim_t* perm,
                                              T* b) {
    permute(a, b, dims[0] * dims[1] * dims[2], perm_indices_3d<dim_t>(dims, perm));
  }

  template <typename T>
  struct perm_indices_4d : public thrust::unary_function<T, T> {
    T _a_ps0, _a_ps1, _a_ps2, _a_ps3; // Permuted strides of the original array.
    T _b_d0, _b_d1, _b_d2, _b_d3;    // Dimension of the permutated array.
    T _b_s0, _b_s1, _b_s2, _b_s3;    // Strides of the permutated array.
    perm_indices_4d(const T* dims, const T* perm) {
      const T a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
      _a_ps0 = a_stride[perm[0]];
      _a_ps1 = a_stride[perm[1]];
      _a_ps2 = a_stride[perm[2]];
      _a_ps3 = a_stride[perm[3]];
      _b_d0 = dims[perm[0]];
      _b_d1 = dims[perm[1]];
      _b_d2 = dims[perm[2]];
      _b_d3 = dims[perm[3]];
      _b_s0 = _b_d1 * _b_d2 * _b_d3;
      _b_s1 = _b_d2 * _b_d3;
      _b_s2 = _b_d3;
      _b_s3 = 1;
    }
    __host__ __device__
    T operator()(const T i) const {
      const T i0 = i / _b_s0;
      const T i1 = i / _b_s1 % _b_d1;
      const T i2 = i / _b_s2 % _b_d2;
      const T i3 = i % _b_d3;
      return i0 * _a_ps0 + i1 * _a_ps1 + i2 * _a_ps2 + i3 * _a_ps3;
    }
  };

  template<>
  template <typename T>
  void primitives<Device::CUDA>::transpose_4d(const T* a,
                                              const dim_t* dims,
                                              const dim_t* perm,
                                              T* b) {
    permute(a, b, dims[0] * dims[1] * dims[2] * dims[3], perm_indices_4d<dim_t>(dims, perm));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const float* a, const float* b,
                                      bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha, float beta,
                                      float* c,
                                      const float*) {
    // Memo: cuBLAS assumes column-major storage.

    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(cuda::get_cublas_handle(),
                             transb, transa,
                             n, m, k,
                             &alpha,
                             b, ldb,
                             a, lda,
                             &beta,
                             c, ldc));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm(const int8_t* a, const int8_t* b,
                                      bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha, float beta,
                                      int32_t* c,
                                      const int32_t*) {
    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int32_t alpha_i = alpha;
    int32_t beta_i = beta;

    // cuBLAS assumes column-major storage, so swap a and b accordingly.
    CUBLAS_CHECK(cublasGemmEx(cuda::get_cublas_handle(),
                              transb, transa,
                              n, m, k,
                              &alpha_i,
                              b, CUDA_R_8I, ldb,
                              a, CUDA_R_8I, lda,
                              &beta_i,
                              c, CUDA_R_32I, ldc,
                              CUDA_R_32I,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  template<>
  template<>
  void primitives<Device::CUDA>::gemm_batch(const float* a, const float* b,
                                            bool transpose_a, bool transpose_b,
                                            dim_t batch_size,
                                            dim_t m, dim_t n, dim_t k,
                                            float alpha, float beta,
                                            float* c) {
    // Memo: cuBLAS assumes column-major storage.

    const int lda = transpose_a ? m : k;
    const int ldb = transpose_b ? k : n;
    const int ldc = n;

    const long long int stridea = m * k;
    const long long int strideb = k * n;
    const long long int stridec = m * n;

    const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(cuda::get_cublas_handle(),
                                           transb, transa,
                                           n, m, k,
                                           &alpha,
                                           b, ldb, strideb,
                                           a, lda, stridea,
                                           &beta,
                                           c, ldc, stridec,
                                           batch_size));
  }

  struct exp_func : public thrust::unary_function<float, float> {
    __host__ __device__
    float operator()(float x) { return expf(x); }
  };

  template<>
  void primitives<Device::CUDA>::exp(const float* x, float* y, dim_t size) {
    unary_transform(x, y, size, exp_func());
  }

  struct log_func : public thrust::unary_function<float, float> {
    __host__ __device__
    float operator()(float x) { return logf(x); }
  };

  template<>
  void primitives<Device::CUDA>::log(const float* x, float* y, dim_t size) {
    unary_transform(x, y, size, log_func());
  }


  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::CUDA>::copy(const T* x, T* y, dim_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T), cudaMemcpyHostToDevice, cuda::get_cuda_stream()));
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CUDA, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    CUDA_CHECK(cudaMemcpyAsync(y, x, size * sizeof (T), cudaMemcpyDeviceToHost, cuda::get_cuda_stream()));
  }

#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::CUDA>::deref(const T* x, dim_t index);             \
  template void                                                         \
  primitives<Device::CUDA>::fill(T* x, T a, dim_t size);                \
  template void                                                         \
  primitives<Device::CUDA>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::copy<T>(const T* x, T* y, dim_t size);      \
  template T                                                            \
  primitives<Device::CUDA>::sum(const T* array, dim_t size);            \
  template dim_t                                                        \
  primitives<Device::CUDA>::max_element(const T* array, dim_t size);    \
  template T                                                            \
  primitives<Device::CUDA>::max(const T* array, dim_t size);            \
  template void                                                         \
  primitives<Device::CUDA>::row_max(const T* x,                         \
                                    const dim_t rows,                   \
                                    const dim_t cols,                   \
                                    T* values,                          \
                                    int32_t* indices);                  \
  template void                                                         \
  primitives<Device::CUDA>::add(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CUDA>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::add_batch_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CUDA>::add_depth_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CUDA>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::min(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CUDA>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::max(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CUDA>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::mul(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CUDA>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CUDA>::mul_batch_broadcast(const T* a, const T* b, \
                                                T* c, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CUDA>::transpose_2d(const T* a,                    \
                                         const dim_t* dims,             \
                                         T* b);                         \
  template void                                                         \
  primitives<Device::CUDA>::transpose_3d(const T* a,                    \
                                         const dim_t* dims,             \
                                         const dim_t* perm,             \
                                         T* b);                         \
  template void                                                         \
  primitives<Device::CUDA>::transpose_4d(const T* a,                    \
                                         const dim_t* dims,             \
                                         const dim_t* perm,             \
                                         T* b);                         \
  template void                                                         \
  cross_device_primitives<Device::CPU, Device::CUDA>::copy<T>(const T*, T*, dim_t); \
  template void                                                         \
  cross_device_primitives<Device::CUDA, Device::CPU>::copy<T>(const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

}

