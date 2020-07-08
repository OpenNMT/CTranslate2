#pragma once

#include <thrust/device_vector.h>

#include "ctranslate2/types.h"

#include "utils.h"

namespace ctranslate2 {
  namespace cuda {

    template <typename T>
    struct DeviceType {
      using type = T;
    };

    // float16_t (a.k.a. half_float::half) can't be used in device code so there is a bit
    // of template work to cast values and pointers to __half.
    template<>
    struct DeviceType<float16_t> {
      using type = __half;
    };

    template <typename T>
    using device_type = typename DeviceType<T>::type;

    template <typename T>
    inline const device_type<T>* cast(const T* x) {
      return reinterpret_cast<const device_type<T>*>(x);
    }

    template <typename T>
    inline device_type<T>* cast(T* x) {
      return reinterpret_cast<device_type<T>*>(x);
    }

    template <typename T1, typename T2, typename UnaryFunction>
    inline void unary_transform(const T1* x, T2* y, dim_t size, const UnaryFunction& op) {
      THRUST_CALL(thrust::transform, cast(x), cast(x) + size, cast(y), op);
    }

    template <typename T1, typename T2, typename T3, typename BinaryFunction>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 dim_t size,
                                 const BinaryFunction& op) {
      THRUST_CALL(thrust::transform, cast(a), cast(a) + size, cast(b), cast(c), op);
    }

    template <typename T1, typename T2, typename T3, typename BinaryFunction, typename IndexFunction>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 dim_t size,
                                 const BinaryFunction& op,
                                 const IndexFunction& index_a) {
      auto index_it = thrust::make_transform_iterator(thrust::counting_iterator<dim_t>(0), index_a);
      auto a_it = thrust::make_permutation_iterator(cast(a), index_it);
      THRUST_CALL(thrust::transform, a_it, a_it + size, cast(b), cast(c), op);
    }

    // perm_fun is a functor that takes the index in the permuted iterator and
    // return the index in the original iterator.
    template <typename T, typename PermFunction>
    inline void permute(const T* x, T* y, dim_t size, const PermFunction& perm_fun) {
      auto ind_it = thrust::counting_iterator<dim_t>(0);
      auto perm_ind_it = thrust::make_transform_iterator(ind_it, perm_fun);
      auto perm_it = thrust::make_permutation_iterator(cast(x), perm_ind_it);
      THRUST_CALL(thrust::copy, perm_it, perm_it + size, cast(y));
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


  }
}
