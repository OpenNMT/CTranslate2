#pragma once

#include <riscv_vector.h>
#include "vec.h"

namespace ctranslate2 {
  namespace cpu {

    template<>
    struct Vec<float, CpuIsa::RVV> {
      using value_type = vfloat32m1_t;
      using mask_type = vbool32_t;
      static constexpr dim_t width = 4; 
      static inline value_type load(float value) {
        return __riscv_vfmv_v_f_f32m1(value, width);
      }

      static inline value_type load(const float* ptr) {
        return __riscv_vle32_v_f32m1(ptr, width);
      }

      static inline value_type load(const float* ptr, dim_t count, float default_value = 0) {
        if (count == width) {
          return __riscv_vle32_v_f32m1(ptr, width);
        } else {
          float tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return __riscv_vle32_v_f32m1(tmp_values, width);
        }
      }

      static inline value_type load_and_convert(const int32_t* ptr) {
        return __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(ptr, width), width);
      }

      static inline value_type load_and_convert(const int32_t* ptr, dim_t count, int32_t default_value = 0) {
        if (count == width) {
          return load_and_convert(ptr);
        } else {
          int32_t tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return load_and_convert(tmp_values);
        }
      }

      static inline void store(value_type value, float* ptr) {
        __riscv_vse32_v_f32m1(ptr, value, width);
      }

      static inline void store(value_type value, float* ptr, dim_t count) {
        if (count == width) {
          __riscv_vse32_v_f32m1(ptr, value, width);
        } else {
          float tmp_values[width];
          __riscv_vse32_v_f32m1(tmp_values, value, width);
          std::copy(tmp_values, tmp_values + count, ptr);
        }
      }

      static inline value_type bit_and(value_type a, value_type b) {
        return  __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vand_vv_u32m1( __riscv_vreinterpret_v_f32m1_u32m1(a),  __riscv_vreinterpret_v_f32m1_u32m1(b), width)); 
      }

      static inline value_type bit_xor(value_type a, value_type b) {
        return  __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vxor_vv_u32m1( __riscv_vreinterpret_v_f32m1_u32m1(a),  __riscv_vreinterpret_v_f32m1_u32m1(b), width)); 
      }

      static inline mask_type lt(value_type a, value_type b) {
        return __riscv_vmflt_vv_f32m1_b32(a, b, width);
      }

      static inline value_type select(mask_type mask, value_type a, value_type b) {
        return __riscv_vmerge_vvm_f32m1( a, b,mask, width);
      }

      static inline value_type abs(value_type a) {
        return __riscv_vfabs_v_f32m1(a, width);
      }

      static inline value_type neg(value_type a) {
        return __riscv_vfneg_v_f32m1(a, width);
      }

      static inline value_type rcp(value_type a) {
        return __riscv_vfrdiv_vf_f32m1(a, 1.f, width);
      }

      static inline value_type exp(value_type a) {
        // Need to implement exp using RVV intrinsics
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::exp(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      static inline value_type log(value_type a) {
        // Need to implement log using RVV intrinsics
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::log(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      static inline value_type sin(value_type a) {
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::sin(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      static inline value_type cos(value_type a) {
        // Need to implement cos using RVV intrinsics
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::cos(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      static inline value_type tanh(value_type a) {
        // Need to implement tanh using RVV intrinsics
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::tanh(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      static inline value_type erf(value_type a) {
        // Need to implement erf using RVV intrinsics
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::erf(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      static inline value_type max(value_type a, value_type b) {
        return __riscv_vfmax_vv_f32m1(a, b, width);
      }

      static inline value_type min(value_type a, value_type b) {
        return __riscv_vfmin_vv_f32m1(a, b, width);
      }

      static inline value_type add(value_type a, value_type b) {
        return __riscv_vfadd_vv_f32m1(a, b, width);
      }

      static inline value_type sub(value_type a, value_type b) {
        return __riscv_vfsub_vv_f32m1(a, b, width);
      }

      static inline value_type mul(value_type a, value_type b) {
        return __riscv_vfmul_vv_f32m1(a, b, width);
      }

      static inline value_type div(value_type a, value_type b) {
        return __riscv_vfdiv_vv_f32m1(a, b, width);
      }

      static inline value_type mul_add(value_type a, value_type b, value_type c) {
        return __riscv_vfmacc_vv_f32m1(c, a, b, width);
      }

      static inline float reduce_add(value_type a) {
        // 使用RVV reduce sum内在函数
        value_type result = __riscv_vfredusum_vs_f32m1_f32m1(a, a, width);
        return __riscv_vfmv_f_s_f32m1_f32(result);
      }

      static inline float reduce_max(value_type a) {
        // 使用RVV reduce max内在函数
        value_type result = __riscv_vfredmax_vs_f32m1_f32m1(a, a, width);
        return __riscv_vfmv_f_s_f32m1_f32(result);
      }

      static inline value_type round(value_type a) {
        // Need to implement erf using RVV intrinsics
      float temp_a[4];
      float temp_result[4];

      __riscv_vse32_v_f32m1(temp_a, a, width);

      for (size_t i = 0; i < width; ++i) {
        temp_result[i] = std::round(temp_a[i]);
      }

      return __riscv_vle32_v_f32m1(temp_result, width);
      }

      template<typename T>
      static void convert_and_store(value_type v, T* a, dim_t count) {
        auto i32 = __riscv_vfcvt_x_f_v_i32m1(v, width);
        int32_t tmp[width];
        __riscv_vse32_v_i32m1(tmp, i32, width);
        std::copy(tmp, tmp + count, a);
      }
    };

  }
}
