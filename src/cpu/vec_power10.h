#pragma once


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <altivec.h>

#include <sleefinline_vsx3.h>

#include "vec.h"

#if defined(__GNUC__) || defined(__clang__)
#  define __ct2_align16__ __attribute__((aligned(16)))
#else
#  define __ct2_align16__
#endif

namespace ctranslate2 {
  namespace cpu {

    #define ALIGNMENT_VALUE     16u
    
    template<>
    struct Vec<float, TARGET_ISA> {

      using value_type = __ct2_align16__  __vector float;
      using mask_type = __ct2_align16__ __vector bool int;
      static constexpr dim_t width = 4;

      static inline value_type unaligned_load(const float* ptr){
	return (value_type){*ptr,*(ptr+1),*(ptr+2),*(ptr+3)};
      }
      

      static inline value_type load(float value) {
	return (value_type){value,value,value,value};
      }

      static inline value_type load(const float* ptr) {
	return (value_type){*ptr,*(ptr+1),*(ptr+2),*(ptr+3)};
      }

      static inline value_type load(const float* ptr, dim_t count, float default_value = float(0)) {
	if (count == width) {
          return load(ptr);
        } else {
          __ct2_align16__ float tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return load(tmp_values);
        }
      }

      static inline value_type load_and_convert(const int32_t* ptr) {
	return vec_ctf((vector signed int){*ptr,*(ptr+1),*(ptr+2),*(ptr+3)},0);
      }

      static inline value_type load_and_convert(const int32_t* ptr,
                                                dim_t count,
                                                int32_t default_value = 0) {
        if (count == width) {
          return load_and_convert(ptr);
        } else {
          __ct2_align16__ int32_t tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return load_and_convert(tmp_values);
        }
      }
      static inline void unaligned_store(value_type value, float* ptr) {
	vec_xst(value,0,ptr);
      }

      static inline void store(value_type value, float* ptr) {
	if (((uintptr_t)ptr % ALIGNMENT_VALUE) != 0)
        {
	  unaligned_store(value,ptr);
        } else
	  vec_st(value,0,ptr);
      }

      static inline void store(value_type value, float* ptr, dim_t count) {
        if (count == width) {
	  store(value,ptr);
        } else {
          __ct2_align16__ float tmp_values[width];
          store(value,tmp_values);
          std::copy(tmp_values, tmp_values + count, ptr);
        }
      }

      static inline value_type bit_and(value_type  a, value_type b) {
	return vec_and(a,b);
      }

      static inline value_type bit_xor(value_type a, value_type b) {
        return vec_xor(a,b);
      }

      static inline mask_type lt(value_type a, value_type b) {
        return vec_cmplt(a,b);
      }

      static inline value_type select(mask_type mask, value_type a, value_type b) {
	return vec_sel(a,b,mask);
      }

      static inline value_type abs(value_type a) {
        return vec_abs(a);
      }

      static inline value_type neg(value_type a) {
        return vec_neg(a);
      }

      static inline value_type rcp(value_type a) {
        return vec_re(a);
      }

      static inline value_type exp(value_type a) {
	return Sleef_expf4_u10vsx3(a);	 
      }

      static inline value_type log(value_type a) {
         return Sleef_logf4_u35vsx3(a);

      }
      static inline value_type sin(value_type a) {
        return Sleef_sinf4_u35vsx3(a);
      }

      static inline value_type cos(value_type a) {
        return Sleef_cosf4_u35vsx3(a);

      }

      static inline value_type tanh(value_type a) {
	return Sleef_tanhf4_u35vsx3(a);

      }

      static inline value_type erf(value_type a) {
        return Sleef_erff4_u10vsx3(a);
      }

      static inline value_type max(value_type a, value_type b) {
        return vec_max(a, b);
      }

      static inline value_type min(value_type a, value_type b) {
        return vec_min(a, b);
      }

      static inline value_type add(value_type a, value_type b) {
        return vec_add(a,b);
      }

      static inline value_type sub(value_type a, value_type b) {
        return vec_sub(a,b);
      }

      static inline value_type mul(value_type a, value_type b) {
        return vec_mul(a,b);
      }

      static inline value_type div(value_type a, value_type b) {
        return vec_div(a,b);
      }

      static inline value_type mul_add(value_type a, value_type b, value_type c) {
        
	return vec_madd(a,b,c);
      }

      static inline float reduce_add(value_type a) {


        unsigned long __element_selector_10 = 1 & 0x03;
        unsigned long __element_selector_32 = (1 >> 2) & 0x03;
        unsigned long __element_selector_54 = (1 >> 4) & 0x03;
        unsigned long __element_selector_76 = (1 >> 6) & 0x03;
        static const unsigned int __permute_selectors[4] =
          {
#ifdef __LITTLE_ENDIAN__
            0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C
#else
            0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F
#endif
          };
        __vector unsigned int __t;
        __t[0] = __permute_selectors[__element_selector_10];
        __t[1] = __permute_selectors[__element_selector_32];
        __t[2] = __permute_selectors[__element_selector_54] + 0x10101010;
        __t[3] = __permute_selectors[__element_selector_76] + 0x10101010;

        __vector unsigned long long v1 = vec_mergel((__vector unsigned long long)a,(__vector unsigned long long)a);
	value_type v2 = (value_type)a + (value_type)v1;
        value_type v3 = vec_perm (v2, v2,(__vector unsigned char) __t);
	return  v2[0]+v3[0];
      }
      
      static inline float reduce_max(value_type a) {
	float t0 = a[0] > a[1] ? a[0] : a[1];
        float t1 = a[2] > a[3] ? a[2] : a[3];
	return t0 > t1 ? t0 : t1;
      }

      static inline value_type round(value_type a) {
	return vec_round(a);
      }

      static inline void convert_and_store(value_type v, int8_t *a, dim_t count) {
	auto i32 = vec_cts(v,0);
	
	int8_t tmp[4];
	tmp[0]=i32[0];
	tmp[1]=i32[1];
	tmp[2]=i32[2];
	tmp[3]=i32[3];
	std::copy(tmp, tmp + count, a);
      }

      static inline void convert_and_store(value_type v, uint8_t *a, dim_t count) {
	auto u32 = vec_ctu(v,0);
        uint8_t tmp[4];
        tmp[0]=u32[0];
        tmp[1]=u32[1];
        tmp[2]=u32[2];
        tmp[3]=u32[3];
        std::copy(tmp, tmp + count, a);	
      }
    };
  }
}
