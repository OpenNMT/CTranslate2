#pragma once


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <altivec.h>

#include <sleefinline_vsx.h>


#include<xmmintrin.h>

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
        return vec_perm(vec_ld(0, ptr), vec_ld(16, ptr), vec_lvsl(0, ptr));
      }
      

      static inline value_type load(float value) {
	return vec_splats(value);
      }

      static inline value_type load(const float* ptr) {
	return vec_perm(vec_ld(0, ptr), vec_ld(16, ptr), vec_lvsl(0, ptr));
	
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
	
        return vec_ctf(vec_perm(vec_ld(0, ptr), vec_ld(16, ptr), vec_lvsl(0, ptr)),0);
      }

      static inline value_type load_and_convert(const int32_t* ptr,
                                                dim_t count,
                                                int32_t default_value = 0) {
        if (count == width) {
          return load_and_convert(ptr);
        } else {
          __ct2_align16__ int32_t tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
	  for (int i=0;i<width;i+=1){
            std::cout << "load_and_convert, before filltmp_values["<<i<<"]:"<<tmp_values[i]<<"\n";
	    std::cout << "load_and_convert, before ptr["<<i<<"]:"<<*(ptr+i)<<"\n";
	  }
          std::copy(ptr, ptr + count, tmp_values);
	  for (int i=0;i<width;i+=1){
	    std::cout << "load_and_convert,tmp_values["<<i<<"]:"<<tmp_values[i]<<"\n";
	    std::cout << "load_and_convert,ptr["<<i<<"]:"<<ptr[i]<<"\n";
	  }
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
	return Sleef_expf4_u10vsx(a);

	 
      }

      static inline value_type log(value_type a) {
         return Sleef_logf4_u35vsx(a);

      }
      static inline value_type sin(value_type a) {
        return Sleef_sinf4_u35vsx(a);
      }

      static inline value_type cos(value_type a) {
        return Sleef_cosf4_u35vsx(a);

      }

      static inline value_type tanh(value_type a) {
	return Sleef_tanhf4_u35vsx(a);

      }

      static inline value_type erf(value_type a) {
        return Sleef_erff4_u10vsx(a);
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
	/*float f=0;
	for (int i=0;i<width;i+=1) {
	  f+=a[i];
	}
	std::cout << "reduce_add ";
        return f;*/
        __m128 t1 = _mm_movehl_ps(a, a);
	__m128 t2 = _mm_add_ps(a, t1);
	__m128 t3 = _mm_shuffle_ps(t2, t2, 1);
	__m128 t4 = _mm_add_ss(t2, t3);
	return _mm_cvtss_f32(t4);
	
      }

      static inline float reduce_max4(value_type[4] a){

	return 0;
      }
      static inline float reduce_max(value_type a) {
	/*float max=0;
	for (int i=0;i<width;i+=1) {
          if (a[i]>max) {
	    max=a[i];
	  }
        }
	//std::cout << "reduce_max ";
        return max;*/
	float t0 = a[0] > a[1] ? a[0] : a[1];
        float t1 = a[2] > a[3] ? a[2] : a[3];
	return t0 > t1 ? t0 : t1;
      }
      static inline void output_vec(value_type v)
      {
	for(int a=0;a<4;a+=1)
	  std::cout<< " "<<a<<":"<<v[a];
	std::cout <<"\n";
	return;
      }
      static inline void output_vec_mask(mask_type v)
      {
        for(int a=0;a<4;a+=1)
          std::cout<< " "<<a<<":"<<(bool)v[a];
        std::cout <<"\n";
        return;
      }

    //template <CpuIsa ISA>
    static inline value_type vec_tanh(value_type a) {
      using VecType = Vec<float, TARGET_ISA>;

      // Implementation ported from Eigen:
      // https://gitlab.com/libeigen/eigen/-/blob/3.4.0/Eigen/src/Core/MathFunctionsImpl.h#L18-L76
      //std::cout << "Starting Power10::vec_tanh\n";
      const auto plus_clamp = VecType::load(7.90531110763549805f);
      //std::cout << " plus_clamp:";
      //VecType::output_vec(plus_clamp);
      const auto minus_clamp = VecType::load(-7.90531110763549805f);
      //std::cout << " minus_clamp:";
      //VecType::output_vec(minus_clamp);
      const auto tiny = VecType::load(0.0004f);
      //std::cout << "tiny:";
      //VecType::output_vec(tiny);
      const auto x = VecType::max(VecType::min(a, plus_clamp), minus_clamp);
      //std::cout << "x:";
      //VecType::output_vec(x);
      const auto tiny_mask = VecType::lt(VecType::abs(a), tiny);
      //std::cout << "tiny_mask:";
      //VecType::output_vec_mask(tiny_mask);


      const auto alpha_1 = VecType::load(4.89352455891786e-03f);
      const auto alpha_3 = VecType::load(6.37261928875436e-04f);
      const auto alpha_5 = VecType::load(1.48572235717979e-05f);
      const auto alpha_7 = VecType::load(5.12229709037114e-08f);
      const auto alpha_9 = VecType::load(-8.60467152213735e-11f);
      const auto alpha_11 = VecType::load(2.00018790482477e-13f);
      const auto alpha_13 = VecType::load(-2.76076847742355e-16f);

      const auto beta_0 = VecType::load(4.89352518554385e-03f);
      const auto beta_2 = VecType::load(2.26843463243900e-03f);
      const auto beta_4 = VecType::load(1.18534705686654e-04f);
      const auto beta_6 = VecType::load(1.19825839466702e-06f);

      const auto x2 = VecType::mul(x, x);
      //std::cout << "x2:";
      //output_vec(x2);
 

      auto p = VecType::mul_add(x2, alpha_13, alpha_11);
      //std::cout << "p:";
      //output_vec(p);

      p = VecType::mul_add(x2, p, alpha_9);
      p = VecType::mul_add(x2, p, alpha_7);
      p = VecType::mul_add(x2, p, alpha_5);
      p = VecType::mul_add(x2, p, alpha_3);
      p = VecType::mul_add(x2, p, alpha_1);
      p = VecType::mul(x, p);

      auto q = VecType::mul_add(x2, beta_6, beta_4);
      q = VecType::mul_add(x2, q, beta_2);
      q = VecType::mul_add(x2, q, beta_0);

      return VecType::select(tiny_mask, x, VecType::div(p, q));
    }
    };
  }
}
