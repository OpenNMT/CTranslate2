#pragma once
#include <cuda_fp16.h>

namespace ctranslate2 {
  namespace ops {
    __device__ __forceinline__ int make_divisible(int c, int divisor){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
      assert(false);
#else
      return (c + divisor - 1) / divisor;
#endif
    }

    __inline__ __device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
    {
      uint4 result;

      uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
      uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

      // First, we extract the i4s and construct an intermediate fp16 number.
      static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
      static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
      static constexpr uint32_t TOP_MASK              = 0x00f000f0;
      static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

      // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
      // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
      // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
      // elt_67 to fp16 without having to shift them to the bottom bits before hand.

      // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
      // immediately before required.
      const uint32_t top_i4s = i4s >> 8;
      // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
      // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
      // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[2])
        : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
      // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
      asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

      // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
      // half2 ctor. In this case, I chose performance reliability over code readability.

      // This is the half2 {1032, 1032} represented as an integer.
      // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
      // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
      static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
      // This is the half2 {1 / 16, 1 / 16} represented as an integer.
      static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
      // This is the half2 {-72, -72} represented as an integer.
      // static constexpr uint32_t NEG_72 = 0xd480d480;
      // Haotian: Let's use {-64, -64}.
      static constexpr uint32_t NEG_64 = 0xd400d400;

      // Finally, we construct the output numbers.
      // Convert elt_01
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
      // Convert elt_23
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
      // Convert elt_45
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
      // Convert elt_67
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

      return result;
    }
  }
}