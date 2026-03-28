// RotorQuant encode/decode CUDA kernels.
//
// Ported from RotorQuant csrc/rotor_fused_kernel.cu (MIT licence).
// PyTorch / ATen dependencies replaced with CTranslate2 CUDA helpers.
//
// Phase 1: per-token min-max quantisation, identity rotors.
// Phase 2 (TODO): per-group Clifford rotor sandwich, Lloyd-Max codebook.
//
// Kernel signatures follow CTranslate2 conventions:
//   - Raw device pointers (no torch::Tensor)
//   - Float16 via cuda::device_type<float16_t> = __half
//   - CUDA stream via cuda::get_cuda_stream()

#include "ctranslate2/ops/rotor_quant_kv.h"

#ifdef CT2_WITH_CUDA

#include <cuda_fp16.h>
#include <stdexcept>

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // -----------------------------------------------------------------------
    // Bit-pack helpers (device-side)
    // -----------------------------------------------------------------------

    // Read `bits`-bit code at position `idx` from packed byte array `src`.
    __device__ inline uint32_t read_code(const int8_t* src, int idx, int bits) {
      const int bit_pos  = idx * bits;
      const int byte_idx = bit_pos >> 3;
      const int shift    = bit_pos & 7;
      // Read 2 bytes to handle cross-byte codes (bits ≤ 8 so 2 bytes always sufficient).
      uint32_t val = static_cast<uint32_t>(static_cast<uint8_t>(src[byte_idx]));
      if (shift + bits > 8)
        val |= static_cast<uint32_t>(static_cast<uint8_t>(src[byte_idx + 1])) << 8;
      return (val >> shift) & ((1u << bits) - 1u);
    }

    // Write `bits`-bit value `v` at position `idx` in packed byte array `dst`.
    // Note: assumes the target bytes are zero-initialised before writing.
    __device__ inline void write_code(int8_t* dst, int idx, int bits, uint32_t v) {
      const int bit_pos  = idx * bits;
      const int byte_idx = bit_pos >> 3;
      const int shift    = bit_pos & 7;
      atomicOr(reinterpret_cast<unsigned int*>(dst + byte_idx),
               (v & ((1u << bits) - 1u)) << shift);
    }

    // -----------------------------------------------------------------------
    // rotor_encode_kernel
    //
    // Each thread processes one token (one row of shape [d_head]).
    // Grid: (n_tokens,), Block: (1,)
    // -----------------------------------------------------------------------
    template <typename T>
    __global__ void rotor_encode_kernel(const T*    __restrict__ kv,      // [n_tokens, d_head]
                                        int8_t*     __restrict__ packed,  // [n_tokens, packed_stride]
                                        int         d_head,
                                        int         packed_stride,
                                        int         bits,
                                        int         codes_bytes) {
      const int t = blockIdx.x * blockDim.x + threadIdx.x;
      if (t >= gridDim.x * blockDim.x) return;

      const T*  src = kv     + (long long)t * d_head;
      int8_t*   dst = packed + (long long)t * packed_stride;

      // ---- Step 1: find min / max ----
      float mn = __half2float(__float2half(static_cast<float>(src[0])));
      float mx = mn;
      for (int i = 1; i < d_head; ++i) {
        const float v = static_cast<float>(src[i]);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      float scale = mx - mn;
      if (scale < 1e-9f) scale = 1e-9f;

      // ---- Step 2: quantise + pack ----
      const float qmax  = static_cast<float>((1 << bits) - 1);
      const uint32_t mask = (1u << bits) - 1u;

      // Zero the code bytes.
      for (int b = 0; b < codes_bytes; ++b)
        dst[b] = 0;

      uint32_t buf  = 0u;
      int      fill = 0;
      int      byte_out = 0;
      for (int i = 0; i < d_head; ++i) {
        float q = (static_cast<float>(src[i]) - mn) / scale * qmax;
        if (q < 0.f) q = 0.f;
        if (q > qmax) q = qmax;
        const uint32_t code = static_cast<uint32_t>(__float2int_rn(q));
        buf  |= (code & mask) << fill;
        fill += bits;
        while (fill >= 8) {
          dst[byte_out++] = static_cast<int8_t>(buf & 0xFFu);
          buf >>= 8;
          fill -= 8;
        }
      }
      if (fill > 0)
        dst[byte_out] = static_cast<int8_t>(buf & 0xFFu);

      // ---- Step 3: write metadata ----
      memcpy(dst + codes_bytes,     &mn,    sizeof(float));
      memcpy(dst + codes_bytes + 4, &scale, sizeof(float));
    }

    // -----------------------------------------------------------------------
    // rotor_decode_kernel
    // -----------------------------------------------------------------------
    template <typename T>
    __global__ void rotor_decode_kernel(const int8_t* __restrict__ packed,  // [n_tokens, packed_stride]
                                        T*            __restrict__ kv,      // [n_tokens, d_head]
                                        int           d_head,
                                        int           packed_stride,
                                        int           bits,
                                        int           codes_bytes) {
      const int t = blockIdx.x * blockDim.x + threadIdx.x;
      if (t >= gridDim.x * blockDim.x) return;

      const int8_t* src = packed + (long long)t * packed_stride;
      T*            dst = kv     + (long long)t * d_head;

      // Read metadata.
      float mn, scale;
      memcpy(&mn,    src + codes_bytes,     sizeof(float));
      memcpy(&scale, src + codes_bytes + 4, sizeof(float));

      // Unpack codes and dequantise.
      const float  qmax = static_cast<float>((1 << bits) - 1);
      const uint32_t mask = (1u << bits) - 1u;

      uint32_t      buf   = 0u;
      int           avail = 0;
      const int8_t* p     = src;
      for (int i = 0; i < d_head; ++i) {
        while (avail < bits) {
          buf   |= static_cast<uint32_t>(static_cast<uint8_t>(*p++)) << avail;
          avail += 8;
        }
        const uint32_t code = buf & mask;
        buf   >>= bits;
        avail  -= bits;
        const float val = static_cast<float>(code) / qmax * scale + mn;
        dst[i] = static_cast<T>(val);
      }
    }

    // -----------------------------------------------------------------------
    // Launch wrappers
    // -----------------------------------------------------------------------
    template <typename T>
    void launch_rotor_encode(const T*    kv,
                             int8_t*     packed,
                             int         n_tokens,
                             int         d_head,
                             int         packed_stride,
                             int         bits) {
      if (n_tokens == 0) return;
      const int  codes_bytes = (d_head * bits + 7) / 8;
      const int  threads     = 128;
      const int  blocks      = (n_tokens + threads - 1) / threads;
      rotor_encode_kernel<cuda::device_type<T>>
          <<<blocks, threads, 0, get_cuda_stream()>>>(
              cuda::device_cast(kv), packed,
              d_head, packed_stride, bits, codes_bytes);
    }

    template <typename T>
    void launch_rotor_decode(const int8_t* packed,
                             T*            kv,
                             int           n_tokens,
                             int           d_head,
                             int           packed_stride,
                             int           bits) {
      if (n_tokens == 0) return;
      const int  codes_bytes = (d_head * bits + 7) / 8;
      const int  threads     = 128;
      const int  blocks      = (n_tokens + threads - 1) / threads;
      rotor_decode_kernel<cuda::device_type<T>>
          <<<blocks, threads, 0, get_cuda_stream()>>>(
              packed, cuda::device_cast(kv),
              d_head, packed_stride, bits, codes_bytes);
    }

    // Explicit instantiations.
    template void launch_rotor_encode<float>(
        const float*, int8_t*, int, int, int, int);
    template void launch_rotor_encode<float16_t>(
        const float16_t*, int8_t*, int, int, int, int);

    template void launch_rotor_decode<float>(
        const int8_t*, float*, int, int, int, int);
    template void launch_rotor_decode<float16_t>(
        const int8_t*, float16_t*, int, int, int, int);

  } // cuda
} // ctranslate2

// -----------------------------------------------------------------------
// Hook rotor_quant_kv_gpu.cu encode_cuda / decode_cuda into the kernels.
// The Phase-1 GPU stub in rotor_quant_kv_gpu.cu round-trips through CPU.
// This file provides the real CUDA path but is wired in Phase 2 only
// (see WITH_ROTOR_QUANT_CUDA define below).
// -----------------------------------------------------------------------

#endif // CT2_WITH_CUDA
