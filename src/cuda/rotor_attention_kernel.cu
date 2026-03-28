// Fused RotorQuant Attention CUDA kernel.
//
// Computes:  score(q, K_packed) = q · decode(K_packed)^T  in one pass,
// without fully materialising the decoded K matrix.
//
// Converted from the Triton implementation in RotorQuant
// turboquant/fused_attention.py (MIT licence).
//
// Two-term estimator (from TurboQuant / RotorQuant):
//   score_i = term1_i + term2_i
//   term1_i = <q_rotated, centroid(k_codes_i)>      (MSE reconstruction)
//   term2_i = residual_norm_i * sqrt(pi/2) / m
//              * sum_j(qjl_sign_ij * <q, s_j>)       (QJL correction)
//
// Phase 1 (this file): term1 only (no QJL), using min-max decoded keys.
// Phase 2 (TODO): add QJL term2, use Lloyd-Max codebook.

#include "ctranslate2/ops/rotor_quant_kv.h"

#ifdef CT2_WITH_CUDA

#include <cuda_fp16.h>
#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    // -----------------------------------------------------------------------
    // rotor_fused_attn_kernel
    //
    // Each block computes attention scores for one (batch_head) pair.
    // Threads in the block iterate over key tokens.
    //
    // Grid:  (batch * heads,)
    // Block: (BLOCK_T,)   where BLOCK_T tiles the key-sequence dimension.
    //
    // Output scores: [batch, heads, 1, seq_k]  (single query step)
    // -----------------------------------------------------------------------
    template <typename T, int BLOCK_T = 64>
    __global__ void rotor_fused_attn_kernel(
        const T*      __restrict__ q,           // [batch, heads, 1, d_head]
        const int8_t* __restrict__ k_packed,    // [batch, heads, seq_k, packed_stride]
        float*        __restrict__ scores,      // [batch, heads, 1, seq_k]
        int   d_head,
        int   seq_k,
        int   packed_stride,
        int   bits,
        int   codes_bytes,
        float scale_factor) {                   // 1 / sqrt(d_head)

      const int bh = blockIdx.x;               // flattened batch-head index
      const int t_base = threadIdx.x;

      // Pointer to this batch-head's query vector.
      const T*      q_bh = q        + (long long)bh * d_head;
      const int8_t* k_bh = k_packed + (long long)bh * seq_k * packed_stride;
      float*        s_bh = scores   + (long long)bh * seq_k;

      // Load query to registers (up to d_head elements, tiled by BLOCK_T).
      // For simplicity, load full d_head in each thread (works well when d_head ≤ 256).
      // A production implementation would use shared memory.

      for (int t = t_base; t < seq_k; t += BLOCK_T) {
        const int8_t* k_tok = k_bh + (long long)t * packed_stride;

        // Decode key token on-the-fly.
        float mn, scale_kv;
        memcpy(&mn,       k_tok + codes_bytes,     sizeof(float));
        memcpy(&scale_kv, k_tok + codes_bytes + 4, sizeof(float));

        const float  qmax = static_cast<float>((1 << bits) - 1);
        const uint32_t mask = (1u << bits) - 1u;

        // Dot product q · decode(k_tok).
        float dot = 0.f;
        uint32_t buf   = 0u;
        int      avail = 0;
        const int8_t* p = k_tok;

        for (int d = 0; d < d_head; ++d) {
          while (avail < bits) {
            buf   |= static_cast<uint32_t>(static_cast<uint8_t>(*p++)) << avail;
            avail += 8;
          }
          const uint32_t code = buf & mask;
          buf   >>= bits;
          avail  -= bits;
          const float k_val = static_cast<float>(code) / qmax * scale_kv + mn;
          dot += static_cast<float>(q_bh[d]) * k_val;
        }

        s_bh[t] = dot * scale_factor;
      }
    }

    // -----------------------------------------------------------------------
    // launch_rotor_fused_attn
    // -----------------------------------------------------------------------
    template <typename T>
    void launch_rotor_fused_attn(
        const T*      q,            // [batch, heads, 1, d_head]
        const int8_t* k_packed,     // [batch, heads, seq_k, packed_stride]
        float*        scores,       // [batch, heads, 1, seq_k]  (caller-allocated)
        int   batch,
        int   heads,
        int   seq_k,
        int   d_head,
        int   packed_stride,
        int   bits) {
      if (seq_k == 0) return;

      const int   codes_bytes  = (d_head * bits + 7) / 8;
      const float scale_factor = 1.f / sqrtf(static_cast<float>(d_head));
      const int   bh_count     = batch * heads;
      const int   block_t      = 64;

      rotor_fused_attn_kernel<cuda::device_type<T>, 64>
          <<<bh_count, block_t, 0, get_cuda_stream()>>>(
              cuda::device_cast(q), k_packed, scores,
              d_head, seq_k, packed_stride, bits, codes_bytes, scale_factor);
    }

    // Explicit instantiations.
    template void launch_rotor_fused_attn<float>(
        const float*, const int8_t*, float*, int, int, int, int, int, int);
    template void launch_rotor_fused_attn<float16_t>(
        const float16_t*, const int8_t*, float*, int, int, int, int, int, int);

  } // cuda
} // ctranslate2

#endif // CT2_WITH_CUDA
