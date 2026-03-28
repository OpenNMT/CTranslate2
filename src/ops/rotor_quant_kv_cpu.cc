#include "ctranslate2/ops/rotor_quant_kv.h"

// CPU implementation of RotorQuantKV encode/decode.
//
// Encoding algorithm (per token vector v of d_head floats):
//  1. Apply Clifford rotor sandwich per 3-element group (identity in Phase 1).
//  2. Compute min_val = min(v), scale = max(v) - min(v).
//  3. Quantise each element to [0, 2^bits-1]:
//       code_i = round((v_i - min_val) / scale * (2^bits - 1))
//  4. Pack codes as 'bits'-bit integers into bytes (LSB first).
//  5. Write [packed_codes | min_val_bytes | scale_bytes] to the output buffer.
//
// Decoding reverses the above.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "ctranslate2/types.h"
#include "cpu/clifford_ops.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    // -----------------------------------------------------------------------
    // Bit-packing helpers
    //
    // We pack `n_vals` values, each occupying `bits` bits, LSB-first into
    // consecutive bytes starting at `out`.
    // -----------------------------------------------------------------------

    static inline void pack_bits(const uint8_t* codes,
                                 int8_t*        out,
                                 dim_t          n_vals,
                                 int            bits) {
      uint32_t buf = 0;
      int      fill = 0;
      int8_t*  p = out;
      for (dim_t i = 0; i < n_vals; ++i) {
        buf  |= (static_cast<uint32_t>(codes[i]) & ((1u << bits) - 1u)) << fill;
        fill += bits;
        while (fill >= 8) {
          *p++ = static_cast<int8_t>(buf & 0xFF);
          buf >>= 8;
          fill -= 8;
        }
      }
      if (fill > 0)
        *p = static_cast<int8_t>(buf & 0xFF);
    }

    static inline void unpack_bits(const int8_t* in,
                                   uint8_t*      codes,
                                   dim_t         n_vals,
                                   int           bits) {
      uint32_t           buf  = 0;
      int                avail = 0;
      const int8_t*      p    = in;
      const uint32_t     mask = (1u << bits) - 1u;
      for (dim_t i = 0; i < n_vals; ++i) {
        while (avail < bits) {
          buf  |= (static_cast<uint32_t>(static_cast<uint8_t>(*p++))) << avail;
          avail += 8;
        }
        codes[i] = static_cast<uint8_t>(buf & mask);
        buf   >>= bits;
        avail  -= bits;
      }
    }

    // -----------------------------------------------------------------------
    // encode_cpu
    // -----------------------------------------------------------------------
    template <typename T>
    void RotorQuantKV::encode_cpu(const T*    kv_ptr,
                                  int8_t*     packed_ptr,
                                  dim_t       n_tokens) const {
      const dim_t  d       = _d_head;
      const int    bits    = _cfg.bits;
      const dim_t  cbytes  = (d * bits + 7) / 8;  // codes bytes per token
      const float  qmax    = static_cast<float>((1 << bits) - 1);

      cpu::parallel_for(0, n_tokens, 1, [&](dim_t begin, dim_t end) {
        std::vector<float>   fvec(d);
        std::vector<uint8_t> codes(d);

        for (dim_t t = begin; t < end; ++t) {
          const T*  src  = kv_ptr   + t * d;
          int8_t*   dst  = packed_ptr + t * _packed_stride;

          // Step 1: convert to float and optionally apply rotor.
          // Phase 1: identity rotors → no rotation, just copy to float.
          for (dim_t i = 0; i < d; ++i)
            fvec[i] = static_cast<float>(src[i]);

          // (Phase 2 will call clifford::rotor_sandwich per group here.)

          // Step 2: find min / max.
          float mn = fvec[0], mx = fvec[0];
          for (dim_t i = 1; i < d; ++i) {
            if (fvec[i] < mn) mn = fvec[i];
            if (fvec[i] > mx) mx = fvec[i];
          }
          float scale = mx - mn;
          if (scale < 1e-9f) scale = 1e-9f;  // avoid division by zero

          // Step 3: quantise.
          for (dim_t i = 0; i < d; ++i) {
            float q = (fvec[i] - mn) / scale * qmax;
            q = std::max(0.f, std::min(qmax, std::round(q)));
            codes[i] = static_cast<uint8_t>(q);
          }

          // Step 4: pack bits.
          pack_bits(codes.data(), dst, d, bits);

          // Step 5: write metadata (min and scale as raw float bytes).
          std::memcpy(dst + cbytes,     &mn,    sizeof(float));
          std::memcpy(dst + cbytes + 4, &scale, sizeof(float));
        }
      });
    }

    // -----------------------------------------------------------------------
    // decode_cpu
    // -----------------------------------------------------------------------
    template <typename T>
    void RotorQuantKV::decode_cpu(const int8_t* packed_ptr,
                                  T*            kv_ptr,
                                  dim_t         n_tokens) const {
      const dim_t  d       = _d_head;
      const int    bits    = _cfg.bits;
      const dim_t  cbytes  = (d * bits + 7) / 8;
      const float  qmax    = static_cast<float>((1 << bits) - 1);

      cpu::parallel_for(0, n_tokens, 1, [&](dim_t begin, dim_t end) {
        std::vector<uint8_t> codes(d);
        std::vector<float>   fvec(d);

        for (dim_t t = begin; t < end; ++t) {
          const int8_t* src = packed_ptr + t * _packed_stride;
          T*            dst = kv_ptr     + t * d;

          // Unpack codes.
          unpack_bits(src, codes.data(), d, bits);

          // Read metadata.
          float mn, scale;
          std::memcpy(&mn,    src + cbytes,     sizeof(float));
          std::memcpy(&scale, src + cbytes + 4, sizeof(float));

          // Dequantise.
          for (dim_t i = 0; i < d; ++i)
            fvec[i] = static_cast<float>(codes[i]) / qmax * scale + mn;

          // (Phase 2: inverse rotor sandwich here.)

          // Convert back to T.
          for (dim_t i = 0; i < d; ++i)
            dst[i] = static_cast<T>(fvec[i]);
        }
      });
    }

    // -----------------------------------------------------------------------
    // Explicit instantiations for the types used by CTranslate2.
    // -----------------------------------------------------------------------
    template void RotorQuantKV::encode_cpu<float>(
        const float*, int8_t*, dim_t) const;
    template void RotorQuantKV::encode_cpu<float16_t>(
        const float16_t*, int8_t*, dim_t) const;

    template void RotorQuantKV::decode_cpu<float>(
        const int8_t*, float*, dim_t) const;
    template void RotorQuantKV::decode_cpu<float16_t>(
        const int8_t*, float16_t*, dim_t) const;

  } // ops
} // ctranslate2
