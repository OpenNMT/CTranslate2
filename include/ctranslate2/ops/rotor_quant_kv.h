#pragma once

// RotorQuant KV-cache compression operator.
//
// Compresses float16/float32 KV-cache tensors using Clifford Cl(3,0) rotor
// rotation followed by per-token min-max quantization to 3 or 4 bits.
//
// Phase 1 (this implementation):
//   - Identity rotors (no actual rotation applied, rotor sandwich = identity)
//   - Per-token symmetric min-max quantization
//   - No QJL residual correction
//   Phase 2 (future):
//   - Learned / random unit rotors per group
//   - Lloyd-Max codebook quantization
//   - QJL residual correction
//
// Packed buffer layout (per token vector of d_head elements):
//   [0 .. codes_bytes-1]       : quantized codes, LSB-packed at `bits` bits/dim
//   [codes_bytes .. +3]        : float32 min_val (raw bytes)
//   [codes_bytes+4 .. +7]      : float32 scale = max_val - min_val (raw bytes)
//
//   codes_bytes = (d_head * bits + 7) / 8
//   total packed_stride = codes_bytes + 8
//
// Cached buffers use DataType::INT8 as raw byte storage.
// The caller detects compression by checking cache.dtype() == DataType::INT8.

#include <memory>
#include <vector>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace ops {

    class RotorQuantKV {
    public:
      struct Config {
        int bits = 4;  // quantisation bits per dimension (3 or 4)
      };

      explicit RotorQuantKV(dim_t d_head, const Config& cfg = {});

      // Returns the packed stride (bytes per token) for the given d_head / config.
      static dim_t compute_packed_stride(dim_t d_head, int bits);

      dim_t packed_stride() const { return _packed_stride; }
      int   bits()          const { return _cfg.bits; }
      dim_t d_head()        const { return _d_head; }

      // Detect whether a cache StorageView is in compressed format.
      static bool is_packed(const StorageView& v) {
        return !v.empty() && v.dtype() == DataType::INT8;
      }

      // encode: kv [*, d_head] float → packed [*, packed_stride] INT8
      // Works for any leading dimensions (batch * heads * time flattened).
      void encode(const StorageView& kv, StorageView& packed) const;

      // decode: packed [*, packed_stride] INT8 → kv [*, d_head] same dtype/device as `kv_out`
      void decode(const StorageView& packed,
                  StorageView& kv_out,
                  DataType out_dtype,
                  Device   out_device) const;

      // append: encode `new_kv` and concat to existing packed cache along the time dimension.
      // packed_cache: [batch, heads, time_old, packed_stride] INT8
      // new_kv:       [batch, heads, 1,        d_head]       float
      // Result:       [batch, heads, time_old+1, packed_stride] INT8
      void append(const StorageView& new_kv,
                  StorageView& packed_cache) const;

    private:
      dim_t  _d_head;
      Config _cfg;
      dim_t  _packed_stride;

      // Per-group rotors: shape [n_groups, 4]  (s, b12, b13, b23)
      // Initialised to identity; future phases will populate from learned weights.
      std::vector<std::array<float, 4>> _rotors;  // size = n_groups

      template <typename T>
      void encode_cpu(const T* kv_ptr,
                      int8_t*  packed_ptr,
                      dim_t    n_tokens) const;

      template <typename T>
      void decode_cpu(const int8_t* packed_ptr,
                      T*            kv_ptr,
                      dim_t         n_tokens) const;

#ifdef CT2_WITH_CUDA
      template <typename T>
      void encode_cuda(const T* kv_ptr, int8_t* packed_ptr, dim_t n_tokens) const;

      template <typename T>
      void decode_cuda(const int8_t* packed_ptr, T* kv_ptr, dim_t n_tokens) const;
#endif
    };

  } // ops
} // ctranslate2
