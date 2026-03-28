#include "ctranslate2/ops/rotor_quant_kv.h"

#include <algorithm>
#include <stdexcept>

#include "ctranslate2/types.h"
#include "ctranslate2/ops/concat.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // -------------------------------------------------------------------------
    // compute_packed_stride
    // -------------------------------------------------------------------------
    dim_t RotorQuantKV::compute_packed_stride(dim_t d_head, int bits) {
      const dim_t codes_bytes = (d_head * bits + 7) / 8;
      return codes_bytes + 8;  // +8 for float32 min + float32 scale
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------
    RotorQuantKV::RotorQuantKV(dim_t d_head, const Config& cfg)
      : _d_head(d_head)
      , _cfg(cfg)
      , _packed_stride(compute_packed_stride(d_head, cfg.bits))
    {
      if (cfg.bits != 3 && cfg.bits != 4)
        throw std::invalid_argument("RotorQuantKV: bits must be 3 or 4");

      // Initialise rotors to identity for all groups.
      const dim_t n_groups = (d_head + 2) / 3;
      _rotors.resize(n_groups, {1.f, 0.f, 0.f, 0.f});
    }

    // -------------------------------------------------------------------------
    // encode (dispatch wrapper)
    // -------------------------------------------------------------------------
    void RotorQuantKV::encode(const StorageView& kv, StorageView& packed) const {
      PROFILE("RotorQuantKV::encode");

      const dim_t n_tokens = kv.size() / _d_head;
      const Device dev = kv.device();
      packed = StorageView({n_tokens, _packed_stride}, DataType::INT8, dev);
      if (dev == Device::CPU) {
        if (kv.dtype() == DataType::FLOAT32)
          encode_cpu(kv.data<float>(), packed.data<int8_t>(), n_tokens);
        else if (kv.dtype() == DataType::FLOAT16)
          encode_cpu(kv.data<float16_t>(), packed.data<int8_t>(), n_tokens);
        else
          throw std::invalid_argument("RotorQuantKV::encode: unsupported dtype "
                                      + dtype_name(kv.dtype()));
      }
#ifdef CT2_WITH_CUDA
      else if (dev == Device::CUDA) {
        if (kv.dtype() == DataType::FLOAT32)
          encode_cuda(kv.data<float>(), packed.data<int8_t>(), n_tokens);
        else if (kv.dtype() == DataType::FLOAT16)
          encode_cuda(kv.data<float16_t>(), packed.data<int8_t>(), n_tokens);
        else
          throw std::invalid_argument("RotorQuantKV::encode (CUDA): unsupported dtype "
                                      + dtype_name(kv.dtype()));
      }
#endif
      else {
        throw std::invalid_argument("RotorQuantKV::encode: unsupported device");
      }
    }

    // -------------------------------------------------------------------------
    // decode (dispatch wrapper)
    // -------------------------------------------------------------------------
    void RotorQuantKV::decode(const StorageView& packed,
                              StorageView& kv_out,
                              DataType out_dtype,
                              Device   out_device) const {
      PROFILE("RotorQuantKV::decode");

      const dim_t n_tokens = packed.size() / _packed_stride;
      kv_out = StorageView({n_tokens, _d_head}, out_dtype, out_device);

      if (out_device == Device::CPU) {
        if (out_dtype == DataType::FLOAT32)
          decode_cpu(packed.data<int8_t>(), kv_out.data<float>(), n_tokens);
        else if (out_dtype == DataType::FLOAT16)
          decode_cpu(packed.data<int8_t>(), kv_out.data<float16_t>(), n_tokens);
        else
          throw std::invalid_argument("RotorQuantKV::decode: unsupported dtype "
                                      + dtype_name(out_dtype));
      }
#ifdef CT2_WITH_CUDA
      else if (out_device == Device::CUDA) {
        if (out_dtype == DataType::FLOAT32)
          decode_cuda(packed.data<int8_t>(), kv_out.data<float>(), n_tokens);
        else if (out_dtype == DataType::FLOAT16)
          decode_cuda(packed.data<int8_t>(), kv_out.data<float16_t>(), n_tokens);
        else
          throw std::invalid_argument("RotorQuantKV::decode (CUDA): unsupported dtype "
                                      + dtype_name(out_dtype));
      }
#endif
      else {
        throw std::invalid_argument("RotorQuantKV::decode: unsupported device");
      }
    }

    // -------------------------------------------------------------------------
    // append
    // Encodes new_kv and appends to packed_cache along the time dimension.
    // packed_cache: [batch, heads, time_old, packed_stride] INT8 (or empty)
    // new_kv:       [batch, heads, time_new, d_head]        float
    // -------------------------------------------------------------------------
    void RotorQuantKV::append(const StorageView& new_kv,
                              StorageView& packed_cache) const {
      PROFILE("RotorQuantKV::append");

      if (packed_cache.empty()) {
        // First call: just encode.
        encode(new_kv, packed_cache);
        // Restore 4-dim shape [batch, heads, time, packed_stride] if new_kv is 4D.
        if (new_kv.rank() == 4) {
          packed_cache.reshape({new_kv.dim(0), new_kv.dim(1),
                                new_kv.dim(2), _packed_stride});
        }
        return;
      }

      // Encode new tokens.
      StorageView new_packed(DataType::INT8, new_kv.device());
      encode(new_kv, new_packed);
      if (new_kv.rank() == 4)
        new_packed.reshape({new_kv.dim(0), new_kv.dim(1),
                            new_kv.dim(2), _packed_stride});

      // Concat along time dimension (dim 2 for 4D, dim 0 for 2D).
      const dim_t cat_dim = (packed_cache.rank() == 4) ? 2 : 0;
      StorageView tmp(DataType::INT8, new_kv.device());
      const ops::Concat concat_op(cat_dim);
      tmp = std::move(packed_cache);
      concat_op({&tmp, &new_packed}, packed_cache);
    }

  } // ops
} // ctranslate2
