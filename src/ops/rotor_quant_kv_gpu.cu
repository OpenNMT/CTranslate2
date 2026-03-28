// GPU implementation of RotorQuantKV encode/decode.
//
// Dispatches to native CUDA kernels declared in cuda/rotor_quant_kernel.cu.

#include "ctranslate2/ops/rotor_quant_kv.h"

#ifdef CT2_WITH_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "ctranslate2/types.h"

// Forward-declare the kernel launchers from rotor_quant_kernel.cu.
namespace ctranslate2 {
  namespace cuda {
    template <typename T>
    void launch_rotor_encode(const T* kv, int8_t* packed,
                             int n_tokens, int d_head,
                             int packed_stride, int bits);
    template <typename T>
    void launch_rotor_decode(const int8_t* packed, T* kv,
                             int n_tokens, int d_head,
                             int packed_stride, int bits);
  }
}

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    void RotorQuantKV::encode_cuda(const T*    kv_ptr,
                                   int8_t*     packed_ptr,
                                   dim_t       n_tokens) const {
      cuda::launch_rotor_encode<T>(kv_ptr, packed_ptr,
                                   static_cast<int>(n_tokens),
                                   static_cast<int>(_d_head),
                                   static_cast<int>(_packed_stride),
                                   _cfg.bits);
    }

    template <typename T>
    void RotorQuantKV::decode_cuda(const int8_t* packed_ptr,
                                   T*            kv_ptr,
                                   dim_t         n_tokens) const {
      cuda::launch_rotor_decode<T>(packed_ptr, kv_ptr,
                                   static_cast<int>(n_tokens),
                                   static_cast<int>(_d_head),
                                   static_cast<int>(_packed_stride),
                                   _cfg.bits);
    }

    // Explicit instantiations.
    template void RotorQuantKV::encode_cuda<float>(
        const float*, int8_t*, dim_t) const;
    template void RotorQuantKV::encode_cuda<float16_t>(
        const float16_t*, int8_t*, dim_t) const;

    template void RotorQuantKV::decode_cuda<float>(
        const int8_t*, float*, dim_t) const;
    template void RotorQuantKV::decode_cuda<float16_t>(
        const int8_t*, float16_t*, dim_t) const;

  } // ops
} // ctranslate2

#endif // CT2_WITH_CUDA
