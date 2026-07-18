#pragma once

#include <string>

#include "ctranslate2/types.h"

#ifdef CT2_WITH_RUY
#include <ruy/ruy.h>
#endif

namespace ctranslate2 {
  namespace cpu {

    enum class GemmBackend {
      NONE,
      MKL,
      DNNL,
      ACCELERATE,
      OPENBLAS,
      RUY,
    };

    std::string gemm_backend_to_str(GemmBackend gemm_backend);
    bool mayiuse_mkl();
    GemmBackend get_gemm_backend(ComputeType compute_type);
    bool has_gemm_backend(ComputeType compute_type);
    bool prefer_u8s8s32_gemm();
    bool pack_gemm_weights(ComputeType compute_type);
#ifdef CT2_WITH_RUY
    ruy::Context *get_ruy_context();
    // Destroy the calling thread's ruy::Context (joins ruy's thread pool). Must be
    // called from a normal execution context, not during thread exit — see backend.cc.
    void clear_ruy_context();
#endif

  }
}
