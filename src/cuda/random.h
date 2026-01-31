#pragma once

#ifdef CT2_USE_HIP
#include <hiprand/hiprand_kernel.h>
#define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#else
#include <curand_kernel.h>
#endif

namespace ctranslate2 {
  namespace cuda {

    curandStatePhilox4_32_10_t* get_curand_states(size_t num_states);

  }
}
