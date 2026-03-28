// SIMD-optimised Clifford Cl(3,0) rotor operations.
//
// This file is compiled once per ISA (similar to kernels.cc) and provides
// vectorised encode/decode for groups of 3 dimensions.
//
// Phase 2 uses these to replace the scalar loop in rotor_quant_kv_cpu.cc.
//
// Currently provides the fallback scalar + portable optimised paths.
// AVX2 / NEON specialisations are added incrementally.

#include "clifford_ops.h"
#include "clifford_kernels.h"

#include <cmath>
#include <cstring>

namespace ctranslate2 {
  namespace cpu {
    namespace clifford {

      // -----------------------------------------------------------------------
      // rotate_groups
      //
      // Apply the rotor sandwich to each group of 3 consecutive floats in `vec`,
      // reading rotor parameters from `rotors` (n_groups × 4 floats: s,b12,b13,b23).
      // -----------------------------------------------------------------------
      void rotate_groups(float*       vec,
                         const float* rotors,  // [n_groups][4]
                         int64_t      d,
                         bool         inverse) {
        const int n_groups = static_cast<int>((d + 2) / 3);
        for (int g = 0; g < n_groups; ++g) {
          const float* R = rotors + g * 4;
          const Rotor  rotor{R[0], R[1], R[2], R[3]};

          const int base = g * 3;
          // Handle last (possibly partial) group.
          float a = vec[base];
          float b = (base + 1 < static_cast<int>(d)) ? vec[base + 1] : 0.f;
          float c = (base + 2 < static_cast<int>(d)) ? vec[base + 2] : 0.f;

          float y0, y1, y2;
          if (!inverse)
            rotor_sandwich(rotor, a, b, c, y0, y1, y2);
          else
            rotor_sandwich_inv(rotor, a, b, c, y0, y1, y2);

          vec[base] = y0;
          if (base + 1 < static_cast<int>(d)) vec[base + 1] = y1;
          if (base + 2 < static_cast<int>(d)) vec[base + 2] = y2;
        }
      }

    } // clifford
  } // cpu
} // ctranslate2
