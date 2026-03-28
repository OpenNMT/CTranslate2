#pragma once

#include <cstdint>

namespace ctranslate2 {
  namespace cpu {
    namespace clifford {

      // Apply Clifford rotor sandwich to groups of 3 consecutive floats.
      // vec:     in/out buffer of length `d`
      // rotors:  [n_groups x 4] packed as (s, b12, b13, b23) per group
      // inverse: if true, apply the inverse sandwich
      void rotate_groups(float*       vec,
                         const float* rotors,
                         int64_t      d,
                         bool         inverse);

    } // clifford
  } // cpu
} // ctranslate2
