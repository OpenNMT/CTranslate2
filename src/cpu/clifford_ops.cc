#include "clifford_ops.h"

#include <cmath>
#include <cstdint>

namespace ctranslate2 {
  namespace cpu {
    namespace clifford {

      // -----------------------------------------------------------------------
      // Rotor::normalised
      // -----------------------------------------------------------------------
      Rotor Rotor::normalised() const {
        const float norm = std::sqrt(s*s + b12*b12 + b13*b13 + b23*b23);
        if (norm < 1e-9f)
          return {};  // fall back to identity
        const float inv = 1.f / norm;
        return {s*inv, b12*inv, b13*inv, b23*inv};
      }

      // -----------------------------------------------------------------------
      // Forward sandwich: y = R · v · R̃
      //
      // Derivation (Cl(3,0) geometric product, explicit component expansion):
      //
      // Step 1 — T = R · v  where v = a·e1 + b·e2 + c·e3
      //   T[e1]   =  s·a + p12·b + p13·c
      //   T[e2]   =  s·b - p12·a + p23·c
      //   T[e3]   =  s·c - p13·a - p23·b
      //   T[e123] = p12·c - p13·b + p23·a
      //
      // Step 2 — y = T · R̃  where R̃ = s - p12·e12 - p13·e13 - p23·e23
      //   y[e1] =  s·T1 + p12·T2 + p13·T3 + p23·T123
      //   y[e2] = -p12·T1 + s·T2 + p23·T3 - p13·T123
      //   y[e3] = -p13·T1 - p23·T2 + s·T3 + p12·T123
      //
      // (verified against quaternion rotation for 90° test cases)
      // -----------------------------------------------------------------------
      void rotor_sandwich(const Rotor& R,
                          float a, float b, float c,
                          float& y0, float& y1, float& y2) {
        const float s   = R.s;
        const float p12 = R.b12;
        const float p13 = R.b13;
        const float p23 = R.b23;

        // Step 1: T = R · v
        const float T1   =  s*a + p12*b + p13*c;
        const float T2   =  s*b - p12*a + p23*c;
        const float T3   =  s*c - p13*a - p23*b;
        const float T123 = p12*c - p13*b + p23*a;

        // Step 2: y = T · R̃
        y0 =  s*T1 + p12*T2 + p13*T3 + p23*T123;
        y1 = -p12*T1 + s*T2 + p23*T3 - p13*T123;
        y2 = -p13*T1 - p23*T2 + s*T3 + p12*T123;
      }

      // -----------------------------------------------------------------------
      // Inverse sandwich: y = R̃ · v · R
      //
      // Step 1 — T = R̃ · v  (sign-flip on bivector components)
      //   T[e1]   =  s·a - p12·b - p13·c
      //   T[e2]   =  s·b + p12·a - p23·c
      //   T[e3]   =  s·c + p13·a + p23·b
      //   T[e123] = -p12·c + p13·b - p23·a
      //
      // Step 2 — y = T · R
      //   y[e1] =  s·T1 - p12·T2 - p13·T3 - p23·T123
      //   y[e2] =  p12·T1 + s·T2 + p13·T123 - p23·T3
      //   y[e3] = -p12·T123 + p13·T1 + p23·T2 + s·T3
      // -----------------------------------------------------------------------
      void rotor_sandwich_inv(const Rotor& R,
                              float a, float b, float c,
                              float& y0, float& y1, float& y2) {
        const float s   = R.s;
        const float p12 = R.b12;
        const float p13 = R.b13;
        const float p23 = R.b23;

        // Step 1: T = R̃ · v
        const float T1   =  s*a - p12*b - p13*c;
        const float T2   =  s*b + p12*a - p23*c;
        const float T3   =  s*c + p13*a + p23*b;
        const float T123 = -p12*c + p13*b - p23*a;

        // Step 2: y = T · R
        y0 =  s*T1 - p12*T2 - p13*T3 - p23*T123;
        y1 =  p12*T1 + s*T2 - p23*T3 + p13*T123;
        y2 =  p13*T1 + p23*T2 + s*T3 - p12*T123;
      }

      // -----------------------------------------------------------------------
      // random_rotor
      //
      // Deterministic "random" unit rotor from a simple hash of the indices.
      // When all indices are 0, returns the identity rotor.
      // -----------------------------------------------------------------------
      Rotor random_rotor(int group_idx, int head_idx, int layer_idx) {
        if (group_idx == 0 && head_idx == 0 && layer_idx == 0)
          return {};

        // Deterministic hash: mix indices into a 32-bit seed.
        uint32_t seed = static_cast<uint32_t>(group_idx)
                      ^ (static_cast<uint32_t>(head_idx) * 2654435761u)
                      ^ (static_cast<uint32_t>(layer_idx) * 1234567891u);
        // Xorshift32
        auto xorshift = [](uint32_t x) -> uint32_t {
          x ^= x << 13; x ^= x >> 17; x ^= x << 5;
          return x;
        };
        seed = xorshift(seed ? seed : 1u);
        const float u0 = static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);
        seed = xorshift(seed);
        const float u1 = static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);
        seed = xorshift(seed);
        const float u2 = static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);
        seed = xorshift(seed);
        const float u3 = static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);

        // Uniform distribution on S³ via Marsaglia (1972):
        const float sq1 = std::sqrt(1.f - u0);
        const float sq2 = std::sqrt(u0);
        const float a   = sq1 * std::sin(2.f * 3.14159265358979323846f * u1);
        const float b   = sq1 * std::cos(2.f * 3.14159265358979323846f * u1);
        const float c   = sq2 * std::sin(2.f * 3.14159265358979323846f * u2);
        const float d   = sq2 * std::cos(2.f * 3.14159265358979323846f * u2);
        (void)u3;

        // Map (a,b,c,d) → (s, b12, b13, b23)
        return Rotor{a, b, c, d};
      }

    } // clifford
  } // cpu
} // ctranslate2
