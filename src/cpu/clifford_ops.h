#pragma once

// Clifford algebra Cl(3,0) rotor operations for RotorQuant KV-cache compression.
//
// A rotor R in Cl(3,0) lives in the even sub-algebra:
//   R = s·1 + b12·e12 + b13·e13 + b23·e23
// with norm s²+b12²+b13²+b23² = 1 (unit rotor).
//
// The sandwich product  R·v·R̃  rotates a grade-1 vector v ∈ span{e1,e2,e3}
// to another grade-1 vector.  R̃ is the reverse (conjugate) of R.
//
// Correspondence with quaternions (for reference):
//   s  ↔ w,  b23 ↔ i,  -b13 ↔ j,  b12 ↔ k

namespace ctranslate2 {
  namespace cpu {
    namespace clifford {

      // Clifford Cl(3,0) rotor: R = s + b12·e12 + b13·e13 + b23·e23
      // Identity rotor: s=1, b12=b13=b23=0
      struct Rotor {
        float s   = 1.f;
        float b12 = 0.f;
        float b13 = 0.f;
        float b23 = 0.f;

        Rotor() = default;
        Rotor(float s_, float b12_, float b13_, float b23_)
          : s(s_), b12(b12_), b13(b13_), b23(b23_) {}

        // Return normalised version (unit rotor).
        Rotor normalised() const;

        // Return the reverse/conjugate: R̃ = s - b12·e12 - b13·e13 - b23·e23
        Rotor reversed() const { return {s, -b12, -b13, -b23}; }
      };

      // Forward sandwich: y = R · [a,b,c]ᵀ · R̃
      // (rotation of the grade-1 vector a·e1+b·e2+c·e3)
      void rotor_sandwich(const Rotor& R,
                          float a, float b, float c,
                          float& y0, float& y1, float& y2);

      // Inverse sandwich: y = R̃ · [a,b,c]ᵀ · R
      // Undoes the forward rotation.
      void rotor_sandwich_inv(const Rotor& R,
                              float a, float b, float c,
                              float& y0, float& y1, float& y2);

      // Generate a random unit rotor seeded by (group_idx, head_idx, layer_idx).
      // Returns identity when all indices are 0.
      Rotor random_rotor(int group_idx, int head_idx, int layer_idx = 0);

    } // clifford
  } // cpu
} // ctranslate2
