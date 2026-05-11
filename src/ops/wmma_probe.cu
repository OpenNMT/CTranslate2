// -----------------------------------------------------------------------------
// WMMA probe kernel — verifies the wave32 RDNA3 fragment layout empirically.
//
// Strategy: feed a known A and B = identity into one 16x16x16 WMMA, then have
// every lane write its 8 fp32 accumulator elements to an output buffer indexed
// by lane and slot.  Comparing the result on the host to a reference GEMM
// reveals which (row, col) of C each (lane, slot) pair corresponds to.
//
// Only compiled for HIP gfx11* targets.  Plain HIP / Clang built-ins, no
// rocWMMA dependency.  Lives in a separate translation unit so it doesn't
// pollute the production Flash Attention build.
// -----------------------------------------------------------------------------
#ifdef CT2_USE_HIP

#include <hip/hip_runtime.h>

namespace ctranslate2 {
  namespace ops {
    namespace wmma_probe {

      using v16f16 = _Float16 __attribute__((ext_vector_type(16)));
      using v8f32  = float    __attribute__((ext_vector_type(8)));
      // __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 is a Clang-internal
      // __device__ built-in for gfx11.  No forward declaration needed.

      // ---------------------------------------------------------------------
      // Probe kernel.  One block, one wave (32 threads).
      //
      // Inputs:
      //   A[16][16] row-major fp16    -- caller fills with A[i][j] = i*16 + j
      //   B[16][16] col-major fp16    -- caller fills with identity (B[i][j] = i==j)
      //
      // Output:
      //   out[lane*8 + slot] = the FP32 accumulator value held by `lane` at
      //                        slot `slot` (0..7) after WMMA.
      //
      // With B = I, the WMMA result D = A * I = A, so out[lane*8 + slot] tells
      // us which A[row][col] that (lane, slot) corresponds to.  This is the
      // ground-truth layout map we need to design real WMMA kernels.
      // ---------------------------------------------------------------------
      __global__ void wmma_probe_kernel(const _Float16* __restrict__ A,
                                        const _Float16* __restrict__ B,
                                        float* __restrict__ out)
      {
        const int lane = threadIdx.x;            // 0 .. 31

        // -- Load A row for this lane (rows 0..15; lanes 16..31 duplicate) --
        v16f16 a_frag;
        const int row = lane % 16;
        #pragma unroll
        for (int j = 0; j < 16; ++j)
          a_frag[j] = A[row * 16 + j];

        // -- Load B column for this lane (cols 0..15; lanes 16..31 duplicate) --
        v16f16 b_frag;
        const int col = lane % 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          b_frag[i] = B[i * 16 + col];

        // -- Zero the accumulator --
        v8f32 c_frag;
        #pragma unroll
        for (int s = 0; s < 8; ++s) c_frag[s] = 0.f;

        // -- The matmul --
        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

        // -- Spill every lane's 8 accumulator slots so the host can inspect --
        #pragma unroll
        for (int s = 0; s < 8; ++s)
          out[lane * 8 + s] = c_frag[s];
      }

    }  // namespace wmma_probe
  }  // namespace ops
}  // namespace ctranslate2

// ---------------------------------------------------------------------------
// C-ABI driver — invoked via ctypes from a small Python script.
// Allocates a 16x16 fp16 A (A[i][j] = i*16 + j) and identity B, runs the
// probe kernel on the current device, and copies the per-lane accumulator
// dump into the caller-provided buffer.
// out must point to at least 32*8 floats of host memory.
// Returns 0 on success, a hipError_t code otherwise.
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport)
int ct2_wmma_probe_run(float* out_host)
{
  using ctranslate2::ops::wmma_probe::wmma_probe_kernel;

  const int N = 16 * 16;
  _Float16 A_host[N];
  _Float16 B_host[N];
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      A_host[i * 16 + j] = static_cast<_Float16>(i * 16 + j);
      B_host[i * 16 + j] = static_cast<_Float16>(i == j ? 1 : 0);
    }

  _Float16 *A_d = nullptr, *B_d = nullptr;
  float* out_d = nullptr;
  hipError_t err;

  err = hipMalloc(&A_d, N * sizeof(_Float16));   if (err) return (int)err;
  err = hipMalloc(&B_d, N * sizeof(_Float16));   if (err) return (int)err;
  err = hipMalloc(&out_d, 32 * 8 * sizeof(float)); if (err) return (int)err;

  err = hipMemcpy(A_d, A_host, N * sizeof(_Float16), hipMemcpyHostToDevice);
  if (err) return (int)err;
  err = hipMemcpy(B_d, B_host, N * sizeof(_Float16), hipMemcpyHostToDevice);
  if (err) return (int)err;

  hipLaunchKernelGGL(wmma_probe_kernel, dim3(1), dim3(32), 0, 0, A_d, B_d, out_d);
  err = hipDeviceSynchronize();
  if (err) return (int)err;

  err = hipMemcpy(out_host, out_d, 32 * 8 * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(A_d); hipFree(B_d); hipFree(out_d);
  return (int)err;
}

#endif  // CT2_USE_HIP
