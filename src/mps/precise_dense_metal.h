// FP16 model matrices with FP32 activations/results. No expanded weight copy.
R"METAL(
struct PreciseDenseArgs { uint m, n, k, has_bias, has_residual; };
kernel void precise_dense_gemv(
    device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant PreciseDenseArgs& a [[buffer(3)]],
    device const half* bias [[buffer(4)]], device const float* residual [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.y, col = (group.x * 4 + simd) * 4;
  float sums[4] = {0, 0, 0, 0};
  for (uint k = lane; k < a.k; k += 32) {
    const float value = x[ulong(row) * a.k + k];
    for (uint j = 0; j < 4; ++j)
      if (col + j < a.n)
        sums[j] = fma(value, float(w[ulong(col+j) * a.k + k]), sums[j]);
  }
  for (uint j = 0; j < 4; ++j) {
    const float sum = simd_sum(sums[j]);
    if (lane == 0 && col + j < a.n) {
      const ulong index = ulong(row) * a.n + col + j;
      y[index] = sum + (a.has_bias ? float(bias[col+j]) : 0)
                     + (a.has_residual ? residual[index] : 0);
    }
  }
}
kernel void precise_dense_gemm(
    device const float* x [[buffer(0)]], device const half* w [[buffer(1)]],
    device float* y [[buffer(2)]], constant PreciseDenseArgs& a [[buffer(3)]],
    device const half* bias [[buffer(4)]], device const float* residual [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint LD = 20;
  threadgroup float xs[16 * LD];
  threadgroup half ws[32 * LD];
  const uint row0 = group.y * 16, col0 = group.x * 32;
  const uint sr = (simd >> 1) * 8, sc = (simd & 1) * 16;
  const uint quad = lane >> 2;
  const uint fr = (quad & 4) + ((lane >> 1) & 3);
  const uint fc = (quad & 2) * 2 + (lane & 1) * 2;
  simdgroup_matrix<float, 8, 8> accum[2];
  for (uint j = 0; j < 2; ++j) {
    accum[j].thread_elements()[0] = 0;
    accum[j].thread_elements()[1] = 0;
  }
  for (uint start = 0; start < a.k; start += 16) {
    for (uint i = tid; i < 16*16; i += 128) {
      const uint r = row0 + i/16, k = start + i%16;
      xs[i/16*LD+i%16] = r < a.m && k < a.k ? x[ulong(r)*a.k+k] : 0;
    }
    for (uint i = tid; i < 32*16; i += 128) {
      const uint c = col0 + i/16, k = start + i%16;
      ws[i/16*LD+i%16] = c < a.n && k < a.k ? w[ulong(c)*a.k+k] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ki = 0; ki < 2; ++ki) {
      simdgroup_matrix<float, 8, 8> af;
      af.thread_elements()[0] = xs[(sr+fr)*LD+ki*8+fc];
      af.thread_elements()[1] = xs[(sr+fr)*LD+ki*8+fc+1];
      for (uint j = 0; j < 2; ++j) {
        simdgroup_matrix<float, 8, 8> bf;
        const uint c = sc+j*8+fc, r = ki*8+fr;
        bf.thread_elements()[0] = float(ws[c*LD+r]);
        bf.thread_elements()[1] = float(ws[(c+1)*LD+r]);
        simdgroup_multiply_accumulate(accum[j], af, bf, accum[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row = row0+sr+fr;
  for (uint j = 0; j < 2; ++j)
    for (uint element = 0; element < 2; ++element) {
      const uint col = col0+sc+j*8+fc+element;
      if (row < a.m && col < a.n) {
        const ulong index = ulong(row)*a.n+col;
        y[index] = accum[j].thread_elements()[element]
          + (a.has_bias ? float(bias[col]) : 0)
          + (a.has_residual ? residual[index] : 0);
      }
    }
}
)METAL"
