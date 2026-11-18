// Uses the existing Dense SIMD fragment layout, adapted from Apple MLX Steel
// (Copyright (c) 2024 Apple Inc., Apache-2.0), with GPU expert permutation.
R"METAL(
template<bool GATED, uint BM>
inline void moe_wide_f16(
    device const half* input, device const half* gate, device const half* up,
    device half* output, device const uint* offsets, device const uint* permutation,
    device const float* gate_scale, device const float* up_scale,
    constant MoeGemmArgs& a, uint2 group, uint lane, uint simd,
    threadgroup half* as, threadgroup half* gs, threadgroup half* us) {
  constexpr uint BN = 64, BK = 32, LD = 40, RN = BM/16;
  const uint first = offsets[a.expert], count = offsets[a.expert+1]-first;
  const uint tile_row = group.y*BM, tile_col = group.x*BN;
  if (tile_row >= count) return;
  struct alignas(16) Half8Bytes { uchar bytes[16]; };
  const uint tid = simd*32+lane, simd_row = (simd>>1)*(BM/2), simd_col = (simd&1)*32;
  const uint quad = lane>>2, fr = (quad&4)+((lane>>1)&3), fc = (quad&2)*2+(lane&1)*2;
  simdgroup_matrix<float,8,8> cg[RN][4], cu[RN][4];
  #pragma clang loop unroll(full)
  for (uint i = 0; i < RN; ++i) {
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
      cg[i][j].thread_elements()[0] = cg[i][j].thread_elements()[1] = 0;
      if (GATED) cu[i][j].thread_elements()[0] = cu[i][j].thread_elements()[1] = 0;
    }
  }
  for (uint start = 0; start < a.k; start += BK) {
    for (uint index = tid; index < BM*4; index += 128) {
      const uint row = tile_row+index/4, kk = start+(index%4)*8;
      threadgroup half* dst = as+(index/4)*LD+(index%4)*8;
      if (row < count) {
        const uint token = GATED ? permutation[first+row]/a.topk : first+row;
        const ulong at = ulong(token)*a.k+kk;
        if (a.vector_loads && kk+8 <= a.k)
          *(threadgroup Half8Bytes*)dst = *(device const Half8Bytes*)(input+at);
        else {
          #pragma clang loop unroll(full)
          for (uint el = 0; el < 8; ++el) dst[el] = kk+el < a.k ? input[at+el] : half(0);
        }
      } else {
        #pragma clang loop unroll(full)
        for (uint el = 0; el < 8; ++el) dst[el] = half(0);
      }
    }
    for (uint index = tid; index < BN*4; index += 128) {
      const uint col = tile_col+index/4, kk = start+(index%4)*8;
      threadgroup half* gdst = gs+(index/4)*LD+(index%4)*8;
      threadgroup half* udst = us+(index/4)*LD+(index%4)*8;
      const ulong at = ulong(col)*a.k+kk;
      if (!a.int4 && a.vector_loads && col < a.n && kk+8 <= a.k) {
        *(threadgroup Half8Bytes*)gdst = *(device const Half8Bytes*)(gate+at);
        if (GATED) *(threadgroup Half8Bytes*)udst = *(device const Half8Bytes*)(up+at);
      } else {
        #pragma clang loop unroll(full)
        for (uint el = 0; el < 8; ++el) {
          gdst[el] = col < a.n && kk+el < a.k ? half(moe_weight((device const uchar*)gate,gate_scale,col,kk+el,a)) : half(0);
          if (GATED) udst[el] = col < a.n && kk+el < a.k ? half(moe_weight((device const uchar*)up,up_scale,col,kk+el,a)) : half(0);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint ki = 0; ki < 4; ++ki) {
      simdgroup_matrix<float,8,8> af[RN], gf[4], uf[4];
      #pragma clang loop unroll(full)
      for (uint i = 0; i < RN; ++i) {
        const uint r = simd_row+i*8+fr, c = ki*8+fc;
        af[i].thread_elements()[0] = float(as[r*LD+c]);
        af[i].thread_elements()[1] = float(as[r*LD+c+1]);
      }
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 4; ++j) {
        const uint r = ki*8+fr, c = simd_col+j*8+fc;
        gf[j].thread_elements()[0] = float(gs[c*LD+r]);
        gf[j].thread_elements()[1] = float(gs[(c+1)*LD+r]);
        if (GATED) {
          uf[j].thread_elements()[0] = float(us[c*LD+r]);
          uf[j].thread_elements()[1] = float(us[(c+1)*LD+r]);
        }
      }
      #pragma clang loop unroll(full)
      for (uint i = 0; i < RN; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
          simdgroup_multiply_accumulate(cg[i][j], af[i], gf[j], cg[i][j]);
          if (GATED) simdgroup_multiply_accumulate(cu[i][j], af[i], uf[j], cu[i][j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  #pragma clang loop unroll(full)
  for (uint i = 0; i < RN; ++i) {
    const uint row = tile_row+simd_row+i*8+fr;
    if (row >= count) continue;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
      #pragma clang loop unroll(full)
      for (uint el = 0; el < 2; ++el) {
        const uint col = tile_col+simd_col+j*8+fc+el;
        if (col < a.n) {
          const float value = cg[i][j].thread_elements()[el];
          output[ulong(first+row)*a.n+col] = half(GATED
            ? moe_swiglu(value, cu[i][j].thread_elements()[el], 1) : value);
        }
      }
    }
  }
}
kernel void moe_expert_gate_up_f16_64x64(
    device const half* input [[buffer(0)]], device const half* gate [[buffer(1)]],
    device const half* up [[buffer(2)]], device half* output [[buffer(3)]],
    device const uint* offsets [[buffer(4)]], device const uint* permutation [[buffer(5)]],
    constant MoeGemmArgs& a [[buffer(6)]],
    device const float* gate_scale [[buffer(7)]], device const float* up_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup half as[64*40], gs[64*40], us[64*40];
  moe_wide_f16<true,64>(input, gate, up, output, offsets, permutation, gate_scale, up_scale, a, group, lane, simd, as, gs, us);
}
kernel void moe_expert_down_f16_64x64(
    device const half* input [[buffer(0)]], device const half* gate [[buffer(1)]],
    device const half* up [[buffer(2)]], device half* output [[buffer(3)]],
    device const uint* offsets [[buffer(4)]], device const uint* permutation [[buffer(5)]],
    constant MoeGemmArgs& a [[buffer(6)]],
    device const float* gate_scale [[buffer(7)]], device const float* up_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup half as[64*40], gs[64*40];
  moe_wide_f16<false,64>(input, gate, up, output, offsets, permutation, gate_scale, up_scale, a, group, lane, simd, as, gs, gs);
}
kernel void moe_expert_gate_up_f16_16x64(
    device const half* input [[buffer(0)]], device const half* gate [[buffer(1)]],
    device const half* up [[buffer(2)]], device half* output [[buffer(3)]],
    device const uint* offsets [[buffer(4)]], device const uint* permutation [[buffer(5)]],
    constant MoeGemmArgs& a [[buffer(6)]],
    device const float* gate_scale [[buffer(7)]], device const float* up_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup half as[16*40], gs[64*40], us[64*40];
  moe_wide_f16<true,16>(input, gate, up, output, offsets, permutation, gate_scale, up_scale, a, group, lane, simd, as, gs, us);
}
kernel void moe_expert_down_f16_16x64(
    device const half* input [[buffer(0)]], device const half* gate [[buffer(1)]],
    device const half* up [[buffer(2)]], device half* output [[buffer(3)]],
    device const uint* offsets [[buffer(4)]], device const uint* permutation [[buffer(5)]],
    constant MoeGemmArgs& a [[buffer(6)]],
    device const float* gate_scale [[buffer(7)]], device const float* up_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup half as[16*40], gs[64*40];
  moe_wide_f16<false,16>(input, gate, up, output, offsets, permutation, gate_scale, up_scale, a, group, lane, simd, as, gs, gs);
}
)METAL"
