R"METAL(
struct MoePlanArgs { uint tokens, experts, topk, hidden, intermediate, decode; };
// Decode plans fit in one threadgroup. Each expert independently derives its
// prefix from the selected IDs, then writes its stable grouping and both
// indirect grids. This replaces count/prefix/group without global atomics.
kernel void moe_plan_small(device const int* ids [[buffer(0)]], device uint* offsets [[buffer(1)]],
    device uint* permutation [[buffer(2)]], device uint* inverse [[buffer(3)]],
    device uint* indirect [[buffer(4)]], constant MoePlanArgs& a [[buffer(5)]],
    uint expert [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint counts[4], earlier[4];
  const uint assignments = a.tokens*a.topk;
  const uint hit = tid < assignments && ids[tid] == int(expert);
  const uint before = tid < assignments && ids[tid] < int(expert);
  const uint prefix = simd_prefix_exclusive_sum(hit);
  const uint count = simd_sum(hit), first = simd_sum(before);
  if (lane == 0) { counts[simd] = count; earlier[simd] = first; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint start = earlier[0]+earlier[1]+earlier[2]+earlier[3];
  if (hit) {
    uint destination = start+prefix;
    for (uint i = 0; i < simd; ++i) destination += counts[i];
    permutation[destination] = tid;
    inverse[tid] = destination;
  }
  if (tid == 0) {
    const uint total = counts[0]+counts[1]+counts[2]+counts[3];
    offsets[expert] = start;
    if (expert+1 == a.experts) offsets[expert+1] = assignments;
    for (uint stage = 0; stage < 2; ++stage) {
      const uint at = (expert*2+stage)*3;
      indirect[at] = total ? ((stage ? a.hidden : a.intermediate)+7)/8 : 1;
      indirect[at+1] = total ? total : 1;
      indirect[at+2] = 1;
    }
  }
}
// One group per expert counts routed assignments. Counts never leave the GPU.
kernel void moe_count(device const int* ids [[buffer(0)]], device uint* offsets [[buffer(1)]],
                       constant MoePlanArgs& a [[buffer(2)]],
                       uint expert [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint sums[4];
  uint count = 0;
  for (uint i = tid; i < a.tokens*a.topk; i += 128) count += ids[i] == int(expert);
  count = simd_sum(count);
  if (lane == 0) sums[simd] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) offsets[expert+1] = sums[0]+sums[1]+sums[2]+sums[3];
}
// E is small metadata. Prefix offsets and indirect grids are generated in one
// dispatch without CPU readbacks; an empty expert receives one no-op group.
kernel void moe_offsets(device uint* offsets [[buffer(0)]], device uint* indirect [[buffer(1)]],
                         constant MoePlanArgs& a [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  offsets[0] = 0;
  for (uint e = 0; e < a.experts; ++e) {
    const uint count = offsets[e+1];
    offsets[e+1] = offsets[e]+count;
    for (uint stage = 0; stage < 2; ++stage) {
      const uint columns = stage ? a.hidden : a.intermediate;
      const uint at = (e*2+stage)*3;
      const uint bm = a.decode == 1 ? 1 : a.decode == 2 ? 64 : 16;
      const uint bn = a.decode == 1 ? 8 : a.decode >= 2 ? 64 : 32;
      indirect[at] = count ? (columns+bn-1)/bn : 1;
      indirect[at+1] = count ? (count+bm-1)/bm : 1;
      indirect[at+2] = 1;
    }
  }
}
// Stable grouping of flattened [token, selected slot] assignments. The inverse
// mapping makes aggregation independent of group order and expert load.
kernel void moe_group(device const int* ids [[buffer(0)]], device const uint* offsets [[buffer(1)]],
                       device uint* permutation [[buffer(2)]], device uint* inverse [[buffer(3)]],
                       constant MoePlanArgs& a [[buffer(4)]], uint expert [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint sums[4];
  uint cursor = offsets[expert];
  for (uint base = 0; base < a.tokens*a.topk; base += 128) {
    const uint assignment = base+tid;
    const uint hit = assignment < a.tokens*a.topk && ids[assignment] == int(expert);
    const uint prefix = simd_prefix_exclusive_sum(hit), count = simd_sum(hit);
    if (lane == 0) sums[simd] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    for (uint group = 0; group < simd; ++group) before += sums[group];
    if (hit) {
      const uint destination = cursor+before+prefix;
      permutation[destination] = assignment; inverse[assignment] = destination;
    }
    cursor += sums[0]+sums[1]+sums[2]+sums[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
inline float moe_load(device const uchar* x, ulong i, uint dtype) {
  if (dtype == 0) return ((device const float*)x)[i];
  const ushort value = ((device const ushort*)x)[i];
  return dtype == 1 ? float(as_type<half>(value)) : bf16_to_f32(value);
}
inline float moe_round(float value, uint dtype) {
  return dtype == 0 ? value : dtype == 1 ? float(half(value)) : bf16_to_f32(f32_to_bf16(value));
}
inline void moe_store(device uchar* x, ulong i, uint dtype, float value) {
  if (dtype == 0) ((device float*)x)[i] = value;
  else ((device ushort*)x)[i] = dtype == 1 ? as_type<ushort>(half(value)) : f32_to_bf16(value);
}
// Mixed-storage router projection: retain FP32 router weights/accumulation
// without converting every activation tensor through the host.
struct MoeRouterArgs { uint experts, hidden, dtype; };
kernel void moe_router(device const uchar* input [[buffer(0)]],
                        device const float* weights [[buffer(1)]], device float* logits [[buffer(2)]],
                        constant MoeRouterArgs& a [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint groups = (a.experts+3)/4, token = group/groups;
  const uint expert = (group%groups)*4+simd;
  float sum = 0;
  if (expert < a.experts)
    for (uint k = lane; k < a.hidden; k += 32)
      sum = fma(moe_load(input, ulong(token)*a.hidden+k, a.dtype),
                weights[ulong(expert)*a.hidden+k], sum);
  sum = simd_sum(sum);
  if (lane == 0 && expert < a.experts) logits[ulong(token)*a.experts+expert] = sum;
}
inline float moe_swiglu(float gate, float up, uint dtype) {
  const float g = moe_round(gate, dtype), u = moe_round(up, dtype);
  const float activation = moe_round(g / (1.f+exp(-g)), dtype);
  return moe_round(activation*u, dtype);
}
struct MoeGemmArgs { uint expert, topk, n, k, dtype, gated, vector_loads, int4; };
inline float moe_weight(device const uchar* packed, device const float* scales,
                         uint row, uint col, constant MoeGemmArgs& a) {
  if (!a.int4) return moe_load(packed,ulong(row)*a.k+col,a.dtype);
  const uchar byte=packed[ulong(row)*((a.k+1)/2)+col/2];
  const int q=int((byte>>((col&1)*4))&15)-8;
  return float(half(float(q)*scales[ulong(row)*((a.k+63)/64)+col/64]));
}
inline float4 moe_int4_weight4(device const uchar* packed, device const float* scales,
                               uint row, uint col, uint depth) {
  const ushort bits = *(device const ushort*)(packed+ulong(row)*(depth/2)+col/2);
  const int4 q = int4(int(bits&15), int((bits>>4)&15),
                      int((bits>>8)&15), int((bits>>12)&15))-8;
  // Four adjacent columns never cross the 64-column quantization group.
  return float4(half4(float4(q)*scales[ulong(row)*((depth+63)/64)+col/64]));
}
// Incremental/small-batch projection: each SIMD group produces two columns.
// K/V-style contiguous reads reuse each input for paired gate/up projections.
kernel void moe_expert_gemv(device const uchar* input [[buffer(0)]], device const uchar* gate [[buffer(1)]],
                             device const uchar* up [[buffer(2)]], device uchar* output [[buffer(3)]],
                             device const uint* offsets [[buffer(4)]], device const uint* permutation [[buffer(5)]],
                             constant MoeGemmArgs& a [[buffer(6)]],
                             device const float* gate_scale [[buffer(7)]], device const float* up_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint first = offsets[a.expert], count = offsets[a.expert+1]-first;
  if (group.y >= count) return;
  const uint sorted = first+group.y, token = a.gated ? permutation[sorted]/a.topk : sorted;
  const uint column = (group.x*4+simd)*2;
  float g[2] = {0,0}, u[2] = {0,0};
  if (a.dtype == 1 && a.vector_loads) {
    // Move the storage-format branches outside the reduction and issue aligned
    // 64-bit loads. Each SIMD group still owns two adjacent output columns.
    for (uint kk = lane*4; kk < a.k; kk += 128) {
      const float4 x = float4(*(device const half4*)((device const half*)input+ulong(token)*a.k+kk));
      for (uint j = 0; j < 2; ++j) if (column+j < a.n) {
        const ulong at = ulong(column+j)*a.k+kk;
        const float4 gv = a.int4 ? moe_int4_weight4(gate, gate_scale, column+j, kk, a.k)
          : float4(*(device const half4*)((device const half*)gate+at));
        for (uint el = 0; el < 4; ++el) g[j] = fma(x[el], gv[el], g[j]);
        if (a.gated) {
          const float4 uv = a.int4 ? moe_int4_weight4(up, up_scale, column+j, kk, a.k)
            : float4(*(device const half4*)((device const half*)up+at));
          for (uint el = 0; el < 4; ++el) u[j] = fma(x[el], uv[el], u[j]);
        }
      }
    }
  } else {
    for (uint kk = lane; kk < a.k; kk += 32) {
      const float x = moe_load(input, ulong(token)*a.k+kk, a.dtype);
      for (uint j = 0; j < 2; ++j) if (column+j < a.n) {
        g[j] = fma(x, moe_weight(gate, gate_scale, column+j, kk, a), g[j]);
        if (a.gated) u[j] = fma(x, moe_weight(up, up_scale, column+j, kk, a), u[j]);
      }
    }
  }
  for (uint j = 0; j < 2; ++j) {
    const float gs = simd_sum(g[j]), us = simd_sum(u[j]);
    if (lane == 0 && column+j < a.n)
      moe_store(output, ulong(sorted)*a.n+column+j, a.dtype,
                a.gated ? moe_swiglu(gs, us, a.dtype) : gs);
  }
}

// 16x32x16 FP32-accumulating SIMD-matrix tile. Gate/up share the input tile;
// expert output offsets and the dispatch grid are generated on the GPU.
kernel void moe_expert_gemm(device const uchar* input [[buffer(0)]], device const uchar* gate [[buffer(1)]],
                             device const uchar* up [[buffer(2)]], device uchar* output [[buffer(3)]],
                             device const uint* offsets [[buffer(4)]], device const uint* permutation [[buffer(5)]],
                             constant MoeGemmArgs& a [[buffer(6)]],
                             device const float* gate_scale [[buffer(7)]], device const float* up_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint first = offsets[a.expert], count = offsets[a.expert+1]-first;
  const uint tile_row = group.y*16, tile_col = group.x*32;
  if (tile_row >= count) return;
  constexpr uint LD = 20;
  threadgroup float as[16*LD], gs[32*LD], us[32*LD];
  const uint tid = simd*32+lane, simd_row = (simd>>1)*8, simd_col = (simd&1)*16;
  const uint quad = lane>>2, fr = (quad&4)+((lane>>1)&3), fc = (quad&2)*2+(lane&1)*2;
  simdgroup_matrix<float,8,8> cg[2], cu[2];
  for (uint j = 0; j < 2; ++j) for (uint el = 0; el < 2; ++el) {
    cg[j].thread_elements()[el] = 0; cu[j].thread_elements()[el] = 0;
  }
  for (uint start = 0; start < a.k; start += 16) {
    for (uint i = tid; i < 16*16; i += 128) {
      const uint row = tile_row+i/16, kk = start+i%16;
      float value = 0;
      if (row < count && kk < a.k) {
        const uint token = a.gated ? permutation[first+row]/a.topk : first+row;
        value = moe_load(input, ulong(token)*a.k+kk, a.dtype);
      }
      as[(i/16)*LD+i%16] = value;
    }
    for (uint i = tid; i < 32*16; i += 128) {
      const uint column = tile_col+i/16, kk = start+i%16;
      gs[(i/16)*LD+i%16] = column < a.n && kk < a.k
        ? moe_weight(gate, gate_scale, column, kk, a) : 0;
      if (a.gated) us[(i/16)*LD+i%16] = column < a.n && kk < a.k
        ? moe_weight(up, up_scale, column, kk, a) : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ki = 0; ki < 2; ++ki) {
      simdgroup_matrix<float,8,8> af;
      af.thread_elements()[0] = as[(simd_row+fr)*LD+ki*8+fc];
      af.thread_elements()[1] = as[(simd_row+fr)*LD+ki*8+fc+1];
      for (uint j = 0; j < 2; ++j) {
        simdgroup_matrix<float,8,8> bf;
        const uint r = ki*8+fr, c = simd_col+j*8+fc;
        bf.thread_elements()[0] = gs[c*LD+r]; bf.thread_elements()[1] = gs[(c+1)*LD+r];
        simdgroup_multiply_accumulate(cg[j], af, bf, cg[j]);
        if (a.gated) {
          bf.thread_elements()[0] = us[c*LD+r]; bf.thread_elements()[1] = us[(c+1)*LD+r];
          simdgroup_multiply_accumulate(cu[j], af, bf, cu[j]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row = tile_row+simd_row+fr;
  if (row >= count) return;
  for (uint j = 0; j < 2; ++j) for (uint el = 0; el < 2; ++el) {
    const uint column = tile_col+simd_col+j*8+fc+el;
    if (column < a.n) moe_store(output, ulong(first+row)*a.n+column, a.dtype,
      a.gated ? moe_swiglu(cg[j].thread_elements()[el], cu[j].thread_elements()[el], a.dtype)
              : cg[j].thread_elements()[el]);
  }
}

// Accumulate in ascending expert order, matching the source model's index_add.
// Round each product/add at the model dtype boundaries, including BF16.
kernel void moe_aggregate(device const uchar* expert_output [[buffer(0)]], device const uint* inverse [[buffer(1)]],
                           device const int* ids [[buffer(2)]], device const float* weights [[buffer(3)]],
                           device uchar* output [[buffer(4)]], constant MoePlanArgs& a [[buffer(5)]],
                           constant uint& dtype [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
  const uint token = gid/a.hidden, column = gid%a.hidden;
  if (token >= a.tokens) return;
  float sum = 0; int previous = -1;
  for (uint step = 0; step < a.topk; ++step) {
    uint selected = 0; int minimum = 0x7fffffff;
    for (uint slot = 0; slot < a.topk; ++slot) {
      const int expert = ids[token*a.topk+slot];
      if (expert > previous && expert < minimum) { minimum = expert; selected = slot; }
    }
    previous = minimum;
    const uint assignment = token*a.topk+selected;
    const float weight = moe_round(weights[assignment], dtype);
    const float value = moe_load(expert_output, ulong(inverse[assignment])*a.hidden+column, dtype);
    const float product = moe_round(weight*value, dtype);
    sum = moe_round(sum+product, dtype);
  }
  moe_store(output, gid, dtype, sum);
}
struct MoeSharedArgs { uint hidden, dtype; };
kernel void moe_add_shared(device const uchar* shared [[buffer(0)]], device const uchar* gates [[buffer(1)]],
                           device uchar* output [[buffer(2)]], constant MoeSharedArgs& a [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
  const float product=moe_round(moe_load(shared,gid,a.dtype)*moe_load(gates,gid/a.hidden,a.dtype),a.dtype);
  moe_store(output,gid,a.dtype,moe_round(moe_load(output,gid,a.dtype)+product,a.dtype));
}
)METAL"
