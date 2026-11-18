// Compact BTHD tiled prefill. All score/softmax/output accumulators are FP32.
R"METAL(
struct FlashPrefillArgs {
  uint queries, keys, capacity, heads, kv_heads, depth;
  uint offset, window, causal, has_lengths, interleaved_lengths, bf16, cache_bits;
  float scale;
};
template<uint D, bool QUANTIZED = false>
inline void flash_prefill_tile(
    device const ushort* queries, device const ushort* keys, device const ushort* values,
    device const int* lengths, device ushort* output, constant FlashPrefillArgs& a,
    uint2 group, uint tid, uint lane, uint simd,
    threadgroup float* qt, threadgroup float* kt, threadgroup float* probabilities,
    threadgroup float* maxima, threadgroup float* denominators, threadgroup float* corrections,
    threadgroup uint* limits, threadgroup uint* starts) {
  constexpr uint BQ = 32, BK = 32, LD = 20;
  const uint batch = group.y/a.heads, head = group.y%a.heads;
  const uint kvhead = head/(a.heads/a.kv_heads), query_start = group.x*BQ;
  const uint quad = lane>>2, fr = (quad&4)+((lane>>1)&3), fc = (quad&2)*2+(lane&1)*2;
  if (tid < BQ) {
    const uint q = query_start+tid, position = a.offset+q;
    uint last = a.keys;
    if (q >= a.queries) last = 0;
    else if (a.has_lengths) {
      const ulong index = a.interleaved_lengths
        ? (ulong(batch)*a.queries+q)*a.heads+head : (ulong(batch)*a.heads+head)*a.queries+q;
      last = uint(clamp(lengths[index], 0, int(a.keys)));
    }
    if (a.causal) last = min(last, position+1);
    limits[tid] = last;
    starts[tid] = q < a.queries ? (a.window && position+1 > a.window ? position+1-a.window : 0) : 0xffffffffu;
    maxima[tid] = -INFINITY; denominators[tid] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint last = simd_max(limits[lane]);
  const uint first = simd_min(starts[lane]);
  simdgroup_matrix<float,8,8> accum[D/8];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < D/8; ++j)
    accum[j].thread_elements()[0] = accum[j].thread_elements()[1] = 0;
  for (uint base = first/BK*BK; base < last; base += BK) {
    simdgroup_matrix<float,8,8> scores[4];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j)
      scores[j].thread_elements()[0] = scores[j].thread_elements()[1] = 0;
    for (uint start = 0; start < a.depth; start += 16) {
      for (uint i = tid; i < BQ*16; i += 128) {
        const uint row = i/16, d = start+i%16;
        const uint q = query_start+row, key = base+row;
        qt[row*LD+i%16] = q < a.queries && d < a.depth
          ? attention_value(queries[((ulong(batch)*a.queries+q)*a.heads+head)*a.depth+d], a.bf16) : 0;
        kt[row*LD+i%16] = key >= first && key < last && d < a.depth
          ? kv_cache_value<QUANTIZED>(keys,(ulong(batch)*a.capacity+key)*a.kv_heads+kvhead,d,a.depth,a.cache_bits,a.bf16) : 0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      #pragma clang loop unroll(full)
      for (uint ki = 0; ki < 2; ++ki) {
        simdgroup_matrix<float,8,8> af;
        af.thread_elements()[0] = qt[(simd*8+fr)*LD+ki*8+fc];
        af.thread_elements()[1] = qt[(simd*8+fr)*LD+ki*8+fc+1];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
          simdgroup_matrix<float,8,8> bf;
          bf.thread_elements()[0] = kt[(j*8+fc)*LD+ki*8+fr];
          bf.thread_elements()[1] = kt[(j*8+fc+1)*LD+ki*8+fr];
          simdgroup_multiply_accumulate(scores[j], af, bf, scores[j]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint r = simd*8+fr;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) {
      #pragma clang loop unroll(full)
      for (uint el = 0; el < 2; ++el) {
        const uint c = j*8+fc+el, key = base+c;
        probabilities[r*BK+c] = key >= starts[r] && key < limits[r]
          ? scores[j].thread_elements()[el]*a.scale : -INFINITY;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Four adjacent lanes own a row; max and sum stay in that subgroup.
    const uint row = tid/4, sub = tid%4;
    float m = -INFINITY;
    for (uint c = sub; c < BK; c += 4) m = max(m, probabilities[row*BK+c]);
    m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2));
    const float next = max(maxima[row], m);
    const float correction = maxima[row] == -INFINITY ? 0 : exp(maxima[row]-next);
    float sum = 0;
    for (uint c = sub; c < BK; c += 4) {
      const float score = probabilities[row*BK+c];
      const float p = score == -INFINITY ? 0 : exp(score-next);
      probabilities[row*BK+c] = p; sum += p;
    }
    sum += simd_shuffle_xor(sum, 1); sum += simd_shuffle_xor(sum, 2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0) {
      maxima[row] = next;
      denominators[row] = denominators[row]*correction+sum;
      corrections[row] = correction;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < D/8; ++j) {
      accum[j].thread_elements()[0] *= corrections[r];
      accum[j].thread_elements()[1] *= corrections[r];
    }
    #pragma clang loop unroll(full)
    for (uint ki = 0; ki < 4; ++ki) {
      simdgroup_matrix<float,8,8> af;
      af.thread_elements()[0] = probabilities[r*BK+ki*8+fc];
      af.thread_elements()[1] = probabilities[r*BK+ki*8+fc+1];
      const uint key = base+ki*8+fr;
      #pragma clang loop unroll(full)
      for (uint j = 0; j < D/8; ++j) {
        simdgroup_matrix<float,8,8> bf;
        const uint col = j*8+fc;
        const ulong cache_row = (ulong(batch)*a.capacity+key)*a.kv_heads+kvhead;
        bf.thread_elements()[0] = key >= first && key < last && col < a.depth
          ? kv_cache_value<QUANTIZED>(values,cache_row,col,a.depth,a.cache_bits,a.bf16) : 0;
        bf.thread_elements()[1] = key >= first && key < last && col+1 < a.depth
          ? kv_cache_value<QUANTIZED>(values,cache_row,col+1,a.depth,a.cache_bits,a.bf16) : 0;
        simdgroup_multiply_accumulate(accum[j], af, bf, accum[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row = simd*8+fr, query = query_start+row;
  if (query >= a.queries) return;
  const float denominator = denominators[row];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < D/8; ++j) {
    #pragma clang loop unroll(full)
    for (uint el = 0; el < 2; ++el) {
      const uint col = j*8+fc+el;
      if (col < a.depth) {
        const ulong at = ((ulong(batch)*a.queries+query)*a.heads+head)*a.depth+col;
        const float value = denominator > 0 ? accum[j].thread_elements()[el]/denominator : 0;
        if (a.bf16) store_value_bf16(output, at, value);
        else output[at] = as_type<ushort>(half(value));
      }
    }
  }
}
#define FLASH_PREFILL_ENTRY(NAME, DEPTH, QUANTIZED) \
kernel void NAME( \
    device const ushort* q [[buffer(0)]], device const ushort* k [[buffer(1)]], \
    device const ushort* v [[buffer(2)]], device const int* lengths [[buffer(3)]], \
    device ushort* out [[buffer(4)]], constant FlashPrefillArgs& a [[buffer(5)]], \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float qt[32*20], kt[32*20], p[32*32]; \
  threadgroup float maxima[32], denominators[32], corrections[32]; \
  threadgroup uint limits[32], starts[32]; \
  flash_prefill_tile<DEPTH,QUANTIZED>(q,k,v,lengths,out,a,group,tid,lane,simd, \
                           qt,kt,p,maxima,denominators,corrections,limits,starts); \
}
FLASH_PREFILL_ENTRY(flash_prefill_64,64,false)
FLASH_PREFILL_ENTRY(flash_prefill_quantized_64,64,true)
FLASH_PREFILL_ENTRY(flash_prefill_128,128,false)
FLASH_PREFILL_ENTRY(flash_prefill_quantized_128,128,true)
FLASH_PREFILL_ENTRY(flash_prefill_256,256,false)
FLASH_PREFILL_ENTRY(flash_prefill_quantized_256,256,true)
)METAL"
