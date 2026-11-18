// Appended to the shared Metal library, after the storage conversion helpers.
R"METAL(
struct CacheAppendArgs {
  uint batch, queries, heads, depth, offset, old_capacity, capacity, copy_past;
};

kernel void append_compact_kv(
    device const ushort* keys [[buffer(0)]], device const ushort* values [[buffer(1)]],
    device const ushort* old_keys [[buffer(2)]], device const ushort* old_values [[buffer(3)]],
    device ushort* cached_keys [[buffer(4)]], device ushort* cached_values [[buffer(5)]],
    constant CacheAppendArgs& a [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const ulong width = ulong(a.heads) * a.depth;
  const ulong rows = a.copy_past ? a.offset + a.queries : a.queries;
  if (ulong(gid) >= ulong(a.batch) * rows * width) return;
  const ulong column = gid % width, row = gid / width;
  const ulong batch = row / rows, token = row % rows + (a.copy_past ? 0 : a.offset);
  const ulong out = (batch * a.capacity + token) * width + column;
  if (a.copy_past && token < a.offset) {
    const ulong in = (batch * a.old_capacity + token) * width + column;
    cached_keys[out] = old_keys[in]; cached_values[out] = old_values[in];
  } else {
    const ulong in = (batch * a.queries + token - a.offset) * width + column;
    cached_keys[out] = keys[in]; cached_values[out] = values[in];
  }
}

struct FlashDecodeArgs {
  uint batch, heads, kv_heads, keys, capacity, depth, splits;
  uint offset, window, has_lengths, causal, bf16;
  uint queries, kv_interleaved, queries_interleaved, cache_bits, key_start;
  float scale;
};

// One query/head, split into 256-key blocks. Each thread computes two scores;
// all threads then cooperate on the value product. Partial FP32 numerators,
// maxima and denominators permit a stable merge without storing attention.
template<bool COOPERATIVE, bool QUANTIZED = false>
inline void flash_decode_partial_impl(
    device const ushort* q, device const ushort* k, device const ushort* v,
    device const int* lengths, device float* partials, device ushort* output,
    constant FlashDecodeArgs& a, uint2 group, uint tid, uint lane, uint simd,
    threadgroup float* probabilities, threadgroup float* maxima, threadgroup float* sums) {
  const uint query = group.y % a.queries;
  const uint head = (group.y / a.queries) % a.heads, batch = group.y / a.queries / a.heads;
  const uint kv_head = head / (a.heads / a.kv_heads), start = a.key_start + group.x * 256;
  uint valid = a.keys;
  if (a.has_lengths) valid = uint(clamp(lengths[group.y], 0, int(a.keys)));
  const uint position = a.offset + query;
  if (a.causal) valid = min(valid, position + 1);
  const uint first = a.window && position + 1 > a.window ? position + 1 - a.window : 0;
  const ulong qbase = (a.queries_interleaved
    ? (ulong(batch)*a.queries+query)*a.heads+head : ulong(group.y)) * a.depth;
  const ulong kvrowbase = (a.kv_interleaved
    ? ulong(batch)*a.capacity*a.kv_heads+kv_head : (ulong(batch)*a.kv_heads+kv_head)*a.capacity);
  const ulong kvrowstride = a.kv_interleaved ? a.kv_heads : 1;
  float scores[2];
  if (COOPERATIVE) {
    // One SIMD group cooperates on each dot product. Adjacent lanes read
    // adjacent head dimensions instead of striding across entire KV rows.
    // Query fragments remain in registers across the 64 keys per SIMD group.
    float query_values[8];
    for (uint i = 0; i < 8; ++i) {
      const uint d = lane+i*32;
      query_values[i] = d < a.depth ? attention_value(q[qbase+d], a.bf16) : 0;
    }
    for (uint i = 0; i < 64; ++i) {
      const uint token = start+simd+i*4;
      float score = 0;
      if (token >= first && token < valid) {
        for (uint j = 0; j < 8; ++j) {
          const uint d = lane+j*32;
          if (d < a.depth)
            score = fma(query_values[j], kv_cache_value<QUANTIZED>(k,kvrowbase+ulong(token)*kvrowstride,d,a.depth,a.cache_bits,a.bf16), score);
        }
      }
      score = simd_sum(score);
      if (lane == 0) probabilities[simd+i*4] = token >= first && token < valid ? score*a.scale : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scores[0] = probabilities[tid];
    scores[1] = probabilities[tid+128];
    // All groups must finish reading the score tile before it is reused for
    // probabilities below; this barrier is required even on unified memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
  for (uint i = 0; i < 2; ++i) {
    const uint token = start + tid + i * 128;
    float score = 0;
    if (token >= first && token < valid) {
      for (uint d = 0; d < a.depth; ++d)
        score = fma(attention_value(q[qbase+d], a.bf16),
                    kv_cache_value<QUANTIZED>(k,kvrowbase+ulong(token)*kvrowstride,d,a.depth,a.cache_bits,a.bf16), score);
      scores[i] = score * a.scale;
    } else scores[i] = -INFINITY;
  }
  }
  const float local_max = simd_max(max(scores[0], scores[1]));
  if (lane == 0) maxima[simd] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float maximum = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
  float local_sum = 0;
  for (uint i = 0; i < 2; ++i) {
    const float p = scores[i] == -INFINITY ? 0 : exp(scores[i] - maximum);
    probabilities[tid+i*128] = p; local_sum += p;
  }
  const float sum = simd_sum(local_sum);
  if (lane == 0) sums[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float denominator = sums[0]+sums[1]+sums[2]+sums[3];
  const ulong partial_base = (ulong(group.y)*a.splits+group.x)*(a.depth+2);
  for (uint d = tid; d < a.depth; d += 128) {
    float numerator = 0;
    for (uint token = max(start, first); token < min(start+256, valid); ++token)
      numerator = fma(probabilities[token-start],
                      kv_cache_value<QUANTIZED>(v,kvrowbase+ulong(token)*kvrowstride,
                                                d,a.depth,a.cache_bits,a.bf16),numerator);
    if (a.splits > 1) partials[partial_base+d] = numerator;
    else {
      const float result = denominator > 0 ? numerator / denominator : 0;
      if (a.bf16) store_value_bf16(output, qbase+d, result);
      else output[qbase+d] = as_type<ushort>(half(result));
    }
  }
  if (tid == 0 && a.splits > 1) {
    partials[partial_base+a.depth] = maximum;
    partials[partial_base+a.depth+1] = denominator;
  }
}

kernel void flash_decode_partial(
    device const ushort* q [[buffer(0)]], device const ushort* k [[buffer(1)]],
    device const ushort* v [[buffer(2)]], device const int* lengths [[buffer(3)]],
    device float* partials [[buffer(4)]], device ushort* output [[buffer(5)]],
    constant FlashDecodeArgs& a [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float probabilities[256], maxima[4], sums[4];
  flash_decode_partial_impl<false>(q, k, v, lengths, partials, output, a,
                                      group, tid, lane, simd, probabilities, maxima, sums);
}

kernel void flash_decode_partial_cooperative(
    device const ushort* q [[buffer(0)]], device const ushort* k [[buffer(1)]],
    device const ushort* v [[buffer(2)]], device const int* lengths [[buffer(3)]],
    device float* partials [[buffer(4)]], device ushort* output [[buffer(5)]],
    constant FlashDecodeArgs& a [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float probabilities[256], maxima[4], sums[4];
  flash_decode_partial_impl<true>(q, k, v, lengths, partials, output, a,
                                      group, tid, lane, simd, probabilities, maxima, sums);
}

kernel void flash_decode_quantized(
    device const ushort* q [[buffer(0)]], device const ushort* k [[buffer(1)]],
    device const ushort* v [[buffer(2)]], device const int* lengths [[buffer(3)]],
    device float* partials [[buffer(4)]], device ushort* output [[buffer(5)]],
    constant FlashDecodeArgs& a [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float p[256], maxima[4], sums[4];
  flash_decode_partial_impl<true,true>(q,k,v,lengths,partials,output,a,
                                      group,tid,lane,simd,p,maxima,sums);
}

kernel void flash_decode_merge(device const float* partials [[buffer(0)]],
                                device ushort* output [[buffer(1)]],
                                constant FlashDecodeArgs& a [[buffer(2)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  const uint query = head % a.queries, h = (head/a.queries)%a.heads, b = head/a.queries/a.heads;
  const ulong outbase = (a.queries_interleaved
    ? (ulong(b)*a.queries+query)*a.heads+h : ulong(head))*a.depth;
  float maximum = -INFINITY;
  for (uint split = 0; split < a.splits; ++split)
    maximum = max(maximum, partials[(ulong(head)*a.splits+split)*(a.depth+2)+a.depth]);
  float denominator = 0;
  for (uint split = 0; split < a.splits; ++split) {
    const ulong base = (ulong(head)*a.splits+split)*(a.depth+2);
    const float sum = partials[base+a.depth+1];
    if (sum > 0) denominator += exp(partials[base+a.depth]-maximum)*sum;
  }
  for (uint d = tid; d < a.depth; d += 128) {
    float numerator = 0;
    for (uint split = 0; split < a.splits; ++split) {
      const ulong base = (ulong(head)*a.splits+split)*(a.depth+2);
      if (partials[base+a.depth+1] > 0)
        numerator = fma(exp(partials[base+a.depth]-maximum), partials[base+d], numerator);
    }
    const float result = denominator > 0 ? numerator / denominator : 0;
    if (a.bf16) store_value_bf16(output, outbase+d, result);
    else output[outbase+d] = as_type<ushort>(half(result));
  }
}

struct AttentionMaskArgs { uint queries, keys, offset, window, has_lengths, causal, bf16, heads, interleaved_lengths; };
kernel void mask_flash_scores(device ushort* scores [[buffer(0)]], device const int* lengths [[buffer(1)]],
                               device int* bounds [[buffer(2)]], constant AttentionMaskArgs& a [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  const uint row = gid / a.keys, key = gid % a.keys, position = a.offset + row % a.queries;
  const uint mask_row = a.interleaved_lengths
    ? ((row/a.queries/a.heads)*a.queries+row%a.queries)*a.heads+(row/a.queries)%a.heads : row;
  uint valid = a.has_lengths ? uint(clamp(lengths[mask_row], 0, int(a.keys))) : a.keys;
  if (a.causal) valid = min(valid, position+1);
  const uint first = a.window && position+1 > a.window ? position+1-a.window : 0;
  if (key == 0) bounds[row] = valid > first ? int(valid) : 0;
  if (key < first) scores[gid] = a.bf16 ? ushort(0xff80u) : ushort(0xfc00u);
}
)METAL"
