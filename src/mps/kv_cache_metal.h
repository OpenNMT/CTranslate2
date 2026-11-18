// Group-32 symmetric KV quantization. Scales and packed codes share each row,
// so ordinary GPU gather/reordering preserves the entire cache representation.
R"METAL(
inline uint packed_kv_width(uint depth, uint bits) {
  return ((depth+31)/32)*(4+4*bits);
}

template<bool QUANTIZED>
inline float kv_cache_value(device const ushort* input, ulong row, uint column,
                            uint depth, uint bits, bool bf16) {
  if (!QUANTIZED) return attention_value(input[row*depth+column], bf16);
  device const uchar* group = (device const uchar*)input
    + row*packed_kv_width(depth,bits)+(column/32)*(4+4*bits);
  const uint bit = (column%32)*bits, byte = bit/8, shift = bit%8;
  uint code = uint(group[4+byte]) >> shift;
  if (shift+bits > 8) code |= uint(group[5+byte]) << (8-shift);
  const int value = int(code&((1u<<bits)-1))-int(1u<<(bits-1));
  // Quantized cache options require FP16 activations. Match the FP16 CPU oracle.
  return float(half(float(value)*(*(device const float*)group)));
}

struct QuantizedCacheArgs {
  uint batch, queries, heads, depth, offset, old_capacity, capacity, copy_past, bits;
};
kernel void append_quantized_kv(
    device const half* keys [[buffer(0)]], device const half* values [[buffer(1)]],
    device const uchar* old_keys [[buffer(2)]], device const uchar* old_values [[buffer(3)]],
    device uchar* cached_keys [[buffer(4)]], device uchar* cached_values [[buffer(5)]],
    constant QuantizedCacheArgs& a [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint groups = (a.depth+31)/32, block = group%groups;
  const ulong head_row = group/groups, head = head_row%a.heads;
  const uint rows = a.copy_past ? a.offset+a.queries : a.queries;
  const ulong batch = head_row/a.heads/rows;
  const uint token = uint(head_row/a.heads%rows)+(a.copy_past ? 0 : a.offset);
  const uint bytes = 4+4*a.bits;
  const ulong out = ((batch*a.capacity+token)*a.heads+head)*groups*bytes+block*bytes;
  if (a.copy_past && token < a.offset) {
    const ulong in = ((batch*a.old_capacity+token)*a.heads+head)*groups*bytes+block*bytes;
    for (uint i = lane; i < bytes; i += 32) {
      cached_keys[out+i] = old_keys[in+i]; cached_values[out+i] = old_values[in+i];
    }
    return;
  }
  const uint d = block*32+lane;
  const ulong in = ((batch*a.queries+token-a.offset)*a.heads+head)*a.depth+d;
  for (uint which = 0; which < 2; ++which) {
    const float x = d < a.depth ? float(which ? values[in] : keys[in]) : 0;
    const float maximum = simd_max(abs(x));
    const int zero = 1u<<(a.bits-1), limit = zero-1;
    const float reciprocal_limit = a.bits == 4 ? 1.f/7.f : a.bits == 5 ? 1.f/15.f : 1.f/127.f;
    const float scale = simd_any(!isfinite(x)) ? NAN : maximum == 0 ? 1.f : maximum*reciprocal_limit;
    // FP16 inputs make x*limit and n*maximum exact in FP32. Correct the
    // approximate quotient with an FMA remainder before nearest-even rounding;
    // reciprocal approximations must not choose the wrong side of a half tie.
    const float target = abs(x)*float(limit);
    const uint lower = maximum > 0 ? uint(floor(target/maximum)) : 0;
    const float remainder = fma(-float(lower),maximum,target), halfway = maximum*.5f;
    const uint rounded = min(uint(limit),lower+uint(remainder > halfway || (remainder == halfway && (lower&1))));
    const uint code = uint(x < 0 ? -int(rounded) : int(rounded))+zero;
    device uchar* dst = (which ? cached_values : cached_keys)+out;
    if (lane == 0) *(device float*)dst = scale;
    // Every lane must execute the shuffles, including lanes that do not
    // write a payload byte; their codes may be read by an active lane.
    const uint bit = lane*8, index = min(bit/a.bits,31u);
    uint word = simd_shuffle(code,index);
    const uint next = simd_shuffle(code,min(index+1,31u));
    const uint last = simd_shuffle(code,min(index+2,31u));
    word |= index+1 < 32 ? next<<a.bits : 0;
    word |= index+2 < 32 ? last<<(2*a.bits) : 0;
    if (lane < 4*a.bits) dst[4+lane] = uchar(word>>(bit%a.bits));
  }
}

struct UnpackCacheArgs { uint tokens, heads, depth, capacity, bits; };
kernel void unpack_quantized_kv(device const ushort* input [[buffer(0)]],
                                device half* output [[buffer(1)]],
                                constant UnpackCacheArgs& a [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  const ulong row = gid/a.depth, batch = row/a.heads/a.tokens;
  const ulong source_row = (batch*a.capacity+row/a.heads%a.tokens)*a.heads+row%a.heads;
  output[gid] = half(kv_cache_value<true>(input,source_row,gid%a.depth,a.depth,a.bits,false));
}
)METAL"
