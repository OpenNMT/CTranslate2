// Measured decoder shape: D=128, compact GQA/MQA, one query. Vector loads
// amortize packed-code extraction and scale loads across four adjacent values.
R"METAL(
template<uint BITS>
inline float4 flash_cache_vector(device const ushort* data, ulong row, uint lane) {
  if (BITS == 0) return float4(((device const half4*)data)[row*32+lane]);
  device const uchar* block = (device const uchar*)data
    + row*(4*(4+4*BITS))+(lane/8)*(4+4*BITS);
  device const uchar* payload = block+4+(lane%8)*(4*BITS)/8;
  uint packed;
  if (BITS == 8) packed = *(device const uint*)payload;
  else if (BITS == 4) packed = *(device const ushort*)payload;
  else packed = uint(payload[0]) | (uint(payload[1])<<8) | (uint(payload[2])<<16);
  packed >>= ((lane%8)*4*BITS)%8;
  const uint mask = (1u<<BITS)-1, zero = 1u<<(BITS ? BITS-1 : 0);
  const int4 codes = int4(packed&mask,(packed>>BITS)&mask,
                          (packed>>(2*BITS))&mask,(packed>>(3*BITS))&mask)-int(zero);
  return float4(half4(float4(codes)*(*(device const float*)block)));
}

template<uint BITS>
inline void flash_decode_vector(
    device const half4* q, device const ushort* k, device const ushort* v,
    device const int* lengths, device float* partials, device half4* output,
    constant FlashDecodeArgs& a, uint2 group, uint tid, uint lane, uint simd,
    threadgroup float* p, threadgroup float* maxima, threadgroup float* sums,
    threadgroup float4* pv) {
  const uint head = group.y%a.heads, batch = group.y/a.heads;
  const uint kvhead = head/(a.heads/a.kv_heads), start = a.key_start+group.x*256;
  const uint position = a.offset;
  uint valid = a.keys;
  if (a.has_lengths) valid = uint(clamp(lengths[group.y],0,int(a.keys)));
  if (a.causal) valid = min(valid,position+1);
  const uint first = a.window && position+1 > a.window ? position+1-a.window : 0;
  const ulong kvbase = ulong(batch)*a.capacity*a.kv_heads+kvhead;
  const float4 query = float4(q[ulong(group.y)*32+lane]);
  for (uint i = 0; i < 64; ++i) {
    const uint token = start+simd+i*4;
    float score = 0;
    if (token >= first && token < valid)
      score = dot(query,flash_cache_vector<BITS>(k,kvbase+ulong(token)*a.kv_heads,lane));
    score = simd_sum(score);
    if (lane == 0) p[simd+i*4] = token >= first && token < valid ? score*a.scale : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float x = p[tid], y = p[tid+128];
  const float m = simd_max(max(x,y));
  if (lane == 0) maxima[simd] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float maximum = max(max(maxima[0],maxima[1]),max(maxima[2],maxima[3]));
  const float px = x == -INFINITY ? 0 : exp(x-maximum), py = y == -INFINITY ? 0 : exp(y-maximum);
  // This barrier also protects score reads before p is reused for probabilities.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  p[tid] = px; p[tid+128] = py;
  const float sum = simd_sum(px+py);
  if (lane == 0) sums[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 acc(0.f);
  for (uint token = max(start+simd*64,first); token < min(start+(simd+1)*64,valid); ++token)
    acc = fma(float4(p[token-start]),flash_cache_vector<BITS>(v,kvbase+ulong(token)*a.kv_heads,lane),acc);
  pv[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const float denominator = sums[0]+sums[1]+sums[2]+sums[3];
    const float4 numerator = ((pv[lane]+pv[32+lane])+pv[64+lane])+pv[96+lane];
    const ulong base = (ulong(group.y)*a.splits+group.x)*130;
    if (a.splits > 1) {
      for (uint i = 0; i < 4; ++i) partials[base+lane*4+i] = numerator[i];
      if (lane == 0) { partials[base+128] = maximum; partials[base+129] = denominator; }
    } else output[ulong(group.y)*32+lane] = denominator > 0 ? half4(numerator/denominator) : half4(0);
  }
}
#define FLASH_VECTOR_ENTRY(NAME, BITS) \
kernel void NAME( \
    device const half4* q [[buffer(0)]], device const ushort* k [[buffer(1)]], \
    device const ushort* v [[buffer(2)]], device const int* lengths [[buffer(3)]], \
    device float* partials [[buffer(4)]], device half4* output [[buffer(5)]], \
    constant FlashDecodeArgs& a [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float p[256], maxima[4], sums[4]; threadgroup float4 pv[128]; \
  flash_decode_vector<BITS>(q,k,v,lengths,partials,output,a,group,tid,lane,simd,p,maxima,sums,pv); \
}
FLASH_VECTOR_ENTRY(flash_decode_vector_f16,0)
FLASH_VECTOR_ENTRY(flash_decode_vector_int8,8)
FLASH_VECTOR_ENTRY(flash_decode_vector_int5,5)
FLASH_VECTOR_ENTRY(flash_decode_vector_int4,4)
)METAL"
