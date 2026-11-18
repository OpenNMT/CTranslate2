// INT4 format: row-major low/high nibbles, signed value = nibble - 8,
// FP32 scale per 64 columns. No complete dequantized matrix is allocated.
R"METAL(
struct Int4Args { uint m,n,k,has_bias,unused; };
inline half int4_weight(device const uchar* packed, device const float* scales,
                        uint row, uint col, uint depth) {
  const uchar byte=packed[ulong(row)*((depth+1)/2)+col/2];
  const int q=int((byte >> ((col&1)*4))&15)-8;
  return half(float(q)*scales[ulong(row)*((depth+63)/64)+col/64]);
}
kernel void gemv_int4_f16(device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]], device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]], constant Int4Args& a [[buffer(4)]],
    device const float* scales [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row=group.y, col=(group.x*4+simd)*4;
  float sums[4]={0,0,0,0};
  for(uint k=lane;k<a.k;k+=32) {
    const float x=float(input[ulong(row)*a.k+k]);
    for(uint j=0;j<4;++j) if(col+j<a.n)
      sums[j]=fma(x,float(int4_weight(weight,scales,col+j,k,a.k)),sums[j]);
  }
  for(uint j=0;j<4;++j) {
    const float sum=simd_sum(sums[j]);
    if(lane==0&&col+j<a.n) output[ulong(row)*a.n+col+j]=half(sum+(a.has_bias?float(bias[col+j]):0));
  }
}
kernel void gemm_int4_f16(
    device const half* input [[buffer(0)]], device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]], device half* output [[buffer(3)]],
    constant Int4Args& a [[buffer(4)]], device const float* scales [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint LD=24;
  threadgroup half xs[16*LD],ws[32*LD];
  const uint row0=group.y*16,col0=group.x*32,sr=(simd>>1)*8,sc=(simd&1)*16;
  const uint quad=lane>>2,fr=(quad&4)+((lane>>1)&3),fc=(quad&2)*2+(lane&1)*2;
  simdgroup_matrix<float,8,8> accum[2];
  for(uint j=0;j<2;++j) accum[j].thread_elements()[0]=accum[j].thread_elements()[1]=0;
  for(uint start=0;start<a.k;start+=16) {
    for(uint i=tid;i<16*16;i+=128) {
      const uint r=row0+i/16,k=start+i%16;
      xs[i/16*LD+i%16]=r<a.m&&k<a.k?input[ulong(r)*a.k+k]:half(0);
    }
    for(uint i=tid;i<32*16;i+=128) {
      const uint c=col0+i/16,k=start+i%16;
      ws[i/16*LD+i%16]=c<a.n&&k<a.k?int4_weight(weight,scales,c,k,a.k):half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint ki=0;ki<2;++ki) {
      simdgroup_matrix<float,8,8> af;
      af.thread_elements()[0]=float(xs[(sr+fr)*LD+ki*8+fc]);
      af.thread_elements()[1]=float(xs[(sr+fr)*LD+ki*8+fc+1]);
      for(uint j=0;j<2;++j) {
        simdgroup_matrix<float,8,8> bf;
        const uint c=sc+j*8+fc,r=ki*8+fr;
        bf.thread_elements()[0]=float(ws[c*LD+r]);
        bf.thread_elements()[1]=float(ws[(c+1)*LD+r]);
        simdgroup_multiply_accumulate(accum[j],af,bf,accum[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row=row0+sr+fr;
  for(uint j=0;j<2;++j) for(uint el=0;el<2;++el) {
    const uint col=col0+sc+j*8+fc+el;
    if(row<a.m&&col<a.n) output[ulong(row)*a.n+col]=half(accum[j].thread_elements()[el]+(a.has_bias?float(bias[col]):0));
  }
}
kernel void gemv_int4_f32(device const float* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]], device const half* bias [[buffer(2)]],
    device float* output [[buffer(3)]], constant Int4Args& a [[buffer(4)]],
    device const float* scales [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row=group.y, col=(group.x*4+simd)*4;
  float sums[4]={0,0,0,0};
  for(uint k=lane;k<a.k;k+=32) {
    const float x=float(input[ulong(row)*a.k+k]);
    for(uint j=0;j<4;++j) if(col+j<a.n)
      sums[j]=fma(x,float(int4_weight(weight,scales,col+j,k,a.k)),sums[j]);
  }
  for(uint j=0;j<4;++j) {
    const float sum=simd_sum(sums[j]);
    if(lane==0&&col+j<a.n) output[ulong(row)*a.n+col+j]=sum+(a.has_bias?float(bias[col+j]):0);
  }
}
kernel void gemm_int4_f32(
    device const float* input [[buffer(0)]], device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]], device float* output [[buffer(3)]],
    constant Int4Args& a [[buffer(4)]], device const float* scales [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr uint LD=24;
  threadgroup float xs[16*LD];
  threadgroup half ws[32*LD];
  const uint row0=group.y*16,col0=group.x*32,sr=(simd>>1)*8,sc=(simd&1)*16;
  const uint quad=lane>>2,fr=(quad&4)+((lane>>1)&3),fc=(quad&2)*2+(lane&1)*2;
  simdgroup_matrix<float,8,8> accum[2];
  for(uint j=0;j<2;++j) accum[j].thread_elements()[0]=accum[j].thread_elements()[1]=0;
  for(uint start=0;start<a.k;start+=16) {
    for(uint i=tid;i<16*16;i+=128) {
      const uint r=row0+i/16,k=start+i%16;
      xs[i/16*LD+i%16]=r<a.m&&k<a.k?input[ulong(r)*a.k+k]:half(0);
    }
    for(uint i=tid;i<32*16;i+=128) {
      const uint c=col0+i/16,k=start+i%16;
      ws[i/16*LD+i%16]=c<a.n&&k<a.k?int4_weight(weight,scales,c,k,a.k):half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint ki=0;ki<2;++ki) {
      simdgroup_matrix<float,8,8> af;
      af.thread_elements()[0]=float(xs[(sr+fr)*LD+ki*8+fc]);
      af.thread_elements()[1]=float(xs[(sr+fr)*LD+ki*8+fc+1]);
      for(uint j=0;j<2;++j) {
        simdgroup_matrix<float,8,8> bf;
        const uint c=sc+j*8+fc,r=ki*8+fr;
        bf.thread_elements()[0]=float(ws[c*LD+r]);
        bf.thread_elements()[1]=float(ws[(c+1)*LD+r]);
        simdgroup_multiply_accumulate(accum[j],af,bf,accum[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint row=row0+sr+fr;
  for(uint j=0;j<2;++j) for(uint el=0;el<2;++el) {
    const uint col=col0+sc+j*8+fc+el;
    if(row<a.m&&col<a.n) output[ulong(row)*a.n+col]=accum[j].thread_elements()[el]+(a.has_bias?float(bias[col]):0);
  }
}
)METAL"
