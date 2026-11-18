R"METAL(
struct MixedOutputArgs { uint m,n,k,has_bias,vector_loads; };
// Preserve FP32 accumulators through the vocabulary output. FP16 storage here
// can collapse distinct top logits into an artificial greedy tie.
kernel void gemv_f16_f32_output(
    device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]], device float* output [[buffer(3)]],
    constant MixedOutputArgs& a [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]]) {
  const uint row=group.y, col=(group.x*simds+simd)*4;
  float sums[4]={0,0,0,0};
  if (a.vector_loads) {
    device const half4* x=(device const half4*)(input+ulong(row)*a.k);
    for(uint k=lane;k<a.k/4;k+=32) {
      const float4 v=float4(x[k]);
      #pragma clang loop unroll(full)
      for(uint j=0;j<4;++j) if(col+j<a.n)
        sums[j]+=dot(v,float4(((device const half4*)(weight+ulong(col+j)*a.k))[k]));
    }
  } else {
    for(uint k=lane;k<a.k;k+=32) {
      const float x=float(input[ulong(row)*a.k+k]);
      #pragma clang loop unroll(full)
      for(uint j=0;j<4;++j) if(col+j<a.n)
        sums[j]=fma(x,float(weight[ulong(col+j)*a.k+k]),sums[j]);
    }
  }
  #pragma clang loop unroll(full)
  for(uint j=0;j<4;++j) {
    const float sum=simd_sum(sums[j]);
    if(lane==0 && col+j<a.n) output[ulong(row)*a.n+col+j]=sum+(a.has_bias?float(bias[col+j]):0);
  }
}
kernel void gemm_f16_f32_output(
    device const half* input [[buffer(0)]], device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]], device float* output [[buffer(3)]],
    constant MixedOutputArgs& a [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
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
      ws[i/16*LD+i%16]=c<a.n&&k<a.k?weight[ulong(c)*a.k+k]:half(0);
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
