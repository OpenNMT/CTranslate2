void dense_f16_fp32(const float* input, const float16_t* weight,
                     const float16_t* bias, const float* residual, float* output,
                     dim_t rows, dim_t columns, dim_t depth) {
  StreamGuard guard;
  if (rows <= 0 || columns <= 0 || depth <= 0 || rows > INT32_MAX ||
      columns > INT32_MAX || depth > INT32_MAX ||
      uint64_t(rows) > SIZE_MAX / uint64_t(columns) / sizeof(float) ||
      uint64_t(rows) > SIZE_MAX / uint64_t(depth) / sizeof(float) ||
      uint64_t(columns) > SIZE_MAX / uint64_t(depth) / sizeof(float16_t))
    throw std::invalid_argument("Unsupported precise FP16 dense dimensions");
  id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
  const bool gemv = rows <= 8 || ![device supportsFamily:MTLGPUFamilyApple7];
  const char* kernel = gemv ? "precise_dense_gemv" : "precise_dense_gemm";
  auto state = pipeline(kernel);
  if (state.threadExecutionWidth != 32 || state.maxTotalThreadsPerThreadgroup < 128)
    throw std::invalid_argument("Precise FP16 dense requires 32-lane Metal SIMD groups");
  struct Args { uint32_t m, n, k, has_bias, has_residual; };
  const Args args{uint32_t(rows), uint32_t(columns), uint32_t(depth),
                  bias ? 1u : 0u, residual ? 1u : 0u};
  const void* pointers[] = {input, weight, output, bias ? bias : weight,
                            residual ? residual : input};
  const size_t sizes[] = {size_t(rows)*depth*sizeof(float),
                          size_t(columns)*depth*sizeof(float16_t),
                          size_t(rows)*columns*sizeof(float),
                          bias ? size_t(columns)*sizeof(float16_t) : sizeof(float16_t),
                          residual ? size_t(rows)*columns*sizeof(float) : sizeof(float)};
  id<MTLBuffer> buffers[5];
  NSUInteger offsets[5] = {};
  for (size_t i = 0; i < 5; ++i)
    buffers[i] = mtl_buffer(pointers[i], sizes[i], offsets[i]);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (NSUInteger i = 0; i < 3; ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder setBytes:&args length:sizeof(args) atIndex:3];
  [encoder setBuffer:buffers[3] offset:offsets[3] atIndex:4];
  [encoder setBuffer:buffers[4] offset:offsets[4] atIndex:5];
  [encoder dispatchThreadgroups:MTLSizeMake(gemv ? (columns+15)/16 : (columns+31)/32,
                                           gemv ? rows : (rows+15)/16, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(kernel);
}
