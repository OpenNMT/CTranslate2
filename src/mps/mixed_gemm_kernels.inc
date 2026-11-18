void gemm_f16_f32_output(const float16_t* input, const float16_t* weight,
                         const float16_t* bias, float* output, dim_t rows,
                         dim_t columns, dim_t depth) {
  StreamGuard guard;
  PROFILE("PreciseVocabularyProjection");
  if (rows <= 0 || columns <= 0 || depth <= 0 || rows > INT32_MAX ||
      columns > INT32_MAX || depth > INT32_MAX ||
      uint64_t(rows) > SIZE_MAX / uint64_t(columns) / sizeof(float) ||
      uint64_t(rows) > SIZE_MAX / uint64_t(depth) / sizeof(float16_t) ||
      uint64_t(columns) > SIZE_MAX / uint64_t(depth) / sizeof(float16_t))
    throw std::invalid_argument(
        "Unsupported precise MPS projection dimensions");
  const bool gemv = rows <= 2;
  const auto kernel = gemv ? "gemv_f16_f32_output" : "gemm_f16_f32_output";
  auto state = pipeline(kernel);
  const NSUInteger simds = gemv && columns >= 4096 ? 8 : 4;
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < simds * 32)
    throw std::invalid_argument(
        "Precise MPS projection requires 32-lane SIMD groups");
  struct Args {
    uint32_t m, n, k, has_bias, vector_loads;
  };
  const Args args{
      uint32_t(rows), uint32_t(columns), uint32_t(depth), bias ? 1u : 0u,
      (depth % 4 == 0 && (reinterpret_cast<uintptr_t>(input) & 7u) == 0 &&
       (reinterpret_cast<uintptr_t>(weight) & 7u) == 0)
          ? 1u
          : 0u};
  const void* pointers[] = {
      input, weight, bias ? static_cast<const void*>(bias) : input, output};
  const size_t sizes[] = {size_t(rows) * depth * sizeof(float16_t),
                          size_t(columns) * depth * sizeof(float16_t),
                          bias ? size_t(columns) * sizeof(float16_t)
                               : sizeof(float16_t),
                          size_t(rows) * columns * sizeof(float)};
  id<MTLBuffer> buffers[4];
  NSUInteger offsets[4] = {};
  for (size_t i = 0; i < 4; ++i)
    buffers[i] = mtl_buffer(pointers[i], sizes[i], offsets[i]);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (NSUInteger i = 0; i < 4; ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder setBytes:&args length:sizeof(args) atIndex:4];
  [encoder dispatchThreadgroups:MTLSizeMake(gemv ? (columns + simds * 4 - 1) /
                                                       (simds * 4)
                                                 : (columns + 31) / 32,
                                            gemv ? rows : (rows + 15) / 16, 1)
          threadsPerThreadgroup:MTLSizeMake(simds * 32, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(kernel);
  record_profile_detail("fp32_vocabulary_logits");
}
