// Packed weights and scales belong to the immutable model, not a global cache.
template<typename T>
static void gemm_int4_typed(const T* input, const int8_t* weight,
               const float* scales, const float16_t* bias, T* output,
               dim_t rows, dim_t columns, dim_t depth) {
  StreamGuard guard;
  PROFILE("NativeInt4Projection");
  if (rows <= 0 || columns <= 0 || depth <= 0 || rows > INT32_MAX ||
      columns > INT32_MAX || depth > INT32_MAX ||
      uint64_t(rows) > SIZE_MAX / uint64_t(columns) / sizeof(T) ||
      uint64_t(rows) > SIZE_MAX / uint64_t(depth) / sizeof(T) ||
      uint64_t(columns) >
          SIZE_MAX / ((uint64_t(depth) + 63) / 64) / sizeof(float))
    throw std::invalid_argument(
        "Unsupported native INT4 projection dimensions");
  id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
  const bool gemv = rows <= 8 || ![device supportsFamily:MTLGPUFamilyApple7];
  const char* kernel = sizeof(T) == 4
    ? (gemv ? "gemv_int4_f32" : "gemm_int4_f32")
    : (gemv ? "gemv_int4_f16" : "gemm_int4_f16");
  auto state = pipeline(kernel);
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < 128)
    throw std::invalid_argument("INT4 requires 32-lane Metal SIMD groups");
  struct Args {
    uint32_t m, n, k, has_bias, unused;
  };
  const Args args{uint32_t(rows), uint32_t(columns), uint32_t(depth),
                  bias ? 1u : 0u, 0};
  const void* pointers[] = {input, weight,
                            bias ? static_cast<const void*>(bias) : input,
                            output, scales};
  const size_t sizes[] = {
      size_t(rows) * depth * sizeof(T), size_t(columns) * ((depth + 1) / 2),
      bias ? size_t(columns) * 2 : 2, size_t(rows) * columns * sizeof(T),
      size_t(columns) * ((depth + 63) / 64) * sizeof(float)};
  id<MTLBuffer> buffers[5];
  NSUInteger offsets[5] = {};
  for (size_t i = 0; i < 5; ++i)
    buffers[i] = mtl_buffer(pointers[i], sizes[i], offsets[i]);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (NSUInteger i = 0; i < 4; ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder setBytes:&args length:sizeof(args) atIndex:4];
  [encoder setBuffer:buffers[4] offset:offsets[4] atIndex:5];
  [encoder dispatchThreadgroups:MTLSizeMake(gemv ? (columns + 15) / 16
                                                 : (columns + 31) / 32,
                                            gemv ? rows : (rows + 15) / 16, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(kernel);
  record_profile_detail("native_int4_fused_dequantization");
}

void gemm_int4(const float16_t* input, const int8_t* weight, const float* scales,
                 const float16_t* bias, float16_t* output,
                 dim_t rows, dim_t columns, dim_t depth) {
  gemm_int4_typed(input, weight, scales, bias, output, rows, columns, depth);
}

void gemm_int4_f32(const float* input, const int8_t* weight, const float* scales,
                 const float16_t* bias, float* output,
                 dim_t rows, dim_t columns, dim_t depth) {
  gemm_int4_typed(input, weight, scales, bias, output, rows, columns, depth);
}
