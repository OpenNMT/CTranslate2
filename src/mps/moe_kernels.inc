// Included inside ctranslate2::mps to use the shared Metal pipeline cache.
void moe_topk(const float* logits, int32_t* indices, float* weights, dim_t rows,
              dim_t experts, dim_t k, bool normalize_selected) {
  StreamGuard guard;
  if (rows < 0 || !fits_u32(rows) || experts <= 0 || experts > INT32_MAX ||
      k <= 0 || k > experts ||
      uint64_t(rows) > SIZE_MAX / uint64_t(experts) / sizeof(float))
    throw std::invalid_argument(
        "unsupported native MPS MoE routing dimensions");
  if (!rows)
    return;
  id<MTLComputePipelineState> state = pipeline("moe_topk_f32");
  const uint32_t threads = experts <= 32 ? 32 : 128;
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < threads)
    throw std::invalid_argument(
        "native MPS MoE routing requires 32-lane SIMD groups");
  struct Args {
    uint32_t rows, experts, k, threads, normalize_selected;
  };
  const Args args{uint32_t(rows), uint32_t(experts), uint32_t(k), threads,
                  normalize_selected ? 1u : 0u};
  const size_t input_bytes = size_t(rows) * experts * sizeof(float);
  const size_t output_bytes = size_t(rows) * k * sizeof(float);
  // Register every buffer before encoding, preserving asynchronous lifetimes.
  NSUInteger input_offset = 0, ids_offset = 0, probabilities_offset = 0;
  const auto input = mtl_buffer(logits, input_bytes, input_offset);
  const auto ids = mtl_buffer(indices, output_bytes, ids_offset);
  const auto probabilities =
      mtl_buffer(weights, output_bytes, probabilities_offset);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  [encoder setBuffer:input offset:input_offset atIndex:0];
  [encoder setBuffer:ids offset:ids_offset atIndex:1];
  [encoder setBuffer:probabilities offset:probabilities_offset atIndex:2];
  [encoder setBytes:&args length:sizeof(args) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch("moe_topk_f32");
  record_profile_detail("native_moe_topk");
}

bool moe_use_gemv(dim_t rows) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)get_device();
  return rows <= 8 || ![device supportsFamily:MTLGPUFamilyApple7];
}

static void
run_moe_groups(const char* kernel, dim_t groups,
               const std::vector<std::pair<const void*, size_t>>& inputs,
               const void* args, size_t args_size, NSUInteger args_index) {
  auto state = pipeline(kernel);
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < 128)
    throw std::invalid_argument(
        "MPS MoE dispatch requires 32-lane SIMD groups");
  std::vector<id<MTLBuffer>> buffers(inputs.size());
  std::vector<NSUInteger> offsets(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
    buffers[i] = mtl_buffer(inputs[i].first, inputs[i].second, offsets[i]);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (NSUInteger i = 0; i < inputs.size(); ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder setBytes:args length:args_size atIndex:args_index];
  [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(kernel);
}

void moe_router(DataType dtype, const void* input, const float* router,
                float* logits, dim_t rows, dim_t experts, dim_t hidden) {
  PROFILE("MoeRouterProjection");
  StreamGuard guard;
  const uint32_t storage = dtype == DataType::FLOAT32    ? 0
                           : dtype == DataType::FLOAT16  ? 1
                           : dtype == DataType::BFLOAT16 ? 2
                                                         : 3;
  if (storage == 3 || rows <= 0 || experts <= 0 || experts > INT32_MAX ||
      hidden <= 0 || !fits_u32(hidden) ||
      uint64_t(rows) > UINT32_MAX / ((uint64_t(experts) + 3) / 4) ||
      uint64_t(rows) > SIZE_MAX / uint64_t(hidden) / dtype_size(dtype) ||
      uint64_t(experts) > SIZE_MAX / uint64_t(hidden) / sizeof(float) ||
      uint64_t(rows) > SIZE_MAX / uint64_t(experts) / sizeof(float))
    throw std::invalid_argument(
        "Unsupported MPS MoE router projection dimensions");
  struct Args {
    uint32_t experts, hidden, dtype;
  };
  const Args args{uint32_t(experts), uint32_t(hidden), storage};
  run_moe_groups("moe_router", rows * ((experts + 3) / 4),
                 {{input, size_t(rows) * hidden * dtype_size(dtype)},
                  {router, size_t(experts) * hidden * sizeof(float)},
                  {logits, size_t(rows) * experts * sizeof(float)}},
                 &args, sizeof(args), 3);
  record_profile_detail("native_moe_router");
}

void moe_dispatch_plan(const int32_t* ids, int32_t* offsets,
                       int32_t* permutation, int32_t* inverse,
                       int32_t* indirect, dim_t rows, dim_t experts, dim_t topk,
                       dim_t hidden, dim_t intermediate, bool decode, bool wide,
                       bool force_generic) {
  PROFILE("MoeDispatchPlan");
  StreamGuard guard;
  if (rows <= 0 || experts <= 0 || topk <= 0 || topk > experts || hidden <= 0 ||
      intermediate <= 0 || !fits_u32(experts) || !fits_u32(hidden) ||
      !fits_u32(intermediate) || uint64_t(rows) > INT32_MAX / uint64_t(topk) ||
      uint64_t(rows) * topk >
          UINT32_MAX / uint64_t(std::max(hidden, intermediate)))
    throw std::invalid_argument("Unsupported MPS MoE dispatch dimensions");
  struct Args {
    uint32_t tokens, experts, topk, hidden, intermediate, decode;
  };
  const Args args{
      uint32_t(rows),         uint32_t(experts),
      uint32_t(topk),         uint32_t(hidden),
      uint32_t(intermediate), decode ? 1u : wide ? (rows < 64 ? 3u : 2u) : 0u};
  const size_t assignment_bytes = size_t(rows) * topk * sizeof(int32_t);
  const size_t offset_bytes = size_t(experts + 1) * sizeof(int32_t);
  static const bool disable_small = [] {
    const char* value = std::getenv("CT2_MPS_DISABLE_SMALL_MOE_PLAN");
    return value && value[0] && std::string(value) != "0";
  }();
  if (decode && rows * topk <= 128 && !force_generic && !disable_small) {
    run_moe_groups("moe_plan_small", experts,
                   {{ids, assignment_bytes},
                    {offsets, offset_bytes},
                    {permutation, assignment_bytes},
                    {inverse, assignment_bytes},
                    {indirect, size_t(experts) * 6 * sizeof(int32_t)}},
                   &args, sizeof(args), 5);
    record_profile_detail("native_moe_small_plan");
    return;
  }
  run_moe_groups("moe_count", experts,
                 {{ids, assignment_bytes}, {offsets, offset_bytes}}, &args,
                 sizeof(args), 2);
  run_1d("moe_offsets", 1,
         {{offsets, offset_bytes},
          {indirect, size_t(experts) * 6 * sizeof(int32_t)}},
         &args, sizeof(args), 2);
  run_moe_groups("moe_group", experts,
                 {{ids, assignment_bytes},
                  {offsets, offset_bytes},
                  {permutation, assignment_bytes},
                  {inverse, assignment_bytes}},
                 &args, sizeof(args), 4);
  record_profile_detail("native_moe_dispatch_plan");
}

void moe_expert_project(DataType dtype, const void* input, const void* gate,
                        const void* up, void* output, const int32_t* offsets,
                        const int32_t* permutation, const int32_t* indirect,
                        dim_t rows, dim_t experts, dim_t expert, dim_t topk,
                        dim_t n, dim_t k, bool gated, bool decode, bool wide,
                        const float* gate_scale, const float* up_scale) {
  PROFILE(gated ? "MoeExpertGateUp" : "MoeExpertDown");
  StreamGuard guard;
  const uint32_t storage = dtype == DataType::FLOAT32    ? 0
                           : dtype == DataType::FLOAT16  ? 1
                           : dtype == DataType::BFLOAT16 ? 2
                                                         : 3;
  if (storage == 3 || rows <= 0 || experts <= 0 || expert < 0 ||
      expert >= experts || topk <= 0 || topk > experts || n <= 0 || k <= 0 ||
      !fits_u32(n) || !fits_u32(k) ||
      uint64_t(n) > SIZE_MAX / uint64_t(k) / dtype_size(dtype) ||
      uint64_t(rows) > UINT32_MAX / uint64_t(topk) ||
      uint64_t(rows) * topk > UINT32_MAX / uint64_t(std::max(n, k)))
    throw std::invalid_argument(
        "Unsupported MPS MoE expert projection dimensions");
  if ((gate_scale && (dtype != DataType::FLOAT16 || (gated && !up_scale))) ||
      (!gate_scale && up_scale))
    throw std::invalid_argument("Invalid native INT4 expert scales");
  if (wide && (decode || dtype != DataType::FLOAT16))
    throw std::invalid_argument("Wide MPS MoE tiles require FP16 GEMM");
  const char* kernel = decode ? "moe_expert_gemv"
                       : wide && rows < 64
                           ? (gated ? "moe_expert_gate_up_f16_16x64"
                                    : "moe_expert_down_f16_16x64")
                       : wide ? (gated ? "moe_expert_gate_up_f16_64x64"
                                       : "moe_expert_down_f16_64x64")
                              : "moe_expert_gemm";
  id<MTLComputePipelineState> state = pipeline(kernel);
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < 128)
    throw std::invalid_argument(
        "MPS MoE expert kernels require 32-lane SIMD groups");
  struct Args {
    uint32_t expert, topk, n, k, dtype, gated, vector_loads, int4;
  };
  static const bool disable_vector = [] {
    const char* value = std::getenv("CT2_MPS_DISABLE_VECTOR_MOE");
    return value && value[0] && std::string(value) != "0";
  }();
  const Args args{uint32_t(expert),
                  uint32_t(topk),
                  uint32_t(n),
                  uint32_t(k),
                  storage,
                  gated ? 1u : 0u,
                  ((!decode || !disable_vector) && k % 8 == 0 &&
                   (reinterpret_cast<uintptr_t>(input) & 15u) == 0 &&
                   (reinterpret_cast<uintptr_t>(gate) & 15u) == 0 &&
                   (!gated || (reinterpret_cast<uintptr_t>(up) & 15u) == 0))
                      ? 1u
                      : 0u,
                  gate_scale ? 1u : 0u};
  const size_t item = dtype_size(dtype), assignments = size_t(rows) * topk;
  const size_t weight_bytes =
      gate_scale ? size_t(n) * ((k + 1) / 2) : size_t(n) * k * item;
  const size_t scale_bytes =
      gate_scale ? size_t(n) * ((k + 63) / 64) * sizeof(float) : sizeof(float);
  const void* pointers[] = {input,
                            gate,
                            gated ? up : gate,
                            output,
                            offsets,
                            permutation,
                            gate_scale ? static_cast<const void*>(gate_scale)
                                       : gate,
                            up_scale     ? static_cast<const void*>(up_scale)
                            : gate_scale ? static_cast<const void*>(gate_scale)
                                         : gate};
  const size_t sizes[] = {size_t(gated ? rows : assignments) * k * item,
                          weight_bytes,
                          weight_bytes,
                          assignments * n * item,
                          size_t(experts + 1) * sizeof(int32_t),
                          assignments * sizeof(int32_t),
                          scale_bytes,
                          scale_bytes};
  id<MTLBuffer> buffers[8];
  NSUInteger buffer_offsets[8] = {};
  for (size_t i = 0; i < 8; ++i)
    buffers[i] = mtl_buffer(pointers[i], sizes[i], buffer_offsets[i]);
  NSUInteger indirect_offset = 0;
  auto indirect_buffer = mtl_buffer(
      indirect, size_t(experts) * 6 * sizeof(int32_t), indirect_offset);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (NSUInteger i = 0; i < 6; ++i)
    [encoder setBuffer:buffers[i] offset:buffer_offsets[i] atIndex:i];
  [encoder setBytes:&args length:sizeof(args) atIndex:6];
  [encoder setBuffer:buffers[6] offset:buffer_offsets[6] atIndex:7];
  [encoder setBuffer:buffers[7] offset:buffer_offsets[7] atIndex:8];
  [encoder dispatchThreadgroupsWithIndirectBuffer:indirect_buffer
                             indirectBufferOffset:indirect_offset +
                                                  (size_t(expert) * 6 +
                                                   (gated ? 0 : 3)) *
                                                      sizeof(int32_t)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(kernel);
  record_profile_detail(gated ? "native_moe_gate_up_swiglu"
                              : "native_moe_down");
}

void moe_aggregate(DataType dtype, const void* expert_output,
                   const int32_t* inverse, const int32_t* ids,
                   const float* weights, void* output, dim_t rows,
                   dim_t experts, dim_t topk, dim_t hidden) {
  PROFILE("MoeAggregate");
  StreamGuard guard;
  struct Args {
    uint32_t tokens, experts, topk, hidden, intermediate, decode;
  };
  const Args args{uint32_t(rows),
                  uint32_t(experts),
                  uint32_t(topk),
                  uint32_t(hidden),
                  0,
                  0};
  const uint32_t storage = dtype == DataType::FLOAT32   ? 0
                           : dtype == DataType::FLOAT16 ? 1
                                                        : 2;
  // Bind model dtype separately from the dispatch-plan metadata.
  const size_t assignments = size_t(rows) * topk, item = dtype_size(dtype);
  const void* pointers[] = {expert_output, inverse, ids, weights, output};
  const size_t sizes[] = {
      assignments * hidden * item, assignments * sizeof(int32_t),
      assignments * sizeof(int32_t), assignments * sizeof(float),
      size_t(rows) * hidden * item};
  id<MTLBuffer> buffers[5];
  NSUInteger offsets[5] = {};
  for (size_t i = 0; i < 5; ++i)
    buffers[i] = mtl_buffer(pointers[i], sizes[i], offsets[i]);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:pipeline("moe_aggregate")];
  for (NSUInteger i = 0; i < 5; ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder setBytes:&args length:sizeof(args) atIndex:5];
  [encoder setBytes:&storage length:sizeof(storage) atIndex:6];
  [encoder dispatchThreads:MTLSizeMake(size_t(rows) * hidden, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch("moe_aggregate");
}

void moe_add_shared(DataType dtype, const void* shared, const void* gates,
                    void* output, dim_t rows, dim_t hidden) {
  StreamGuard guard;
  const uint32_t storage = dtype == DataType::FLOAT32   ? 0
                           : dtype == DataType::FLOAT16 ? 1
                                                        : 2;
  struct Args {
    uint32_t hidden, dtype;
  };
  const Args args{uint32_t(hidden), storage};
  run_1d("moe_add_shared", size_t(rows) * hidden,
         {{shared, size_t(rows) * hidden * dtype_size(dtype)},
          {gates, size_t(rows) * dtype_size(dtype)},
          {output, size_t(rows) * hidden * dtype_size(dtype)}},
         &args, sizeof(args), 3);
}
