// Included inside ctranslate2::mps to share the backend's pipeline/lifetime
// machinery. All buffers are registered before encoding; no host reads occur.
void append_compact_kv(const void* keys, const void* values,
                       const void* old_keys, const void* old_values,
                       void* cached_keys, void* cached_values, dim_t batch,
                       dim_t queries, dim_t heads, dim_t depth, dim_t offset,
                       dim_t old_capacity, dim_t capacity, bool copy_past) {
  StreamGuard guard;
  if (batch <= 0 || queries <= 0 || heads <= 0 || depth <= 0 || offset < 0 ||
      capacity < queries || offset > capacity - queries || old_capacity < 0 ||
      (copy_past && offset > old_capacity) || !fits_u32(batch) ||
      !fits_u32(queries) || !fits_u32(heads) || !fits_u32(depth) ||
      !fits_u32(capacity) || !fits_u32(old_capacity) ||
      uint64_t(batch) * heads > UINT32_MAX ||
      uint64_t(capacity) * depth > UINT32_MAX ||
      uint64_t(batch) * heads * (uint64_t(capacity) * depth) > UINT32_MAX)
    throw std::invalid_argument("invalid compact MPS KV cache dimensions");
  struct Args {
    uint32_t batch, queries, heads, depth, offset, old_capacity, capacity,
        copy_past;
  };
  const Args args{uint32_t(batch),    uint32_t(queries),
                  uint32_t(heads),    uint32_t(depth),
                  uint32_t(offset),   uint32_t(old_capacity),
                  uint32_t(capacity), copy_past ? 1u : 0u};
  const size_t width = size_t(heads) * depth * sizeof(uint16_t);
  const size_t input_bytes = size_t(batch) * queries * width;
  const size_t cache_bytes = size_t(batch) * capacity * width;
  const bool read_old = copy_past && offset > 0;
  run_1d("append_compact_kv",
         size_t(batch) * (copy_past ? offset + queries : queries) * heads *
             depth,
         {{keys, input_bytes},
          {values, input_bytes},
          {read_old ? old_keys : keys,
           read_old ? size_t(batch) * old_capacity * width : input_bytes},
          {read_old ? old_values : values,
           read_old ? size_t(batch) * old_capacity * width : input_bytes},
          {cached_keys, cache_bytes},
          {cached_values, cache_bytes}},
         &args, sizeof(args), 6);
  record_profile_detail(copy_past ? "compact_kv_growth" : "compact_kv_append");
}

bool flash_decode(DataType dtype, const void* queries, const void* keys,
                  const void* values, const int32_t* lengths, void* output,
                  float* workspace, size_t workspace_floats, dim_t batch,
                  dim_t heads, dim_t kv_heads, dim_t keys_length,
                  dim_t capacity, dim_t depth, float scale, dim_t offset,
                  dim_t window, bool causal, dim_t query_length,
                  bool kv_interleaved, bool queries_interleaved,
                  unsigned cache_bits) {
  StreamGuard guard;
  if ((dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) ||
      batch <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads ||
      keys_length <= 0 || keys_length > INT32_MAX || capacity < keys_length ||
      depth <= 0 || depth > 256 || offset < 0 || !fits_u32(offset) ||
      query_length <= 0 || !fits_u32(query_length) ||
      uint64_t(offset) + query_length >= UINT32_MAX || window < 0 ||
      !fits_u32(window) || (window && !causal) || !std::isfinite(scale) ||
      !queries || !keys || !values || !output || !fits_u32(batch) ||
      !fits_u32(heads) || !fits_u32(kv_heads) ||
      uint64_t(batch) * heads > UINT32_MAX / uint64_t(query_length) ||
      uint64_t(capacity) * depth > UINT32_MAX)
    return false;
  if (cache_bits && (dtype != DataType::FLOAT16 || !kv_interleaved ||
                     (cache_bits != 4 && cache_bits != 5 && cache_bits != 8)))
    return false;
  const size_t key_start =
      causal && window && query_length == 1 && offset + 1 > window
          ? size_t(offset + 1 - window) / 256 * 256
          : 0;
  const size_t splits =
      (size_t(keys_length) - std::min(size_t(keys_length), key_start) + 255) /
      256;
  if (!splits)
    return false;
  const size_t matrices = size_t(batch) * heads * query_length;
  if (matrices > SIZE_MAX / splits / (size_t(depth) + 2) / sizeof(float) ||
      uint64_t(batch) * kv_heads >
          SIZE_MAX / (uint64_t(capacity) * depth) / sizeof(uint16_t))
    return false;
  const size_t required = splits * matrices * (depth + 2);
  if (splits > 1 && (!workspace || workspace_floats < required))
    return false;
  const size_t qbytes = matrices * depth * sizeof(uint16_t);
  const size_t kvbytes = size_t(batch) * kv_heads * capacity *
                         (cache_bits ? packed_kv_row_bytes(depth, cache_bits)
                                     : depth * sizeof(uint16_t));
  const auto overlaps = [&](const void* source, size_t bytes) {
    const auto a = reinterpret_cast<uintptr_t>(source),
               b = reinterpret_cast<uintptr_t>(output);
    return a <= b ? b - a < bytes : a - b < qbytes;
  };
  if (overlaps(keys, kvbytes) || overlaps(values, kvbytes) ||
      (queries != output && overlaps(queries, qbytes)))
    return false;
  static const bool disable_cooperative_qk = [] {
    const char* value = std::getenv("CT2_MPS_DISABLE_COOPERATIVE_QK");
    return value && value[0] && std::string(value) != "0";
  }();
  static const bool disable_vector_cache = [] {
    const char* value = std::getenv("CT2_MPS_DISABLE_VECTOR_CACHE");
    return value && value[0] && std::string(value) != "0";
  }();
  const bool vector = !disable_vector_cache && kv_interleaved && depth == 128 &&
                      dtype == DataType::FLOAT16 && query_length == 1 &&
                      keys_length >= 1024 &&
                      !(reinterpret_cast<uintptr_t>(queries) % 8) &&
                      !(reinterpret_cast<uintptr_t>(keys) % 8) &&
                      !(reinterpret_cast<uintptr_t>(values) % 8) &&
                      !(reinterpret_cast<uintptr_t>(output) % 8);
  const char* kernel =
      vector       ? (cache_bits == 4   ? "flash_decode_vector_int4"
                      : cache_bits == 5 ? "flash_decode_vector_int5"
                      : cache_bits == 8 ? "flash_decode_vector_int8"
                                        : "flash_decode_vector_f16")
      : cache_bits ? "flash_decode_quantized"
      : kv_interleaved && depth >= 64 && !disable_cooperative_qk
          ? "flash_decode_partial_cooperative"
          : "flash_decode_partial";
  id<MTLComputePipelineState> state = pipeline(kernel);
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < 128)
    return false;
  struct Args {
    uint32_t batch, heads, kv_heads, keys, capacity, depth, splits;
    uint32_t offset, window, has_lengths, causal, bf16;
    uint32_t queries, kv_interleaved, queries_interleaved, cache_bits,
        key_start;
    float scale;
  };
  const Args args{uint32_t(batch),
                  uint32_t(heads),
                  uint32_t(kv_heads),
                  uint32_t(keys_length),
                  uint32_t(capacity),
                  uint32_t(depth),
                  uint32_t(splits),
                  uint32_t(offset),
                  uint32_t(window),
                  lengths ? 1u : 0u,
                  causal ? 1u : 0u,
                  dtype == DataType::BFLOAT16 ? 1u : 0u,
                  uint32_t(query_length),
                  kv_interleaved ? 1u : 0u,
                  queries_interleaved ? 1u : 0u,
                  cache_bits,
                  uint32_t(key_start),
                  scale};
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  set_buffer(encoder, queries, qbytes, 0);
  set_buffer(encoder, keys, kvbytes, 1);
  set_buffer(encoder, values, kvbytes, 2);
  set_buffer(encoder, lengths ? static_cast<const void*>(lengths) : queries,
             lengths ? matrices * sizeof(int32_t) : sizeof(uint16_t), 3);
  set_buffer(encoder,
             splits > 1 ? static_cast<const void*>(workspace) : queries,
             splits > 1 ? required * sizeof(float) : sizeof(uint16_t), 4);
  set_buffer(encoder, output, qbytes, 5);
  [encoder setBytes:&args length:sizeof(args) atIndex:6];
  [encoder dispatchThreadgroups:MTLSizeMake(splits, matrices, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(kernel);
  if (splits > 1) {
    encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
    [encoder setComputePipelineState:pipeline("flash_decode_merge")];
    set_buffer(encoder, workspace, required * sizeof(float), 0);
    set_buffer(encoder, output, qbytes, 1);
    [encoder setBytes:&args length:sizeof(args) atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(matrices, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    record_compute_dispatch("flash_decode_merge");
  }
  record_profile_detail("native_flash_decode");
  return true;
}

bool flash_prefill(DataType dtype, const void* queries, const void* keys,
                   const void* values, const int32_t* lengths, void* output,
                   dim_t batch, dim_t heads, dim_t kv_heads,
                   dim_t queries_length, dim_t keys_length, dim_t capacity,
                   dim_t depth, float scale, dim_t offset, dim_t window,
                   bool causal, bool interleaved_lengths, unsigned cache_bits) {
  StreamGuard guard;
  static const bool disabled = []() {
    const char* value = std::getenv("CT2_MPS_DISABLE_FLASH_PREFILL");
    return value && value[0] && value[0] != '0';
  }();
  if ((cache_bits &&
       (dtype != DataType::FLOAT16 ||
        (cache_bits != 4 && cache_bits != 5 && cache_bits != 8))) ||
      disabled || (dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) ||
      batch <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads ||
      queries_length <= 1 || queries_length > INT32_MAX || keys_length <= 0 ||
      keys_length > INT32_MAX || capacity < keys_length || depth <= 0 ||
      depth > 256 || offset < 0 || window < 0 || (window && !causal) ||
      uint64_t(offset) + queries_length > INT32_MAX || !fits_u32(capacity) ||
      !fits_u32(window) || !std::isfinite(scale) || !queries || !keys ||
      !values || !output || uint64_t(batch) > UINT32_MAX / uint64_t(heads) ||
      uint64_t(batch) * heads >
          SIZE_MAX / uint64_t(queries_length) / depth / sizeof(uint16_t) ||
      uint64_t(batch) * heads >
          SIZE_MAX / uint64_t(queries_length) / sizeof(int32_t) ||
      uint64_t(batch) * kv_heads >
          SIZE_MAX / uint64_t(capacity) / depth / sizeof(uint16_t))
    return false;
  auto device = (__bridge id<MTLDevice>)get_device();
  if (![device supportsFamily:MTLGPUFamilyApple7])
    return false;
  const size_t qbytes =
      size_t(batch) * queries_length * heads * depth * sizeof(uint16_t);
  const size_t kvbytes = size_t(batch) * capacity * kv_heads *
                         (cache_bits ? packed_kv_row_bytes(depth, cache_bits)
                                     : depth * sizeof(uint16_t));
  const auto overlaps = [&](const void* source, size_t bytes) {
    const auto a = reinterpret_cast<uintptr_t>(source),
               b = reinterpret_cast<uintptr_t>(output);
    return a <= b ? b - a < bytes : a - b < qbytes;
  };
  if (overlaps(keys, kvbytes) || overlaps(values, kvbytes) ||
      (queries != output && overlaps(queries, qbytes)))
    return false;
  const char* kernel = depth <= 64    ? "flash_prefill_64"
                       : depth <= 128 ? "flash_prefill_128"
                                      : "flash_prefill_256";
  const std::string selected_kernel =
      cache_bits ? std::string("flash_prefill_quantized_") +
                       std::to_string(depth <= 64    ? 64
                                      : depth <= 128 ? 128
                                                     : 256)
                 : kernel;
  auto state = pipeline(selected_kernel);
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < 128)
    return false;
  struct Args {
    uint32_t queries, keys, capacity, heads, kv_heads, depth;
    uint32_t offset, window, causal, has_lengths, interleaved_lengths, bf16,
        cache_bits;
    float scale;
  };
  const Args args{uint32_t(queries_length),
                  uint32_t(keys_length),
                  uint32_t(capacity),
                  uint32_t(heads),
                  uint32_t(kv_heads),
                  uint32_t(depth),
                  uint32_t(offset),
                  uint32_t(window),
                  causal ? 1u : 0u,
                  lengths ? 1u : 0u,
                  interleaved_lengths ? 1u : 0u,
                  dtype == DataType::BFLOAT16 ? 1u : 0u,
                  cache_bits,
                  scale};
  const void* pointers[] = {
      queries, keys, values,
      lengths ? static_cast<const void*>(lengths) : queries, output};
  const size_t sizes[] = {qbytes, kvbytes, kvbytes,
                          lengths ? size_t(batch) * heads * queries_length *
                                        sizeof(int32_t)
                                  : sizeof(uint16_t),
                          qbytes};
  id<MTLBuffer> buffers[5];
  NSUInteger offsets[5] = {};
  for (size_t i = 0; i < 5; ++i)
    buffers[i] = mtl_buffer(pointers[i], sizes[i], offsets[i]);
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (NSUInteger i = 0; i < 5; ++i)
    [encoder setBuffer:buffers[i] offset:offsets[i] atIndex:i];
  [encoder setBytes:&args length:sizeof(args) atIndex:5];
  [encoder dispatchThreadgroups:MTLSizeMake((queries_length + 31) / 32,
                                            batch * heads, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch(selected_kernel.c_str());
  record_profile_detail("native_flash_prefill");
  return true;
}

void mask_flash_scores(DataType dtype, void* scores, const int32_t* lengths,
                       int32_t* bounds, dim_t rows, dim_t heads, dim_t queries,
                       dim_t keys, dim_t offset, dim_t window, bool causal,
                       bool interleaved_lengths) {
  StreamGuard guard;
  if ((dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) ||
      rows <= 0 || heads <= 0 || queries <= 0 || keys <= 0 ||
      keys > INT32_MAX || offset < 0 || window < 0 || !fits_u32(window) ||
      uint64_t(offset) + queries >= UINT32_MAX ||
      uint64_t(rows) > UINT32_MAX / uint64_t(keys))
    throw std::invalid_argument(
        "unsupported materialized MPS flash attention mask");
  struct Args {
    uint32_t queries, keys, offset, window, has_lengths, causal, bf16, heads,
        interleaved_lengths;
  };
  const Args args{uint32_t(queries),
                  uint32_t(keys),
                  uint32_t(offset),
                  uint32_t(window),
                  lengths ? 1u : 0u,
                  causal ? 1u : 0u,
                  dtype == DataType::BFLOAT16 ? 1u : 0u,
                  uint32_t(heads),
                  interleaved_lengths ? 1u : 0u};
  run_1d("mask_flash_scores", size_t(rows) * keys,
         {{scores, size_t(rows) * keys * sizeof(uint16_t)},
          {lengths ? static_cast<const void*>(lengths) : scores,
           lengths ? size_t(rows) * sizeof(int32_t) : sizeof(uint16_t)},
          {bounds, size_t(rows) * sizeof(int32_t)}},
         &args, sizeof(args), 3);
}
