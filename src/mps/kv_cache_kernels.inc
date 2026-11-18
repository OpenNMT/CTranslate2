// Included in ctranslate2::mps. Packing, growth and unpacking stay on the GPU.
void append_quantized_kv(const void* keys, const void* values,
                         const void* old_keys, const void* old_values,
                         void* cached_keys, void* cached_values, dim_t batch,
                         dim_t queries, dim_t heads, dim_t depth, dim_t offset,
                         dim_t old_capacity, dim_t capacity, bool copy_past,
                         unsigned bits) {
  StreamGuard guard;
  const auto width = packed_kv_row_bytes(depth, bits);
  if (batch <= 0 || heads <= 0 || queries <= 0 || offset < 0 ||
      old_capacity < 0 || capacity < queries || offset > capacity - queries ||
      (copy_past && offset > old_capacity) || !fits_u32(batch) ||
      !fits_u32(heads) || !fits_u32(capacity) || !fits_u32(old_capacity) ||
      uint64_t(batch) * heads > UINT32_MAX ||
      uint64_t(batch) * heads * capacity > UINT32_MAX / uint64_t(width) ||
      !keys || !values || !cached_keys || !cached_values ||
      (copy_past && offset && (!old_keys || !old_values)))
    throw std::invalid_argument("invalid quantized MPS KV cache append");
  struct Args {
    uint32_t batch, queries, heads, depth, offset, old_capacity, capacity,
        copy_past, bits;
  };
  const Args args{
      uint32_t(batch),    uint32_t(queries),   uint32_t(heads),
      uint32_t(depth),    uint32_t(offset),    uint32_t(old_capacity),
      uint32_t(capacity), copy_past ? 1u : 0u, bits};
  const size_t input_bytes =
      size_t(batch) * queries * heads * depth * sizeof(uint16_t);
  const size_t cache_bytes = size_t(batch) * capacity * heads * width;
  const bool read_old = copy_past && offset;
  const size_t old_bytes =
      read_old ? size_t(batch) * old_capacity * heads * width : input_bytes;
  const void* pointers[] = {keys,
                            values,
                            read_old ? old_keys : keys,
                            read_old ? old_values : values,
                            cached_keys,
                            cached_values};
  const size_t sizes[] = {input_bytes, input_bytes, old_bytes,
                          old_bytes,   cache_bytes, cache_bytes};
  auto state = pipeline("append_quantized_kv");
  if (state.threadExecutionWidth != 32 ||
      state.maxTotalThreadsPerThreadgroup < 32)
    throw std::runtime_error(
        "Quantized MPS KV caches require 32-lane SIMD groups");
  auto encoder = (__bridge id<MTLComputeCommandEncoder>)compute_encoder();
  [encoder setComputePipelineState:state];
  for (size_t i = 0; i < 6; ++i)
    set_buffer(encoder, pointers[i], sizes[i], i);
  [encoder setBytes:&args length:sizeof(args) atIndex:6];
  const size_t rows = copy_past ? size_t(offset) + queries : size_t(queries);
  [encoder dispatchThreadgroups:MTLSizeMake(size_t(batch) * rows * heads *
                                                ((depth + 31) / 32),
                                            1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  record_compute_dispatch("append_quantized_kv");
}

void unpack_quantized_kv(const void* input, void* output, dim_t batch,
                         dim_t tokens, dim_t heads, dim_t depth, dim_t capacity,
                         unsigned bits) {
  StreamGuard guard;
  const auto width = packed_kv_row_bytes(depth, bits);
  if (batch <= 0 || heads <= 0 || tokens <= 0 || capacity < tokens ||
      !fits_u32(batch) || !fits_u32(heads) || !fits_u32(capacity) ||
      uint64_t(batch) * heads > UINT32_MAX ||
      uint64_t(batch) * heads * capacity >
          UINT32_MAX / uint64_t(std::max(width, depth)) ||
      !input || !output || input == output)
    throw std::invalid_argument("invalid quantized MPS KV cache unpack");
  struct Args {
    uint32_t tokens, heads, depth, capacity, bits;
  };
  const Args args{uint32_t(tokens), uint32_t(heads), uint32_t(depth),
                  uint32_t(capacity), bits};
  const size_t count = size_t(batch) * tokens * heads * depth;
  run_1d("unpack_quantized_kv", count,
         {{input, size_t(batch) * capacity * heads * width},
          {output, count * sizeof(uint16_t)}},
         &args, sizeof(args), 2);
}
