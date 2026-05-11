#include "ctranslate2/ops/flash_attention.h"
#if defined(CT2_WITH_FLASH_ATTN) && !defined(CT2_USE_HIP)
#include "ctranslate2/ops/flash-attention/flash.h"
#include "ctranslate2/ops/flash-attention/static_switch.h"
#endif
#include "ctranslate2/ops/transpose.h"
#include "cuda/utils.h"
#include "cuda/helpers.h"

#include "dispatch.h"

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

namespace ctranslate2 {
  namespace ops {

#ifndef CT2_USE_HIP  // CUDA-only: CUTLASS/CuTe Flash Attention kernels

#ifdef CT2_WITH_FLASH_ATTN
    static void set_params_fprop(Flash_fwd_params &params,
      // sizes
                                 const size_t b,
                                 const size_t seqlen_q,
                                 const size_t seqlen_k,
                                 const size_t seqlen_q_rounded,
                                 const size_t seqlen_k_rounded,
                                 const size_t h,
                                 const size_t h_k,
                                 const size_t d,
                                 const size_t d_rounded,
      // device pointers
                                 StorageView* q,
                                 StorageView* k,
                                 StorageView* v,
                                 StorageView* out,
                                 void *cu_seqlens_q_d,
                                 void *cu_seqlens_k_d,
                                 void *seqused_k,
                                 void *p_d,
                                 void *softmax_lse_d,
                                 float softmax_scale,
                                 int window_size_left,
                                 int window_size_right,
                                 bool seqlenq_ngroups_swapped=false) {

      // Reset the parameters
      memset(&params, 0, sizeof(params));

      params.is_bf16 = q->dtype() == DataType::BFLOAT16;

      // Set the pointers and strides.
      params.q_ptr = q->buffer();
      params.k_ptr = k->buffer();
      params.v_ptr = v->buffer();
      // All stride are in elements, not bytes.
      params.q_row_stride = q->stride(-3);
      params.k_row_stride = k->stride(-3);
      params.v_row_stride = v->stride(-3);
      params.q_head_stride = q->stride(-2);
      params.k_head_stride = k->stride(-2);
      params.v_head_stride = v->stride(-2);
      params.o_ptr = out->buffer();
      params.o_row_stride = out->stride(-3);
      params.o_head_stride = out->stride(-2);

      if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q->stride(0);
        params.k_batch_stride = k->stride(0);
        params.v_batch_stride = v->stride(0);
        params.o_batch_stride = out->stride(0);
        if (seqlenq_ngroups_swapped) {
          params.q_batch_stride *= seqlen_q;
          params.o_batch_stride *= seqlen_q;
        }
      }

      params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
      params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
      params.seqused_k = static_cast<int *>(seqused_k);

      // P = softmax(QK^T)
      params.p_ptr = p_d;

      // Softmax sum
      params.softmax_lse_ptr = softmax_lse_d;

      // Set the dimensions.
      params.b = b;
      params.h = h;
      params.h_k = h_k;
      params.h_h_k_ratio = h / h_k;
      params.seqlen_q = seqlen_q;
      params.seqlen_k = seqlen_k;
      params.seqlen_q_rounded = seqlen_q_rounded;
      params.seqlen_k_rounded = seqlen_k_rounded;
      params.d = d;
      params.d_rounded = d_rounded;

      // Set the different scale values.
      params.scale_softmax = softmax_scale;
      params.scale_softmax_log2 = softmax_scale * M_LOG2E;

      // Set this to probability of keeping an element to simplify things.
      // not use dropout
      params.p_dropout = 1.f;
      params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
      params.rp_dropout = 1.f / params.p_dropout;
      params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

      // Causal is the special case where window_size_right == 0 and window_size_left < 0.
      // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
      params.is_causal = window_size_left < 0 && window_size_right == 0;

      if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
      if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
      params.window_size_left = window_size_left;
      params.window_size_right = window_size_right;

      params.is_seqlens_k_cumulative = true;
    }

    // Find the number of splits that maximizes the occupancy. For example, if we have
    // batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
    // better than having 3 splits (efficiency = 0.67). However, we also don't want too many
    // splits as that would incur more HBM reads/writes.
    // So we find the best efficiency, then find the smallest number of splits that gets 85%
    // of the best efficiency.
    static int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
      // If we have enough to almost fill the SMs, then just use 1 split
      if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
      max_splits = std::min({max_splits, num_SMs, num_n_blocks});
      float max_efficiency = 0.f;
      std::vector<float> efficiency;
      efficiency.reserve(max_splits);
      auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
      // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
      // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
      // (i.e. it's 11 splits anyway).
      // So we check if the number of blocks per split is the same as the previous num_splits.
      auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
      };
      for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
          efficiency.push_back(0.f);
        } else {
          float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
          float eff = n_waves / ceil(n_waves);
          // printf("num_splits = %d, eff = %f\n", num_splits, eff);
          if (eff > max_efficiency) { max_efficiency = eff; }
          efficiency.push_back(eff);
        }
      }
      for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
          // printf("num_splits chosen = %d\n", num_splits);
          return num_splits;
        }
      }
      return 1;
    }

    static void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
                                   const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
                                   const int head_size_rounded,
                                   const int num_splits, cudaDeviceProp *dprops) {

      // This needs to match with run_mha_fwd_splitkv_dispatch
      const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
      const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
      // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
      // In any case we don't expect seqlen_q to be larger than 64 for inference.
      const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
      params.num_splits = num_splits;
      if (num_splits < 1) {
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount, num_n_blocks, 128);
      }
      TENSOR_CHECK((params.num_splits <= 128), "[FlashAttention] num_splits > 128 not supported");
    }

    void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
      FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
          if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
            run_mha_fwd_<elem_type, kHeadDim>(params, stream);
          } else {
            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
          }
        });
      });
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});
#endif
    template<>
    void FlashAttention::compute<Device::CUDA>(StorageView& queries,
                                               StorageView& keys,
                                               StorageView& values,
                                               StorageView& output,
                                               StorageView* cached_keys,
                                               StorageView* cached_values,
                                               StorageView* attention,
                                               bool return_normalized_attention,
                                               StorageView* rotary_cos,
                                               StorageView* rotary_sin,
                                               const bool rotary_interleave,
                                               StorageView* alibi,
                                               dim_t offset) const {
#ifdef CT2_WITH_FLASH_ATTN
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      dim_t window_size_left = _sliding_window > 0 ? _sliding_window : -1;
      dim_t window_size_right = _sliding_window > 0 ? 0 : -1;

      int device_id = ctranslate2::get_device_index(ctranslate2::Device::CUDA);
      auto dprops = ctranslate2::cuda::get_device_properties(device_id);

      const auto shape = queries.shape();
      const dim_t batch_size = shape[0];
      dim_t seqlen_q = shape[1];
      dim_t num_heads = shape[2];
      const dim_t head_size_og = shape[3];

      dim_t seqlen_k, num_heads_k;
      if (offset == 0) {
        seqlen_k = keys.dim(1);
        num_heads_k = keys.dim(2);
        if (window_size_left >= seqlen_k) { window_size_left = -1; }
        if (window_size_right >= seqlen_k) { window_size_right = -1; }
      } else {
        seqlen_k = cached_keys->dim(1);
        num_heads_k = cached_keys->dim(2);
      }

      bool is_causal = _is_causal;
      // causal=true is the same as causal=false in this case
      if (seqlen_q == 1 && !alibi) { is_causal = false; }
      if (is_causal) { window_size_right = 0; }

      // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
      // H/t Daniel Haziza
      const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0;
      if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        StorageView tmp(dtype, device);
        transpose_op(queries.reshape({batch_size, num_heads_k, ngroups, head_size_og}), tmp);
        queries = std::move(tmp);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
      }

      if (offset > 0) {
        if (window_size_left >= seqlen_k) { window_size_left = -1; }
        if (window_size_right >= seqlen_k) { window_size_right = -1; }
      }

      auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int head_size = round_multiple(head_size_og, 8);
      const int head_size_rounded = round_multiple(head_size, 32);
      const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
      const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

      StorageView softmax_lse({batch_size, num_heads, seqlen_q}, DataType::FLOAT32, device);
      output.resize(queries.shape());
      if (attention && return_normalized_attention) {
        attention->resize({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      }
      bool force_split_kernel = false;
      StorageView seqlens_k({batch_size}, static_cast<int>(offset), device);

      Flash_fwd_params params;
      if (offset == 0) {
        set_params_fprop(params,
                         batch_size,
                         seqlen_q, seqlen_k,
                         seqlen_q_rounded, seqlen_k_rounded,
                         num_heads, num_heads_k,
                         head_size, head_size_rounded,
                         &queries, &keys, &values, &output,
                        /*cu_seqlens_q_d=*/nullptr,
                        /*cu_seqlens_k_d=*/nullptr,
                        /*seqused_k=*/nullptr,
                         (return_normalized_attention && attention) ? attention->buffer() : /*p_ptr=*/nullptr,
                         softmax_lse.buffer(),
                         _queries_scale,
                         window_size_left,
                         window_size_right);

        // set params splitkv
        set_params_splitkv(params, batch_size, num_heads,
                           head_size, seqlen_k, seqlen_q,
                           head_size_rounded, /*num_splits*/0, &dprops);
      }
      else {
        const int page_block_size = 1;

        set_params_fprop(params,
                         batch_size,
                         seqlen_q, seqlen_k,
                         seqlen_q_rounded, seqlen_k_rounded,
                         num_heads, num_heads_k,
                         head_size, head_size_rounded,
                         &queries, cached_keys, cached_values, &output,
                         /*cu_seqlens_q_d=*/nullptr,
                         /*cu_seqlens_k_d=*/nullptr,
                         /*seqused_k=*/nullptr,
                         /*p_ptr=*/nullptr,
                         softmax_lse.buffer(),
                         _queries_scale,
                         window_size_left,
                         window_size_right);

        int seqlen_knew = keys.dim(1);
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = keys.buffer();
        params.vnew_ptr = values.buffer();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = keys.stride(0);
        params.vnew_batch_stride = values.stride(0);
        params.knew_row_stride = keys.stride(-3);
        params.vnew_row_stride = values.stride(-3);
        params.knew_head_stride = keys.stride(-2);
        params.vnew_head_stride = values.stride(-2);
        params.cu_seqlens_k =  static_cast<int *>(seqlens_k.buffer());
        params.is_seqlens_k_cumulative = false;

        if (rotary_cos && rotary_sin) {
          params.rotary_dim = rotary_cos->dim(1) * 2;
          params.rotary_cos_ptr = rotary_cos->buffer();
          params.rotary_sin_ptr = rotary_sin->buffer();
          params.is_rotary_interleaved = rotary_interleave;
        }
        else
          params.rotary_dim = 0;

        set_params_splitkv(params, batch_size, num_heads,
                           head_size, seqlen_k, seqlen_q,
                           head_size_rounded, /*num_splits*/0, &dprops);
        params.page_block_size = page_block_size;
        force_split_kernel = true;
      }

      StorageView softmax_lse_accum(DataType::FLOAT32, device);
      StorageView out_accum(DataType::FLOAT32, device);
      if (params.num_splits > 1) {
        softmax_lse_accum.resize({params.num_splits, batch_size, num_heads, seqlen_q});
        out_accum.resize({params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded});
        params.softmax_lseaccum_ptr = softmax_lse_accum.buffer();
        params.oaccum_ptr = out_accum.buffer();
      }
      params.alibi_slopes_ptr = nullptr;

      cudaStream_t stream = ctranslate2::cuda::get_cuda_stream();
      run_mha_fwd(params, stream, force_split_kernel);

      if (seqlenq_ngroups_swapped) {
        StorageView tmp(dtype, device);
        transpose_op(output, tmp);
        output = std::move(tmp);
        output.reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
      }
#else
      throw std::runtime_error("Flash attention 2 is not supported");
#endif
    }

#else  // CT2_USE_HIP — native HIP Flash Attention implementation

// ---------------------------------------------------------------------------
// HIP Flash Attention — native implementation for AMD GPUs (gfx1100 / RDNA3+)
//
// Two code paths are provided:
//
// 1) Tiled fused kernel (hip_flash_attn_fwd_tiled<DevT, BM, BN, D>)
//    Implements the Flash Attention 2 forward algorithm: a single grid block
//    processes BM contiguous query rows of one (batch, head), streams K/V in
//    BN-wide tiles through LDS, and maintains an online softmax state
//    (m_i, l_i) plus an FP32 output accumulator in registers.  S = Q@K^T is
//    NEVER materialised in HBM — memory and bandwidth scale with O(N·D)
//    instead of O(N^2).  Used whenever head_dim matches one of the
//    specialised values (64 / 80 / 128).
//
// 2) Three-pass fallback (hip_attn_qk_kernel + softmax + ov)
//    Simple and provably correct reference path that materialises the full
//    [batch, nheads, seqlen_q, seqlen_k] score buffer.  Kept as a fallback
//    for head dimensions that the tiled kernel is not specialised for, and
//    as an oracle for correctness comparisons.
//
// Memory layout of all tensors: [batch, seqlen, nheads, head_dim]
// Supports FP16 and BF16 inputs; FP32 accumulators throughout.
//
// Limitations in this initial implementation:
//   - Rotary embeddings and ALiBi are expected to be pre-applied by the caller.
//   - No backward pass (inference only).
// ---------------------------------------------------------------------------


    // -------------------------------------------------------------------
    // Kernel 0 — KV-cache write
    // Copies new_kv[b, t, h, d] → cache[b, write_offset+t, h, d].
    // Grid:  (seqlen_new, nheads, batch)
    // Block: (head_dim, 1, 1)
    // -------------------------------------------------------------------
    template <typename scalar_t>
    __global__ void hip_kv_cache_write_kernel(
        const scalar_t* __restrict__ new_kv,  // [batch, seqlen_new, nheads, head_dim]
        scalar_t*       __restrict__ cache,   // [batch, cache_size, nheads, head_dim]
        const int seqlen_new,
        const int cache_size,
        const int nheads,
        const int head_dim,
        const int write_offset)
    {
      const int b = blockIdx.z;
      const int t = blockIdx.y;
      const int h = blockIdx.x;
      const int d = threadIdx.x;
      if (d >= head_dim) return;

      const int src = b * seqlen_new * nheads * head_dim + t * nheads * head_dim + h * head_dim + d;
      const int dst = b * cache_size  * nheads * head_dim
                    + (write_offset + t) * nheads * head_dim + h * head_dim + d;
      cache[dst] = new_kv[src];
    }

    // -------------------------------------------------------------------
    // Kernel 1 — Q @ K^T
    // Grid:  (ceildiv(seqlen_q * seqlen_k, 256), nheads, batch)
    // Block: (256, 1, 1)
    // Each thread computes one element of S[b, h, q, k].
    //
    // k_time_stride: stride (in elements) between consecutive K time steps
    //   within one batch.  For a K tensor of shape [batch, K_seqlen, nheads, head_dim]
    //   this equals K_seqlen * nheads * head_dim.  When K is a view into a
    //   larger KV-cache buffer the allocation seqlen may differ from the
    //   logically active seqlen, so the stride must be passed explicitly.
    // -------------------------------------------------------------------
    template <typename scalar_t>
    __global__ void hip_attn_qk_kernel(
        const scalar_t* __restrict__ Q,  // [batch, seqlen_q, nheads, head_dim]
        const scalar_t* __restrict__ K,  // [batch, K_alloc, nheads, head_dim]
        float*          __restrict__ S,  // [batch, nheads, seqlen_q, seqlen_k]
        const int seqlen_q,
        const int seqlen_k,
        const int k_time_stride,         // K_alloc * nheads * head_dim
        const int nheads,
        const int head_dim,
        const float scale)
    {
      const int b   = blockIdx.z;
      const int h   = blockIdx.y;
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int q   = idx / seqlen_k;
      const int k   = idx % seqlen_k;
      if (q >= seqlen_q) return;

      float dot = 0.f;
      const int q_base = b * seqlen_q * nheads * head_dim + q * nheads * head_dim + h * head_dim;
      const int k_base = b * k_time_stride               + k * nheads * head_dim + h * head_dim;
      for (int d = 0; d < head_dim; ++d)
        dot += static_cast<float>(Q[q_base + d]) * static_cast<float>(K[k_base + d]);

      S[b * nheads * seqlen_q * seqlen_k + h * seqlen_q * seqlen_k + q * seqlen_k + k] =
          dot * scale;
    }

    // -------------------------------------------------------------------
    // Kernel 2 — row-wise softmax with optional causal mask
    // Grid:  (seqlen_q, nheads, batch)   — so gridDim = (seqlen_q, nheads, batch)
    // Block: (min(seqlen_k, 256), 1, 1)
    // Uses shared memory reduction; wavefront-safe for both wf32 and wf64.
    // -------------------------------------------------------------------
    __global__ void hip_attn_softmax_kernel(
        float* __restrict__ S,       // [batch, nheads, seqlen_q, seqlen_k] — modified in place
        const int nheads,
        const int seqlen_q,
        const int seqlen_k,
        const bool is_causal,
        const int q_offset)          // position of q[0] in the full sequence (for KV-cache)
    {
      const int b = blockIdx.z;
      const int h = blockIdx.y;
      const int q = blockIdx.x;

      float* row = S + (b * nheads + h) * seqlen_q * seqlen_k + q * seqlen_k;

      extern __shared__ float smem[];  // [blockDim.x] for reduction

      // --- apply causal mask ---
      const int q_pos = q + q_offset;
      for (int k = threadIdx.x; k < seqlen_k; k += blockDim.x)
        if (is_causal && k > q_pos) row[k] = -1e9f;
      __syncthreads();

      // --- find row max ---
      float local_max = -1e9f;
      for (int k = threadIdx.x; k < seqlen_k; k += blockDim.x)
        local_max = fmaxf(local_max, row[k]);
      smem[threadIdx.x] = local_max;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
      }
      const float row_max = smem[0];
      __syncthreads();

      // --- exp and row sum ---
      float local_sum = 0.f;
      for (int k = threadIdx.x; k < seqlen_k; k += blockDim.x) {
        const float e = expf(row[k] - row_max);
        row[k] = e;
        local_sum += e;
      }
      smem[threadIdx.x] = local_sum;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
      }
      const float row_sum = smem[0];
      __syncthreads();

      // --- normalise ---
      const float inv_sum = (row_sum > 0.f) ? 1.f / row_sum : 0.f;
      for (int k = threadIdx.x; k < seqlen_k; k += blockDim.x)
        row[k] *= inv_sum;
    }

    // -------------------------------------------------------------------
    // Kernel 3 — O = P @ V
    // Grid:  (seqlen_q, nheads, batch)
    // Block: (head_dim, 1, 1)   — head_dim <= 1024
    // Each thread accumulates one output channel O[b, q, h, d].
    //
    // v_time_stride: same concept as k_time_stride in Kernel 1.
    // -------------------------------------------------------------------
    template <typename scalar_t>
    __global__ void hip_attn_ov_kernel(
        const float*    __restrict__ P,  // [batch, nheads, seqlen_q, seqlen_k]
        const scalar_t* __restrict__ V,  // [batch, V_alloc, nheads, head_dim]
        scalar_t*       __restrict__ O,  // [batch, seqlen_q, nheads, head_dim]
        const int seqlen_k,
        const int v_time_stride,         // V_alloc * nheads * head_dim
        const int nheads,
        const int head_dim)
    {
      const int b = blockIdx.z;
      const int h = blockIdx.y;
      const int q = blockIdx.x;
      const int d = threadIdx.x;
      if (d >= head_dim) return;

      const int seqlen_q = gridDim.x;
      const float* p_row =
          P + b * nheads * seqlen_q * seqlen_k + h * seqlen_q * seqlen_k + q * seqlen_k;

      float out = 0.f;
      for (int k = 0; k < seqlen_k; ++k)
        out += p_row[k] *
               static_cast<float>(V[b * v_time_stride + k * nheads * head_dim + h * head_dim + d]);

      O[b * seqlen_q * nheads * head_dim + q * nheads * head_dim + h * head_dim + d] =
          static_cast<scalar_t>(out);
    }

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || !defined(__HIP_DEVICE_COMPILE__)
    // -------------------------------------------------------------------
    // RDNA3 WMMA Flash Attention forward kernel — FP16 only.
    //
    // Uses the wave32 16x16x16 fp16-input/fp32-accumulator WMMA built-in
    // (__builtin_amdgcn_wmma_f32_16x16x16_f16_w32) for Q·K^T and P·V.
    // Other paths in this file (3-pass / scalar tiled) remain for BF16,
    // for non-multiple-of-16 head dimensions, and as the correctness oracle.
    //
    // Layout (wave32, RDNA3 — verified empirically via wmma_probe.cu):
    //   A-fragment (16x16 fp16, row-major): each lane holds one full row
    //     a_frag[j] = A[lane % 16, j],  for j = 0..15
    //     (Lanes 16..31 carry a duplicate of rows 0..15.)
    //   B-fragment (16x16 fp16, col-major): each lane holds one full column
    //     b_frag[i] = B[i, lane % 16],  for i = 0..15
    //   C-fragment (16x16 fp32, accumulator): each lane holds 8 elements
    //     c_frag[s] = C[2*s + (lane >> 4), lane % 16],  for s = 0..7
    //     (Lanes 0..15 hold even rows, lanes 16..31 hold odd rows.)
    //
    // S = Q · K^T  is mapped to D = A · B with:
    //   A[i][k] = Q[i][k]                (Q tile, row-major in LDS)
    //   B[k][j] = K[j][k]                (K tile transposed view)
    //
    // Block layout:
    //   grid: (ceildiv(seqlen_q, BM), nheads, batch_size)
    //   block: 32 threads (one wave32)
    //   BM = 16 query rows per block,  BN = 16 key tokens per K/V tile
    //
    // LDS per block (D = 64):
    //   q_lds[BM][D]    fp16  — Q tile, scaled by softmax-scale on load
    //   k_lds[BN][D]    fp16  — current K tile
    //   v_lds[BN][D]    fp16  — current V tile
    //   p_lds[BM][BN]   fp16  — softmaxed S (used as A-fragment for P·V)
    //   s_scratch[BM][BN] fp32 — temporary S before softmax
    //   m_lds[BM]       fp32  — running max per query row
    //   l_lds[BM]       fp32  — running sum (denominator) per query row
    //   alpha_lds[BM]   fp32  — softmax correction factor per row
    //   Total ≈ 18 KiB / block, well within the 64 KiB budget.
    //
    // O is held entirely in registers as `o_frag[D/16]` (4 v8f32 fragments
    // for D = 64) per lane.  Each fragment covers a 16-column slice of O.
    // -------------------------------------------------------------------
    // HalfT is the wave-level fp16/bf16 element type and determines which
    // WMMA built-in we dispatch (f16 vs bf16).  Both variants share the
    // same wave32 accumulator-fragment layout on RDNA3.
    template <typename HalfT, int D>
    __global__ void hip_flash_attn_wmma_fp16(
        const HalfT* __restrict__ Q,
        const HalfT* __restrict__ K,
        const HalfT* __restrict__ V,
        HalfT*       __restrict__ O,
        const int seqlen_q,
        const int seqlen_k,
        const int k_time_stride,
        const int v_time_stride,
        const int nheads,
        const float scale,
        const bool is_causal,
        const int q_offset)
    {
      static_assert(D % 16 == 0, "head_dim must be a multiple of 16 for WMMA");
      constexpr int BM = 16;
      constexpr int BN = 16;
      constexpr int DT = D / 16;  // number of D-tiles

      using v16f16 = HalfT __attribute__((ext_vector_type(16)));
      using v8f32  = float __attribute__((ext_vector_type(8)));

      const int lane     = threadIdx.x;       // 0..31
      const int b        = blockIdx.z;
      const int h        = blockIdx.y;
      const int q_tile   = blockIdx.x;
      const int q_row_0  = q_tile * BM;

      __shared__ HalfT q_lds[BM][D];
      __shared__ HalfT k_lds[BN][D];
      __shared__ HalfT v_lds[BN][D];
      __shared__ HalfT p_lds[BM][BN];
      __shared__ float s_scratch[BM][BN];
      __shared__ float m_lds[BM];
      __shared__ float l_lds[BM];
      __shared__ float alpha_lds[BM];

      // ---- Load Q tile, pre-scaled (so S = Q·K^T already has softmax-scale) ----
      for (int idx = lane; idx < BM * D; idx += 32) {
        const int row = idx / D;
        const int col = idx % D;
        const int q_row = q_row_0 + row;
        float v = 0.f;
        if (q_row < seqlen_q)
          v = static_cast<float>(Q[b * seqlen_q * nheads * D
                                   + q_row * nheads * D + h * D + col]) * scale;
        q_lds[row][col] = static_cast<HalfT>(v);
      }

      // ---- Initialise running state and output accumulators ----
      if (lane < BM) {
        m_lds[lane] = -1e30f;
        l_lds[lane] = 0.f;
      }
      v8f32 o_frag[DT];
      #pragma unroll
      for (int t = 0; t < DT; ++t)
        #pragma unroll
        for (int s = 0; s < 8; ++s) o_frag[t][s] = 0.f;

      __syncthreads();

      // ---- Iterate over K/V tiles ----
      const int num_k_tiles = (seqlen_k + BN - 1) / BN;
      for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_row_0 = kt * BN;

        // -- Load K and V tiles --
        for (int idx = lane; idx < BN * D; idx += 32) {
          const int row = idx / D;
          const int col = idx % D;
          const int k_row = k_row_0 + row;
          HalfT kv_k = static_cast<HalfT>(0);
          HalfT kv_v = static_cast<HalfT>(0);
          if (k_row < seqlen_k) {
            kv_k = K[b * k_time_stride + k_row * nheads * D + h * D + col];
            kv_v = V[b * v_time_stride + k_row * nheads * D + h * D + col];
          }
          k_lds[row][col] = kv_k;
          v_lds[row][col] = kv_v;
        }
        __syncthreads();

        // -- Compute S[BM=16][BN=16] = Q · K^T using WMMA --
        v8f32 s_frag;
        #pragma unroll
        for (int s = 0; s < 8; ++s) s_frag[s] = 0.f;

        // Inner reduction over D in 16-element chunks.
        // A (Q row, row-major): a_frag[j] = q_lds[lane & 15][inner*16 + j]
        // B (K transposed, col-major): b_frag[i] = k_lds[lane & 15][inner*16 + i]
        // (col-major B's column index = lane mod 16, mapping to the K row j;
        //  inner reduction index k maps to the D-position within the chunk.)
        #pragma unroll
        for (int inner = 0; inner < DT; ++inner) {
          v16f16 a_frag, b_frag;
          const int a_row = lane & 15;
          const int b_col = lane & 15;
          #pragma unroll
          for (int x = 0; x < 16; ++x) {
            a_frag[x] = q_lds[a_row][inner * 16 + x];
            b_frag[x] = k_lds[b_col][inner * 16 + x];
          }
          if constexpr (std::is_same<HalfT, _Float16>::value)
            s_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, s_frag);
          else
            s_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, s_frag);
        }

        // -- Write S to LDS with causal/OOB masking baked in --
        {
          const int s_col = lane & 15;
          const int k_col_g = k_row_0 + s_col;
          #pragma unroll
          for (int s = 0; s < 8; ++s) {
            const int s_row = 2 * s + (lane >> 4);
            const int q_row_g = q_row_0 + s_row;
            const int q_pos   = q_row_g + q_offset;
            const bool oob    = (k_col_g >= seqlen_k) || (q_row_g >= seqlen_q);
            const bool masked = is_causal && k_col_g > q_pos;
            s_scratch[s_row][s_col] = (oob || masked) ? -1e30f : s_frag[s];
          }
        }
        __syncthreads();

        // -- Per-row softmax + online update (one thread per row, 16 rows / 32 threads) --
        if (lane < BM) {
          const int row = lane;
          const float m_old = m_lds[row];
          const float l_old = l_lds[row];

          float m_tile = -1e30f;
          #pragma unroll
          for (int c = 0; c < BN; ++c)
            m_tile = fmaxf(m_tile, s_scratch[row][c]);

          const float m_new = fmaxf(m_old, m_tile);
          const float alpha = (m_old == -1e30f) ? 0.f : __expf(m_old - m_new);

          float l_tile = 0.f;
          #pragma unroll
          for (int c = 0; c < BN; ++c) {
            const float v = s_scratch[row][c];
            const float e = (v <= -1e29f) ? 0.f : __expf(v - m_new);
            p_lds[row][c] = static_cast<HalfT>(e);  // P for the P·V WMMA
            l_tile += e;
          }

          m_lds[row]     = m_new;
          l_lds[row]     = alpha * l_old + l_tile;
          alpha_lds[row] = alpha;
        }
        __syncthreads();

        // -- Scale o_frag by per-row alpha --
        {
          #pragma unroll
          for (int t = 0; t < DT; ++t) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
              const int o_row = 2 * s + (lane >> 4);
              o_frag[t][s] *= alpha_lds[o_row];
            }
          }
        }

        // -- Compute O += P · V using WMMA (one V-N-tile per D-tile of O) --
        // A (P, row-major):  a_frag[j] = p_lds[lane & 15][j]      (one row of P)
        // B (V, col-major):  b_frag[i] = v_lds[i][t*16 + lane&15] (column t*16+col of V)
        // Inner reduction goes over BN = 16 (the K-token dimension), in one step.
        {
          v16f16 a_frag;
          const int a_row = lane & 15;
          #pragma unroll
          for (int x = 0; x < 16; ++x)
            a_frag[x] = p_lds[a_row][x];

          #pragma unroll
          for (int t = 0; t < DT; ++t) {
            v16f16 b_frag;
            const int b_col = lane & 15;
            #pragma unroll
            for (int i = 0; i < 16; ++i)
              b_frag[i] = v_lds[i][t * 16 + b_col];

            if constexpr (std::is_same<HalfT, _Float16>::value)
              o_frag[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, o_frag[t]);
            else
              o_frag[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, o_frag[t]);
          }
        }
        __syncthreads();
      }

      // ---- Normalise and store O ----
      if (lane < BM)
        alpha_lds[lane] = (l_lds[lane] > 0.f) ? 1.f / l_lds[lane] : 0.f;  // reuse alpha_lds as inv_l
      __syncthreads();

      #pragma unroll
      for (int t = 0; t < DT; ++t) {
        const int o_col_base = t * 16 + (lane & 15);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int o_row = 2 * s + (lane >> 4);
          const int q_row_g = q_row_0 + o_row;
          if (q_row_g < seqlen_q && o_col_base < D) {
            const float val = o_frag[t][s] * alpha_lds[o_row];
            O[b * seqlen_q * nheads * D + q_row_g * nheads * D + h * D + o_col_base]
                = static_cast<HalfT>(val);
          }
        }
      }
    }
    // -------------------------------------------------------------------
    // Larger WMMA kernel — BM_W = BN = 64, 4 wave32 wavefronts per block.
    //
    // The key win over the BM_W = 16 kernel is K/V HBM-reuse: each K-tile
    // and V-tile is loaded ONCE into LDS per block, and the four waves
    // share the loaded tiles to compute attention over 64 query rows.
    // Compared to BM_W = 16, total K-data / V-data read from HBM per
    // encoder layer drops by ~4x.
    //
    // Block layout:
    //   blockDim = 128 = 4 wave32
    //   wave w (0..3) handles Q-rows [w*16, w*16 + 16) of this block's tile
    //   All 128 threads cooperate on Q / K / V tile loads
    //   WMMA, softmax, and P*V are per-wave (each wave on its 16-row slice)
    //
    // LDS layout:
    //   q_lds[64][D]     fp16  — 8 KiB at D=64
    //   k_lds[64][D]     fp16  — 8 KiB
    //   v_lds[64][D]     fp16  — 8 KiB
    //   s_scratch[4][16][64] fp32 — 16 KiB (one 16x64 slab per wave)
    //   p_lds[4][16][64] fp16  — 8 KiB (P input for P*V WMMA)
    //   m_lds/l_lds/alpha_lds[64] fp32 — ~0.8 KiB
    //   Total ~49 KiB / block (fits in 64 KiB).
    // -------------------------------------------------------------------
    template <int D>
    __global__ void hip_flash_attn_wmma_fp16_bm64(
        const _Float16* __restrict__ Q,
        const _Float16* __restrict__ K,
        const _Float16* __restrict__ V,
        _Float16*       __restrict__ O,
        const int seqlen_q,
        const int seqlen_k,
        const int k_time_stride,
        const int v_time_stride,
        const int nheads,
        const float scale,
        const bool is_causal,
        const int q_offset)
    {
      static_assert(D % 16 == 0, "head_dim must be a multiple of 16 for WMMA");
      constexpr int BM_W = 64;
      constexpr int BN   = 64;
      constexpr int WAVES = 4;     // 128 / 32
      constexpr int DT = D / 16;   // 4 for D=64
      constexpr int NT = BN / 16;  // 4

      using v16f16 = _Float16 __attribute__((ext_vector_type(16)));
      using v8f32  = float    __attribute__((ext_vector_type(8)));

      const int tid     = threadIdx.x;     // 0..127
      const int wave    = tid >> 5;        // 0..3
      const int lane    = tid & 31;        // 0..31
      const int b       = blockIdx.z;
      const int h       = blockIdx.y;
      const int q_tile  = blockIdx.x;
      const int q_block_0 = q_tile * BM_W;
      const int q_row_0   = q_block_0 + wave * 16;  // this wave's first row

      __shared__ _Float16 q_lds[BM_W][D];
      __shared__ _Float16 k_lds[BN][D];
      __shared__ _Float16 v_lds[BN][D];
      __shared__ float    s_scratch[WAVES][16][BN];
      __shared__ _Float16 p_lds[WAVES][16][BN];
      __shared__ float    m_lds[BM_W];
      __shared__ float    l_lds[BM_W];
      __shared__ float    alpha_lds[BM_W];

      // ---- Load Q tile (pre-scaled by softmax-scale) ----
      for (int idx = tid; idx < BM_W * D; idx += 128) {
        const int row = idx / D;
        const int col = idx % D;
        const int q_row = q_block_0 + row;
        float v = 0.f;
        if (q_row < seqlen_q)
          v = static_cast<float>(Q[b * seqlen_q * nheads * D
                                   + q_row * nheads * D + h * D + col]) * scale;
        q_lds[row][col] = static_cast<_Float16>(v);
      }

      if (tid < BM_W) {
        m_lds[tid] = -1e30f;
        l_lds[tid] = 0.f;
      }

      v8f32 o_frag[DT];
      #pragma unroll
      for (int t = 0; t < DT; ++t)
        #pragma unroll
        for (int s = 0; s < 8; ++s) o_frag[t][s] = 0.f;

      __syncthreads();

      // ---- Loop over K/V tiles ----
      const int num_k_tiles = (seqlen_k + BN - 1) / BN;
      for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_block_0 = kt * BN;

        // -- Cooperative K/V load (all 128 threads) --
        for (int idx = tid; idx < BN * D; idx += 128) {
          const int row = idx / D;
          const int col = idx % D;
          const int k_row = k_block_0 + row;
          _Float16 kv_k = static_cast<_Float16>(0);
          _Float16 kv_v = static_cast<_Float16>(0);
          if (k_row < seqlen_k) {
            kv_k = K[b * k_time_stride + k_row * nheads * D + h * D + col];
            kv_v = V[b * v_time_stride + k_row * nheads * D + h * D + col];
          }
          k_lds[row][col] = kv_k;
          v_lds[row][col] = kv_v;
        }
        __syncthreads();

        // -- Per-wave Q*K^T into 4 S-fragments (one per N-tile) --
        v8f32 s_frag[NT];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          #pragma unroll
          for (int s = 0; s < 8; ++s) s_frag[nt][s] = 0.f;

          // Inner reduction over the head dimension
          #pragma unroll
          for (int inner = 0; inner < DT; ++inner) {
            v16f16 a_frag, b_frag;
            const int a_row = wave * 16 + (lane & 15);
            const int b_col = nt   * 16 + (lane & 15);
            #pragma unroll
            for (int x = 0; x < 16; ++x) {
              a_frag[x] = q_lds[a_row][inner * 16 + x];
              b_frag[x] = k_lds[b_col][inner * 16 + x];
            }
            s_frag[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_frag, b_frag, s_frag[nt]);
          }
        }

        // -- Write S to per-wave scratch, with causal / OOB mask --
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int s_col_local = lane & 15;
          const int s_col_bn    = nt * 16 + s_col_local;
          const int k_col_g     = k_block_0 + s_col_bn;
          #pragma unroll
          for (int s = 0; s < 8; ++s) {
            const int s_row_w = 2 * s + (lane >> 4);
            const int q_row_g = q_row_0 + s_row_w;
            const int q_pos   = q_row_g + q_offset;
            const bool oob    = (k_col_g >= seqlen_k) || (q_row_g >= seqlen_q);
            const bool masked = is_causal && k_col_g > q_pos;
            s_scratch[wave][s_row_w][s_col_bn] = (oob || masked) ? -1e30f : s_frag[nt][s];
          }
        }
        __syncthreads();

        // -- Per-row softmax (one thread per row of the 64-row block) --
        if (tid < BM_W) {
          const int row   = tid;
          const int w_row = row >> 4;            // which wave's chunk
          const int r_w   = row & 15;            // row index within that chunk
          const float m_old = m_lds[row];
          const float l_old = l_lds[row];

          float m_tile = -1e30f;
          #pragma unroll
          for (int c = 0; c < BN; ++c)
            m_tile = fmaxf(m_tile, s_scratch[w_row][r_w][c]);

          const float m_new = fmaxf(m_old, m_tile);
          const float alpha = (m_old == -1e30f) ? 0.f : __expf(m_old - m_new);

          float l_tile = 0.f;
          #pragma unroll
          for (int c = 0; c < BN; ++c) {
            const float v = s_scratch[w_row][r_w][c];
            const float e = (v <= -1e29f) ? 0.f : __expf(v - m_new);
            p_lds[w_row][r_w][c] = static_cast<_Float16>(e);
            l_tile += e;
          }

          m_lds[row]     = m_new;
          l_lds[row]     = alpha * l_old + l_tile;
          alpha_lds[row] = alpha;
        }
        __syncthreads();

        // -- Scale o_frag by per-row alpha (using this wave's 16 rows) --
        {
          #pragma unroll
          for (int t = 0; t < DT; ++t) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
              const int o_row_w  = 2 * s + (lane >> 4);
              const int o_row_b  = wave * 16 + o_row_w;
              o_frag[t][s] *= alpha_lds[o_row_b];
            }
          }
        }

        // -- O += P * V via WMMA (inner reduction over BN in 16-chunks) --
        #pragma unroll
        for (int t = 0; t < DT; ++t) {
          #pragma unroll
          for (int n_in = 0; n_in < NT; ++n_in) {
            v16f16 a_frag, b_frag;
            const int a_row = lane & 15;
            const int b_col = lane & 15;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
              a_frag[j] = p_lds[wave][a_row][n_in * 16 + j];
            #pragma unroll
            for (int i = 0; i < 16; ++i)
              b_frag[i] = v_lds[n_in * 16 + i][t * 16 + b_col];

            o_frag[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_frag, b_frag, o_frag[t]);
          }
        }
        __syncthreads();
      }

      // ---- Normalise (1/l) ----
      if (tid < BM_W)
        alpha_lds[tid] = (l_lds[tid] > 0.f) ? 1.f / l_lds[tid] : 0.f;
      __syncthreads();

      // ---- Store O ----
      #pragma unroll
      for (int t = 0; t < DT; ++t) {
        const int o_col = t * 16 + (lane & 15);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int o_row_w = 2 * s + (lane >> 4);
          const int o_row_b = wave * 16 + o_row_w;
          const int q_row_g = q_block_0 + o_row_b;
          if (q_row_g < seqlen_q && o_col < D) {
            const float val = o_frag[t][s] * alpha_lds[o_row_b];
            O[b * seqlen_q * nheads * D + q_row_g * nheads * D + h * D + o_col]
                = static_cast<_Float16>(val);
          }
        }
      }
    }
#endif  // gfx11

    // -------------------------------------------------------------------
    // Tiled fused forward attention (Flash Attention 2 algorithm)
    //
    // Each grid block processes BM contiguous query rows of one (batch, head).
    //   Grid:  (ceildiv(seqlen_q, BM), nheads, batch)
    //   Block: (BM, 1, 1)        — one thread per query row.
    //
    // Per-thread state (kept in registers, never spilled to HBM):
    //   q_reg[D]   : the thread's Q row, FP32
    //   acc[D]     : running output accumulator, FP32
    //   m_i, l_i   : running max / normaliser of the online softmax
    //
    // Per-block shared memory:
    //   s_k[BN][D] : current K tile
    //   s_v[BN][D] : current V tile
    //   (BM threads cooperatively load BN·D elements per tile.)
    //
    // For every K/V tile we compute s_tile[BN] = q · k_t  for the local Q row,
    // apply the bounds + causal mask, then perform the standard online-softmax
    // update of (m_i, l_i, acc).  After all tiles we divide the accumulator
    // by l_i and store it back to HBM.
    //
    // Template parameters allow the compiler to fully unroll the inner D loops
    // and to keep q_reg/acc entirely in registers.  Supported D values are
    // currently 64, 80, 128 (covering Whisper / common transformer heads).
    // -------------------------------------------------------------------
    template <typename scalar_t, int BM, int BN, int D>
    __global__ void hip_flash_attn_fwd_tiled(
        const scalar_t* __restrict__ Q,  // [batch, seqlen_q, nheads, D]
        const scalar_t* __restrict__ K,  // [batch, K_alloc,  nheads, D]
        const scalar_t* __restrict__ V,  // [batch, V_alloc,  nheads, D]
        scalar_t*       __restrict__ O,  // [batch, seqlen_q, nheads, D]
        const int seqlen_q,
        const int seqlen_k,
        const int k_time_stride,         // K_alloc * nheads * D (batch stride)
        const int v_time_stride,         // V_alloc * nheads * D
        const int nheads,
        const float scale,
        const bool is_causal,
        const int q_offset)              // absolute position of q[0]
    {
      const int b       = blockIdx.z;
      const int h       = blockIdx.y;
      const int q_tile  = blockIdx.x;
      const int tid     = threadIdx.x;          // 0 .. BM-1
      const int q       = q_tile * BM + tid;    // global query row

      __shared__ scalar_t s_k[BN][D];
      __shared__ scalar_t s_v[BN][D];

      // -- Load this thread's Q row into registers (FP32, scaled once) --
      float q_reg[D];
      const bool q_valid = (q < seqlen_q);
      if (q_valid) {
        const int q_base = b * seqlen_q * nheads * D + q * nheads * D + h * D;
        #pragma unroll
        for (int d = 0; d < D; ++d)
          q_reg[d] = static_cast<float>(Q[q_base + d]) * scale;
      } else {
        #pragma unroll
        for (int d = 0; d < D; ++d) q_reg[d] = 0.f;
      }

      // -- Online softmax state --
      float m_i = -1e30f;
      float l_i = 0.f;
      float acc[D];
      #pragma unroll
      for (int d = 0; d < D; ++d) acc[d] = 0.f;

      const int num_tiles = (seqlen_k + BN - 1) / BN;
      const int q_pos     = q + q_offset;

      for (int t = 0; t < num_tiles; ++t) {
        const int k_start = t * BN;

        // -- Cooperatively load K and V tiles into LDS --
        //    BM threads load BN*D elements each (== BN*D / BM per thread).
        for (int idx = tid; idx < BN * D; idx += BM) {
          const int kk = idx / D;
          const int dd = idx % D;
          const int kpos = k_start + kk;
          if (kpos < seqlen_k) {
            const int k_base = b * k_time_stride + kpos * nheads * D + h * D;
            const int v_base = b * v_time_stride + kpos * nheads * D + h * D;
            s_k[kk][dd] = K[k_base + dd];
            s_v[kk][dd] = V[v_base + dd];
          } else {
            s_k[kk][dd] = static_cast<scalar_t>(0);
            s_v[kk][dd] = static_cast<scalar_t>(0);
          }
        }
        __syncthreads();

        if (q_valid) {
          // -- s_tile[kk] = q · k_t, with bounds + causal mask --
          float s_tile[BN];
          float m_tile = -1e30f;
          #pragma unroll
          for (int kk = 0; kk < BN; ++kk) {
            const int kpos = k_start + kk;
            const bool oob    = kpos >= seqlen_k;
            const bool masked = is_causal && kpos > q_pos;
            if (oob || masked) {
              s_tile[kk] = -1e30f;
            } else {
              float dot = 0.f;
              #pragma unroll
              for (int d = 0; d < D; ++d)
                dot += q_reg[d] * static_cast<float>(s_k[kk][d]);
              s_tile[kk] = dot;
              if (dot > m_tile) m_tile = dot;
            }
          }

          // -- Online softmax update --
          const float m_new = fmaxf(m_i, m_tile);
          const float alpha = (m_i == -1e30f) ? 0.f : __expf(m_i - m_new);

          float l_tile = 0.f;
          #pragma unroll
          for (int kk = 0; kk < BN; ++kk) {
            // exp(-inf - finite) == 0, but __expf is undefined for -inf
            // on some HIP targets, so guard explicitly.
            float e = (s_tile[kk] <= -1e29f) ? 0.f : __expf(s_tile[kk] - m_new);
            s_tile[kk] = e;
            l_tile += e;
          }

          // acc = alpha * acc + s_tile @ V_tile
          #pragma unroll
          for (int d = 0; d < D; ++d) {
            float a = alpha * acc[d];
            #pragma unroll
            for (int kk = 0; kk < BN; ++kk)
              a += s_tile[kk] * static_cast<float>(s_v[kk][d]);
            acc[d] = a;
          }
          l_i = alpha * l_i + l_tile;
          m_i = m_new;
        }
        __syncthreads();
      }

      // -- Final normalisation and store --
      if (q_valid) {
        const float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
        const int o_base = b * seqlen_q * nheads * D + q * nheads * D + h * D;
        #pragma unroll
        for (int d = 0; d < D; ++d)
          O[o_base + d] = static_cast<scalar_t>(acc[d] * inv_l);
      }
    }

    // -------------------------------------------------------------------
    // Decode-optimised kernel (seqlen_q == 1)
    //
    // The tiled forward kernel above uses one thread per Q row.  During
    // autoregressive generation seqlen_q == 1, so that design would leave
    // BM-1 of BM threads idle.  This kernel inverts the parallelisation:
    // a single block handles one (batch, head), and the threads cooperate
    // along the K dimension instead.
    //
    // Grid:  (1, nheads, batch)
    // Block: BLOCK threads (BLOCK >= D, both powers of two)
    //
    // Phase 1  Compute all seqlen_k scores S[k] = Q . K[k]  (each thread
    //          handles k = tid, tid+BLOCK, …) and stash them in LDS.
    // Phase 2  Block-wide tree reduction for row max → exp(S - max) → sum.
    // Phase 3  Output: thread tid (tid < D) accumulates one channel
    //          O[d] = Σ_k P[k] · V[k][d] / sum.  V[k][.] is loaded with
    //          the natural [k, d] memory layout, so the BLOCK threads
    //          read coalesced D-wide vectors per k.
    //
    // LDS layout (one dynamic shared array; scratch is reused across
    // phases — reduce buffer in phase 2, V-tile in phase 3):
    //   [0           .. D)                      q_lds   (FP32 scaled Q)
    //   [D           .. D+seqlen_k)             s_lds   (FP32 scores)
    //   [D+seqlen_k  .. +max(BLOCK, V_TILE*D))  scratch
    //
    // V_TILE controls Phase-3 V-tiling: instead of every output channel
    // streaming all seqlen_k V values from HBM with no reuse, the BLOCK
    // threads cooperatively stage a V_TILE-wide slab of V into LDS once,
    // then the D output-channel threads accumulate against the cached
    // tile.  HBM reads for V drop by ~BLOCK/D for that phase.
    //
    // gfx1100 LDS budget is 64 KiB.  For D=64, BLOCK=64, V_TILE=64 the
    // scratch region is 64*64*4 = 16 KiB, so seqlen_k can reach ~12 k.
    // -------------------------------------------------------------------
    template <typename scalar_t, int BLOCK, int D, int V_TILE>
    __global__ void hip_flash_decode_kernel(
        const scalar_t* __restrict__ Q,  // [batch, 1,        nheads, D]
        const scalar_t* __restrict__ K,  // [batch, K_alloc,  nheads, D]
        const scalar_t* __restrict__ V,  // [batch, V_alloc,  nheads, D]
        scalar_t*       __restrict__ O,  // [batch, 1,        nheads, D]
        const int seqlen_k,
        const int k_time_stride,
        const int v_time_stride,
        const int nheads,
        const float scale,
        const bool is_causal,
        const int q_offset)
    {
      const int b   = blockIdx.z;
      const int h   = blockIdx.y;
      const int tid = threadIdx.x;

      extern __shared__ float smem[];
      float* q_lds   = smem;                       // D floats
      float* s_lds   = smem + D;                   // seqlen_k floats
      float* scratch = s_lds + seqlen_k;           // reduce_buf OR v_tile

      // ---- Load Q (FP32, scaled once) into LDS ----
      for (int d = tid; d < D; d += BLOCK)
        q_lds[d] = static_cast<float>(Q[b * nheads * D + h * D + d]) * scale;
      __syncthreads();

      // ---- Phase 1: S[k] = q · k_t  (+ causal mask) ----
      const int q_pos = q_offset;  // seqlen_q == 1, so q_pos == offset
      for (int k = tid; k < seqlen_k; k += BLOCK) {
        if (is_causal && k > q_pos) {
          s_lds[k] = -1e30f;
          continue;
        }
        const int k_base = b * k_time_stride + k * nheads * D + h * D;
        float dot = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d)
          dot += q_lds[d] * static_cast<float>(K[k_base + d]);
        s_lds[k] = dot;
      }
      __syncthreads();

      // ---- Phase 2a: row max via block-wide reduction (scratch=reduce_buf) ----
      float local_max = -1e30f;
      for (int k = tid; k < seqlen_k; k += BLOCK)
        local_max = fmaxf(local_max, s_lds[k]);
      scratch[tid] = local_max;
      __syncthreads();
      for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
        __syncthreads();
      }
      const float row_max = scratch[0];
      __syncthreads();

      // ---- Phase 2b: exp(S - max) and row sum ----
      float local_sum = 0.f;
      for (int k = tid; k < seqlen_k; k += BLOCK) {
        const float e = (s_lds[k] <= -1e29f) ? 0.f : __expf(s_lds[k] - row_max);
        s_lds[k] = e;
        local_sum += e;
      }
      scratch[tid] = local_sum;
      __syncthreads();
      for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
      }
      const float row_sum = scratch[0];
      const float inv_sum = (row_sum > 0.f) ? 1.f / row_sum : 0.f;
      __syncthreads();

      // ---- Phase 3: O[d] = Σ_k P[k] · V[k][d] / sum, with V-tiling ----
      // scratch is reinterpreted as v_tile (row-major [V_TILE][D] FP32).
      float* v_tile = scratch;
      float acc = 0.f;

      for (int kt = 0; kt < seqlen_k; kt += V_TILE) {
        // -- Cooperative load: BLOCK threads stage V_TILE * D elements --
        for (int idx = tid; idx < V_TILE * D; idx += BLOCK) {
          const int kk = idx / D;
          const int dd = idx % D;
          const int kpos = kt + kk;
          float v_val = 0.f;
          if (kpos < seqlen_k) {
            const int v_idx = b * v_time_stride + kpos * nheads * D + h * D + dd;
            v_val = static_cast<float>(V[v_idx]);
          }
          v_tile[idx] = v_val;
        }
        __syncthreads();

        // -- Accumulate: each output-channel thread sums over this tile --
        if (tid < D) {
          const int kk_max = min((int)V_TILE, seqlen_k - kt);
          for (int kk = 0; kk < kk_max; ++kk) {
            const float p = s_lds[kt + kk];
            acc += p * v_tile[kk * D + tid];
          }
        }
        __syncthreads();
      }

      if (tid < D)
        O[b * nheads * D + h * D + tid] = static_cast<scalar_t>(acc * inv_sum);
    }

    // -------------------------------------------------------------------
    // Dispatcher called from FlashAttention::compute<Device::CUDA> below.
    //
    // KV-cache semantics (mirrors the CUDA CUTLASS path):
    //   offset == 0  — prefilling / encoder self-attention:
    //     keys/values contain the full context; cached_keys/values are
    //     filled or updated by the layer *before* this call.
    //   offset > 0   — autoregressive decoder step:
    //     keys/values contain only the NEW tokens (seqlen_new, typically 1).
    //     cached_keys/values hold tokens [0 .. offset-1] and have capacity
    //     for at least offset+seqlen_new tokens.
    //     We must: 1) write new tokens into the cache at position offset,
    //              2) run attention with Q vs. full cache [0 .. offset+seqlen_new).
    // -------------------------------------------------------------------
    template <typename scalar_t>
    static void flash_attention_hip_impl(
        StorageView& queries,
        StorageView& keys,
        StorageView& values,
        StorageView& output,
        StorageView* cached_keys,
        StorageView* cached_values,
        float queries_scale,
        bool is_causal,
        dim_t offset)
    {
      const dim_t batch_size = queries.dim(0);
      const dim_t seqlen_q   = queries.dim(1);
      const dim_t num_heads  = queries.dim(2);
      const dim_t head_dim   = queries.dim(3);
      const dim_t seqlen_new = keys.dim(1);   // NEW tokens this step

      using DevT = typename cuda::DeviceType<scalar_t>::type;
      const DevT* q_ptr = reinterpret_cast<const DevT*>(queries.data<scalar_t>());

      hipStream_t stream = cuda::get_cuda_stream();

      // Determine K/V pointers and their batch-stride depending on whether
      // we are using the KV-cache or the raw keys/values tensors.
      const DevT* k_ptr;
      const DevT* v_ptr;
      dim_t seqlen_k;        // number of KEY tokens to attend over
      dim_t k_time_stride;   // K_alloc * nheads * head_dim (batch stride for K)
      dim_t v_time_stride;   // V_alloc * nheads * head_dim

      if (offset == 0) {
        // --- Prefilling / encoder ---
        k_ptr         = reinterpret_cast<const DevT*>(keys.data<scalar_t>());
        v_ptr         = reinterpret_cast<const DevT*>(values.data<scalar_t>());
        seqlen_k      = seqlen_new;
        k_time_stride = seqlen_k * num_heads * head_dim;
        v_time_stride = seqlen_k * num_heads * head_dim;
      } else {
        // --- Autoregressive decode step ---
        // 1. Write new keys/values into the cache at position `offset`.
        const dim_t cache_size = cached_keys->dim(1);
        {
          // grid: (num_heads, seqlen_new, batch)
          //   blockIdx.x = head index  → matches kernel's h = blockIdx.x
          //   blockIdx.y = time step   → matches kernel's t = blockIdx.y
          //   blockIdx.z = batch index → matches kernel's b = blockIdx.z
          dim3 grid(num_heads, seqlen_new, batch_size);
          const int block = min((int)head_dim, 1024);
          hipLaunchKernelGGL(
              hip_kv_cache_write_kernel<DevT>,
              grid, block, 0, stream,
              reinterpret_cast<const DevT*>(keys.data<scalar_t>()),
              reinterpret_cast<DevT*>(cached_keys->data<scalar_t>()),
              (int)seqlen_new, (int)cache_size,
              (int)num_heads, (int)head_dim, (int)offset);
          hipLaunchKernelGGL(
              hip_kv_cache_write_kernel<DevT>,
              grid, block, 0, stream,
              reinterpret_cast<const DevT*>(values.data<scalar_t>()),
              reinterpret_cast<DevT*>(cached_values->data<scalar_t>()),
              (int)seqlen_new, (int)cache_size,
              (int)num_heads, (int)head_dim, (int)offset);
        }
        // 2. Attend over the full (now updated) cache.
        k_ptr         = reinterpret_cast<const DevT*>(cached_keys->data<scalar_t>());
        v_ptr         = reinterpret_cast<const DevT*>(cached_values->data<scalar_t>());
        seqlen_k      = offset + seqlen_new;
        k_time_stride = cache_size * num_heads * head_dim;
        v_time_stride = cache_size * num_heads * head_dim;
      }

      output.resize(queries.shape());
      DevT* o_ptr = reinterpret_cast<DevT*>(output.data<scalar_t>());

      // ----------------------------------------------------------------
      // Fast path A: decode-optimised kernel for seqlen_q == 1.
      // Used for every autoregressive generation step.  One block per
      // (batch, head) — threads parallelise across the K dimension.
      // ----------------------------------------------------------------
      if (seqlen_q == 1) {
        // LDS budget: D + seqlen_k + max(BLOCK, V_TILE*D) FP32 elements.
        // gfx1100 has 64 KiB per block.  V_TILE is chosen per head_dim to
        // maximise per-sync reuse without busting LDS.
        const dim_t lds_budget_floats = 64 * 1024 / sizeof(float);
        auto launch_decode =
            [&](auto head_dim_const, auto block_const, auto vtile_const) -> bool {
          constexpr int D = decltype(head_dim_const)::value;
          constexpr int BLOCK = decltype(block_const)::value;
          constexpr int V_TILE = decltype(vtile_const)::value;
          if (head_dim != D) return false;
          const size_t scratch =
              std::max((size_t)BLOCK, (size_t)V_TILE * D);
          const size_t lds_floats = (size_t)D + (size_t)seqlen_k + scratch;
          if ((dim_t)lds_floats > lds_budget_floats) return false;
          dim3 grid(1, num_heads, batch_size);
          dim3 block(BLOCK);
          const size_t lds_bytes = lds_floats * sizeof(float);
          hipLaunchKernelGGL((hip_flash_decode_kernel<DevT, BLOCK, D, V_TILE>),
                             grid, block, lds_bytes, stream,
                             q_ptr, k_ptr, v_ptr, o_ptr,
                             (int)seqlen_k,
                             (int)k_time_stride, (int)v_time_stride,
                             (int)num_heads, queries_scale,
                             is_causal, (int)offset);
          return true;
        };
        // BLOCK > D so the V-tile load is co-operative across more threads
        // than there are output channels — gives ~BLOCK/D HBM-bandwidth
        // reduction on V in phase 3 plus more parallelism in phase 1.
        // V_TILE picked large to amortise the sync cost over more useful
        // multiply-adds (D=64 -> 128 K-tokens per tile -> 32 KiB scratch).
        if (launch_decode(std::integral_constant<int,  64>{},
                          std::integral_constant<int, 128>{},
                          std::integral_constant<int, 128>{})) return;
        if (launch_decode(std::integral_constant<int, 128>{},
                          std::integral_constant<int, 256>{},
                          std::integral_constant<int,  64>{})) return;
        // else: fall through to 3-pass for very long sequences or
        //       unsupported head dimensions.
      }

      // ----------------------------------------------------------------
      // Fast path B: WMMA-accelerated kernels on RDNA3.
      //   - BM_W = 16 variant: single wave32 per block, used for both FP16
      //     and BF16 inputs (same wave32 fragment layout; different built-in).
      //   - BM_W = 64 variant exists in this TU as future-work; currently
      //     not dispatched — see the comment above launch_wmma_bm64.
      // ----------------------------------------------------------------
      if constexpr (std::is_same<scalar_t, float16_t>::value
                 || std::is_same<scalar_t, bfloat16_t>::value) {
        using HalfT = std::conditional_t<
            std::is_same<scalar_t, float16_t>::value, _Float16, __bf16>;

        auto launch_wmma_bm64 = [&](auto head_dim_const) -> bool {
          constexpr int D = decltype(head_dim_const)::value;
          constexpr int BM_W = 64;
          if (head_dim != D) return false;
          if (D % 16 != 0)   return false;
          if (seqlen_q < BM_W) return false;
          if constexpr (!std::is_same<HalfT, _Float16>::value) return false;
          // (BM_W=64 BF16 variant not generated; only FP16 specialisation exists.)
          dim3 grid((seqlen_q + BM_W - 1) / BM_W, num_heads, batch_size);
          dim3 block(128);  // 4 wave32
          if constexpr (std::is_same<HalfT, _Float16>::value) {
            hipLaunchKernelGGL((hip_flash_attn_wmma_fp16_bm64<D>),
                               grid, block, 0, stream,
                               reinterpret_cast<const _Float16*>(q_ptr),
                               reinterpret_cast<const _Float16*>(k_ptr),
                               reinterpret_cast<const _Float16*>(v_ptr),
                               reinterpret_cast<_Float16*>(o_ptr),
                               (int)seqlen_q, (int)seqlen_k,
                               (int)k_time_stride, (int)v_time_stride,
                               (int)num_heads, queries_scale,
                               is_causal, (int)offset);
          }
          return true;
        };
        // BM_W=64 currently disabled (see kernel comment for tuning notes).
        // if (launch_wmma_bm64(std::integral_constant<int, 64>{})) return;
        (void)launch_wmma_bm64;

        auto launch_wmma_bm16 = [&](auto head_dim_const) -> bool {
          constexpr int D = decltype(head_dim_const)::value;
          constexpr int BM_W = 16;
          if (head_dim != D) return false;
          if (D % 16 != 0)   return false;
          if (seqlen_q < BM_W) return false;
          dim3 grid((seqlen_q + BM_W - 1) / BM_W, num_heads, batch_size);
          dim3 block(32);  // one wave32
          hipLaunchKernelGGL((hip_flash_attn_wmma_fp16<HalfT, D>),
                             grid, block, 0, stream,
                             reinterpret_cast<const HalfT*>(q_ptr),
                             reinterpret_cast<const HalfT*>(k_ptr),
                             reinterpret_cast<const HalfT*>(v_ptr),
                             reinterpret_cast<HalfT*>(o_ptr),
                             (int)seqlen_q, (int)seqlen_k,
                             (int)k_time_stride, (int)v_time_stride,
                             (int)num_heads, queries_scale,
                             is_causal, (int)offset);
          return true;
        };
        if (launch_wmma_bm16(std::integral_constant<int,  64>{})) return;
        if (launch_wmma_bm16(std::integral_constant<int, 128>{})) return;
      }

      // Fast path C: scalar tiled Flash Attention 2 kernel.
      // Fallback for BF16 and head dimensions WMMA doesn't cover.
      // Used when seqlen_q >= BM (one thread per query row).
      // ----------------------------------------------------------------
      constexpr int BM = 64;
      constexpr int BN = 64;
      if (seqlen_q >= BM) {
        auto launch_tiled = [&](auto head_dim_const) -> bool {
          constexpr int D = decltype(head_dim_const)::value;
          if (head_dim != D) return false;
          dim3 grid((seqlen_q + BM - 1) / BM, num_heads, batch_size);
          dim3 block(BM);
          hipLaunchKernelGGL((hip_flash_attn_fwd_tiled<DevT, BM, BN, D>),
                             grid, block, 0, stream,
                             q_ptr, k_ptr, v_ptr, o_ptr,
                             (int)seqlen_q, (int)seqlen_k,
                             (int)k_time_stride, (int)v_time_stride,
                             (int)num_heads, queries_scale,
                             is_causal, (int)offset);
          return true;
        };

        if (launch_tiled(std::integral_constant<int, 64>{}))  return;
        if (launch_tiled(std::integral_constant<int, 80>{}))  return;
        if (launch_tiled(std::integral_constant<int, 128>{})) return;
      }

      // ----------------------------------------------------------------
      // Fallback: three-pass reference implementation.  Used when head_dim
      // is not one of the specialised values above.  Materialises the full
      // [batch, nheads, seqlen_q, seqlen_k] score buffer in HBM.
      // ----------------------------------------------------------------
      StorageView scores_buf({batch_size, num_heads, seqlen_q, seqlen_k},
                             DataType::FLOAT32, Device::CUDA);
      float* s_ptr = scores_buf.data<float>();

      // --- Pass 1: Q @ K^T ---
      {
        const int total = seqlen_q * seqlen_k;
        const int block = 256;
        dim3 grid((total + block - 1) / block, num_heads, batch_size);
        hipLaunchKernelGGL(hip_attn_qk_kernel<DevT>, grid, block, 0, stream,
                           q_ptr, k_ptr, s_ptr,
                           (int)seqlen_q, (int)seqlen_k, (int)k_time_stride,
                           (int)num_heads, (int)head_dim,
                           queries_scale);
      }

      // --- Pass 2: row-wise softmax (with causal mask) ---
      {
        // The tree reduction inside hip_attn_softmax_kernel requires
        // blockDim.x to be a power of 2.  Round up min(seqlen_k, 256) to
        // the next power of 2 (256 itself is already a power of 2, so the
        // cap never breaks the invariant).  Extra threads contribute the
        // identity values (-1e9 / 0.0) and are harmless.
        int block = 1;
        while (block < (int)std::min(seqlen_k, (dim_t)256)) block <<= 1;
        dim3 grid(seqlen_q, num_heads, batch_size);
        hipLaunchKernelGGL(hip_attn_softmax_kernel, grid, block,
                           block * sizeof(float),  // dynamic shared mem
                           stream,
                           s_ptr, (int)num_heads, (int)seqlen_q, (int)seqlen_k,
                           is_causal, (int)offset);
      }

      // --- Pass 3: P @ V ---
      {
        const int block = min((int)head_dim, 1024);
        dim3 grid(seqlen_q, num_heads, batch_size);
        hipLaunchKernelGGL(hip_attn_ov_kernel<DevT>, grid, block, 0, stream,
                           s_ptr, v_ptr, o_ptr,
                           (int)seqlen_k, (int)v_time_stride,
                           (int)num_heads, (int)head_dim);
      }
    }

    template <>
    void FlashAttention::compute<Device::CUDA>(
        StorageView& queries,
        StorageView& keys,
        StorageView& values,
        StorageView& output,
        StorageView* cached_keys,
        StorageView* cached_values,
        StorageView* /*attention*/,
        bool         /*return_normalized_attention*/,
        StorageView* /*rotary_cos*/,
        StorageView* /*rotary_sin*/,
        const bool   /*rotary_interleave*/,
        StorageView* /*alibi*/,
        dim_t        offset) const
    {
      const DataType dtype = queries.dtype();
      switch (dtype) {
        case DataType::FLOAT16:
          flash_attention_hip_impl<float16_t>(
              queries, keys, values, output,
              cached_keys, cached_values,
              _queries_scale, _is_causal, offset);
          break;
        case DataType::BFLOAT16:
          flash_attention_hip_impl<bfloat16_t>(
              queries, keys, values, output,
              cached_keys, cached_values,
              _queries_scale, _is_causal, offset);
          break;
        default:
          throw std::invalid_argument(
              "Flash Attention HIP only supports float16 and bfloat16 inputs.");
      }
    }

#endif  // CT2_USE_HIP / !CT2_USE_HIP

  }  // namespace ops
}  // namespace ctranslate2
