#include "ctranslate2/layers/flash-attention/flash.h"
#include "ctranslate2/layers/flash-attention/static_switch.h"
#include "ctranslate2/layers/flash_attention.h"
#include "cuda/utils.h"

#define TENSOR_CHECK(ans, message)                      \
    {                                                   \
      if (ans)                                          \
        THROW_RUNTIME_ERROR("tensor error: "            \
                            + message);                 \
    }


namespace ctranslate2 {
  namespace layers {
    void set_params_fprop(Flash_fwd_params &params,
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

#ifdef FLASHATTENTION_DISABLE_LOCAL
      TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
#endif

      params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
      TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
#endif
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

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
    inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
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

    static std::vector<Dense> make_linear_layers(const models::Model& model,
                                                 const std::string& scope,
                                                 bool self_attention) {
      const dim_t num_linear_layers = self_attention ? 2 : 3;
      std::vector<Dense> layers;
      layers.reserve(num_linear_layers);
      for (dim_t i = 0; i < num_linear_layers; ++i)
        if (i == (num_linear_layers - 1)) {
          layers.emplace_back(model, scope + "/linear_" + std::to_string(i), nullptr, true);
        } else
          layers.emplace_back(model, scope + "/linear_" + std::to_string(i));
      return layers;
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    static void split_heads(StorageView& x,
                            dim_t num_heads,
                            const Padder* padder = nullptr,
                            dim_t beam_size = 1) {
      if (padder)
        padder->add_padding(x);

      if (beam_size > 1)
        x.reshape({x.dim(0) / beam_size, beam_size, x.dim(2)});

      // x has shape [batch_size, time, depth]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t head_dim = x.dim(2) / num_heads;

      if (time == 1) {
        x.reshape({batch_size, num_heads, 1, head_dim});
      } else {
        x.reshape({batch_size, time, num_heads, head_dim});
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }
    }

    static void replicate_heads(StorageView& x, dim_t repeats) {
      x.expand_dims(2);
      ops::Tile(2, repeats)(x);
      x.reshape({x.dim(0), x.dim(1) * x.dim(2), x.dim(3), x.dim(4)});
    }

    static void combine_heads(StorageView& x,
                              dim_t num_heads,
                              const Padder* padder = nullptr,
                              dim_t beam_size = 1) {
      // x has shape [batch_size, num_heads, time, head_dim]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(2);
      const dim_t depth = x.dim(3) * num_heads;

      if (time > 1) {
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }

      x.reshape({batch_size, time, depth});

      if (beam_size > 1)
        x.reshape({batch_size * beam_size, 1, depth});

      if (padder)
        padder->remove_padding(x);
    }

    static std::unique_ptr<RotaryEmbeddings> make_rotary_embeddings(const models::Model& model,
                                                                    const std::string& scope) {
      const dim_t rotary_dim = model.get_attribute_with_default<int32_t>(scope + "/rotary_dim", -1);
      if (rotary_dim < 0)
        return nullptr;

      const bool interleave = model.get_flag_with_default(scope + "/rotary_interleave", true);
      const float base = model.get_attribute_with_default<float>(scope + "/rotary_base", 10000.f);

      const auto scaling_type = model.get_enum_value<RotaryScalingType>(
        scope + "/rotary_scaling_type", -1);
      const auto scaling_factor = model.get_attribute_with_default<float>(
        scope + "/rotary_scaling_factor", 1.f);

      return std::make_unique<RotaryEmbeddings>(rotary_dim,
                                                interleave,
                                                scaling_type,
                                                scaling_factor,
                                                base);
    }

    FlashMultiHeadAttention::FlashMultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder,
                                           Alibi* alibi)
      : _tensor_parallel(model.tensor_parallel())
      , _num_heads(_tensor_parallel ? SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks()) : num_heads)
      , _self_attention(self_attention)
      , _is_decoder(is_decoder)
      , _linear(make_linear_layers(model, scope, self_attention))
      , _d_model(_tensor_parallel ? SAFE_DIVIDE(_linear.back().output_size(),  ScopedMPISetter::getNRanks()) : _linear.back().output_size())
      , _d_head(model.get_attribute_with_default<int32_t >(scope + "/head_dim", _d_model / _num_heads))
      , _pre_norm(pre_norm)
      , _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _rotary_embeddings(make_rotary_embeddings(model, scope))
      , _alibi(alibi)
      , _relative_attention_bias(model.get_variable_if_exists(scope + "/relative_attention_bias"))
      , _relative_position_keys(model.get_variable_if_exists(scope + "/relative_position_keys"))
      , _relative_position_values(model.get_variable_if_exists(scope + "/relative_position_values"))
      , _queries_scale(model.get_attribute_with_default<float>(
        scope + "/queries_scale",
        1.f / std::sqrt(static_cast<float>(_d_head))))
      , _multi_query(model.get_flag_with_default(scope + "/multi_query", false))
      , _num_heads_kv(_multi_query
                      ? 1
                      : (_tensor_parallel ? model.get_attribute_with_default<int32_t>(scope + "/num_heads_kv",
                                                                                      _num_heads * ScopedMPISetter::getNRanks()) / ScopedMPISetter::getNRanks()
                                          : model.get_attribute_with_default<int32_t>(scope + "/num_heads_kv", _num_heads)))
      , _merge_time_and_head_dims(_multi_query
                                  && !_relative_attention_bias
                                  && !_relative_position_keys
                                  && !_relative_position_values)
      , _cache_time_dim(_merge_time_and_head_dims ? 1 : 2)
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
    {
      if (_relative_position_keys)
        _maximum_relative_position = (_relative_position_keys->dim(0) - 1) / 2;
      else if (_relative_attention_bias)
        _maximum_relative_position = model.get_attribute<int32_t>(
          scope + "/relative_attention_max_distance");
      else
        _maximum_relative_position = 0;
    }

    DataType FlashMultiHeadAttention::output_type() const {
      return _linear.back().output_type();
    }

    dim_t FlashMultiHeadAttention::output_size() const {
      return _d_model;
    }

    void FlashMultiHeadAttention::operator()(const StorageView& queries,
                                             const StorageView& values,
                                             const StorageView* values_lengths,
                                             StorageView& output,
                                             StorageView* cached_keys,
                                             StorageView* cached_values,
                                             StorageView* attention,
                                             const Padder* queries_padder,
                                             const Padder* values_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset) const {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);

      const StorageView* q = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, queries_proj);
        q = &queries_proj;
      }

      _linear[0](*q, fused_proj);

      dim_t beam_size = 1;

      bool prefilling = (_sliding_window > 0 && values_lengths);

      if (!_self_attention) {
        queries_proj = std::move(fused_proj);

        if (cached_keys == nullptr || cached_keys->empty()) {
          _linear[1](values, fused_proj);

          if (_multi_query) {
            if (values_padder)
              values_padder->add_padding(fused_proj);
            ops::Split(2, {_d_head, _d_head})(fused_proj, keys_proj, values_proj);
          } else {
            split_heads(fused_proj, 2 * _num_heads, values_padder);
            ops::Split(1)(fused_proj, keys_proj, values_proj);
          }

          /*if (cached_keys != nullptr) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          }*/
        }

        if (queries_proj.dim(1) == 1 && cached_keys)
          beam_size = queries_proj.dim(0) / cached_keys->dim(0);

        if (_multi_query) {
          if (queries_padder)
            queries_padder->add_padding(queries_proj);
          queries_proj.reshape({queries_proj.dim(0) / beam_size, -1, _d_head});
        } else {
          split_heads(queries_proj, _num_heads, queries_padder, beam_size);
        }

      } else {

        if (_num_heads_kv < _num_heads) {
          if (queries_padder)
            queries_padder->add_padding(fused_proj);

          const ops::Split split_op(2, {_d_model, _num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(fused_proj, queries_proj, keys_proj, values_proj);

          split_heads(queries_proj, _num_heads);
          split_heads(keys_proj, _num_heads_kv);
          split_heads(values_proj, _num_heads_kv);

          //replicate_heads(keys_proj, _num_heads / _num_heads_kv);
          //replicate_heads(values_proj, _num_heads / _num_heads_kv);

        } else {
          split_heads(fused_proj, 3 * _num_heads, queries_padder);
          ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);
        }

        if (_rotary_embeddings) {
          _rotary_embeddings->apply(queries_proj, offset);
          _rotary_embeddings->apply(keys_proj, offset);
        }
      }

      if (cached_keys != nullptr) {
        if (cached_keys->empty()) {
          *cached_keys = std::move(keys_proj);
          *cached_values = std::move(values_proj);
        } else {
          const ops::Concat concat_op(_cache_time_dim);
          StorageView& tmp = fused_proj;  // Reuse storage.
          tmp = std::move(*cached_keys);
          concat_op({&tmp, &keys_proj}, *cached_keys);
          tmp = std::move(*cached_values);
          concat_op({&tmp, &values_proj}, *cached_values);

          if (!prefilling && _sliding_window > 0 && cached_keys->shape()[2] > _sliding_window) {
            // only for generation
            const ops::Slide slide_op(2, 1, cached_keys->shape()[2] - 1);
            slide_op(*cached_keys, tmp);
            *cached_keys = std::move(tmp);
            slide_op(*cached_values, tmp);
            *cached_values = std::move(tmp);
          }
        }
      }

      dim_t window_size_right = -1;
      dim_t window_size_left = -1;
      int device_id = ctranslate2::get_device_index(ctranslate2::Device::CUDA);
      auto dprops = ctranslate2::cuda::get_device_properties(device_id);
      // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
      bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
      bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
      // TODO: Thuc add this in model loading
      //TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
      // We will support Turing in the near future
      // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

      //auto q_dtype = q.dtype();
      //TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
      //            "FlashAttention only support fp16 and bf16 data type");
      //if (q_dtype == torch::kBFloat16) {
      //  TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
      //}
      //TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
      //TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

      //CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

      //TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
      //TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
      //TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

      const auto shape = queries.shape();
      const dim_t batch_size = shape[0];
      dim_t seqlen_q = shape[1];
      dim_t num_heads = shape[2];
      const dim_t head_size_og = shape[3];
      const dim_t seqlen_k = keys_proj.dim(1);
      const dim_t num_heads_k = keys_proj.dim(2);
      TENSOR_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256")
      TENSOR_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

      // causal=true is the same as causal=false in this case
      bool is_causal = false;
      if (seqlen_q == 1 && !_alibi) { is_causal = false; }
      if (is_causal) { window_size_right = 0; }

      // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
      // H/t Daniel Haziza
      const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0;
      if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        transpose_op(queries_proj.reshape({batch_size, num_heads_k, ngroups, head_size_og}), fused_proj);
        queries_proj = std::move(fused_proj);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
      }

      if (window_size_left >= seqlen_k) { window_size_left = -1; }
      if (window_size_right >= seqlen_k) { window_size_right = -1; }

      // init output
      StorageView context(dtype, device);  // Reuse storage.
      context.resize(queries_proj.shape());

      auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int head_size = round_multiple(head_size_og, 8);
      const int head_size_rounded = round_multiple(head_size, 32);
      const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
      const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

      // Otherwise the kernel will be launched from cuda:0 device
      // Cast to char to avoid compiler warning about narrowing
      //at::cuda::CUDAGuard device_guard{(char)q.get_device()};

      StorageView softmax_lse({batch_size, num_heads, seqlen_q}, dtype, device);
      if (return_normalized_attention) {
        attention->resize({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
      }

      Flash_fwd_params params;
      set_params_fprop(params,
                       batch_size,
                       seqlen_q, seqlen_k,
                       seqlen_q_rounded, seqlen_k_rounded,
                       num_heads, num_heads_k,
                       head_size, head_size_rounded,
                       &queries_proj, &keys_proj, &values_proj, &context,
                       /*cu_seqlens_q_d=*/nullptr,
                       /*cu_seqlens_k_d=*/nullptr,
                       /*seqused_k=*/nullptr,
                       return_normalized_attention ? attention->buffer() : /*p_ptr=*/nullptr,
                       softmax_lse.buffer(),
                       _queries_scale,
                       window_size_left,
                       window_size_right);

      // set params splitkv
      params.num_splits = 0;
      params.alibi_slopes_ptr = nullptr;

      auto stream = ctranslate2::cuda::get_cuda_stream();
      run_mha_fwd(params, stream);

      if (seqlenq_ngroups_swapped) {
        transpose_op(context, fused_proj);
        context = std::move(fused_proj);
        context.reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        //softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
      }

      if (prefilling && cached_keys && cached_keys->shape()[2] > _sliding_window) {
        // set only last sliding_window tokens to cached_keys and cached_values after computing attention
        const ops::Slide slide_op(2, cached_keys->shape()[2] - _sliding_window, _sliding_window);
        StorageView tmp(dtype, device);
        slide_op(*cached_keys, tmp);
        *cached_keys = std::move(tmp);
        slide_op(*cached_values, tmp);
        *cached_values = std::move(tmp);
      }
      combine_heads(context, _num_heads, queries_padder, beam_size);

      _linear.back()(context, output);
      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll ops_reduce_all(ops::ReduceAll::RED_OP::SUM);
        ops_reduce_all(output, tmp);
        output = std::move(tmp);
      }
      if (_layer_norm) {
        ops::Add()(queries, output, output);

        if (!_pre_norm)
          (*_layer_norm)(output, output);
      }
    }
  }
}