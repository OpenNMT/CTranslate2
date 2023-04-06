#include "ctranslate2/layers/attention.h"

#include <algorithm>
#include <cmath>

#include "dispatch.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t length, dim_t max_position, bool with_cache) {
      StorageView positions({with_cache ? 1 : length, length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      if (with_cache) {
        for (dim_t i = 0; i < length; ++i) {
          positions_data[i] = std::max(i - length + 1, -max_position) + max_position;
        }
      } else {
        for (dim_t i = 0; i < length; ++i) {
          auto* row = positions_data + i * length;
          for (dim_t j = 0; j < length; ++j) {
            row[j] = std::min(std::max(j - i, -max_position), max_position) + max_position;
          }
        }
      }

      return positions;
    }

    static StorageView get_relative_position_bucket(bool bidirectional,
                                                    dim_t query_length,
                                                    dim_t key_length,
                                                    dim_t num_buckets,
                                                    dim_t max_distance,
                                                    dim_t query_offset = 0) {
      StorageView relative_buckets({query_length, key_length}, DataType::INT32);

      if (bidirectional)
        num_buckets /= 2;

      const dim_t max_exact = num_buckets / 2;

      for (dim_t i = 0; i < query_length; ++i) {
        for (dim_t j = 0; j < key_length; ++j) {
          int32_t relative_position = j - (i + query_offset);
          int32_t& relative_bucket = relative_buckets.at<int32_t>(i * key_length + j);

          relative_bucket = 0;

          if (bidirectional) {
            if (relative_position > 0)
              relative_bucket += num_buckets;
            else
              relative_position = std::abs(relative_position);
          } else {
            relative_position = -std::min(relative_position, 0);
          }

          const bool is_small = relative_position < max_exact;

          if (!is_small) {
            relative_position = std::min(
              int32_t(float(max_exact)
                      + std::log(float(relative_position) / float(max_exact))
                      / std::log(float(max_distance) / float(max_exact))
                      * float(num_buckets - max_exact)),
              int32_t(num_buckets - 1));
          }

          relative_bucket += relative_position;

        }
      }

      return relative_buckets;
    }

    static StorageView compute_relative_bias(const StorageView& relative_attention_bias,
                                             dim_t query_length,
                                             dim_t key_length,
                                             dim_t max_distance,
                                             bool is_decoder,
                                             dim_t query_offset = 0) {
      const Device device = relative_attention_bias.device();
      const DataType dtype = relative_attention_bias.dtype();
      const dim_t num_buckets = relative_attention_bias.dim(0);

      StorageView relative_attention_bucket = get_relative_position_bucket(!is_decoder,
                                                                           query_length,
                                                                           key_length,
                                                                           num_buckets,
                                                                           max_distance,
                                                                           query_offset);

      StorageView values(dtype, device);
      ops::Gather()(relative_attention_bias, relative_attention_bucket.to(device), values);

      StorageView values_t(dtype, device);
      ops::Transpose({2, 0, 1})(values, values_t);

      return values_t;
    }

    StorageView build_alibi(dim_t batch_size,
                            dim_t num_heads,
                            dim_t query_max_length,
                            dim_t key_max_length,
                            const StorageView* key_lengths) {
      const float closest_power_of_2_f = std::pow(2.f, std::floor(std::log2f(num_heads)));
      const dim_t closest_power_of_2 = closest_power_of_2_f;

      const float base = std::pow(2.f, -std::pow(2.f, -(std::log2f(closest_power_of_2_f) - 3.f)));

      std::vector<float> slopes;
      slopes.reserve(closest_power_of_2);
      for (dim_t power = 1; power <= closest_power_of_2; ++power)
        slopes.emplace_back(std::pow(base, float(power)));

      if (closest_power_of_2 != num_heads) {
        const float extra_base = (
          std::pow(2.f, -std::pow(2.f, -(std::log2f(2 * closest_power_of_2_f) - 3.f))));
        const dim_t num_remaining_heads = std::min(
          closest_power_of_2, num_heads - closest_power_of_2);

        for (dim_t power = 1; power <= 2 * num_remaining_heads; power += 2)
          slopes.emplace_back(std::pow(extra_base, float(power)));
      }

      StorageView alibi({batch_size, num_heads, query_max_length, key_max_length});

      for (dim_t b = 0; b < batch_size; ++b) {
        for (dim_t h = 0; h < num_heads; ++h) {
          for (dim_t q = 0; q < query_max_length; ++q) {
            for (dim_t k = 0; k < key_max_length; ++k) {
              dim_t length = key_lengths ? key_lengths->scalar_at<int32_t>({b}) : key_max_length;
              alibi.at<float>({b, h, q, k}) = k >= length ? 0 : float(k) * slopes[h];
            }
          }
        }
      }

      return alibi;
    }

    static void matmul_with_relative_representations(const ops::MatMul& matmul_op,
                                                     const StorageView& a,
                                                     const StorageView& b,
                                                     StorageView& c) {
      const Device device = a.device();
      const DataType dtype = a.dtype();
      const dim_t batch = a.dim(0);
      const dim_t head = a.dim(1);
      const dim_t time = a.dim(2);

      StorageView a_t(dtype, device);
      ops::Transpose({2, 0, 1, 3})(a, a_t);
      a_t.reshape({time, batch * head, -1});

      StorageView c_t(dtype, device);
      matmul_op(a_t, b, c_t);
      c_t.reshape({time, batch, head, -1});
      ops::Transpose({1, 2, 0, 3})(c_t, c);
    }

    static void add_relative_representations(const StorageView& queries,
                                             const StorageView& relative_positions,
                                             const StorageView& relative_values,
                                             const ops::MatMul& matmul_op,
                                             StorageView& dot) {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      StorageView relative_representations(dtype, device);
      ops::Gather()(relative_values, relative_positions, relative_representations);

      StorageView dot_relative(dtype, device);
      matmul_with_relative_representations(matmul_op,
                                           queries,
                                           relative_representations,
                                           dot_relative);
      ops::Add()(dot_relative, dot, dot);
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    static inline void save_attention(StorageView& attention, StorageView weights, dim_t beam_size) {
      if (beam_size == 1)
        attention = std::move(weights);
      else {
        transpose_op(weights, attention);
        attention.reshape({-1, weights.dim(1), 1, weights.dim(-1)});
      }
    }

    static void dot_product_attention(const StorageView& queries,
                                      const StorageView& keys,
                                      const StorageView& values,
                                      const StorageView* alibi,
                                      const StorageView* values_lengths,
                                      const StorageView* relative_position_keys,
                                      const StorageView* relative_position_values,
                                      const StorageView* relative_attention_bias,
                                      dim_t maximum_relative_position,
                                      StorageView& output,
                                      StorageView* attention = nullptr,
                                      bool return_normalized_attention = true,
                                      float queries_scale = 1,
                                      bool is_decoder = false,
                                      bool with_cache = false,
                                      dim_t beam_size = 1) {
      PROFILE("dot_product_attention");

      std::unique_ptr<const StorageView> relative_positions;
      if (relative_position_keys || relative_position_values) {
        const dim_t max_time = keys.dim(2);
        relative_positions = std::make_unique<StorageView>(
          make_relative_positions(max_time,
                                  maximum_relative_position,
                                  with_cache).to(queries.device()));
      }

      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, queries_scale);
      keys_matmul(queries, keys, output);
      if (relative_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_position_keys,
                                     keys_matmul,
                                     output);

      if (relative_attention_bias) {
        const dim_t query_length = queries.dim(2);
        const dim_t key_length = keys.dim(2);
        const StorageView position_bias = compute_relative_bias(*relative_attention_bias,
                                                                query_length,
                                                                key_length,
                                                                maximum_relative_position,
                                                                is_decoder,
                                                                with_cache ? key_length - 1 : 0);

        DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(),
                                 primitives<D>::add_batch_broadcast(position_bias.data<T>(),
                                                                    output.data<T>(),
                                                                    position_bias.size(),
                                                                    output.size()));
      }

      if (alibi)
        ops::Add()(output, *alibi, output);

      StorageView attn(values.dtype(), values.device());
      ops::SoftMax()(output, values_lengths, attn);

      if (attention && !return_normalized_attention)
        save_attention(*attention, std::move(output), beam_size);

      const ops::MatMul values_matmul;
      values_matmul(attn, values, output);
      if (relative_position_values)
        add_relative_representations(attn,
                                     *relative_positions,
                                     *relative_position_values,
                                     values_matmul,
                                     output);

      if (attention && return_normalized_attention)
        save_attention(*attention, std::move(attn), beam_size);
    }

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

    static std::vector<Dense> make_linear_layers(const models::Model& model,
                                                 const std::string& scope,
                                                 bool self_attention) {
      const dim_t num_linear_layers = self_attention ? 2 : 3;
      std::vector<Dense> layers;
      layers.reserve(num_linear_layers);
      for (dim_t i = 0; i < num_linear_layers; ++i)
        layers.emplace_back(model, scope + "/linear_" + std::to_string(i));
      return layers;
    }

    static std::unique_ptr<RotaryEmbeddings> make_rotary_embeddings(const models::Model& model,
                                                                    const std::string& scope) {
      const dim_t rotary_dim = model.get_attribute_with_default<int32_t>(scope + "/rotary_dim", -1);
      if (rotary_dim < 0)
        return nullptr;
      return std::make_unique<RotaryEmbeddings>(rotary_dim);
    }


    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder)
      : _num_heads(num_heads)
      , _self_attention(self_attention)
      , _is_decoder(is_decoder)
      , _linear(make_linear_layers(model, scope, self_attention))
      , _d_model(_linear.back().output_size())
      , _pre_norm(pre_norm)
      , _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _rotary_embeddings(make_rotary_embeddings(model, scope))
      , _relative_attention_bias(model.get_variable_if_exists(scope + "/relative_attention_bias"))
      , _relative_position_keys(model.get_variable_if_exists(scope + "/relative_position_keys"))
      , _relative_position_values(model.get_variable_if_exists(scope + "/relative_position_values"))
      , _queries_scale(model.get_attribute_with_default<float>(
                         scope + "/queries_scale",
                         1.f / std::sqrt(static_cast<float>(_d_model / _num_heads))))
    {
      if (_relative_position_keys)
        _maximum_relative_position = (_relative_position_keys->dim(0) - 1) / 2;
      else if (_relative_attention_bias)
        _maximum_relative_position = model.get_attribute<int32_t>(
          scope + "/relative_attention_max_distance");
      else
        _maximum_relative_position = 0;
    }

    DataType MultiHeadAttention::output_type() const {
      return _linear.back().output_type();
    }

    dim_t MultiHeadAttention::output_size() const {
      return _d_model;
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView& values,
                                        const StorageView* values_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        StorageView* attention,
                                        const Padder* queries_padder,
                                        const Padder* values_padder,
                                        bool return_normalized_attention,
                                        const StorageView* alibi) const {
      PROFILE("MultiHeadAttention");
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

      if (!_self_attention) {
        queries_proj = std::move(fused_proj);

        if (cached_keys == nullptr || cached_keys->empty()) {
          _linear[1](values, fused_proj);
          split_heads(fused_proj, 2 * _num_heads, values_padder);
          ops::Split(1)(fused_proj, keys_proj, values_proj);

          if (cached_keys != nullptr) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          }
        }

        if (queries_proj.dim(1) == 1 && cached_keys)
          beam_size = queries_proj.dim(0) / cached_keys->dim(0);
        split_heads(queries_proj, _num_heads, queries_padder, beam_size);

      } else {
        split_heads(fused_proj, 3 * _num_heads, queries_padder);
        ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);

        if (_rotary_embeddings) {
          const dim_t offset = cached_keys && !cached_keys->empty() ? cached_keys->dim(2) : 0;
          _rotary_embeddings->apply(queries_proj, offset);
          _rotary_embeddings->apply(keys_proj, offset);
        }

        if (cached_keys != nullptr) {
          if (cached_keys->empty()) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          } else {
            StorageView& tmp = fused_proj;  // Reuse storage.
            tmp = std::move(*cached_keys);
            ops::Concat(2)({&tmp, &keys_proj}, *cached_keys);
            tmp = std::move(*cached_values);
            ops::Concat(2)({&tmp, &values_proj}, *cached_values);
          }
        }
      }

      if (cached_keys) {
        keys_proj.shallow_copy(*cached_keys);
        values_proj.shallow_copy(*cached_values);
      }

      StorageView& context = fused_proj;  // Reuse storage.
      dot_product_attention(queries_proj,
                            keys_proj,
                            values_proj,
                            alibi,
                            values_lengths,
                            _relative_position_keys,
                            _relative_position_values,
                            _relative_attention_bias,
                            _maximum_relative_position,
                            context,
                            attention,
                            return_normalized_attention,
                            _queries_scale,
                            _is_decoder,
                            bool(cached_keys),
                            beam_size);

      combine_heads(context, _num_heads, queries_padder, beam_size);
      _linear.back()(context, output);

      if (_layer_norm) {
        ops::Add()(queries, output, output);

        if (!_pre_norm)
          (*_layer_norm)(output, output);
      }
    }

    StorageView MultiHeadAttention::prepare_length_mask(const StorageView& lengths,
                                                        const dim_t num_heads,
                                                        const dim_t num_queries,
                                                        const bool mask_future) {
      const Device device = lengths.device();
      const dim_t batch_size = lengths.size();
      StorageView mask({batch_size, num_heads, num_queries}, lengths.dtype(), device);
      DEVICE_DISPATCH(device, (primitives<D>::prepare_length_mask(lengths.data<int32_t>(),
                                                                  batch_size,
                                                                  num_heads,
                                                                  num_queries,
                                                                  mask_future,
                                                                  mask.data<int32_t>())));
      return mask;
    }


    static void apply_signal(StorageView& x, const StorageView& signal, const dim_t offset) {
      const dim_t depth = x.dim(-1);
      const dim_t time = x.dim(-2);
      DEVICE_AND_TYPE_DISPATCH(x.device(), x.dtype(),
                               primitives<D>::mul_batch_broadcast(signal.data<T>() + offset * depth,
                                                                  x.data<T>(),
                                                                  time * depth,
                                                                  x.size()));
    }

    RotaryEmbeddings::RotaryEmbeddings(const dim_t dim,
                                       const dim_t num_initial_positions,
                                       const float base)
      : _dim(dim)
      , _num_initial_positions(num_initial_positions)
      , _base(base)
    {
    }

    void RotaryEmbeddings::apply(StorageView& x, const dim_t offset) {
      if (_dim == 0) {
        apply_impl(x, offset);

      } else {
        const ops::Split split_op(-1, {_dim, x.dim(-1) - _dim});
        const ops::Concat concat_op(-1);
        const Device device = x.device();
        const DataType dtype = x.dtype();

        StorageView x_rot(dtype, device);
        StorageView x_pass(dtype, device);
        split_op(x, x_rot, x_pass);
        apply_impl(x_rot, offset);
        concat_op({&x_rot, &x_pass}, x);
      }
    }

    void RotaryEmbeddings::apply_impl(StorageView& x, const dim_t offset) {
      const Device device = x.device();
      const DataType dtype = x.dtype();
      const dim_t max_time = x.dim(-2);

      if (!_sin || offset + max_time > _sin.dim(0)) {
        const dim_t num_positions = std::max(offset + max_time, _num_initial_positions);
        initialize(num_positions, x.dim(-1), device, dtype);
      }

      StorageView x_rot(dtype, device);
      rotate_every_two(x, x_rot);

      apply_signal(x, _cos, offset);
      apply_signal(x_rot, _sin, offset);

      ops::Add()(x, x_rot, x);
    }

    void RotaryEmbeddings::rotate_every_two(StorageView& x, StorageView& y) const {
      const Device device = x.device();
      const DataType dtype = x.dtype();
      const Shape original_shape = x.shape();

      Shape new_shape = original_shape;
      new_shape.push_back(2);
      new_shape[new_shape.size() - 2] /= 2;
      x.reshape(new_shape);

      StorageView x1(dtype, device);
      StorageView x2(dtype, device);
      ops::Split(-1)(x, x1, x2);

      StorageView zero(x2.shape(), dtype, device);
      zero.zero();
      ops::Sub()(zero, x2, x2);

      ops::Concat(-1)({&x2, &x1}, y);

      y.reshape(original_shape);
      x.reshape(original_shape);
    }

    void RotaryEmbeddings::initialize(const dim_t num_positions,
                                      const dim_t dim,
                                      const Device device,
                                      const DataType dtype) {
      StorageView inv_freq({1, dim / 2});
      for (dim_t i = 0; i < inv_freq.size(); ++i)
        inv_freq.at<float>(i) = 1.f / std::pow(_base, float(i * 2) / float(dim));

      StorageView t({num_positions, 1});
      for (dim_t i = 0; i < t.size(); ++i)
        t.at<float>(i) = i;

      StorageView freqs;
      ops::MatMul()(t, inv_freq, freqs);

      freqs.expand_dims(-1);

      StorageView emb;
      ops::Concat(-1)({&freqs, &freqs}, emb);

      emb.reshape({num_positions, dim});

      StorageView sin_tmp;
      ops::Sin()(emb, sin_tmp);
      sin_tmp.move_to(device, dtype);
      _sin = std::move(sin_tmp);

      StorageView cos_tmp;
      ops::Cos()(emb, cos_tmp);
      cos_tmp.move_to(device, dtype);
      _cos = std::move(cos_tmp);
    }

  }
}
