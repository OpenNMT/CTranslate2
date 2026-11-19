#include "layers/drama/drama_tower.h"
#include "ctranslate2/ops/rms_norm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {

    int32_t sample_drama_token(const std::vector<float>& logits,
                               const DramaSamplingOptions& options, std::mt19937& rng) {
      if (logits.empty() || logits.size() > size_t(std::numeric_limits<int32_t>::max())
          || !std::isfinite(options.temperature) || options.temperature <= 0
          || !std::isfinite(options.top_p) || options.top_p < 0 || options.top_p > 1
          || options.top_k <= 0)
        throw std::invalid_argument(
            "Drama sampling requires finite positive temperature/top_k and top_p in [0,1]");
      for (const auto value : logits) {
        if (!std::isfinite(value))
          throw std::invalid_argument("Drama sampling logits must be finite");
      }
      const float temperature = std::max(options.temperature, 1e-5f);
      std::vector<int32_t> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      std::vector<float> values(logits.size()), probabilities(logits.size());
      for (size_t i = 0; i < logits.size(); ++i)
        values[i] = logits[i] / temperature;
      std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return values[a] > values[b];
      });
      float sum = 0;
      for (size_t rank = 0; rank < order.size(); ++rank)
        sum += probabilities[rank] = std::exp(values[order[rank]] - values[order[0]]);
      double cumulative = 0;
      float kept_sum = 0;
      for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / sum;
        if (rank && (float(cumulative) > options.top_p || rank >= size_t(options.top_k)))
          probabilities[rank] = 0;
        kept_sum += probabilities[rank];
      }
      // Match the tested Torch CPU exponential stream in sorted-rank order,
      // including masked ranks. Different platforms may round softmax differently.
      float best = -1;
      int32_t chosen = order[0];
      for (size_t rank = 0; rank < order.size(); ++rank) {
        const uint64_t hi = rng(), lo = rng();
        const uint64_t bits = (hi << 32) | lo;
        const double uniform = double(bits & ((uint64_t(1) << 53) - 1)) * 0x1.0p-53;
        const float noise = float(-std::log1p(-uniform));
        const float score = (probabilities[rank] / kept_sum) / noise;
        if (score > best) {
          best = score;
          chosen = order[rank];
        }
      }
      return chosen;
    }

    namespace {

      void require_cpu_float(const StorageView& tensor) {
        if (tensor.device() != Device::CPU || tensor.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("DramaTower requires CPU float32 activations");
      }

      bool overlaps(const StorageView& a, const StorageView& b) {
        if (!a.buffer() || !b.buffer() || !a.reserved_memory() || !b.reserved_memory())
          return false;
        const auto x = reinterpret_cast<uintptr_t>(a.buffer()),
                   y = reinterpret_cast<uintptr_t>(b.buffer());
        return x <= y ? y - x < static_cast<uintptr_t>(a.reserved_memory())
                      : x - y < static_cast<uintptr_t>(b.reserved_memory());
      }

      void validate_embedding(const models::Model& model, const std::string& scope,
                              const StorageView& weights) {
        if (weights.device() != Device::CPU
            || (weights.dtype() != DataType::FLOAT32 && weights.dtype() != DataType::INT8
                && weights.dtype() != DataType::INT16))
          throw std::invalid_argument("DramaTower embedding type is unsupported");
        if (weights.dtype() != DataType::FLOAT32) {
          const auto* scale = model.get_variable_if_exists(scope + "/weight_scale");
          if (!scale || scale->device() != Device::CPU || scale->dtype() != DataType::FLOAT32
              || (!scale->is_scalar() && scale->shape() != Shape{weights.dim(0)}))
            throw std::invalid_argument(
                "DramaTower quantized embeddings need float32 per-row or scalar scales");
          for (dim_t i = 0; i < scale->size(); ++i) {
            if (!std::isfinite(scale->data<float>()[i]) || scale->data<float>()[i] <= 0)
              throw std::invalid_argument(
                  "DramaTower embedding scales must be finite and positive");
          }
        }
      }

    }

    struct DramaTower::TowerLayer {
      const StorageView& embedding;
      Embeddings embedding_lookup;
      const StorageView& enorm;
      const StorageView& hnorm;
      const StorageView& attention_norm;
      const StorageView& ffn_norm;
      const StorageView& final_norm;
      DramaProjection eh_proj, gate, up, down;
      GatedSelfAttention attention;
      bool resident_gpu;
      int gpu_device_index;
      std::vector<std::shared_ptr<const DramaDeviceWeights>> _shared_gpu_parameters;
      StorageView gpu_attention_norm, gpu_ffn_norm, gpu_final_norm;
      float epsilon;
      dim_t dim;

      TowerLayer(const models::Model& model, const std::string& scope, const DramaTowerConfig& cfg,
                 bool shared, const DramaExecutionOptions& options)
        : embedding(model.get_variable(
              scope + (shared ? "/semantic_embedding/weight" : "/codebook_embedding/weight")))
        , embedding_lookup(model, scope + (shared ? "/semantic_embedding" : "/codebook_embedding"))
        , enorm(model.get_variable(scope + "/enorm/weight"))
        , hnorm(model.get_variable(scope + "/hnorm/weight"))
        , attention_norm(model.get_variable(scope + "/transformer_layer/attention_norm/weight"))
        , ffn_norm(model.get_variable(scope + "/transformer_layer/ffn_norm/weight"))
        , final_norm(model.get_variable(scope + "/final_layernorm/weight"))
        , eh_proj(model, scope + "/eh_proj", options)
        , gate(model, scope + "/transformer_layer/feed_forward/gate_proj", options)
        , up(model, scope + "/transformer_layer/feed_forward/up_proj", options)
        , down(model, scope + "/transformer_layer/feed_forward/down_proj", options)
        , attention(model, scope + "/transformer_layer/attention", cfg.attention, options)
        , resident_gpu(options.resident_gpu())
        , gpu_device_index(resident_gpu ? options.device_index : 0)
        , epsilon(cfg.attention.norm_epsilon)
        , dim(attention.output_size()) {
        if (embedding.shape() != Shape{cfg.vocab_size, dim})
          throw std::invalid_argument("DramaTower layer embedding dimensions are incompatible");
        validate_embedding(model, scope + (shared ? "/semantic_embedding" : "/codebook_embedding"),
                           embedding);
        for (const auto* gamma : {&enorm, &hnorm, &attention_norm, &ffn_norm, &final_norm}) {
          require_cpu_float(*gamma);
          if (gamma->shape() != Shape{dim})
            throw std::invalid_argument("DramaTower norm dimensions are incompatible");
          for (dim_t i = 0; i < gamma->size(); ++i) {
            if (!std::isfinite(gamma->data<float>()[i]))
              throw std::invalid_argument("DramaTower norm coefficients must be finite");
          }
        }
        auto weight = [&](const std::string& path, dim_t rows, dim_t cols) {
          const auto& value = model.get_variable(path + "/weight");
          if (value.device() != Device::CPU || value.shape() != Shape{rows, cols}
              || model.get_variable_if_exists(path + "/weight_packed"))
            throw std::invalid_argument(
                "DramaTower requires compatible unpacked projection matrices");
        };
        if (gate.output_size() <= 0 || gate.output_type() != DataType::FLOAT32)
          throw std::invalid_argument("DramaTower requires float32 activation projections");
        weight(scope + "/eh_proj", dim, 2 * dim);
        weight(scope + "/transformer_layer/feed_forward/gate_proj", gate.output_size(), dim);
        weight(scope + "/transformer_layer/feed_forward/up_proj", gate.output_size(), dim);
        weight(scope + "/transformer_layer/feed_forward/down_proj", dim, gate.output_size());
        if (resident_gpu) {
          const ScopedDeviceSetter setter(Device::CUDA, gpu_device_index);
          gpu_attention_norm = upload_drama_weight(attention_norm, options, _shared_gpu_parameters);
          gpu_ffn_norm = upload_drama_weight(ffn_norm, options, _shared_gpu_parameters);
          gpu_final_norm = upload_drama_weight(final_norm, options, _shared_gpu_parameters);
        }
      }

      void forward_gpu(const StorageView& joined, GatedSelfAttentionCache& cache,
                       StorageView& output) const {
        const ScopedDeviceSetter setter(Device::CUDA, gpu_device_index);
        auto gpu_joined = joined.to(Device::CUDA);
        StorageView gm(DataType::FLOAT32, Device::CUDA), gn(DataType::FLOAT32, Device::CUDA),
            ga(DataType::FLOAT32, Device::CUDA);
        eh_proj(gpu_joined, gm);
        ops::RMSNorm(epsilon, true)(gpu_attention_norm, gm, gn);
        attention(gn, cache, ga);
        ops::Add()(gm, ga, gm);
        ops::RMSNorm(epsilon, true)(gpu_ffn_norm, gm, gn);
        StorageView gates(DataType::FLOAT32, Device::CUDA), ups(DataType::FLOAT32, Device::CUDA);
        StorageView activated(DataType::FLOAT32, Device::CUDA),
            projected(DataType::FLOAT32, Device::CUDA);
        gate(gn, gates);
        up(gn, ups);
        ops::get_activation_op(ops::ActivationType::Swish)(gates, activated);
        ops::Mul()(activated, ups, activated);
        down(activated, projected);
        ops::Add()(gm, projected, gm);
        ops::RMSNorm(epsilon, true)(gpu_final_norm, gm, gn);
        output.copy_from(gn);
        synchronize_stream(Device::CUDA);
      }

      void operator()(const StorageView& hidden, const StorageView& next_frames, dim_t code_column,
                      int32_t semantic_begin, int32_t semantic_end, const StorageView* shared_input,
                      GatedSelfAttentionCache& cache, StorageView& output) const {
        const dim_t rows = hidden.dim(0), columns = next_frames.dim(1);
        StorageView e(hidden.shape(), 0.f);
        StorageView ids({rows}, DataType::INT32), selected;
        for (dim_t t = 0; t < rows; ++t)
          ids.data<int32_t>()[t] = next_frames.data<int32_t>()[t * columns + code_column];
        if (rows)
          embedding_lookup(ids, selected);
        for (dim_t t = 0; t < rows; ++t) {
          const int32_t main = next_frames.data<int32_t>()[t * columns];
          const bool semantic = main >= semantic_begin && main <= semantic_end;
          for (dim_t d = 0; d < dim; ++d)
            e.data<float>()[t * dim + d] =
                (semantic ? selected.data<float>()[t * dim + d] : 0.f)
                + (shared_input ? shared_input->data<float>()[t * dim + d] : 0.f);
        }
        StorageView en, hn;
        ops::RMSNorm(epsilon, true)(enorm, e, en);
        ops::RMSNorm(epsilon, true)(hnorm, hidden, hn);
        StorageView joined;
        ops::Concat(-1)({&en, &hn}, joined);
        StorageView mixed;
        if (rows == 0) {
          output.resize_as(hidden);
          return;
        }
        if (resident_gpu) {
          forward_gpu(joined, cache, output);
          return;
        }
        eh_proj(joined, mixed);
        StorageView normalized, attended;
        ops::RMSNorm(epsilon, true)(attention_norm, mixed, normalized);
        attention(normalized, cache, attended);
        ops::Add()(mixed, attended, mixed);
        ops::RMSNorm(epsilon, true)(ffn_norm, mixed, normalized);
        StorageView gates, ups, activated, projected;
        gate(normalized, gates);
        up(normalized, ups);
        ops::get_activation_op(ops::ActivationType::Swish)(gates, activated);
        ops::Mul()(activated, ups, activated);
        down(activated, projected);
        ops::Add()(mixed, projected, mixed);
        ops::RMSNorm(epsilon, true)(final_norm, mixed, output);
      }
    };

    DramaTower::DramaTower(const models::Model& model, const std::string& scope,
                           DramaTowerConfig config, const DramaExecutionOptions& options)
      : _config(config)
      , _embeddings(model.get_variable(scope + "/codebook_embeddings/weight"))
      , _embedding_lookup(model, scope + "/codebook_embeddings") {
      const auto& cfg = _config;
      if (cfg.num_codebooks < 2 || cfg.vocab_size <= 0 || cfg.shared_layers < 1
          || cfg.vq_backbone_codebooks < 0 || cfg.vq_backbone_codebooks >= cfg.num_codebooks
          || cfg.semantic_begin < 0 || cfg.semantic_end < cfg.semantic_begin || cfg.eos_id < 0
          || int64_t(cfg.semantic_end) - cfg.semantic_begin + 1 > cfg.vocab_size
          || is_semantic(cfg.eos_id) || _embeddings.rank() != 2
          || _embeddings.dim(0) != cfg.num_codebooks * cfg.vocab_size || _embeddings.dim(1) <= 0
          || !cfg.attention.rotary_interleaved || !cfg.attention.zero_centered_qk_norm)
        throw std::invalid_argument(
            "DramaTower configuration or codebook embedding dimensions are incompatible");
      validate_embedding(model, scope + "/codebook_embeddings", _embeddings);
      for (dim_t i = 0; i < cfg.shared_layers; ++i) {
        auto layer = std::make_unique<TowerLayer>(
            model, scope + "/shared_layers_" + std::to_string(i), cfg, true, options);
        if (layer->dim != output_size())
          throw std::invalid_argument("DramaTower shared hidden dimension is incompatible");
        _shared.emplace_back(std::move(layer));
      }
      for (dim_t i = 0; i < cfg.num_codebooks - 1; ++i) {
        auto layer = std::make_unique<TowerLayer>(
            model, scope + "/residual_layers_" + std::to_string(i), cfg, false, options);
        if (layer->dim != output_size())
          throw std::invalid_argument("DramaTower residual hidden dimension is incompatible");
        _residual.emplace_back(std::move(layer));
        const auto head_scope = scope + "/heads_" + std::to_string(i);
        const auto& weight = model.get_variable(head_scope + "/weight");
        if (weight.shape() != Shape{cfg.vocab_size, output_size()}
            || model.get_variable_if_exists(head_scope + "/weight_packed"))
          throw std::invalid_argument("DramaTower requires compatible unpacked head matrices");
        _heads.emplace_back(std::make_unique<DramaProjection>(model, head_scope, options));
      }
      _resident_gpu = options.resident_gpu();
      if (_resident_gpu)
        _gpu_device_index = options.device_index;
    }

    DramaTower::~DramaTower() = default;

    DataType DramaTower::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t DramaTower::output_size() const {
      return _embeddings.dim(1);
    }

    bool DramaTower::is_semantic(int32_t token) const {
      return token >= _config.semantic_begin && token <= _config.semantic_end;
    }

    void DramaTower::initialize_state(DramaTowerState& state, dim_t max_tokens) const {
      if (max_tokens <= 0)
        throw std::invalid_argument("DramaTower requires positive cache capacity");
      const ScopedDeviceSetter setter(_resident_gpu ? Device::CUDA : Device::CPU,
                                      _gpu_device_index);
      DramaTowerState fresh;
      fresh.capacity = max_tokens;
      auto allocate = [&](size_t count, std::vector<GatedSelfAttentionCache>& caches) {
        for (size_t i = 0; i < count; ++i) {
          const Shape shape{max_tokens, _config.attention.num_kv_heads, _config.attention.head_dim};
          const auto device = _resident_gpu ? Device::CUDA : Device::CPU;
          caches.push_back({StorageView(shape, DataType::FLOAT32, device),
                            StorageView(shape, DataType::FLOAT32, device), 0});
        }
      };
      allocate(_shared.size(), fresh.shared);
      allocate(_residual.size(), fresh.residual);
      state = std::move(fresh);
    }

    void DramaTower::validate_frames(const StorageView& frames, dim_t rows) const {
      if (frames.device() != Device::CPU || frames.dtype() != DataType::INT32
          || frames.shape() != Shape{rows, _config.num_codebooks + 1})
        throw std::invalid_argument("DramaTower frames must be CPU int32 [time,1+num_codebooks]");
      for (dim_t t = 0; t < rows; ++t)
        for (dim_t c = 1; c <= _config.num_codebooks; ++c) {
          const auto code = frames.data<int32_t>()[t * (_config.num_codebooks + 1) + c];
          if (code < 0 || code >= _config.vocab_size)
            throw std::invalid_argument("DramaTower frame code is out of range");
        }
    }

    void DramaTower::validate_hidden(const StorageView& hidden) const {
      require_cpu_float(hidden);
      if (hidden.rank() != 2 || hidden.dim(1) != output_size())
        throw std::invalid_argument("DramaTower hidden must be [time,dim]");
      for (dim_t i = 0; i < hidden.size(); ++i) {
        if (!std::isfinite(hidden.data<float>()[i]))
          throw std::invalid_argument("DramaTower hidden values must be finite");
      }
    }

    void DramaTower::validate_state(const DramaTowerState& state, dim_t rows) const {
      if (state.shared.size() != _shared.size() || state.residual.size() != _residual.size()
          || state.capacity <= 0 || state.length < 0 || state.length > state.capacity
          || rows > state.capacity - state.length)
        throw std::invalid_argument("DramaTower state/context limit is incompatible");
      for (const auto* caches : {&state.shared, &state.residual})
        for (const auto& cache : *caches) {
          for (const auto* tensor : {&cache.keys, &cache.values})
            if (tensor->dtype() != DataType::FLOAT32
                || tensor->device() != (_resident_gpu ? Device::CUDA : Device::CPU)
                || (_resident_gpu && tensor->device_index() != _gpu_device_index))
              throw std::invalid_argument("DramaTower cache device/type is incompatible");
          if (cache.length != state.length
              || cache.keys.shape()
                     != Shape{state.capacity, _config.attention.num_kv_heads,
                              _config.attention.head_dim}
              || cache.values.shape() != cache.keys.shape())
            throw std::invalid_argument(
                "DramaTower temporal cache position/dimensions are incompatible");
        }
    }

    void DramaTower::validate_state_aliases(const DramaTowerState& state,
                                            const std::vector<const StorageView*>& tensors) const {
      std::vector<const StorageView*> allocations;
      for (const auto* caches : {&state.shared, &state.residual})
        for (const auto& cache : *caches)
          for (const auto* tensor : {&cache.keys, &cache.values}) {
            for (const auto* input : tensors) {
              if (input == tensor || overlaps(*input, *tensor))
                throw std::invalid_argument(
                    "DramaTower temporal caches must not alias inputs/outputs");
            }
            for (const auto* prior : allocations) {
              if (overlaps(*prior, *tensor))
                throw std::invalid_argument("DramaTower temporal caches must not alias each other");
            }
            allocations.push_back(tensor);
          }
    }

    StorageView DramaTower::embedding_sum(const StorageView& frames, dim_t first, dim_t end) const {
      StorageView result({frames.dim(0), output_size()}, 0.f);
      const dim_t count = end - first;
      if (!count || !frames.dim(0))
        return result;
      StorageView ids({frames.dim(0) * count}, DataType::INT32), selected;
      for (dim_t t = 0; t < frames.dim(0); ++t)
        for (dim_t c = first; c < end; ++c) {
          const dim_t code = frames.data<int32_t>()[t * (_config.num_codebooks + 1) + 1 + c];
          ids.data<int32_t>()[t * count + c - first] =
              static_cast<int32_t>(c * _config.vocab_size + code);
        }
      _embedding_lookup(ids, selected);
      for (dim_t t = 0; t < frames.dim(0); ++t)
        for (dim_t c = first; c < end; ++c)
          for (dim_t d = 0; d < output_size(); ++d)
            result.data<float>()[t * output_size() + d] +=
                selected.data<float>()[((t * count) + c - first) * output_size() + d];
      return result;
    }

    void DramaTower::embed_backbone(const StorageView& frames, const StorageView& text_embeddings,
                                    StorageView& output) const {
      validate_hidden(text_embeddings);
      validate_frames(frames, text_embeddings.dim(0));
      require_cpu_float(output);
      if (&text_embeddings == &output || overlaps(text_embeddings, output)
          || overlaps(frames, output) || overlaps(_embeddings, output))
        throw std::invalid_argument("DramaTower embedding output must not alias inputs/weights");
      auto extra = embedding_sum(frames, 1, 1 + _config.vq_backbone_codebooks);
      const float divisor = std::sqrt(float(1 + _config.vq_backbone_codebooks));
      for (dim_t t = 0; t < text_embeddings.dim(0); ++t) {
        const bool semantic = is_semantic(frames.data<int32_t>()[t * (_config.num_codebooks + 1)]);
        for (dim_t d = 0; d < output_size(); ++d) {
          const dim_t i = t * output_size() + d;
          extra.data<float>()[i] =
              semantic ? (text_embeddings.data<float>()[i] + extra.data<float>()[i]) / divisor
                       : text_embeddings.data<float>()[i];
        }
      }
      output = std::move(extra);
    }

    void DramaTower::run_shared(const StorageView& hidden, const StorageView& frames,
                                const StorageView& next_frames, DramaTowerState& state,
                                StorageView& output) const {
      auto unseen = embedding_sum(frames, 1 + _config.vq_backbone_codebooks, _config.num_codebooks);
      for (dim_t t = 0; t < frames.dim(0); ++t)
        if (!is_semantic(frames.data<int32_t>()[t * (_config.num_codebooks + 1)]))
          for (dim_t d = 0; d < output_size(); ++d)
            unseen.data<float>()[t * output_size() + d] = 0;
      StorageView current(hidden);
      for (size_t i = 0; i < _shared.size(); ++i) {
        StorageView next;
        (*_shared[i])(current, next_frames, 1, _config.semantic_begin, _config.semantic_end,
                      i == 0 ? &unseen : nullptr, state.shared[i], next);
        current = std::move(next);
      }
      output = std::move(current);
    }

    void DramaTower::prefill(const StorageView& hidden, const StorageView& frames,
                             const StorageView& next_frames, DramaTowerState& state) const {
      validate_hidden(hidden);
      validate_frames(frames, hidden.dim(0));
      validate_frames(next_frames, hidden.dim(0));
      validate_state(state, hidden.dim(0));
      validate_state_aliases(state, {&hidden, &frames, &next_frames, &_embeddings});
      StorageView current;
      run_shared(hidden, frames, next_frames, state, current);
      for (size_t i = 0; i < _residual.size(); ++i) {
        StorageView next;
        (*_residual[i])(current, next_frames, i + 1, _config.semantic_begin, _config.semantic_end,
                        nullptr, state.residual[i], next);
        current = std::move(next);
      }
      state.length += hidden.dim(0);
    }

    void DramaTower::decode_greedy(const StorageView& logits, const StorageView& hidden,
                                   const StorageView& previous_frame, DramaTowerState& state,
                                   StorageView& next_frame) const {
      decode_impl(logits, hidden, previous_frame, state, next_frame, nullptr, nullptr, nullptr);
    }

    void DramaTower::decode_sample(const StorageView& logits, const StorageView& hidden,
                                   const StorageView& previous_frame, DramaTowerState& state,
                                   const std::vector<int32_t>& history_main_tokens,
                                   const DramaSamplingOptions& options, std::mt19937& rng,
                                   StorageView& next_frame) const {
      decode_impl(logits, hidden, previous_frame, state, next_frame, &history_main_tokens, &options,
                  &rng);
    }

    int32_t DramaTower::select_main_token(const StorageView& logits,
                                          const std::vector<int32_t>* history_main_tokens,
                                          const DramaSamplingOptions* options,
                                          std::mt19937* rng) const {
      int32_t main = _config.eos_id;
      float best = logits.data<float>()[main];
      for (int64_t candidate = _config.semantic_begin; candidate <= _config.semantic_end;
           ++candidate) {
        const float value = logits.data<float>()[candidate];
        if (!std::isfinite(value) || !std::isfinite(best))
          throw std::invalid_argument("DramaTower candidate logits must be finite");
        if (value > best || (value == best && candidate < main)) {
          main = static_cast<int32_t>(candidate);
          best = value;
        }
      }
      if (options) {
        std::vector<int32_t> candidates{_config.eos_id};
        for (int64_t candidate = _config.semantic_begin; candidate <= _config.semantic_end;
             ++candidate)
          candidates.push_back(static_cast<int32_t>(candidate));
        std::sort(candidates.begin(), candidates.end());
        std::vector<float> candidate_logits;
        for (const auto candidate : candidates)
          candidate_logits.push_back(logits.data<float>()[candidate]);
        main = candidates[sample_drama_token(candidate_logits, *options, *rng)];
        DramaSamplingOptions retry;
        retry.temperature = 1.f;
        retry.top_k = 30;
        retry.top_p = .95f;
        // The official graph draws the retry unconditionally. Only semantic
        // repeats in the last16 main tokens use it; EOS is never resampled.
        const auto replacement = candidates[sample_drama_token(candidate_logits, retry, *rng)];
        const size_t first =
            history_main_tokens->size() > 16 ? history_main_tokens->size() - 16 : 0;
        if (is_semantic(main)
            && std::find(history_main_tokens->begin() + first, history_main_tokens->end(), main)
                   != history_main_tokens->end())
          main = replacement;
      }
      return main;
    }

    void DramaTower::decode_residual_codes(const StorageView& hidden,
                                           const StorageView& previous_frame, StorageView& frame,
                                           DramaTowerState& state,
                                           const DramaSamplingOptions* options,
                                           std::mt19937* rng) const {
      StorageView current;
      run_shared(hidden, previous_frame, frame, state, current);
      for (size_t i = 0; i < _residual.size(); ++i) {
        StorageView next, code_logits;
        (*_residual[i])(current, frame, i + 1, _config.semantic_begin, _config.semantic_end,
                        nullptr, state.residual[i], next);
        _heads[i]->operator()(next, code_logits);
        int32_t chosen = 0;
        for (dim_t code = 0; code < _config.vocab_size; ++code) {
          const float value = code_logits.data<float>()[code];
          if (!std::isfinite(value))
            throw std::runtime_error(
                "DramaTower code logits are non-finite; reset state before retry");
          if (value > code_logits.data<float>()[chosen])
            chosen = static_cast<int32_t>(code);
        }
        if (options) {
          const std::vector<float> values(code_logits.data<float>(),
                                          code_logits.data<float>() + _config.vocab_size);
          chosen = sample_drama_token(values, *options, *rng);
        }
        frame.data<int32_t>()[i + 2] = chosen;
        current = std::move(next);
      }
    }

    void DramaTower::decode_impl(const StorageView& logits, const StorageView& hidden,
                                 const StorageView& previous_frame, DramaTowerState& state,
                                 StorageView& next_frame,
                                 const std::vector<int32_t>* history_main_tokens,
                                 const DramaSamplingOptions* options, std::mt19937* rng) const {
      validate_hidden(hidden);
      validate_frames(previous_frame, 1);
      validate_state(state, 1);
      validate_state_aliases(state, {&logits, &hidden, &previous_frame, &next_frame, &_embeddings});
      require_cpu_float(logits);
      if (hidden.dim(0) != 1 || logits.rank() != 2 || logits.dim(0) != 1
          || logits.dim(1) <= std::max(_config.semantic_end, _config.eos_id)
          || next_frame.device() != Device::CPU || next_frame.dtype() != DataType::INT32)
        throw std::invalid_argument(
            "DramaTower greedy decode expects one hidden/logit row and int32 frame output");
      for (const auto* input : {&logits, &hidden, &previous_frame, &_embeddings}) {
        if (input == &next_frame || overlaps(*input, next_frame))
          throw std::invalid_argument("DramaTower frame output must not alias inputs/weights");
      }
      const auto main = select_main_token(logits, history_main_tokens, options, rng);
      StorageView frame({1, _config.num_codebooks + 1}, int32_t(0));
      frame.data<int32_t>()[0] = main;
      frame.data<int32_t>()[1] = static_cast<int32_t>(std::max<int64_t>(
          0, std::min<int64_t>(_config.vocab_size - 1, int64_t(main) - _config.semantic_begin)));
      decode_residual_codes(hidden, previous_frame, frame, state, options, rng);
      ++state.length;
      next_frame = std::move(frame);
    }

  }
}
