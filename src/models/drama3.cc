#include "ctranslate2/models/drama3.h"
#include <regex>
#include <stdexcept>
#include "layers/drama/drama_backbone.h"
#include "layers/drama/drama_tower.h"
#include "layers/drama/fish_dac_decoder.h"
#include "ctranslate2/utils.h"
#include "ctranslate2/ops/dequantize_fp8.h"
#include "drama3_memory.h"
#include "drama3_checkpoint.h"
#include "env.h"
#include "ctranslate2/ops/quantize.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace ctranslate2 {
  namespace models {
    namespace {

      constexpr dim_t max_drama3_sequence_length = 8192;
      using Clock = std::chrono::steady_clock;

      double seconds_since(const Clock::time_point& start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
      }

      std::string quantization_policy(const nlohmann::json& config) {
        const auto policy = config.value("quantization_policy", std::string());
        if (policy != "experts_fp8_sensitive_float32" && policy != "standard")
          throw std::invalid_argument("Unknown Drama3 quantization policy: " + policy);
        return policy;
      }

    }

    std::shared_ptr<const Model> Drama3Model::load_checkpoint(const std::string& path,
                                                             ComputeType compute_type,
                                                             bool map_experts) {
      if (!map_experts)
        return Model::load(path, Device::CPU, 0, compute_type);
      Drama3CheckpointReader reader(path);
      return Model::load(reader, Device::CPU, 0, compute_type);
    }

    StorageView Drama3Model::load_variable(ModelReader& reader, const std::string& name,
                                           Shape shape, DataType dtype, dim_t num_bytes,
                                           std::istream& stream) {
      if (const auto* checkpoint = dynamic_cast<const Drama3CheckpointReader*>(&reader))
        _checkpoint_storage = checkpoint->storage;
      if (_checkpoint_storage && dtype == DataType::INT8 && shape.size() == 2
          && name.compare(0, 16, "backbone/layers_") == 0
          && name.find("/mlp/experts_") != std::string::npos
          && name.size() >= 7 && name.compare(name.size() - 7, 7, "/weight") == 0)
        return _checkpoint_storage->view(std::move(shape), stream, num_bytes);
      return Model::load_variable(reader, name, std::move(shape), dtype, num_bytes, stream);
    }

    bool Drama3Model::is_quantizable(const std::string& name) const {
      // Model::set_compute_type runs before initialize. Enforce this policy here
      // so the source FP8 bytes and sensitive F32 matrices remain untouched.
      const auto policy = quantization_policy(config);
      if (name.compare(0, 6, "codec/") == 0 || name.size() < 7
          || name.compare(name.size() - 7, 7, "/weight") != 0)
        return false;
      const auto* variable = get_variable_if_exists(name);
      if (!variable || variable->rank() != 2)
        return false;
      if (policy == "standard")
        return true;
      static const std::regex routed(
          "backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight");
      const auto* inverse_scale = get_variable_if_exists(name + "_scale_inv");
      // INT8 is a byte container here, not CTranslate2 integer quantization.
      // Leave the original FP8 encoding and all sensitive matrices untouched.
      if (std::regex_match(name, routed)) {
        if (config.value("fp8_format", std::string()) != "e4m3fn_block128"
            || variable->dtype() != DataType::INT8 || !inverse_scale
            || inverse_scale->dtype() != DataType::FLOAT32
            || inverse_scale->shape()
                   != Shape{(variable->dim(0) + 127) / 128, (variable->dim(1) + 127) / 128}
            || get_variable_if_exists(name + "_scale"))
          throw std::invalid_argument(
              "Drama3 FP8 experts require original E4M3FN bytes and 128x128 block scales");
      } else if (variable->dtype() != DataType::FLOAT32 || inverse_scale) {
        throw std::invalid_argument("Drama3 FP8 policy requires float32 sensitive matrices");
      }
      return false;
    }

    bool Drama3Model::is_linear_weight(const std::string& name) const {
      return is_quantizable(name) && name.find("embedding") == std::string::npos;
    }

    void Drama3Model::register_variable(std::string name, StorageView variable) {
      const std::string suffix = "/weight_scale_inv";
      if (name.size() >= suffix.size()
          && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
        _fp8_weights.push_back(name.substr(0, name.size() - 10));
      if (_checkpoint_storage && name.size() > 13
          && name.compare(name.size() - 13, 13, "_compensation") == 0) {
        const auto* weight = get_variable_if_exists(name.substr(0, name.size() - 13));
        if (weight)
          _checkpoint_storage->release_pages(*weight);
      }
      Model::register_variable(std::move(name), std::move(variable));
    }

    void Drama3Model::prepare_weights(ComputeType& type, Device device, int index,
                                      bool update_weight) {
      if (_checkpoint_storage && (device != Device::CPU || tensor_parallel()))
        throw std::invalid_argument("Drama3 mapped weights require CPU loading without tensor parallelism");
      if (_checkpoint_storage && type == ComputeType::DEFAULT)
        type = config.value("quantization_policy", std::string()) == "standard"
                       && config.value("matrix_dtype", std::string()) == "int8"
                   ? ComputeType::INT8_FLOAT32 : ComputeType::FLOAT32;
      const auto policy = quantization_policy(config);
      if (device != Device::CPU)
        throw std::invalid_argument("Drama3Model requires CPU weight storage; use Drama3 with "
                                    "device=\"cuda\" for GPU execution");
      if (policy == "experts_fp8_sensitive_float32") {
        // Resolve before decoding, so unsupported requests fail without expanding
        // the checkpoint. DEFAULT retains the original FP8 storage.
        const auto resolved = resolve_compute_type(type, ComputeType::FLOAT32, device, index);
        if (update_weight && resolved != ComputeType::FLOAT32) {
          const auto weight_dtype = compute_type_to_data_type(resolved).first;
          StorageView decoded;
          for (const auto& name : _fp8_weights) {
            is_quantizable(name); // Validate FP8 metadata before replacing it.
            ops::DequantizeFP8()(get_variable(name), get_variable(name + "_scale_inv"), decoded);
            // Finish converting this expert before decoding the next one. Keeping
            // all experts in FP32 until Model::set_compute_type runs can require
            // several times the checkpoint size in resident memory.
            StorageView converted(weight_dtype);
            if (is_float_type(weight_dtype)) {
              converted = decoded.to(weight_dtype);
            } else {
              StorageView quantized(weight_dtype), scale;
              ops::Quantize(ops::Quantize::ScaleType::PER_LAYER, false,
                            round_before_cast_in_quantization())(decoded, quantized, scale);
              register_variable(name + "_scale", std::move(scale));
              converted = std::move(quantized);
            }
            remove_variable(name);
            Model::register_variable(name, std::move(converted));
            remove_variable(name + "_scale_inv");
          }
          _fp8_weights.clear();
          config["quantization_policy"] = "standard";
          config.erase("fp8_format");
        }
      } else if (!_fp8_weights.empty()) {
        throw std::invalid_argument(
            "Drama3 standard quantization cannot contain FP8 inverse scales");
      }
    }

    void Drama3Model::initialize(ModelReader&) {
      quantization_policy(config);
      if (get_default_float_type(effective_compute_type()) != DataType::FLOAT32)
        throw std::invalid_argument("Drama3 CPU graph requires float32 activations");
      if (device() != Device::CPU)
        throw std::invalid_argument("Drama3Model requires CPU weight storage; use Drama3 with "
                                    "device=\"cuda\" for GPU execution");
      if (!config.is_object() || config.value("model_type", std::string()) != "drama3"
          || config.value("ct2_layout_version", 0) != 1 || !config.contains("source_config")
          || !config["source_config"].is_object())
        throw std::invalid_argument("Drama3 requires verified layout-v1 source configuration");
    }

    std::unique_ptr<Model> Drama3Model::clone() const {
      return std::make_unique<Drama3Model>(*this);
    }

    namespace {

      layers::GatedSelfAttentionConfig attention_config(const nlohmann::json& cfg, bool tower) {
        layers::GatedSelfAttentionConfig result;
        result.num_heads = cfg.at(tower ? "n_head" : "num_attention_heads").get<dim_t>();
        result.num_kv_heads = cfg.at(tower ? "n_local_heads" : "num_key_value_heads").get<dim_t>();
        result.head_dim = cfg.at("head_dim").get<dim_t>();
        const auto& rope = tower ? cfg : cfg.at("rope_parameters");
        const float fraction =
            rope.at(tower ? "rotary_percent" : "partial_rotary_factor").get<float>();
        result.rotary_dim = static_cast<dim_t>(result.head_dim * fraction);
        result.rope_base = rope.at(tower ? "rope_base" : "rope_theta").get<float>();
        result.norm_epsilon = cfg.at(tower ? "norm_eps" : "rms_norm_eps").get<float>();
        result.rotary_interleaved = cfg.at("rotary_interleaved").get<bool>();
        result.zero_centered_qk_norm = true;
        if (!cfg.value("layernorm_zero_centered_gamma", false)
            || !cfg.value(tower ? "attention_output_gate" : "attn_output_gate", false))
          throw std::invalid_argument(
              "Unsupported Drama3 attention normalization/gate configuration");
        return result;
      }

      template <typename T> StorageView rows(const StorageView& source, dim_t first, dim_t count) {
        const dim_t columns = source.dim(1);
        return StorageView({count, columns},
                           std::vector<T>(source.data<T>() + first * columns,
                                          source.data<T>() + (first + count) * columns));
      }

    }

    struct Drama3Replica::Impl : ModelReplica {
      struct Setup {
        std::shared_ptr<const Model> checkpoint;
        layers::DramaExecutionOptions options;
      };

      const layers::DramaExecutionOptions options;
      std::unique_ptr<layers::Embeddings> embeddings;
      std::unique_ptr<layers::DramaBackbone> backbone;
      std::unique_ptr<layers::DramaTower> tower;
      std::unique_ptr<layers::FishDACDecoder> codec;
      dim_t max_context;
      dim_t vocabulary;
      int32_t eos;

      Impl(Setup setup)
        : ModelReplica(setup.checkpoint)
        , options(setup.options)
      {
        const ScopedDeviceSetter setter(options.device, options.device_index);
        initialize_layers();
        // Publish initialization uploads before a worker uses its own stream.
        synchronize_stream(options.device);
      }

      static Setup prepare(const std::string& path, size_t count, const std::string& compute,
                           const std::string& device, int device_index) {
        if (!count)
          throw std::invalid_argument("Drama3 requires positive intra_threads");
        const auto legacy_budget = read_string_from_env("CT2_DRAMA3_MAX_MEMORY_GB");
        if (!legacy_budget.empty() && legacy_budget != "0")
          throw std::invalid_argument("CT2_DRAMA3_MAX_MEMORY_GB is no longer supported. "
                                      "Drama3 uses allocation controls, without a total RAM ceiling.");
        Setup setup;
        auto& options = setup.options;
        options.device = str_to_device(device);
        options.device_index = device_index;
        options.resident_backbone = read_bool_from_env("CT2_DRAMA_GPU_RESIDENT_BACKBONE", true);
        options.waveform = read_bool_from_env("CT2_DRAMA_GPU_WAVEFORM", true);
        const auto budget = drama3_gpu_expert_cache_budget();
        if (budget && options.device != Device::CUDA)
          throw std::invalid_argument("CT2_DRAMA3_GPU_EXPERT_CACHE_GB requires GPU execution");
        if (device_index < 0 || device_index >= get_device_count(options.device))
          throw std::invalid_argument("Invalid Drama3 device index: " + std::to_string(device_index));
        const ScopedDeviceSetter setter(options.device, device_index);
        const auto requested = str_to_compute_type(compute);
        const bool map_cpu_experts = options.device == Device::CPU
            && read_bool_from_env("CT2_DRAMA3_CPU_MMAP", false);
        if (map_cpu_experts && requested != ComputeType::DEFAULT
            && requested != ComputeType::FLOAT32)
          throw std::invalid_argument("CT2_DRAMA3_CPU_MMAP requires compute_type=default or float32");
        if (options.device == Device::CUDA && requested != ComputeType::DEFAULT)
          resolve_compute_type(requested, ComputeType::FLOAT32, options.device, device_index);
        set_num_threads(count);
        setup.checkpoint = Drama3Model::load_checkpoint(path,
            options.device == Device::CPU ? requested
            : budget ? ComputeType::DEFAULT : ComputeType::FLOAT32,
            budget != 0 || map_cpu_experts);
        if (!dynamic_cast<const Drama3Model*>(setup.checkpoint.get()))
          throw std::invalid_argument("Expected a converted Drama3Spec model");
        options.compute_type = options.device == Device::CPU
            ? setup.checkpoint->effective_compute_type()
            : resolve_compute_type(requested, setup.checkpoint->saved_compute_type(),
                                   options.device, device_index);
        if (budget) {
          const auto count = setup.checkpoint->config.at("source_config").at("text_config")
                                 .at("num_hidden_layers").get<int64_t>();
          if (count <= 0 || budget / count == 0)
            throw std::invalid_argument("Drama3 expert cache requires a positive layer count and budget per layer");
          options.expert_cache_bytes = budget / count;
        }
        return setup;
      }

      void initialize_layers() {
        const auto& source = model()->config.at("source_config");
        const auto& text = source.at("text_config");
        const auto& audio = source.at("audio_decoder_config");
        layers::DramaBackboneConfig bc;
        bc.norm_epsilon = text.at("rms_norm_eps").get<float>();
        bc.max_context = max_context = text.at("max_position_embeddings").get<dim_t>();
        vocabulary = text.at("vocab_size").get<dim_t>();
        if (!text.value("norm_topk_prob", true))
          throw std::invalid_argument("Drama3 requires normalized top-k routing");
        for (const auto& type : text.at("layer_types")) {
          layers::DramaBlockConfig block;
          if (type == "linear_attention")
            block.kind = layers::DramaMixerKind::LinearAttention;
          else if (type == "full_attention")
            block.kind = layers::DramaMixerKind::FullAttention;
          else
            throw std::invalid_argument("Unsupported Drama3 mixer type");
          block.norm_epsilon = bc.norm_epsilon;
          block.num_experts = text.at("num_experts").get<dim_t>();
          block.top_k = text.at("num_experts_per_tok").get<dim_t>();
          block.key_heads = text.at("linear_num_key_heads").get<dim_t>();
          block.value_heads = text.at("linear_num_value_heads").get<dim_t>();
          block.key_dim = text.at("linear_key_head_dim").get<dim_t>();
          block.value_dim = text.at("linear_value_head_dim").get<dim_t>();
          block.attention = attention_config(text, false);
          bc.layers.push_back(block);
        }
        if (bc.layers.size() != text.at("num_hidden_layers").get<size_t>())
          throw std::invalid_argument("Drama3 layer count mismatch");
        layers::DramaTowerConfig tc;
        tc.attention = attention_config(audio, true);
        tc.num_codebooks = audio.at("num_codebooks").get<dim_t>();
        tc.vocab_size = audio.at("vocab_size").get<dim_t>();
        tc.shared_layers = audio.at("shared_enrichment_layers").get<dim_t>();
        tc.vq_backbone_codebooks = audio.at("vq_backbone_codebooks").get<dim_t>();
        tc.semantic_begin = source.at("semantic_start_token_id").get<int32_t>();
        tc.semantic_end = source.at("semantic_end_token_id").get<int32_t>();
        tc.eos_id = eos = source.at("eos_token_id").get<int32_t>();
        if (tc.num_codebooks != 10 || tc.semantic_end - tc.semantic_begin + 1 != 4096)
          throw std::invalid_argument(
              "Drama3 codec requires ten codebooks and 4096 semantic codes");
        bc.lm_candidates.push_back(tc.eos_id);
        for (int32_t id = tc.semantic_begin; id <= tc.semantic_end; ++id)
          bc.lm_candidates.push_back(id);
        embeddings = std::make_unique<layers::Embeddings>(*model(), "backbone/embeddings");
        backbone = std::make_unique<layers::DramaBackbone>(*model(), "backbone", std::move(bc), options);
        tower = std::make_unique<layers::DramaTower>(*model(), "tower", tc, options);
        codec = std::make_unique<layers::FishDACDecoder>(*model(), "codec", options);
      }

      StorageView embed(const StorageView& frames) const {
        std::vector<int32_t> ids(frames.dim(0));
        for (dim_t i = 0; i < frames.dim(0); ++i)
          ids[i] = frames.data<int32_t>()[i * 11];
        StorageView text, result;
        (*embeddings)(StorageView({frames.dim(0)}, ids), text);
        tower->embed_backbone(frames, text, result);
        return result;
      }

      struct GenerationState {
        layers::DramaBackboneState backbone_state;
        layers::DramaTowerState tower_state;
        StorageView hidden;
        StorageView logits;
      };

      void validate_generation(const StorageView& prompt, dim_t max_frames,
                               dim_t prefill_chunk_size, float temperature, dim_t top_k,
                               float top_p) const {
        if (prompt.device() != Device::CPU || prompt.dtype() != DataType::INT32
            || prompt.rank() != 2 || prompt.dim(1) != 11 || prompt.dim(0) <= 0
            || prompt.dim(0) > max_drama3_sequence_length || max_frames <= 0
            || max_frames > max_drama3_sequence_length || prefill_chunk_size <= 0
            || max_frames > max_context - prompt.dim(0) || !std::isfinite(temperature)
            || temperature <= 0 || top_k <= 0 || !std::isfinite(top_p) || top_p <= 0 || top_p > 1)
          throw std::invalid_argument("Invalid Drama3 prompt, frame budget or prefill chunk size");
        for (dim_t t = 0; t < prompt.dim(0); ++t)
          if (prompt.data<int32_t>()[t * 11] < 0 || prompt.data<int32_t>()[t * 11] >= vocabulary)
            throw std::invalid_argument("Drama3 prompt token outside vocabulary");
      }

      void prefill(const StorageView& prompt, dim_t chunk_size, GenerationState& state) const {
        for (dim_t start = 0; start < prompt.dim(0); start += chunk_size) {
          const dim_t count = std::min(chunk_size, prompt.dim(0) - start);
          auto frames = rows<int32_t>(prompt, start, count);
          auto embedded = embed(frames);
          (*backbone)(embedded, state.backbone_state, state.hidden, state.logits);
          const dim_t paired = std::min(count, prompt.dim(0) - 1 - start);
          if (paired > 0)
            tower->prefill(rows<float>(state.hidden, 0, paired),
                           rows<int32_t>(prompt, start, paired),
                           rows<int32_t>(prompt, start + 1, paired), state.tower_state);
        }
        state.hidden = rows<float>(state.hidden, state.hidden.dim(0) - 1, 1);
        state.logits = rows<float>(state.logits, state.logits.dim(0) - 1, 1);
      }

      void generate_codes(const StorageView& prompt, dim_t max_frames,
                          const layers::DramaSamplingOptions& sampling, bool greedy,
                          GenerationState& state, Drama3Result& result) const {
        auto previous = rows<int32_t>(prompt, prompt.dim(0) - 1, 1);
        std::vector<int32_t> codes;
        std::vector<int32_t> history;
        for (dim_t t = 0; t < prompt.dim(0); ++t)
          history.push_back(prompt.data<int32_t>()[t * 11]);
        std::mt19937 rng(static_cast<uint32_t>(sampling.seed));
        result.finish_reason = "length";
        for (dim_t step = 0; step < max_frames; ++step) {
          StorageView next(DataType::INT32);
          const auto tower_start = Clock::now();
          if (greedy)
            tower->decode_greedy(state.logits, state.hidden, previous, state.tower_state, next);
          else
            tower->decode_sample(state.logits, state.hidden, previous, state.tower_state, history,
                                 sampling, rng, next);
          result.timings.tower_decode_seconds += seconds_since(tower_start);
          if (next.data<int32_t>()[0] == eos) {
            result.finish_reason = "eos";
            break;
          }
          codes.insert(codes.end(), next.data<int32_t>() + 1, next.data<int32_t>() + 11);
          history.push_back(next.data<int32_t>()[0]);
          if (step + 1 < max_frames) {
            const auto backbone_start = Clock::now();
            auto embedded = embed(next);
            (*backbone)(embedded, state.backbone_state, state.hidden, state.logits);
            result.timings.backbone_decode_seconds += seconds_since(backbone_start);
          }
          previous = std::move(next);
        }
        result.codes = StorageView({dim_t(codes.size() / 10), 10}, codes);
      }

      void decode_audio(Drama3Result& result) const {
        const dim_t frames = result.codes.dim(0);
        if (frames > 0) {
          const auto codec_start = Clock::now();
          StorageView codec_codes({1, 10, frames}, DataType::INT32);
          for (dim_t t = 0; t < frames; ++t)
            // Official DAC clamps residual lookup indices to its 1024-entry books.
            // Preserve raw tower codes for feedback and in the returned diagnostic.
            for (dim_t b = 0; b < 10; ++b)
              codec_codes.data<int32_t>()[b * frames + t] =
                  std::min(result.codes.data<int32_t>()[t * 10 + b], b == 0 ? 4095 : 1023);
          (*codec)(codec_codes, result.audio);
          result.audio.reshape({result.audio.size()});
          result.timings.codec_seconds = seconds_since(codec_start);
        } else
          result.audio = StorageView({0}, DataType::FLOAT32);
      }
    };

    Drama3Replica::Drama3Replica(std::unique_ptr<Impl> impl)
      : ModelReplica(impl->model())
      , _impl(std::move(impl)) {
    }

    Drama3Replica::~Drama3Replica() {
      const auto device = _impl->options.device;
      const ScopedDeviceSetter setter(device, _impl->options.device_index);
      synchronize_stream(device);
      _impl.reset();
      if (device != model()->device())
        destroy_context(device);
    }

    std::unique_ptr<Drama3Replica> Drama3Replica::create_from_model(const Model& model) {
      const auto* checkpoint = dynamic_cast<const Drama3Model*>(&model);
      if (!checkpoint)
        throw std::invalid_argument("Expected a Drama3 model");
      Impl::Setup setup;
      setup.checkpoint = model.shared_from_this();
      setup.options.compute_type = model.effective_compute_type();
      return std::unique_ptr<Drama3Replica>(new Drama3Replica(
          std::make_unique<Impl>(std::move(setup))));
    }

    Drama3::Drama3(const std::string& path, size_t threads, const std::string& compute,
                   const std::string& device, int device_index,
                   size_t inter_threads, long max_queued_batches) {
      if (!inter_threads)
        throw std::invalid_argument("Drama3 requires positive inter_threads");
      CallingThreadContextGuard context_guard;
      auto setup = Drama3Replica::Impl::prepare(path, threads, compute, device, device_index);
      setup.options.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
      ReplicaPoolConfig config;
      config.num_threads_per_replica = threads;
      config.max_queued_batches = max_queued_batches;
      std::vector<std::shared_ptr<const Model>> models(inter_threads, setup.checkpoint);
      initialize_pool(models, config, [options = setup.options](const Model& model) {
        const auto* checkpoint = dynamic_cast<const Drama3Model*>(&model);
        if (!checkpoint)
          throw std::invalid_argument("Expected a Drama3 model");
        Drama3Replica::Impl::Setup setup;
        setup.checkpoint = model.shared_from_this();
        setup.options = options;
        return std::unique_ptr<Drama3Replica>(new Drama3Replica(
            std::make_unique<Drama3Replica::Impl>(std::move(setup))));
      });
    }

    Drama3Result Drama3::generate(const StorageView& prompt, dim_t max_frames,
                                  dim_t prefill_chunk_size, float temperature, dim_t top_k,
                                  float top_p, uint64_t seed, bool greedy) {
      return generate_async(prompt, {max_frames, prefill_chunk_size, temperature,
                                     top_k, top_p, seed, greedy}).get();
    }

    std::future<Drama3Result> Drama3::generate_async(const StorageView& prompt,
                                                    const Drama3Options& options) {
      // Own the input until the queued job finishes, including borrowed caller views.
      return post<Drama3Result>([prompt = StorageView(prompt), options](Drama3Replica& replica) {
        return replica.generate(prompt, options);
      });
    }

    std::vector<std::future<Drama3Result>>
    Drama3::generate_batch(const std::vector<StorageView>& prompts, const Drama3Options& options) {
      std::vector<std::future<Drama3Result>> results;
      results.reserve(prompts.size());
      for (const auto& prompt : prompts)
        results.emplace_back(generate_async(prompt, options));
      return results;
    }

    Drama3Result Drama3Replica::generate(const StorageView& prompt, const Drama3Options& options) {
      const auto max_frames = options.max_frames;
      const auto prefill_chunk_size = options.prefill_chunk_size;
      const auto temperature = options.temperature;
      const auto top_k = options.top_k;
      const auto top_p = options.top_p;
      const auto seed = options.seed;
      const auto greedy = options.greedy;
      const auto total_start = Clock::now();
      const ScopedDeviceSetter setter(_impl->options.device, _impl->options.device_index);
      _impl->validate_generation(prompt, max_frames, prefill_chunk_size, temperature, top_k, top_p);

      Impl::GenerationState state;
      const dim_t capacity = prompt.dim(0) + max_frames;
      _impl->backbone->initialize_state(state.backbone_state, capacity);
      _impl->tower->initialize_state(state.tower_state, capacity);

      Drama3Result result;
      const auto prefill_start = Clock::now();
      _impl->prefill(prompt, prefill_chunk_size, state);
      result.timings.prefill_seconds = seconds_since(prefill_start);

      layers::DramaSamplingOptions sampling;
      sampling.temperature = temperature;
      sampling.top_k = top_k;
      sampling.top_p = top_p;
      sampling.seed = seed;
      _impl->generate_codes(prompt, max_frames, sampling, greedy, state, result);
      _impl->decode_audio(result);

      result.timings.total_native_seconds = seconds_since(total_start);
      return result;
    }

  }
}
