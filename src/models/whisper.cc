#include "ctranslate2/models/whisper.h"

#include <algorithm>

#include "ctranslate2/decoding.h"
#include "ctranslate2/models/model_factory.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace models {

    static auto register_whisper = register_model<WhisperModel>("WhisperSpec");

    const Vocabulary& WhisperModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t WhisperModel::current_spec_revision() const {
      return 2;
    }

    void WhisperModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "<|endoftext|>";
      vocab_info.bos_token = "<|startoftranscript|>";
      vocab_info.eos_token = "<|endoftext|>";
      _vocabulary = std::make_shared<Vocabulary>(*model_reader.get_required_file("vocabulary.txt"),
                                                 std::move(vocab_info));
    }

    bool WhisperModel::is_quantizable(const std::string& variable_name) const {
      return (Model::is_quantizable(variable_name)
              && variable_name.find("conv") == std::string::npos);
    }

    bool WhisperModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> WhisperModel::clone() const {
      return std::make_unique<WhisperModel>(*this);
    }


    std::unique_ptr<WhisperReplica> WhisperReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const WhisperModel*>(&model))
        throw std::invalid_argument("The model is not a Whisper model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const WhisperModel>(model_ptr);
      return std::make_unique<WhisperReplica>(concrete_model);
    }

    WhisperReplica::WhisperReplica(const std::shared_ptr<const WhisperModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::WhisperEncoder>(*model, "encoder"))
      , _decoder(std::make_unique<layers::WhisperDecoder>(*model, "decoder"))
    {
    }

    StorageView WhisperReplica::encode(const StorageView& features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      StorageView encoder_output(dtype, device);
      if (features.device() == device && features.dtype() == dtype)
        (*_encoder)(features, encoder_output);
      else
        (*_encoder)(features.to(device).to(dtype), encoder_output);

      return encoder_output;
    }

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(const StorageView& features,
                             const std::vector<std::vector<std::string>>& prompts,
                             const WhisperOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(features, vocabulary.to_ids(prompts), options);
    }

    class ApplyTimestampRules;

    std::vector<WhisperGenerationResult>
    WhisperReplica::generate(const StorageView& features,
                             const std::vector<std::vector<size_t>>& prompts,
                             const WhisperOptions& options) {
      PROFILE("WhisperReplica::generate");
      if (prompts.empty())
        return {};

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = options.max_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.return_prefix = false;
      for (const auto& id : _model->config["suppress_ids"])
        decoding_options.disable_ids.push_back(id);
      for (const auto& id : _model->config["suppress_ids_begin"])
        decoding_options.disable_ids_begin.push_back(id);

      const size_t no_timestamps_id = vocabulary.to_id("<|notimestamps|>");
      const bool with_timestamps = (
        std::find(prompts[0].begin(), prompts[0].end(), no_timestamps_id) == prompts[0].end());

      if (with_timestamps) {
        const size_t eot_id = vocabulary.eos_id();
        const size_t timestamp_begin_id = no_timestamps_id + 1;
        const size_t timestamp_end_id = vocabulary.size() - 1;
        const size_t max_initial_timestamp_id = timestamp_begin_id + 50;
        decoding_options.logits_processors.emplace_back(
          std::make_shared<ApplyTimestampRules>(eot_id,
                                                no_timestamps_id,
                                                timestamp_begin_id,
                                                timestamp_end_id,
                                                max_initial_timestamp_id));
      }

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", encode(features));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      std::vector<std::vector<size_t>> start_tokens = prompts;
      std::vector<float> no_speech_probs;

      if (start_tokens.size() == 1 && start_tokens[0].size() > 1) {
        // Initialize the decoder state with the prompt.
        auto& prompt = start_tokens[0];
        const size_t last_token = prompt.back();
        prompt.pop_back();

        const StorageView input = layers::make_sequence_inputs(start_tokens, _model->device());

        if (options.return_no_speech_prob) {
          const auto it = std::find(prompt.begin(), prompt.end(), vocabulary.bos_id());
          if (it == prompt.end())
            throw std::invalid_argument("<|startoftranscript|> token was not found in the prompt");
          const dim_t sot_index = std::distance(prompt.begin(), it);

          const Device device = _decoder->device();
          const DataType dtype = _decoder->output_type();
          StorageView logits_at_sot(dtype, device);
          StorageView probs_at_sot(dtype, device);
          _decoder->forward_prompt(input, state, sot_index, &logits_at_sot);
          ops::SoftMax()(logits_at_sot, probs_at_sot);

          dim_t no_speech_id = vocabulary.to_id("<|nospeech|>");
          if (size_t(no_speech_id) == vocabulary.unk_id())
            no_speech_id = vocabulary.to_id("<|nocaptions|>");

          float no_speech_prob = 0;
          TYPE_DISPATCH(dtype, no_speech_prob = probs_at_sot.scalar_at<T>({0, no_speech_id}));
          no_speech_probs.push_back(no_speech_prob);
        } else {
          _decoder->forward_prompt(input, state);
        }

        start_tokens = {{last_token}};
        decoding_options.start_step = input.dim(1);
        decoding_options.max_length /= 2;

      } else if (options.return_no_speech_prob) {
        throw std::runtime_error("Returning the no speech probability in batch mode is "
                                 "currently not supported");
      }

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   start_tokens,
                                                   vocabulary.eos_id(),
                                                   decoding_options);

      std::vector<WhisperGenerationResult> final_results;
      final_results.reserve(results.size());

      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        WhisperGenerationResult final_result;
        final_result.sequences = vocabulary.to_tokens(result.hypotheses);
        final_result.sequences_ids = std::move(result.hypotheses);
        final_result.scores = std::move(result.scores);
        if (options.return_no_speech_prob)
          final_result.no_speech_prob = no_speech_probs[i];

        final_results.emplace_back(std::move(final_result));
      }

      return final_results;
    }

    std::vector<std::vector<std::pair<std::string, float>>>
    WhisperReplica::detect_language(const StorageView& features) {
      PROFILE("WhisperReplica::detect_language");

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const auto& vocabulary = _model->get_vocabulary();
      const auto device = _model->device();

      const int32_t sot = vocabulary.bos_id();
      std::vector<int32_t> lang_ids;
      for (const auto& id : _model->config["lang_ids"])
        lang_ids.push_back(id);

      const dim_t batch_size = features.dim(0);
      const dim_t num_langs = lang_ids.size();

      StorageView start_ids({batch_size}, sot, device);
      StorageView score_ids({batch_size, num_langs}, DataType::INT32);
      for (dim_t i = 0; i < batch_size; ++i) {
        for (dim_t j = 0; j < num_langs; ++j)
          score_ids.at<int32_t>({i, j}) = lang_ids[j];
      }
      if (score_ids.device() != device)
        score_ids = score_ids.to(device);

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", encode(features));

      StorageView logits(_decoder->output_type(), device);
      StorageView lang_probs(logits.dtype(), device);
      (*_decoder)(0, start_ids, state, &logits);
      ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(logits, score_ids, lang_probs);
      ops::SoftMax()(lang_probs);

      if (lang_probs.dtype() != DataType::FLOAT)
        lang_probs = lang_probs.to_float();
      if (lang_probs.device() != Device::CPU)
        lang_probs = lang_probs.to(Device::CPU);

      std::vector<std::vector<std::pair<std::string, float>>> results;
      results.reserve(batch_size);

      for (dim_t i = 0; i < batch_size; ++i) {
        std::vector<std::pair<std::string, float>> result;
        result.reserve(num_langs);

        for (dim_t j = 0; j < num_langs; ++j) {
          const size_t lang_id = lang_ids[j];
          const float prob = lang_probs.at<float>({i, j});
          result.emplace_back(vocabulary.to_token(lang_id), prob);
        }

        std::sort(result.begin(), result.end(),
                  [](const std::pair<std::string, float>& a,
                     const std::pair<std::string, float>& b) {
                    return a.second > b.second;
                  });

        results.emplace_back(std::move(result));
      }

      return results;
    }


    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(StorageView features,
                      std::vector<std::vector<std::string>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = std::move(features), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(features, prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<WhisperGenerationResult>>
    Whisper::generate(StorageView features,
                      std::vector<std::vector<size_t>> prompts,
                      WhisperOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<WhisperGenerationResult>(
        [features = std::move(features), prompts = std::move(prompts), options]
        (WhisperReplica& replica) {
          return replica.generate(features, prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<std::vector<std::pair<std::string, float>>>>
    Whisper::detect_language(StorageView features) {
      const size_t batch_size = features.dim(0);
      return post_batch<std::vector<std::pair<std::string, float>>>(
        [features = std::move(features)](WhisperReplica& replica) {
          return replica.detect_language(features);
        },
        batch_size);
    }


    class ApplyTimestampRules : public LogitsProcessor {
    private:
      const size_t _eot_id;
      const size_t _no_timestamps_id;
      const size_t _timestamp_begin_id;
      const size_t _timestamp_end_id;
      const size_t _max_initial_timestamp_id;

    public:
      ApplyTimestampRules(const size_t eot_id,
                          const size_t no_timestamps_id,
                          const size_t timestamp_begin_id,
                          const size_t timestamp_end_id,
                          const size_t max_initial_timestamp_id)
        : _eot_id(eot_id)
        , _no_timestamps_id(no_timestamps_id)
        , _timestamp_begin_id(timestamp_begin_id)
        , _timestamp_end_id(timestamp_end_id)
        , _max_initial_timestamp_id(max_initial_timestamp_id)
      {
      }

      void apply(dim_t step,
                 StorageView& logits,
                 DisableTokens& disable_tokens,
                 const StorageView& sequences,
                 const std::vector<dim_t>& batch_offset,
                 const std::vector<std::vector<size_t>>* prefix) override {
        std::vector<dim_t> check_timestamps_prob_for_batch;
        const dim_t batch_size = logits.dim(0);

        for (dim_t batch_id = 0; batch_id < batch_size; ++batch_id) {
          const dim_t sample_begin = get_sample_begin(batch_size, batch_id, batch_offset, prefix);

          // Suppress <|notimestamps|>.
          disable_tokens.add(batch_id, _no_timestamps_id);

          if (step == sample_begin) {
            // Suppress non timestamps at the beginning.
            for (size_t i = 0; i < _timestamp_begin_id; ++i)
              disable_tokens.add(batch_id, i);

            // Apply max_initial_timestamp option.
            for (size_t i = _max_initial_timestamp_id + 1; i <= _timestamp_end_id; ++i)
              disable_tokens.add(batch_id, i);

          } else if (step > sample_begin) {
            // Timestamps have to appear in pairs, except directly before EOT.
            const size_t last_token = sequences.at<int32_t>({batch_id, step - 1});

            if (last_token >= _timestamp_begin_id) {
              const size_t penultimate_token = (step - 1 > sample_begin
                                                ? sequences.at<int32_t>({batch_id, step - 2})
                                                : last_token);

              if (penultimate_token >= _timestamp_begin_id) {  // has to be non-timestamp
                for (size_t i = _timestamp_begin_id; i <= _timestamp_end_id; ++i)
                  disable_tokens.add(batch_id, i);
              } else {  // cannot be normal text tokens
                for (size_t i = 0; i < _eot_id; ++i)
                  disable_tokens.add(batch_id, i);
              }
            } else {
              check_timestamps_prob_for_batch.push_back(batch_id);
            }
          }
        }

        if (!check_timestamps_prob_for_batch.empty()) {
          // Apply all changes to the logits before computing the log softmax.
          disable_tokens.apply();

          StorageView log_probs(logits.dtype(), logits.device());
          ops::LogSoftMax()(logits, log_probs);

          for (const dim_t batch_id : check_timestamps_prob_for_batch) {
            bool sample_timestamp = false;

            if (log_probs.device() == Device::CPU)
              sample_timestamp = should_sample_timestamp<Device::CPU, float>(log_probs, batch_id);
#ifdef CT2_WITH_CUDA
            else if (log_probs.dtype() == DataType::FLOAT)
              sample_timestamp = should_sample_timestamp<Device::CUDA, float>(log_probs, batch_id);
            else
              sample_timestamp = should_sample_timestamp<Device::CUDA, float16_t>(log_probs, batch_id);
#endif

            if (sample_timestamp) {
              for (size_t i = 0; i < _timestamp_begin_id; ++i)
                disable_tokens.add(batch_id, i);
            }
          }
        }
      }

      template <Device D, typename T>
      bool should_sample_timestamp(const StorageView& log_probs, const dim_t batch_id) {
        const dim_t num_text_tokens = _timestamp_begin_id;
        const dim_t num_timestamp_tokens = _timestamp_end_id - _timestamp_begin_id + 1;

        const T* text_log_probs = log_probs.index<T>({batch_id, 0});
        const T* timestamp_log_probs = text_log_probs + num_text_tokens;

        // If sum of probability over timestamps is above any other token, sample timestamp.
        const float max_text_token_log_prob = primitives<D>::max(text_log_probs, num_text_tokens);
        const float timestamp_log_prob = primitives<D>::logsumexp(timestamp_log_probs,
                                                                  num_timestamp_tokens);

        return timestamp_log_prob > max_text_token_log_prob;
      }

    };

  }
}
