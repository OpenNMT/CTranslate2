#include "ctranslate2/models/moonshine.h"

#include <algorithm>

#include "ctranslate2/decoding.h"

#include "dispatch.h"
#include "dtw.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

    const Vocabulary& MoonshineModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t MoonshineModel::current_spec_revision() const {
      return 0;
    }

    void MoonshineModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "<unk>";
      vocab_info.bos_token = "<s>";
      vocab_info.eos_token = "</s>";

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");
    }

    bool MoonshineModel::is_quantizable(const std::string& variable_name) const {
      return Model::is_quantizable(variable_name);
    }

    bool MoonshineModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> MoonshineModel::clone() const {
      return std::make_unique<MoonshineModel>(*this);
    }


    std::unique_ptr<MoonshineReplica> MoonshineReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const MoonshineModel*>(&model))
        throw std::invalid_argument("The model is not a Moonshine model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const MoonshineModel>(model_ptr);
      return std::make_unique<MoonshineReplica>(concrete_model);
    }

    MoonshineReplica::MoonshineReplica(const std::shared_ptr<const MoonshineModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _preprocessor(std::make_unique<layers::MoonshinePreprocessor>(*model, "preprocessor"))
      , _encoder(std::make_unique<layers::MoonshineEncoder>(*model, "encoder"))
      , _decoder(std::make_unique<layers::MoonshineDecoder>(*model, "decoder"))
    {
      const auto& vocabulary = model->get_vocabulary();
      _sot_id = vocabulary.bos_id();
      _eot_id = vocabulary.eos_id();
    }

    StorageView MoonshineReplica::encode(StorageView features, const bool to_cpu) {
      PROFILE("MoonshineReplica::encode");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();
      features.move_to(device, dtype);

      StorageView encoder_input(dtype, device);
      StorageView encoder_output(dtype, device);
      (*_preprocessor)(features, encoder_input);
      (*_encoder)(encoder_input, encoder_output);

      if (to_cpu) {
        if (device != Device::CPU)
          encoder_output = encoder_output.to(Device::CPU);
        return encoder_output;
      }

      // Ensure all operations are finished before returning the output.
      synchronize_stream(device);

      return encoder_output;
    }

    StorageView MoonshineReplica::maybe_encode(StorageView features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      features.move_to(device, dtype);

      if (_encoder->is_encoded(features))
        return features;

      StorageView encoder_input(dtype, device);
      StorageView encoder_output(dtype, device);
      (*_preprocessor)(features, encoder_input);
      (*_encoder)(encoder_input, encoder_output);
      return encoder_output;
    }

    std::vector<MoonshineGenerationResult>
    MoonshineReplica::generate(StorageView features,
                             const std::vector<std::vector<std::string>>& prompts,
                             const MoonshineOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(std::move(features), vocabulary.to_ids(prompts), options);
    }

    std::vector<MoonshineGenerationResult>
    MoonshineReplica::generate(StorageView features,
                             const std::vector<std::vector<size_t>>& prompts,
                             const MoonshineOptions& options) {
      PROFILE("MoonshineReplica::generate");
      if (prompts.empty())
        return {};

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(features)));

      _decoder->update_output_layer(_model->preferred_size_multiple());

      const dim_t total_max_length = options.max_length;

      DecodingOptions decoding_options;
      decoding_options.start_step = 0;
      decoding_options.beam_size = options.beam_size;
      decoding_options.patience = options.patience;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = total_max_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;
      decoding_options.include_eos_in_hypotheses = false;

      for (const auto& id : options.suppress_tokens) {
        if (id >= 0)
          decoding_options.disable_ids.push_back(id);
        else if (id == -1) {
          for (const auto& default_id : _model->config["suppress_ids"])
            decoding_options.disable_ids.push_back(default_id);
        }
      }

      if (options.suppress_blank) {
        for (const auto& id : _model->config["suppress_ids_begin"])
          decoding_options.disable_ids_begin.push_back(id);
      }

      std::vector<DecodingResult> results = decode(*_decoder,
                                                   state,
                                                   prompts,
                                                   {_eot_id},
                                                   decoding_options);

      std::vector<MoonshineGenerationResult> final_results;
      final_results.reserve(results.size());

      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];

        MoonshineGenerationResult final_result;
        final_result.sequences = vocabulary.to_tokens(result.hypotheses);
        final_result.sequences_ids = std::move(result.hypotheses);
        final_result.scores = std::move(result.scores);

        final_results.emplace_back(std::move(final_result));
      }

      return final_results;
    }

    std::future<StorageView> Moonshine::encode(const StorageView& features, const bool to_cpu) {
      return post<StorageView>(
        [features = features.sync_copy(), to_cpu](MoonshineReplica& replica) mutable {
          return replica.encode(std::move(features), to_cpu);
        });
    }

    std::vector<std::future<MoonshineGenerationResult>>
    Moonshine::generate(const StorageView& features,
                      std::vector<std::vector<std::string>> prompts,
                      MoonshineOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<MoonshineGenerationResult>(
        [features = features.sync_copy(),
         prompts = std::move(prompts),
         options = std::move(options)]
        (MoonshineReplica& replica) mutable {
          return replica.generate(std::move(features), prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<MoonshineGenerationResult>>
    Moonshine::generate(const StorageView& features,
                      std::vector<std::vector<size_t>> prompts,
                      MoonshineOptions options) {
      const size_t batch_size = features.dim(0);
      return post_batch<MoonshineGenerationResult>(
        [features = features.sync_copy(),
         prompts = std::move(prompts),
         options = std::move(options)]
        (MoonshineReplica& replica) mutable {
          return replica.generate(std::move(features), prompts, options);
        },
        batch_size);
    }
  }
}
