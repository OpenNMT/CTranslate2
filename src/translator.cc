#include "ctranslate2/translator.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/decoding.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/profiler.h"

namespace ctranslate2 {

  static std::unique_ptr<const Sampler> make_sampler(const TranslationOptions& options) {
    if (options.sampling_topk != 1)
      return std::make_unique<RandomSampler>(options.sampling_topk, options.sampling_temperature);
    else
      return std::make_unique<BestSampler>();
  }

  static std::unique_ptr<const SearchStrategy>
  make_search_strategy(const TranslationOptions& options) {
    if (options.beam_size == 1)
      return std::make_unique<GreedySearch>();
    else
      return std::make_unique<BeamSearch>(options.beam_size,
                                          options.length_penalty,
                                          options.coverage_penalty,
                                          options.prefix_bias_beta,
                                          options.allow_early_exit);
  }

  static bool should_rebatch(const std::vector<std::vector<std::string>>& source,
                             const bool allow_batch = true) {
    const size_t batch_size = source.size();
    if (!allow_batch && batch_size > 1)
      return true;
    for (size_t i = 0; i < batch_size; ++i) {
      if (source[i].empty())
        return true;
      if (i > 0 && source[i].size() > source[i - 1].size())
        return true;
    }
    return false;
  }


  void TranslationOptions::validate() const {
    if (num_hypotheses == 0)
      throw std::invalid_argument("num_hypotheses must be > 0");
    if (beam_size == 0)
      throw std::invalid_argument("beam_size must be > 0");
    if (num_hypotheses > beam_size && !return_alternatives)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (sampling_topk != 1 && beam_size != 1)
      throw std::invalid_argument("Random sampling should be used with beam_size = 1");
    if (min_decoding_length > max_decoding_length)
      throw std::invalid_argument("min_decoding_length is greater than max_decoding_length");
    if (prefix_bias_beta >= 1)
      throw std::invalid_argument("prefix_bias_beta must be less than 1.0");
    if (prefix_bias_beta > 0 && return_alternatives)
      throw std::invalid_argument("prefix_bias_beta is not compatible with return_alternatives");
    if (prefix_bias_beta > 0 && beam_size <= 1)
      throw std::invalid_argument("prefix_bias_beta is not compatible with greedy-search");
  }

  bool TranslationOptions::support_batch_translation() const {
    return !return_alternatives;
  }


  Translator::Translator(const std::string& model_dir,
                         Device device,
                         int device_index,
                         ComputeType compute_type) {
    set_model(models::Model::load(model_dir, device, device_index, compute_type));
  }

  Translator::Translator(const std::shared_ptr<const models::Model>& model) {
    set_model(model);
  }

  Translator::Translator(const Translator& other) {
    if (other._model)
      set_model(other._model);
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens) {
    return translate(tokens, TranslationOptions());
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens,
                        const TranslationOptions& options) {
    return translate_batch({tokens}, options)[0];
  }

  TranslationResult
  Translator::translate_with_prefix(const std::vector<std::string>& source,
                                    const std::vector<std::string>& target_prefix,
                                    const TranslationOptions& options) {
    return translate_batch_with_prefix({source}, {target_prefix}, options)[0];
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    return translate_batch(batch_tokens, TranslationOptions());
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens,
                              const TranslationOptions& options) {
    return translate_batch_with_prefix(batch_tokens, {}, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch_with_prefix(const std::vector<std::vector<std::string>>& source,
                                          const std::vector<std::vector<std::string>>& target_prefix,
                                          const TranslationOptions& options) {
    options.validate();
    if (source.empty())
      return {};
    if (!should_rebatch(source, options.support_batch_translation()))
      return run_batch_translation(source, target_prefix, options);

    // Rebatching is required to filter out empty inputs and sort by length.
    const TranslationResult empty_result(options.num_hypotheses,
                                         options.return_attention,
                                         options.return_scores);
    std::vector<TranslationResult> results(source.size(), empty_result);

    const size_t max_batch_size = options.support_batch_translation() ? 0 : 1;
    for (const auto& batch : rebatch_input(source, target_prefix, max_batch_size)) {
      auto batch_results = run_batch_translation(batch.source, batch.target, options);
      for (size_t i = 0; i < batch_results.size(); ++i)
        results[batch.example_index[i]] = std::move(batch_results[i]);
    }

    return results;
  }

  static void
  replace_unknowns(const std::vector<std::string>& source,
                   std::vector<std::string>& hypotheses,
                   const std::vector<std::vector<float>>& attention) {
    for (size_t t = 0; t < hypotheses.size(); ++t) {
      if (hypotheses[t] == Vocabulary::unk_token) {
        const std::vector<float>& attention_values = attention[t];
        const size_t pos = std::distance(attention_values.begin(),
                                         std::max_element(attention_values.begin(),
                                                          attention_values.end()));

        hypotheses[t] = source[pos];
      }
    }
  }

  std::vector<TranslationResult>
  Translator::run_batch_translation(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target_prefix,
                                    const TranslationOptions& options) {
    PROFILE("run_batch_translation");
    assert_has_model();
    auto scoped_device_setter = _model->get_scoped_device_setter();
    const Device device = _model->device();

    if (!_allocator)
      _allocator = &ctranslate2::get_allocator(device);

    const auto& source_vocabulary = _seq2seq_model->get_source_vocabulary();
    const auto& target_vocabulary = _seq2seq_model->get_target_vocabulary();
    const auto source_ids = source_vocabulary.to_ids(source,
                                                     _seq2seq_model->with_source_bos(),
                                                     _seq2seq_model->with_source_eos());
    const auto target_prefix_ids = target_vocabulary.to_ids(target_prefix);

    const dim_t preferred_size_multiple = _model->preferred_size_multiple();
    std::pair<StorageView, StorageView> inputs = layers::make_sequence_inputs(
      source_ids,
      device,
      preferred_size_multiple);
    StorageView& ids = inputs.first;
    StorageView& lengths = inputs.second;

    // Encode sequence.
    StorageView encoded(_encoder->output_type(), device);
    (*_encoder)(ids, lengths, encoded);

    // If set, extract the subset of candidates to generate.
    const auto* vocabulary_map = _seq2seq_model->get_vocabulary_map();
    std::vector<size_t> output_ids_map;
    if (options.use_vmap && vocabulary_map) {
      output_ids_map = vocabulary_map->get_candidates(source);
    } else if (target_vocabulary.size() % preferred_size_multiple != 0) {
      output_ids_map.resize(target_vocabulary.size());
      std::iota(output_ids_map.begin(), output_ids_map.end(), size_t(0));
    }

    if (!output_ids_map.empty()) {
      // Pad vocabulary size to the preferred size multiple.
      while (output_ids_map.size() % preferred_size_multiple != 0)
        output_ids_map.push_back(0);

      _decoder->set_vocabulary_mask(
        StorageView({static_cast<dim_t>(output_ids_map.size())},
                    std::vector<int32_t>(output_ids_map.begin(), output_ids_map.end()),
                    device));
    } else {
      _decoder->reset_vocabulary_mask();
    }

    // Decode.
    layers::DecoderState state = _decoder->initial_state();
    state.emplace(std::string("memory"), std::move(encoded));
    state.emplace(std::string("memory_lengths"), std::move(lengths));
    const size_t start_id = target_vocabulary.to_id(_seq2seq_model->with_target_bos()
                                                    ? Vocabulary::bos_token
                                                    : Vocabulary::eos_token);
    const size_t end_id = target_vocabulary.to_id(Vocabulary::eos_token);
    const size_t batch_size = source.size();
    const std::vector<size_t> start_ids(batch_size, start_id);
    std::vector<GenerationResult<size_t>> results = decode(
      *_decoder,
      state,
      *make_search_strategy(options),
      *make_sampler(options),
      start_ids,
      !target_prefix_ids.empty() ? &target_prefix_ids : nullptr,
      !output_ids_map.empty() ? &output_ids_map : nullptr,
      end_id,
      options.max_decoding_length,
      options.min_decoding_length,
      options.num_hypotheses,
      options.return_alternatives,
      options.return_scores,
      options.return_attention || options.replace_unknowns,
      options.normalize_scores);

    // Convert generated ids to tokens.
    std::vector<TranslationResult> final_results;
    final_results.reserve(results.size());
    for (size_t i = 0; i < batch_size; ++i) {
      GenerationResult<size_t>& result = results[i];
      auto hypotheses = target_vocabulary.to_tokens(result.hypotheses);

      if (result.has_attention()) {
        // Remove padding and special tokens in attention vectors.
        const size_t offset = size_t(_seq2seq_model->with_source_bos());
        const size_t length = source[i].size();

        for (size_t h = 0; h < result.attention.size(); ++h) {
          auto& attention = result.attention[h];

          for (auto& vector : attention) {
            vector = std::vector<float>(vector.begin() + offset, vector.begin() + offset + length);
          }

          if (options.replace_unknowns)
            replace_unknowns(source[i], hypotheses[h], attention);
        }

        if (!options.return_attention)
          result.attention.clear();
      }

      final_results.emplace_back(std::move(hypotheses),
                                 std::move(result.scores),
                                 std::move(result.attention));
    }
    return final_results;
  }

  Device Translator::device() const {
    assert_has_model();
    return _model->device();
  }

  int Translator::device_index() const {
    assert_has_model();
    return _model->device_index();
  }

  ComputeType Translator::compute_type() const {
    assert_has_model();
    return _model->compute_type();
  }

  void Translator::set_model(const std::string& model_dir) {
    models::ModelFileReader model_reader(model_dir);
    set_model(model_reader);
  }

  void Translator::set_model(models::ModelReader& model_reader) {
    Device device = Device::CPU;
    int device_index = 0;
    ComputeType compute_type = ComputeType::DEFAULT;
    if (_model) {
      device = _model->device();
      device_index = _model->device_index();
      compute_type = _model->compute_type();
    }
    set_model(models::Model::load(model_reader, device, device_index, compute_type));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    const auto* seq2seq_model = dynamic_cast<const models::SequenceToSequenceModel*>(model.get());
    if (!seq2seq_model)
      throw std::invalid_argument("Translator expects a model of type SequenceToSequenceModel");
    _model = model;
    _seq2seq_model = seq2seq_model;
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder = seq2seq_model->make_encoder();
    _decoder = seq2seq_model->make_decoder();
  }

  void Translator::detach_model() {
    if (!_model)
      return;
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder.reset();
    _decoder.reset();
    _model.reset();
    _seq2seq_model = nullptr;
  }

  void Translator::assert_has_model() const {
    if (!_model)
      throw std::runtime_error("No model is attached to this translator");
  }

  std::vector<Batch>
  rebatch_input(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                size_t max_batch_size,
                BatchType batch_type) {
    if (!target.empty() && target.size() != source.size())
      throw std::invalid_argument("Batch size mismatch: got "
                                  + std::to_string(source.size()) + " for source and "
                                  + std::to_string(target.size()) + " for target");

    const size_t global_batch_size = source.size();
    if (max_batch_size == 0) {
      max_batch_size = global_batch_size;
      batch_type = BatchType::Examples;
    }

    // Sorting the source inputs from the longest to the shortest has 2 benefits:
    //
    // 1. When max_batch_size is smaller that the number of inputs, we prefer translating
    //    together sentences that have a similar length for improved efficiency.
    // 2. Decoding functions remove finished translations from the batch. On CPU, arrays are
    //    updated in place so it is more efficient to remove content at the end. Shorter sentences
    //    are more likely to finish first so we sort the batch accordingly.
    std::vector<size_t> example_index(global_batch_size);
    std::iota(example_index.begin(), example_index.end(), 0);
    std::sort(example_index.begin(), example_index.end(),
              [&source](size_t i1, size_t i2) {
                return source[i1].size() > source[i2].size();
              });

    // Ignore empty examples.
    // As example_index is sorted from longest to shortest, we simply pop empty examples
    // from the back.
    while (!example_index.empty() && source[example_index.back()].empty())
      example_index.pop_back();

    std::vector<Batch> batches;
    if (example_index.empty())
      return batches;
    batches.reserve(example_index.size());

    ParallelBatchReader batch_reader;
    batch_reader.add(std::make_unique<VectorReader>(index_vector(source, example_index)));
    if (!target.empty())
      batch_reader.add(std::make_unique<VectorReader>(index_vector(target, example_index)));

    for (size_t offset = 0;;) {
      auto batch_tokens = batch_reader.get_next(max_batch_size, batch_type);
      if (batch_tokens[0].empty())
        break;

      Batch batch;
      batch.source = std::move(batch_tokens[0]);
      if (batch_tokens.size() > 1)
        batch.target = std::move(batch_tokens[1]);

      const size_t batch_size = batch.source.size();
      batch.example_index.insert(batch.example_index.begin(),
                                 example_index.begin() + offset,
                                 example_index.begin() + offset + batch_size);
      offset += batch_size;

      batches.emplace_back(std::move(batch));
    }

    return batches;
  }

}
