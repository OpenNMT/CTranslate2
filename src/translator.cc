#include "ctranslate2/translator.h"

#include <algorithm>
#include <numeric>

#include "ctranslate2/decoding.h"

namespace ctranslate2 {

  static std::vector<size_t>
  tokens_to_ids(const std::vector<std::string>& tokens,
                const Vocabulary& vocab) {
    std::vector<size_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens)
      ids.push_back(vocab.to_id(token));
    return ids;
  }

  static std::vector<std::vector<size_t>>
  tokens_to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
                const Vocabulary& vocab) {
    std::vector<std::vector<size_t>> batch_ids;
    batch_ids.reserve(batch_tokens.size());
    for (const auto& tokens : batch_tokens)
      batch_ids.emplace_back(tokens_to_ids(tokens, vocab));
    return batch_ids;
  }

  static void sort_from_longest_to_shortest(std::vector<std::vector<size_t>>& ids,
                                            std::vector<size_t>& original_to_sorted_index) {
    std::vector<size_t> sorted_to_original_index(ids.size());
    std::iota(sorted_to_original_index.begin(), sorted_to_original_index.end(), 0);
    std::sort(sorted_to_original_index.begin(), sorted_to_original_index.end(),
              [&ids](size_t i1, size_t i2) { return ids[i1].size() > ids[i2].size(); });

    original_to_sorted_index.resize(ids.size());
    std::vector<std::vector<size_t>> new_ids;
    new_ids.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      size_t original_index = sorted_to_original_index[i];
      original_to_sorted_index[original_index] = i;
      new_ids.emplace_back(std::move(ids[original_index]));
    }
    ids = std::move(new_ids);
  }

  static std::pair<StorageView, StorageView>
  make_inputs(const std::vector<std::vector<size_t>>& ids, Device device) {
    size_t batch_size = ids.size();

    // Record lengths and maximum length.
    size_t max_length = 0;
    StorageView lengths({batch_size}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = ids[i].size();
      lengths.at<int32_t>(i) = length;
      max_length = std::max(max_length, length);
    }

    // Make 2D input.
    StorageView input({batch_size, max_length}, DataType::DT_INT32);
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t t = 0; t < ids[i].size(); ++t)
        input.at<int32_t>({i, t}) = ids[i][t];
    }

    return std::make_pair(input.to(device), lengths.to(device));
  }


  Translator::Translator(const std::string& model_dir, Device device, int device_index)
    : _model(models::Model::load(model_dir, device, device_index)) {
    make_graph();
  }

  Translator::Translator(const std::shared_ptr<models::Model>& model)
    : _model(model) {
    make_graph();
  }

  Translator::Translator(const Translator& other)
    : _model(other._model) {
    make_graph();
  }

  void Translator::make_graph() {
    auto scoped_device_setter = _model->get_scoped_device_setter();
    _encoder = _model->make_encoder();
    _decoder = _model->make_decoder();
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens) {
    TranslationOptions options;
    return translate(tokens, options);
  }

  TranslationResult
  Translator::translate(const std::vector<std::string>& tokens,
                        const TranslationOptions& options) {
    std::vector<std::vector<std::string>> batch_tokens(1, tokens);
    return translate_batch(batch_tokens, options)[0];
  }

  TranslationResult
  Translator::translate_with_prefix(const std::vector<std::string>& source,
                                    const std::vector<std::string>& target_prefix,
                                    const TranslationOptions& options) {
    std::vector<std::vector<std::string>> batch_source(1, source);
    std::vector<std::vector<std::string>> batch_target_prefix(1, target_prefix);
    return translate_batch_with_prefix(batch_source, batch_target_prefix, options)[0];
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens) {
    TranslationOptions options;
    return translate_batch(batch_tokens, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch(const std::vector<std::vector<std::string>>& batch_tokens,
                              const TranslationOptions& options) {
    std::vector<std::vector<std::string>> target_prefix;
    return translate_batch_with_prefix(batch_tokens, target_prefix, options);
  }

  std::vector<TranslationResult>
  Translator::translate_batch_with_prefix(const std::vector<std::vector<std::string>>& source,
                                          const std::vector<std::vector<std::string>>& target_prefix,
                                          const TranslationOptions& options) {
    const auto& source_vocab = _model->get_source_vocabulary();
    const auto& target_vocab = _model->get_target_vocabulary();
    const auto& vocab_map = _model->get_vocabulary_map();
    auto& encoder = *_encoder;
    auto& decoder = *_decoder;

    size_t batch_size = source.size();
    bool with_prefix = !target_prefix.empty();

    // Check options and inputs.
    if (options.num_hypotheses > options.beam_size)
      throw std::invalid_argument("The number of hypotheses can not be greater than the beam size");
    if (options.use_vmap && vocab_map.empty())
      throw std::invalid_argument("use_vmap is set but the model does not include a vocabulary map");
    if (options.min_decoding_length > options.max_decoding_length)
      throw std::invalid_argument("min_decoding_length is greater than max_decoding_length");
    if (with_prefix) {
      if (options.return_attention)
        throw std::invalid_argument(
          "Prefixed translation currently does not support returning attention vectors");
      if (batch_size > 1)
        throw std::invalid_argument(
          "Prefixed translation currently does not support batch inputs");
      if (target_prefix.size() != batch_size)
        throw std::invalid_argument("Batch size mismatch: got "
                                    + std::to_string(batch_size) + " for source and "
                                    + std::to_string(target_prefix.size()) + " for target prefix");
    }

    auto scoped_device_setter = _model->get_scoped_device_setter();
    auto device = _model->device();

    auto source_ids = tokens_to_ids(source, source_vocab);

    // Decoding functions remove finished translations from the batch. On CPU, arrays are
    // updated in place so it is more efficient to remove content at the end. Shorter sentences
    // are more likely to finish first so we sort the batch accordingly.
    std::vector<size_t> sorted_index;
    if (batch_size > 1 && device == Device::CPU)
      sort_from_longest_to_shortest(source_ids, sorted_index);

    auto inputs = make_inputs(source_ids, device);
    StorageView& ids = inputs.first;
    StorageView& lengths = inputs.second;

    // Encode sequence.
    StorageView encoded(device);
    encoder(ids, lengths, encoded);

    // If set, extract the subset of candidates to generate.
    StorageView candidates(DataType::DT_INT32, device);
    if (options.use_vmap && !vocab_map.empty()) {
      auto candidates_vec = vocab_map.get_candidates<int32_t>(source);
      candidates.resize({candidates_vec.size()});
      candidates.copy_from(candidates_vec.data(), candidates_vec.size(), Device::CPU);
    }
    decoder.reduce_vocab(candidates);

    // Decode.
    size_t start_step = 0;
    size_t start_token = target_vocab.to_id(Vocabulary::bos_token);
    size_t end_token = target_vocab.to_id(Vocabulary::eos_token);
    StorageView sample_from({batch_size}, static_cast<int32_t>(start_token));
    std::vector<std::vector<std::vector<size_t>>> sampled_ids;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> attention;
    auto* attention_ptr = options.return_attention ? &attention : nullptr;
    auto state = decoder.initial_state();

    // Forward target prefix, if set (only batch_size = 1 for now).
    if (with_prefix) {
      // TODO: Forward all timesteps at once. This requires supporting the masking
      // of future steps.
      const auto& prefix = target_prefix.front();
      auto prefix_ids = tokens_to_ids(prefix, target_vocab);
      start_step = prefix.size();
      for (size_t i = 0; i < start_step; ++i) {
        auto input = sample_from.to(device);
        input.reshape({batch_size, 1});
        decoder(i, input, encoded, lengths, state);
        sample_from.at<int32_t>(0) = prefix_ids[i];
      }
    }

    if (options.beam_size == 1)
      greedy_search(decoder,
                    state,
                    sample_from,
                    candidates,
                    encoded,
                    lengths,
                    start_step,
                    end_token,
                    options.max_decoding_length,
                    options.min_decoding_length,
                    sampled_ids,
                    scores,
                    attention_ptr);
    else
      beam_search(decoder,
                  state,
                  sample_from,
                  candidates,
                  encoded,
                  lengths,
                  start_step,
                  end_token,
                  options.max_decoding_length,
                  options.min_decoding_length,
                  options.beam_size,
                  options.num_hypotheses,
                  options.length_penalty,
                  sampled_ids,
                  scores,
                  attention_ptr);

    // Build results.
    std::vector<TranslationResult> results;
    results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      std::vector<std::vector<std::string>> hypotheses;
      size_t num_hypotheses = sampled_ids[i].size();
      hypotheses.resize(num_hypotheses);
      for (size_t h = 0; h < num_hypotheses; ++h) {
        if (with_prefix)
          hypotheses[h] = target_prefix[i];
        for (auto id : sampled_ids[i][h])
          hypotheses[h].push_back(target_vocab.to_token(id));
      }
      const auto* attn = i < attention.size() ? &attention[i] : nullptr;
      results.emplace_back(hypotheses, scores[i], attn);
    }

    if (sorted_index.empty())
      return results;
    else {
      // Reorder results based on original batch index.
      std::vector<TranslationResult> final_results;
      final_results.reserve(results.size());
      for (auto index : sorted_index)
        final_results.emplace_back(std::move(results[index]));
      return final_results;
    }
  }

  Device Translator::device() const {
    return _model->device();
  }

}
