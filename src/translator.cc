#include "ctranslate2/translator.h"

namespace ctranslate2 {

  Translator::Translator(const std::string& model_dir,
                         Device device,
                         int device_index,
                         ComputeType compute_type) {
    set_model(models::Model::load(model_dir, device, device_index, compute_type));
  }

  Translator::Translator(models::ModelReader& model_reader,
                         Device device,
                         int device_index,
                         ComputeType compute_type)
  {
    set_model(models::Model::load(model_reader, device, device_index, compute_type));
  }

  Translator::Translator(const std::shared_ptr<const models::Model>& model) {
    set_model(model);
  }

  Translator::Translator(const Translator& other) {
    set_model(other.get_model());
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
    std::vector<std::vector<std::string>> batch_source(1, source);
    std::vector<std::vector<std::string>> batch_target_prefix;
    if (!target_prefix.empty())
      batch_target_prefix.emplace_back(target_prefix);
    return translate_batch_with_prefix(batch_source, batch_target_prefix, options)[0];
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
    return _replica->translate(source, target_prefix, options);
  }

  std::vector<ScoringResult>
  Translator::score_batch(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target,
                          const ScoringOptions& options) {
    return _replica->score(source, target, options);
  }

  Device Translator::device() const {
    return get_model()->device();
  }

  int Translator::device_index() const {
    return get_model()->device_index();
  }

  ComputeType Translator::compute_type() const {
    return get_model()->compute_type();
  }

  void Translator::set_model(const std::string& model_dir) {
    models::ModelFileReader model_reader(model_dir);
    set_model(model_reader);
  }

  void Translator::set_model(models::ModelReader& model_reader) {
    set_model(models::Model::load(model_reader, device(), device_index(), compute_type()));
  }

  void Translator::set_model(const std::shared_ptr<const models::Model>& model) {
    _replica = model->as_sequence_to_sequence();
  }

}
