#pragma once

#include <string>
#include <vector>

#include "models/model.h"
#include "translation_result.h"

namespace ctranslate2 {

  struct TranslationOptions {
    // Maximum batch size to run the model on (set 0 to forward the input as is).
    // When more inputs are passed to translate(), they will be internally sorted by length
    // to increase efficiency.
    size_t max_batch_size = 0;

    // Beam size to use for beam search (set 1 to run greedy search).
    size_t beam_size = 2;
    // Length penalty value to apply during beam search.
    float length_penalty = 0;

    // Decoding length constraints.
    size_t max_decoding_length = 250;
    size_t min_decoding_length = 1;

    // Randomly sample from the top K candidates (not compatible with beam search, set to 0
    // to sample from the full output distribution).
    size_t sampling_topk = 1;
    // High temperature increase randomness.
    float sampling_temperature = 1;

    // Use the vocabulary map included in the model directory.
    bool use_vmap = false;

    // Number of hypotheses to store in the TranslationResult class (should be smaller than
    // beam_size).
    size_t num_hypotheses = 1;

    // Store attention vectors in the TranslationResult class.
    bool return_attention = false;
  };

  // This class holds all information required to translate from a model. Copying
  // a Translator instance does not duplicate the model data and the copy can
  // be safely executed in parallel.
  class Translator {
  public:
    Translator(const std::string& model_dir, Device device = Device::CPU, int device_index = 0);
    Translator(const std::shared_ptr<const models::Model>& model);
    Translator(const Translator& other);

    TranslationResult
    translate(const std::vector<std::string>& tokens);
    TranslationResult
    translate(const std::vector<std::string>& tokens,
              const TranslationOptions& options);
    TranslationResult
    translate_with_prefix(const std::vector<std::string>& source,
                          const std::vector<std::string>& target_prefix,
                          const TranslationOptions& options);

    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& tokens);
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& tokens,
                    const TranslationOptions& options);
    std::vector<TranslationResult>
    translate_batch_with_prefix(const std::vector<std::vector<std::string>>& source,
                                const std::vector<std::vector<std::string>>& target_prefix,
                                const TranslationOptions& options);

    Device device() const;
    int device_index() const;
    ComputeType compute_type() const;

    //Change only the model while keeping the same device
    //and compute type
    void set_model(const std::string& model_dir);
    void set_model(const std::shared_ptr<const models::Model>& model);

  private:
    void make_graph();

    std::vector<TranslationResult>
    translate_tokens(const std::vector<std::vector<std::string>>& source,
                     const std::vector<std::vector<std::string>>& target_prefix,
                     const TranslationOptions& options);
    std::vector<TranslationResult>
    run_translation(const std::vector<std::vector<std::string>>& source,
                    const std::vector<std::vector<std::string>>& target_prefix,
                    const TranslationOptions& options);

    std::shared_ptr<const models::Model> _model;
    std::unique_ptr<layers::Encoder> _encoder;
    std::unique_ptr<layers::Decoder> _decoder;
  };

}
