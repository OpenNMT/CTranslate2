#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/moonshine.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct MoonshineOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 5;

      // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
      // The decoding will continue until beam_size*patience hypotheses are finished.
      float patience = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Suppress blank outputs at the beginning of the sampling.
      bool suppress_blank = true;

      // List of token IDs to suppress.
      // -1 will suppress a default set of symbols as defined in the model config.json file.
      std::vector<int> suppress_tokens = {-1};
    };

    struct MoonshineGenerationResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    class MoonshineModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class MoonshineReplica : public ModelReplica {
    public:
      static std::unique_ptr<MoonshineReplica> create_from_model(const Model& model);

      MoonshineReplica(const std::shared_ptr<const MoonshineModel>& model);

      StorageView encode(StorageView features, const bool to_cpu);

      std::vector<MoonshineGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<std::string>>& prompts,
               const MoonshineOptions& options);

      std::vector<MoonshineGenerationResult>
      generate(StorageView features,
               const std::vector<std::vector<size_t>>& prompts,
               const MoonshineOptions& options);

    private:
      const std::shared_ptr<const MoonshineModel> _model;
      const std::unique_ptr<layers::MoonshinePreprocessor> _preprocessor;
      const std::unique_ptr<layers::MoonshineEncoder> _encoder;
      const std::unique_ptr<layers::MoonshineDecoder> _decoder;

      size_t _sot_id;
      size_t _eot_id;

      StorageView maybe_encode(StorageView features);
    };

    class Moonshine : public ReplicaPool<MoonshineReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

      std::vector<std::future<MoonshineGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<std::string>> prompts,
               MoonshineOptions options = {});

      std::vector<std::future<MoonshineGenerationResult>>
      generate(const StorageView& features,
               std::vector<std::vector<size_t>> prompts,
               MoonshineOptions options = {});
    };

  }
}
