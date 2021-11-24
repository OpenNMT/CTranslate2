#pragma once

#include "ctranslate2/decoding.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/scoring.h"
#include "ctranslate2/vocabulary.h"
#include "ctranslate2/vocabulary_map.h"

namespace ctranslate2 {
  namespace models {

    class SequenceToSequenceModel : public Model {
    public:
      const Vocabulary& get_source_vocabulary() const;
      const Vocabulary& get_target_vocabulary() const;
      const VocabularyMap* get_vocabulary_map() const;

      void forward_encoder(const std::vector<std::vector<std::string>>& source,
                           StorageView& memory,
                           StorageView& memory_lengths) const;

      void forward_decoder(layers::DecoderState& state,
                           const std::vector<std::vector<std::string>>& target,
                           StorageView& logits) const;

      void forward(const std::vector<std::vector<std::string>>& source,
                   const std::vector<std::vector<std::string>>& target,
                   StorageView& logits) const;

      std::vector<ScoringResult>
      score(const std::vector<std::vector<std::string>>& source,
            const std::vector<std::vector<std::string>>& target,
            const size_t max_input_length = 0) const;

      std::vector<GenerationResult<std::string>>
      sample(const std::vector<std::vector<std::string>>& source,
             const std::vector<std::vector<std::string>>& target_prefix = {},
             const SearchStrategy& search_strategy = GreedySearch(),
             const Sampler& sampler = BestSampler(),
             const bool use_vmap = false,
             const size_t max_input_length = 0,
             const size_t max_output_length = 256,
             const size_t min_output_length = 1,
             const size_t num_hypotheses = 1,
             const bool return_alternatives = false,
             const bool return_scores = false,
             const bool return_attention = false,
             const bool replace_unknowns = false,
             const bool normalize_scores = false,
             const float repetition_penalty = 1) const;

    protected:
      SequenceToSequenceModel(ModelReader& model_reader, size_t spec_revision);
      SequenceToSequenceModel(const SequenceToSequenceModel& model);
      virtual void finalize() override;
      virtual void build() override;

      virtual std::unique_ptr<layers::Encoder> make_encoder() const = 0;
      virtual std::unique_ptr<layers::Decoder> make_decoder() const = 0;

      std::unique_ptr<layers::Encoder> _encoder;
      std::unique_ptr<layers::Decoder> _decoder;

    private:
      std::shared_ptr<const Vocabulary> _source_vocabulary;
      std::shared_ptr<const Vocabulary> _target_vocabulary;
      std::shared_ptr<const VocabularyMap> _vocabulary_map;

      bool _with_source_bos = false;
      bool _with_source_eos = false;
      bool _with_target_bos = true;
    };

  }
}
