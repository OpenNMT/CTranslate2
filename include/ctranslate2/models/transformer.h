#pragma once

#include "sequence_to_sequence.h"
#include "language_model.h"

#include "ctranslate2/ops/activation.h"

namespace ctranslate2 {
  namespace models {

    class TransformerModel : public SequenceToSequenceModel {
    public:
      TransformerModel(size_t num_heads = 0);
      size_t current_spec_revision() const override;
      std::unique_ptr<SequenceToSequenceReplica> as_sequence_to_sequence() const override;

    protected:
      void update_variable_name(std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      bool is_packable(const std::string& variable_name) const override;
      void initialize(ModelReader& model_reader) override;
      std::unique_ptr<Model> clone() const override;

    private:
      size_t _num_heads;
    };


    class TransformerDecoderModel : public LanguageModel {
    public:
      size_t current_spec_revision() const override;
      std::unique_ptr<SequenceGeneratorReplica> as_sequence_generator() const override;

    protected:
      bool is_linear_weight(const std::string& variable_name) const override;
      void initialize(ModelReader& model_reader) override;
      std::unique_ptr<Model> clone() const override;
    };

  }
}
