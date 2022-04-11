#pragma once

#include "sequence_to_sequence.h"

namespace ctranslate2 {
  namespace models {

    class TransformerModel : public SequenceToSequenceModel {
    public:
      TransformerModel(size_t num_heads = 0);
      size_t current_spec_revision() const override;
      std::unique_ptr<SequenceToSequenceReplica> as_sequence_to_sequence() const override;

    protected:
      bool is_linear_weight(const std::string& variable_name) const override;
      bool is_packable(const std::string& variable_name) const override;
      void register_variable(std::string name, StorageView variable) override;
      void register_variable_alias(std::string alias, std::string variable_name) override;

    private:
      size_t _num_heads;
    };

  }
}
