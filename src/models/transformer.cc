#include "ctranslate2/models/transformer.h"

#include "ctranslate2/models/model_factory.h"
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace models {

    static bool replace(std::string& str, const std::string& from, const std::string& to) {
      size_t start_pos = str.find(from);
      if (start_pos == std::string::npos)
        return false;
      str.replace(start_pos, from.length(), to);
      return true;
    }


    // Empty spec name, TransformerBase, and TransformerBig are there for backward compatibility.
    static auto register_empty = register_model<TransformerModel>("", /*num_heads=*/8);
    static auto register_base = register_model<TransformerModel>("TransformerBase", /*num_heads=*/8);
    static auto register_big = register_model<TransformerModel>("TransformerBig", /*num_heads=*/16);
    static auto register_generic = register_model<TransformerModel>("TransformerSpec");

    TransformerModel::TransformerModel(size_t num_heads)
      : _num_heads(num_heads) {
    }

    size_t TransformerModel::current_spec_revision() const {
      return 6;
    }

    void TransformerModel::update_variable_name(std::string& variable_name) const {
      if (spec_revision() == 1) {
        // In the first specification, variable names were the names defined by OpenNMT-tf V1.
        replace(variable_name, "transformer/", "");
        replace(variable_name, ":0", "");
        replace(variable_name, "w_embs", "embeddings/weight");
        replace(variable_name, "kernel", "weight");
        replace(variable_name, "LayerNorm", "layer_norm");
        replace(variable_name, "dense", "projection");
        replace(variable_name, "conv1d_", "linear_");
        replace(variable_name, "conv1d", "linear_0");
        if (variable_name.find("encoder") != std::string::npos) {
          replace(variable_name, "multi_head", "self_attention");
        } else {
          replace(variable_name, "masked_multi_head", "self_attention");
          replace(variable_name, "multi_head", "attention");
        }
      }
    }

    bool TransformerModel::is_linear_weight(const std::string& variable_name) const {
      // Linear weights are all variables that are quantizable and not under the "embeddings" scope.
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    bool TransformerModel::is_packable(const std::string& variable_name) const {
      // Disallow packing for the last linear layer if it can be dynamically masked.
      return (is_linear_weight(variable_name)
              && (!get_vocabulary_map() || variable_name.find("projection") == std::string::npos));
    }

    void TransformerModel::initialize(ModelReader& model_reader) {
      SequenceToSequenceModel::initialize(model_reader);

      if (spec_revision() < 3) {
        register_variable("num_heads", StorageView(int8_t(_num_heads)));
      }

      if (spec_revision() < 5) {
        register_variable_alias("encoder/num_heads", "num_heads");
        register_variable_alias("encoder/pre_norm", "pre_norm");
        register_variable_alias("encoder/activation", "activation");
        register_variable_alias("encoder/embeddings_merge", "embeddings_merge");

        register_variable_alias("decoder/num_heads", "num_heads");
        register_variable_alias("decoder/pre_norm", "pre_norm");
        register_variable_alias("decoder/activation", "activation");
        register_variable_alias("decoder/alignment_layer", "alignment_layer");
        register_variable_alias("decoder/alignment_heads", "alignment_heads");
      }
    }

    std::unique_ptr<SequenceToSequenceReplica> TransformerModel::as_sequence_to_sequence() const {
      const auto scoped_device_setter = get_scoped_device_setter();

      auto encoder = std::make_unique<layers::TransformerEncoder>(*this, "encoder");
      auto decoder = std::make_unique<layers::TransformerDecoder>(*this, "decoder");

      const auto model = std::static_pointer_cast<const TransformerModel>(shared_from_this());
      return std::make_unique<EncoderDecoderReplica>(model, std::move(encoder), std::move(decoder));
    }

    std::unique_ptr<Model> TransformerModel::clone() const {
      return std::make_unique<TransformerModel>(*this);
    }


    static auto register_decoder = register_model<TransformerDecoderModel>("TransformerDecoderSpec");

    size_t TransformerDecoderModel::current_spec_revision() const {
      return 2;
    }

    void TransformerDecoderModel::initialize(ModelReader& model_reader) {
      LanguageModel::initialize(model_reader);

      if (spec_revision() < 2) {
        register_variable_alias("decoder/num_heads", "num_heads");
        register_variable_alias("decoder/pre_norm", "pre_norm");
        register_variable_alias("decoder/activation", "activation");
      }
    }

    std::unique_ptr<SequenceGeneratorReplica>
    TransformerDecoderModel::as_sequence_generator() const {
      const auto scoped_device_setter = get_scoped_device_setter();

      auto decoder = std::make_unique<layers::TransformerDecoder>(*this, "decoder");

      const auto model = std::static_pointer_cast<const TransformerDecoderModel>(shared_from_this());
      return std::make_unique<DecoderReplica>(model, std::move(decoder));
    }

    bool TransformerDecoderModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> TransformerDecoderModel::clone() const {
      return std::make_unique<TransformerDecoderModel>(*this);
    }

  }
}
