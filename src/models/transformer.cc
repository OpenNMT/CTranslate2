#include "ctranslate2/models/transformer.h"

#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/ops/activation.h"

namespace ctranslate2 {
  namespace models {

    static bool replace(std::string& str, const std::string& from, const std::string& to) {
      size_t start_pos = str.find(from);
      if (start_pos == std::string::npos)
        return false;
      str.replace(start_pos, from.length(), to);
      return true;
    }

    static std::string map_v1_variable_name(std::string name) {
      // V1 variable names were simply the names defined by OpenNMT-tf.
      replace(name, "transformer/", "");
      replace(name, ":0", "");
      replace(name, "w_embs", "embeddings/weight");
      replace(name, "kernel", "weight");
      replace(name, "LayerNorm", "layer_norm");
      replace(name, "dense", "projection");
      replace(name, "conv1d_", "linear_");
      replace(name, "conv1d", "linear_0");
      if (name.find("encoder") != std::string::npos) {
        replace(name, "multi_head", "self_attention");
      } else {
        replace(name, "masked_multi_head", "self_attention");
        replace(name, "multi_head", "attention");
      }
      return name;
    }

    TransformerModel::TransformerModel(size_t num_heads)
      : _num_heads(num_heads) {
    }

    size_t TransformerModel::current_spec_revision() const {
      return 4;
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

    void TransformerModel::register_variable(std::string name, StorageView variable) {
      if (spec_revision() == 1)
        name = map_v1_variable_name(std::move(name));
      SequenceToSequenceModel::register_variable(std::move(name), std::move(variable));
    }

    void TransformerModel::register_variable_alias(std::string alias, std::string variable_name) {
      if (spec_revision() == 1) {
        alias = map_v1_variable_name(std::move(alias));
        variable_name = map_v1_variable_name(std::move(variable_name));
      }
      SequenceToSequenceModel::register_variable_alias(std::move(alias), std::move(variable_name));
    }

    std::unique_ptr<SequenceToSequenceReplica> TransformerModel::as_sequence_to_sequence() const {
      const size_t num_heads = get_attribute_with_default<int8_t>("num_heads", _num_heads);
      const bool with_relative_position = get_flag_with_default("with_relative_position", false);
      const bool pre_norm = get_flag_with_default("pre_norm", true);
      const auto activation_type = static_cast<ops::ActivationType>(
        get_attribute_with_default<int8_t>("activation", 0));
      const auto embeddings_merge = static_cast<layers::EmbeddingsMerge>(
        get_attribute_with_default<int8_t>("embeddings_merge", 0));
      const dim_t alignment_layer = get_attribute_with_default<int16_t>("alignment_layer", -1);
      const dim_t alignment_heads = get_attribute_with_default<int16_t>("alignment_heads", 1);
      const bool layernorm_embedding = get_flag_with_default("layernorm_embedding", false);

      const auto scoped_device_setter = get_scoped_device_setter();

      auto encoder = std::make_unique<layers::TransformerEncoder>(*this,
                                                                  "encoder",
                                                                  num_heads,
                                                                  !with_relative_position,
                                                                  pre_norm,
                                                                  activation_type,
                                                                  embeddings_merge,
                                                                  layernorm_embedding);
      auto decoder = std::make_unique<layers::TransformerDecoder>(*this,
                                                                  "decoder",
                                                                  num_heads,
                                                                  !with_relative_position,
                                                                  /*with_encoder_attention=*/true,
                                                                  pre_norm,
                                                                  activation_type,
                                                                  alignment_layer,
                                                                  alignment_heads,
                                                                  layernorm_embedding);

      const auto model = std::static_pointer_cast<const TransformerModel>(shared_from_this());
      return std::make_unique<EncoderDecoderReplica>(model, std::move(encoder), std::move(decoder));
    }

  }
}
