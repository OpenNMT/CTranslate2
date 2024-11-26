#include "ctranslate2/layers/moonshine.h"

namespace ctranslate2 {
  namespace layers {
    MoonshinePreprocessor::MoonshinePreprocessor(const models::Model& model, const std::string& scope)
      : _conv1(model, scope + "/conv1", /*stride=*/64, /*padding=*/0),
        _tanh(),
        _norm(model, scope + "/layernorm"),
        _conv2(model, scope + "/conv2", /*stride=*/3, /*padding=*/0),
        _gelu1(),
        _conv3(model, scope + "/conv3", /*stride=*/2, /*padding=*/0),
        _gelu2(),
        _transpose({0, 2, 1}) {}

    void MoonshinePreprocessor::operator()(const StorageView& features, StorageView& output) {
      if (features.rank() != 2)
        throw std::invalid_argument("Expected input features to have 2 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");

      StorageView input(output_type(), features.device());
      StorageView input_reshaped = std::move(features);
      input_reshaped.expand_dims(1);

      _conv1(input_reshaped, input);
      _tanh(input, input);
      _norm(input, input);

      _conv2(input, output);
      _gelu1(output, output);

      _conv3(output, input);
      _gelu2(input, input);
      _transpose(input, output);
    }


    MoonshineEncoder::MoonshineEncoder(const models::Model& model, const std::string& scope)
      : _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _layers(build_layers_list<const TransformerEncoderLayer>(model,
                                                                 scope + "/layer",
                                                                 _num_heads,
                                                                 /*pre_norm=*/true,
                                                                 ops::ActivationType::GELU))
      , _output_norm(model, scope + "/layer_norm")
    {
    }

    void MoonshineEncoder::operator()(const StorageView& features, StorageView& output) {
      PROFILE("MoonshineEncoder");

      if (features.rank() != 3)
        throw std::invalid_argument("Expected input features to have 3 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");

      StorageView input(output_type(), features.device());

      input = std::move(features);

      for (const auto& layer : _layers) {
        (*layer)(input, nullptr, output);
        input = std::move(output);
      }

      _output_norm(input, output);
    }

  }
}
