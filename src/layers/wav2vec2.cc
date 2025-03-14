#include "ctranslate2/layers/wav2vec2.h"


namespace ctranslate2 {
  namespace layers {

    Wav2Vec2LayerNormConvLayer::Wav2Vec2LayerNormConvLayer(const models::Model& model,
                                                           const std::string& scope,
                                                           dim_t stride,
                                                           dim_t padding)
      : _stride(stride)
      , _padding(padding)
      , _conv(model, scope + "/conv", _stride, _padding)
      , _transpose({0, 2, 1})
      , _output_norm(model, scope + "/layer_norm") {
    }

    void Wav2Vec2LayerNormConvLayer::operator()(const StorageView& input, StorageView& output) const{
      PROFILE("Wav2Vec2LayerNormConvLayer");

      StorageView buffer(input.dtype(), input.device());
      buffer = std::move(input);
      _conv(buffer, output);
      _transpose(output, buffer);
      _output_norm(buffer, output);
      _transpose(output, buffer);
      _gelu(buffer, output);
    }

    Wav2Vec2PosConvLayer::Wav2Vec2PosConvLayer(const models::Model& model, const std::string& scope)
      : _conv(model, scope + "/conv", /*stride=*/1, /*padding=*/64, /*dilation*/1, /*groups*/16)
      , _transpose({0, 2, 1}) {
    }

    void Wav2Vec2PosConvLayer::operator()(const StorageView& input, StorageView& output) const{
      PROFILE("Wav2Vec2PosConvLayer");

      StorageView buffer(input.dtype(), input.device());
      StorageView buffer2(input.dtype(), input.device());
      _transpose(input, buffer);
      _conv(buffer, buffer2);
      ops::Split(2, {buffer.dim(2), 1})(buffer2, buffer, output);
      _gelu(buffer, buffer);
      _transpose(buffer, buffer2);
      ops::Add()(input, buffer2, output);
    }

    Wav2Vec2Encoder::Wav2Vec2Encoder(const models::Model& model, const std::string& scope)
      : _return_logits(model.get_variable_if_exists(scope + "/lm_head/weight"))
      , _upgraded_model(model.get_variable_if_exists(scope + "/fp_projection/weight"))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _transpose({0, 2, 1})
      , _layers(build_layers_list<const TransformerEncoderLayer>(model,
                                                                 scope + "/layer",
                                                                 _num_heads,
                                                                 /*pre_norm=*/true,
                                                                 ops::ActivationType::GELU))
      , _output_norm(model, scope + "/layer_norm")
    {
      if (_upgraded_model) {
        _feat_layer0.emplace(model, scope + "/feat_layer0", /*stride=*/5, /*padding=*/0);
        _feat_layers.emplace(build_layers_list<const Wav2Vec2LayerNormConvLayer>(model,
                                                                                 scope + "/feat_layer",
                                                                                 /*stride=*/2,
                                                                                 /*padding=*/0));
        _fp_norm.emplace(model, scope + "/fp_layer_norm");
        _fp_ff.emplace(model, scope + "/fp_projection", nullptr, true);
        _pos_conv_embed.emplace(model, scope + "/pos_conv_embed");
        if (_return_logits) {
          _lm_head.emplace(model, scope + "/lm_head", nullptr, true);
        }
      }
    }

    void Wav2Vec2Encoder::operator()(const StorageView& features, StorageView& output) {
      PROFILE("Wav2Vec2Encoder");

      // SAD in front-end handles the input length
      if (features.rank() != 3)
        throw std::invalid_argument("Expected input features to have 3 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");
      if (_upgraded_model) {
        // Wav2Vec2FeatureExtractor------------------------------------
        StorageView feat_buffer(features.dtype(), features.device());
        StorageView feat_buffer2(features.dtype(), features.device());
        feat_buffer = std::move(features);
        (*_feat_layer0)(feat_buffer, output); //_feat_layer0(feat_buffer, output);
        feat_buffer = std::move(output);
        for (dim_t l = 0; l < _feat_layers->size(); l++) {
          (*_feat_layers.value()[l])(feat_buffer, output);
          if (l < _feat_layers->size() - 1 ) {
            feat_buffer = std::move(output);
          }
        }
        _transpose(output, feat_buffer);
        // Wav2Vec2FeatureProjection-----------------------------------
        (*_fp_norm)(feat_buffer, output); //_fp_norm(feat_buffer, output);
        (*_fp_ff)(output, feat_buffer); //_fp_ff(output, feat_buffer);
        // Wav2Vec2PositionalConvEmbedding-----------------------------
        (*_pos_conv_embed)(feat_buffer, feat_buffer2); //_pos_conv_embed(feat_buffer, feat_buffer2);
        // Wav2Vec2EncoderLayerStableLayerNorm-------------------------
        for (const auto& layer : _layers) {
          (*layer)(feat_buffer2, nullptr, feat_buffer);
          feat_buffer2 = std::move(feat_buffer);
        }
        if (_return_logits) {
          _output_norm(feat_buffer2, feat_buffer);
          (*_lm_head)(feat_buffer, output);
        }
        else {
          _output_norm(feat_buffer2, output);
        }
      }
      else { // backward compatibility for the previous converted model
        StorageView input(features.dtype(), features.device());
        input = features;
        for (const auto& layer : _layers) {
          (*layer)(input, nullptr, output);
          input = std::move(output);
        }

        _output_norm(input, output);
      }
    }

  }
}
