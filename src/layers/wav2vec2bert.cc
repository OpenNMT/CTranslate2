#include "ctranslate2/layers/wav2vec2bert.h"

namespace ctranslate2 {
  namespace layers {

    EncoderLayer::EncoderLayer(const models::Model& model,
                               const std::string& scope,
                               const bool pre_norm,
                               const ops::ActivationType activation_type,
                               const bool use_flash_attention)
      : _ffn1_layer_norm(model, scope + "/enc_ffn1_layer_norm")
      , _ff1(model, scope + "/enc_ffn1", pre_norm, activation_type)
      , _self_attn_layer_norm(model, scope + "/enc_attn_layer_norm")
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 16))
      , _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/enc_attn",
                        _num_heads,
                        /*self_attention=*/true,
                        pre_norm)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/enc_attn",
                        _num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _transpose({0, 2, 1})
      , _layer_norm(model, scope + "/enc_conv_layer_norm")
      , _pconv1(model, scope + "/enc_conv_pointwise_conv1", /*stride=*/1, /*padding=*/0)
      , _dconv(model, scope + "/enc_conv_depthwise_conv", /*stride=*/1, /*padding=*/0, /*dilation*/1, /*groups*/1024)
      , _dlayer_norm(model, scope +"/enc_conv_depthwise_layer_norm")
      , _pconv2(model, scope + "/enc_conv_pointwise_conv2", /*stride=*/1, /*padding=*/0)
      , _ffn2_layer_norm(model, scope + "/enc_ffn2_layer_norm")
      , _ff2(model, scope + "/enc_ffn2", pre_norm, activation_type)
      , _final_layer_norm(model, scope + "/enc_final_layer_norm") {
      }

    void EncoderLayer::operator()(const StorageView& input, StorageView& output) const{
      PROFILE("EncoderLayer");

      StorageView buffer1(input.dtype(), input.device());
      StorageView buffer2(input.dtype(), input.device());
      StorageView buffer3(input.dtype(), input.device());
      StorageView residual(input.dtype(), input.device());
      StorageView m(static_cast<float>(0.5));

      _ffn1_layer_norm(input, buffer1);
      _ff1(buffer1, buffer2);
      ops::Mul()(buffer2, m, buffer1);
      ops::Add()(buffer1, input, buffer2);
      residual.copy_from(buffer2);

      _self_attn_layer_norm(buffer2, buffer1);
      (*_self_attention)(buffer1,
                         buffer1,
                         nullptr,
                         buffer2,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         true,
                         nullptr);
      ops::Add()(buffer2, residual, buffer1);

      residual.copy_from(buffer1);
      _layer_norm(buffer1, buffer2);

      _transpose(buffer2, buffer1);

      _pconv1(buffer1, buffer2);
      std::vector<StorageView*> out{&buffer1, &buffer3};
      ops::Split(1, {buffer2.dim(1)/2, buffer2.dim(1)/2})(buffer2, out);
      _sigmoid(buffer3, buffer3);
      ops::Mul()(buffer1, buffer3, buffer2);

      StorageView buffer_zeros({buffer2.dim(0), buffer2.dim(1), 30},
                               buffer2.dtype(),
                               buffer2.device());
      buffer_zeros.zero();
      ops::Concat(-1)({&buffer_zeros, &buffer2}, buffer1);
      _dconv(buffer1, buffer2);
      _transpose(buffer2, buffer1);
      _dlayer_norm(buffer1, buffer2);
      _transpose(buffer2, buffer1);
      _swish(buffer1, buffer2);
      _pconv2(buffer2, buffer1);
      _transpose(buffer1, buffer2);
      ops::Add()(buffer2, residual, buffer1);

      residual.copy_from(buffer1);
      _ffn2_layer_norm(buffer1, buffer2);
      _ff2(buffer2, buffer1);
      ops::Mul()(buffer1, m, buffer2);
      ops::Add()(buffer2, residual, buffer1);

      _final_layer_norm(buffer1, output);
    }

    AdapterLayer::AdapterLayer(const models::Model& model,
                               const std::string& scope,
                               const bool pre_norm,
                               const ops::ActivationType activation_type,
                               const bool use_flash_attention)
      : _residual_layer_norm(model, scope + "/adpt_residual_layer_norm")
      , _transpose({0, 2, 1})
      , _residual_conv(model, scope + "/adpt_residual_conv", /*stride=*/2, /*padding=*/1)
      , _attn_layer_norm(model, scope + "/adpt_attn_layer_norm")
      , _attn_conv(model, scope + "/adpt_attn_conv", /*stride=*/2, /*padding=*/1)
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 16))
      , _self_attention(!use_flash_attention ? std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/adpt_attn_layer",
                        _num_heads,
                        /*self_attention=*/true,
                        pre_norm)) : std::unique_ptr<AttentionLayer>(new FlashMultiHeadAttention(model,
                        scope + "/adpt_attn_layer",
                        _num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _ffn_layer_norm(model, scope + "/adpt_ffn_layer_norm")
      , _ffn(model, scope + "/adpt_ffn", pre_norm, activation_type) {
      }

    void AdapterLayer::operator()(const StorageView& input, StorageView& output) const{
      PROFILE("AdapterLayer");

      StorageView buffer1(input.dtype(), input.device());
      StorageView buffer2(input.dtype(), input.device());
      StorageView buffer3(input.dtype(), input.device());
      StorageView residual(input.dtype(), input.device());
      std::vector<StorageView*> out{&buffer2, &buffer3};

      _residual_layer_norm(input, buffer1);
      _transpose(buffer1, buffer2);
      _residual_conv(buffer2, buffer1);
      ops::Split(1, {buffer1.dim(1)/2, buffer1.dim(1)/2})(buffer1, out);
      _sigmoid(buffer3, buffer3);
      ops::Mul()(buffer2, buffer3, buffer1);

      _transpose(buffer1, residual);
      _attn_layer_norm(input, buffer1);
      _transpose(buffer1, buffer2);
      _attn_conv(buffer2, buffer1);
      ops::Split(1, {buffer1.dim(1)/2, buffer1.dim(1)/2})(buffer1, out);
      _sigmoid(buffer3, buffer3);
      ops::Mul()(buffer2, buffer3, buffer1);

      _transpose(buffer1, buffer2);
      (*_self_attention)(buffer2,
                         buffer2,
                         nullptr,
                         buffer1,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         true,
                         nullptr);
      ops::Add()(buffer1, residual, buffer2);

      residual.copy_from(buffer2);
      _ffn_layer_norm(buffer2, buffer1);
      _ffn(buffer1, buffer2);
      ops::Add()(buffer1, residual, output);
    }

    Wav2Vec2BertEncoder::Wav2Vec2BertEncoder(const models::Model& model, const std::string& scope)
      : _return_logits(model.get_variable_if_exists(scope + "/lm_head/weight"))
      , _fp_layer_norm(model, scope + "/fp_layer_norm")
      , _fp_projection(model, scope + "/fp_projection", nullptr, true)
      , _encoder_layers(build_layers_list<const EncoderLayer>(model,
                                                              scope + "/encoder_layers",
                                                              /*pre_norm=*/true,
                                                              ops::ActivationType::Swish,
                                                              /*use_flash_attention=*/false))
      , _adapt_layers(build_layers_list<const AdapterLayer>(model,
                                                            scope + "/adapter_layers",
                                                            /*pre_norm=*/true,
                                                            ops::ActivationType::ReLU,
                                                            /*use_flash_attention=*/false)) {
      if (_return_logits) {
        _lm_head.emplace(model, scope + "/lm_head", nullptr, true);
      }
    }

    void Wav2Vec2BertEncoder::operator()(const StorageView& features, StorageView& output) {
      PROFILE("Wav2Vec2BertEncoder");

      // SAD in front-end handles the input length
      if (features.rank() != 3)
        throw std::invalid_argument("Expected input features to have 3 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");

      StorageView buffer1(features.dtype(), features.device());
      StorageView buffer2(features.dtype(), features.device());
      _fp_layer_norm(features, buffer1);
      _fp_projection(buffer1, buffer2);

      for (const auto& layer : _encoder_layers) {
        (*layer)(buffer2, buffer1);
        buffer2 = std::move(buffer1);
      }

      for (const auto& layer : _adapt_layers) {
        (*layer)(buffer2, buffer1);
        buffer2 = std::move(buffer1);
      }

      if (_return_logits) {
        (*_lm_head)(buffer2, output);
      }
      else {
        output = std::move(buffer2);
      }
    }

  }
}
