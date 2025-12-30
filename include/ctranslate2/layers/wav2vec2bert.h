#pragma once

#include <optional>
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/flash_attention.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    class EncoderLayer : public Layer {
    public:
      EncoderLayer(const models::Model& model,
                   const std::string& scope,
                   const bool pre_norm = true,
                   const ops::ActivationType activation_type = ops::ActivationType::ReLU,
                   const bool use_flash_attention = false);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _final_layer_norm.output_type();
      }

      dim_t output_size() const override {
        return _final_layer_norm.output_size();
      }

      const AttentionLayer& get_self_attention() const {
        return *_self_attention;
      }

    private:
      const dim_t _num_heads;
      const LayerNorm _ffn1_layer_norm;
      const FeedForwardNetwork _ff1;
      const LayerNorm _self_attn_layer_norm;
      std::unique_ptr<AttentionLayer> _self_attention;
      const ops::Transpose _transpose;
      const LayerNorm _layer_norm;
      const Conv1D _pconv1;
      const ops::Sigmoid _sigmoid;
      const Conv1D _dconv;
      const LayerNorm _dlayer_norm;
      const ops::Swish _swish;
      const Conv1D _pconv2;
      const LayerNorm _ffn2_layer_norm;
      const FeedForwardNetwork _ff2;
      const LayerNorm _final_layer_norm;
    };

    class AdapterLayer : public Layer {
    public:
      AdapterLayer(const models::Model& model,
                   const std::string& scope,
                   const bool pre_norm = true,
                   const ops::ActivationType activation_type = ops::ActivationType::ReLU,
                   const bool use_flash_attention = false);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _ffn.output_type();
      }

      dim_t output_size() const override {
        return _ffn.output_size();
      }

      const AttentionLayer& get_self_attention() const {
        return *_self_attention;
      }

    private:
      const dim_t _num_heads;
      const LayerNorm _residual_layer_norm;
      const ops::Transpose _transpose;
      const Conv1D _residual_conv;
      const ops::Sigmoid _sigmoid;
      const LayerNorm _attn_layer_norm;
      const Conv1D _attn_conv;
      std::unique_ptr<AttentionLayer> _self_attention;
      const LayerNorm _ffn_layer_norm;
      const FeedForwardNetwork _ffn;
    };

    class Wav2Vec2BertEncoder : public Layer {
    public:
      Wav2Vec2BertEncoder(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features, StorageView& output);

      DataType output_type() const override {
        if (_lm_head) {
          return (*_lm_head).output_type();
        }
        else {
          return DataType::FLOAT32;
        }
      }

      dim_t output_size() const override {
        if (_lm_head) {
          return (*_lm_head).output_size();
        }
        else {
          return 1024;
        }
      }

      dim_t input_size() const {
        return 1024;
      }

      bool is_encoded(const StorageView& features) const {
        // Input features shape: [batch_size, input_size, input_time]
        // Encoder output shape: [batch_size, input_time // 2, output_size]
        //
        // input_time is variable so we check that dimension 1 is different than its original value.

        return (features.rank() == 3
                && features.dim(2) == output_size()
                && features.dim(1) != input_size());
      }

    private:
      const StorageView* _return_logits;
      const LayerNorm _fp_layer_norm;
      const Dense _fp_projection;
      const std::vector<std::unique_ptr<const EncoderLayer>> _encoder_layers;
      const std::vector<std::unique_ptr<const AdapterLayer>> _adapt_layers;
      std::optional<Dense> _lm_head;
    };

  }
}
