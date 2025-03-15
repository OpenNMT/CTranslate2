#pragma once

#include <optional>
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    class Wav2Vec2LayerNormConvLayer : public Layer {
    public:
      Wav2Vec2LayerNormConvLayer(const models::Model& model,
                                 const std::string& scope,
                                 dim_t stride,
                                 dim_t padding);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _conv.output_type();
      }

      dim_t output_size() const override {
        return _conv.output_size();
      }

    private:
      dim_t _stride;
      dim_t _padding;
      const Conv1D _conv;
      const LayerNorm _output_norm;
      const ops::Transpose _transpose;
      const ops::GELU _gelu;
    };

    class Wav2Vec2PosConvLayer : public Layer {
    public:
      Wav2Vec2PosConvLayer(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _conv.output_type();
      }

      dim_t output_size() const override {
        return _conv.output_size();
      }

    private:
      const Conv1D _conv;
      const ops::Transpose _transpose;
      const ops::GELU _gelu;
    };

    class Wav2Vec2Encoder : public Layer {
    public:
      Wav2Vec2Encoder(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features, StorageView& output);

      DataType output_type() const override {
        if (_lm_head) {
          return (*_lm_head).output_type();
        }
        else {
          return _output_norm.output_type();
        }
      }

      dim_t output_size() const override {
        if (_lm_head) {
          return (*_lm_head).output_size();
        }
        else {
          return _output_norm.output_size();
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

      const StorageView* _upgraded_model;

    private:
      const StorageView* _return_logits;
      std::optional<Wav2Vec2LayerNormConvLayer> _feat_layer0;
      std::optional<std::vector<std::unique_ptr<const Wav2Vec2LayerNormConvLayer>>> _feat_layers;
      std::optional<LayerNorm> _fp_norm;
      std::optional<Dense> _fp_ff;
      std::optional<Wav2Vec2PosConvLayer> _pos_conv_embed;
      const ops::Transpose _transpose;
      const ops::GELU _gelu;
      const dim_t _num_heads;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const LayerNorm _output_norm;
      std::optional<Dense> _lm_head;
    };

  }
}
