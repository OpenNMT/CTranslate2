#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    class MoonshinePreprocessor : public Layer {
    public:
      MoonshinePreprocessor(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features, StorageView& output);

      DataType output_type() const override {
        return _conv3.output_type();
      }

      dim_t output_size() const override {
        return _conv3.output_size();
      }

      dim_t input_size() const {
        return _conv1.input_size();
      }
    private:
      const Conv1D _conv1;
      const ops::Tanh _tanh;
      const LayerNorm _norm;
      const Conv1D _conv2;
      const ops::GELU _gelu1;
      const Conv1D _conv3;
      const ops::GELU _gelu2;
      const ops::Transpose _transpose;
    };


    class MoonshineEncoder : public Layer {
    public:
      MoonshineEncoder(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features, StorageView& output);

      DataType output_type() const override {
        return _output_norm.output_type();
      }

      dim_t output_size() const override {
        return _output_norm.output_size();
      }

      bool is_encoded(const StorageView& features) const {
        // Input features shape: [batch_size, input_size, input_time]
        // Encoder output shape: [batch_size, input_time // 2, output_size]
        //
        // input_time is variable so we check that dimension 1 is different than its original value.

        return (features.rank() == 3
                && features.dim(2) == output_size()
                && features.dim(1) != 1);
      }

    private:
      const dim_t _num_heads;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const LayerNorm _output_norm;
    };

    class MoonshineDecoder : public TransformerDecoder {
    public:
      using TransformerDecoder::TransformerDecoder;

      bool return_normalized_attention() const override {
        return false;
      }
    };
  }
}
