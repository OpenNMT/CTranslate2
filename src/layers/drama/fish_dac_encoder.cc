#include "layers/drama/fish_dac_encoder.h"
#include "ctranslate2/ops/snake.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {

    FishDACReferenceEncoder::FishDACReferenceEncoder(const models::Model& model)
      : _model(model)
      , _encoder_transformer(model, "codec/encoder/block/4/block/5", {16, 4, 512})
      , _pre_transformer(model, "codec/quantizer/pre_module")
      , _downsample(model, "codec/quantizer/downsample", 2, true) {
    }

    const StorageView& FishDACReferenceEncoder::parameter(const std::string& name) const {
      const auto& value = _model.get_variable("codec/" + name);
      if (value.device() != Device::CPU || value.dtype() != DataType::FLOAT32)
        throw std::invalid_argument("FishDAC reference weights must be CPU float32");
      for (dim_t i = 0; i < value.size(); ++i)
        if (!std::isfinite(value.data<float>()[i]))
          throw std::invalid_argument("FishDAC reference weights must be finite");
      return value;
    }

    void FishDACReferenceEncoder::convolution(const StorageView& input, const std::string& name,
                                               StorageView& output, dim_t stride,
                                               dim_t dilation) const {
      const auto& weight = parameter(name + "/weight");
      const auto& bias = parameter(name + "/bias");
      if (weight.rank() != 3 || weight.dim(0) <= 0 || weight.dim(1) != input.dim(1)
          || bias.shape() != Shape{weight.dim(0)} || weight.dim(2) <= 0)
        throw std::invalid_argument("FishDAC reference convolution shape mismatch: " + name);
      const dim_t left = dilation * (weight.dim(2) - 1) + 1 - stride;
      const dim_t right = (stride - input.dim(2) % stride) % stride;
      if (left < 0)
        throw std::invalid_argument("FishDAC reference convolution kernel smaller than stride");
      StorageView padded({1, input.dim(1), input.dim(2) + left + right}, 0.f);
      for (dim_t c = 0; c < input.dim(1); ++c)
        std::copy_n(input.data<float>() + c * input.dim(2), input.dim(2),
                    padded.data<float>() + c * padded.dim(2) + left);
      ops::Conv1D(stride, 0, dilation)(padded, weight, bias, output);
    }

    void FishDACReferenceEncoder::snake(const StorageView& input, const std::string& name,
                                        StorageView& output) const {
      const auto& alpha = parameter(name + "/alpha");
      if (alpha.shape() != Shape{1, input.dim(1), 1})
        throw std::invalid_argument("FishDAC reference Snake shape mismatch");
      ops::Snake()(input, alpha, output);
    }

    StorageView FishDACReferenceEncoder::operator()(const StorageView& waveform) const {
      if (waveform.device() != Device::CPU || waveform.dtype() != DataType::FLOAT32
          || waveform.rank() != 1 || waveform.size() <= 0
          || waveform.size() > std::numeric_limits<dim_t>::max() - 2048)
        throw std::invalid_argument("reference waveform must be nonempty CPU float32 [samples]");
      for (dim_t i = 0; i < waveform.size(); ++i)
        if (!std::isfinite(waveform.data<float>()[i]))
          throw std::invalid_argument("reference waveform must be finite");
      const dim_t length = ((waveform.size() + 2047) / 2048) * 2048;
      StorageView padded({1, 1, length}, 0.f);
      std::copy_n(waveform.data<float>(), waveform.size(), padded.data<float>());
      StorageView current;
      convolution(padded, "encoder/block/0/conv", current);
      const dim_t strides[] = {2, 4, 8, 8};
      for (int stage = 0; stage < 4; ++stage) {
        const auto prefix = "encoder/block/" + std::to_string(stage + 1) + "/block/";
        dim_t dilation = 1;
        for (int unit = 0; unit < 3; ++unit) {
          const auto r = prefix + std::to_string(unit) + "/block/";
          StorageView a, b, c, branch;
          snake(current, r + "0", a);
          convolution(a, r + "1/conv", b, 1, dilation);
          snake(b, r + "2", c);
          convolution(c, r + "3/conv", branch);
          ops::Add()(current, branch, current);
          dilation *= 3;
        }
        StorageView activated, down;
        snake(current, prefix + "3", activated);
        convolution(activated, prefix + "4/conv", down, strides[stage]);
        if (stage == 3)
          _encoder_transformer(down, current);
        else
          current = std::move(down);
      }
      StorageView activated, latent;
      snake(current, "encoder/block/5", activated);
      convolution(activated, "encoder/block/6/conv", latent);
      StorageView downsampled;
      _downsample(latent, downsampled);
      StorageView residual;
      _pre_transformer(downsampled, residual);
      const dim_t time = residual.dim(2), channels = residual.dim(1);
      StorageView codes({time, 10}, DataType::INT32);
      for (dim_t book = 0; book < 10; ++book) {
        const std::string prefix = "quantizer/" +
            (book == 0 ? std::string("semantic_quantizer/quantizers/0")
                       : "quantizer/quantizers/" + std::to_string(book - 1));
        StorageView projected;
        convolution(residual, prefix + "/in_proj", projected);
        for (dim_t i = 0; i < projected.size(); ++i)
          if (!std::isfinite(projected.data<float>()[i]))
            throw std::runtime_error("FishDAC reference quantizer produced nonfinite values");
        const auto& embedding = parameter(prefix + "/codebook/weight");
        const dim_t width = projected.dim(1);
        if (embedding.rank() != 2 || embedding.dim(1) != width || embedding.dim(0) <= 0)
          throw std::invalid_argument("FishDAC reference codebook shape mismatch");
        StorageView vectors;
        projected.reshape({width, time});
        ops::Transpose()(projected, vectors);
        StorageView normalized(embedding);
        // Match DAC's normalized squared Euclidean distance, including zero rows.
        for (auto* matrix : {&vectors, &normalized}) {
          for (dim_t row = 0; row < matrix->dim(0); ++row) {
            auto* values = matrix->data<float>() + row * width;
            float norm = 0;
            for (dim_t d = 0; d < width; ++d)
              norm += values[d] * values[d];
            norm = std::max(std::sqrt(norm), 1e-12f);
            for (dim_t d = 0; d < width; ++d)
              values[d] /= norm;
          }
        }
        StorageView scores;
        ops::MatMul(false, true, 2.f)(vectors, normalized, scores);
        for (dim_t k = 0; k < embedding.dim(0); ++k) {
          float norm = 0;
          for (dim_t d = 0; d < width; ++d)
            norm += std::pow(normalized.data<float>()[k * width + d], 2);
          for (dim_t t = 0; t < time; ++t)
            scores.data<float>()[t * embedding.dim(0) + k] -= norm;
        }
        StorageView values, indices(DataType::INT32), selected, quantized;
        ops::TopK(1)(scores, values, indices);
        indices.reshape({time});
        ops::Gather()(embedding, indices, selected);
        ops::Transpose()(selected, quantized);
        quantized.reshape({1, width, time});
        for (dim_t t = 0; t < time; ++t)
          codes.data<int32_t>()[t * 10 + book] = indices.data<int32_t>()[t];
        StorageView decoded;
        convolution(quantized, prefix + "/out_proj", decoded);
        if (decoded.shape() != Shape{1, channels, time})
          throw std::invalid_argument("FishDAC reference quantizer projection shape mismatch");
        ops::Sub()(residual, decoded, residual);
      }
      return codes;
    }

  }
}
