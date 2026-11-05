#include "layers/drama/fish_dac_upsample.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ctranslate2/ops/conv_transpose1d.h"

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& source, const StorageView& output) {
        const auto size = source.size() * source.item_size();
        const auto capacity =
            std::max(output.reserved_memory(), output.size() * output.item_size());
        if (!source.buffer() || !output.buffer() || !size || !capacity)
          return false;
        const auto a = reinterpret_cast<uintptr_t>(source.buffer());
        const auto b = reinterpret_cast<uintptr_t>(output.buffer());
        return a <= b ? b - a < static_cast<uintptr_t>(size)
                      : a - b < static_cast<uintptr_t>(capacity);
      }

      void require_finite(const StorageView& tensor) {
        for (dim_t i = 0; i < tensor.size(); ++i) {
          if (!std::isfinite(tensor.data<float>()[i]))
            throw std::invalid_argument("FishDAC latent upsampler requires finite values");
        }
      }

    }

    FishDACLatentResampler::Block::Block(const models::Model& model, const std::string& scope)
      : pw1(model, scope + "/1/pwconv1")
      , pw2(model, scope + "/1/pwconv2") {
    }

    FishDACLatentResampler::FishDACLatentResampler(const models::Model& model,
                                               const std::string& scope, dim_t block_count,
                                               bool downsample)
      : _downsample(downsample) {
      if (model.device() != Device::CPU || block_count <= 0)
        throw std::invalid_argument(
            "FishDAC latent resampler requires CPU and positive block count");
      auto parameter = [&](const std::string& name) {
        const auto& value = model.get_variable(name);
        if (value.device() != Device::CPU || value.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("FishDAC latent weights must be CPU float32");
        require_finite(value);
        _parameters.push_back(&value);
        return &value;
      };
      dim_t previous_channels = 0;
      for (dim_t i = 0; i < block_count; ++i) {
        const auto prefix = scope + "/" + std::to_string(i);
        auto block = std::make_unique<Block>(model, prefix);
        if (block->pw1.output_type() != DataType::FLOAT32
            || block->pw2.output_type() != DataType::FLOAT32)
          throw std::invalid_argument("FishDAC latent Dense output must be float32");
        block->transpose_weight = parameter(prefix + "/0/conv/weight");
        block->transpose_bias = parameter(prefix + "/0/conv/bias");
        const auto& tw = *block->transpose_weight;
        if (tw.rank() != 3 || tw.dim(0) <= 0 || tw.dim(1) <= 0 || tw.dim(2) != 2
            || (i && tw.dim(_downsample ? 1 : 0) != previous_channels))
          throw std::invalid_argument("FishDAC latent resampling weight/channel mismatch");
        if (!i)
          _input_channels = tw.dim(_downsample ? 1 : 0);
        block->channels = tw.dim(_downsample ? 0 : 1);
        previous_channels = block->channels;
        const auto channels = block->channels;
        if (block->transpose_bias->shape() != Shape{channels})
          throw std::invalid_argument("FishDAC latent resampling bias mismatch");
        const auto cp = prefix + "/1";
        block->depthwise_weight = parameter(cp + "/dwconv/conv/weight");
        block->depthwise_bias = parameter(cp + "/dwconv/conv/bias");
        block->norm_weight = parameter(cp + "/norm/weight");
        block->norm_bias = parameter(cp + "/norm/bias");
        block->gamma = parameter(cp + "/gamma");
        if (block->depthwise_weight->shape() != Shape{channels, 1, 7}
            || block->depthwise_bias->shape() != Shape{channels}
            || block->norm_weight->shape() != Shape{channels}
            || block->norm_bias->shape() != Shape{channels}
            || block->gamma->shape() != Shape{channels})
          throw std::invalid_argument("FishDAC latent ConvNeXt parameter dimensions mismatch");
        const auto& pw1 = *parameter(cp + "/pwconv1/weight");
        const auto& pb1 = *parameter(cp + "/pwconv1/bias");
        const auto& pw2 = *parameter(cp + "/pwconv2/weight");
        const auto& pb2 = *parameter(cp + "/pwconv2/bias");
        if (model.get_variable_if_exists(cp + "/pwconv1/weight_packed")
            || model.get_variable_if_exists(cp + "/pwconv2/weight_packed") || pw1.rank() != 2
            || pw1.dim(0) <= 0 || pw1.dim(1) != channels || pb1.shape() != Shape{pw1.dim(0)}
            || pw2.shape() != Shape{channels, pw1.dim(0)} || pb2.shape() != Shape{channels})
          throw std::invalid_argument("FishDAC latent pointwise Dense dimensions mismatch");
        _blocks.emplace_back(std::move(block));
      }
    }

    void FishDACLatentResampler::forward_block(const Block& block, const StorageView& input,
                                               StorageView& output) const {
      StorageView upsampled;
      if (_downsample)
        ops::Conv1D(2)(input, *block.transpose_weight, *block.transpose_bias, upsampled);
      else
        ops::ConvTranspose1D(2)(input, *block.transpose_weight, *block.transpose_bias, upsampled);
      const auto time = upsampled.dim(2);
      const auto channels = block.channels;
      StorageView padded({1, channels, time + 6}, 0.f);
      for (dim_t c = 0; c < channels; ++c)
        std::copy(upsampled.data<float>() + c * time, upsampled.data<float>() + (c + 1) * time,
                  padded.data<float>() + c * (time + 6) + 6);
      StorageView depthwise;
      ops::Conv1D(1, 0, 1, channels)(padded, *block.depthwise_weight, *block.depthwise_bias,
                                     depthwise);
      StorageView rows({time, channels});
      for (dim_t t = 0; t < time; ++t)
        for (dim_t c = 0; c < channels; ++c)
          rows.data<float>()[t * channels + c] = depthwise.data<float>()[c * time + t];
      StorageView normalized, wide, activated, projected;
      ops::LayerNorm(-1, 1e-6f)(*block.norm_bias, *block.norm_weight, rows, normalized);
      block.pw1(normalized, wide);
      ops::get_activation_op(ops::ActivationType::GELU)(wide, activated);
      block.pw2(activated, projected);
      for (dim_t c = 0; c < channels; ++c)
        for (dim_t t = 0; t < time; ++t)
          upsampled.data<float>()[c * time + t] +=
              block.gamma->data<float>()[c] * projected.data<float>()[t * channels + c];
      require_finite(upsampled);
      output = std::move(upsampled);
    }

    void FishDACLatentResampler::operator()(const StorageView& input, StorageView& output) const {
      if (input.device() != Device::CPU || output.device() != Device::CPU
          || input.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32
          || input.rank() != 3 || input.dim(0) != 1 || input.dim(1) != _input_channels
          || input.dim(2) < 0)
        throw std::invalid_argument("FishDAC latent input must be CPU float32 [1,C,T]");
      if (&input == &output || overlaps(input, output))
        throw std::invalid_argument("FishDAC latent output must not alias input");
      for (const auto* parameter : _parameters) {
        if (&output == parameter || overlaps(*parameter, output))
          throw std::invalid_argument("FishDAC latent output must not alias weights");
      }
      require_finite(input);
      dim_t length = input.dim(2);
      for (size_t i = 0; i < _blocks.size(); ++i) {
        if (length > (std::numeric_limits<dim_t>::max() - 6) / 2)
          throw std::invalid_argument("FishDAC latent sequence length overflows");
        if (_downsample && length % 2)
          throw std::invalid_argument("FishDAC latent downsampling requires even frame counts");
        length = _downsample ? length / 2 : length * 2;
      }
      if (!length) {
        output.resize({1, _blocks.back()->channels, 0});
        return;
      }
      StorageView current(input);
      for (const auto& block : _blocks)
        forward_block(*block, current, current);
      output.copy_from(current);
    }

  }
}
