#include "layers/drama/fish_dac.h"
#include "ctranslate2/ops/snake.h"
#include "ctranslate2/ops/conv_transpose1d.h"
#include "ops/drama/fish_dac_gpu.h"
#include "env.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& source, const StorageView& output) {
        if (!source.buffer() || !output.buffer())
          return false;
        const auto a = reinterpret_cast<uintptr_t>(source.buffer());
        const auto b = reinterpret_cast<uintptr_t>(output.buffer());
        const auto as = static_cast<uintptr_t>(source.size() * source.item_size());
        const auto bs = static_cast<uintptr_t>(
            std::max(output.reserved_memory(), output.size() * output.item_size()));
        return a <= b ? b - a < as : a - b < bs;
      }

    }

    FishDACWaveformDecoder::FishDACWaveformDecoder(const models::Model& model,
                                                   const std::string& scope,
                                                   std::vector<dim_t> strides, const DramaExecutionOptions& options) {
      if (model.device() != Device::CPU || strides.empty())
        throw std::invalid_argument("FishDAC waveform decoder requires CPU and nonempty strides");
      auto parameter = [&](std::string name) {
        std::replace(name.begin(), name.end(), '.', '/');
        const auto& value = model.get_variable(name);
        if (value.dtype() != DataType::FLOAT32 || value.device() != Device::CPU)
          throw std::invalid_argument("FishDAC waveform weights require CPU float32");
        for (dim_t i = 0; i < value.size(); ++i)
          if (!std::isfinite(value.data<float>()[i]))
            throw std::invalid_argument("FishDAC waveform weights must be finite");
        _parameters.push_back(&value);
        return &value;
      };
      auto convolution = [&](const std::string& name, dim_t input_channels, bool transpose) {
        Convolution conv{parameter(name + ".weight"), parameter(name + ".bias")};
        const auto& w = *conv.weight;
        if (w.rank() != 3 || w.dim(0) <= 0 || w.dim(1) <= 0 || w.dim(2) <= 0
            || (transpose ? w.dim(0) : w.dim(1)) != input_channels
            || conv.bias->shape() != Shape{transpose ? w.dim(1) : w.dim(0)})
          throw std::invalid_argument("FishDAC convolution shape mismatch: " + name);
        return conv;
      };
      auto alpha = [&](const std::string& name, dim_t channels) {
        const auto* a = parameter(name + ".alpha");
        if (a->shape() != Shape{1, channels, 1} && a->shape() != Shape{channels})
          throw std::invalid_argument("FishDAC Snake shape mismatch: " + name);
        return a;
      };
      const auto& first = *parameter(scope + ".model.0.conv.weight");
      if (first.rank() != 3)
        throw std::invalid_argument("FishDAC initial weight must be rank 3");
      _initial = convolution(scope + ".model.0.conv", first.dim(1), false);
      dim_t channels = first.dim(0);
      for (size_t i = 0; i < strides.size(); ++i) {
        const auto prefix = scope + ".model." + std::to_string(i + 1) + ".block.";
        if (strides[i] <= 0 || strides[i] > std::numeric_limits<dim_t>::max() / 2)
          throw std::invalid_argument("FishDAC stride must be positive and bounded");
        Upsample block;
        block.stride = strides[i];
        block.alpha = alpha(prefix + "0", channels);
        block.transpose = convolution(prefix + "1.conv", channels, true);
        if (block.transpose.weight->dim(2) != 2 * block.stride)
          throw std::invalid_argument("FishDAC transpose kernel must be twice its stride");
        channels = block.transpose.weight->dim(1);
        for (int j = 0; j < 3; ++j) {
          const auto r = prefix + std::to_string(j + 2) + ".block.";
          Residual residual{alpha(r + "0", channels), convolution(r + "1.conv", channels, false),
                            alpha(r + "2", channels), convolution(r + "3.conv", channels, false)};
          if (residual.conv0.weight->dim(0) != channels || residual.conv0.weight->dim(2) != 7
              || residual.conv1.weight->dim(0) != channels || residual.conv1.weight->dim(2) != 1)
            throw std::invalid_argument("FishDAC residual kernel/channel mismatch");
          block.residuals.push_back(residual);
        }
        _blocks.push_back(std::move(block));
      }
      const auto end = scope + ".model.";
      _final_alpha = alpha(end + std::to_string(strides.size() + 1), channels);
      _final = convolution(end + std::to_string(strides.size() + 2) + ".conv", channels, false);
      if (_initial.weight->dim(2) != 7 || _final.weight->dim(2) != 7 || _final.weight->dim(0) != 1)
        throw std::invalid_argument(
            "FishDAC boundary convolutions require kernel7 and mono output");
      if (options.device == Device::CUDA && options.waveform) {
        _gpu_device_index = options.device_index;
        const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
        for (const auto* parameter : _parameters)
          _gpu_parameters.emplace(
              parameter, upload_drama_weight(*parameter, options, _shared_gpu_parameters));
      }
    }

    void FishDACWaveformDecoder::causal(const StorageView& input, const Convolution& conv,
                                        dim_t dilation, StorageView& output) {
      const dim_t left = dilation * (conv.weight->dim(2) - 1);
      if (input.dim(2) > std::numeric_limits<dim_t>::max() - left)
        throw std::invalid_argument("FishDAC causal length overflow");
      StorageView padded({1, input.dim(1), input.dim(2) + left}, 0.f);
      for (dim_t c = 0; c < input.dim(1); ++c)
        std::copy_n(input.data<float>() + c * input.dim(2), input.dim(2),
                    padded.data<float>() + c * padded.dim(2) + left);
      ops::Conv1D(1, 0, dilation)(padded, *conv.weight, *conv.bias, output);
    }

    void FishDACWaveformDecoder::forward_gpu(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_CUDA
      const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
      auto parameter = [&](const StorageView* p) -> const StorageView& {
        return _gpu_parameters.at(p);
      };
      auto convolution = [&](const StorageView& x, const Convolution& conv, dim_t dilation,
                             StorageView& y) {
        ops::fish_causal_gpu(x, parameter(conv.weight), parameter(conv.bias), dilation, y);
      };
      auto gpu_input = input.to(Device::CUDA);
      StorageView current(DataType::FLOAT32, Device::CUDA);
      convolution(gpu_input, _initial, 1, current);
      for (const auto& block : _blocks) {
        StorageView activated(DataType::FLOAT32, Device::CUDA),
            transposed(DataType::FLOAT32, Device::CUDA);
        ops::Snake()(current, parameter(block.alpha), activated);
        ops::fish_transpose_gpu(activated, parameter(block.transpose.weight),
                                parameter(block.transpose.bias), block.stride, transposed);
        current = std::move(transposed);
        dim_t dilation = 1;
        for (const auto& residual : block.residuals) {
          StorageView a(DataType::FLOAT32, Device::CUDA), b(DataType::FLOAT32, Device::CUDA);
          StorageView c(DataType::FLOAT32, Device::CUDA), branch(DataType::FLOAT32, Device::CUDA);
          ops::Snake()(current, parameter(residual.alpha0), a);
          convolution(a, residual.conv0, dilation, b);
          ops::Snake()(b, parameter(residual.alpha1), c);
          convolution(c, residual.conv1, 1, branch);
          ops::Add()(current, branch, current);
          dilation *= 3;
        }
      }
      StorageView activated(DataType::FLOAT32, Device::CUDA),
          result(DataType::FLOAT32, Device::CUDA);
      ops::Snake()(current, parameter(_final_alpha), activated);
      convolution(activated, _final, 1, result);
      ops::fish_tanh_gpu(result);
      output.copy_from(result);
      synchronize_stream(Device::CUDA);
      for (dim_t i = 0; i < output.size(); ++i)
        if (!std::isfinite(output.data<float>()[i]))
          throw std::runtime_error("GPU FishDAC produced nonfinite waveform");
#else
      throw std::runtime_error("GPU FishDAC requires GPU support");
#endif
    }

    void FishDACWaveformDecoder::validate_input(const StorageView& input,
                                                const StorageView& output) const {
      if (input.device() != Device::CPU || input.dtype() != DataType::FLOAT32
          || output.device() != Device::CPU || output.dtype() != DataType::FLOAT32
          || input.rank() != 3 || input.dim(0) != 1 || input.dim(1) != _initial.weight->dim(1)
          || input.dim(2) <= 0 || &input == &output || overlaps(input, output))
        throw std::invalid_argument("FishDAC waveform input requires nonempty CPUfloat32 [1,C,T]");
      for (const auto* p : _parameters)
        if (overlaps(*p, output))
          throw std::invalid_argument("FishDAC output aliases weights");
      for (dim_t i = 0; i < input.size(); ++i)
        if (!std::isfinite(input.data<float>()[i]))
          throw std::invalid_argument("FishDAC input must be finite");
    }

    void FishDACWaveformDecoder::forward_cpu(const StorageView& input, StorageView& output) const {
      StorageView current;
      causal(input, _initial, 1, current);
      for (const auto& block : _blocks) {
        StorageView activated, transposed;
        ops::Snake()(current, *block.alpha, activated);
        ops::ConvTranspose1D(block.stride)(activated, *block.transpose.weight,
                                           *block.transpose.bias, transposed);
        // Fish causal transpose discards k-stride samples exclusively at right.
        const dim_t length = transposed.dim(2) - block.stride;
        StorageView cropped({1, transposed.dim(1), length}, DataType::FLOAT32);
        for (dim_t c = 0; c < cropped.dim(1); ++c)
          std::copy_n(transposed.data<float>() + c * transposed.dim(2), length,
                      cropped.data<float>() + c * length);
        current = std::move(cropped);
        dim_t dilation = 1;
        for (const auto& residual : block.residuals) {
          StorageView a, b, c, branch;
          ops::Snake()(current, *residual.alpha0, a);
          causal(a, residual.conv0, dilation, b);
          ops::Snake()(b, *residual.alpha1, c);
          causal(c, residual.conv1, 1, branch);
          for (dim_t i = 0; i < current.size(); ++i)
            current.data<float>()[i] += branch.data<float>()[i];
          dilation *= 3;
        }
      }
      StorageView activated, result;
      ops::Snake()(current, *_final_alpha, activated);
      causal(activated, _final, 1, result);
      for (dim_t i = 0; i < result.size(); ++i) {
        const float value = result.data<float>()[i];
        if (!std::isfinite(value))
          throw std::runtime_error("FishDAC produced nonfinite waveform");
        result.data<float>()[i] = std::tanh(value);
      }
      output = std::move(result);
    }

    void FishDACWaveformDecoder::operator()(const StorageView& input, StorageView& output) const {
      validate_input(input, output);
      if (_gpu_device_index >= 0) {
        forward_gpu(input, output);
        return;
      }
      forward_cpu(input, output);
    }

  }
}
