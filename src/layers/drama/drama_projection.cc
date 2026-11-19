#include "layers/drama/drama_projection.h"

#include "ctranslate2/ops/dequantize_fp8.h"

namespace ctranslate2 {
  namespace layers {

    DramaProjection::DramaProjection(const models::Model& model, const std::string& scope,
                                     const DramaExecutionOptions& options)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _fp8_scale_inv(model.get_variable_if_exists(scope + "/weight_scale_inv"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), _bias ? _bias->dtype() : DataType::FLOAT32)
      , _partial_qscale(_weight.device(), DataType::FLOAT32)
      , _gpu_device_index(options.device == Device::CUDA ? options.device_index : -1)
      , _gpu_compute_type(options.compute_type)
      , _device_weights(options.device_weights)
      , _device_weight_key(scope + ":" + std::to_string(reinterpret_cast<uintptr_t>(_weight.buffer()))
                           + ":" + std::to_string(_gpu_device_index)
                           + ":" + std::to_string(static_cast<int>(_gpu_compute_type)))
    {
      if (_weight.device() != Device::CPU || model.tensor_parallel()
          || model.get_variable_if_exists(scope + "/weight_packed")
          || model.get_variable_if_exists(scope + "/weight_zero")
          || get_default_float_type(model.effective_compute_type()) != DataType::FLOAT32)
        throw std::invalid_argument("Drama3 projections require unpacked CPU weights and float32 graph state");
      if (_fp8_scale_inv) {
        if (model.config.value("fp8_format", std::string()) != "e4m3fn_block128"
            || _weight.dtype() != DataType::INT8 || _qscale)
          throw std::invalid_argument("Block FP8 projections require explicit E4M3FN metadata");
      } else if (_gpu_device_index < 0) {
        _cpu = std::make_unique<Dense>(model, scope);
      }
    }

    void DramaProjection::prepare_gpu_weight(const StorageView& weight, const StorageView* scale) const {
      const auto types = compute_type_to_data_type(_gpu_compute_type);
      const auto weight_type = types.first, float_type = types.second;
      if (_gpu_weights)
        return;
      auto prepare = [&]() {
        auto prepared = std::make_shared<DramaDeviceWeights>();
        StorageView uploaded;
        if (!_fp8_scale_inv && !is_float_type(weight.dtype())) {
          StorageView decoded;
          ops::Dequantize()(weight, *scale, decoded);
          uploaded = decoded.to(Device::CUDA);
        } else {
          uploaded = weight.to(Device::CUDA);
        }
        if (_fp8_scale_inv) {
          prepared->scale_inv = _fp8_scale_inv->to(Device::CUDA);
          // Prepare half-precision experts once, just like INT8 experts, instead
          // of decoding and synchronizing their FP8 weights on every projection.
          if (weight_type == DataType::INT8 || weight_type == DataType::FLOAT16
              || weight_type == DataType::BFLOAT16) {
            StorageView decoded(DataType::FLOAT32, Device::CUDA);
            ops::DequantizeFP8()(uploaded, prepared->scale_inv, decoded);
            uploaded = std::move(decoded);
            prepared->scale_inv.release();
          }
        }
        if (weight_type == DataType::INT8) {
          StorageView quantized(DataType::INT8, Device::CUDA),
              scales(DataType::FLOAT32, Device::CUDA);
          ops::Quantize(ops::Quantize::ScaleType::PER_LAYER, false)(uploaded, quantized, scales);
          prepared->scale = std::move(scales);
          uploaded = std::move(quantized);
        } else if (prepared->scale_inv.empty() && uploaded.dtype() != float_type) {
          uploaded = uploaded.to(float_type);
        }
        prepared->weight = std::move(uploaded);
        const auto* bias = _partial_bias.empty() ? _bias : &_partial_bias;
        if (bias)
          prepared->bias = bias->to(Device::CUDA).to(float_type);
        // Other replicas execute on different streams. Finish preparation before
        // publishing immutable buffers to them (also covers conversion kernels).
        synchronize_stream(Device::CUDA);
        return prepared;
      };
      _gpu_weights = _device_weights && _share_selected_weights
          ? _device_weights->get(_device_weight_key + _selection_key, prepare) : prepare();
    }

    void DramaProjection::forward_gpu(const StorageView& input, const StorageView& weight,
                            const StorageView* bias, const StorageView* scale,
                            const StorageView* residual, StorageView& output) const {
      const ScopedDeviceSetter device_setter(Device::CUDA, _gpu_device_index);
      if (input.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32)
        throw std::invalid_argument("Drama3 GPU projections require float32 input/output");
      if ((input.device() == Device::CUDA && input.device_index() != _gpu_device_index)
          || (output.device() == Device::CUDA && output.device_index() != _gpu_device_index))
        throw std::invalid_argument("Drama3 GPU projection tensors must use the configured device");
      const auto types = compute_type_to_data_type(_gpu_compute_type);
      const auto weight_type = types.first, float_type = types.second;
      prepare_gpu_weight(weight, scale);
      StorageView converted_input;
      const StorageView* gpu_input = &input;
      if (input.device() != Device::CUDA) {
        converted_input = input.to(Device::CUDA);
        gpu_input = &converted_input;
      }
      if (gpu_input->dtype() != float_type) {
        converted_input = gpu_input->to(float_type);
        gpu_input = &converted_input;
      }
      const auto& gpu_bias = _gpu_weights->bias;
      StorageView gpu_residual;
      if (residual)
        gpu_residual = residual->to(Device::CUDA).to(float_type);
      StorageView result(float_type, Device::CUDA);
      const ops::Gemm gemm(1, 0, false, true, false, false,
                           weight_type == DataType::INT8 ? nullptr : nullptr);
      if (weight_type == DataType::INT8) {
        StorageView quantized(DataType::INT8, Device::CUDA),
            scales(DataType::FLOAT32, Device::CUDA);
        StorageView accumulator(DataType::INT32, Device::CUDA);
        ops::Quantize(ops::Quantize::ScaleType::PER_LAYER, false)(*gpu_input, quantized, scales);
        gemm(quantized, _gpu_weights->weight, accumulator);
        _dequantize_op(accumulator, scales, _gpu_weights->scale, false, true, result,
                       bias ? &gpu_bias : nullptr);
        if (residual)
          ops::Add()(gpu_residual, result, result);
      } else {
        StorageView decoded(DataType::FLOAT32, Device::CUDA);
        const StorageView* gpu_weight = &_gpu_weights->weight;
        if (!_gpu_weights->scale_inv.empty()) {
          ops::DequantizeFP8()(_gpu_weights->weight, _gpu_weights->scale_inv, decoded);
          if (decoded.dtype() != float_type)
            decoded = decoded.to(float_type);
          gpu_weight = &decoded;
        }
        gemm(*gpu_input, *gpu_weight, result, nullptr, bias ? &gpu_bias : nullptr,
             residual ? &gpu_residual : nullptr);
      }
      if (result.dtype() != DataType::FLOAT32)
        result = result.to(DataType::FLOAT32);
      if (output.device() == Device::CUDA) {
        output = std::move(result);
      } else {
        output.copy_from(result);
        synchronize_stream(Device::CUDA); // The following graph operations read host memory.
      }
    }

    DataType DramaProjection::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t DramaProjection::output_size() const {
      return _partial_weight.empty() ? _weight.dim(0) : _partial_weight.dim(0);
    }

    void DramaProjection::select_weights(const StorageView* index, const StorageView* extra_bias) {
      release_device_weights();
      _selection_key.clear();
      _share_selected_weights = !extra_bias;
      if (index && _device_weights) {
        const auto host_index = index->to(Device::CPU);
        _selection_key = ":selected:";
        for (const auto dim : index->shape())
          _selection_key += std::to_string(dim) + ":";
        _selection_key.append(static_cast<const char*>(host_index.buffer()),
                              host_index.size() * host_index.item_size());
      }
      if (index && _fp8_scale_inv)
        throw std::invalid_argument("Selecting block FP8 projection weights is unsupported");
      if (_cpu)
        _cpu->select_weights(index, extra_bias);
      if (index) {
        ops::Gather()(_weight, *index, _partial_weight);
        if (_bias) {
          ops::Gather()(*_bias, *index, _partial_bias);
          if (extra_bias)
            ops::Add()(_partial_bias, *extra_bias, _partial_bias);
        } else if (extra_bias) {
          _partial_bias = *extra_bias;
        }
        if (_qscale && !_qscale->is_scalar())
          ops::Gather()(*_qscale, *index, _partial_qscale);
      } else {
        _partial_weight.clear();
        _partial_bias.clear();
        _partial_qscale.clear();
      }
    }

    void DramaProjection::operator()(const StorageView& input, StorageView& output,
                                     const StorageView* residual) const {
      if (_cpu) {
        (*_cpu)(input, output, residual);
        return;
      }
      const auto& weight = _partial_weight.empty() ? _weight : _partial_weight;
      const auto* bias = _partial_bias.empty() ? _bias : &_partial_bias;
      const auto* scale = _partial_qscale.empty() ? _qscale : &_partial_qscale;
      if (_gpu_device_index >= 0) {
        forward_gpu(input, weight, bias, scale, residual, output);
      } else {
        if (input.device() != Device::CPU || input.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("Block FP8 projections require CPU float32 input");
        StorageView decoded;
        ops::DequantizeFP8()(weight, *_fp8_scale_inv, decoded);
        ops::Gemm(1, 0, false, true)(input, decoded, output, nullptr, bias, residual);
      }
    }

    DramaProjection::DeviceWeights DramaProjection::device_weights() const {
      if (_gpu_device_index < 0)
        throw std::invalid_argument("Device weights require GPU execution");
      const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
      prepare_gpu_weight(_partial_weight.empty() ? _weight : _partial_weight,
                         _partial_qscale.empty() ? _qscale : &_partial_qscale);
      return {_gpu_weights->weight, _gpu_weights->scale};
    }

    void DramaProjection::release_device_weights() const {
      _gpu_weights.reset();
    }

    size_t DramaProjection::device_weight_bytes() const {
      return _gpu_weights ? _gpu_weights->bytes() : 0;
    }

    size_t DramaProjection::estimated_device_weight_bytes() const {
      const auto types = compute_type_to_data_type(_gpu_compute_type);
      const auto type = types.first;
      const auto bias_bytes = _bias ? _bias->size() * (types.second == DataType::FLOAT32 ? 4 : 2) : 0;
      if (type == DataType::INT8)
        return _weight.size() + _weight.dim(0) * sizeof(float) + bias_bytes;
      if (type == DataType::FLOAT32 && _fp8_scale_inv)
        return _weight.size() + _fp8_scale_inv->size() * sizeof(float) + bias_bytes;
      return _weight.size() * (type == DataType::FLOAT32 ? 4 : 2) + bias_bytes;
    }

  }
}
