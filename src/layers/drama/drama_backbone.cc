#include "layers/drama/drama_backbone.h"
#include "ctranslate2/ops/rms_norm.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        if (!a.buffer() || !b.buffer() || !a.reserved_memory() || !b.reserved_memory())
          return false;
        const auto x = reinterpret_cast<uintptr_t>(a.buffer());
        const auto y = reinterpret_cast<uintptr_t>(b.buffer());
        return x <= y ? y - x < static_cast<uintptr_t>(a.reserved_memory())
                      : x - y < static_cast<uintptr_t>(b.reserved_memory());
      }

    }

    DramaBackbone::DramaBackbone(const models::Model& model, const std::string& scope,
                                 DramaBackboneConfig config, const DramaExecutionOptions& options)
      : _config(std::move(config))
      , _norm(model.get_variable(scope + "/norm/weight"))
      , _lm_head(model, scope + "/lm_head", options)
      , _vocabulary_size(_lm_head.output_size()) {
      if (_config.layers.empty() || _config.max_context <= 0 || !std::isfinite(_config.norm_epsilon)
          || _config.norm_epsilon <= 0 || model.device() != Device::CPU
          || _lm_head.output_type() != DataType::FLOAT32 || _norm.device() != Device::CPU
          || _norm.dtype() != DataType::FLOAT32 || _norm.rank() != 1 || _norm.dim(0) <= 0)
        throw std::invalid_argument("DramaBackbone configuration or final norm is incompatible");
      for (dim_t d = 0; d < _norm.size(); ++d) {
        if (!std::isfinite(_norm.data<float>()[d]))
          throw std::invalid_argument("DramaBackbone norm weights must be finite");
      }
      const auto& head = model.get_variable(scope + "/lm_head/weight");
      if (head.rank() != 2 || head.dim(0) <= 0 || head.dim(1) != output_size()
          || model.get_variable_if_exists(scope + "/lm_head/weight_packed"))
        throw std::invalid_argument("DramaBackbone requires compatible unpacked lm_head weights");
      _head_parameters = {&_norm, &head};
      for (const auto* suffix :
           {"/bias", "/weight_scale", "/weight_compensation", "/weight_zero"}) {
        if (const auto* variable = model.get_variable_if_exists(scope + "/lm_head" + suffix)) {
          if (std::string(suffix) == "/bias" && variable->shape() != Shape{vocabulary_size()})
            throw std::invalid_argument("DramaBackbone lm_head bias dimension is incompatible");
          _head_parameters.push_back(variable);
        }
      }
      for (size_t i = 0; i < _config.layers.size(); ++i) {
        auto layer = std::make_unique<DramaBlock>(model, scope + "/layers_" + std::to_string(i),
                                                  _config.layers[i], options);
        if (layer->output_size() != output_size())
          throw std::invalid_argument("DramaBackbone layers must have the same hidden dimension");
        _layers.emplace_back(std::move(layer));
      }
      if (!_config.lm_candidates.empty()) {
        std::unordered_set<int32_t> seen;
        for (const auto id : _config.lm_candidates) {
          if (id < 0 || id >= _vocabulary_size || !seen.insert(id).second)
            throw std::invalid_argument(
                "DramaBackbone LM candidates must be unique vocabulary ids");
        }
        const StorageView ids({dim_t(_config.lm_candidates.size())}, _config.lm_candidates);
        _lm_head.select_weights(&ids);
      }
      _resident_gpu = options.resident_gpu();
      if (_resident_gpu) {
        _gpu_device_index = options.device_index;
        const ScopedDeviceSetter setter(Device::CUDA, _gpu_device_index);
        _gpu_norm = upload_drama_weight(_norm, options, _shared_gpu_parameters);
      }
    }

    DataType DramaBackbone::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t DramaBackbone::output_size() const {
      return _norm.dim(0);
    }

    dim_t DramaBackbone::vocabulary_size() const {
      return _vocabulary_size;
    }

    void DramaBackbone::initialize_state(DramaBackboneState& state, dim_t max_tokens) const {
      if (max_tokens == 0)
        max_tokens = _config.max_context;
      if (max_tokens < 1 || max_tokens > _config.max_context)
        throw std::invalid_argument("DramaBackbone cache capacity must be within max_context");
      DramaBackboneState initialized;
      initialized.capacity = max_tokens;
      initialized.layers.resize(_layers.size());
      for (size_t i = 0; i < _layers.size(); ++i)
        _layers[i]->initialize_state(initialized.layers[i], max_tokens);
      state = std::move(initialized);
    }

    void DramaBackbone::validate_input(const StorageView& embedded, const DramaBackboneState& state,
                                       const StorageView& hidden, const StorageView& logits) const {
      for (const auto* tensor : {&embedded, static_cast<const StorageView*>(&hidden),
                                 static_cast<const StorageView*>(&logits)}) {
        if (tensor->device() != Device::CPU || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("DramaBackbone requires CPU float32 inputs and outputs");
      }
      if (embedded.rank() != 2 || embedded.dim(1) != output_size()
          || state.layers.size() != _layers.size() || state.capacity <= 0
          || state.capacity > _config.max_context || state.length < 0
          || state.length > state.capacity || embedded.dim(0) > state.capacity - state.length)
        throw std::invalid_argument(
            "DramaBackbone input/state dimensions or context limit are incompatible");
      if (&hidden == &logits || &embedded == &hidden || &embedded == &logits
          || overlaps(embedded, hidden) || overlaps(embedded, logits) || overlaps(hidden, logits))
        throw std::invalid_argument("DramaBackbone inputs and outputs must not alias");
      for (size_t i = 0; i < state.layers.size(); ++i) {
        const auto& layer = state.layers[i];
        if (_config.layers[i].kind == DramaMixerKind::FullAttention
            && (layer.attention.length != state.length || layer.attention.keys.rank() != 3
                || layer.attention.keys.dim(0) != state.capacity))
          throw std::invalid_argument(
              "DramaBackbone attention cache position must match sequence position");
        for (const auto* tensor :
             {&layer.recurrent, &layer.history, &layer.attention.keys, &layer.attention.values}) {
          if (&embedded == tensor || &hidden == tensor || &logits == tensor
              || overlaps(embedded, *tensor) || overlaps(hidden, *tensor)
              || overlaps(logits, *tensor))
            throw std::invalid_argument("DramaBackbone cache must not alias inputs/outputs");
          for (const auto* parameter : _head_parameters) {
            if (overlaps(*parameter, *tensor))
              throw std::invalid_argument("DramaBackbone cache must not alias head parameters");
          }
        }
      }
      for (const auto* parameter : _head_parameters) {
        if (overlaps(*parameter, hidden) || overlaps(*parameter, logits))
          throw std::invalid_argument("DramaBackbone outputs must not alias head parameters");
      }
      for (dim_t i = 0; i < embedded.size(); ++i) {
        if (!std::isfinite(embedded.data<float>()[i]))
          throw std::invalid_argument("DramaBackbone embedded input must be finite");
      }
    }

    void DramaBackbone::operator()(const StorageView& embedded, DramaBackboneState& state,
                                   StorageView& hidden, StorageView& logits) const {
      validate_input(embedded, state, hidden, logits);
      if (!embedded.dim(0)) {
        hidden.resize({0, output_size()});
        logits.resize({0, vocabulary_size()});
        return;
      }
      const ScopedDeviceSetter setter(_resident_gpu ? Device::CUDA : Device::CPU,
                                      _gpu_device_index);
      StorageView current = _resident_gpu ? embedded.to(Device::CUDA) : StorageView(embedded);
      for (size_t i = 0; i < _layers.size(); ++i) {
        StorageView next(DataType::FLOAT32, _resident_gpu ? Device::CUDA : Device::CPU);
        (*_layers[i])(current, state.layers[i], next);
        current = std::move(next);
      }
      StorageView normalized(DataType::FLOAT32, _resident_gpu ? Device::CUDA : Device::CPU);
      ops::RMSNorm(_config.norm_epsilon, true)(_resident_gpu ? _gpu_norm : _norm, current,
                                               normalized);
      StorageView projected;
      if (embedded.dim(0) == 0)
        projected.resize({0, vocabulary_size()});
      else if (_config.lm_candidates.empty())
        _lm_head(normalized, projected);
      else {
        StorageView selected;
        _lm_head(normalized, selected);
        projected = StorageView({embedded.dim(0), vocabulary_size()}, 0.f);
        for (dim_t t = 0; t < embedded.dim(0); ++t)
          for (size_t i = 0; i < _config.lm_candidates.size(); ++i)
            projected.data<float>()[t * vocabulary_size() + _config.lm_candidates[i]] =
                selected.data<float>()[t * _config.lm_candidates.size() + i];
      }
      if (_resident_gpu) {
        normalized = normalized.to(Device::CPU);
        synchronize_stream(Device::CUDA);
      }
      for (const auto* tensor : {&normalized, &projected}) {
        for (dim_t i = 0; i < tensor->size(); ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::runtime_error(
                "DramaBackbone produced non-finite outputs; reset state before retry");
        }
      }
      state.length += embedded.dim(0);
      hidden = std::move(normalized);
      logits = std::move(projected);
    }

  }
}
