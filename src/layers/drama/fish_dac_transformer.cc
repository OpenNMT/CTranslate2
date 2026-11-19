#include "layers/drama/fish_dac_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        const auto an = std::max(a.reserved_memory(), a.size() * a.item_size());
        const auto bn = std::max(b.reserved_memory(), b.size() * b.item_size());
        if (!an || !bn)
          return false;
        const auto ap = reinterpret_cast<uintptr_t>(a.buffer());
        const auto bp = reinterpret_cast<uintptr_t>(b.buffer());
        return ap <= bp ? bp - ap < static_cast<uintptr_t>(an)
                        : ap - bp < static_cast<uintptr_t>(bn);
      }

      void require_finite(const StorageView& tensor) {
        for (dim_t i = 0; i < tensor.size(); ++i)
          if (!std::isfinite(tensor.data<float>()[i]))
            throw std::invalid_argument("FishDAC transformer tensors must be finite");
      }

    }

    FishDACTransformer::FishDACTransformer(const models::Model& model, const std::string& scope,
                                           FishDACTransformerConfig config)
      : _config(config) {
      if (model.device() != Device::CPU || config.heads <= 0 || config.layers <= 0
          || config.window <= 0 || !std::isfinite(config.epsilon) || config.epsilon <= 0
          || !std::isfinite(config.rope_theta) || config.rope_theta <= 0)
        throw std::invalid_argument("invalid FishDAC transformer configuration");
      auto parameter = [&](const std::string& name) {
        const auto& value = model.get_variable(name);
        if (value.device() != Device::CPU || value.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("FishDAC transformer requires CPU float32 weights");
        require_finite(value);
        _parameters.push_back(&value);
        return &value;
      };
      _norm = parameter(scope + "/norm/weight");
      if (_norm->rank() != 1 || _norm->dim(0) <= 0)
        throw std::invalid_argument("FishDAC transformer final norm must be a nonempty vector");
      _channels = _norm->dim(0);
      if (_channels % config.heads || (_channels / config.heads) % 2
          || _channels > std::numeric_limits<dim_t>::max() / 3)
        throw std::invalid_argument(
            "FishDAC transformer requires an even full-head rotary dimension");
      auto vector = [&](const std::string& name) {
        const auto* value = parameter(name);
        if (value->shape() != Shape{_channels})
          throw std::invalid_argument("FishDAC transformer norm/scale shape mismatch");
        return value;
      };
      auto dense = [&](const std::string& name, dim_t rows, dim_t columns) {
        const auto* weight = parameter(name + "/weight");
        if (weight->shape() != Shape{rows, columns} || model.get_variable_if_exists(name + "/bias"))
          throw std::invalid_argument("FishDAC transformer projection shape/bias mismatch: "
                                      + name);
        return std::make_unique<Dense>(model, name);
      };
      for (dim_t index = 0; index < config.layers; ++index) {
        const auto prefix = scope + "/layers/" + std::to_string(index);
        Block block;
        block.attention_norm = vector(prefix + "/attention_norm/weight");
        block.ffn_norm = vector(prefix + "/ffn_norm/weight");
        block.attention_scale = vector(prefix + "/attention_layer_scale/gamma");
        block.ffn_scale = vector(prefix + "/ffn_layer_scale/gamma");
        block.qkv = dense(prefix + "/attention/wqkv", 3 * _channels, _channels);
        block.out = dense(prefix + "/attention/wo", _channels, _channels);
        const auto& gate_weight = model.get_variable(prefix + "/feed_forward/w1/weight");
        if (gate_weight.rank() != 2 || gate_weight.dim(0) <= 0)
          throw std::invalid_argument("FishDAC transformer FFN must be a nonempty matrix");
        const auto intermediate = gate_weight.dim(0);
        block.gate = dense(prefix + "/feed_forward/w1", intermediate, _channels);
        block.down = dense(prefix + "/feed_forward/w2", _channels, intermediate);
        block.up = dense(prefix + "/feed_forward/w3", intermediate, _channels);
        _blocks.push_back(std::move(block));
      }
    }

    void FishDACTransformer::validate_input(const StorageView& input,
                                            const StorageView& output) const {
      if (input.device() != Device::CPU || input.dtype() != DataType::FLOAT32
          || output.device() != Device::CPU || output.dtype() != DataType::FLOAT32
          || input.rank() != 3 || input.dim(0) != 1 || input.dim(1) != _channels
          || input.dim(2) <= 0 || &input == &output || overlaps(input, output))
        throw std::invalid_argument("FishDAC transformer expects nonaliasing float32 [1,C,T]");
      for (const auto* parameter : _parameters)
        if (overlaps(*parameter, output))
          throw std::invalid_argument("FishDAC transformer output aliases weights");
      require_finite(input);
      const dim_t time = input.dim(2);
      if (time > std::numeric_limits<dim_t>::max() / _channels / 3 / sizeof(float))
        throw std::invalid_argument("FishDAC transformer dimensions overflow");
    }

    void FishDACTransformer::apply_rotary(StorageView& qkv) const {
      const dim_t time = qkv.dim(0), head_dim = _channels / _config.heads;
      // Q, K and V occupy consecutive C-sized sections of every output row.
      for (dim_t t = 0; t < time; ++t) {
        for (dim_t part = 0; part < 2; ++part) {
          for (dim_t h = 0; h < _config.heads; ++h) {
            auto* values = qkv.data<float>() + t * 3 * _channels + part * _channels + h * head_dim;
            for (dim_t d = 0; d < head_dim; d += 2) {
              const double angle =
                  double(t) / std::pow(double(_config.rope_theta), double(d) / head_dim);
              const float cosine = std::cos(angle), sine = std::sin(angle);
              const float x = values[d], y = values[d + 1];
              values[d] = x * cosine - y * sine;
              values[d + 1] = x * sine + y * cosine;
            }
          }
        }
      }
    }

    void FishDACTransformer::apply_attention(const Block& block, StorageView& hidden) const {
      const dim_t time = hidden.dim(0), head_dim = _channels / _config.heads;
      StorageView normalized, qkv;
      ops::RMSNorm(_config.epsilon, false)(*block.attention_norm, hidden, normalized);
      (*block.qkv)(normalized, qkv);
      apply_rotary(qkv);
      StorageView context({time, _channels}, 0.f);
      std::vector<double> scores(std::min(time, _config.window));
      const double scale = 1. / std::sqrt(double(head_dim));
      for (dim_t t = 0; t < time; ++t) {
        const dim_t begin = std::max<dim_t>(0, t - _config.window + 1);
        for (dim_t h = 0; h < _config.heads; ++h) {
          double maximum = -std::numeric_limits<double>::infinity();
          for (dim_t s = begin; s <= t; ++s) {
            double dot = 0;
            for (dim_t d = 0; d < head_dim; ++d)
              dot += double(qkv.data<float>()[t * 3 * _channels + h * head_dim + d])
                     * qkv.data<float>()[s * 3 * _channels + _channels + h * head_dim + d];
            scores[s - begin] = dot * scale;
            maximum = std::max(maximum, scores[s - begin]);
          }
          double denominator = 0;
          for (dim_t s = begin; s <= t; ++s)
            denominator += (scores[s - begin] = std::exp(scores[s - begin] - maximum));
          for (dim_t d = 0; d < head_dim; ++d) {
            double value = 0;
            for (dim_t s = begin; s <= t; ++s)
              value += scores[s - begin] / denominator
                       * qkv.data<float>()[s * 3 * _channels + 2 * _channels + h * head_dim + d];
            context.data<float>()[t * _channels + h * head_dim + d] = value;
          }
        }
      }
      StorageView projected;
      (*block.out)(context, projected);
      for (dim_t i = 0; i < hidden.size(); ++i)
        hidden.data<float>()[i] +=
            projected.data<float>()[i] * block.attention_scale->data<float>()[i % _channels];
    }

    void FishDACTransformer::apply_feed_forward(const Block& block, StorageView& hidden) const {
      StorageView normalized;
      ops::RMSNorm(_config.epsilon, false)(*block.ffn_norm, hidden, normalized);
      StorageView gate, up, activated, down;
      (*block.gate)(normalized, gate);
      (*block.up)(normalized, up);
      ops::Swish()(gate, activated);
      ops::Mul()(activated, up, activated);
      (*block.down)(activated, down);
      for (dim_t i = 0; i < hidden.size(); ++i)
        hidden.data<float>()[i] +=
            down.data<float>()[i] * block.ffn_scale->data<float>()[i % _channels];
    }

    void FishDACTransformer::operator()(const StorageView& input, StorageView& output) const {
      validate_input(input, output);
      const dim_t time = input.dim(2);
      StorageView hidden({time, _channels}, DataType::FLOAT32);
      for (dim_t t = 0; t < time; ++t)
        for (dim_t c = 0; c < _channels; ++c)
          hidden.data<float>()[t * _channels + c] = input.data<float>()[c * time + t];
      for (const auto& block : _blocks) {
        apply_attention(block, hidden);
        apply_feed_forward(block, hidden);
      }
      StorageView normalized;
      ops::RMSNorm(_config.epsilon, false)(*_norm, hidden, normalized);
      require_finite(normalized);
      StorageView result(input.shape(), DataType::FLOAT32);
      for (dim_t t = 0; t < time; ++t)
        for (dim_t c = 0; c < _channels; ++c)
          result.data<float>()[c * time + t] = normalized.data<float>()[t * _channels + c];
      output = std::move(result);
    }

  }
}
