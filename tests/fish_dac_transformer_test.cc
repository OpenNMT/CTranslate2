#include <array>
#include <cmath>
#include <limits>

#include "test_utils.h"
#include "layers/drama/fish_dac_transformer.h"

namespace {

  const std::string prefix = "codec/quantizer/post_module";

  class TinyCodecTransformer : public models::Model {
  public:
    TinyCodecTransformer() {
      add("norm/weight", {4}, {1.1f, .7f, -.9f, 1.3f});
      const std::string p = "layers/0/";
      add(p + "attention_norm/weight", {4}, {.8f, 1.2f, -.6f, .9f});
      add(p + "ffn_norm/weight", {4}, {1.1f, .7f, 1.3f, -.8f});
      add(p + "attention_layer_scale/gamma", {4}, {.2f, .4f, .3f, .1f});
      add(p + "ffn_layer_scale/gamma", {4}, {.1f, .3f, .2f, .4f});
      StorageView qkv({12, 4}, 0.f);
      for (dim_t part = 0; part < 3; ++part)
        for (dim_t c = 0; c < 4; ++c)
          qkv.data<float>()[(part * 4 + c) * 4 + c] = .7f + .2f * part;
      register_variable(prefix + "/" + p + "attention/wqkv/weight", std::move(qkv));
      add(p + "attention/wo/weight", {4, 4},
          {1, .2f, 0, 0, 0, .8f, -.1f, 0, 0, 0, .6f, .3f, -.2f, 0, 0, .9f});
      add(p + "feed_forward/w1/weight", {3, 4},
          {.1f, -.2f, .3f, .4f, .2f, .1f, -.1f, .3f, -.3f, .2f, .1f, -.2f});
      add(p + "feed_forward/w3/weight", {3, 4},
          {.2f, .1f, -.3f, .2f, -.1f, .3f, .2f, .1f, .4f, -.2f, .3f, .1f});
      add(p + "feed_forward/w2/weight", {4, 3},
          {.1f, .2f, -.1f, .3f, -.2f, .1f, -.1f, .2f, .3f, .1f, -.1f, .2f});
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void replace(const std::string& name, StorageView value) {
      remove_variable(prefix + "/" + name);
      register_variable(prefix + "/" + name, std::move(value));
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyCodecTransformer>(*this);
    }

  private:
    void add(const std::string& name, Shape shape, std::vector<float> values) {
      register_variable(prefix + "/" + name, StorageView(shape, values));
    }
  };

  using Vector = std::vector<double>;
  using Matrix = std::vector<Vector>;

  Vector linear(const TinyCodecTransformer& model, const std::string& name, const Vector& x) {
    const auto& w = model.get_variable(prefix + "/" + name + "/weight");
    Vector result(w.dim(0), 0);
    for (dim_t r = 0; r < w.dim(0); ++r)
      for (dim_t c = 0; c < w.dim(1); ++c)
        result[r] += x[c] * w.data<float>()[r * w.dim(1) + c];
    return result;
  }

  Vector norm(const TinyCodecTransformer& model, const std::string& name, const Vector& x) {
    const auto& gamma = model.get_variable(prefix + "/" + name + "/weight");
    double sum = 0;
    for (auto v : x)
      sum += v * v;
    Vector result(x.size());
    for (size_t c = 0; c < x.size(); ++c)
      result[c] = x[c] * gamma.data<float>()[c] / std::sqrt(sum / x.size() + 1e-5);
    return result;
  }

  Matrix scalar_reference(const TinyCodecTransformer& model, const StorageView& input,
                          dim_t window) {
    const dim_t time = input.dim(2);
    const std::string p = "layers/0/";
    Matrix x(time, Vector(4)), qkv(time);
    for (dim_t t = 0; t < time; ++t) {
      for (dim_t c = 0; c < 4; ++c)
        x[t][c] = input.data<float>()[c * time + t];
      qkv[t] = linear(model, p + "attention/wqkv", norm(model, p + "attention_norm", x[t]));
      for (dim_t part = 0; part < 2; ++part)
        for (dim_t h = 0; h < 2; ++h) {
          const auto i = part * 4 + h * 2;
          const auto a = qkv[t][i], b = qkv[t][i + 1];
          qkv[t][i] = a * std::cos(double(t)) - b * std::sin(double(t));
          qkv[t][i + 1] = a * std::sin(double(t)) + b * std::cos(double(t));
        }
    }
    for (dim_t t = 0; t < time; ++t) {
      Vector context(4, 0);
      const auto first = std::max<dim_t>(0, t - window + 1);
      for (dim_t h = 0; h < 2; ++h) {
        Vector logits(t - first + 1);
        double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
        for (dim_t s = first; s <= t; ++s) {
          logits[s - first] =
              (qkv[t][h * 2] * qkv[s][4 + h * 2] + qkv[t][h * 2 + 1] * qkv[s][4 + h * 2 + 1])
              / std::sqrt(2.);
          maximum = std::max(maximum, logits[s - first]);
        }
        for (auto& value : logits)
          denominator += (value = std::exp(value - maximum));
        for (dim_t s = first; s <= t; ++s)
          for (dim_t d = 0; d < 2; ++d)
            context[h * 2 + d] += logits[s - first] / denominator * qkv[s][8 + h * 2 + d];
      }
      const auto attention = linear(model, p + "attention/wo", context);
      const auto* a_scale =
          model.get_variable(prefix + "/" + p + "attention_layer_scale/gamma").data<float>();
      for (dim_t c = 0; c < 4; ++c)
        x[t][c] += attention[c] * a_scale[c];
      const auto normalized = norm(model, p + "ffn_norm", x[t]);
      auto gate = linear(model, p + "feed_forward/w1", normalized);
      const auto up = linear(model, p + "feed_forward/w3", normalized);
      for (size_t c = 0; c < gate.size(); ++c)
        gate[c] = gate[c] / (1 + std::exp(-gate[c])) * up[c];
      const auto down = linear(model, p + "feed_forward/w2", gate);
      const auto* f_scale =
          model.get_variable(prefix + "/" + p + "ffn_layer_scale/gamma").data<float>();
      for (dim_t c = 0; c < 4; ++c)
        x[t][c] += down[c] * f_scale[c];
      x[t] = norm(model, "norm", x[t]);
    }
    return x;
  }

  StorageView signal(dim_t time) {
    StorageView input({1, 4, time}, DataType::FLOAT32);
    for (dim_t i = 0; i < input.size(); ++i)
      input.data<float>()[i] = float((i * 3) % 17 - 8) / 7;
    return input;
  }

  layers::FishDACTransformerConfig config(dim_t window) {
    return {2, 1, window, 1e-5f, 10000.f};
  }

}

TEST(FishDACTransformerTest, MatchesIndependentScalarAttentionRoPESwiGLUNormAndLayerScale) {
  TinyCodecTransformer model;
  layers::FishDACTransformer layer(model, prefix, config(3));
  auto input = signal(7);
  const auto reference = scalar_reference(model, input, 3);
  StorageView output;
  layer(input, output);
  ASSERT_EQ(output.shape(), input.shape());
  for (dim_t t = 0; t < 7; ++t)
    for (dim_t c = 0; c < 4; ++c)
      EXPECT_NEAR(output.data<float>()[c * 7 + t], reference[t][c], 3e-5);
}

TEST(FishDACTransformerTest, Window128BoundaryAndNoFutureLeakage) {
  TinyCodecTransformer model;
  layers::FishDACTransformer layer(model, prefix, config(128));
  auto input = signal(130);
  StorageView original, changed;
  layer(input, original);
  // With one layer only, position zero must be excluded at t=128, but included
  // at t=127. Change its values without changing any other input row.
  for (dim_t c = 0; c < 4; ++c)
    input.data<float>()[c * 130] = float(c + 1) * 2;
  layer(input, changed);
  double included_difference = 0;
  for (dim_t c = 0; c < 4; ++c) {
    EXPECT_NEAR(original.data<float>()[c * 130 + 128], changed.data<float>()[c * 130 + 128], 1e-7);
    included_difference +=
        std::abs(original.data<float>()[c * 130 + 127] - changed.data<float>()[c * 130 + 127]);
  }
  EXPECT_GT(included_difference, 1e-5);
  const auto reference = scalar_reference(model, input, 128);
  for (dim_t t : {127, 128, 129})
    for (dim_t c = 0; c < 4; ++c)
      EXPECT_NEAR(changed.data<float>()[c * 130 + t], reference[t][c], 3e-5);
  input.data<float>()[129] += 100;
  layer(input, changed);
  for (dim_t c = 0; c < 4; ++c)
    EXPECT_NEAR(changed.data<float>()[c * 130 + 128], reference[128][c], 3e-5);
}

TEST(FishDACTransformerTest, ZeroInputAndRepeatedCallsDoNotRetainState) {
  TinyCodecTransformer model;
  layers::FishDACTransformer layer(model, prefix, config(3));
  auto input = signal(4);
  StorageView first, second;
  layer(input, first);
  StorageView zero({1, 4, 1}, 0.f);
  layer(zero, second);
  expect_storage_eq(second, StorageView({1, 4, 1}, 0.f));
  layer(input, second);
  expect_storage_eq(first, second, 1e-6f);
}

TEST(FishDACTransformerTest, RejectsInvalidInputAndOutputAliases) {
  TinyCodecTransformer model;
  layers::FishDACTransformer layer(model, prefix, config(3));
  StorageView output;
  for (const auto& shape : {Shape{4, 1}, Shape{2, 4, 1}, Shape{1, 3, 1}, Shape{1, 4, 0}}) {
    StorageView input(shape, DataType::FLOAT32);
    EXPECT_THROW(layer(input, output), std::invalid_argument);
  }
  auto input = signal(4);
  EXPECT_THROW(layer(input, input), std::invalid_argument);
  StorageView alias({1}, input.data<float>() + 2);
  EXPECT_THROW(layer(input, alias), std::invalid_argument);
  StorageView wrong_type(input.shape(), DataType::INT32);
  EXPECT_THROW(layer(wrong_type, output), std::invalid_argument);
  input.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(layer(input, output), std::invalid_argument);
}

TEST(FishDACTransformerTest, RejectsInvalidConfigurationWeightsAndBiases) {
  TinyCodecTransformer model;
  auto invalid = config(0);
  EXPECT_THROW(layers::FishDACTransformer(model, prefix, invalid), std::invalid_argument);
  invalid = config(3);
  invalid.heads = 4;
  EXPECT_THROW(layers::FishDACTransformer(model, prefix, invalid), std::invalid_argument);
  invalid = config(3);
  invalid.epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(layers::FishDACTransformer(model, prefix, invalid), std::invalid_argument);
  model.replace("layers/0/attention/wqkv/weight", StorageView({11, 4}, 0.f));
  EXPECT_THROW(layers::FishDACTransformer(model, prefix, config(3)), std::invalid_argument);
  TinyCodecTransformer with_bias;
  with_bias.replace("layers/0/attention/wo/bias", StorageView({4}, 0.f));
  EXPECT_THROW(layers::FishDACTransformer(with_bias, prefix, config(3)), std::invalid_argument);
}
