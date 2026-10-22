#include <algorithm>
#include <cmath>
#include <limits>

#include "test_utils.h"
#include "layers/drama/fish_dac.h"

namespace {

  const std::string scope = "codec/decoder";

  class TinyFishDACModel : public models::Model {
  public:
    explicit TinyFishDACModel(bool zero = false) {
      conv("model/0/conv", 7, .01f, zero);
      alpha("model/1/block/0", .8f, zero);
      conv("model/1/block/1/conv", 4, .11f, zero);
      for (int r = 0; r < 3; ++r) {
        const auto prefix = "model/1/block/" + std::to_string(r + 2) + "/block/";
        alpha(prefix + "0", .6f + .1f * r, zero);
        conv(prefix + "1/conv", 7, .002f * (r + 1), zero);
        alpha(prefix + "2", .9f, zero);
        conv(prefix + "3/conv", 1, .15f, zero);
      }
      alpha("model/2", .7f, zero);
      conv("model/3/conv", 7, .02f, zero);
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void replace(const std::string& name, StorageView value) {
      remove_variable(scope + "/" + name);
      register_variable(scope + "/" + name, std::move(value));
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyFishDACModel>(*this);
    }

  private:
    void conv(const std::string& name, dim_t kernel, float start, bool zero) {
      StorageView weight({1, 1, kernel}, 0.f);
      for (dim_t k = 0; k < kernel; ++k)
        weight.data<float>()[k] = zero ? 0.f : start + .003f * k;
      register_variable(scope + "/" + name + "/weight", std::move(weight));
      register_variable(scope + "/" + name + "/bias", StorageView({1}, zero ? 0.f : -.003f));
    }

    void alpha(const std::string& name, float value, bool zero) {
      register_variable(scope + "/" + name + "/alpha", StorageView({1, 1, 1}, zero ? 0.f : value));
    }
  };

  // Scalar equations only: no CT2 convolution or activation operators appear in
  // this reference. Accumulate in double to reduce shared rounding assumptions.
  std::vector<double> snake(std::vector<double> values, double alpha) {
    for (auto& value : values) {
      const double s = std::sin(alpha * value);
      value += s * s / (alpha + 1e-9);
    }
    return values;
  }

  std::vector<double> causal(const TinyFishDACModel& model, const std::string& name,
                             const std::vector<double>& values, int dilation) {
    const auto& weight = model.get_variable(scope + "/" + name + "/weight");
    const auto bias = model.get_variable(scope + "/" + name + "/bias").data<float>()[0];
    std::vector<double> result(values.size(), bias);
    for (size_t t = 0; t < values.size(); ++t) {
      for (dim_t k = 0; k < weight.dim(2); ++k) {
        const auto position = static_cast<int64_t>(t) - (weight.dim(2) - 1 - k) * dilation;
        if (position >= 0)
          result[t] += values[position] * weight.data<float>()[k];
      }
    }
    return result;
  }

  double coefficient(const TinyFishDACModel& model, const std::string& name) {
    return model.get_variable(scope + "/" + name + "/alpha").data<float>()[0];
  }

  std::vector<double> reference(const TinyFishDACModel& model, const StorageView& input) {
    std::vector<double> x(input.data<float>(), input.data<float>() + input.size());
    x = causal(model, "model/0/conv", x, 1);
    x = snake(std::move(x), coefficient(model, "model/1/block/0"));
    const auto& filter = model.get_variable(scope + "/model/1/block/1/conv/weight");
    const auto bias = model.get_variable(scope + "/model/1/block/1/conv/bias").data<float>()[0];
    // Keep exactly 2*T samples: discard the two rightmost samples of the
    // ordinary stride-two transposed convolution, without any left crop.
    std::vector<double> expanded(x.size() * 2, bias);
    for (size_t t = 0; t < x.size(); ++t)
      for (dim_t k = 0; k < 4; ++k)
        if (2 * t + k < expanded.size())
          expanded[2 * t + k] += x[t] * filter.data<float>()[k];
    x = std::move(expanded);
    int dilation = 1;
    for (int r = 0; r < 3; ++r) {
      const auto prefix = "model/1/block/" + std::to_string(r + 2) + "/block/";
      auto branch = snake(x, coefficient(model, prefix + "0"));
      branch = causal(model, prefix + "1/conv", branch, dilation);
      branch = snake(std::move(branch), coefficient(model, prefix + "2"));
      branch = causal(model, prefix + "3/conv", branch, 1);
      for (size_t t = 0; t < x.size(); ++t)
        x[t] += branch[t];
      dilation *= 3;
    }
    x = snake(std::move(x), coefficient(model, "model/2"));
    x = causal(model, "model/3/conv", x, 1);
    for (auto& value : x)
      value = std::tanh(value);
    return x;
  }

}

TEST(FishDACTest, MatchesIndependentScalarGraphIncludingCausalDilationsAndRightCrop) {
  TinyFishDACModel model;
  layers::FishDACWaveformDecoder decoder(model, scope, {2});
  StorageView input({1, 1, 35}, DataType::FLOAT32), output;
  for (dim_t t = 0; t < input.size(); ++t)
    input.data<float>()[t] = float((t * 7) % 13 - 6) / 9;
  const auto expected = reference(model, input);
  decoder(input, output);
  ASSERT_EQ(output.shape(), (Shape{1, 1, 70}));
  for (dim_t i = 0; i < output.size(); ++i)
    EXPECT_NEAR(output.data<float>()[i], expected[i], 2e-6) << "sample " << i;
}

TEST(FishDACTest, ZeroWeightsAndSingleFrameAreFiniteAndBounded) {
  TinyFishDACModel model(true);
  layers::FishDACWaveformDecoder decoder(model, scope, {2});
  StorageView input({1, 1, 1}, 100.f), output;
  decoder(input, output);
  expect_storage_eq(output, StorageView({1, 1, 2}, 0.f));
}

TEST(FishDACTest, RejectsInvalidInputShapesTypesAndNonfiniteSamples) {
  TinyFishDACModel model;
  layers::FishDACWaveformDecoder decoder(model, scope, {2});
  StorageView output;
  for (const auto& shape : {Shape{1, 3}, Shape{2, 1, 3}, Shape{1, 2, 3}, Shape{1, 1, 0}}) {
    StorageView input(shape, DataType::FLOAT32);
    EXPECT_THROW(decoder(input, output), std::invalid_argument);
  }
  StorageView wrong_type({1, 1, 3}, DataType::INT32);
  EXPECT_THROW(decoder(wrong_type, output), std::invalid_argument);
  StorageView input({1, 1, 3}, 1.f), wrong_output(DataType::INT32);
  EXPECT_THROW(decoder(input, wrong_output), std::invalid_argument);
  for (const float value :
       {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    input.data<float>()[1] = value;
    EXPECT_THROW(decoder(input, output), std::invalid_argument);
  }
}

TEST(FishDACTest, RejectsDirectPartialAndRetainedOutputAliasing) {
  TinyFishDACModel model;
  layers::FishDACWaveformDecoder decoder(model, scope, {2});
  StorageView input({1, 1, 3}, 1.f);
  EXPECT_THROW(decoder(input, input), std::invalid_argument);
  StorageView partial({1}, input.data<float>() + 1);
  EXPECT_THROW(decoder(input, partial), std::invalid_argument);
  StorageView retained({8}, 0.f);
  StorageView view({1, 1, 3}, retained.data<float>() + 4);
  retained.resize({0});
  EXPECT_THROW(decoder(view, retained), std::invalid_argument);
  const auto& weights = model.get_variable(scope + "/model/0/conv/weight");
  StorageView weight_alias({1}, const_cast<float*>(weights.data<float>()) + 2);
  EXPECT_THROW(decoder(input, weight_alias), std::invalid_argument);
}

TEST(FishDACTest, RejectsMalformedWeightsAndStrides) {
  TinyFishDACModel model;
  EXPECT_THROW(layers::FishDACWaveformDecoder(model, scope, {}), std::invalid_argument);
  EXPECT_THROW(layers::FishDACWaveformDecoder(model, scope, {0}), std::invalid_argument);
  model.replace("model/1/block/1/conv/weight", StorageView({1, 1, 3}, 1.f));
  EXPECT_THROW(layers::FishDACWaveformDecoder(model, scope, {2}), std::invalid_argument);
  TinyFishDACModel bad_alpha;
  bad_alpha.replace("model/2/alpha", StorageView({2}, 1.f));
  EXPECT_THROW(layers::FishDACWaveformDecoder(bad_alpha, scope, {2}), std::invalid_argument);
  TinyFishDACModel bad_finite;
  bad_finite.replace("model/3/conv/bias",
                     StorageView({1}, std::numeric_limits<float>::quiet_NaN()));
  EXPECT_THROW(layers::FishDACWaveformDecoder(bad_finite, scope, {2}), std::invalid_argument);
}

TEST(FishDACTest, GpuWaveformMatchesCpuAndIndependentScalarReference) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  for (int index = 0; index < get_device_count(Device::CUDA); ++index) {
    const ScopedDeviceSetter setter(Device::CUDA, index);
    TinyFishDACModel cpu_model, gpu_model;
    layers::DramaExecutionOptions execution;
    execution.device = Device::CUDA;
    execution.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
    execution.device_index = index;
    layers::FishDACWaveformDecoder cpu(cpu_model, scope, {2}), gpu(gpu_model, scope, {2}, execution);
    for (dim_t time : {1, 7, 37}) {
      StorageView input({1, 1, time});
      for (dim_t i = 0; i < time; ++i)
        input.data<float>()[i] = .4f * std::sin(float(i));
      StorageView expected, actual;
      cpu(input, expected);
      gpu(input, actual);
      expect_storage_eq(actual, expected, 2e-6f);
      const auto scalar = reference(cpu_model, input);
      ASSERT_EQ(actual.size(), dim_t(scalar.size()));
      for (dim_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual.data<float>()[i], scalar[i], 2e-6);
    }
  }
}

#ifdef CT2_WITH_CUDA
#  include "ops/drama/fish_dac_gpu.h"

TEST(FishDACTest, GpuCausalConvolutionTilesLongMultichannelInput) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  const ScopedDeviceSetter setter(Device::CUDA, 0);
  constexpr dim_t time = 45000, cin = 16, cout = 8, kernel = 7, dilation = 3,
                  left = (kernel - 1) * dilation;
  StorageView input({1, cin, time}), weight({cout, cin, kernel}), bias({cout}, .01f);
  for (dim_t i = 0; i < input.size(); ++i)
    input.data<float>()[i] = .2f * std::sin(float(i));
  for (dim_t i = 0; i < weight.size(); ++i)
    weight.data<float>()[i] = .02f * std::cos(float(i));
  StorageView padded({1, cin, time + left}, 0.f);
  for (dim_t c = 0; c < cin; ++c)
    std::copy_n(input.data<float>() + c * time, time,
                padded.data<float>() + c * (time + left) + left);
  StorageView reference;
  ops::Conv1D(1, 0, dilation)(padded, weight, bias, reference);
  auto gx = input.to(Device::CUDA), gw = weight.to(Device::CUDA), gb = bias.to(Device::CUDA);
  StorageView output(DataType::FLOAT32, Device::CUDA);
  ops::fish_causal_gpu(gx, gw, gb, dilation, output);
  auto host = output.to(Device::CPU);
  synchronize_stream(Device::CUDA);
  expect_storage_eq(host, reference, 2e-5f);
}
#endif
