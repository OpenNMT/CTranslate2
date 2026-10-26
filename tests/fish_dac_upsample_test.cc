#include <cmath>
#include <limits>

#include "test_utils.h"
#include "layers/drama/fish_dac_upsample.h"

namespace {

  const std::string scope = "codec/quantizer/upsample";

  class LatentModel : public models::Model {
  public:
    explicit LatentModel(dim_t blocks = 2) {
      for (dim_t i = 0; i < blocks; ++i) {
        const auto p = std::to_string(i);
        StorageView transpose({2, 2, 2}, 0.f);
        transpose.data<float>()[0] = 1;
        transpose.data<float>()[1] = 2;
        transpose.data<float>()[6] = 1;
        transpose.data<float>()[7] = 2;
        put(p + "/0/conv/weight", std::move(transpose));
        put(p + "/0/conv/bias", StorageView({2}, std::vector<float>{.1f, -.2f}));
        StorageView depthwise({2, 1, 7}, 0.f);
        depthwise.data<float>()[5] = .2f;
        depthwise.data<float>()[6] = .5f;
        depthwise.data<float>()[12] = .3f;
        depthwise.data<float>()[13] = -.25f;
        put(p + "/1/dwconv/conv/weight", std::move(depthwise));
        put(p + "/1/dwconv/conv/bias", StorageView({2}, std::vector<float>{.05f, -.1f}));
        put(p + "/1/norm/weight", StorageView({2}, std::vector<float>{1.2f, .7f}));
        put(p + "/1/norm/bias", StorageView({2}, std::vector<float>{-.1f, .2f}));
        put(p + "/1/gamma", StorageView({2}, std::vector<float>{.2f, -.4f}));
        StorageView w1({8, 2}), w2({2, 8});
        for (dim_t j = 0; j < 16; ++j) {
          w1.data<float>()[j] = float(j % 5 - 2) / 7;
          w2.data<float>()[j] = float(j % 7 - 3) / 11;
        }
        put(p + "/1/pwconv1/weight", std::move(w1));
        put(p + "/1/pwconv2/weight", std::move(w2));
        put(p + "/1/pwconv1/bias", StorageView({8}, .1f));
        put(p + "/1/pwconv2/bias", StorageView({2}, std::vector<float>{-.3f, .25f}));
      }
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void put(const std::string& name, StorageView value) {
      remove_variable(scope + "/" + name);
      register_variable(scope + "/" + name, std::move(value));
    }

  protected:
    // Materialized codec weights must bypass Model's generic convolution-name
    // heuristic: pwconv matrices are rank 2 despite containing "conv".
    bool is_quantizable(const std::string&) const override {
      return false;
    }

    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<LatentModel>(*this);
    }
  };

  // Entire block evaluated by independent double scalar equations, without
  // invoking any CT2 operation or layer in the reference calculation.
  std::vector<double> scalar_block(const LatentModel& model, dim_t index,
                                   const std::vector<double>& input, dim_t length) {
    const auto prefix = scope + "/" + std::to_string(index);
    auto data = [&](const std::string& suffix) {
      return model.get_variable(prefix + suffix).data<float>();
    };
    const dim_t time = 2 * length;
    std::vector<double> upsampled(2 * time);
    for (dim_t c = 0; c < 2; ++c)
      for (dim_t t = 0; t < time; ++t)
        upsampled[c * time + t] = data("/0/conv/bias")[c];
    for (dim_t ic = 0; ic < 2; ++ic)
      for (dim_t oc = 0; oc < 2; ++oc)
        for (dim_t t = 0; t < length; ++t)
          for (dim_t k = 0; k < 2; ++k)
            upsampled[oc * time + 2 * t + k] +=
                input[ic * length + t] * data("/0/conv/weight")[(ic * 2 + oc) * 2 + k];
    auto result = upsampled;
    for (dim_t t = 0; t < time; ++t) {
      double dw[2];
      for (dim_t c = 0; c < 2; ++c) {
        dw[c] = data("/1/dwconv/conv/bias")[c];
        for (dim_t k = 0; k < 7; ++k) {
          const auto position = t - 6 + k;
          if (position >= 0)
            dw[c] += upsampled[c * time + position] * data("/1/dwconv/conv/weight")[c * 7 + k];
        }
      }
      const double mean = (dw[0] + dw[1]) / 2;
      const double variance =
          ((dw[0] - mean) * (dw[0] - mean) + (dw[1] - mean) * (dw[1] - mean)) / 2;
      double normalized[2];
      for (dim_t c = 0; c < 2; ++c)
        normalized[c] = (dw[c] - mean) / std::sqrt(variance + 1e-6) * data("/1/norm/weight")[c]
                        + data("/1/norm/bias")[c];
      double gelu[8];
      for (dim_t h = 0; h < 8; ++h) {
        double wide = data("/1/pwconv1/bias")[h];
        for (dim_t c = 0; c < 2; ++c)
          wide += normalized[c] * data("/1/pwconv1/weight")[h * 2 + c];
        gelu[h] = .5 * wide * (1 + std::erf(wide / std::sqrt(2.)));
      }
      for (dim_t c = 0; c < 2; ++c) {
        double projected = data("/1/pwconv2/bias")[c];
        for (dim_t h = 0; h < 8; ++h)
          projected += gelu[h] * data("/1/pwconv2/weight")[c * 8 + h];
        result[c * time + t] += data("/1/gamma")[c] * projected;
      }
    }
    return result;
  }

}

TEST(FishDACLatentUpsampleTest, IndependentDoubleEquationsForTwoBlocks) {
  LatentModel model;
  layers::FishDACLatentUpsampler layer(model, scope);
  StorageView input({1, 2, 3}, std::vector<float>{1, -1, .2f, 2, .5f, -.7f});
  std::vector<double> expected(input.data<float>(), input.data<float>() + input.size());
  expected = scalar_block(model, 0, expected, 3);
  expected = scalar_block(model, 1, expected, 6);
  StorageView output;
  layer(input, output);
  EXPECT_EQ(output.shape(), (Shape{1, 2, 12}));
  for (dim_t i = 0; i < output.size(); ++i)
    EXPECT_NEAR(output.data<float>()[i], expected[i], 3e-6);
}

TEST(FishDACLatentUpsampleTest, CausalPrefixAndSingleFrame) {
  LatentModel model;
  layers::FishDACLatentUpsampler layer(model, scope);
  StorageView input({1, 2, 3}, std::vector<float>{1, 20, -7, 2, -10, 30});
  StorageView prefix({1, 2, 1}, std::vector<float>{1, 2});
  StorageView full, first;
  layer(input, full);
  layer(prefix, first);
  EXPECT_EQ(first.shape(), (Shape{1, 2, 4}));
  for (dim_t c = 0; c < 2; ++c)
    expect_array_eq(first.data<float>() + c * 4, full.data<float>() + c * 12, 4, 2e-6f);
}

TEST(FishDACLatentUpsampleTest, EmptyTimeAndConfigurableBlockCount) {
  LatentModel model(1);
  layers::FishDACLatentUpsampler layer(model, scope, 1);
  StorageView empty({1, 2, 0}), output;
  layer(empty, output);
  EXPECT_EQ(output.shape(), (Shape{1, 2, 0}));
  StorageView input({1, 2, 1}, 1.f);
  layer(input, output);
  EXPECT_EQ(output.shape(), (Shape{1, 2, 2}));
}

TEST(FishDACLatentUpsampleTest, RejectsInvalidParametersAndShapes) {
  LatentModel model;
  EXPECT_THROW(layers::FishDACLatentUpsampler(model, scope, 0), std::invalid_argument);
  model.put("0/1/norm/bias", StorageView({3}, 0.f));
  EXPECT_THROW(layers::FishDACLatentUpsampler(model, scope), std::invalid_argument);
  model.put("0/1/norm/bias", StorageView({2}, 0.f));
  model.put("0/0/conv/weight", StorageView({2, 2, 3}, 1.f));
  EXPECT_THROW(layers::FishDACLatentUpsampler(model, scope), std::invalid_argument);
}

TEST(FishDACLatentUpsampleTest, RejectsAliasingAndInvalidInputWithoutChangingOutput) {
  LatentModel model;
  layers::FishDACLatentUpsampler layer(model, scope);
  StorageView input({1, 2, 1}, 1.f), output({1}, 42.f);
  StorageView wrong_batch({2, 2, 1}, 1.f), integers({1, 2, 1}, int32_t(1));
  EXPECT_THROW(layer(wrong_batch, output), std::invalid_argument);
  EXPECT_THROW(layer(integers, output), std::invalid_argument);
  EXPECT_THROW(layer(input, input), std::invalid_argument);
  StorageView allocation({32}, 1.f);
  StorageView aliased({1, 2, 1}, allocation.data<float>() + 5);
  allocation.clear();
  EXPECT_THROW(layer(aliased, allocation), std::invalid_argument);
  const auto& gamma = model.get_variable(scope + "/0/1/gamma");
  StorageView weight_output({1}, const_cast<float*>(gamma.data<float>()));
  EXPECT_THROW(layer(input, weight_output), std::invalid_argument);
  input.data<float>()[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(layer(input, output), std::invalid_argument);
  EXPECT_EQ(output.data<float>()[0], 42.f);
}
