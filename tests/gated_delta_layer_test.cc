#include <cmath>
#include <limits>

#include "test_utils.h"
#include "layers/drama/gated_delta.h"

namespace {

  class GatedDeltaModel : public models::Model {
  public:
    GatedDeltaModel(dim_t hidden = 3, dim_t kh = 1, dim_t vh = 2, dim_t kd = 2, dim_t vd = 2,
                    dim_t kernel = 3) {
      const dim_t channels = 2 * kh * kd + vh * vd;
      matrix("in_proj_qkv", channels, hidden);
      matrix("in_proj_z", vh * vd, hidden);
      matrix("in_proj_a", vh, hidden);
      matrix("in_proj_b", vh, hidden);
      matrix("out_proj", hidden, vh * vd);
      put("conv1d/weight", StorageView({channels, 1, kernel}, .4f));
      put("A_log", StorageView({vh}, -.3f));
      put("dt_bias", StorageView({vh}, .2f));
      put("norm/weight", StorageView({vd}, 1.3f));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void put(const std::string& name, StorageView value) {
      remove_variable("delta/" + name);
      register_variable("delta/" + name, std::move(value));
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<GatedDeltaModel>(*this);
    }

  private:
    void matrix(const std::string& name, dim_t rows, dim_t columns) {
      StorageView weights({rows, columns});
      for (dim_t i = 0; i < weights.size(); ++i)
        weights.data<float>()[i] = float(i % 7 - 3) / 5;
      put(name + "/weight", std::move(weights));
    }
  };

}

TEST(GatedDeltaLayerTest, IndependentNonzeroStateScalarEquation) {
  GatedDeltaModel model(1, 1, 1, 1, 1, 2);
  model.put("in_proj_qkv/weight", StorageView({3, 1}, std::vector<float>{1, 2, 3}));
  model.put("in_proj_z/weight", StorageView({1, 1}, .5f));
  model.put("in_proj_a/weight", StorageView({1, 1}, -.2f));
  model.put("in_proj_b/weight", StorageView({1, 1}, .4f));
  model.put("out_proj/weight", StorageView({1, 1}, -2.f));
  model.put("conv1d/weight", StorageView({3, 1, 2}, std::vector<float>{.5f, 1, 1, -.25f, -.5f, 2}));
  model.put("A_log", StorageView({1}, std::log(2.f)));
  model.put("dt_bias", StorageView({1}, .3f));
  model.put("norm/weight", StorageView({1}, 1.5f));
  layers::GatedDeltaNet layer(model, "delta", 1, 1, 1, 1, .01f);
  StorageView state({1, 1, 1, 1}, .7f);
  StorageView history({1, 3, 1}, std::vector<float>{1, -2, .25f});
  StorageView input({1, 1}, 2.f), output;
  layer(input, state, history, output);
  const auto silu = [](double x) {
    return x / (1 + std::exp(-x));
  };
  // Causal conv uses raw previous projected QKV, followed by SiLU.
  double q = silu(2.5), k = silu(-3), v = silu(11.875);
  q /= std::sqrt(q * q + 1e-6);
  k /= std::sqrt(k * k + 1e-6);
  const double decay = std::pow(1 + std::exp(-.1), -2);
  const double beta = 1 / (1 + std::exp(-.8));
  const double decayed_state = .7 * decay;
  const double expected_state = (1 - beta * k * k) * decayed_state + beta * k * v;
  const double core = q * expected_state;
  const double expected = -2 * 1.5 * core / std::sqrt(core * core + .01) * silu(1);
  EXPECT_NEAR(state.data<float>()[0], expected_state, 2e-6);
  EXPECT_NEAR(output.data<float>()[0], expected, 2e-6);
  expect_storage_eq(history, StorageView({1, 3, 1}, std::vector<float>{2, 4, 6}));
}

TEST(GatedDeltaLayerTest, ChunkStepParityWithGroupedHeadsAndNonzeroCaches) {
  GatedDeltaModel model;
  layers::GatedDeltaNet layer(model, "delta", 1, 2, 2, 2, 1e-5f);
  StorageView input({5, 3});
  for (dim_t i = 0; i < input.size(); ++i)
    input.data<float>()[i] = float(i % 11 - 5) / 4;
  StorageView state, history;
  layer.initialize_state(state, history);
  state.fill(.15f);
  for (dim_t i = 0; i < history.size(); ++i)
    history.data<float>()[i] = float(i % 5 - 2) / 3;
  StorageView step_state(state), step_history(history), output;
  layer(input, state, history, output);
  for (dim_t t = 0; t < 5; ++t) {
    StorageView token({1, 3}, input.data<float>() + t * 3), step_output;
    layer(token, step_state, step_history, step_output);
    expect_array_eq(step_output.data<float>(), output.data<float>() + t * 3, 3, 2e-6f);
  }
  expect_storage_eq(step_state, state, 2e-6f);
  expect_storage_eq(step_history, history, 2e-6f);
  EXPECT_EQ(layer.output_size(), 3);
  EXPECT_EQ(layer.output_type(), DataType::FLOAT32);
}

TEST(GatedDeltaLayerTest, EmptyInputPreservesCachesAndKernelOneWorks) {
  GatedDeltaModel model(3, 1, 2, 2, 2, 1);
  layers::GatedDeltaNet layer(model, "delta", 1, 2, 2, 2, 1e-5f);
  StorageView state, history;
  layer.initialize_state(state, history);
  state.fill(.2f);
  const StorageView original_state(state), original_history(history);
  StorageView empty({0, 3}), output;
  layer(empty, state, history, output);
  expect_storage_eq(state, original_state);
  expect_storage_eq(history, original_history);
  EXPECT_EQ(output.shape(), (Shape{0, 3}));
  StorageView input({1, 3}, .1f);
  layer(input, state, history, output);
  EXPECT_EQ(output.shape(), (Shape{1, 3}));
  for (dim_t i = 0; i < output.size(); ++i)
    EXPECT_TRUE(std::isfinite(output.data<float>()[i]));
}

TEST(GatedDeltaLayerTest, GuardsShapesTypesAndRetainedAllocationAliases) {
  GatedDeltaModel model;
  layers::GatedDeltaNet layer(model, "delta", 1, 2, 2, 2, 1e-5f);
  StorageView state, history, input({1, 3}, .2f), output;
  layer.initialize_state(state, history);
  const StorageView original_state(state), original_history(history);
  StorageView bad_input({1, 2}, .1f), integers({1, 3}, int32_t(1));
  EXPECT_THROW(layer(bad_input, state, history, output), std::invalid_argument);
  EXPECT_THROW(layer(integers, state, history, output), std::invalid_argument);
  EXPECT_THROW(layer(input, state, history, state), std::invalid_argument);
  StorageView allocation({64}, .2f);
  StorageView aliased_input({1, 3}, allocation.data<float>() + 2);
  allocation.clear();
  EXPECT_THROW(layer(aliased_input, state, history, allocation), std::invalid_argument);
  StorageView parameter_output(
      {1}, const_cast<float*>(model.get_variable("delta/norm/weight").data<float>()));
  EXPECT_THROW(layer(input, state, history, parameter_output), std::invalid_argument);
  StorageView wrong_state({1, 2, 1, 4}, .2f);
  EXPECT_THROW(layer(input, wrong_state, history, output), std::invalid_argument);
  StorageView wrong_history({1, 8, 1}, .2f);
  EXPECT_THROW(layer(input, state, wrong_history, output), std::invalid_argument);
  expect_storage_eq(state, original_state);
  expect_storage_eq(history, original_history);
}

TEST(GatedDeltaLayerTest, RejectsInvalidConfiguration) {
  GatedDeltaModel model;
  EXPECT_THROW(layers::GatedDeltaNet(model, "delta", 0, 2, 2, 2, 1e-5f), std::invalid_argument);
  EXPECT_THROW(layers::GatedDeltaNet(model, "delta", 3, 2, 2, 2, 1e-5f), std::invalid_argument);
  EXPECT_THROW(layers::GatedDeltaNet(model, "delta", 1, 2, 2, 2, 0.f), std::invalid_argument);
  model.put("norm/weight", StorageView({4}, 1.f));
  EXPECT_THROW(layers::GatedDeltaNet(model, "delta", 1, 2, 2, 2, 1e-5f), std::invalid_argument);
}

TEST(GatedDeltaLayerTest, NumericalFailureDoesNotCommitCaches) {
  GatedDeltaModel model;
  layers::GatedDeltaNet layer(model, "delta", 1, 2, 2, 2, 1e-5f);
  StorageView state, history, output({1}, 42.f);
  layer.initialize_state(state, history);
  state.fill(.3f);
  history.fill(.2f);
  const StorageView original_state(state), original_history(history);
  StorageView input({1, 3}, .2f);
  input.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(layer(input, state, history, output), std::invalid_argument);
  expect_storage_eq(state, original_state);
  expect_storage_eq(history, original_history);
  EXPECT_EQ(output.data<float>()[0], 42.f);
}
