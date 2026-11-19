#include <cstdlib>
#include <fstream>

#include "drama_parity_test_utils.h"
#include "layers/drama/gated_delta.h"
#include "ctranslate2/models/model_factory.h"

namespace {

  class BackboneParityFixture : public models::Model {
  protected:
    bool is_quantizable(const std::string&) const override {
      return false;
    }

    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<BackboneParityFixture>(*this);
    }
  };

}

TEST(Drama3BackboneParityTest, ActualGdnWeightsMatchIndependentFunctionalOracle) {
  const char* path = std::getenv("CT2_DRAMA3_BACKBONE_FIXTURE");
  if (!path)
    GTEST_SKIP() << "Set CT2_DRAMA3_BACKBONE_FIXTURE to an independent backbone fixture";
  const std::string directory(path);
  std::ifstream stream(directory + "/manifest.json");
  ASSERT_TRUE(stream.good());
  const auto manifest = nlohmann::json::parse(stream);
  ASSERT_EQ(manifest.at("schema_version").get<int>(), 1);
  models::register_model<BackboneParityFixture>("Drama3BackboneParityFixture");
  const auto model = models::Model::load(directory, Device::CPU, 0, ComputeType::FLOAT32);
  const dim_t kh = manifest.at("key_heads"), vh = manifest.at("value_heads");
  const dim_t kd = manifest.at("key_dim"), vd = manifest.at("value_dim");
  const dim_t hidden = manifest.at("hidden"), kernel = manifest.at("kernel");
  const Shape state_shape{1, vh, kd, vd};
  const Shape history_shape{1, 2 * kh * kd + vh * vd, kernel - 1};
  layers::GatedDeltaNet layer(*model, "gdn", kh, vh, kd, vd, manifest.at("rms_epsilon"));
  for (const auto& fixture : manifest.at("cases")) {
    const dim_t steps = fixture.at("steps");
    SCOPED_TRACE("actual-weight GDN steps=" + std::to_string(steps));
    auto read = [&](const char* key, Shape shape) {
      return drama_test::read_tensor(directory, fixture.at(key), std::move(shape));
    };
    auto input = read("input", {steps, hidden});
    auto state = read("initial_state", state_shape);
    auto history = read("initial_history", history_shape);
    auto step_state = StorageView(state), step_history = StorageView(history);
    StorageView output;
    layer(input, state, history, output);
    const std::string label = "t" + std::to_string(steps);
    drama_test::compare(output, read("expected", {steps, hidden}), 5e-4f, 5e-5, label + "_output");
    drama_test::compare(state, read("final_state", state_shape), 5e-6f, 5e-7, label + "_state");
    drama_test::compare(history, read("final_history", history_shape), 5e-5f, 5e-6,
                        label + "_history");
    for (dim_t t = 0; t < steps; ++t) {
      StorageView row({1, hidden}, input.data<float>() + t * hidden), one;
      layer(row, step_state, step_history, one);
      StorageView expected_row({1, hidden}, output.data<float>() + t * hidden);
      drama_test::compare(one, expected_row, 5e-4f, 5e-5, label + "_step" + std::to_string(t));
    }
    drama_test::compare(step_state, state, 5e-6f, 5e-7, label + "_step_state");
    drama_test::compare(step_history, history, 5e-5f, 5e-6, label + "_step_history");
  }
}
