#include <cstdlib>
#include <fstream>

#include "drama_parity_test_utils.h"
#include "layers/drama/gated_self_attention.h"
#include "ctranslate2/models/model_factory.h"

namespace {

  class AttentionParityFixture : public models::Model {
  protected:
    bool is_quantizable(const std::string&) const override {
      return false;
    }

    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<AttentionParityFixture>(*this);
    }
  };

}

TEST(Drama3AttentionParityTest, ActualBackboneAndFusedTowerWeightsMatchIndependentOracle) {
  const char* path = std::getenv("CT2_DRAMA3_ATTENTION_FIXTURE");
  if (!path)
    GTEST_SKIP() << "Set CT2_DRAMA3_ATTENTION_FIXTURE to an independent attention fixture";
  models::register_model<AttentionParityFixture>("Drama3AttentionParityFixture");
  for (const auto* kind : {"backbone", "tower"}) {
    const std::string directory = std::string(path) + "/" + kind;
    std::ifstream stream(directory + "/manifest.json");
    ASSERT_TRUE(stream.good());
    const auto manifest = nlohmann::json::parse(stream);
    const auto model = models::Model::load(directory, Device::CPU, 0, ComputeType::FLOAT32);
    const dim_t hidden = manifest.at("hidden"), kv_heads = manifest.at("kv_heads"),
                dim = manifest.at("head_dim");
    layers::GatedSelfAttentionConfig cfg{manifest.at("heads"),
                                         kv_heads,
                                         dim,
                                         manifest.at("rotary_dim"),
                                         manifest.at("rope_base"),
                                         manifest.at("norm_epsilon"),
                                         manifest.at("interleaved"),
                                         true};
    layers::GatedSelfAttention layer(*model, "attn", cfg);
    for (const auto& fixture : manifest.at("cases")) {
      const dim_t steps = fixture.at("steps");
      const std::string label = std::string(kind) + "_t" + std::to_string(steps);
      SCOPED_TRACE(label);
      auto get = [&](const char* name, Shape shape) {
        return drama_test::read_tensor(directory, fixture.at(name), std::move(shape));
      };
      auto input = get("input", {steps, hidden});
      auto make_cache = [&]() {
        return layers::GatedSelfAttentionCache{
            StorageView({steps, kv_heads, dim}, DataType::FLOAT32),
            StorageView({steps, kv_heads, dim}, DataType::FLOAT32), 0};
      };
      auto cache = make_cache(), incremental = make_cache();
      StorageView output;
      layer(input, cache, output);
      drama_test::compare(output, get("expected", {steps, hidden}), 2e-4f, 2e-5, label + "_output");
      drama_test::compare(cache.keys, get("keys", {steps, kv_heads, dim}), 5e-5f, 5e-6,
                          label + "_keys");
      drama_test::compare(cache.values, get("values", {steps, kv_heads, dim}), 5e-5f, 5e-6,
                          label + "_values");
      for (dim_t t = 0; t < steps; ++t) {
        StorageView row({1, hidden}, input.data<float>() + t * hidden), one;
        layer(row, incremental, one);
        StorageView full_row({1, hidden}, output.data<float>() + t * hidden);
        drama_test::compare(one, full_row, 2e-4f, 2e-5, label + "_step" + std::to_string(t));
      }
      drama_test::compare(incremental.keys, cache.keys, 5e-5f, 5e-6, label + "_step_keys");
      drama_test::compare(incremental.values, cache.values, 5e-5f, 5e-6, label + "_step_values");
    }
  }
}
