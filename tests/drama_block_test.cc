#include <cmath>
#include <limits>

#include "drama_test_utils.h"
#include "layers/drama/drama_block.h"

namespace {

  const std::string block_scope = "backbone/layers_0";

  class TinyBlockModel : public models::Model {
  public:
    explicit TinyBlockModel(bool attention) {
      put("input_layernorm/weight", {2}, {.1f, -.2f});
      put("post_attention_layernorm/weight", {2}, {-.1f, .3f});
      put("mlp/router/weight", {2, 2}, {1, 0, 0, 1});
      put("mlp/shared_expert_gate/weight", {1, 2}, {.2f, -.1f});
      for (const auto* expert : {"experts_0", "experts_1", "shared_expert"}) {
        const std::string name = std::string("mlp/") + expert;
        put(name + "/gate_proj/weight", {1, 2}, {.4f, -.2f});
        put(name + "/up_proj/weight", {1, 2}, {.2f, .5f});
        put(name + "/down_proj/weight", {2, 1}, {.5f, -.3f});
      }
      drama_test::add_mixer(
          [&](const std::string& name, Shape shape, std::vector<float> values) {
            put(name, std::move(shape), std::move(values));
          },
          attention);
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyBlockModel>(*this);
    }

  private:
    void put(const std::string& path, Shape shape, std::vector<float> values) {
      register_variable(block_scope + "/" + path, StorageView(std::move(shape), values));
    }
  };

  StorageView independent_norm(const StorageView& input, const std::vector<double>& gamma) {
    StorageView result(input.shape(), DataType::FLOAT32);
    for (dim_t t = 0; t < input.dim(0); ++t) {
      const double a = input.data<float>()[2 * t], b = input.data<float>()[2 * t + 1];
      const double rms = std::sqrt((a * a + b * b) / 2 + 1e-5);
      result.data<float>()[2 * t] = float(a / rms * (1 + gamma[0]));
      result.data<float>()[2 * t + 1] = float(b / rms * (1 + gamma[1]));
    }
    return result;
  }

}

TEST(DramaBlockTest, IndependentCompositionHasTwoPrenormResiduals) {
  for (const bool attention : {false, true}) {
    TinyBlockModel model(attention);
    const auto cfg = drama_test::block_config(attention);
    layers::DramaBlock block(model, block_scope, cfg);
    layers::DramaBlockState state;
    block.initialize_state(state, 3);
    const auto input = drama_test::inputs();
    StorageView actual;
    block(input, state, actual);
    const auto normalized = independent_norm(input, {.1, -.2});
    StorageView mixed;
    if (attention) {
      layers::GatedSelfAttention mixer(model, block_scope + "/self_attn", cfg.attention);
      layers::GatedSelfAttentionCache cache{StorageView({3, 1, 2}, DataType::FLOAT32),
                                            StorageView({3, 1, 2}, DataType::FLOAT32), 0};
      mixer(normalized, cache, mixed);
    } else {
      layers::GatedDeltaNet mixer(model, block_scope + "/linear_attn", 1, 1, 1, 1, 1e-5f);
      StorageView recurrent, history;
      mixer.initialize_state(recurrent, history);
      mixer(normalized, recurrent, history, mixed);
    }
    for (dim_t i = 0; i < mixed.size(); ++i)
      mixed.data<float>()[i] += input.data<float>()[i];
    const auto post = independent_norm(mixed, {-.1, .3});
    layers::SparseMoE moe(model, block_scope + "/mlp", 2, 1);
    StorageView expert;
    moe(post, expert);
    for (dim_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual.data<float>()[i], mixed.data<float>()[i] + expert.data<float>()[i], 1e-5);
  }
}

TEST(DramaBlockTest, SingleStepsAndResetMatchPrefillForBothMixers) {
  for (const bool attention : {false, true}) {
    TinyBlockModel model(attention);
    layers::DramaBlock block(model, block_scope, drama_test::block_config(attention));
    layers::DramaBlockState full_state, step_state;
    block.initialize_state(full_state, 3);
    block.initialize_state(step_state, 3);
    auto input = drama_test::inputs();
    StorageView full;
    block(input, full_state, full);
    for (dim_t t = 0; t < 3; ++t) {
      StorageView row({1, 2}, input.data<float>() + t * 2), output;
      block(row, step_state, output);
      for (dim_t d = 0; d < 2; ++d)
        EXPECT_NEAR(output.data<float>()[d], full.data<float>()[t * 2 + d], 1e-6);
    }
    if (!attention) {
      expect_storage_eq(full_state.recurrent, step_state.recurrent);
      expect_storage_eq(full_state.history, step_state.history);
    } else {
      expect_storage_eq(full_state.attention.keys, step_state.attention.keys);
      expect_storage_eq(full_state.attention.values, step_state.attention.values);
    }
    block.initialize_state(step_state, 3);
    StorageView reset_output;
    block(input, step_state, reset_output);
    expect_storage_eq(reset_output, full);
  }
}

TEST(DramaBlockTest, InputAndStateGuards) {
  TinyBlockModel model(false);
  layers::DramaBlock block(model, block_scope, drama_test::block_config(false));
  layers::DramaBlockState state;
  block.initialize_state(state);
  StorageView wrong({1, 3}, 1.f), output;
  EXPECT_THROW(block(wrong, state, output), std::invalid_argument);
  auto input = drama_test::inputs();
  EXPECT_THROW(block(input, state, input), std::invalid_argument);
  EXPECT_THROW(block(input, state, state.history), std::invalid_argument);
  input.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(block(input, state, output), std::invalid_argument);
  TinyBlockModel attention_model(true);
  layers::DramaBlock attention(attention_model, block_scope, drama_test::block_config(true));
  EXPECT_THROW(attention.initialize_state(state, 0), std::invalid_argument);
}
