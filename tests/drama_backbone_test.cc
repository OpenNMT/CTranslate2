#include <cmath>

#include "drama_test_utils.h"
#include "layers/drama/drama_backbone.h"

namespace {

  layers::DramaBackboneConfig config(bool hybrid = true) {
    return {{drama_test::block_config(false, 1), drama_test::block_config(hybrid, 1)}, 1e-5f, 8};
  }

  class TinyBackboneModel : public models::Model {
  public:
    explicit TinyBackboneModel(bool hybrid = true) {
      for (dim_t i = 0; i < 2; ++i) {
        const auto path = "backbone/layers_" + std::to_string(i) + "/";
        put(path + "input_layernorm/weight", {2}, {.1f, -.1f});
        put(path + "post_attention_layernorm/weight", {2}, {0, .2f});
        put(path + "mlp/router/weight", {1, 2}, {.1f, -.2f});
        put(path + "mlp/shared_expert_gate/weight", {1, 2}, {0, 0});
        for (const auto* expert : {"experts_0", "shared_expert"}) {
          const auto scope = path + "mlp/" + expert + "/";
          put(scope + "gate_proj/weight", {1, 2}, {.3f, -.2f});
          put(scope + "up_proj/weight", {1, 2}, {-.2f, .1f});
          put(scope + "down_proj/weight", {2, 1}, {.5f, -.1f});
        }
        drama_test::add_mixer(
            [&](const std::string& name, Shape shape, std::vector<float> values) {
              put(path + name, std::move(shape), std::move(values));
            },
            i == 1 && hybrid);
      }
      put("backbone/norm/weight", {2}, {.2f, -.1f});
      put("backbone/lm_head/weight", {3, 2}, {1, 0, 0, 1, .5f, -.25f});
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void install_int8_lm_head() {
      StorageView weight({3, 2}, std::vector<int8_t>{4, 0, 0, 4, 2, -1});
      auto compensation = ops::Gemm::compensate_u8_input(weight, true, 2, 3, 1.f);
      remove_variable("backbone/lm_head/weight");
      register_variable("backbone/lm_head/weight", std::move(weight));
      register_variable("backbone/lm_head/weight_scale", StorageView({3}, 4.f));
      register_variable("backbone/lm_head/weight_compensation", std::move(compensation));
      set_compute_type(ComputeType::INT8_FLOAT32, Device::CPU, 0, false);
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyBackboneModel>(*this);
    }

  private:
    void put(std::string name, Shape shape, std::vector<float> values) {
      register_variable(std::move(name), StorageView(std::move(shape), values));
    }
  };

}

TEST(DramaBackboneTest, HybridCompositionAndIndependentFinalLogits) {
  TinyBackboneModel model;
  const auto cfg = config();
  layers::DramaBackbone backbone(model, "backbone", cfg);
  layers::DramaBackboneState state;
  backbone.initialize_state(state, 3);
  const auto embedded = drama_test::inputs();
  StorageView hidden, logits;
  backbone(embedded, state, hidden, logits);
  EXPECT_EQ(backbone.output_size(), 2);
  EXPECT_EQ(backbone.vocabulary_size(), 3);
  EXPECT_EQ(logits.shape(), (Shape{3, 3}));
  EXPECT_EQ(state.length, 3);
  StorageView composed(embedded);
  for (size_t i = 0; i < 2; ++i) {
    layers::DramaBlock layer(model, "backbone/layers_" + std::to_string(i), cfg.layers[i]);
    layers::DramaBlockState local;
    layer.initialize_state(local, 3);
    StorageView output;
    layer(composed, local, output);
    composed = std::move(output);
  }
  for (dim_t t = 0; t < 3; ++t) {
    const double a = composed.data<float>()[t * 2], b = composed.data<float>()[t * 2 + 1];
    const double rms = std::sqrt((a * a + b * b) / 2 + 1e-5);
    const double x = 1.2 * a / rms, y = .9 * b / rms;
    EXPECT_NEAR(hidden.data<float>()[t * 2], x, 2e-6);
    EXPECT_NEAR(hidden.data<float>()[t * 2 + 1], y, 2e-6);
    EXPECT_NEAR(logits.data<float>()[t * 3], x, 2e-6);
    EXPECT_NEAR(logits.data<float>()[t * 3 + 1], y, 2e-6);
    EXPECT_NEAR(logits.data<float>()[t * 3 + 2], .5 * x - .25 * y, 2e-6);
  }
}

TEST(DramaBackboneTest, HybridChunkParityAndReset) {
  TinyBackboneModel model;
  layers::DramaBackbone backbone(model, "backbone", config());
  layers::DramaBackboneState whole, chunked;
  backbone.initialize_state(whole);
  backbone.initialize_state(chunked);
  auto embedded = drama_test::inputs();
  StorageView full_hidden, full_logits;
  backbone(embedded, whole, full_hidden, full_logits);
  dim_t offset = 0;
  for (const dim_t size : {2, 1}) {
    StorageView chunk({size, 2}, embedded.data<float>() + offset * 2), hidden, logits;
    backbone(chunk, chunked, hidden, logits);
    for (dim_t t = 0; t < size; ++t) {
      for (dim_t d = 0; d < 2; ++d)
        EXPECT_NEAR(hidden.data<float>()[t * 2 + d],
                    full_hidden.data<float>()[(offset + t) * 2 + d], 1e-6);
      for (dim_t v = 0; v < 3; ++v)
        EXPECT_NEAR(logits.data<float>()[t * 3 + v],
                    full_logits.data<float>()[(offset + t) * 3 + v], 1e-6);
    }
    offset += size;
  }
  EXPECT_EQ(chunked.length, whole.length);
  backbone.initialize_state(chunked);
  StorageView reset_hidden, reset_logits;
  backbone(embedded, chunked, reset_hidden, reset_logits);
  expect_storage_eq(reset_hidden, full_hidden);
  expect_storage_eq(reset_logits, full_logits);
}

TEST(DramaBackboneTest, ContextLimitAlsoAppliesToGdnOnlyBackbones) {
  TinyBackboneModel model(false);
  layers::DramaBackbone backbone(model, "backbone", config(false));
  layers::DramaBackboneState state;
  backbone.initialize_state(state, 3);
  auto embedded = drama_test::inputs();
  StorageView hidden, logits;
  backbone(embedded, state, hidden, logits);
  StorageView row({1, 2}, embedded.data<float>());
  EXPECT_THROW(backbone(row, state, hidden, logits), std::invalid_argument);
  EXPECT_EQ(state.length, 3);
  EXPECT_THROW(backbone.initialize_state(state, 9), std::invalid_argument);
}

TEST(DramaBackboneTest, InputStateAndAliasGuards) {
  TinyBackboneModel model;
  layers::DramaBackbone backbone(model, "backbone", config());
  layers::DramaBackboneState state;
  auto embedded = drama_test::inputs();
  StorageView hidden, logits;
  EXPECT_THROW(backbone(embedded, state, hidden, logits), std::invalid_argument);
  backbone.initialize_state(state);
  EXPECT_THROW(backbone(embedded, state, hidden, hidden), std::invalid_argument);
  EXPECT_THROW(backbone(embedded, state, embedded, logits), std::invalid_argument);
  state.layers[1].attention.length = 1;
  EXPECT_THROW(backbone(embedded, state, hidden, logits), std::invalid_argument);
}

TEST(DramaBackboneTest, CandidateHeadMatchesFullProjectionAtRequestedIds) {
  TinyBackboneModel model;
  auto selected_config = config();
  selected_config.lm_candidates = {2, 0}; // Deliberately preserve arbitrary order.
  layers::DramaBackbone full(model, "backbone", config()),
      selected(model, "backbone", selected_config);
  EXPECT_EQ(selected.vocabulary_size(), full.vocabulary_size());
  layers::DramaBackboneState fs, ss;
  full.initialize_state(fs);
  selected.initialize_state(ss);
  const auto input = drama_test::inputs();
  StorageView fh, fl, sh, sl;
  full(input, fs, fh, fl);
  selected(input, ss, sh, sl);
  expect_storage_eq(fh, sh);
  EXPECT_EQ(sl.shape(), fl.shape());
  for (dim_t t = 0; t < input.dim(0); ++t) {
    EXPECT_NEAR(sl.data<float>()[t * 3], fl.data<float>()[t * 3], 1e-6);
    EXPECT_NEAR(sl.data<float>()[t * 3 + 2], fl.data<float>()[t * 3 + 2], 1e-6);
    EXPECT_EQ(sl.data<float>()[t * 3 + 1], 0.f);
  }
}

TEST(DramaBackboneTest, CandidateHeadGathersInt8ScalesAndCompensation) {
  if (!mayiuse_int8(Device::CPU))
    GTEST_SKIP() << "CPU INT8 GEMM unavailable";
  TinyBackboneModel model;
  model.install_int8_lm_head();
  auto cfg = config();
  cfg.lm_candidates = {2, 0};
  layers::DramaBackbone full(model, "backbone", config()), selected(model, "backbone", cfg);
  layers::DramaBackboneState fs, ss;
  full.initialize_state(fs);
  selected.initialize_state(ss);
  StorageView fh, fl, sh, sl;
  const auto input = drama_test::inputs();
  full(input, fs, fh, fl);
  selected(input, ss, sh, sl);
  for (dim_t t = 0; t < input.dim(0); ++t) {
    EXPECT_NEAR(sl.data<float>()[t * 3], fl.data<float>()[t * 3], 1e-6);
    EXPECT_NEAR(sl.data<float>()[t * 3 + 2], fl.data<float>()[t * 3 + 2], 1e-6);
    EXPECT_EQ(sl.data<float>()[t * 3 + 1], 0.f);
  }
}

TEST(DramaBackboneTest, CandidateHeadValidatesUniqueInRangeIds) {
  TinyBackboneModel model;
  for (const std::vector<int32_t> ids : {std::vector<int32_t>{0, 0}, {-1}, {3}}) {
    auto cfg = config();
    cfg.lm_candidates = ids;
    EXPECT_THROW((layers::DramaBackbone(model, "backbone", cfg)), std::invalid_argument);
  }
}

// Compare the experimental device path with the original CPU graph using the
// same GPU INT8 Dense projections. Exercise recurrent state, grouped KV heads,
// rotary positions, selected experts and multi-token prefill followed by decode.
TEST(DramaBackboneTest, ResidentGpuMatchesHybridProjectionReference) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";

  class Fixture : public models::Model {
  public:
    layers::DramaExecutionOptions execution;

    Fixture(bool resident, int device_index, dim_t key_dim, ComputeType type) {
      auto matrix = [&](const std::string& name, dim_t rows, dim_t cols) {
        StorageView weight({rows, cols});
        int salt = 0;
        for (char c : name)
          salt += c;
        for (dim_t i = 0; i < weight.size(); ++i)
          weight.data<float>()[i] = .035f * std::sin(float(i * 3 + salt));
        register_variable(name + "/weight", std::move(weight));
      };
      for (int i = 0; i < 2; ++i) {
        const auto s = "backbone/layers_" + std::to_string(i) + "/";
        register_variable(s + "input_layernorm/weight", StorageView({32}, .1f));
        register_variable(s + "post_attention_layernorm/weight", StorageView({32}, -.05f));
        matrix(s + "mlp/router", 4, 32);
        matrix(s + "mlp/shared_expert_gate", 1, 32);
        for (int e = -1; e < 4; ++e) {
          const auto path = s + "mlp/" + (e < 0 ? "shared_expert" : "experts_" + std::to_string(e));
          matrix(path + "/gate_proj", 32, 32);
          matrix(path + "/up_proj", 32, 32);
          matrix(path + "/down_proj", 32, 32);
        }
        if (!i) {
          matrix(s + "linear_attn/in_proj_qkv", 2 * key_dim + 32, 32);
          matrix(s + "linear_attn/in_proj_z", 32, 32);
          matrix(s + "linear_attn/in_proj_a", 2, 32);
          matrix(s + "linear_attn/in_proj_b", 2, 32);
          matrix(s + "linear_attn/out_proj", 32, 32);
          register_variable(s + "linear_attn/conv1d/weight",
                            StorageView({2 * key_dim + 32, 1, 3}, .4f));
          register_variable(s + "linear_attn/conv1d/bias", StorageView({2 * key_dim + 32}, .01f));
          register_variable(s + "linear_attn/A_log", StorageView({2}, -.3f));
          register_variable(s + "linear_attn/dt_bias", StorageView({2}, .2f));
          register_variable(s + "linear_attn/norm/weight", StorageView({16}, 1.3f));
        } else {
          matrix(s + "self_attn/q_proj", 64, 32);
          matrix(s + "self_attn/k_proj", 16, 32);
          matrix(s + "self_attn/v_proj", 16, 32);
          matrix(s + "self_attn/o_proj", 32, 32);
          register_variable(s + "self_attn/q_norm/weight", StorageView({16}, .1f));
          register_variable(s + "self_attn/k_norm/weight", StorageView({16}, -.1f));
        }
      }
      register_variable("backbone/norm/weight", StorageView({32}, .1f));
      matrix("backbone/lm_head", 32, 32);
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
      execution.device = Device::CUDA;
      execution.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
      execution.compute_type = type;
      execution.device_index = device_index;
      execution.resident_backbone = resident;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<Fixture>(*this);
    }
  };

  for (int index = 0; index < get_device_count(Device::CUDA); ++index)
    for (auto type :
         {ComputeType::FLOAT32, ComputeType::FLOAT16, ComputeType::BFLOAT16,
          ComputeType::INT8_FLOAT32, ComputeType::INT8_FLOAT16, ComputeType::INT8_BFLOAT16}) {
      SCOPED_TRACE(compute_type_to_str(type));
      try {
        resolve_compute_type(type, ComputeType::FLOAT32, Device::CUDA, index);
      } catch (const std::invalid_argument&) {
        continue;
      }
      const ScopedDeviceSetter setter(Device::CUDA, index);
      for (dim_t key_dim : {16, 64, 128})
        for (bool interleaved : {false, true}) {
          SCOPED_TRACE(key_dim);
          Fixture reference_model(false, index, key_dim, type),
              resident_model(true, index, key_dim, type);
          layers::DramaBlockConfig linear, attention;
          linear.num_experts = attention.num_experts = 4;
          linear.top_k = attention.top_k = 2;
          linear.key_heads = 1;
          linear.value_heads = 2;
          linear.key_dim = key_dim;
          linear.value_dim = 16;
          attention.kind = layers::DramaMixerKind::FullAttention;
          attention.attention = {2, 1, 16, 16, 10000.f, 1e-5f, interleaved, true};
          layers::DramaBackboneConfig cfg{{linear, attention}, 1e-5f, 64, {1, 3, 7, 15}};
          layers::DramaBackbone reference(reference_model, "backbone", cfg, reference_model.execution),
              resident(resident_model, "backbone", cfg, resident_model.execution);
          layers::DramaBackboneState rs, gs;
          reference.initialize_state(rs);
          resident.initialize_state(gs);
          for (dim_t steps : {key_dim == 16 && !interleaved ? 33 : 3, 1, 2}) {
            StorageView input({steps, 32});
            for (dim_t i = 0; i < input.size(); ++i)
              input.data<float>()[i] = std::sin(float(i + rs.length));
            StorageView rh, rl, gh, gl;
            reference(input, rs, rh, rl);
            resident(input, gs, gh, gl);
            expect_storage_eq(gh, rh, 2e-3f);
            expect_storage_eq(gl, rl, 2e-3f);
            EXPECT_EQ(rs.length, gs.length);
            auto state = gs.layers[0].recurrent.to(Device::CPU);
            auto history = gs.layers[0].history.to(Device::CPU);
            synchronize_stream(Device::CUDA);
            expect_storage_eq(state, rs.layers[0].recurrent, 2e-4f);
            expect_storage_eq(history, rs.layers[0].history, 2e-4f);
            EXPECT_EQ(gs.layers[1].attention.length, rs.layers[1].attention.length);
          }
          resident.initialize_state(gs);
          EXPECT_EQ(gs.length, 0);
        }
    }
}
