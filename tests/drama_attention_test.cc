#include <algorithm>
#include <cmath>
#include <limits>

#include "test_utils.h"
#include "layers/drama/gated_self_attention.h"

namespace {

  class TinyAttentionModel : public models::Model {
  public:
    explicit TinyAttentionModel(float gamma = 0.f, float gate_scale = 0.f) {
      StorageView q({16, 4}, 0.f), k({4, 4}, 0.f), v({4, 4}, 0.f), o({4, 8}, 0.f);
      for (dim_t d = 0; d < 4; ++d) {
        q.data<float>()[d * 4 + d] = 1;
        q.data<float>()[(8 + d) * 4 + d] = -1;
        q.data<float>()[(4 + d) * 4 + d] = gate_scale;
        q.data<float>()[(12 + d) * 4 + d] = gate_scale * 2;
        k.data<float>()[d * 4 + d] = 1;
        v.data<float>()[d * 4 + d] = 1;
        o.data<float>()[d * 8 + d] = .25f;
        o.data<float>()[d * 8 + 4 + d] = .75f;
      }
      register_variable("attn/q_proj/weight", std::move(q));
      register_variable("attn/k_proj/weight", std::move(k));
      register_variable("attn/v_proj/weight", std::move(v));
      register_variable("attn/o_proj/weight", std::move(o));
      register_variable("attn/q_norm/weight", StorageView({4}, gamma));
      register_variable("attn/k_norm/weight", StorageView({4}, gamma));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyAttentionModel>(*this);
    }
  };

  layers::GatedSelfAttentionConfig config(bool interleaved = false, dim_t rotary = 4) {
    return {2, 1, 4, rotary, 100.f, 1e-5f, interleaved, true};
  }

  layers::GatedSelfAttentionCache cache(dim_t capacity) {
    return {StorageView({capacity, 1, 4}, DataType::FLOAT32),
            StorageView({capacity, 1, 4}, DataType::FLOAT32), 0};
  }

  StorageView inputs() {
    return StorageView({3, 4}, std::vector<float>{1, 2, -1, .5f, -.5f, 1, 2, -1, 2, -.25f, .5f, 1});
  }

  // Independent double-precision causal attention over dense tiny arrays.
  std::vector<double> reference(const StorageView& x, bool interleaved, dim_t rotary) {
    std::vector<std::vector<double>> vectors(3, std::vector<double>(4));
    for (dim_t t = 0; t < 3; ++t) {
      double square = 0;
      for (dim_t d = 0; d < 4; ++d)
        square += std::pow(double(x.data<float>()[t * 4 + d]), 2);
      for (dim_t d = 0; d < 4; ++d)
        vectors[t][d] = x.data<float>()[t * 4 + d] / std::sqrt(square / 4 + 1e-5);
      const auto normalized = vectors[t];
      for (dim_t p = 0; p < rotary / 2; ++p) {
        const dim_t a = interleaved ? 2 * p : p;
        const dim_t b = interleaved ? 2 * p + 1 : p + rotary / 2;
        const double angle = t / std::pow(100., 2. * p / rotary);
        vectors[t][a] = normalized[a] * std::cos(angle) - normalized[b] * std::sin(angle);
        vectors[t][b] = normalized[a] * std::sin(angle) + normalized[b] * std::cos(angle);
      }
    }
    std::vector<double> output(12, 0);
    for (dim_t t = 0; t < 3; ++t) {
      for (dim_t h = 0; h < 2; ++h) {
        std::vector<double> scores(t + 1);
        double sum = 0;
        for (dim_t s = 0; s <= t; ++s) {
          double dot = 0;
          for (dim_t d = 0; d < 4; ++d)
            dot += vectors[t][d] * vectors[s][d];
          scores[s] = std::exp((h ? -1 : 1) * dot / 2);
          sum += scores[s];
        }
        for (dim_t d = 0; d < 4; ++d)
          for (dim_t s = 0; s <= t; ++s)
            output[t * 4 + d] +=
                .5 * (h ? .75 : .25) * (scores[s] / sum) * x.data<float>()[s * 4 + d];
      }
    }
    return output;
  }

}

TEST(GatedSelfAttentionTest, IndependentGroupedAttentionAndBothRopeLayouts) {
  TinyAttentionModel model;
  const auto x = inputs();
  for (const bool interleaved : {false, true}) {
    for (const dim_t rotary : {0, 2, 4}) {
      layers::GatedSelfAttention layer(model, "attn", config(interleaved, rotary));
      auto state = cache(3);
      StorageView output;
      layer(x, state, output);
      const auto expected = reference(x, interleaved, rotary);
      for (dim_t i = 0; i < output.size(); ++i)
        EXPECT_NEAR(output.data<float>()[i], expected[i], 2e-6);
      EXPECT_EQ(state.length, 3);
    }
  }
}

TEST(GatedSelfAttentionTest, CachedStepsMatchPrefill) {
  TinyAttentionModel model;
  layers::GatedSelfAttention layer(model, "attn", config());
  const auto x = inputs();
  auto full_cache = cache(3), step_cache = cache(3);
  StorageView full;
  layer(x, full_cache, full);
  for (dim_t t = 0; t < 3; ++t) {
    StorageView row({1, 4}, const_cast<float*>(x.data<float>()) + 4 * t), output;
    layer(row, step_cache, output);
    for (dim_t d = 0; d < 4; ++d)
      EXPECT_FLOAT_EQ(output.data<float>()[d], full.data<float>()[t * 4 + d]);
  }
  expect_storage_eq(step_cache.keys, full_cache.keys);
  expect_storage_eq(step_cache.values, full_cache.values);
}

TEST(GatedSelfAttentionTest, PerHeadQueryGatePacking) {
  TinyAttentionModel model(0.f, 1.f);
  layers::GatedSelfAttention layer(model, "attn", config());
  auto state = cache(1);
  StorageView input({1, 4}, std::vector<float>{1, -1, .5f, -2}), output;
  layer(input, state, output);
  for (dim_t d = 0; d < 4; ++d) {
    const double x = input.data<float>()[d];
    const double expected = x * (.25 / (1 + std::exp(-x)) + .75 / (1 + std::exp(-2 * x)));
    EXPECT_NEAR(output.data<float>()[d], expected, 1e-6);
  }
}

TEST(GatedSelfAttentionTest, DirectAndZeroCenteredGammaAgree) {
  TinyAttentionModel zero_centered(0), direct(1);
  auto cfg = config();
  layers::GatedSelfAttention a(zero_centered, "attn", cfg);
  cfg.zero_centered_qk_norm = false;
  layers::GatedSelfAttention b(direct, "attn", cfg);
  auto ac = cache(3), bc = cache(3);
  StorageView ay, by;
  const auto x = inputs();
  a(x, ac, ay);
  b(x, bc, by);
  expect_storage_eq(ay, by);
}

TEST(GatedSelfAttentionTest, CapacityAndShapeGuardsDoNotUpdateCache) {
  TinyAttentionModel model;
  layers::GatedSelfAttention layer(model, "attn", config());
  auto state = cache(2);
  const auto x = inputs();
  StorageView y;
  EXPECT_THROW(layer(x, state, y), std::invalid_argument);
  EXPECT_EQ(state.length, 0);
  StorageView wrong({1, 3}, 1.f);
  EXPECT_THROW(layer(wrong, state, y), std::invalid_argument);
  state.length = -1;
  EXPECT_THROW(layer(x, state, y), std::invalid_argument);
  auto cfg = config();
  cfg.rotary_dim = 3;
  EXPECT_THROW((layers::GatedSelfAttention(model, "attn", cfg)), std::invalid_argument);
}

TEST(GatedSelfAttentionTest, EmptyAndAliasingAndNonfiniteGuards) {
  TinyAttentionModel model;
  layers::GatedSelfAttention layer(model, "attn", config());
  auto state = cache(3);
  StorageView empty({0, 4}, DataType::FLOAT32), output;
  layer(empty, state, output);
  EXPECT_EQ(output.shape(), empty.shape());
  EXPECT_EQ(state.length, 0);
  auto x = inputs();
  EXPECT_THROW(layer(x, state, x), std::invalid_argument);
  StorageView alias({3, 4}, state.keys.data<float>());
  EXPECT_THROW(layer(alias, state, output), std::invalid_argument);
  x.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(layer(x, state, output), std::invalid_argument);
  EXPECT_EQ(state.length, 0);
}

TEST(GatedSelfAttentionTest, MultiTokenContinuationMatchesFullCausalAttention) {
  TinyAttentionModel model;
  const auto x = inputs();
  for (const bool interleaved : {false, true}) {
    for (const dim_t rotary : {0, 2, 4}) {
      layers::GatedSelfAttention layer(model, "attn", config(interleaved, rotary));
      auto full_cache = cache(3), chunk_cache = cache(3);
      StorageView full, output;
      layer(x, full_cache, full);
      StorageView prefix({1, 4}, const_cast<float*>(x.data<float>()));
      layer(prefix, chunk_cache, output);
      StorageView continuation({2, 4}, const_cast<float*>(x.data<float>()) + 4);
      layer(continuation, chunk_cache, output);
      const auto expected = reference(x, interleaved, rotary);
      for (dim_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output.data<float>()[i], full.data<float>()[4 + i], 2e-6);
        EXPECT_NEAR(output.data<float>()[i], expected[4 + i], 2e-6);
      }
      EXPECT_EQ(chunk_cache.length, 3);
      expect_storage_eq(chunk_cache.keys, full_cache.keys);
      expect_storage_eq(chunk_cache.values, full_cache.values);
    }
  }
}

TEST(GatedSelfAttentionTest, ProjectionOverflowDoesNotCommitCacheOrOutput) {
  TinyAttentionModel model;
  // Finite parameters and finite input whose projection overflows.
  auto& weights = const_cast<StorageView&>(model.get_variable("attn/q_proj/weight"));
  weights.data<float>()[0] = std::numeric_limits<float>::max();
  layers::GatedSelfAttention layer(model, "attn", config());
  auto state = cache(3);
  std::fill_n(state.keys.data<float>(), state.keys.size(), 7.f);
  std::fill_n(state.values.data<float>(), state.values.size(), 9.f);
  StorageView input({1, 4}, std::vector<float>{2.f, 1.f, -1.f, .5f});
  StorageView output({1, 4}, 11.f);
  EXPECT_THROW(layer(input, state, output), std::runtime_error);
  EXPECT_EQ(state.length, 0);
  for (dim_t i = 0; i < state.keys.size(); ++i) {
    EXPECT_EQ(state.keys.data<float>()[i], 7.f);
    EXPECT_EQ(state.values.data<float>()[i], 9.f);
  }
  for (dim_t i = 0; i < output.size(); ++i)
    EXPECT_EQ(output.data<float>()[i], 11.f);
}

TEST(DramaAttentionTest, ResidentGpuRotaryCacheCanSwitchDevices) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  TinyAttentionModel model(.1f, .2f);
  layers::DramaExecutionOptions execution;
  execution.device = Device::CUDA;
  execution.compute_type = ComputeType::INT8_FLOAT32;
  const ScopedDeviceSetter setter(Device::CUDA, 0);
  layers::GatedSelfAttention layer(model, "attn", config(), execution);
  auto cpu_cache = cache(8);
  auto input = inputs();
  StorageView reference;
  layer(input, cpu_cache, reference);
  auto gpu_cache = cache(8);
  gpu_cache.keys = gpu_cache.keys.to(Device::CUDA);
  gpu_cache.values = gpu_cache.values.to(Device::CUDA);
  auto gpu_input = input.to(Device::CUDA);
  StorageView result(DataType::FLOAT32, Device::CUDA);
  layer(gpu_input, gpu_cache, result);
  auto host = result.to(Device::CPU);
  synchronize_stream(Device::CUDA);
  expect_storage_eq(host, reference, 2e-3f);
  cpu_cache.length = 0;
  StorageView switched;
  layer(input, cpu_cache, switched);
  expect_storage_eq(switched, reference, 2e-3f);
}
