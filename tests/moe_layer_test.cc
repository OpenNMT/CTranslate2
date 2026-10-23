#include <algorithm>
#include <cmath>
#include <limits>

#include "test_utils.h"
#include "layers/drama/moe.h"
#include "cpu/backend.h"

namespace {

  class TinyMoEModel : public models::Model {
  public:
    explicit TinyMoEModel(float unused_expert_scale = 100.f) {
      register_variable("moe/router/weight", StorageView({3, 1}, std::vector<float>{0, 1, -1}));
      register_variable("moe/shared_expert_gate/weight", StorageView({1, 1}, 0.f));
      add_expert("moe/shared_expert", 6.f);
      add_expert("moe/experts_0", 2.f);
      add_expert("moe/experts_1", 4.f);
      add_expert("moe/experts_2", unused_expert_scale);
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }

    void replace(const std::string& name, StorageView value) {
      remove_variable(name);
      register_variable(name, std::move(value));
    }

    void install_explicit_int8_weights(dim_t width = 1) {
      const auto variables = get_variables();
      for (const auto& pair : variables) {
        if (pair.first.size() < 7 || pair.first.substr(pair.first.size() - 7) != "/weight")
          continue;
        const auto& original = pair.second;
        const dim_t rows = pair.first.find("_proj/") != std::string::npos ? width : original.dim(0);
        StorageView quantized({rows, width}, int8_t(0));
        // These fixture weights are small integers. Scale=1 means the stored
        // INT8 value is exactly the original coefficient, independent of CT2's
        // weight-quantization implementation.
        for (dim_t i = 0; i < original.size(); ++i)
          quantized.data<int8_t>()[i * width] = static_cast<int8_t>(original.data<float>()[i]);
        auto compensation = ops::Gemm::compensate_u8_input(quantized, true, quantized.dim(1),
                                                           quantized.dim(0), 1.f);
        replace(pair.first, std::move(quantized));
        register_variable(pair.first + "_scale", StorageView({rows}, 1.f));
        register_variable(pair.first + "_compensation", std::move(compensation));
      }
      set_compute_type(ComputeType::INT8_FLOAT32, Device::CPU, 0, false);
    }

  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyMoEModel>(*this);
    }

  private:
    void add_expert(const std::string& scope, float scale) {
      register_variable(scope + "/gate_proj/weight", StorageView({1, 1}, 1.f));
      register_variable(scope + "/up_proj/weight", StorageView({1, 1}, 1.f));
      register_variable(scope + "/down_proj/weight", StorageView({1, 1}, scale));
    }
  };

}

TEST(SparseMoETest, GpuExpertMatchesProjectionByProjectionReference) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";

  class GpuFixture : public models::Model {
  public:
    layers::DramaExecutionOptions execution;

    GpuFixture(ComputeType type, int index) {
      auto add = [&](const std::string& scope) {
        StorageView weight({16, 16}, DataType::FLOAT32);
        for (dim_t i = 0; i < weight.size(); ++i)
          weight.data<float>()[i] = .1f * std::sin(float(i + scope.size()));
        register_variable(scope + "/weight", std::move(weight));
      };
      add("moe/router");
      register_variable("moe/shared_expert_gate/weight", StorageView({1, 16}, .01f));
      for (int e = -1; e < 16; ++e) {
        const auto scope = e == -1 ? "moe/shared_expert" : "moe/experts_" + std::to_string(e);
        for (const auto* projection : {"/gate_proj", "/up_proj", "/down_proj"})
          add(scope + projection);
      }
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
      execution.device = Device::CUDA;
      execution.compute_type = type;
      execution.device_index = index;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<GpuFixture>(*this);
    }
  };

  for (int index = 0; index < get_device_count(Device::CUDA); ++index) {
    const ScopedDeviceSetter setter(Device::CUDA, index);
    for (auto type :
         {ComputeType::FLOAT32, ComputeType::FLOAT16, ComputeType::BFLOAT16,
          ComputeType::INT8_FLOAT32, ComputeType::INT8_FLOAT16, ComputeType::INT8_BFLOAT16}) {
      try {
        resolve_compute_type(type, ComputeType::FLOAT32, Device::CUDA, index);
      } catch (const std::invalid_argument&) {
        continue;
      }
      SCOPED_TRACE(compute_type_to_str(type));
      GpuFixture model(type, index);
      layers::SparseMoE layer(model, "moe", 16, 2, model.execution);
      std::unique_ptr<layers::SparseMoE> cached;
      const auto weight_type = compute_type_to_data_type(type).first;
      const size_t cache_bytes = weight_type == DataType::INT8
                                     ? 2880
                                     : 3 * 3 * 256 * (weight_type == DataType::FLOAT32 ? 4 : 2);
      model.execution.expert_cache_bytes = cache_bytes;
      cached = std::make_unique<layers::SparseMoE>(model, "moe", 16, 2, model.execution);
      layers::DramaProjection router(model, "moe/router", model.execution),
          shared_gate(model, "moe/shared_expert_gate", model.execution);
      auto expert_reference = [&](const std::string& scope, const StorageView& row) {
        layers::DramaProjection gate(model, scope + "/gate_proj", model.execution),
            up(model, scope + "/up_proj", model.execution),
            down(model, scope + "/down_proj", model.execution);
        StorageView gated, values, activated, output;
        gate(row, gated);
        up(row, values);
        ops::get_activation_op(ops::ActivationType::Swish)(gated, activated);
        ops::Mul()(activated, values, activated);
        down(activated, output);
        return output;
      };
      for (dim_t rows : {1, 3, 1}) {
        StorageView input({rows, 16}, DataType::FLOAT32), output;
        for (dim_t i = 0; i < input.size(); ++i)
          input.data<float>()[i] = std::sin(float(i + 1));
        layer(input, output);
        StorageView logits, indices(DataType::INT32), weights, reference(input.shape(), 0.f);
        router(input, logits);
        ops::MoeTopK(2)(logits, indices, weights);
        for (dim_t r = 0; r < rows; ++r) {
          StorageView row({1, 16}, input.data<float>() + r * 16);
          for (dim_t i = 0; i < 2; ++i) {
            auto result = expert_reference(
                "moe/experts_" + std::to_string(indices.data<int32_t>()[r * 2 + i]), row);
            for (dim_t d = 0; d < 16; ++d)
              reference.data<float>()[r * 16 + d] +=
                  weights.data<float>()[r * 2 + i] * result.data<float>()[d];
          }
          auto shared = expert_reference("moe/shared_expert", row);
          StorageView gate;
          shared_gate(row, gate);
          const float probability = 1.f / (1.f + std::exp(-gate.data<float>()[0]));
          for (dim_t d = 0; d < 16; ++d)
            reference.data<float>()[r * 16 + d] += probability * shared.data<float>()[d];
        }
        expect_storage_eq(output, reference, type == ComputeType::FLOAT32 ? 1e-5f : 5e-4f);
        StorageView device_input = input.to(Device::CUDA);
        StorageView device_output(DataType::FLOAT32, Device::CUDA);
        layer(device_input, device_output);
        auto resident = device_output.to(Device::CPU);
        synchronize_stream(Device::CUDA);
        expect_storage_eq(resident, reference, type == ComputeType::FLOAT32 ? 1e-5f : 5e-4f);
        if (cached) {
          (*cached)(device_input, device_output);
          auto result = device_output.to(Device::CPU);
          synchronize_stream(Device::CUDA);
          expect_storage_eq(result, resident, 1e-6f);
          EXPECT_EQ(cached->gpu_expert_cache_bytes(), cache_bytes);
        }
      }
    }
  }
}

TEST(SparseMoETest, BoundedGpuCacheEvictsAndReloadsWithoutChangingResults) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  for (int device = 0; device < get_device_count(Device::CUDA); ++device) {
    const ScopedDeviceSetter setter(Device::CUDA, device);
    TinyMoEModel model;
    model.install_explicit_int8_weights(4);  // cuBLAS INT8 requires aligned input widths.
    layers::DramaExecutionOptions execution;
    execution.device = Device::CUDA;
    execution.compute_type = ComputeType::INT8_FLOAT32;
    execution.device_index = device;
    layers::SparseMoE baseline(model, "moe", 3, 1, execution);
    // One 4x4 INT8 matrix and four float scales per projection. Two slots
    // fit the shared expert and exactly one routed expert, forcing eviction.
    execution.expert_cache_bytes = 192;
    layers::SparseMoE cached(model, "moe", 3, 1, execution);
    size_t expected_misses = 0;
    for (float value : {1.f, 1.f, -1.f, -1.f, 2.f}) {
      SCOPED_TRACE(value);
      auto input = StorageView({1, 4}, std::vector<float>{value, 0, 0, 0}).to(Device::CUDA);
      StorageView expected(DataType::FLOAT32, Device::CUDA),
          actual(DataType::FLOAT32, Device::CUDA);
      baseline(input, expected);
      cached(input, actual);
      auto reference = expected.to(Device::CPU), result = actual.to(Device::CPU);
      synchronize_stream(Device::CUDA);
      expect_storage_eq(result, reference, 1e-6f);
      EXPECT_EQ(cached.gpu_expert_cache_bytes(), 192);
      if (!expected_misses)
        expected_misses = 2;
      else if (value == -1.f && expected_misses == 2)
        ++expected_misses;
      else if (value == 2.f)
        ++expected_misses;
      EXPECT_EQ(cached.gpu_expert_cache_misses(), expected_misses);
    }
    // Exercise several routing decisions in one prefill and the CPU-input path.
    StorageView input({4, 4}, std::vector<float>{1, 0, 0, 0, -1, 0, 0, 0,
                                               2, 0, 0, 0, -2, 0, 0, 0}), expected, actual;
    baseline(input, expected);
    cached(input, actual);
    expect_storage_eq(actual, expected, 1e-6f);
    EXPECT_LE(cached.gpu_expert_cache_bytes(), 192);
    EXPECT_EQ(baseline.gpu_expert_cache_bytes(), 0);
    EXPECT_EQ(baseline.gpu_expert_cache_misses(), 0);
    execution.expert_cache_bytes = 191;
    EXPECT_THROW((layers::SparseMoE(model, "moe", 3, 1, execution)), std::invalid_argument);
    execution.compute_type = ComputeType::FLOAT32;
    execution.expert_cache_bytes = 383;
    EXPECT_THROW((layers::SparseMoE(model, "moe", 3, 1, execution)), std::invalid_argument);
  }
}

TEST(SparseMoETest, IndependentScalarEquation) {
  TinyMoEModel model;
  layers::SparseMoE layer(model, "moe", 3, 2);
  EXPECT_EQ(layer.output_type(), DataType::FLOAT32);
  EXPECT_EQ(layer.output_size(), 1);
  StorageView input({3, 1}, std::vector<float>{1, -1, .5f}), output;
  layer(input, output);
  ASSERT_EQ(output.shape(), input.shape());
  for (dim_t r = 0; r < 3; ++r) {
    const double x = input.data<float>()[r];
    const double selected = x > 0 ? 4 : 100;
    const double probability = std::exp(std::abs(x)) / (1 + std::exp(std::abs(x)));
    const double ffn = x * x / (1 + std::exp(-x));
    const double expected = ffn * (probability * selected + (1 - probability) * 2 + .5 * 6);
    EXPECT_NEAR(output.data<float>()[r], expected, 1e-5);
  }
}

TEST(SparseMoETest, UnselectedExpertDoesNotAffectOutput) {
  TinyMoEModel model_a(100.f), model_b(-10000.f);
  layers::SparseMoE layer_a(model_a, "moe", 3, 1), layer_b(model_b, "moe", 3, 1);
  StorageView input({2, 1}, std::vector<float>{1, 2}), a, b;
  layer_a(input, a);
  layer_b(input, b);
  expect_storage_eq(a, b);
}

TEST(SparseMoETest, SharedGateIsInputDependent) {
  TinyMoEModel model;
  model.replace("moe/shared_expert_gate/weight", StorageView({1, 1}, 2.f));
  layers::SparseMoE layer(model, "moe", 3, 1);
  StorageView input({1, 1}, 1.f), output;
  layer(input, output);
  const double ffn = 1 / (1 + std::exp(-1.));
  EXPECT_NEAR(output.data<float>()[0], ffn * (4 + 6 / (1 + std::exp(-2.))), 1e-6);
}

TEST(SparseMoETest, EmptyRows) {
  TinyMoEModel model;
  layers::SparseMoE layer(model, "moe", 3, 2);
  StorageView input({0, 1}, DataType::FLOAT32), output;
  layer(input, output);
  EXPECT_EQ(output.shape(), input.shape());
}

TEST(SparseMoETest, ConstructorGuards) {
  TinyMoEModel model;
  EXPECT_THROW((layers::SparseMoE(model, "moe", 3, 0)), std::invalid_argument);
  EXPECT_THROW((layers::SparseMoE(model, "moe", 3, 4)), std::invalid_argument);
  EXPECT_THROW((layers::SparseMoE(model, "moe", 2, 1)), std::invalid_argument);
  model.replace("moe/experts_0/up_proj/weight", StorageView({2, 1}, 1.f));
  EXPECT_THROW((layers::SparseMoE(model, "moe", 3, 1)), std::invalid_argument);
}

TEST(SparseMoETest, InputAndAliasingGuards) {
  TinyMoEModel model;
  layers::SparseMoE layer(model, "moe", 3, 1);
  StorageView rank3({1, 1, 1}, 1.f), wrong_width({1, 2}, 1.f), integers({1, 1}, int32_t(1)), output;
  EXPECT_THROW(layer(rank3, output), std::invalid_argument);
  EXPECT_THROW(layer(wrong_width, output), std::invalid_argument);
  EXPECT_THROW(layer(integers, output), std::invalid_argument);
  StorageView input({1, 1}, 1.f);
  EXPECT_THROW(layer(input, input), std::invalid_argument);
  StorageView allocation({8}, 0.f);
  StorageView view({1, 1}, allocation.data<float>() + 3);
  allocation.clear();
  EXPECT_THROW(layer(view, allocation), std::invalid_argument);
  input.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(layer(input, output), std::invalid_argument);
}

TEST(SparseMoETest, ExplicitInt8WeightsIndependentQuantizedReference) {
  if (!mayiuse_int8(Device::CPU))
    GTEST_SKIP() << "CPU INT8 GEMM is unavailable in this build";
  TinyMoEModel model;
  model.install_explicit_int8_weights();
  ASSERT_EQ(model.get_variable("moe/experts_1/down_proj/weight").dtype(), DataType::INT8);
  layers::SparseMoE layer(model, "moe", 3, 2);
  StorageView input({4, 1}, std::vector<float>{1, -1, .37f, -2.1f}), output;
  layer(input, output);
  // An independent scalar simulation of each dynamic-INT8 Dense invocation.
  // This fixture Model has binary_version=0, so activation quantization uses
  // truncation, not rounding. Scale=127/maxabs and the explicit weight scale=1.
  const bool unsigned_activation = cpu::prefer_u8s8s32_gemm();
  auto linear = [unsigned_activation](float value, int32_t weight) {
    const float scale = value == 0.f ? 1.f : 127.f / std::abs(value);
    // Legacy U8 quantization offsets before truncation, then compensates the
    // integer dot product. Its boundary behavior differs from signed casting.
    const int32_t quantized = unsigned_activation
                                  ? static_cast<int32_t>(value * scale + 128.f) - 128
                                  : static_cast<int32_t>(value * scale);
    return static_cast<float>(quantized * weight) * (1.f / scale);
  };
  auto expert = [&](float value, int32_t down) {
    const float gate = linear(value, 1);
    const float up = linear(value, 1);
    const float activation = (gate / (1.f + std::exp(-gate))) * up;
    return linear(activation, down);
  };
  for (dim_t r = 0; r < input.dim(0); ++r) {
    const float value = input.data<float>()[r];
    const double router_max = std::abs(linear(value, 1));
    const float probability = static_cast<float>(1. / (1. + std::exp(-router_max)));
    const int32_t chosen_down = value > 0 ? 4 : 100;
    const float expected = probability * expert(value, chosen_down)
                           + (1.f - probability) * expert(value, 2) + .5f * expert(value, 6);
    EXPECT_NEAR(output.data<float>()[r], expected, 1e-4 * std::max(1.f, std::abs(expected)));
  }
}
