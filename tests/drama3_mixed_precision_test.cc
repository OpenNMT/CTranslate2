#include <cstring>
#include "test_utils.h"
#include "ctranslate2/models/drama3.h"
#include "ctranslate2/models/model_reader.h"
#include "layers/drama/drama_projection.h"
#include "ctranslate2/ops/dequantize.h"

namespace {

  void integer(std::string& bytes, uint32_t value, size_t width) {
    for (size_t i = 0; i < width; ++i)
      bytes.push_back(char((value >> (8 * i)) & 255));
  }

  void string(std::string& bytes, const std::string& text) {
    integer(bytes, text.size() + 1, 2);
    bytes += text;
    bytes.push_back('\0');
  }

  template <typename T>
  void tensor(std::string& bytes, const std::string& name, Shape shape, uint8_t type,
              const std::vector<T>& values) {
    string(bytes, name);
    integer(bytes, shape.size(), 1);
    for (auto d : shape)
      integer(bytes, d, 4);
    integer(bytes, type, 1);
    integer(bytes, values.size() * sizeof(T), 4);
    for (T v : values) {
      if constexpr (sizeof(T) == 4) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        integer(bytes, bits, 4);
      } else
        bytes.push_back(char(v));
    }
  }

  models::ModelMemoryReader mixed_fixture(const std::string& policy, bool fp8 = false,
                                          const std::string& format = "e4m3fn_block128") {
    std::string bytes;
    integer(bytes, 6, 4);
    string(bytes, "Drama3Spec");
    integer(bytes, 1, 4);
    integer(bytes, fp8 ? 9 : 8, 4);
    if (fp8) {
      tensor<int8_t>(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight", {2, 2}, 1,
                     {0x38, 0x40, -72, -64});
      tensor<float>(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight_scale_inv", {1, 1}, 0,
                    {2});
      tensor<float>(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/bias", {2}, 0, {.25f, -.5f});
    } else {
      tensor<int8_t>(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight", {2, 2}, 1,
                     {127, -127, 64, -64});
      tensor<float>(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight_scale", {2}, 0,
                    {127, 64});
    }
    tensor<float>(bytes, "backbone/layers_0/mlp/router/weight", {2, 2}, 0,
                  {.125f, -.375f, .5f, .75f});
    tensor<float>(bytes, "backbone/layers_0/mlp/shared_expert/up_proj/weight", {2, 2}, 0,
                  {.125f, -.375f, .5f, .75f});
    tensor<float>(bytes, "backbone/embeddings/weight", {2, 2}, 0, {.125f, -.375f, .5f, .75f});
    tensor<float>(bytes, "tower/heads_0/weight", {2, 2}, 0, {.125f, -.375f, .5f, .75f});
    tensor<float>(bytes, "backbone/norm/weight", {2}, 0, {.25f, -.5f});
    tensor<float>(bytes, "codec/quantizer/upsample/0/1/pwconv1/weight", {2, 2}, 0,
                  {.125f, -.375f, .5f, .75f});
    integer(bytes, 0, 4);
    models::ModelMemoryReader reader("mixed_precision_fixture");
    reader.register_file("model.bin", std::move(bytes));
    nlohmann::json config = {{"model_type", "drama3"},
                             {"ct2_layout_version", 1},
                             {"source_config", nlohmann::json::object()},
                             {"quantization_policy", policy}};
    if (fp8)
      config["fp8_format"] = format;
    reader.register_file("config.json", config.dump());
    return reader;
  }

}

TEST(Drama3MixedPrecisionTest, RejectsLegacyMissingAndUnknownPolicies) {
  for (const auto* policy :
       {"all_matrices_int8", "experts_int8_sensitive_float32", "", "typo_sensitive_float32"}) {
    auto reader = mixed_fixture(policy, true);
    EXPECT_THROW(models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32),
                 std::invalid_argument);
  }
}

TEST(Drama3MixedPrecisionTest, SourceFp8SurvivesLoadingAndUsesFloat32Dense) {
  const std::string scope = "backbone/layers_0/mlp/experts_0/gate_proj";
  for (auto compute : {ComputeType::DEFAULT, ComputeType::FLOAT32}) {
    auto reader = mixed_fixture("experts_fp8_sensitive_float32", true);
    auto model = models::Model::load(reader, Device::CPU, 0, compute);
    EXPECT_EQ(model->effective_compute_type(), ComputeType::FLOAT32);
    expect_storage_eq(model->get_variable(scope + "/weight"),
                      StorageView({2, 2}, std::vector<int8_t>{0x38, 0x40, -72, -64}));
    expect_storage_eq(model->get_variable(scope + "/weight_scale_inv"), StorageView({1, 1}, 2.f));
    for (const auto* suffix : {"_scale", "_compensation", "_packed"})
      EXPECT_EQ(model->get_variable_if_exists(scope + "/weight" + suffix), nullptr);
    EXPECT_EQ(model->get_variable("backbone/layers_0/mlp/router/weight").dtype(),
              DataType::FLOAT32);
    layers::DramaProjection dense(*model, scope);
    StorageView input({1, 2}, std::vector<float>{2, 4}), residual({1, 2}, std::vector<float>{1, 2}),
        output;
    dense(input, output, &residual);
    expect_storage_eq(output, StorageView({1, 2}, std::vector<float>{21.25f, -18.5f}));
    StorageView indices({1}, int32_t(0)), invalid({1, 2}, int8_t(1));
    EXPECT_THROW(dense.select_weights(&indices), std::invalid_argument);
    EXPECT_THROW(dense(invalid, output), std::invalid_argument);
  }
}

TEST(Drama3MixedPrecisionTest, SourceFp8RejectsAmbiguousPolicy) {
  auto malformed = mixed_fixture("experts_fp8_sensitive_float32");
  EXPECT_THROW(models::Model::load(malformed, Device::CPU, 0, ComputeType::FLOAT32),
               std::invalid_argument);
  auto wrong_format = mixed_fixture("experts_fp8_sensitive_float32", true, "e5m2");
  EXPECT_THROW(models::Model::load(wrong_format, Device::CPU, 0, ComputeType::FLOAT32),
               std::invalid_argument);
  auto wrong_policy = mixed_fixture("experts_int8_sensitive_float32", true);
  EXPECT_THROW(models::Model::load(wrong_policy, Device::CPU, 0, ComputeType::FLOAT32),
               std::invalid_argument);
}

TEST(Drama3MixedPrecisionTest, StandardComputeTypesUseBackendResolutionAndDense) {
  for (bool fp8 : {false, true}) {
    for (const auto* compute : {"default", "auto", "float32", "int8", "int8_float32", "int16",
                                "float16", "bfloat16", "int8_float16", "int8_bfloat16"}) {
      SCOPED_TRACE(std::string(compute) + (fp8 ? " FP8" : " standard"));
      auto reader = mixed_fixture(fp8 ? "experts_fp8_sensitive_float32" : "standard", fp8);
      const auto requested = str_to_compute_type(compute);
      ComputeType expected;
      try {
        expected = resolve_compute_type(
            requested, fp8 ? ComputeType::FLOAT32 : ComputeType::INT8_FLOAT32, Device::CPU);
      } catch (const std::invalid_argument&) {
        EXPECT_THROW(models::Model::load(reader, Device::CPU, 0, requested), std::invalid_argument);
        continue;
      }
      const auto model = models::Model::load(reader, Device::CPU, 0, requested);
      EXPECT_EQ(model->effective_compute_type(), expected);
      const std::string scope = "backbone/layers_0/mlp/experts_0/gate_proj";
      layers::DramaProjection dense(*model, scope);
      // Mixed signs avoid the u8*s8 pairwise saturation of pre-VNNI oneDNN
      // kernels. This test checks compute-type resolution, not that limitation.
      StorageView input({1, 2}, std::vector<float>{2, -4}), output;
      dense(input, output);
      const std::vector<float> values =
          fp8 ? std::vector<float>{-11.75f, 11.5f} : std::vector<float>{6, 6};
      expect_storage_eq(output, StorageView({1, 2}, values), .12f);
      EXPECT_EQ(model->get_variable("backbone/norm/weight").dtype(), DataType::FLOAT32);
      EXPECT_EQ(model->get_variable("codec/quantizer/upsample/0/1/pwconv1/weight").dtype(),
                DataType::FLOAT32);
      if (fp8 && expected != ComputeType::FLOAT32)
        EXPECT_EQ(model->get_variable_if_exists(scope + "/weight_scale_inv"), nullptr);
    }
  }
}

TEST(Drama3MixedPrecisionTest, GpuProjectionsMatchCpuWithCachedAndSelectedWeights) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  const std::string scope = "backbone/layers_0/mlp/experts_0/gate_proj";
  for (int index = 0; index < get_device_count(Device::CUDA); ++index) {
    const ScopedDeviceSetter setter(Device::CUDA, index);
    for (bool fp8 : {false, true}) {
      for (auto type :
           {ComputeType::FLOAT32, ComputeType::FLOAT16, ComputeType::BFLOAT16,
            ComputeType::INT8_FLOAT32, ComputeType::INT8_FLOAT16, ComputeType::INT8_BFLOAT16}) {
        try {
          resolve_compute_type(type, ComputeType::FLOAT32, Device::CUDA, index);
        } catch (const std::invalid_argument&) {
          continue;
        }
        SCOPED_TRACE(compute_type_to_str(type) + (fp8 ? " FP8" : " F32"));
        std::string bytes;
        integer(bytes, 6, 4);
        string(bytes, "Drama3Spec");
        integer(bytes, 1, 4);
        integer(bytes, fp8 ? 3 : 2, 4);
        if (fp8) {
          tensor<int8_t>(bytes, scope + "/weight", {16, 16}, 1, std::vector<int8_t>(256, 0x38));
          tensor<float>(bytes, scope + "/weight_scale_inv", {1, 1}, 0, {2.f});
        } else {
          tensor<float>(bytes, scope + "/weight", {16, 16}, 0, std::vector<float>(256, .5f));
        }
        tensor<float>(bytes, scope + "/bias", {16}, 0, std::vector<float>(16, -.25f));
        integer(bytes, 0, 4);
        models::ModelMemoryReader reader("drama3_gpu_projection");
        reader.register_file("model.bin", std::move(bytes));
        reader.register_file("config.json",
                             nlohmann::json{{"model_type", "drama3"},
                                            {"ct2_layout_version", 1},
                                            {"source_config", nlohmann::json::object()},
                                            {"quantization_policy",
                                             fp8 ? "experts_fp8_sensitive_float32" : "standard"},
                                            {"fp8_format", "e4m3fn_block128"}}
                                 .dump());
        auto model = models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32);
        layers::DramaExecutionOptions execution;
        execution.device = Device::CUDA;
        execution.device_index = index;
        execution.compute_type = type;
        execution.device_weights = std::make_shared<layers::DramaDeviceWeightRegistry>();
        layers::DramaProjection dense(*model, scope, execution);
        layers::DramaProjection second(*model, scope, execution);
        EXPECT_EQ(dense.device_weights().weight.buffer(), second.device_weights().weight.buffer());
        for (dim_t rows : {1, 4, 1}) { // Repeat after cache creation with another batch shape.
          StorageView input({rows, 16}, .25f), residual({rows, 16}, .5f), output;
          dense(input, output, &residual);
          expect_storage_eq(output, StorageView({rows, 16}, fp8 ? 8.25f : 2.25f), .08f);
          StorageView gpu_input = input.to(Device::CUDA);
          StorageView gpu_output(DataType::FLOAT32, Device::CUDA);
          dense(gpu_input, gpu_output, &residual);
          auto host_output = gpu_output.to(Device::CPU);
          synchronize_stream(Device::CUDA);
          expect_storage_eq(host_output, output, .0001f);
        }
        if (!fp8) {
          StorageView selected({4}, std::vector<int32_t>{3, 0, 2, 1});
          StorageView extra_bias({4}, .5f), input({1, 16}, .25f), output;
          dense.select_weights(&selected, &extra_bias);
          dense(input, output);
          expect_storage_eq(output, StorageView({1, 4}, 2.25f), .08f);
          dense.select_weights(nullptr);
          dense(input, output);
          expect_storage_eq(output, StorageView({1, 16}, 1.75f), .08f);
        }
      }
    }
  }
}

namespace {

  // Observe the model while conversion is in progress, rather than just its
  // final weights: no previously decoded expert may remain in FP32.
  class ObservedExpertConversion : public models::Drama3Model {
  public:
    using Drama3Model::set_compute_type;
    std::vector<std::string> experts;
    size_t scales_registered = 0;

    void register_variable(std::string name, StorageView value) override {
      if (name.size() >= 6 && name.compare(name.size() - 6, 6, "_scale") == 0) {
        ++scales_registered;
        for (const auto& expert : experts) {
          const auto* weight = get_variable_if_exists(expert);
          ASSERT_NE(weight, nullptr);
          EXPECT_NE(weight->dtype(), DataType::FLOAT32) << expert;
        }
      }
      Drama3Model::register_variable(std::move(name), std::move(value));
    }
  };

}

TEST(Drama3MixedPrecisionTest, ConvertsEachExpertBeforeDecodingTheNext) {
  for (const auto requested : {ComputeType::INT8_FLOAT32, ComputeType::INT16}) {
    ComputeType resolved;
    try {
      resolved = resolve_compute_type(requested, ComputeType::FLOAT32, Device::CPU);
    } catch (const std::invalid_argument&) {
      continue;
    }
    if (resolved == ComputeType::FLOAT32)
      continue;
    SCOPED_TRACE(compute_type_to_str(resolved));
    ObservedExpertConversion model;
    model.config = {{"quantization_policy", "experts_fp8_sensitive_float32"},
                    {"fp8_format", "e4m3fn_block128"}};
    for (size_t i = 0; i < 3; ++i) {
      const std::string name =
          "backbone/layers_0/mlp/experts_" + std::to_string(i) + "/gate_proj/weight";
      model.experts.push_back(name);
      model.register_variable(name, StorageView({2, 2}, std::vector<int8_t>{0x38, 0x40, -72, -64}));
      model.register_variable(name + "_scale_inv", StorageView({1, 1}, float(i + 1)));
    }
    model.set_compute_type(requested, Device::CPU, 0);
    EXPECT_EQ(model.scales_registered, model.experts.size());
    EXPECT_EQ(model.config.at("quantization_policy"), "standard");
    EXPECT_FALSE(model.config.contains("fp8_format"));
    for (size_t i = 0; i < model.experts.size(); ++i) {
      const auto& name = model.experts[i];
      EXPECT_EQ(model.get_variable(name).dtype(), compute_type_to_data_type(resolved).first);
      EXPECT_EQ(model.get_variable_if_exists(name + "_scale_inv"), nullptr);
      StorageView decoded;
      ops::Dequantize()(model.get_variable(name), model.get_variable(name + "_scale"), decoded);
      const float scale = float(i + 1);
      expect_storage_eq(
          decoded, StorageView({2, 2}, std::vector<float>{scale, 2 * scale, -scale, -2 * scale}),
          .05f);
    }
  }
}
