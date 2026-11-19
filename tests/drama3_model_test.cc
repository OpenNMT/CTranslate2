#include <cstring>
#include <cstdlib>
#include <fstream>

#include "test_utils.h"
#include "ctranslate2/models/drama3.h"
#include "ctranslate2/models/model_reader.h"
#include "layers/drama/drama_projection.h"
#include "models/drama3_memory.h"

namespace {

  struct ScopedMemoryBudget {
    const char* name;
    std::string previous;
    bool had_previous;

    explicit ScopedMemoryBudget(const char* value,
                                const char* variable = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB")
      : name(variable) {
      const char* old = std::getenv(name);
      had_previous = old != nullptr;
      if (old)
        previous = old;
#if defined(_WIN32)
      _putenv_s(name, value ? value : "");
#else
      if (value)
        setenv(name, value, 1);
      else
        unsetenv(name);
#endif
    }

    ~ScopedMemoryBudget() {
#if defined(_WIN32)
      _putenv_s(name, had_previous ? previous.c_str() : "");
#else
      if (had_previous)
        setenv(name, previous.c_str(), 1);
      else
        unsetenv(name);
#endif
    }
  };

  void append_integer(std::string& bytes, uint32_t value, size_t width) {
    for (size_t i = 0; i < width; ++i)
      bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
  }

  void append_string(std::string& bytes, const std::string& value) {
    append_integer(bytes, value.size() + 1, 2);
    bytes += value;
    bytes.push_back('\0');
  }

  void append_tensor(std::string& bytes, const std::string& name, Shape shape,
                     std::vector<float> values) {
    append_string(bytes, name);
    append_integer(bytes, shape.size(), 1);
    for (auto dim : shape)
      append_integer(bytes, dim, 4);
    append_integer(bytes, 0, 1); // binary-v6 FLOAT32 dtype id
    append_integer(bytes, values.size() * sizeof(float), 4);
    for (float value : values) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      append_integer(bytes, bits, 4);
    }
  }

  models::ModelMemoryReader fixture(nlohmann::json config = {
                                        {"model_type", "drama3"},
                                        {"ct2_layout_version", 1},
                                        {"source_config", nlohmann::json::object()},
                                        {"quantization_policy", "experts_fp8_sensitive_float32"},
                                        {"fp8_format", "e4m3fn_block128"}}) {
    // Deliberately a model-loading fixture, not a complete inference graph.
    std::string bytes;
    append_integer(bytes, 6, 4);
    append_string(bytes, "Drama3Spec");
    append_integer(bytes, 1, 4);
    append_integer(bytes, 5, 4);
    append_tensor(bytes, "backbone/projection/weight", {2, 2}, {1, -2, 3, -4});
    append_tensor(bytes, "backbone/norm/weight", {2}, {.25f, -.5f});
    append_tensor(bytes, "tower/codebook_embeddings/weight", {2, 2}, {.1f, -.2f, .3f, -.4f});
    // This rank-two name contains 'conv': treating it as a generic quantizable
    // convolution would access dim(2), so protect the codec-specific contract.
    append_tensor(bytes, "codec/quantizer/upsample/0/1/pwconv1/weight", {2, 2},
                  {.125f, -.25f, .5f, -1});
    append_tensor(bytes, "codec/decoder/model/0/conv/weight", {1, 1, 2}, {.2f, -.3f});
    append_integer(bytes, 0, 4); // no aliases
    models::ModelMemoryReader reader("independent_drama3_model_fixture");
    reader.register_file("model.bin", std::move(bytes));
    reader.register_file("config.json", config.dump());
    return reader;
  }

}

TEST(Drama3ModelTest, MemoryBudgetParsesDecimalGbAndRejectsInvalidValues) {
  EXPECT_EQ(parse_drama3_memory_budget(""), 0);
  EXPECT_EQ(parse_drama3_memory_budget("0"), 0);
  EXPECT_EQ(parse_drama3_memory_budget("20"), 20000000000ULL);
  EXPECT_EQ(parse_drama3_memory_budget("0.5"), 500000000ULL);
  for (const char* value : {"-1", "nan", "inf", "twenty", "20GB", "20junk", "1e100", "1e-20"})
    EXPECT_THROW(parse_drama3_memory_budget(value), std::invalid_argument) << value;
  ScopedMemoryBudget unset(nullptr);
  EXPECT_EQ(drama3_gpu_expert_cache_budget(), 0);
}

TEST(Drama3ModelTest, GpuExpertCacheValidatesBudgetAndDeviceBeforeLoading) {
  for (const char* value : {"", "0", "8", "0.5", "-1", "nan", "8GB"}) {
    ScopedMemoryBudget cache(value, "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
    if (!*value || std::string(value) == "0") {
      EXPECT_EQ(drama3_gpu_expert_cache_budget(), 0);
      auto reader = fixture();
      EXPECT_NO_THROW(models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32));
    } else {
      EXPECT_THROW(models::Drama3("missing", 1, "int8", "cpu"), std::invalid_argument);
    }
  }
  ScopedMemoryBudget cache("8", "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
  EXPECT_EQ(drama3_gpu_expert_cache_budget(), 8000000000ULL);
  auto reader = fixture();
  EXPECT_NO_THROW(models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32));
}

#if defined(__linux__) || defined(__APPLE__)
namespace {

  struct MappedDramaFixture {
    std::string directory;

    explicit MappedDramaFixture(bool fp8 = false) {
      char pattern[] = "/tmp/ct2-drama-mmap-XXXXXX";
      const char* path = mkdtemp(pattern);
      if (!path)
        throw std::runtime_error("Unable to create mapped model fixture");
      directory = path;
      std::string bytes;
      append_integer(bytes, 6, 4);
      append_string(bytes, "Drama3Spec");
      append_integer(bytes, 1, 4);
      append_integer(bytes, 3, 4);
      append_string(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight");
      append_integer(bytes, 2, 1);
      append_integer(bytes, 2, 4);
      append_integer(bytes, 2, 4);
      append_integer(bytes, 1, 1); // INT8
      append_integer(bytes, 4, 4);
      const std::vector<int8_t> values = fp8 ? std::vector<int8_t>{56, 64, -72, -64}
                                           : std::vector<int8_t>{-7, 2, 4, -1};
      for (int8_t item : values)
        bytes.push_back(static_cast<char>(item));
      if (fp8)
        append_tensor(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight_scale_inv", {1, 1},
                      {1.f});
      else
        append_tensor(bytes, "backbone/layers_0/mlp/experts_0/gate_proj/weight_scale", {2},
                      {8.f, 16.f});
      append_tensor(bytes, "backbone/norm/weight", {2}, {.25f, -.5f});
      append_integer(bytes, 0, 4);
      std::ofstream(directory + "/model.bin", std::ios::binary).write(bytes.data(), bytes.size());
      const nlohmann::json config = {
          {"model_type", "drama3"},
          {"ct2_layout_version", 1},
          {"source_config", nlohmann::json::object()},
          {"quantization_policy", fp8 ? "experts_fp8_sensitive_float32" : "standard"},
          {"fp8_format", "e4m3fn_block128"},
          {"matrix_dtype", fp8 ? "fp8_float32" : "int8"},
          {"activation_dtype", "float32"}};
      std::ofstream(directory + "/config.json") << config.dump();
    }

    ~MappedDramaFixture() {
      std::remove((directory + "/model.bin").c_str());
      std::remove((directory + "/config.json").c_str());
      std::remove(directory.c_str());
    }
  };

}

TEST(Drama3ModelTest, GpuCachePreservesStandardInt8Storage) {
  MappedDramaFixture files;
  ScopedMemoryBudget cache("8", "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
  auto model = models::Drama3Model::load_checkpoint(files.directory, ComputeType::DEFAULT, true);
  EXPECT_EQ(model->effective_compute_type(), ComputeType::INT8_FLOAT32);
  const auto& weight = model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight");
  EXPECT_EQ(weight.dtype(), DataType::INT8);
  EXPECT_FALSE(weight.owns_data());
}

TEST(Drama3ModelTest, GpuCacheLoadsFloatingPointCheckpointsWithoutQuantizing) {
  MappedDramaFixture files;
  ScopedMemoryBudget cache("8", "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
  for (const auto* policy : {"standard", "experts_fp8_sensitive_float32"}) {
    auto reader = fixture({{"model_type", "drama3"},
                           {"ct2_layout_version", 1},
                           {"source_config", nlohmann::json::object()},
                           {"quantization_policy", policy},
                           {"fp8_format", "e4m3fn_block128"}});
    for (const auto* filename : {"config.json", "model.bin"}) {
      auto input = reader.get_required_file(filename, true);
      std::ofstream(files.directory + "/" + filename, std::ios::binary) << input->rdbuf();
    }
    auto model = models::Drama3Model::load_checkpoint(files.directory, ComputeType::DEFAULT, true);
    EXPECT_EQ(model->effective_compute_type(), ComputeType::FLOAT32);
    expect_storage_eq(model->get_variable("backbone/projection/weight"),
                      StorageView({2, 2}, std::vector<float>{1, -2, 3, -4}));
  }
}

TEST(Drama3ModelTest, MappedExpertsPreserveBytesScalesAndModelCopyLifetime) {
  if (!mayiuse_int8(Device::CPU))
    GTEST_SKIP() << "INT8 unavailable";
  MappedDramaFixture fixture;
  auto model = models::Drama3Model::load_checkpoint(fixture.directory, ComputeType::INT8_FLOAT32, true);
  const auto& weight = model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight");
  EXPECT_FALSE(weight.owns_data());
  const StorageView expected({2, 2}, std::vector<int8_t>{-7, 2, 4, -1});
  expect_storage_eq(weight, expected);
  const auto& scale = model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight_scale");
  EXPECT_TRUE(scale.owns_data());
  expect_storage_eq(scale, StorageView({2}, std::vector<float>{8.f, 16.f}));
  EXPECT_TRUE(model->get_variable("backbone/norm/weight").owns_data());
  auto copy = model->copy_to(Device::CPU, 0);
  model.reset();
  expect_storage_eq(copy->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight"),
                    expected);
}

TEST(Drama3ModelTest, CpuExpertsUseAllocatedStorage) {
  if (!mayiuse_int8(Device::CPU))
    GTEST_SKIP() << "INT8 unavailable";
  MappedDramaFixture files;
  ScopedMemoryBudget cache(nullptr);
  auto model = models::Model::load(files.directory, Device::CPU, 0, ComputeType::INT8_FLOAT32);
  const auto& weight = model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight");
  EXPECT_TRUE(weight.owns_data());
  expect_storage_eq(weight, StorageView({2, 2}, std::vector<int8_t>{-7, 2, 4, -1}));
}

TEST(Drama3ModelTest, MappedExpertsAllowRuntimeConversionAndRejectTruncatedPayloads) {
  ScopedMemoryBudget cache("8");
  MappedDramaFixture files;
  auto model = models::Drama3Model::load_checkpoint(files.directory, ComputeType::FLOAT32, true);
  EXPECT_EQ(model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight").dtype(),
            DataType::FLOAT32);
  EXPECT_TRUE(model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight").owns_data());
  std::ofstream(files.directory + "/model.bin", std::ios::binary | std::ios::trunc) << "short";
  EXPECT_THROW(models::Drama3Model::load_checkpoint(files.directory, ComputeType::INT8, true),
               std::exception);
}

TEST(Drama3ModelTest, MemoryReadersFallBackToAllocatedWeights) {
  auto reader = fixture();
  auto model = models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32);
  EXPECT_TRUE(model->get_variable("backbone/projection/weight").owns_data());
}

TEST(Drama3ModelTest, SourceFp8CpuExpertsUseAllocatedStorage) {
  MappedDramaFixture files(true);
  ScopedMemoryBudget cache(nullptr);
  auto model = models::Model::load(files.directory, Device::CPU, 0, ComputeType::FLOAT32);
  const auto& weight = model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight");
  EXPECT_TRUE(weight.owns_data());
  expect_storage_eq(weight, StorageView({2, 2}, std::vector<int8_t>{56, 64, -72, -64}));
}

TEST(Drama3ModelTest, GpuCacheMapsSourceFp8Experts) {
  MappedDramaFixture files(true);
  ScopedMemoryBudget cache("8");
  auto model = models::Drama3Model::load_checkpoint(files.directory, ComputeType::FLOAT32, true);
  const auto& weight = model->get_variable("backbone/layers_0/mlp/experts_0/gate_proj/weight");
  EXPECT_FALSE(weight.owns_data());
  expect_storage_eq(weight, StorageView({2, 2}, std::vector<int8_t>{56, 64, -72, -64}));
}

TEST(Drama3ModelTest, MappedFp8CpuProjectionMatchesResidentWeights) {
  MappedDramaFixture files(true);
  auto mapped = models::Drama3Model::load_checkpoint(files.directory, ComputeType::FLOAT32, true);
  auto resident = models::Drama3Model::load_checkpoint(files.directory, ComputeType::FLOAT32, false);
  const std::string scope = "backbone/layers_0/mlp/experts_0/gate_proj";
  const auto& weight = mapped->get_variable(scope + "/weight");
  ASSERT_FALSE(weight.owns_data());
  EXPECT_TRUE(resident->get_variable(scope + "/weight").owns_data());
  EXPECT_TRUE(mapped->get_variable(scope + "/weight_scale_inv").owns_data());
  EXPECT_EQ(mapped->effective_compute_type(), ComputeType::FLOAT32);
  layers::DramaProjection mapped_projection(*mapped, scope);
  layers::DramaProjection resident_projection(*resident, scope);
  StorageView input({1, 2}, std::vector<float>{3, 4}), actual, expected;
  for (int i = 0; i < 2; ++i) {
    mapped_projection(input, actual);
    resident_projection(input, expected);
    expect_storage_eq(actual, expected);
    expect_storage_eq(actual, StorageView({1, 2}, std::vector<float>{11, -11}));
    EXPECT_FALSE(weight.owns_data());
  }
}

TEST(Drama3ModelTest, CpuMappingRejectsEagerConversionBeforeLoading) {
  ScopedMemoryBudget mapping("1", "CT2_DRAMA3_CPU_MMAP");
  for (const auto* type : {"int8", "int8_float32", "float16"}) {
    try {
      models::Drama3 model("missing", 1, type);
      FAIL() << "Expected mapped CPU compute type validation";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("CT2_DRAMA3_CPU_MMAP requires"), std::string::npos);
    }
  }
}

TEST(Drama3ModelTest, RemovedProcessBudgetFailsExplicitly) {
  ScopedMemoryBudget budget("12", "CT2_DRAMA3_MAX_MEMORY_GB");
  auto reader = fixture();
  EXPECT_THROW(models::Drama3("missing", 1, "float32"), std::invalid_argument);
  EXPECT_NO_THROW(models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32));
}

#endif

TEST(Drama3ModelTest, SupportsFloat32LoadingWithoutQuantizingCodecOrBackbone) {
  auto reader = fixture();
  const auto model = models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32);
  ASSERT_NE(dynamic_cast<const models::Drama3Model*>(model.get()), nullptr);
  EXPECT_EQ(model->device(), Device::CPU);
  EXPECT_EQ(model->effective_compute_type(), ComputeType::FLOAT32);
  expect_storage_eq(model->get_variable("backbone/projection/weight"),
                    StorageView({2, 2}, std::vector<float>{1, -2, 3, -4}));
  EXPECT_EQ(model->get_variable_if_exists("backbone/projection/weight_scale"), nullptr);
  expect_storage_eq(model->get_variable("backbone/norm/weight"),
                    StorageView({2}, std::vector<float>{.25f, -.5f}));
  expect_storage_eq(model->get_variable("codec/quantizer/upsample/0/1/pwconv1/weight"),
                    StorageView({2, 2}, std::vector<float>{.125f, -.25f, .5f, -1}));
  EXPECT_EQ(model->get_variable("codec/decoder/model/0/conv/weight").dtype(), DataType::FLOAT32);
  for (const auto& variable : model->get_variables())
    EXPECT_EQ(variable.first.find("weight_packed"), std::string::npos) << variable.first;
}

TEST(Drama3ModelTest, RejectsInvalidLayoutAndMissingSourceConfiguration) {
  const std::vector<nlohmann::json> invalid{nlohmann::json::array(),
                                            {{"model_type", "other"},
                                             {"ct2_layout_version", 1},
                                             {"source_config", nlohmann::json::object()}},
                                            {{"model_type", "drama3"},
                                             {"ct2_layout_version", 2},
                                             {"source_config", nlohmann::json::object()}},
                                            {{"model_type", "drama3"}, {"ct2_layout_version", 1}},
                                            {{"model_type", "drama3"},
                                             {"ct2_layout_version", 1},
                                             {"source_config", nlohmann::json::array()}}};
  for (const auto& config : invalid) {
    SCOPED_TRACE(config.dump());
    auto reader = fixture(config);
    EXPECT_THROW(models::Model::load(reader, Device::CPU, 0, ComputeType::FLOAT32), std::exception);
  }
}

TEST(Drama3ModelTest, SupportsInt16WhenAvailable) {
  if (!mayiuse_int16(Device::CPU))
    GTEST_SKIP() << "INT16 unavailable before model-specific validation";
  auto reader = fixture();
  const auto model = models::Model::load(reader, Device::CPU, 0, ComputeType::INT16);
  EXPECT_EQ(model->effective_compute_type(), ComputeType::INT16);
}

TEST(Drama3ModelTest, RejectsInvalidDeviceBeforeLoadingCheckpoint) {
  EXPECT_THROW(models::Drama3("missing", 1, "default", "invalid"), std::invalid_argument);
  EXPECT_THROW(models::Drama3("missing", 1, "default", "cpu", -1), std::invalid_argument);
  EXPECT_THROW(models::Drama3("missing", 1, "default", "cpu", 1), std::invalid_argument);
  EXPECT_THROW(models::Drama3("missing", 1, "default", "cuda", -1), std::invalid_argument);
  EXPECT_THROW(models::Drama3("missing", 1, "default", "cuda", get_device_count(Device::CUDA)),
               std::invalid_argument);
}
