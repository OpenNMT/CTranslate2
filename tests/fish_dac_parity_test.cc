#include <cstdlib>
#include <fstream>
#include <cmath>

#include "test_utils.h"
#include "layers/drama/fish_dac.h"
#include "ctranslate2/models/model_factory.h"

namespace {

  class CodecParityFixture : public models::Model {
  protected:
    bool is_quantizable(const std::string&) const override {
      return false;
    }

    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<CodecParityFixture>(*this);
    }
  };

  std::vector<float> read_float_fixture(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("Cannot read parity fixture " + path);
    std::vector<float> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    if (file.gcount() != static_cast<std::streamsize>(count * sizeof(float))
        || file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Unexpected parity fixture size " + path);
    return values;
  }

}

TEST(FishDACParityTest, OfficialWaveformWeightsMatchIndependentTorchGraph) {
  const char* model_directory = std::getenv("CT2_DRAMA3_CODEC_FIXTURE");
  const char* data_directory = std::getenv("CT2_DRAMA3_WAVEFORM_FIXTURE");
  if (!model_directory || !data_directory)
    GTEST_SKIP() << "Requires converted official codec and independent oracle fixture";
  models::register_model<CodecParityFixture>("Drama3CodecFixture");
  const auto model = models::Model::load(model_directory, Device::CPU, 0, ComputeType::FLOAT32);
  layers::FishDACWaveformDecoder decoder(*model, "codec/decoder");
  StorageView input({1, 1024, 4},
                    read_float_fixture(std::string(data_directory) + "/input.f32", 4096));
  const auto expected = read_float_fixture(std::string(data_directory) + "/expected.f32", 2048);
  StorageView actual;
  decoder(input, actual);
  ASSERT_EQ(actual.shape(), (Shape{1, 1, 2048}));
  double squared_error = 0;
  float maximum_error = 0;
  for (dim_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.data<float>()[i]));
    const float error = std::abs(actual.data<float>()[i] - expected[i]);
    maximum_error = std::max(maximum_error, error);
    squared_error += error * error;
  }
  RecordProperty("max_absolute_error", std::to_string(maximum_error));
  RecordProperty("rmse", std::to_string(std::sqrt(squared_error / actual.size())));
  EXPECT_LT(maximum_error, 1e-3f);
  EXPECT_LT(std::sqrt(squared_error / actual.size()), 2e-4);
}
