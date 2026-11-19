#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include "test_utils.h"
#include "layers/drama/fish_dac_decoder.h"
#include "ctranslate2/models/model_factory.h"
#include "ctranslate2/utils.h"

namespace {

  class FullCodecParityFixture : public models::Model {
  protected:
    bool is_quantizable(const std::string&) const override {
      return false;
    }

    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<FullCodecParityFixture>(*this);
    }
  };

  template <typename T> std::vector<T> read_raw(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("Cannot read full codec fixture: " + path);
    std::vector<T> result(count);
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (file.gcount() != static_cast<std::streamsize>(count * sizeof(T))
        || file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Wrong raw full codec fixture size: " + path);
    return result;
  }

}

TEST(FishDACFullParityTest, OfficialCompleteDecoderMatchesIndependentTorchFunctionalOracle) {
  const char* model_directory = std::getenv("CT2_DRAMA3_CODEC_FIXTURE");
  const char* fixture_directory = std::getenv("CT2_DRAMA3_FULL_CODEC_FIXTURE");
  if (!model_directory || !fixture_directory)
    GTEST_SKIP() << "Requires official converted codec and full functional oracle fixture";
  if (const char* thread_count = std::getenv("CT2_DRAMA3_CODEC_THREADS")) {
    const std::string value(thread_count);
    size_t consumed = 0;
    const auto threads = std::stoul(value, &consumed);
    ASSERT_EQ(consumed, value.size());
    ASSERT_GT(threads, 0u);
    ASSERT_EQ(value.find_first_not_of("0123456789"), std::string::npos);
    set_num_threads(threads);
    RecordProperty("codec_threads", std::to_string(threads));
  }
  const std::string directory(fixture_directory);
  std::ifstream manifest_file(directory + "/manifest.json");
  ASSERT_TRUE(manifest_file.good());
  const auto manifest = nlohmann::json::parse(manifest_file);
  ASSERT_EQ(manifest.at("schema_version").get<int>(), 1);
  ASSERT_FALSE(manifest.at("cases").empty());
  // The waveform-only parity test can register this same fixture name first;
  // both factories disable quantization and packing and contain no product API.
  models::register_model<FullCodecParityFixture>("Drama3CodecFixture");
  const auto model = models::Model::load(model_directory, Device::CPU, 0, ComputeType::FLOAT32);
  layers::FishDACDecoder decoder(*model, "codec");
  for (const auto& fixture : manifest.at("cases")) {
    const auto frames = fixture.at("frames").get<dim_t>();
    SCOPED_TRACE("codec frames=" + std::to_string(frames));
    ASSERT_GT(frames, 0);
    ASSERT_EQ(fixture.at("input_shape").get<std::vector<dim_t>>(), (Shape{1, 10, frames}));
    ASSERT_EQ(fixture.at("output_shape").get<std::vector<dim_t>>(), (Shape{1, 1, frames * 2048}));
    StorageView codes(
        {1, 10, frames},
        read_raw<int32_t>(directory + "/" + fixture.at("codes").get<std::string>(), 10 * frames));
    const auto expected =
        read_raw<float>(directory + "/" + fixture.at("expected").get<std::string>(), 2048 * frames);
    StorageView actual;
    decoder(codes, actual);
    ASSERT_EQ(actual.shape(), (Shape{1, 1, frames * 2048}));
    double squared_error = 0;
    float maximum_error = 0;
    for (dim_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(expected[i]));
      ASSERT_TRUE(std::isfinite(actual.data<float>()[i]));
      const auto error = std::abs(actual.data<float>()[i] - expected[i]);
      maximum_error = std::max(maximum_error, error);
      squared_error += double(error) * error;
    }
    const double rmse = std::sqrt(squared_error / actual.size());
    RecordProperty("frames_" + std::to_string(frames) + "_max_absolute_error",
                   std::to_string(maximum_error));
    RecordProperty("frames_" + std::to_string(frames) + "_rmse", std::to_string(rmse));
    EXPECT_LT(maximum_error, 1e-3f);
    EXPECT_LT(rmse, 2e-4);
  }
}
