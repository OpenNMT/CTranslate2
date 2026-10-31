#include <cstdlib>
#include <fstream>
#include <limits>

#include "test_utils.h"
#include "layers/drama/fish_dac_encoder.h"
#include "ctranslate2/models/model_factory.h"
#include "ctranslate2/utils.h"

namespace {
  class ReferenceFixture : public models::Model {
  protected:
    bool is_quantizable(const std::string&) const override { return false; }
    bool is_packable(const std::string&) const override { return false; }
    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<ReferenceFixture>(*this);
    }
  };

  template <typename T>
  std::vector<T> read_raw(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    std::vector<T> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!file || file.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Invalid reference fixture: " + path);
    return values;
  }
}

TEST(FishDACReferenceTest, MatchesOfficialEncoderAndRejectsInvalidPCM) {
  const auto* path = std::getenv("CT2_DRAMA3_REFERENCE_FIXTURE");
  if (!path)
    GTEST_SKIP() << "Requires tools/generate_drama3_reference_fixture.py output";
  set_num_threads(8);
  const std::string directory(path);
  models::register_model<ReferenceFixture>("Drama3ReferenceFixture");
  auto model = models::Model::load(directory + "/model", Device::CPU, 0, ComputeType::FLOAT32);
  layers::FishDACReferenceEncoder encoder(*model);
  EXPECT_THROW(encoder(StorageView({0}, DataType::FLOAT32)), std::invalid_argument);
  EXPECT_THROW(encoder(StorageView({1, 8}, 0.f)), std::invalid_argument);
  EXPECT_THROW(encoder(StorageView({8}, DataType::INT32)), std::invalid_argument);
  EXPECT_THROW(encoder(StorageView({8}, std::numeric_limits<float>::quiet_NaN())),
               std::invalid_argument);
  EXPECT_THROW(encoder(StorageView({8}, std::numeric_limits<float>::infinity())),
               std::invalid_argument);
  std::ifstream file(directory + "/manifest.json");
  const auto manifest = nlohmann::json::parse(file);
  for (const auto& item : manifest.at("cases")) {
    const auto samples = item.at("samples").get<dim_t>();
    const auto frames = item.at("frames").get<dim_t>();
    SCOPED_TRACE("samples=" + std::to_string(samples));
    StorageView waveform({samples}, read_raw<float>(
        directory + "/" + item.at("waveform").get<std::string>(), samples));
    const auto expected = read_raw<int32_t>(
        directory + "/" + item.at("codes").get<std::string>(), frames * 10);
    const auto alternate = read_raw<int32_t>(
        directory + "/" + item.at("alternate").get<std::string>(), frames * 10);
    const auto actual = encoder(waveform);
    ASSERT_EQ(actual.shape(), (Shape{frames, 10}));
    for (dim_t i = 0; i < actual.size(); ++i)
      EXPECT_TRUE(actual.data<int32_t>()[i] == expected[i]
                  || actual.data<int32_t>()[i] == alternate[i])
          << "frame/codebook offset " << i << ": " << actual.data<int32_t>()[i]
          << ", oracle " << expected[i] << ", near-tied alternate " << alternate[i];
    const auto again = encoder(waveform);
    EXPECT_EQ(std::vector<int32_t>(actual.data<int32_t>(), actual.data<int32_t>() + actual.size()),
              std::vector<int32_t>(again.data<int32_t>(), again.data<int32_t>() + again.size()));
  }
}
