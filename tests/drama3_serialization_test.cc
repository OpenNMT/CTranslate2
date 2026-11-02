#include <cstdlib>

#include "test_utils.h"
#include "ctranslate2/models/model_factory.h"

namespace {

  // Test-only registration: this is a serialization fixture, not a TTS model.
  class Drama3SerializationFixture : public models::Model {
  protected:
    bool is_packable(const std::string&) const override {
      return false;
    }

    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<Drama3SerializationFixture>(*this);
    }
  };

}

TEST(Drama3SerializationTest, PythonModelSpecNativeReader) {
  const char* directory = std::getenv("CT2_DRAMA3_SERIALIZATION_FIXTURE");
  if (!directory)
    GTEST_SKIP() << "Set CT2_DRAMA3_SERIALIZATION_FIXTURE to a standard serialization fixture";
  models::register_model<Drama3SerializationFixture>("Drama3SerializationFixture");
  const auto model = models::Model::load(directory, Device::CPU, 0, ComputeType::INT8_FLOAT32);
  const auto& weight = model->get_variable("projection/weight");
  EXPECT_EQ(weight.dtype(), DataType::INT8);
  ASSERT_EQ(weight.shape(), (Shape{3, 2}));
  const std::vector<int8_t> expected{-3, 2, 0, 7, 8, -9};
  for (dim_t i = 0; i < weight.size(); ++i)
    EXPECT_EQ(weight.data<int8_t>()[i], expected[i]);
  const auto& scale = model->get_variable("projection/weight_scale");
  ASSERT_EQ(scale.shape(), (Shape{3}));
  EXPECT_EQ(scale.data<float>()[0], 1.f);
  EXPECT_EQ(scale.data<float>()[1], 2.f);
  EXPECT_EQ(scale.data<float>()[2], 4.f);
  EXPECT_EQ(&weight, &model->get_variable("tied/weight"));
  EXPECT_EQ(&scale, &model->get_variable("tied/weight_scale"));
  const auto& marker = model->get_variable("marker");
  ASSERT_EQ(marker.dtype(), DataType::INT32);
  ASSERT_EQ(marker.shape(), (Shape{2}));
  EXPECT_EQ(marker.data<int32_t>()[0], 17);
  EXPECT_EQ(marker.data<int32_t>()[1], -8);
}
