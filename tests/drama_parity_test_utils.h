#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>

#include "test_utils.h"

namespace drama_test {

  inline StorageView read_tensor(const std::string& directory, const std::string& name,
                                 Shape shape) {
    StorageView result(std::move(shape), DataType::FLOAT32);
    std::ifstream stream(directory + "/" + name, std::ios::binary);
    if (!stream)
      throw std::runtime_error("cannot read oracle fixture " + directory + "/" + name);
    const auto bytes = result.size() * sizeof(float);
    stream.read(reinterpret_cast<char*>(result.data<float>()), bytes);
    if (stream.gcount() != std::streamsize(bytes) || stream.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("oracle fixture has incorrect byte count");
    return result;
  }

  inline void compare(const StorageView& actual, const StorageView& expected, float max_tolerance,
                      double rms_tolerance, const std::string& label) {
    ASSERT_EQ(actual.shape(), expected.shape());
    float maximum = 0;
    double squared = 0;
    for (dim_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual.data<float>()[i]));
      ASSERT_TRUE(std::isfinite(expected.data<float>()[i]));
      const float error = std::abs(actual.data<float>()[i] - expected.data<float>()[i]);
      maximum = std::max(maximum, error);
      squared += double(error) * error;
    }
    const double rms = std::sqrt(squared / std::max(dim_t(1), actual.size()));
    ::testing::Test::RecordProperty(label + "_max_absolute_error", std::to_string(maximum));
    ::testing::Test::RecordProperty(label + "_rmse", std::to_string(rms));
    EXPECT_LT(maximum, max_tolerance) << label;
    EXPECT_LT(rms, rms_tolerance) << label;
  }

}
