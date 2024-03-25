#pragma once

#include <gtest/gtest.h>
#include <mutex>

#include "ctranslate2/storage_view.h"

#include "type_dispatch.h"
#ifdef CT2_WITH_CANN
#include "cann/utils.h"
#endif

using namespace ctranslate2;

const std::string& get_data_dir();
std::string default_model_dir();

#define ASSERT_RAISES(STMT, EXCEPT)                     \
  do {                                                  \
    try {                                               \
      STMT;                                             \
      FAIL() << "Expected "#EXCEPT" exception";         \
    } catch (EXCEPT&) {                                 \
    } catch (...) {                                     \
      FAIL() << "Expected "#EXCEPT" exception";         \
    }                                                   \
  } while (false)

template <typename T>
void expect_array_eq(const T* x, const T* y, size_t n, T abs_diff = 0) {
  for (size_t i = 0; i < n; ++i) {
    if (abs_diff == 0) {
      EXPECT_EQ(x[i], y[i]) << "Value mismatch at index " << i;
    } else {
      EXPECT_NEAR(x[i], y[i], abs_diff) << "Absolute difference greater than "
                                        << abs_diff << " at index " << i;
    }
  }
}

template<>
inline void expect_array_eq(const float* x, const float* y, size_t n, float abs_diff) {
  for (size_t i = 0; i < n; ++i) {
    if (abs_diff == 0 || !std::isfinite(y[i])) {
      EXPECT_FLOAT_EQ(x[i], y[i]) << "Value mismatch at index " << i;
    } else {
      EXPECT_NEAR(x[i], y[i], abs_diff) << "Absolute difference greater than "
                                        << abs_diff << " at index " << i;
    }
  }
}

template <typename T>
void expect_vector_eq(const std::vector<T>& got, const std::vector<T>& expected, T abs_diff) {
  ASSERT_EQ(got.size(), expected.size());
  expect_array_eq(got.data(), expected.data(), got.size(), abs_diff);
}

template <typename T>
void assert_vector_eq(const std::vector<T>& got, const std::vector<T>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(got[i], expected[i]) << "Value mismatch for dimension " << i;
  }
}

inline void expect_storage_eq(const StorageView& got,
                              const StorageView& expected,
                              float abs_diff = 0) {
  StorageView got_cpu = got.to(Device::CPU);
  StorageView expected_cpu = expected.to(Device::CPU);
  ASSERT_EQ(got.dtype(), expected.dtype());
  assert_vector_eq(got.shape(), expected.shape());
  TYPE_DISPATCH(got.dtype(), expect_array_eq(got_cpu.data<T>(), expected_cpu.data<T>(), got.size(), static_cast<T>(abs_diff)));
}

struct FloatType {
  Device device;
  DataType dtype;
  float error = 0;
};

inline std::string fp_test_name(::testing::TestParamInfo<FloatType> param_info) {
  return dtype_name(param_info.param.dtype);
}
#ifdef CT2_WITH_CANN
#  define GUARD_BFLOAT16_NPU_TEST GTEST_SKIP() << "BFLOAT16 not supported"
#  define GUARD_OPERATOR_NPU_TEST GTEST_SKIP() << "Operator not implemented in CANN"

inline void cann_test_setup() {
  ctranslate2::initialize_device();
  // set_device_index has to always be called at least once before get_device_index
  const auto device_index = 0;
  ctranslate2::set_device_index(Device::CANN, device_index);
}

inline void cann_test_allow_acl_finalize() {
  ctranslate2::cann::AclDeviceEnabler::set_allow_acl_finalize(true);
}

inline void cann_test_disallow_acl_finalize() {
  ctranslate2::cann::AclDeviceEnabler::set_allow_acl_finalize(false);
}

inline void cann_test_tear_down() {
  cann_test_allow_acl_finalize(); // Make sure that acl_finalize can be invoked
  ctranslate2::cann::AclDeviceEnabler::acl_finalize();
}

#else
#  define GUARD_BFLOAT16_NPU_TEST do {} while (0)
#  define GUARD_OPERATOR_NPU_TEST do {} while (0)
#endif

