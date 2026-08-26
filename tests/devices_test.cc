#include <gtest/gtest.h>

#include "ctranslate2/devices.h"

using namespace ctranslate2;

TEST(DevicesTest, ExplicitCPU) {
  EXPECT_EQ(str_to_device("cpu"), Device::CPU);
  EXPECT_EQ(str_to_device("CPU"), Device::CPU);
}

#ifdef CT2_WITH_MPS
TEST(DevicesTest, MPSIsExplicitOptIn) {
  EXPECT_EQ(str_to_device("auto"), Device::CPU);
  EXPECT_EQ(str_to_device("AUTO"), Device::CPU);
  EXPECT_EQ(str_to_device("mps"), Device::MPS);
  EXPECT_EQ(str_to_device("MPS"), Device::MPS);
}
#endif
