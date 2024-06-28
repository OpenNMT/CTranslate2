#include <gtest/gtest.h>

std::string g_data_dir;

#ifdef CT2_WITH_CANN
#include "test_utils.h"

class CannTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    cann_test_setup();
  }
  void TearDown() override {
    cann_test_tear_down();
  }
};
#endif

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  if (argc < 2)
    throw std::invalid_argument("missing data directory");
  g_data_dir = argv[1];

#ifdef CT2_WITH_CANN
  ::testing::AddGlobalTestEnvironment(new CannTestEnvironment);
#endif
  return RUN_ALL_TESTS();
}
