#include "test_utils.h"
#include "ctranslate2/primitives.h"
#include "dispatch.h"

class PrimitiveTest : public ::testing::TestWithParam<Device> {
};

TEST_P(PrimitiveTest, StridedFill) {
  const Device device = GetParam();
  StorageView x({3, 2}, float(0), device);
  StorageView expected({3, 2}, std::vector<float>{1, 0, 1, 0, 1, 0}, device);
  DEVICE_DISPATCH(device, primitives<D>::strided_fill(x.data<float>(), 1.f, 2, 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, IndexedFill) {
  const Device device = GetParam();
  StorageView x({6}, float(0), device);
  StorageView ids({3}, std::vector<int32_t>{0, 2, 5}, device);
  StorageView expected({6}, std::vector<float>{1, 0, 1, 0, 0, 1}, device);
  DEVICE_DISPATCH(device, primitives<D>::indexed_fill(x.data<float>(), 1.f, ids.data<int32_t>(), 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, LogSumExp) {
  const Device device = GetParam();
  StorageView x({8}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2}, device);
  float result = 0;
  DEVICE_DISPATCH(device, result = primitives<D>::logsumexp(x.data<float>(), x.size()));
  EXPECT_NEAR(result, 2.1908040046691895, 1e-6);
}

TEST_P(PrimitiveTest, PenalizePreviousTokens) {
  const Device device = GetParam();
  const float penalty = 1.2f;
  StorageView scores({2, 4}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2});
  StorageView previous_ids({2, 2}, std::vector<int32_t>{2, 2, 1, 2}, device);
  StorageView previous_scores({2, 2}, std::vector<float>{-1.2, -1.2, 0.5, -1.3}, device);
  StorageView expected = scores;
  expected.at<float>({0, 2}) *= penalty;
  expected.at<float>({1, 1}) /= penalty;
  expected.at<float>({1, 2}) *= penalty;
  scores = scores.to(device);
  DEVICE_DISPATCH(device, primitives<D>::penalize_previous_tokens(scores.data<float>(),
                                                                  previous_scores.data<float>(),
                                                                  previous_ids.data<int32_t>(),
                                                                  penalty,
                                                                  scores.dim(0),
                                                                  previous_ids.dim(1),
                                                                  scores.dim(1)));
  expect_storage_eq(scores, expected);
}

#ifdef CT2_WITH_MPS
TEST(MPSPrimitiveTest, PenalizePreviousTokensFloat16OddLength) {
  const std::vector<float> score_values{
    0.5f, 1.f, -1.5f, 2.f, -2.5f, 3.f, -3.5f,
    -4.f, 4.5f, -5.f, 5.5f, -6.f, 6.5f, -7.f};
  StorageView scores = StorageView({2, 7}, score_values).to_float16().to(Device::MPS);
  const StorageView previous_ids(
    {2, 3}, std::vector<int32_t>{0, 2, 6, 1, 4, 4}, Device::MPS);
  const StorageView previous_scores =
    StorageView({2, 3}, std::vector<float>{0.5f, -1.5f, -3.5f, 4.5f, -6.f, -6.f})
      .to_float16()
      .to(Device::MPS);

  std::vector<float> expected_values = score_values;
  expected_values[0] = 0.25f;
  expected_values[2] = -3.f;
  expected_values[6] = -7.f;
  expected_values[8] = 2.25f;
  expected_values[11] = -12.f;
  const StorageView expected = StorageView({2, 7}, expected_values).to_float16();

  primitives<Device::MPS>::penalize_previous_tokens(scores.data<float16_t>(),
                                                     previous_scores.data<float16_t>(),
                                                     previous_ids.data<int32_t>(),
                                                     float16_t(2.f),
                                                     scores.dim(0),
                                                     previous_ids.dim(1),
                                                     scores.dim(1));
  expect_storage_eq(scores, expected);
}
#endif

INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PrimitiveTest, ::testing::Values(Device::CUDA));
#endif
#ifdef CT2_WITH_MPS
INSTANTIATE_TEST_SUITE_P(MPS, PrimitiveTest, ::testing::Values(Device::MPS));
#endif
