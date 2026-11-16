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

// The integer primitives run on the host. Building each operand queues a fill on
// the device, so unless they wait for it the fill lands after their result, and
// in the in-place case overwrites it.
TEST(MPSPrimitiveTest, IntegerHostLoopsWaitForPendingWork) {
  using P = primitives<Device::MPS>;
  const auto queued = [](int32_t value, const Shape& shape = {6}) {
    return StorageView(shape, value, Device::MPS);
  };
  const auto check = [](const char* name, const StorageView& got, int32_t expected) {
    SCOPED_TRACE(name);
    expect_storage_eq(got, StorageView(got.shape(), expected, Device::CPU));
  };
  const dim_t size = 6;

  {
    StorageView x = queued(1);
    P::add(int32_t(1), x.data<int32_t>(), x.data<int32_t>(), size);
    check("add scalar in place", x, 2);
  }
  {
    StorageView a = queued(1), b = queued(2), c = queued(0);
    P::add(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), size);
    check("add", c, 3);
  }
  {
    StorageView a = queued(1, {2}), b = queued(2), c = queued(0);
    P::add_batch_broadcast(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), 2, size);
    check("add_batch_broadcast", c, 3);
  }
  {
    StorageView a = queued(1, {2}), b = queued(2), c = queued(0);
    P::add_depth_broadcast(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), 2, size);
    check("add_depth_broadcast", c, 3);
  }
  {
    StorageView a = queued(1, {2}), b = queued(2), c = queued(0);
    P::add_block_broadcast(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), 3, 2, size);
    check("add_block_broadcast", c, 3);
  }
  {
    StorageView a = queued(5), b = queued(2), c = queued(0);
    P::sub(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), size);
    check("sub", c, 3);
  }
  {
    StorageView x = queued(1), y = queued(0);
    P::max(int32_t(3), x.data<int32_t>(), y.data<int32_t>(), size);
    check("max scalar", y, 3);
  }
  {
    StorageView a = queued(1), b = queued(3), c = queued(0);
    P::max(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), size);
    check("max", c, 3);
  }
  {
    StorageView x = queued(3), y = queued(0);
    P::min(int32_t(1), x.data<int32_t>(), y.data<int32_t>(), size);
    check("min scalar", y, 1);
  }
  {
    StorageView a = queued(1), b = queued(3), c = queued(0);
    P::min(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), size);
    check("min", c, 1);
  }
  {
    StorageView x = queued(3), y = queued(0);
    P::mul(int32_t(2), x.data<int32_t>(), y.data<int32_t>(), size);
    check("mul scalar", y, 6);
  }
  {
    StorageView a = queued(2), b = queued(3), c = queued(0);
    P::mul(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), size);
    check("mul", c, 6);
  }
  {
    StorageView a = queued(2, {2}), b = queued(3), c = queued(0);
    P::mul_batch_broadcast(a.data<int32_t>(), b.data<int32_t>(), c.data<int32_t>(), 2, size);
    check("mul_batch_broadcast", c, 6);
  }
  {
    const dim_t dims[] = {2, 3};
    StorageView a = queued(7, {2, 3}), b = queued(0, {3, 2});
    P::transpose_2d(a.data<int32_t>(), dims, b.data<int32_t>());
    check("transpose_2d", b, 7);
  }
  {
    const dim_t dims[] = {1, 2, 3};
    const dim_t perm[] = {2, 0, 1};
    StorageView a = queued(7, {1, 2, 3}), b = queued(0, {3, 1, 2});
    P::transpose_3d(a.data<int32_t>(), dims, perm, b.data<int32_t>());
    check("transpose_3d", b, 7);
  }
  {
    const dim_t dims[] = {1, 1, 2, 3};
    const dim_t perm[] = {3, 0, 1, 2};
    StorageView a = queued(7, {1, 1, 2, 3}), b = queued(0, {3, 1, 1, 2});
    P::transpose_4d(a.data<int32_t>(), dims, perm, b.data<int32_t>());
    check("transpose_4d", b, 7);
  }
}
#endif

INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PrimitiveTest, ::testing::Values(Device::CUDA));
#endif
#ifdef CT2_WITH_MPS
INSTANTIATE_TEST_SUITE_P(MPS, PrimitiveTest, ::testing::Values(Device::MPS));
#endif
