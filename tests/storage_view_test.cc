#include "test_utils.h"
#include "ctranslate2/storage_view.h"

TEST(StorageViewTest, ZeroDim) {
  StorageView a({2, 0, 2});
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.rank(), 3);
  EXPECT_EQ(a.dim(0), 2);
  EXPECT_EQ(a.dim(1), 0);
  EXPECT_EQ(a.dim(2), 2);

  StorageView b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.rank(), 3);
  EXPECT_EQ(b.dim(0), 2);
  EXPECT_EQ(b.dim(1), 0);
  EXPECT_EQ(b.dim(2), 2);
}

TEST(StorageViewTest, BoolOperator) {
  StorageView a;
  EXPECT_FALSE(bool(a));
  a.resize({4});
  EXPECT_TRUE(bool(a));
}

TEST(StorageViewTest, Reshape) {
  StorageView a(Shape{16});
  assert_vector_eq(a.shape(), Shape{16});
  a.reshape({4, 4});
  assert_vector_eq(a.shape(), Shape{4, 4});
  a.reshape({2, -1});
  assert_vector_eq(a.shape(), Shape{2, 8});
  a.reshape({-1, 1});
  assert_vector_eq(a.shape(), Shape{16, 1});
  a.reshape({2, -1, 2});
  assert_vector_eq(a.shape(), Shape{2, 4, 2});
  a.reshape({-1});
  assert_vector_eq(a.shape(), Shape{16});
}

TEST(StorageViewTest, ExpandDimsAndSqueeze) {
  {
    StorageView a(Shape{4});
    a.expand_dims(0);
    assert_vector_eq(a.shape(), Shape{1, 4});
    a.expand_dims(-1);
    assert_vector_eq(a.shape(), Shape{1, 4, 1});
    a.squeeze(0);
    assert_vector_eq(a.shape(), Shape{4, 1});
    a.squeeze(1);
    assert_vector_eq(a.shape(), Shape{4});
  }

  {
    StorageView a(Shape{4, 2});
    a.expand_dims(1);
    assert_vector_eq(a.shape(), Shape{4, 1, 2});
    a.expand_dims(3);
    assert_vector_eq(a.shape(), Shape{4, 1, 2, 1});
  }
}

class StorageViewDeviceTest : public ::testing::TestWithParam<Device> {
};

TEST_P(StorageViewDeviceTest, StorageViewConstruction) {
  StorageView x({2, 2}, std::vector<float>{1, 2, 3, 4}, GetParam());
}

TEST_P(StorageViewDeviceTest, StorageViewEquality) {
  const Device device = GetParam();
  StorageView x({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView expected({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  expect_storage_eq(x, expected);
}

TEST_P(StorageViewDeviceTest, Shape) {
  StorageView a({2, 2}, std::vector<float>{1, 2, 3, 4}, GetParam());
  EXPECT_EQ(a.size(), 4);
  EXPECT_EQ(a.dim(0), 2);
  EXPECT_EQ(a.dim(1), 2);
}

TEST_P(StorageViewDeviceTest, ToVector) {
  StorageView a({2, 2}, std::vector<float>{1, 2, 3, 4}, GetParam());
  auto vec = a.to_vector<float>();
  EXPECT_EQ(vec[0], 1.);
  EXPECT_EQ(vec[1], 2.);
  EXPECT_EQ(vec[2], 3.);
  EXPECT_EQ(vec[3], 4.);
}

TEST_P(StorageViewDeviceTest, ScalarAt) {
  const Device device = GetParam();
  const dim_t index = 1;
  {
    StorageView values({2}, std::vector<int>{22, 33}, device);
    EXPECT_EQ(values.scalar_at<int>({index}), 33);
  }
  {
    StorageView values({2}, std::vector<signed char>{'c', 'd'}, device);
    EXPECT_EQ(values.scalar_at<signed char>({index}), 'd');
  }
  {
    StorageView values({2}, std::vector<float>{0.2f, -1.f}, device);
    EXPECT_EQ(values.scalar_at<float>({index}), -1.f);
  }
}

TEST_P(StorageViewDeviceTest, ScalarFill) {
  const Device device = GetParam();
  StorageView a({2, 2}, 42.f, device);
  StorageView expected({2, 2}, std::vector<float>{42.f, 42.f,
                                                  42.f, 42.f}, device);
  expect_storage_eq(a, expected);
  StorageView b({3, 4}, 17, device);
  StorageView expected2({3, 4}, std::vector<int>{17, 17, 17, 17,
                                                 17, 17, 17, 17,
                                                 17, 17, 17, 17,}, device);
  expect_storage_eq(b, expected2);
  StorageView c({200, 201}, (float16_t) 42, device);
  StorageView expected3({200, 201}, (float16_t) 42, device);
  expect_storage_eq(c, expected3);
}

TEST_P(StorageViewDeviceTest, HalfConversion) {
  const Device device = GetParam();
  const StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  EXPECT_EQ(a.reserved_memory(), 4 * 4);
  const StorageView b = a.to_float16();
  EXPECT_EQ(b.dtype(), DataType::FLOAT16);
  EXPECT_EQ(b.reserved_memory(), 4 * 2);
  expect_storage_eq(b.to_float32(), a);
}

INSTANTIATE_TEST_SUITE_P(CPU, StorageViewDeviceTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, StorageViewDeviceTest, ::testing::Values(Device::CUDA));
#elif CT2_WITH_CANN
INSTANTIATE_TEST_SUITE_P(CANN, StorageViewDeviceTest, ::testing::Values(Device::CANN));
#endif
