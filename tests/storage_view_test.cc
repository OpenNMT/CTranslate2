#include <limits>

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

TEST(StorageViewTest, InvalidNegativeDim) {
  StorageView scalar(1.0f);
  EXPECT_THROW(scalar.dim(-1), std::invalid_argument);
}

TEST(StorageViewTest, InvalidNegativeIndex) {
  StorageView storage({1}, std::vector<float>{0});
  EXPECT_THROW(storage.at<float>(-1), std::invalid_argument);
}

TEST(StorageViewTest, ReserveByteSizeOverflow) {
  for (const DataType dtype : {DataType::FLOAT16, DataType::FLOAT32}) {
    StorageView sv(dtype);
    const dim_t max_elements = std::numeric_limits<dim_t>::max() / sv.item_size();
    for (const dim_t size : {max_elements + 1, std::numeric_limits<dim_t>::max()}) {
      try {
        sv.reserve(size);
        FAIL() << "Expected byte-size overflow rejection";
      } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("tensor byte size overflows dim_t"),
                  std::string::npos);
      }
    }
  }
}

TEST(StorageViewTest, ReserveNegativeSize) {
  StorageView sv(DataType::FLOAT32);
  EXPECT_THROW(sv.reserve(-1), std::runtime_error);
  EXPECT_THROW(sv.reserve(std::numeric_limits<dim_t>::min()), std::runtime_error);
}

TEST(StorageViewTest, InvalidReservePreservesStorage) {
  StorageView sv({2}, std::vector<float>{1, 2});
  const auto* data = sv.data<float>();
  const auto reserved_memory = sv.reserved_memory();
  EXPECT_THROW(sv.reserve(-1), std::runtime_error);
  EXPECT_THROW(sv.reserve(std::numeric_limits<dim_t>::max()), std::runtime_error);
  EXPECT_EQ(sv.data<float>(), data);
  EXPECT_EQ(sv.reserved_memory(), reserved_memory);
  EXPECT_TRUE(sv.owns_data());
  EXPECT_EQ(sv.size(), 2);
  ASSERT_EQ(sv.shape(), Shape({2}));
  EXPECT_EQ(sv.at<float>(0), 1);
  EXPECT_EQ(sv.at<float>(1), 2);
}

TEST(StorageViewTest, ReserveWithinCapacity) {
  StorageView empty(DataType::FLOAT32);
  EXPECT_NO_THROW(empty.reserve(0));
  EXPECT_EQ(empty.reserved_memory(), 0);

  StorageView sv({2}, std::vector<float>{1, 2});
  const auto* data = sv.data<float>();
  const auto reserved_memory = sv.reserved_memory();
  EXPECT_NO_THROW(sv.reserve(0));
  EXPECT_NO_THROW(sv.reserve(1));
  EXPECT_NO_THROW(sv.reserve(2));
  EXPECT_EQ(sv.data<float>(), data);
  EXPECT_EQ(sv.reserved_memory(), reserved_memory);
  ASSERT_EQ(sv.shape(), Shape({2}));
  EXPECT_EQ(sv.at<float>(0), 1);
  EXPECT_EQ(sv.at<float>(1), 2);
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
#endif

TEST(StorageViewTest, OverflowingShapeThrows) {
  // (2^62 + 1) * 8 overflows dim_t: the shape must be rejected, not wrapped
  // to a small size that passes allocation.
  EXPECT_THROW(StorageView(Shape{(int64_t{1} << 62) + 1, 8}, DataType::INT32),
               std::invalid_argument);
}

TEST(StorageViewTest, OverflowingReshapeThrows) {
  StorageView view(Shape{8}, DataType::INT32);
  EXPECT_THROW(view.reshape(Shape{(int64_t{1} << 62) + 1, 8}), std::invalid_argument);
}

TEST(StorageViewTest, ZeroDimIsValidWhateverItsPosition) {
  // The size of a shape with a zero dimension is 0, so the other dimensions are
  // never multiplied and the overflowing prefix below is not an error. Multiplying
  // before looking for the zero dimension made the result depend on its position.
  EXPECT_THROW(compute_size(Shape{(int64_t{1} << 62) + 1, 8}), std::invalid_argument);
  EXPECT_EQ(compute_size(Shape{(int64_t{1} << 62) + 1, 8, 0}), 0);
  EXPECT_EQ(compute_size(Shape{0, (int64_t{1} << 62) + 1, 8}), 0);

  StorageView view(Shape{(int64_t{1} << 62) + 1, 8, 0}, DataType::INT32);
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.dim(2), 0);
  view.reshape(Shape{0, (int64_t{1} << 62) + 1, 8});
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.dim(0), 0);
}

TEST(StorageViewTest, ZeroDimDoesNotSkipNegativeCheck) {
  EXPECT_THROW(compute_size(Shape{0, -2}), std::invalid_argument);
  EXPECT_THROW(compute_size(Shape{(int64_t{1} << 62) + 1, 0, -1}), std::invalid_argument);

  StorageView view(Shape{0}, DataType::INT32);
  EXPECT_THROW(view.reshape(Shape{0, 8, -3}), std::invalid_argument);
}

TEST(StorageViewTest, UnknownDimWithZeroDimThrows) {
  // An unknown dimension can't be inferred from an empty shape, and the known size
  // of 0 used to be the divisor of the compatibility check below.
  StorageView view(Shape{8}, DataType::INT32);
  EXPECT_THROW(view.reshape(Shape{0, -1}), std::invalid_argument);
  EXPECT_THROW(view.reshape(Shape{-1, 0}), std::invalid_argument);
}

TEST(StorageViewTest, GrowRejectsNegativeIncrement) {
  // grow() with a negative size used to reach resize() and silently shrink the
  // dimension, and INT64_MIN made the addition wrap.
  StorageView view(Shape{4}, DataType::INT32);
  for (const dim_t size : {dim_t(-1), std::numeric_limits<dim_t>::min()}) {
    try {
      view.grow(0, size);
      FAIL() << "Expected a negative growth to be rejected";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("can't grow a dimension by a negative size"),
                std::string::npos);
    }
  }

  // A rejected grow leaves the storage untouched.
  EXPECT_EQ(view.dim(0), 4);
  EXPECT_EQ(view.size(), 4);
}

TEST(StorageViewTest, GrowRejectsAdditionOverflow) {
  // The zero dimension makes the size 0, so this overflowing first dimension costs
  // no memory and still exercises the guard on the addition itself.
  StorageView view(Shape{std::numeric_limits<dim_t>::max(), 0}, DataType::INT32);
  EXPECT_EQ(view.size(), 0);

  EXPECT_THROW(view.grow(0, 1), std::invalid_argument);
  EXPECT_THROW(view.grow(0, std::numeric_limits<dim_t>::max()), std::invalid_argument);

  // Growing the zero dimension by INT64_MAX doesn't overflow the addition (0 +
  // INT64_MAX), so grow() lets it through and compute_size() rejects the resulting
  // shape instead of wrapping it.
  EXPECT_THROW(view.grow(1, std::numeric_limits<dim_t>::max()), std::invalid_argument);

  EXPECT_EQ(view.dim(0), std::numeric_limits<dim_t>::max());
  EXPECT_EQ(view.dim(1), 0);
  EXPECT_EQ(view.size(), 0);
}

TEST(StorageViewTest, GrowingToAnUnallocatableSizeIsRejected) {
  // INT64_MAX is a valid dimension and INT64_MAX elements still fit in a dim_t, so
  // neither grow() nor compute_size() rejects it: reserve() catches the byte size
  // before allocating. The three guards leave no path through to the allocator.
  StorageView view(Shape{0}, DataType::INT32);
  EXPECT_THROW(view.grow(0, std::numeric_limits<dim_t>::max()), std::runtime_error);
  EXPECT_EQ(view.size(), 0);
}

TEST(StorageViewTest, GrowIncreasesShape) {
  StorageView view(Shape{2, 3}, DataType::INT32);
  EXPECT_EQ(view.size(), 6);

  view.grow(0, 1);
  EXPECT_EQ(view.dim(0), 3);
  EXPECT_EQ(view.dim(1), 3);
  EXPECT_EQ(view.size(), 9);

  view.grow(1, 2);
  EXPECT_EQ(view.dim(1), 5);
  EXPECT_EQ(view.size(), 15);

  // Growing an empty dimension back to a positive size stays valid.
  StorageView empty_view(Shape{0}, DataType::INT32);
  empty_view.grow(0, 4);
  EXPECT_EQ(empty_view.dim(0), 4);
  EXPECT_EQ(empty_view.size(), 4);
}
