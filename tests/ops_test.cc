#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/ops.h"

TEST(OpTest, Transpose1D) {
  StorageView x({4}, std::vector<float>{1, 2, 3, 4});
  StorageView y;
  ops::Transpose()(x, y);
  expect_storage_eq(y, x);
}

TEST(OpTest, Squeeze) {
  StorageView x({2, 1, 3}, DataType::FLOAT32);
  StorageView y;
  ops::Squeeze({1})(x, y);
  assert_vector_eq(y.shape(), {2, 3});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  y.release();
  EXPECT_THROW(ops::Squeeze({0})(x, y), std::invalid_argument);
}

TEST(OpTest, Unsqueeze) {
  StorageView x({2, 3}, DataType::FLOAT32);
  StorageView y;
  ops::Unsqueeze({1})(x, y);
  assert_vector_eq(y.shape(), {2, 1, 3});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  StorageView z;
  ops::Unsqueeze({0})(y, z);
  assert_vector_eq(z.shape(), {1, 2, 1, 3});
  EXPECT_EQ(z.data<float>(), y.data<float>());
}

TEST(OpTest, SplitNoCopyInvalidArgument) {
  ASSERT_RAISES(ops::Split(1, /*no_copy=*/true), std::invalid_argument);
}

TEST(OpDeviceTest, SplitInvalidSize) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView a, b;
  ASSERT_RAISES(ops::Split(0, {3, 2})(x, a, b), std::invalid_argument);
}

TEST(OpDeviceTest, SplitInvalidNumSplits) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView a, b, c;
  ASSERT_RAISES(ops::Split(0, {3, 1})(x, a, b, c), std::invalid_argument);
}

TEST(OpDeviceTest, SplitInvalidNumOutputs) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView a, b, c;
  ASSERT_RAISES(ops::Split(0)(x, a, b, c), std::invalid_argument);
}

TEST(OpDeviceTest, GatherInPlaceStrictlyIncreasing) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({2}, std::vector<int32_t>{1, 2});
  StorageView expected({2, 2}, std::vector<float>{2, 2, 3, 3});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_EQ(data.buffer(), data_ptr);
}

TEST(OpDeviceTest, GatherInPlaceIncreasing) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({3}, std::vector<int32_t>{0, 0, 1});
  StorageView expected({3, 2}, std::vector<float>{1, 1, 1, 1, 2, 2});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_NE(data.buffer(), data_ptr);
}

TEST(OpDeviceTest, GatherInPlaceDecreasing) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({2}, std::vector<int32_t>{1, 0});
  StorageView expected({2, 2}, std::vector<float>{2, 2, 1, 1});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_NE(data.buffer(), data_ptr);
}

TEST(OpDeviceTest, GatherInPlaceLarger) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({5}, std::vector<int32_t>{0, 1, 2, 3, 3});
  StorageView expected({5, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 4, 4});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_NE(data.buffer(), data_ptr);
}

TEST(OpTest, GemmInt16) {
  if (!mayiuse_int16(Device::CPU))
    return;
  StorageView a({64, 64}, static_cast<int16_t>(1));
  StorageView b(a);
  StorageView y({64, 64}, static_cast<int32_t>(2));
  StorageView expected({64, 64}, static_cast<int32_t>(130));
  ops::Gemm op(2.0, 1.0, false, true);
  op(a, b, y);
  expect_storage_eq(y, expected);
};

TEST(OpTest, QuantizeINT16) {
  StorageView scale;
  StorageView input({4}, std::vector<float>{0.1f, -0.5f, 2.0f, 0.0f});
  StorageView expected({4}, std::vector<int16_t>{100, -500, 2000, 0});
  StorageView output(expected.dtype());
  StorageView reverse(input.dtype());
  ops::Quantize()(input, output, scale);
  expect_storage_eq(output, expected);
  ops::Dequantize()(output, scale, reverse);
  expect_storage_eq(reverse, input);
}

class OpDeviceTest : public ::testing::TestWithParam<Device> {
};

class OpDeviceFPTest : public ::testing::TestWithParam<FloatType> {
};


TEST_P(OpDeviceFPTest, MedianFilter) {
  Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x({2, 8}, std::vector<float>{
      0.2556743323802948, 0.8028775453567505, 0.3514494299888611, 0.3542254865169525,
      0.5881291031837463, 0.1458204835653305, 0.6845740675926208, 0.543143630027771,
      0.9039326310157776, 0.38000917434692383, 0.9094009399414062, 0.4063926637172699,
      0.7943458557128906, 0.289182186126709, 0.9932224750518799, 0.01137143187224865},
      device);
  StorageView expected({2, 8}, std::vector<float>{
      0.3514494299888611, 0.3542254865169525, 0.3542254865169525, 0.3542254865169525,
      0.3542254865169525, 0.543143630027771, 0.5881291031837463, 0.543143630027771,
      0.9039326310157776, 0.4063926637172699, 0.7943458557128906, 0.4063926637172699,
      0.7943458557128906, 0.4063926637172699, 0.7943458557128906, 0.289182186126709},
      device);
  StorageView y(dtype, device);
  ops::MedianFilter(5)(x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, MedianFilterShortAxis) {
  // An axis shorter than the filter rank (including a zero-size axis, as
  // produced by Whisper align() when num_frames < 2) must pass the input
  // through instead of crashing on an integer division by zero.
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  {
    StorageView x({2, 0}, std::vector<float>{}, device);
    StorageView y(dtype, device);
    ops::MedianFilter(5)(x.to(dtype), y);
    EXPECT_EQ(y.size(), 0);
  }
  {
    StorageView x({2, 2}, std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, device);
    StorageView y(dtype, device);
    ops::MedianFilter(5)(x.to(dtype), y);
    expect_storage_eq(y.to_float32(), x, error);
  }
}

TEST_P(OpDeviceTest, Add) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{3, 5, 7, 9}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddScalar) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b(static_cast<float>(3));
  StorageView expected({4}, std::vector<float>{4, 5, 6, 7}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, Mul) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{2, 6, 12, 20}, device);
  StorageView c(a.device());
  ops::Mul()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MulScalar) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b(static_cast<float>(3));
  StorageView expected({4}, std::vector<float>{3, 6, 9, 12}, device);
  StorageView c(a.device());
  ops::Mul()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, Sub) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{-1, -1, -1, -1}, device);
  StorageView c(a.device());
  ops::Sub()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, TileFirstDim) {
  Device device = GetParam();
  StorageView input({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView expected_output({4, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView output(device);
  ops::Tile(0, 2)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, TileLastDim) {
  Device device = GetParam();
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView expected_output({2, 4}, std::vector<float>{1, 2, 1, 2, 3, 4, 3, 4}, device);
  StorageView output(device);
  ops::Tile(1, 2)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, TileMiddleDim) {
  Device device = GetParam();
  StorageView input({2, 1, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView expected_output({2, 3, 3}, std::vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6}, device);
  StorageView output(device);
  ops::Tile(1, 3)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, ConcatEmpty) {
  Device device = GetParam();
  StorageView a({2, 1, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({2, 0, 2}, DataType::FLOAT32, device);
  StorageView x(device);
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, a);
}

TEST_P(OpDeviceTest, ConcatSplitBatch) {
  Device device = GetParam();
  StorageView a({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({1, 2}, std::vector<float>{5, 6}, device);
  StorageView c({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView x(device);
  ops::Concat(0)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(0, {2, 1})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, ConcatSplitTime) {
  Device device = GetParam();
  StorageView a({2, 2, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, device);
  StorageView b({2, 1, 2}, std::vector<float>{5, 5, 6, 6}, device);
  StorageView c({2, 3, 2}, std::vector<float>{1, 1, 2, 2, 5, 5, 3, 3, 4, 4, 6, 6}, device);
  StorageView x(device);
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(1, {2, 1})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, ConcatSplitDepth) {
  Device device = GetParam();
  StorageView a({2, 1}, std::vector<float>{1, 4}, device);
  StorageView b({2, 2}, std::vector<float>{2, 3, 5, 6}, device);
  StorageView c({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(-1, {1, 2})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, ConcatSplitDepth3) {
  Device device = GetParam();
  StorageView a({2, 2}, std::vector<float>{1, 2, 6, 7}, device);
  StorageView b({2, 1}, std::vector<float>{3, 8}, device);
  StorageView c({2, 2}, std::vector<float>{4, 5, 9, 10}, device);
  StorageView d({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b, &c}, x);
  expect_storage_eq(x, d);
  StorageView w(device);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&w, &y, &z};
  ops::Split(-1, {2, 1, 2})(d, out);
  expect_storage_eq(w, a);
  expect_storage_eq(y, b);
  expect_storage_eq(z, c);
}

TEST_P(OpDeviceTest, ConcatSplitDepthEqualParts) {
  Device device = GetParam();
  StorageView a({2, 2}, std::vector<float>{1, 2, 5, 6}, device);
  StorageView b({2, 2}, std::vector<float>{3, 4, 7, 8}, device);
  StorageView c({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(-1)(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, SplitNoCopy) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView y(device);
  StorageView z(device);
  ops::Split(0, {3, 1}, /*no_copy=*/true)(x, y, z);
  assert_vector_eq(y.shape(), Shape{3, 2});
  assert_vector_eq(z.shape(), Shape{1, 2});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  EXPECT_EQ(z.data<float>(), x.data<float>() + 3 * 2);
}

TEST_P(OpDeviceTest, SplitNoCopyEqualParts) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView y(device);
  StorageView z(device);
  ops::Split(0, /*no_copy=*/true)(x, y, z);
  assert_vector_eq(y.shape(), Shape{2, 2});
  assert_vector_eq(z.shape(), Shape{2, 2});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  EXPECT_EQ(z.data<float>(), x.data<float>() + 4);
}

TEST_P(OpDeviceTest, Mean) {
  const Device device = GetParam();
  const StorageView input({2, 3, 2}, std::vector<float>{
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12
    }, device);
  StorageView output(device);

  {
    ops::Mean(0)(input, output);
    const StorageView expected({3, 2}, std::vector<float>{4, 5, 6, 7, 8, 9}, device);
    expect_storage_eq(output, expected);
  }

  {
    ops::Mean(1)(input, output);
    const StorageView expected({2, 2}, std::vector<float>{3, 4, 9, 10}, device);
    expect_storage_eq(output, expected);
  }

  {
    ops::Mean(-1)(input, output);
    const StorageView expected({2, 3}, std::vector<float>{1.5, 3.5, 5.5, 7.5, 9.5, 11.5}, device);
    expect_storage_eq(output, expected);
  }
}

TEST_P(OpDeviceTest, GatherData1D) {
  Device device = GetParam();
  StorageView data({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({2}, std::vector<float>{2, 4}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData1DIndex2D) {
  Device device = GetParam();
  StorageView data({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView ids({2, 3}, std::vector<int32_t>{1, 3, 1, 1, 2, 0}, device);
  StorageView expected({2, 3}, std::vector<float>{2, 4, 2, 2, 3, 1}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData2D) {
  Device device = GetParam();
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({2, 2}, std::vector<float>{2, 2, 4, 4}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData3D) {
  Device device = GetParam();
  StorageView data({2, 3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 1}, device);
  StorageView expected({2, 3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData2DIndex2D) {
  Device device = GetParam();
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, device);
  StorageView ids({2, 3}, std::vector<int32_t>{1, 3, 3, 2, 1, 0}, device);
  StorageView expected({2, 3, 2}, std::vector<float>{2, 2, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherInDepthWith1DInput) {
  Device device = GetParam();
  StorageView data({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({2}, std::vector<float>{2, 8}, device);
  StorageView output(device);
  ops::Gather(-1, 1)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherInDepthWith2DInput) {
  Device device = GetParam();
  StorageView data({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView ids({2, 2}, std::vector<int32_t>{1, 2, 0, 3}, device);
  StorageView expected({2, 2}, std::vector<float>{2, 3, 5, 8}, device);
  StorageView output(device);
  ops::Gather(-1, 1)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherInTime) {
  Device device = GetParam();
  StorageView data({2, 3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
  StorageView ids({2, 2}, std::vector<int32_t>{1, 1, 2, 0}, device);
  StorageView expected({2, 2, 2}, std::vector<float>{3, 4, 3, 4, 11, 12, 7, 8}, device);
  StorageView output(device);
  ops::Gather(1, 1)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, Transpose2D) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView expected({2, 4}, std::vector<float>{1, 3, 5, 7, 2, 4, 6, 8}, device);
  StorageView y(device);
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
  y.release();
  ops::Transpose({1, 0})(x, y);
  expect_storage_eq(y, expected);
  y.release();
  ops::Transpose({0, 1})(x, y);
  expect_storage_eq(y, x);
}

TEST_P(OpDeviceTest, Transpose2DInt16) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<int16_t>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView expected({2, 4}, std::vector<int16_t>{1, 3, 5, 7, 2, 4, 6, 8}, device);
  StorageView y(x.dtype(), x.device());
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
}

TEST_P(OpDeviceTest, Transpose3D) {
  Device device = GetParam();
  StorageView x({3, 2, 3},
                std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 3}, device);
  StorageView expected({2, 3, 3},
                       std::vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9, 1, 1, 1, 2, 2, 2, 3, 3, 3}, device);
  StorageView y(x.dtype(), x.device());
  ops::Transpose({1, 2, 0})(x, y);
  expect_storage_eq(y, expected);
}

TEST_P(OpDeviceTest, Transpose3DReverse) {
  Device device = GetParam();
  StorageView x({3, 2, 3},
                std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 3}, device);
  StorageView expected({3, 2, 3},
                       std::vector<float>{1, 4, 7, 1, 1, 1, 2, 5, 8, 2, 2, 2, 3, 6, 9, 3, 3, 3}, device);
  StorageView y(x.dtype(), x.device());
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
}

static const StorageView gemm_a({4, 5}, std::vector<float>{
    1.229355, 0.613804, 0.400132, -1.239135, -1.782431,
    0.952620, -0.849226, 1.196964, -0.167603, 0.188656,
    0.701979, 0.952979, 0.849013, 1.627527, 0.742893,
    0.052268, 1.588265, -0.497923, 0.071221, -0.320372});

static const StorageView gemm_b({5, 2}, std::vector<float>{
    -2.159117, -0.547031, 0.287564, -1.467412, 0.007335,
    1.249786, -0.143760, -0.952920, 0.821347, 1.406196});

static const StorageView gemm_y({4, 2}, std::vector<float>{
    0.399763, 0.328475, 0.448811, 0.777195,
    -0.327203, 0.279757, -0.434141, -0.743254});

TEST_P(OpDeviceFPTest, Gemm) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView expected({4, 2}, std::vector<float>{
      -3.360972, -2.070295, -1.664387, 3.423195,
      -1.186388, -0.947825, -0.367293, -4.243156}, device);
  const ops::Transpose transpose_op;

  // a @ b + y
  {
    ops::Gemm op(1.0, 1.0, false, false);
    StorageView a = gemm_a.to(device).to(dtype);
    StorageView b = gemm_b.to(device).to(dtype);
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y);
    expect_storage_eq(y.to_float32(), expected, error);
  }

  // a^T @ b + y
  {
    ops::Gemm op(1.0, 1.0, true, false);
    StorageView a(dtype, device);
    transpose_op(gemm_a.to(device).to(dtype), a);
    StorageView b = gemm_b.to(device).to(dtype);
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y);
    expect_storage_eq(y.to_float32(), expected, error);
  }

  // a @ b^T + y
  {
    ops::Gemm op(1.0, 1.0, false, true);
    StorageView a = gemm_a.to(device).to(dtype);
    StorageView b(dtype, device);
    transpose_op(gemm_b.to(device).to(dtype), b);
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y);
    expect_storage_eq(y.to_float32(), expected, error);
  }

  // a^T @ b^T + y
  {
    ops::Gemm op(1.0, 1.0, true, true);
    StorageView a(dtype, device);
    transpose_op(gemm_a.to(device).to(dtype), a);
    StorageView b(dtype, device);
    transpose_op(gemm_b.to(device).to(dtype), b);
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y);
    expect_storage_eq(y.to_float32(), expected, error);
  }
};

TEST_P(OpDeviceFPTest, GemmBias) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView bias({2}, std::vector<float>{0.5f, -0.1f}, device);
  bias = bias.to(dtype);
  StorageView expected({4, 2}, std::vector<float>{
      -2.860972, -2.170295, -1.164387, 3.323195,
      -0.686388, -1.047825, 0.132707, -4.343156});
  ops::Gemm op(1.0, 1.0, false, false);
  const StorageView a = gemm_a.to(device).to(dtype);
  const StorageView b = gemm_b.to(device).to(dtype);
  StorageView y = gemm_y.to(device).to(dtype);
  // a @ b + bias + y
  op(a, b, y, nullptr, &bias);
  expect_storage_eq(y.to_float32(), expected, error);
};

TEST_P(OpDeviceFPTest, GemmResidual) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView residual({4, 2}, std::vector<float>{
    0.274953, -0.023021, 0.669752, 1.203405,
    1.116680, -1.247854, 0.920804, 0.063622}, device);
  residual = residual.to(dtype);
  ops::Gemm op(1.0, 1.0, false, false);
  const StorageView a = gemm_a.to(device).to(dtype);
  const StorageView b = gemm_b.to(device).to(dtype);

  // a @ b + bias + y + residual
  {
    StorageView bias({2}, std::vector<float>{-0.2f, 0.6f}, device);
    bias = bias.to(dtype);
    StorageView expected({4, 2}, std::vector<float>{
        -3.286019, -1.493316, -1.194635, 5.226600,
        -0.269708, -1.595679, 0.353511, -3.579534});
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y, nullptr, &bias, &residual);
    expect_storage_eq(y.to_float32(), expected, error);
  }

  // a @ b + y + residual
  {
    StorageView expected({4, 2}, std::vector<float>{
        -3.086019, -2.093316, -0.994635, 4.626600,
        -0.069708, -2.195679, 0.553511, -4.179534});
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y, nullptr, nullptr, &residual);
    expect_storage_eq(y.to_float32(), expected, error);
  }
};

TEST_P(OpDeviceFPTest, GemmGELU) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView bias({2}, std::vector<float>{-0.3f, 0.2f}, device);
  bias = bias.to(dtype);
  StorageView residual({4, 2}, std::vector<float>{
    2.484578, 1.774646, 1.115868, -0.369928,
    1.179388, -0.002726, -2.600337, 0.369746}, device);
  residual = residual.to(dtype);
  const ops::ActivationType activation_type = ops::ActivationType::GELU;
  ops::Gemm op(1.0, 0.0, false, false, false, false, &activation_type);
  const StorageView a = gemm_a.to(device).to(dtype);
  const StorageView b = gemm_b.to(device).to(dtype);

  // GELU(a @ b + bias + residual)
  {
    StorageView expected({4, 2}, std::vector<float>{
        -0.090621, -0.142394, -0.126177, 2.459627,
        0.010264, -0.156022, -0.006523, -0.004964});
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y, nullptr, &bias, &residual);
    expect_storage_eq(y.to_float32(), expected, error);
  }

  // GELU(a @ b + bias)
  {
    StorageView expected({4, 2}, std::vector<float>{
        -0.000099, -0.030667, -0.019080, 2.839700,
        -0.142800, -0.156268, -0.095085, -0.001596});
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y, nullptr, &bias);
    expect_storage_eq(y.to_float32(), expected, error);
  }

  // GELU(a @ b)
  {
    StorageView expected({4, 2}, std::vector<float>{
        -0.000319, -0.019730, -0.036541, 2.635224,
        -0.167644, -0.134791, 0.035205, -0.000814});
    StorageView y = gemm_y.to(device).to(dtype);
    op(a, b, y);
    expect_storage_eq(y.to_float32(), expected, error);
  }
};

TEST_P(OpDeviceTest, GemmInt8) {
  Device device = GetParam();
  if (!mayiuse_int8(device))
    return;
  StorageView a({3, 8}, std::vector<int8_t>{
      -31, 14, -39, 36, 17, 4, -10, 15,
      -58, 8, 0, -26, -18, -42, -3, -21,
      -27, -63, -51, -4, -37, -63,  2, -4}, device);
  StorageView b({8, 4}, std::vector<int8_t>{
      42, -59, -28, 50,
      56, -17, 14, -57,
      -15, 37, 37, 63,
      -29, 41, -41, 9,
      -47, 38, -20, 27,
      54, 16, -11, -31,
      32, -17, -10, -58,
      45, -17, 58, -44}, device);
  StorageView c(DataType::INT32, device);
  StorageView expected({3, 4}, std::vector<int32_t>{
      -1205, 2249, -1269, -4226,
      -3697, 1272, 2436, -1676,
      -5560, -1767, -668, 6}, device);
  ops::Gemm op(1.0, 0.0, false, false);
  op(a, b, c);
  expect_storage_eq(c, expected);
};

TEST_P(OpDeviceFPTest, TopK) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const int k = 3;
  StorageView input({2, 6}, std::vector<float>{0.1, -0.5, 2.0, 0.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3, -0.2, 0.0}, device);
  StorageView expected_values({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values(dtype, device);
  StorageView indices(expected_indices.dtype(), device);
  ops::TopK op(k);
  op(input.to(dtype), values, indices);
  expect_storage_eq(values.to_float32(), expected_values, error);
  expect_storage_eq(indices, expected_indices);
}

TEST_P(OpDeviceFPTest, TopKWithExhaustedCandidates) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const dim_t rows = 4;
  const dim_t depth = 1025;
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const float lowest = dtype == DataType::FLOAT16 ? -65504.f
    : dtype == DataType::BFLOAT16 ? -std::ldexp(255.f, 120)
    : std::numeric_limits<float>::lowest();

  // Cover each specialized launcher and the generic k=3 branch. Exhaustion
  // must also be safe for non-first rows and candidates outside block lane 0.
  for (const int k : {1, 2, 3, 4, 6, 8, 10, 16, 32, 64}) {
    for (const float masked : {neg_inf, lowest}) {
      SCOPED_TRACE(::testing::Message() << "k=" << k << ", masked=" << masked);
      std::vector<float> data(rows * depth, masked);
      data[depth + 17] = 4.f;
      data[2 * depth + 137] = 4.f;
      data[2 * depth + 513] = 2.f;
      for (int i = 0; i < k; ++i)
        data[3 * depth + 100 + 3 * i] = float(k - i);

      const StorageView expected_input({rows, depth}, data);
      StorageView input = expected_input.to(dtype).to(device);
      StorageView values(dtype, device);
      StorageView indices(DataType::INT32, device);

      const ops::TopK topk(k);
      // Reuse the input: stage 1 temporarily mutates it and must restore it.
      for (int repeat = 0; repeat < 2; ++repeat) {
        topk(input, values, indices);
        const StorageView result_values = values.to_float32().to(Device::CPU);
        const StorageView result_indices = indices.to(Device::CPU);
        assert_vector_eq(result_values.shape(), {rows, dim_t(k)});
        assert_vector_eq(result_indices.shape(), {rows, dim_t(k)});
        expect_storage_eq(input.to_float32().to(Device::CPU), expected_input);

        for (dim_t row = 0; row < rows; ++row) {
          SCOPED_TRACE(::testing::Message() << "row=" << row << ", repeat=" << repeat);
          const int valid = row == 0 ? 0 : row == 1 ? 1 : row == 2 ? 2 : k;
          for (int i = 0; i < k; ++i) {
            const dim_t offset = row * k + i;
            const int32_t index = result_indices.at<int32_t>(offset);
            EXPECT_GE(index, 0);
            EXPECT_LT(index, depth);
            if (i < valid) {
              const int32_t expected_index = row == 1 ? 17
                : row == 2 ? (i == 0 ? 137 : 513) : 100 + 3 * i;
              EXPECT_EQ(index, expected_index);
              EXPECT_FLOAT_EQ(result_values.at<float>(offset),
                              data[row * depth + expected_index]);
            } else {
              // Preserve the existing CUDA sentinel value behavior without
              // prescribing a tied/sentinel index as a public TopK contract.
              EXPECT_FLOAT_EQ(result_values.at<float>(offset),
                              device == Device::CUDA ? lowest : masked);
            }
          }
        }
      }
    }
  }
}

TEST_P(OpDeviceTest, TopKVariableDepth) {
  Device device = GetParam();
  const int k = 3;
  ops::TopK op(k);
  StorageView input({2, 6}, std::vector<float>{0.1, -0.5, 2.0, 0.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3, -0.2, 0.0}, device);
  StorageView expected_values({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values(expected_values.dtype(), device);
  StorageView indices(expected_indices.dtype(), device);
  op(input, values, indices);
  expect_storage_eq(values, expected_values);
  expect_storage_eq(indices, expected_indices);
  StorageView input2({2, 4}, std::vector<float>{0.1, 2.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3}, device);
  StorageView expected_values2({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices2({2, 3}, std::vector<int32_t>{1, 3, 2, 1, 0, 3}, device);
  op(input2, values, indices);
  expect_storage_eq(values, expected_values2);
  expect_storage_eq(indices, expected_indices2);
}

TEST_P(OpDeviceTest, TopKChangeK) {
  const Device device = GetParam();
  const StorageView input({2, 6},
                          std::vector<float>{
                            0.1, -0.5, 2.0, 0.0, 0.2, 0.6,
                            1.0, 1.1, 0.2, 0.3, -0.2, 0.0
                          },
                          device);

  const StorageView expected_values_k2({2, 2}, std::vector<float>{2.0, 0.6, 1.1, 1.0}, device);
  const StorageView expected_indices_k2({2, 2}, std::vector<int32_t>{2, 5, 1, 0}, device);
  StorageView values_k2(expected_values_k2.dtype(), device);
  StorageView indices_k2(expected_indices_k2.dtype(), device);
  ops::TopK(2)(input, values_k2, indices_k2);
  expect_storage_eq(values_k2, expected_values_k2);
  expect_storage_eq(indices_k2, expected_indices_k2);

  const StorageView expected_values_k3({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  const StorageView expected_indices_k3({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values_k3(expected_values_k3.dtype(), device);
  StorageView indices_k3(expected_indices_k3.dtype(), device);
  ops::TopK(3)(input, values_k3, indices_k3);
  expect_storage_eq(values_k3, expected_values_k3);
  expect_storage_eq(indices_k3, expected_indices_k3);
}

TEST_P(OpDeviceFPTest, TopPMask) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  constexpr float inf = std::numeric_limits<float>::infinity();

  StorageView x = StorageView({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device).to(dtype);
  StorageView expected = StorageView({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -inf, 0.0,
      4.6, 3.3, -inf, -inf, 1.0}, device);
  StorageView y(dtype, device);

  ops::TopPMask(0.97)(x, y);

  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, SoftMax) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x = StorageView({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device).to(dtype);
  StorageView expected({2, 5}, std::vector<float>{
      0.032035, 0.785904, 0.129909, 0.013025, 0.039128,
      0.760941, 0.207381, 0.009342, 0.001544, 0.020792}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x, y);
  expect_storage_eq(y.to_float32(), expected, error);
  ops::SoftMax()(x);
  expect_storage_eq(x.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LogSoftMax) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x = StorageView({2, 10}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0, 0.2, -3.0, -1.2, 1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0, -4.6, -3.3, -0.2, 1.6, -1.0}, device).to(dtype);
  StorageView expected({2, 10}, std::vector<float>{
      -3.638294, -0.438294, -2.238294, -4.538294, -3.438294, -3.238294, -6.438294, -4.638294, -2.338294, -3.438294,
      -0.319434, -1.619434, -4.719434, -6.519434, -3.919434, -9.519434, -8.219434, -5.119434, -3.319434, -5.919434}, device);
  StorageView y(dtype, device);
  ops::LogSoftMax()(x, y);
  expect_storage_eq(y.to_float32(), expected, error);
  ops::LogSoftMax()(x);
  expect_storage_eq(x.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, MaskedSoftMax) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView lengths({2}, std::vector<int32_t>{3, 4}, device);
  StorageView expected({2, 5}, std::vector<float>{
      0.033797, 0.829145, 0.137056,        0, 0,
      0.777098, 0.211783, 0.009540, 0.001577, 0}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x.to(dtype), lengths, y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, MaskedSoftMaxTriangular) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x({2, 2, 3, 3}, std::vector<float>{
      0.08784354, 0.67030656, 0.8866086,
      0.08053982, 0.9826797, 0.7965635,
      0.48865926, 0.8635745, 0.21703207,
      0.0742166, 0.0623771, 0.7590432,
      0.43742728, 0.12613738, 0.53697634,
      0.05396891, 0.04152167, 0.66332567,
      0.6386628, 0.23325896, 0.6977577,
      0.06948507, 0.10246396, 0.6232395,
      0.7822603, 0.3168552, 0.11804962,
      0.1133163, 0.29983068, 0.43074536,
      0.7321733, 0.48709297, 0.35727918,
      0.8421174, 0.9135181, 0.77135813
    }, device);
  StorageView lengths({2}, std::vector<int32_t>{3, 2}, device);
  StorageView mask = layers::MultiHeadAttention::prepare_length_mask(lengths, 2, 3, true);
  StorageView expected({2, 2, 3, 3}, std::vector<float>{
      1, 0, 0,
      0.28861094, 0.71138906, 0,
      0.310848, 0.45224282, 0.23690917,
      1, 0, 0,
      0.57720006, 0.42279992, 0,
      0.26130962, 0.25807717, 0.48061317,
      1, 0, 0,
      0.49175602, 0.508244, 0,
      0.61429566, 0.3857044, 0,
      1, 0, 0,
      0.56096524, 0.43903476, 0,
      0.48215744, 0.5178426, 0
    }, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x.to(dtype), mask, y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LayerNorm) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView gamma({5}, std::vector<float>{0.2, 2.1, 1.1, -0.6, 0.7}, device);
  StorageView beta({5}, std::vector<float>{-6.6, -5.7, 0.01, 2.0, 0}, device);
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView expected({2, 5}, std::vector<float>{
      -6.710264, -2.107929, 0.492053, 2.712477, -0.286970,
      -6.319339, -3.988876, -0.637330, 2.841982, -0.158437}, device);
  StorageView y(dtype, device);
  ops::LayerNorm()(beta.to(dtype), gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LayerNormAxis) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x({2, 3, 2}, std::vector<float>{
      0.08830845355987549, 0.7807812690734863,
      0.34740084409713745, 0.8272842764854431,
      0.3155772089958191, 0.21066278219223022,
      0.02693861722946167, 0.6299145221710205,
      0.05086874961853027, 0.6894713640213013,
      0.7736693620681763, 0.4071813225746155}, device);
  StorageView expected({2, 3, 2}, std::vector<float>{
      -1.4052178859710693, 0.6225495338439941,
      0.8405286073684692, 0.7884179353713989,
      0.5646895170211792, -1.410967469215393,
      -0.7413559556007385, 0.4476976990699768,
      -0.6722954511642456, 0.9379060864448547,
      1.4136513471603394, -1.3856042623519897}, device);
  StorageView y(dtype, device);
  ops::LayerNorm(1, 0)(x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, RMSNorm) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView gamma({5}, std::vector<float>{0.2, 2.1, 1.1, -0.6, 0.7}, device);
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView expected({2, 5}, std::vector<float>{
      -0.026160, 4.120200, 0.863280, 0.431640, 0.000000,
      0.344543, 2.595305, 0.082391, 0.359523, 0.262152}, device);
  StorageView y(dtype, device);
  ops::RMSNorm()(gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceTest, QuantizeINT8) {
  Device device = GetParam();
  StorageView a({2, 4}, std::vector<float>{-10, -3, 5, 2, 5, 21, -3, 0}, device);
  StorageView scale(DataType::FLOAT32, device);
  StorageView qa(DataType::INT8, device);
  StorageView expected_scale({2}, std::vector<float>{12.7, 6.047619}, device);

  // With rounding before cast.
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 64, 25, 30, 127, -18, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, true)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }

  // With rounding before cast and shift to uint8.
  // Shift to uin8_t is not defined on CUDA
  if (device != Device::CUDA) {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{1, 90, -64, -103, -98, -1, 110, -128});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, true, true)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }

  // Without rounding before cast (legacy behavior).
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 63, 25, 30, 127, -18, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, false)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }
}

TEST_P(OpDeviceTest, QuantizeINT8ZeroRow) {
  Device device = GetParam();
  StorageView a({2, 4}, std::vector<float>{-10, -3, 5, 2, 0, 0, 0, 0}, device);
  StorageView scale(DataType::FLOAT32, device);
  StorageView qa(DataType::INT8, device);
  StorageView expected_scale({2}, std::vector<float>{12.7, 1}, device);

  // With rounding before cast.
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 64, 25, 0, 0, 0, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, true)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }

  // Without rounding before cast (legacy behavior).
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 63, 25, 0, 0, 0, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, false)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }
}

TEST_P(OpDeviceFPTest, Multinomial) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  StorageView input({2, 4}, std::vector<float>{0.2, 0.1, 0.6, 0.1, 0.7, 0.2, 0.0, 0.1}, device);
  StorageView output(DataType::INT32, device);
  StorageView counts(input.shape(), int32_t(0));

  constexpr dim_t num_draws = 5000;
  for (dim_t i = 0; i < num_draws; ++i) {
    ops::Multinomial(1)(input.to(dtype), output);
    for (dim_t b = 0; b < output.dim(0); ++b)
      counts.at<int32_t>({b, output.scalar_at<int32_t>({b, 0})}) += 1;
  }

  std::vector<int32_t> counts_vec = counts.to_vector<int32_t>();
  std::vector<float> frequencies(counts_vec.begin(), counts_vec.end());
  for (auto& frequency : frequencies)
    frequency /= num_draws;

  expect_storage_eq(StorageView(input.shape(), frequencies), input, 0.05);
}

TEST_P(OpDeviceFPTest, ReLU) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2, 5}, std::vector<float>{-1, 1, 2, -2, 2, 4, -3, 0, -1, -3}, device);
  StorageView expected({2, 5}, std::vector<float>{0, 1, 2, 0, 2, 4, 0, 0, 0, 0}, device);
  StorageView output(dtype, device);
  ops::ReLU()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, GELU) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11585195362567902, -0.1258406937122345}, device);
  StorageView output(dtype, device);
  ops::GELU()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, GELUTanh) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11585142463445663, -0.1260710209608078}, device);
  StorageView output(dtype, device);
  const ops::GELU gelu_op(ops::GELU::Approximation::Tanh);
  gelu_op(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, GELUSigmoid) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11685754358768463, -0.128212109208107}, device);
  StorageView output(dtype, device);
  const ops::GELU gelu_op(ops::GELU::Approximation::Sigmoid);
  gelu_op(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Sigmoid) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.54983395, 0.21416503}, device);
  StorageView output(dtype, device);
  ops::Sigmoid()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Swish) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.10996679, -0.27841452}, device);
  StorageView output(dtype, device);
  ops::Swish()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Tanh) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x({1, 5}, std::vector<float>{-2, -1.5, 0, 1.5, 2}, device);
  StorageView y(dtype, device);
  StorageView expected({1, 5},
                       std::vector<float>{-0.96402758, -0.90514825, 0., 0.90514825, 0.96402758},
                       device);
  ops::Tanh()(x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Log) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  std::vector<float > input_vec({0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4});
  std::vector<float > expected_vec;
  expected_vec.reserve(input_vec.size());
  std::transform(input_vec.begin(), input_vec.end(), std::back_inserter(expected_vec),
          [](const float& i){return std::log(i);});
  StorageView input({2, 4}, input_vec, device);
  StorageView expected({2, 4}, expected_vec, device);
  StorageView output(dtype, device);
  ops::Log()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LogLimits) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;

  StorageView values({2}, std::vector<float>{0.f, -1.f}, device);
  values = values.to(dtype);
  ops::Log()(values, values);
  values = values.to_float32();

  EXPECT_EQ(values.scalar_at<float>({0}), -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(values.scalar_at<float>({1})));
}

template <typename T, typename Ops, typename Func>
void TestMinMax(Device device, const Ops& ops, const Func& func){
  {
    std::vector<T > input_vec1({0, 1, 1.5, 2, 2.5, 3, 3.5, 4});
    std::vector<T > input_vec2({0, -1, 1.5, -2, 2.5, -3, -3.5, 4});
    std::vector<T > output_vec;
    output_vec.reserve(input_vec1.size());
    std::transform(input_vec1.begin(), input_vec1.end(), input_vec2.begin(),std::back_inserter(output_vec),
            [&func](const T& left, const T& right){return func(left, right);});
    StorageView input1({2, 4}, input_vec1, device);
    StorageView input2({2, 4}, input_vec2, device);
    StorageView expected({2, 4}, output_vec, device);
    StorageView output(device);
    ops(input1, input2, output);
    expect_storage_eq(output, expected);
  }
  {
    std::vector<T > input_vec({0, 1, 1.5, 2, 2.5, 3, 3.5, 4});
    T compare_val = 3;
    std::vector<T > output_vec;
    output_vec.reserve(input_vec.size());
    std::transform(input_vec.begin(), input_vec.end(), std::back_inserter(output_vec),
            [&compare_val, &func](const T& left){return func(left, compare_val);});
    StorageView input({2, 4}, input_vec, device);
    StorageView expected({2, 4}, output_vec, device);
    StorageView output(device);
    ops(input, StorageView(compare_val), output);
    expect_storage_eq(output, expected);
  }
}

TEST_P(OpDeviceTest, Min) {
  Device device = GetParam();
  auto ops = ops::Min();
  TestMinMax<float>(device, ops, [](float left, float right){
    return left > right? right : left;
  });
}

TEST_P(OpDeviceTest, Max) {
  Device device = GetParam();
  auto ops = ops::Max();
  TestMinMax<float>(device, ops, [](float left, float right){
    return left > right? left : right;
  });
}

static const StorageView conv_input({2, 2, 3}, std::vector<float>{
    0.5728129, 0.8784890, 0.2029965, 0.3689166, 0.6570600, 0.9202735,
    0.7081605, 0.3570334, 0.9339380, 0.8162224, 0.0597404, 0.4628246});

static const StorageView conv_weight({4, 2, 2}, std::vector<float>{
    0.4969918, 0.3711241, 0.1489926, -0.3010672,
    -0.2055028, 0.2540314, 0.3566069, -0.1201057,
    -0.0737700, -0.0630847, -0.2370351, -0.0451550,
    0.0186623, 0.3600836, -0.2889268, -0.4857445});

static const StorageView conv_bias({4}, std::vector<float>{
    0.4631361, -0.1047785, 0.1047658, -0.3157263});

TEST_P(OpDeviceFPTest, Conv1D) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.9309945, 0.7959076, 0.0533122, -0.1099610,
      -0.1100256, -0.1701476, -0.4144599, -0.8630960,
      1.0512151, 0.8567453, 0.1242856, 0.0248157,
      -0.1661695, -0.0155492, -0.4387956, -0.2148425});
  StorageView output(dtype, device);
  ops::Conv1D()(conv_input.to(device).to(dtype),
                conv_weight.to(device).to(dtype),
                conv_bias.to(device).to(dtype),
                output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DNoBias) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.4678584, 0.3327716, 0.1580907, -0.005182412,
      -0.2147914, -0.2749133, -0.09873369, -0.5473697,
      0.5880789, 0.3936091, 0.2290641, 0.1295942,
      -0.2709353, -0.120315, -0.1230693, 0.1008837});
  StorageView output(dtype, device);
  ops::Conv1D()(conv_input.to(device).to(dtype),
                conv_weight.to(device).to(dtype),
                output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DPadding) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 4}, std::vector<float>{
      0.5646521, 0.9309945, 0.7959076, 0.7011377,
      -0.0035750, 0.0533122, -0.1099610, 0.1816810,
      0.0519716, -0.1100256, -0.1701476, -0.1283464,
      -0.2886650, -0.4144599, -0.8630960, -0.5778296,
      0.4802138, 1.0512151, 0.8567453, 0.9962531,
      -0.0229165, 0.1242856, 0.0248157, -0.1316590,
      0.0232352, -0.1661695, -0.0155492, -0.0738365,
      -0.4572049, -0.4387956, -0.2148425, -0.4320193});
  StorageView output(dtype, device);
  ops::Conv1D(1, 1)(conv_input.to(device).to(dtype),
                    conv_weight.to(device).to(dtype),
                    conv_bias.to(device).to(dtype),
                    output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DStride) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 1}, std::vector<float>{
      0.9309945, 0.0533122, -0.1100256, -0.4144599,
      1.0512151, 0.1242856, -0.1661695, -0.4387956});
  StorageView output(dtype, device);
  ops::Conv1D(2)(conv_input.to(device).to(dtype),
                 conv_weight.to(device).to(dtype),
                 conv_bias.to(device).to(dtype),
                 output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DPaddingAndStride) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.5646521, 0.7959076, -0.0035750, -0.1099610,
      0.0519716, -0.1701476, -0.2886650, -0.8630960,
      0.4802138, 0.8567453, -0.0229165, 0.0248157,
      0.0232352, -0.0155492, -0.4572049, -0.2148425});
  StorageView output(dtype, device);
  ops::Conv1D(2, 1)(conv_input.to(device).to(dtype),
                    conv_weight.to(device).to(dtype),
                    conv_bias.to(device).to(dtype),
                    output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST(OpTest, Conv1DDilationCPU) {
  const StorageView input({1, 1, 7}, std::vector<float>{1, 2, 3, 4, 5, 6, 7});
  const StorageView weight({1, 1, 3}, std::vector<float>{1, 2, 3});
  const StorageView bias({1}, std::vector<float>{1});
  StorageView output;

  ops::Conv1D(1, 0, 2)(input, weight, output);
  expect_storage_eq(output, StorageView({1, 1, 3}, std::vector<float>{22, 28, 34}));

  ops::Conv1D(1, 0, 3)(input, weight, output);
  expect_storage_eq(output, StorageView({1, 1, 1}, std::vector<float>{30}));

  ops::Conv1D(2, 2, 2)(input, weight, bias, output);
  expect_storage_eq(output, StorageView({1, 1, 4}, std::vector<float>{12, 23, 35, 20}));
}

TEST_P(OpDeviceFPTest, Conv1DDilation) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 1}, std::vector<float>{
      0.601058, -0.149898, -0.079298, -0.785548,
      1.143963, 0.222425, -0.220765, -0.426858});
  StorageView output(dtype, device);
  ops::Conv1D(1, 0, 2, 1)(
      conv_input.to(device).to(dtype),
      conv_weight.to(device).to(dtype),
      conv_bias.to(device).to(dtype),
      output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DGELU) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.7672063, 0.62634534, 0.02778943, -0.05016639,
      -0.05019306, -0.0735798, -0.14061329, -0.16747718,
      0.89712805, 0.6890007, 0.06828941, 0.01265349,
      -0.0721195, -0.00767813, -0.14498018, -0.08914787});
  StorageView output(dtype, device);
  const ops::ActivationType activation_type = ops::ActivationType::GELU;
  ops::Conv1D(1, 0, 1, 1, &activation_type)(
      conv_input.to(device).to(dtype),
      conv_weight.to(device).to(dtype),
      conv_bias.to(device).to(dtype),
      output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DGroupNoBias) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const float error = GetParam().error;
    const StorageView expected({2, 2, 2}, std::vector<float>{
            -0.475623, -0.601933, 0.165541, 0.050849, -0.566024,
            -0.592437, 0.121356, 0.232157});
    const StorageView conv_input({2, 4, 4}, std::vector<float>{
            0.547210, 0.634821, 0.571043, 0.443073, 0.220554, 0.478427,
            0.836031, 0.476906, 0.288942, 0.393840, 0.077658, 0.236493,
            0.759209, 0.826134, 0.728944, 0.130438, 0.355182, 0.884368,
            0.494477, 0.004999, 0.306053, 0.764639, 0.903179, 0.440537,
            0.040332, 0.533495, 0.428653, 0.311188, 0.951956, 0.785873,
            0.443364, 0.065968});
    const StorageView conv_weight({2, 2, 3}, std::vector<float>{
            -0.326986, -0.378711, -0.120962, 0.125665, -0.312741, 0.161123,
            0.226274, 0.340959, -0.127573, 0.094374, -0.164143, 0.054516});
    StorageView output(dtype, device);
    ops::Conv1D(1, 0, 1, 2)(conv_input.to(device).to(dtype),
                            conv_weight.to(device).to(dtype),
                            output);
    EXPECT_EQ(output.dtype(), dtype);
    expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DGroupNoBiasQuantized) {
#ifdef CT2_WITH_DNNL
    GTEST_SKIP() << "Quantized convolution is not implemented for DNNL.";
#endif
    const Device device = GetParam().device;
    if (device != Device::CPU)
        GTEST_SKIP() << "Grouped quantized convolution is not implemented for CUDA.";
    const DataType dtype = GetParam().dtype;
    const float error = std::max(GetParam().error, float(3e-3));
    const StorageView expected({2, 2, 2}, std::vector<float>{
            -0.475623, -0.601933, 0.165541, 0.050849, -0.566024,
            -0.592437, 0.121356, 0.232157});
    const StorageView conv_input({2, 4, 4}, std::vector<float>{
            0.547210, 0.634821, 0.571043, 0.443073, 0.220554, 0.478427,
            0.836031, 0.476906, 0.288942, 0.393840, 0.077658, 0.236493,
            0.759209, 0.826134, 0.728944, 0.130438, 0.355182, 0.884368,
            0.494477, 0.004999, 0.306053, 0.764639, 0.903179, 0.440537,
            0.040332, 0.533495, 0.428653, 0.311188, 0.951956, 0.785873,
            0.443364, 0.065968});
    // These weights correspond to the ones in Conv1DGroupNoBias
    // Hence expected output is same (with quantization error)
    // Therefore we use error = 3e-3
    const StorageView conv_weight({2, 2, 3}, std::vector<int8_t>{
            -110, -127,  -41,   42, -105,   54, 84,  127,  -48,   35,  -61,   20});
    const StorageView conv_qscale({2}, std::vector<float> {335.34806224, 372.47880244});
    StorageView output(dtype, device);
    ops::Conv1D(1, 0, 1, 2)(conv_input.to(device).to(dtype),
                            conv_weight.to(device),
                            output,
                            &conv_qscale);
    EXPECT_EQ(output.dtype(), dtype);
    expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DGroup) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const float error = GetParam().error;
    const StorageView expected({2, 2, 2}, std::vector<float>{
            0.142335, 0.103515, 0.735452, 0.755268, 0.109328, 0.007098, 0.791004, 0.537695});
    const StorageView conv_input({2, 4, 4}, std::vector<float>{
            0.769843, 0.147572, 0.195656, 0.823936, 0.363211, 0.584773, 0.315626, 0.929829,
            0.724258, 0.853388, 0.756254, 0.791604, 0.463644, 0.285105, 0.952018, 0.660709,
            0.557387, 0.147298, 0.473786, 0.566577, 0.255724, 0.488177, 0.534283, 0.678067,
            0.760340, 0.024571, 0.559195, 0.978376, 0.473044, 0.351244, 0.824801, 0.077629});
    const StorageView conv_weight({2, 2, 3}, std::vector<float>{
            0.345985f, -0.071498f, 0.200554f, 0.185144f, -0.015271f, 0.014293f,
            0.006771f, -0.078667f, -0.065937f, 0.382823f, 0.276695f, 0.352038f});
    const StorageView conv_bias({2}, std::vector<float>{-0.215535f, 0.256019f});
    StorageView output(dtype, device);
    ops::Conv1D(1, 0, 1, 2)(conv_input.to(device).to(dtype),
                            conv_weight.to(device).to(dtype),
                            conv_bias.to(device).to(dtype),
                            output);
    EXPECT_EQ(output.dtype(), dtype);
    expect_storage_eq(output.to_float32(), expected, error);
}

static const StorageView bias_value({2, 2, 2}, std::vector<float>{
    -0.356441, 0.914919, -1.626291, 1.481655,
    -1.301039, 0.617229, -0.864480, 1.266894});
static const StorageView bias_bias({2}, std::vector<float>{
    -0.099073, 0.898503});

TEST_P(OpDeviceFPTest, BiasAddGELU) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const float error = GetParam().error;
    const StorageView expected({2, 2, 2}, std::vector<float>{
            -0.147755, 1.750164, -0.072864, 2.359563,
            -0.113045, 1.417522, -0.161525, 2.132529});
    StorageView output(dtype, device);
    const ops::ActivationType activation_type = ops::ActivationType::GELU;
    ops::BiasAdd bias_add_op(&activation_type);
    bias_add_op(bias_value.to(device).to(dtype), bias_bias.to(device).to(dtype), output);
    EXPECT_EQ(output.dtype(), dtype);
    expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, BiasAddAxisGELU) {
    const Device device = GetParam().device;
    const DataType dtype = GetParam().dtype;
    const float error = GetParam().error;
    const StorageView expected({2, 2, 2}, std::vector<float>{
            -0.147755, 0.646726, -0.169845, 2.359563,
            -0.113045, 0.361582, 0.017473, 2.132529});
    StorageView output(dtype, device);
    const ops::ActivationType activation_type = ops::ActivationType::GELU;
    ops::BiasAdd bias_add_op(&activation_type, -2);
    bias_add_op(bias_value.to(device).to(dtype), bias_bias.to(device).to(dtype), output);
    EXPECT_EQ(output.dtype(), dtype);
    expect_storage_eq(output.to_float32(), expected, error);
}

namespace {

void expect_snake_reference(const StorageView& x, const StorageView& alpha,
                            const StorageView& y) {
  ASSERT_EQ(y.shape(), x.shape());
  for (dim_t i = 0; i < x.size(); ++i) {
    const double a = alpha.data<float>()[(i / x.dim(2)) % x.dim(1)];
    const double value = x.data<float>()[i];
    const double expected = value + std::pow(std::sin(a * value), 2) / (a + 1e-9);
    EXPECT_NEAR(y.data<float>()[i], expected, 2e-6) << "index=" << i;
  }
}
}

TEST(OpTest, SnakeAnalyticalAndChannelBroadcast) {
  const float pi = std::acos(-1.f);
  StorageView x({2, 2, 2}, std::vector<float>{0, pi / 2, 0, pi / 4, -pi / 2, pi, -pi / 4, pi / 2});
  StorageView alpha({2}, std::vector<float>{1, 2});
  StorageView y;
  ops::Snake()(x, alpha, y);
  const std::vector<float> expected{0,  pi / 2 + 1,    0,     pi / 4 + .5f, -pi / 2 + 1,
                                    pi, -pi / 4 + .5f, pi / 2};
  ASSERT_EQ(y.shape(), x.shape());
  for (dim_t i = 0; i < y.size(); ++i)
    EXPECT_NEAR(y.data<float>()[i], expected[i], 1e-6);
}

TEST(OpTest, SnakeIndependentDoublePrecisionEquation) {
  StorageView x({2, 3, 11}, DataType::FLOAT32);
  for (dim_t i = 0; i < x.size(); ++i)
    x.data<float>()[i] = float(i - 30) / 9;
  StorageView alpha({1, 3, 1}, std::vector<float>{.25f, -1.5f, 3.f});
  StorageView y;
  ops::Snake()(x, alpha, y);
  expect_snake_reference(x, alpha, y);
}

TEST(OpTest, SnakeZeroAndTinyAlpha) {
  StorageView x({1, 2, 2}, std::vector<float>{-2, 3, 1, -1});
  StorageView alpha({2}, std::vector<float>{0, 1e-9f});
  StorageView y;
  ops::Snake()(x, alpha, y);
  expect_storage_eq(y, x);
}

TEST(OpTest, SnakeParallelChannelBoundaries) {
  StorageView x({2, 3, 10001}, DataType::FLOAT32);
  StorageView alpha({3}, std::vector<float>{.125f, 1.f, 2.5f});
  for (dim_t i = 0; i < x.size(); ++i)
    x.data<float>()[i] = float(i % 73 - 36) / 13;
  StorageView y;
  ops::Snake()(x, alpha, y);
  expect_snake_reference(x, alpha, y);
}

TEST(OpTest, SnakeEmptyTime) {
  StorageView x({1, 2, 0}, DataType::FLOAT32);
  StorageView alpha({2}, std::vector<float>{1, 2});
  StorageView y;
  ops::Snake()(x, alpha, y);
  EXPECT_EQ(y.shape(), x.shape());
  EXPECT_TRUE(y.empty());
}

TEST(OpTest, SnakeInvalidShapeAndDtype) {
  StorageView x({1, 2, 3}, 1.f);
  StorageView alpha({3}, 1.f);
  StorageView y;
  EXPECT_THROW(ops::Snake()(x, alpha, y), std::invalid_argument);
  StorageView wrong_rank({2, 3}, 1.f);
  EXPECT_THROW(ops::Snake()(wrong_rank, alpha, y), std::invalid_argument);
  StorageView valid_alpha({2}, 1.f);
  StorageView integers({1, 2, 3}, int32_t(1));
  EXPECT_THROW(ops::Snake()(integers, valid_alpha, y), std::invalid_argument);
}

TEST(OpTest, SnakeRejectsAliasingIncludingClearedOutput) {
  StorageView x({1, 1, 3}, 1.f);
  StorageView alpha({1}, 1.f);
  EXPECT_THROW(ops::Snake()(x, alpha, x), std::invalid_argument);
  EXPECT_THROW(ops::Snake()(x, alpha, alpha), std::invalid_argument);
  StorageView y({8}, 0.f);
  StorageView view({1, 1, 3}, y.data<float>() + 2);
  y.clear();
  EXPECT_THROW(ops::Snake()(view, alpha, y), std::invalid_argument);
}

TEST_P(OpDeviceTest, SnakeMatchesCpuAcrossBatchesAlphaLayoutsAndEmptyTime) {
  const Device device = GetParam();
  for (dim_t time : {0, 11, 10001}) {
    for (const Shape& shape : {Shape{3}, Shape{1, 3, 1}}) {
      StorageView x({2, 3, time});
      for (dim_t i = 0; i < x.size(); ++i)
        x.data<float>()[i] = float(i % 73 - 36) / 13;
      StorageView alpha(shape, std::vector<float>{0.f, 1e-9f, -1.5f});
      StorageView expected, actual(DataType::FLOAT32, device);
      ops::Snake()(x, alpha, expected);
      auto dx = x.to(device), da = alpha.to(device);
      ops::Snake()(dx, da, actual);
      expect_storage_eq(actual, expected, 2e-6f);
    }
  }
}

TEST_P(OpDeviceTest, SnakeMatchesCpuBeyondCappedGrid) {
  const Device device = GetParam();
  // More elements than 65535 blocks of 256 threads, with a partial final block.
  StorageView x({1, 2, 65535 * 128 + 17}, 0.1f);
  StorageView alpha({2}, std::vector<float>{1.f, -1.5f});
  StorageView expected, actual(DataType::FLOAT32, device);
  ops::Snake()(x, alpha, expected);
  auto dx = x.to(device), da = alpha.to(device);
  ops::Snake()(dx, da, actual);
  expect_storage_eq(actual, expected, 2e-6f);
}

TEST_P(OpDeviceTest, SnakeValidationLeavesOutputUntouched) {
  const Device device = GetParam();
  StorageView host_x({1, 1, 1}, 1.f), host_alpha({1}, 1.f);
  auto x = host_x.to(device), alpha = host_alpha.to(device);
  StorageView y({1}, 42.f, device);
  if (device == Device::CUDA)
    EXPECT_THROW(ops::Snake()(x, host_alpha, y), std::invalid_argument);
  EXPECT_THROW(ops::Snake()(x, alpha, x), std::invalid_argument);
  StorageView retained({8}, 0.f, device);
  StorageView alias({1, 1, 1}, retained.data<float>() + 2, device);
  retained.clear();
  EXPECT_THROW(ops::Snake()(alias, alpha, retained), std::invalid_argument);
  expect_storage_eq(y, StorageView({1}, 42.f));
}

TEST_P(OpDeviceTest, MoeTopKParallelRoutingMatchesCpuAcrossWidthsAndTies) {
  const Device device = GetParam();
  for (dim_t width : {1, 17, 256, 513}) {
    for (dim_t k : {dim_t(1), std::min(dim_t(8), width), width}) {
      SCOPED_TRACE("width=" + std::to_string(width) + ", k=" + std::to_string(k));
      StorageView logits({3, width});
      for (dim_t e = 0; e < width; ++e) {
        logits.data<float>()[e] = 0.f;
        logits.data<float>()[width + e] = float((e * 37) % 19 - 9);
        logits.data<float>()[2 * width + e] = 10000.f + float(e % 13);
      }
      StorageView expected_ids(DataType::INT32), expected_weights;
      ops::MoeTopK{k}(logits, expected_ids, expected_weights);
      auto uploaded = logits.to(device);
      StorageView ids(DataType::INT32, device), weights(DataType::FLOAT32, device);
      ops::MoeTopK{k}(uploaded, ids, weights);
      expect_storage_eq(ids, expected_ids);
      expect_storage_eq(weights, expected_weights, 1e-7f);
    }
  }
}

TEST(OpTest, MoeTopKAnalyticSelectedSoftmax) {
  StorageView logits({2, 4}, std::vector<float>{0, 1, 2, 3, 1, 3, 2, 0});
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(2)(logits, indices, weights);
  const std::vector<int32_t> expected{3, 2, 1, 2};
  EXPECT_EQ(indices.shape(), (Shape{2, 2}));
  EXPECT_EQ(weights.shape(), indices.shape());
  for (dim_t i = 0; i < 4; ++i) {
    EXPECT_EQ(indices.data<int32_t>()[i], expected[i]);
    EXPECT_NEAR(weights.data<float>()[i], i % 2 ? 1 / (1 + std::exp(1.)) : 1 / (1 + std::exp(-1.)),
                1e-7);
  }
}

static void check_moe_topk_nonfinite(Device device) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  StorageView host({6, 4}, std::vector<float>{
      -1, -2, -inf, -3, inf, -1, inf, -inf, nan, -inf, -1, nan,
      -inf, -inf, -inf, -inf, nan, nan, nan, nan, nan, -inf, nan, -inf});
  const int32_t order[][4] = {{0, 1, 3, 2}, {0, 2, 1, 3}, {2, 1, 0, 3},
                              {0, 1, 2, 3}, {0, 1, 2, 3}, {1, 3, 0, 2}};
  const auto logits = host.to(device);
  for (dim_t k : {1, 2, 4}) {
    SCOPED_TRACE(k);
    StorageView ids(DataType::INT32, device), weights(device);
    ops::MoeTopK{k}(logits, ids, weights);
    const auto host_ids = ids.to(Device::CPU), host_weights = weights.to(Device::CPU);
    StorageView cpu_ids(DataType::INT32), cpu_weights;
    ops::MoeTopK{k}(host, cpu_ids, cpu_weights);
    expect_storage_eq(host_ids, cpu_ids);
    expect_storage_eq(host_weights, cpu_weights);
    for (dim_t row = 0; row < 6; ++row) {
      for (dim_t i = 0; i < k; ++i) {
        EXPECT_EQ(host_ids.data<int32_t>()[row * k + i], order[row][i]);
        const double expected = row == 0
          ? (i < 3 ? std::exp(-double(i)) : 0.)
            / (1. + (k >= 2 ? std::exp(-1.) : 0.) + (k == 4 ? std::exp(-2.) : 0.))
          : row == 1 ? (i < 2 ? 1. / std::min<dim_t>(k, 2) : 0.)
          : row == 2 && i == 0 ? 1. : 0.;
        EXPECT_NEAR(host_weights.data<float>()[row * k + i], expected, 1e-7);
      }
    }
  }
}

TEST(OpTest, MoeTopKNonfiniteLogits) {
  check_moe_topk_nonfinite(Device::CPU);
}

TEST_P(OpDeviceTest, MoeTopKNonfiniteLogits) {
  check_moe_topk_nonfinite(GetParam());
}

TEST(OpTest, MoeTopKDeterministicTiesAndSingleExpert) {
  StorageView logits({1, 5}, std::vector<float>{2, 3, 3, 3, 1});
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(2)(logits, indices, weights);
  EXPECT_EQ(indices.data<int32_t>()[0], 1);
  EXPECT_EQ(indices.data<int32_t>()[1], 2);
  EXPECT_EQ(weights.data<float>()[0], .5f);
  EXPECT_EQ(weights.data<float>()[1], .5f);
  ops::MoeTopK(1)(logits, indices, weights);
  EXPECT_EQ(indices.data<int32_t>()[0], 1);
  EXPECT_EQ(weights.data<float>()[0], 1.f);
}

TEST(OpTest, MoeTopKLargeFiniteLogits) {
  const float largest = std::numeric_limits<float>::max();
  StorageView logits({2, 3},
                     std::vector<float>{largest, largest, -largest, -10000, -10001, -10002});
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(3)(logits, indices, weights);
  EXPECT_EQ(weights.data<float>()[0], .5f);
  EXPECT_EQ(weights.data<float>()[1], .5f);
  EXPECT_EQ(weights.data<float>()[2], 0.f);
  EXPECT_NEAR(weights.data<float>()[3], 1. / (1 + std::exp(-1.) + std::exp(-2.)), 1e-7);
}

TEST(OpTest, MoeTopKIndependentFullSoftmaxAndRowPermutation) {
  StorageView logits({13, 17}, DataType::FLOAT32);
  for (dim_t i = 0; i < logits.size(); ++i)
    logits.data<float>()[i] = float((i * 19) % 61 - 30) / 7;
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(8)(logits, indices, weights);
  for (dim_t row = 0; row < 13; ++row) {
    std::vector<double> full(17);
    double total = 0;
    for (dim_t e = 0; e < 17; ++e)
      total += full[e] = std::exp(double(logits.data<float>()[row * 17 + e]));
    for (auto& p : full)
      p /= total;
    auto sorted = full;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    double selected_total = 0;
    for (dim_t i = 0; i < 8; ++i)
      selected_total += sorted[i];
    std::set<int32_t> seen;
    double row_sum = 0;
    for (dim_t i = 0; i < 8; ++i) {
      const auto e = indices.data<int32_t>()[row * 8 + i];
      ASSERT_GE(e, 0);
      ASSERT_LT(e, 17);
      EXPECT_TRUE(seen.insert(e).second);
      EXPECT_NEAR(full[e], sorted[i], 1e-12);
      EXPECT_NEAR(weights.data<float>()[row * 8 + i], full[e] / selected_total, 1e-7);
      row_sum += weights.data<float>()[row * 8 + i];
    }
    EXPECT_NEAR(row_sum, 1, 1e-7);
  }
}

TEST(OpTest, MoeTopKEmptyRows) {
  StorageView logits({0, 3}, DataType::FLOAT32);
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(2)(logits, indices, weights);
  EXPECT_EQ(indices.shape(), (Shape{0, 2}));
  EXPECT_EQ(weights.shape(), indices.shape());
}

TEST(OpTest, MoeTopKValidation) {
  EXPECT_THROW(ops::MoeTopK(0), std::invalid_argument);
  EXPECT_THROW(ops::MoeTopK(-1), std::invalid_argument);
  StorageView logits({1, 3}, 1.f);
  StorageView indices(DataType::INT32), weights;
  EXPECT_THROW(ops::MoeTopK(4)(logits, indices, weights), std::invalid_argument);
  StorageView rank1({3}, 1.f);
  EXPECT_THROW(ops::MoeTopK(2)(rank1, indices, weights), std::invalid_argument);
  StorageView wrong_indices;
  EXPECT_THROW(ops::MoeTopK(2)(logits, wrong_indices, weights), std::invalid_argument);
}

TEST(OpTest, MoeTopKRetainedCapacityAliasing) {
  StorageView weights({12}, 0.f);
  StorageView logits({1, 3}, weights.data<float>() + 5);
  weights.clear();
  StorageView indices(DataType::INT32);
  EXPECT_THROW(ops::MoeTopK(2)(logits, indices, weights), std::invalid_argument);
  StorageView independent_logits({1, 3}, 1.f);
  StorageView overlapping_indices({2}, reinterpret_cast<int32_t*>(weights.data<float>() + 2));
  EXPECT_THROW(ops::MoeTopK(2)(independent_logits, overlapping_indices, weights),
               std::invalid_argument);
}

TEST_P(OpDeviceTest, MoeTopKValidationAndEmptyRows) {
  const Device device = GetParam();
  StorageView host({1, 3}, 1.f);
  auto logits = host.to(device);
  StorageView ids(DataType::INT32, device), weights(DataType::FLOAT32, device);
  EXPECT_THROW(ops::MoeTopK(4)(logits, ids, weights), std::invalid_argument);
  if (device == Device::CUDA)
    EXPECT_THROW(ops::MoeTopK(2)(host, ids, weights), std::invalid_argument);
  EXPECT_THROW(ops::MoeTopK(2)(logits, ids, logits), std::invalid_argument);
  StorageView empty({0, 3}, DataType::FLOAT32, device);
  ops::MoeTopK(2)(empty, ids, weights);
  EXPECT_EQ(ids.shape(), (Shape{0, 2}));
  EXPECT_EQ(weights.shape(), ids.shape());

}

TEST(OpTest, DequantizeFP8EveryFiniteEncodingAndSignedBytes) {
  StorageView input({1, 254}, DataType::INT8), scale({1, 2}, 2.f), output;
  auto* bytes = reinterpret_cast<uint8_t*>(input.data<int8_t>());
  dim_t column = 0;
  for (unsigned code = 0; code < 256; ++code)
    if ((code & 127) != 127)
      bytes[column++] = code;
  ops::DequantizeFP8()(input, scale, output);
  column = 0;
  for (unsigned code = 0; code < 256; ++code) {
    if ((code & 127) == 127)
      continue;
    const unsigned exponent = (code & 127) / 8, mantissa = code % 8;
    const double magnitude = exponent == 0
                                 ? double(mantissa) / 512
                                 : (1. + double(mantissa) / 8) * std::pow(2., int(exponent) - 7);
    const float expected = float((code & 128 ? -magnitude : magnitude) * 2);
    EXPECT_EQ(output.data<float>()[column], expected);
    EXPECT_EQ(std::signbit(output.data<float>()[column]), std::signbit(expected));
    ++column;
  }
}

TEST(OpTest, DequantizeFP8EdgeBlocksAndFloat32ScaleMultiplication) {
  StorageView input({129, 130}, int8_t(0x38)); // E4M3FN 1.0.
  StorageView scale({2, 2}, std::vector<float>{.1f, .3f, 1.7f, 9.1f}), output;
  ops::DequantizeFP8()(input, scale, output);
  EXPECT_EQ(output.shape(), input.shape());
  for (dim_t r = 0; r < 129; ++r)
    for (dim_t c = 0; c < 130; ++c)
      EXPECT_EQ(output.data<float>()[r * 130 + c], scale.data<float>()[(r / 128) * 2 + c / 128]);
}

TEST(OpTest, DequantizeFP8RejectsMalformedShapesTypesAndAliasing) {
  StorageView input({1, 1}, int8_t(0x38)), scale({1, 1}, 1.f), output;
  for (const Shape& shape : {Shape{1}, Shape{0, 1}, Shape{1, 1, 1}}) {
    StorageView bad(shape, DataType::INT8);
    EXPECT_THROW(ops::DequantizeFP8()(bad, scale, output), std::invalid_argument);
  }
  StorageView bad_scale({2, 1}, 1.f), bad_rank({1}, 1.f), bad_type({1, 1}, int8_t(1));
  EXPECT_THROW(ops::DequantizeFP8()(input, bad_scale, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, bad_rank, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(scale, scale, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, bad_type, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, bad_type), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, scale), std::invalid_argument);
  StorageView byte_alias({1, 1}, reinterpret_cast<float*>(input.data<int8_t>()));
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, byte_alias), std::invalid_argument);
  StorageView alias({1, 1}, scale.data<float>());
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, alias), std::invalid_argument);
}

TEST_P(OpDeviceTest, DequantizeFP8MatchesCpuAndValidatesMetadata) {
  const Device device = GetParam();
  StorageView input({129, 130}, DataType::INT8),
      scale({2, 2}, std::vector<float>{.1f, .3f, 1.7f, 9.1f});
  for (dim_t i = 0; i < input.size(); ++i) {
    const uint8_t code = i % 256;
    input.data<int8_t>()[i] = int8_t((code & 127) == 127 ? 0 : code);
  }
  StorageView expected, actual(DataType::FLOAT32, device);
  ops::DequantizeFP8()(input, scale, expected);
  auto device_input = input.to(device), device_scale = scale.to(device);
  ops::DequantizeFP8()(device_input, device_scale, actual);
  expect_storage_eq(actual, expected);
  if (device == Device::CUDA)
    EXPECT_THROW(ops::DequantizeFP8()(device_input, scale, actual), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(device_input, device_scale, device_scale), std::invalid_argument);
}

TEST_P(OpDeviceTest, DequantizeFP8DecodesNaNCodes) {
  const Device device = GetParam();
  StorageView input({1, 2}, std::vector<int8_t>{int8_t(0x7f), int8_t(0xff)}, device);
  StorageView scale({1, 1}, 1.f, device), output(device);
  ops::DequantizeFP8()(input, scale, output);
  const auto host = output.to(Device::CPU);
  EXPECT_TRUE(std::isnan(host.data<float>()[0]));
  EXPECT_TRUE(std::isnan(host.data<float>()[1]));
}

TEST(OpTest, CausalDepthwiseConv1DAnalyticalNonzeroHistory) {
  StorageView x({1, 3, 1}, std::vector<float>{3, 4, 5});
  StorageView weight({1, 3}, std::vector<float>{1, 10, 100});
  StorageView history({1, 1, 2}, std::vector<float>{1, 2});
  StorageView y;
  ops::CausalDepthwiseConv1D(false)(x, weight, nullptr, history, y);
  expect_storage_eq(y, StorageView({1, 3, 1}, std::vector<float>{321, 432, 543}));
  expect_storage_eq(history, StorageView({1, 1, 2}, std::vector<float>{4, 5}));
}

TEST(OpTest, CausalDepthwiseConv1DBatchChannelBiasAndSilu) {
  StorageView x({2, 2, 2}, std::vector<float>{1, 2, 3, 4, -1, -2, -3, -4});
  StorageView weight({2, 2}, std::vector<float>{2, 3, -1, .5f});
  StorageView bias({2}, std::vector<float>{1, -2});
  StorageView history({2, 2, 1}, std::vector<float>{4, 5, -4, -5});
  StorageView y;
  ops::CausalDepthwiseConv1D()(x, weight, &bias, history, y);
  const std::vector<double> preactivation{12, -6, 12, -2, -10, 2, -10, -2};
  for (dim_t i = 0; i < y.size(); ++i)
    EXPECT_NEAR(y.data<float>()[i], preactivation[i] / (1 + std::exp(-preactivation[i])), 1e-6);
  expect_storage_eq(history, StorageView({2, 2, 1}, std::vector<float>{3, 4, -3, -4}));
}

TEST(OpTest, CausalDepthwiseConv1DKernelOneAndEmptyTime) {
  StorageView x({1, 2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView weight({2, 1}, std::vector<float>{2, 3});
  StorageView state({1, 2, 0}, DataType::FLOAT32), y;
  ops::CausalDepthwiseConv1D(false)(x, weight, nullptr, state, y);
  expect_storage_eq(y, StorageView({1, 2, 2}, std::vector<float>{2, 6, 6, 12}));
  StorageView empty({1, 0, 2}, DataType::FLOAT32);
  StorageView wider_weight({2, 3}, 1.f), wider_state({1, 2, 2}, 7.f);
  ops::CausalDepthwiseConv1D()(empty, wider_weight, nullptr, wider_state, y);
  EXPECT_EQ(y.shape(), empty.shape());
  expect_storage_eq(wider_state, StorageView({1, 2, 2}, 7.f));
}

TEST(OpTest, CausalDepthwiseConv1DGuards) {
  StorageView x({1, 2, 2}, 1.f), weight({2, 3}, 1.f), state({1, 2, 2}, 0.f), y;
  StorageView bad_weight({3, 2}, 1.f), bad_state({1, 2, 3}, 0.f), bad_bias({1}, 0.f);
  ops::CausalDepthwiseConv1D op;
  EXPECT_THROW(op(x, bad_weight, nullptr, state, y), std::invalid_argument);
  EXPECT_THROW(op(x, weight, nullptr, bad_state, y), std::invalid_argument);
  EXPECT_THROW(op(x, weight, &bad_bias, state, y), std::invalid_argument);
  StorageView integers({1, 2, 2}, int32_t(1));
  EXPECT_THROW(op(integers, weight, nullptr, state, y), std::invalid_argument);
  expect_storage_eq(state, StorageView({1, 2, 2}, 0.f));
}

TEST(OpTest, CausalDepthwiseConv1DAliasingRetainedAllocations) {
  StorageView x({1, 2, 2}, 1.f), weight({2, 3}, 1.f), state({1, 2, 2}, 0.f), y;
  ops::CausalDepthwiseConv1D op;
  EXPECT_THROW(op(x, weight, nullptr, state, x), std::invalid_argument);
  EXPECT_THROW(op(x, weight, nullptr, state, state), std::invalid_argument);
  StorageView input_alias({1, 2, 2}, state.data<float>());
  EXPECT_THROW(op(input_alias, weight, nullptr, state, y), std::invalid_argument);
  StorageView buffer({12}, 0.f);
  StorageView input_view({1, 2, 2}, buffer.data<float>() + 4);
  buffer.clear();
  EXPECT_THROW(op(input_view, weight, nullptr, state, buffer), std::invalid_argument);
}

TEST(OpTest, ConvTranspose1DScattersOverlappingKernelWithoutReversal) {
  StorageView x({1, 1, 3}, std::vector<float>{1, 2, 3});
  StorageView w({1, 1, 3}, std::vector<float>{1, 10, 100});
  StorageView y;
  ops::ConvTranspose1D()(x, w, y);
  expect_storage_eq(y, StorageView({1, 1, 5}, std::vector<float>{1, 12, 123, 230, 300}));
}

TEST(OpTest, ConvTranspose1DSumsInputChannelsAndAddsBiasOnceForEachBatch) {
  StorageView x({2, 2, 1}, std::vector<float>{2, 3, 4, 5});
  StorageView w({2, 2, 1}, std::vector<float>{1, 10, 100, 1000});
  StorageView b({2}, std::vector<float>{-1, 2});
  StorageView y;
  ops::ConvTranspose1D()(x, w, b, y);
  expect_storage_eq(y, StorageView({2, 2, 1}, std::vector<float>{301, 3022, 503, 5042}));
}

TEST(OpTest, ConvTranspose1DGroupsDoNotMixChannels) {
  StorageView x({1, 2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView w({2, 2, 1}, std::vector<float>{1, 10, 100, 1000});
  StorageView y;
  ops::ConvTranspose1D(1, 0, 0, 1, 2)(x, w, y);
  expect_storage_eq(y,
                    StorageView({1, 4, 2}, std::vector<float>{1, 2, 10, 20, 300, 400, 3000, 4000}));
}

TEST(OpTest, ConvTranspose1DDilationPaddingAndOutputPadding) {
  StorageView x({1, 1, 2}, std::vector<float>{2, 3});
  StorageView w({1, 1, 2}, std::vector<float>{5, 7});
  StorageView b({1}, 1.f), y;
  ops::ConvTranspose1D(2, 1, 1, 2)(x, w, b, y);
  // Uncropped contributions at positions 0,2,4 are 10,29,21.
  expect_storage_eq(y, StorageView({1, 1, 4}, std::vector<float>{1, 30, 1, 22}));
}

TEST(OpTest, ConvTranspose1DOutputPaddingCanBeSmallerThanDilationOnly) {
  StorageView x({1, 1, 1}, 2.f), w({1, 1, 2}, 3.f), y;
  ops::ConvTranspose1D(1, 0, 1, 2)(x, w, y);
  expect_storage_eq(y, StorageView({1, 1, 4}, std::vector<float>{6, 0, 6, 0}));
}

TEST(OpTest, ConvTranspose1DRejectsInvalidParameters) {
  EXPECT_THROW(ops::ConvTranspose1D(0), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, -1), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, -1), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 1), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 0, 1, 0), std::invalid_argument);
}

TEST(OpTest, ConvTranspose1DRejectsInvalidShapesTypesAndLengths) {
  StorageView x({1, 2, 2}, 1.f), w({2, 1, 2}, 1.f), y;
  StorageView rank2({2, 2}, 1.f), channels({1, 1, 2}, 1.f);
  StorageView bias({2}, 1.f), bad_type(x.shape(), DataType::INT32);
  StorageView bad_output(DataType::INT32), empty({1, 2, 0});
  ops::ConvTranspose1D op;
  EXPECT_THROW(op(rank2, w, y), std::invalid_argument);
  EXPECT_THROW(op(x, rank2, y), std::invalid_argument);
  EXPECT_THROW(op(x, channels, y), std::invalid_argument);
  EXPECT_THROW(op(x, w, bias, y), std::invalid_argument);
  EXPECT_THROW(op(bad_type, w, y), std::invalid_argument);
  EXPECT_THROW(op(x, w, bad_output), std::invalid_argument);
  EXPECT_THROW(op(empty, w, y), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 0, 1, 3)(x, w, y), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 10)(x, w, y), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D{std::numeric_limits<dim_t>::max()}(x, w, y),
               std::invalid_argument);
}

TEST(OpTest, ConvTranspose1DRejectsDirectAndPartialOutputAliasing) {
  StorageView x({1, 1, 3}, 1.f), w({1, 1, 2}, 1.f), b({1}, 0.f);
  ops::ConvTranspose1D op;
  EXPECT_THROW(op(x, w, x), std::invalid_argument);
  EXPECT_THROW(op(x, w, w), std::invalid_argument);
  EXPECT_THROW(op(x, w, b, b), std::invalid_argument);
  StorageView partial({1}, x.data<float>() + 1);
  EXPECT_THROW(op(x, w, partial), std::invalid_argument);
}

TEST(OpTest, ConvTranspose1DRejectsAliasingInRetainedOutputCapacity) {
  StorageView y({8}, 1.f);
  StorageView x({1, 1, 2}, y.data<float>() + 4), w({1, 1, 2}, 1.f);
  y.resize({0});
  EXPECT_THROW(ops::ConvTranspose1D()(x, w, y), std::invalid_argument);
}

namespace {

  void compare_transpose_with_scalar(Device device, dim_t batch, dim_t cin, dim_t cout_group, dim_t time,
                                     dim_t kernel, dim_t stride, dim_t padding,
                                     dim_t output_padding, dim_t dilation, dim_t groups,
                                     bool use_bias = true) {
    StorageView x({batch, cin, time}), w({cin, cout_group, kernel});
    const dim_t cout = cout_group * groups;
    StorageView bias(Shape{cout});
    for (dim_t i = 0; i < x.size(); ++i)
      x.data<float>()[i] = (i % 17 - 8) * .03125f;
    for (dim_t i = 0; i < w.size(); ++i)
      w.data<float>()[i] = (i % 13 - 6) * .015625f;
    for (dim_t i = 0; i < cout; ++i)
      bias.data<float>()[i] = i * .00390625f;
    const dim_t length =
        (time - 1) * stride - 2 * padding + (kernel - 1) * dilation + output_padding + 1;
    StorageView expected({batch, cout, length});
    for (dim_t n = 0; n < batch; ++n)
      for (dim_t oc = 0; oc < cout; ++oc) {
        auto* row = expected.data<float>() + (n * cout + oc) * length;
        std::fill_n(row, length, use_bias ? bias.data<float>()[oc] : 0.f);
        const dim_t group = oc / cout_group;
        for (dim_t ic = group * (cin / groups); ic < (group + 1) * (cin / groups); ++ic)
          for (dim_t t = 0; t < time; ++t)
            for (dim_t k = 0; k < kernel; ++k) {
              const dim_t pos = t * stride - padding + k * dilation;
              if (pos >= 0 && pos < length)
                row[pos] += x.data<float>()[(n * cin + ic) * time + t]
                            * w.data<float>()[(ic * cout_group + oc % cout_group) * kernel + k];
            }
      }
    StorageView actual(device);
    ops::ConvTranspose1D op(stride, padding, output_padding, dilation, groups);
    if (use_bias)
      op(x.to(device), w.to(device), bias.to(device), actual);
    else
      op(x.to(device), w.to(device), actual);
    expect_storage_eq(actual, expected, 1e-6f);
  }
}

TEST_P(OpDeviceTest, ConvTranspose1DGemmSizedInputsMatchScalarAcrossGroupsAndBoundaryParameters) {
  const Device device = GetParam();
  for (dim_t groups : {1, 2})
    for (dim_t dilation : {1, 2})
      for (dim_t output_padding : {0, 1})
        compare_transpose_with_scalar(device, 2, 16, 5, 7, 3, 2, 1, output_padding, dilation, groups);
}

TEST_P(OpDeviceTest, ConvTranspose1DGemmTimeTilesPreserveOverlapsAndBiasAcrossTileBoundary) {
  const Device device = GetParam();
  // Contributions alone exceed the 16 MiB workspace bound, forcing multiple
  // input tiles whose overlapping output regions must accumulate, not reset.
  compare_transpose_with_scalar(device, 1, 8, 256, 4096, 5, 2, 1, 1, 2, 1);
}

TEST(OpTest, GatedDeltaNetDecaysStateBeforeComputingResidual) {
  StorageView q({1, 1, 1, 2}, std::vector<float>{2, 1});
  StorageView k({1, 1, 1, 2}, std::vector<float>{1, 2});
  StorageView v({1, 1, 1, 2}, std::vector<float>{11, 6});
  StorageView g({1, 1, 1}, std::log(0.5f));
  StorageView beta({1, 1, 1}, 0.25f);
  StorageView state({1, 1, 2, 2}, std::vector<float>{2, 4, 6, 8});
  StorageView output;
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_storage_eq(state, StorageView({1, 1, 2, 2}, std::vector<float>{2, 1, 5, 2}), 1e-6f);
  expect_storage_eq(
      output, StorageView({1, 1, 1, 2}, std::vector<float>{9 / std::sqrt(2.f), 4 / std::sqrt(2.f)}),
      1e-6f);
}

TEST(OpTest, GatedDeltaNetConsecutiveValueHeadsShareKeysAndQueries) {
  StorageView q({1, 1, 2, 1}, std::vector<float>{2, 3});
  StorageView k({1, 1, 2, 1}, std::vector<float>{1, 2});
  StorageView v({1, 1, 4, 1}, std::vector<float>{1, 2, 3, 4});
  StorageView g({1, 1, 4}, 0.f);
  StorageView beta({1, 1, 4}, 0.5f);
  StorageView state({1, 4, 1, 1}, 0.f);
  StorageView output;
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_storage_eq(output, StorageView({1, 1, 4, 1}, std::vector<float>{1, 2, 9, 12}));
  expect_storage_eq(state, StorageView({1, 4, 1, 1}, std::vector<float>{0.5f, 1, 3, 4}));
}

TEST(OpTest, GatedDeltaNetPrefillMatchesSingleStepsWithNonzeroState) {
  const dim_t batch = 2, steps = 5, key_heads = 2, heads = 4, kd = 3, vd = 2;
  StorageView q({batch, steps, key_heads, kd});
  StorageView k(q.shape());
  StorageView v({batch, steps, heads, vd});
  StorageView g({batch, steps, heads});
  StorageView beta(g.shape());
  StorageView initial({batch, heads, kd, vd});
  for (dim_t i = 0; i < q.size(); ++i) {
    q.data<float>()[i] = float(i % 11 - 5) / 7;
    k.data<float>()[i] = float(i % 7 - 3) / 9;
  }
  for (dim_t i = 0; i < v.size(); ++i)
    v.data<float>()[i] = float(i % 13 - 6) / 5;
  for (dim_t i = 0; i < g.size(); ++i) {
    g.data<float>()[i] = -float(i % 5 + 1) / 10;
    beta.data<float>()[i] = float(i % 4 + 1) / 5;
  }
  for (dim_t i = 0; i < initial.size(); ++i)
    initial.data<float>()[i] = float(i % 9 - 4) / 8;
  StorageView prefill_state(initial), step_state(initial), prefill;
  ops::GatedDeltaNet()(q, k, v, g, beta, prefill_state, prefill);
  // Slice each batch explicitly: a time slice across batches is not contiguous.
  for (dim_t b = 0; b < batch; ++b) {
    StorageView state({1, heads, kd, vd}, step_state.data<float>() + b * heads * kd * vd);
    for (dim_t t = 0; t < steps; ++t) {
      const auto row = b * steps + t;
      StorageView qt({1, 1, key_heads, kd}, q.data<float>() + row * key_heads * kd);
      StorageView kt({1, 1, key_heads, kd}, k.data<float>() + row * key_heads * kd);
      StorageView vt({1, 1, heads, vd}, v.data<float>() + row * heads * vd);
      StorageView gt({1, 1, heads}, g.data<float>() + row * heads);
      StorageView bt({1, 1, heads}, beta.data<float>() + row * heads);
      StorageView output;
      ops::GatedDeltaNet()(qt, kt, vt, gt, bt, state, output);
      expect_array_eq(output.data<float>(), prefill.data<float>() + row * heads * vd, heads * vd,
                      1e-6f);
    }
  }
  expect_storage_eq(step_state, prefill_state, 1e-6f);
}

TEST(OpTest, GatedDeltaNetRejectsInvalidShapesAndTypes) {
  StorageView q({1, 1, 2, 2}, 1.f), k(q);
  StorageView v({1, 1, 4, 3}, 1.f);
  StorageView g({1, 1, 4}, 0.f), beta(g);
  StorageView state({1, 4, 2, 3}, 0.f), output;
  ops::GatedDeltaNet op;
  StorageView bad_key({1, 1, 2, 3}, 1.f);
  EXPECT_THROW(op(q, bad_key, v, g, beta, state, output), std::invalid_argument);
  StorageView bad_value({1, 1, 3, 3}, 1.f);
  EXPECT_THROW(op(q, k, bad_value, g, beta, state, output), std::invalid_argument);
  StorageView bad_gate({1, 4}, 1.f);
  EXPECT_THROW(op(q, k, v, bad_gate, beta, state, output), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, bad_gate, state, output), std::invalid_argument);
  StorageView bad_state({1, 4, 3, 2}, 0.f);
  EXPECT_THROW(op(q, k, v, g, beta, bad_state, output), std::invalid_argument);
  StorageView bad_type(q.shape(), DataType::INT32);
  EXPECT_THROW(op(bad_type, k, v, g, beta, state, output), std::invalid_argument);
  StorageView bad_output(DataType::INT32);
  EXPECT_THROW(op(q, k, v, g, beta, state, bad_output), std::invalid_argument);
}

TEST(OpTest, GatedDeltaNetRejectsAliasingIncludingPartialOverlap) {
  StorageView q({1, 1, 1, 2}, 1.f), k(q), v(q);
  StorageView g({1, 1, 1}, 0.f), beta({1, 1, 1}, 0.5f);
  StorageView state({1, 1, 2, 2}, 0.f), output;
  ops::GatedDeltaNet op;
  EXPECT_THROW(op(q, k, v, g, beta, state, state), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, beta, state, q), std::invalid_argument);
  StorageView overlapping_query({1, 1, 1, 2}, state.data<float>() + 1);
  EXPECT_THROW(op(overlapping_query, k, v, g, beta, state, output), std::invalid_argument);
  StorageView overlapping_output({1}, state.data<float>() + 3);
  EXPECT_THROW(op(q, k, v, g, beta, state, overlapping_output), std::invalid_argument);
  StorageView input_output_alias({1}, v.data<float>() + 1);
  EXPECT_THROW(op(q, k, v, g, beta, state, input_output_alias), std::invalid_argument);
}

TEST(OpTest, GatedDeltaNetRejectsAliasesIntoClearedOutputWithoutMutation) {
  StorageView q({1, 1, 1, 2}, 1.f), k(q), v(q);
  StorageView g({1, 1, 1}, 0.f), beta({1, 1, 1}, 0.5f);
  StorageView state({1, 1, 2, 2}, 2.f);
  StorageView output({16}, 3.f);
  auto* allocation = output.data<float>();
  StorageView aliased_query({1, 1, 1, 2}, allocation + 1);
  output.clear();
  const StorageView original_state(state);
  ops::GatedDeltaNet op;
  EXPECT_THROW(op(aliased_query, k, v, g, beta, state, output), std::invalid_argument);
  expect_storage_eq(state, original_state);
  EXPECT_EQ(output.size(), 0);
  for (int i = 0; i < 16; ++i)
    EXPECT_FLOAT_EQ(allocation[i], 3.f);

  StorageView aliased_state({1, 1, 2, 2}, allocation + 5);
  EXPECT_THROW(op(q, k, v, g, beta, aliased_state, output), std::invalid_argument);
  EXPECT_EQ(output.size(), 0);
  for (int i = 0; i < 16; ++i)
    EXPECT_FLOAT_EQ(allocation[i], 3.f);
}

TEST(OpTest, GatedDeltaNetEmptyTimePreservesState) {
  StorageView q({2, 0, 1, 2}), k(q);
  StorageView v({2, 0, 2, 3});
  StorageView g({2, 0, 2}), beta(g);
  StorageView state({2, 2, 2, 3}, 0.75f);
  const StorageView original(state);
  StorageView output({8}, 9.f);
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_storage_eq(state, original);
  assert_vector_eq(output.shape(), {2, 0, 2, 3});
  EXPECT_EQ(output.size(), 0);
}

TEST(OpTest, GatedDeltaNetMatchesIndependentDenseTransitionReference) {
  // Express the update as A*S + B, with A = exp(g)*(I-beta*k*k^T)
  // and B = beta*k*v^T. Use double matrix arithmetic, independently of
  // the production residual/rank-one update implementation.
  StorageView q({1, 3, 1, 2}, std::vector<float>{0.4f, -0.7f, 1.2f, 0.3f, -0.8f, 0.6f});
  StorageView k({1, 3, 1, 2}, std::vector<float>{1.3f, -0.2f, 0.4f, 1.1f, -0.6f, 0.9f});
  StorageView v({1, 3, 2, 3}, std::vector<float>{1, -2, 3, 4, 0.5f, -1, -0.5f, 2, 1, -3, 0.2f, 0.7f,
                                                 2, 0, -1, 1, -0.4f, 3});
  StorageView g({1, 3, 2}, std::vector<float>{-0.2f, -0.7f, -0.9f, -0.1f, -0.4f, -0.3f});
  StorageView beta({1, 3, 2}, std::vector<float>{0.2f, 0.8f, 0.6f, 0.4f, 0.9f, 0.1f});
  StorageView state({1, 2, 2, 3},
                    std::vector<float>{1, 2, -1, 0.5f, -0.2f, 3, -2, 0.3f, 1, 4, -1, 0.7f});
  std::vector<double> expected_state(state.data<float>(), state.data<float>() + state.size());
  std::vector<float> expected_output(18);
  for (int t = 0; t < 3; ++t) {
    for (int h = 0; h < 2; ++h) {
      const double gate = beta.data<float>()[t * 2 + h];
      const double decay = std::exp(double(g.data<float>()[t * 2 + h]));
      double transition[2][2];
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          transition[i][j] = decay
                             * ((i == j ? 1. : 0.)
                                - gate * k.data<float>()[t * 2 + i] * k.data<float>()[t * 2 + j]);
      const auto previous = expected_state;
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
          double result = gate * k.data<float>()[t * 2 + i] * v.data<float>()[(t * 2 + h) * 3 + j];
          for (int l = 0; l < 2; ++l)
            result += transition[i][l] * previous[(h * 2 + l) * 3 + j];
          expected_state[(h * 2 + i) * 3 + j] = result;
        }
      }
      for (int j = 0; j < 3; ++j) {
        double result = 0;
        for (int i = 0; i < 2; ++i)
          result += q.data<float>()[t * 2 + i] * expected_state[(h * 2 + i) * 3 + j];
        expected_output[(t * 2 + h) * 3 + j] = result / std::sqrt(2.);
      }
    }
  }
  StorageView output;
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_array_eq(output.data<float>(), expected_output.data(), 18, 2e-6f);
  for (dim_t i = 0; i < state.size(); ++i)
    EXPECT_NEAR(state.data<float>()[i], expected_state[i], 2e-6);
}

namespace {

void fill_pattern(StorageView& tensor, float scale = .03125f) {
  for (dim_t i = 0; i < tensor.size(); ++i)
    tensor.data<float>()[i] = float(i % 17 - 8) * scale;
}
}

TEST(OpTest, CausalDepthwiseConv1DTiledMatchesScalar) {
  for (dim_t channels : {1024, 1100})
    for (dim_t kernel : {1, 2, 4})
      for (dim_t steps : {1, 64, 100})
        for (bool silu : {false, true})
          for (bool use_bias : {false, true}) {
            SCOPED_TRACE(::testing::Message() << channels << ", " << kernel << ", " << steps
                         << ", silu=" << silu << ", bias=" << use_bias);
            const dim_t past = kernel - 1;
            StorageView x({2, steps, channels}), w({channels, kernel}), bias(Shape{channels});
            StorageView initial({2, channels, past}), expected(x.shape()), expected_state(initial);
            for (auto* tensor : {&x, &w, &bias, &initial})
              fill_pattern(*tensor);
            // Read a logical concatenation of initial history and input; never mutate it.
            const auto sample = [&](dim_t n, dim_t c, dim_t t) {
              return t < past ? initial.data<float>()[(n * channels + c) * past + t]
                              : x.data<float>()[(n * steps + t - past) * channels + c];
            };
            for (dim_t n = 0; n < 2; ++n)
              for (dim_t c = 0; c < channels; ++c) {
                for (dim_t t = 0; t < steps; ++t) {
                  double value = use_bias ? bias.data<float>()[c] : 0;
                  for (dim_t k = 0; k < kernel; ++k)
                    value += double(sample(n, c, t + k)) * w.data<float>()[c * kernel + k];
                  expected.data<float>()[(n * steps + t) * channels + c] =
                    silu ? value / (1 + std::exp(-value)) : value;
                }
                for (dim_t i = 0; i < past; ++i)
                  expected_state.data<float>()[(n * channels + c) * past + i] = sample(n, c, steps + i);
              }
            StorageView state(initial), actual;
            ops::CausalDepthwiseConv1D{silu}(x, w, use_bias ? &bias : nullptr, state, actual);
            expect_storage_eq(actual, expected, 2e-6f);
            expect_storage_eq(state, expected_state);
          }
}

TEST_P(OpDeviceTest, CausalDepthwiseConv1DMatchesCpuAndPreservesHistory) {
  const Device device = GetParam();
  for (dim_t channels : {5, 1100}) {
    for (dim_t kernel : {1, 4, 7}) {
      for (dim_t steps : {0, 1, 9, 65}) {
        for (bool silu : {false, true}) {
          SCOPED_TRACE("channels=" + std::to_string(channels) + ", kernel=" + std::to_string(kernel)
                       + ", steps=" + std::to_string(steps)
                       + ", silu=" + std::to_string(silu));
          StorageView x({2, steps, channels}), w({channels, kernel}), bias(Shape{channels}),
              initial({2, channels, kernel - 1});
          for (auto* tensor : {&x, &w, &bias, &initial})
            fill_pattern(*tensor);
          ops::CausalDepthwiseConv1D op(silu);
          StorageView expected, expected_state(initial);
          const auto* cpu_bias = silu ? &bias : nullptr;
          op(x, w, cpu_bias, expected_state, expected);
          auto dx = x.to(device), dw = w.to(device), db = bias.to(device), state = initial.to(device);
          StorageView actual(DataType::FLOAT32, device);
          op(dx, dw, silu ? &db : nullptr, state, actual);
          expect_storage_eq(actual, expected, 2e-6f);
          expect_storage_eq(state, expected_state);
        }
      }
    }
  }
}

TEST_P(OpDeviceTest, CausalDepthwiseConv1DArbitraryChunksMatchPrefill) {
  const Device device = GetParam();
  for (dim_t channels : {3, 1100}) {
    SCOPED_TRACE(channels);
    StorageView input({2, 129, channels}), weight({channels, 4}), initial({2, channels, 3});
    for (auto* tensor : {&input, &weight, &initial})
      fill_pattern(*tensor);
    ops::CausalDepthwiseConv1D op;
    StorageView expected, expected_state(initial);
    op(input, weight, nullptr, expected_state, expected);
    auto state = initial.to(device), dw = weight.to(device);
    dim_t offset = 0;
    for (dim_t count : {0, 1, 63, 64, 1}) {
      StorageView chunk({2, count, channels}), expected_chunk(chunk.shape());
      for (dim_t n = 0; n < 2; ++n) {
        if (count) {
          std::copy_n(input.data<float>() + (n * 129 + offset) * channels, count * channels,
                      chunk.data<float>() + n * count * channels);
          std::copy_n(expected.data<float>() + (n * 129 + offset) * channels, count * channels,
                      expected_chunk.data<float>() + n * count * channels);
        }
      }
      auto dx = chunk.to(device);
      StorageView actual(DataType::FLOAT32, device);
      op(dx, dw, nullptr, state, actual);
      expect_storage_eq(actual, expected_chunk, 2e-6f);
      offset += count;
    }
    expect_storage_eq(state, expected_state);
  }
}

TEST_P(OpDeviceTest, CausalDepthwiseConv1DValidationDoesNotMutateHistoryOrOutput) {
  const Device device = GetParam();
  StorageView x({1, 2, 2}, 1.f), w({2, 3}, .5f), bias({2}, .25f), initial({1, 2, 2}, 2.f);
  auto dx = x.to(device), dw = w.to(device), db = bias.to(device), state = initial.to(device);
  StorageView actual({1}, 42.f, device);
  ops::CausalDepthwiseConv1D op;
  EXPECT_THROW(op(dx, dw, &db, state, state), std::invalid_argument);
  EXPECT_THROW(op(dx, dw, &db, state, dx), std::invalid_argument);
  StorageView retained({16}, 3.f, device);
  StorageView alias({1, 2, 2}, retained.data<float>() + 2, device);
  retained.clear();
  EXPECT_THROW(op(alias, dw, &db, state, retained), std::invalid_argument);
  StorageView wrong_weight({2, 2}, 1.f, device), wrong_bias({3}, 1.f, device);
  StorageView wrong_type(x.shape(), DataType::INT32, device);
  EXPECT_THROW(op(dx, wrong_weight, &db, state, actual), std::invalid_argument);
  EXPECT_THROW(op(dx, dw, &wrong_bias, state, actual), std::invalid_argument);
  EXPECT_THROW(op(wrong_type, dw, &db, state, actual), std::invalid_argument);
  expect_storage_eq(state, initial);
  expect_storage_eq(actual, StorageView({1}, 42.f));
  if (device == Device::CUDA) {
    EXPECT_THROW(op(x, dw, &db, state, actual), std::invalid_argument);
  }
}

TEST_P(OpDeviceTest, GatedDeltaNetMatchesCpuAcrossBatchesHeadsAndKeySpecializations) {
  const Device device = GetParam();
  for (dim_t kd : {3, 64, 128}) {
    for (dim_t vd : {2, 257, 1025}) {
      for (dim_t steps : {0, 1, 5}) {
        SCOPED_TRACE("kd=" + std::to_string(kd) + ", vd=" + std::to_string(vd)
                     + ", steps=" + std::to_string(steps));
        StorageView q({2, steps, 2, kd}), k(q.shape()), v({2, steps, 4, vd});
        StorageView g({2, steps, 4}, -.2f), beta({2, steps, 4}, .3f), initial({2, 4, kd, vd});
        for (auto* tensor : {&q, &k, &v, &initial})
          fill_pattern(*tensor, .0078125f);
        StorageView expected, expected_state(initial);
        ops::GatedDeltaNet op;
        op(q, k, v, g, beta, expected_state, expected);
        auto dq = q.to(device), dk = k.to(device), dv = v.to(device);
        auto dg = g.to(device), db = beta.to(device), state = initial.to(device);
        StorageView actual(DataType::FLOAT32, device);
        op(dq, dk, dv, dg, db, state, actual);
        expect_storage_eq(actual, expected, 2e-5f);
        expect_storage_eq(state, expected_state, 2e-5f);
      }
    }
  }
}

TEST_P(OpDeviceTest, GatedDeltaNetSingleStepsMatchPrefill) {
  const Device device = GetParam();
  const dim_t steps = 5, kd = 64, vd = 3;
  StorageView q({2, steps, 2, kd}), k(q.shape()), v({2, steps, 4, vd});
  StorageView g({2, steps, 4}, -.3f), beta({2, steps, 4}, .25f), initial({2, 4, kd, vd});
  for (auto* tensor : {&q, &k, &v, &initial})
    fill_pattern(*tensor, .015625f);
  StorageView expected, expected_state(initial);
  ops::GatedDeltaNet op;
  op(q, k, v, g, beta, expected_state, expected);
  auto dq = q.to(device), dk = k.to(device), dv = v.to(device);
  auto dg = g.to(device), db = beta.to(device), state = initial.to(device);
  for (dim_t n = 0; n < 2; ++n) {
    StorageView s({1, 4, kd, vd}, state.data<float>() + n * 4 * kd * vd, device);
    for (dim_t t = 0; t < steps; ++t) {
      const dim_t row = n * steps + t;
      StorageView qt({1, 1, 2, kd}, dq.data<float>() + row * 2 * kd, device);
      StorageView kt({1, 1, 2, kd}, dk.data<float>() + row * 2 * kd, device);
      StorageView vt({1, 1, 4, vd}, dv.data<float>() + row * 4 * vd, device);
      StorageView gt({1, 1, 4}, dg.data<float>() + row * 4, device);
      StorageView bt({1, 1, 4}, db.data<float>() + row * 4, device);
      StorageView actual(DataType::FLOAT32, device);
      op(qt, kt, vt, gt, bt, s, actual);
      StorageView reference({1, 1, 4, vd}, expected.data<float>() + row * 4 * vd);
      expect_storage_eq(actual, reference, 2e-5f);
    }
  }
  expect_storage_eq(state, expected_state, 2e-5f);
}

TEST_P(OpDeviceTest, GatedDeltaNetValidationPreservesState) {
  const Device device = GetParam();
  StorageView q({1, 1, 1, 2}, 1.f, device), k(q), v(q);
  StorageView g({1, 1, 1}, 0.f, device), beta({1, 1, 1}, .5f, device);
  StorageView state({1, 1, 2, 2}, 2.f, device), actual({1}, 42.f, device);
  ops::GatedDeltaNet op;
  StorageView bad_key({1, 1, 1, 3}, 1.f, device), bad_gate({1}, 0.f, device);
  EXPECT_THROW(op(q, bad_key, v, g, beta, state, actual), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, bad_gate, beta, state, actual), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, beta, state, q), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, beta, state, state), std::invalid_argument);
  StorageView alias({1, 1, 1, 2}, state.data<float>() + 1, device);
  EXPECT_THROW(op(alias, k, v, g, beta, state, actual), std::invalid_argument);
  StorageView wrong_type(q.shape(), DataType::INT32, device);
  EXPECT_THROW(op(wrong_type, k, v, g, beta, state, actual), std::invalid_argument);
  if (device == Device::CUDA) {
    auto host = q.to(Device::CPU);
    EXPECT_THROW(op(host, k, v, g, beta, state, actual), std::invalid_argument);
  }
  expect_storage_eq(state, StorageView({1, 1, 2, 2}, 2.f));
  expect_storage_eq(actual, StorageView({1}, 42.f));
}

TEST_P(OpDeviceTest, ConvTranspose1DMatchesScalarAcrossParameters) {
  const Device device = GetParam();
  for (dim_t groups : {1, 2})
    for (dim_t dilation : {1, 2})
      for (dim_t output_padding : {0, 1})
        for (bool use_bias : {false, true})
          compare_transpose_with_scalar(device, 2, 4, 3, 7, 3, 2, 1,
                                        output_padding, dilation, groups, use_bias);
}

TEST_P(OpDeviceTest, ConvTranspose1DBoundaryCases) {
  const Device device = GetParam();
  StorageView x({1, 2, 1}, std::vector<float>{2, 3}, device);
  StorageView w({2, 2, 1}, std::vector<float>{1, 10, 100, 1000}, device);
  StorageView bias({2}, std::vector<float>{-1, 2}, device);
  StorageView actual(DataType::FLOAT32, device);
  ops::ConvTranspose1D()(x, w, bias, actual);
  expect_storage_eq(actual, StorageView({1, 2, 1}, std::vector<float>{301, 3022}));
  StorageView single({1, 1, 1}, 2.f, device), filter({1, 1, 2}, 3.f, device);
  ops::ConvTranspose1D(1, 0, 1, 2)(single, filter, actual);
  expect_storage_eq(actual, StorageView({1, 1, 4}, std::vector<float>{6, 0, 6, 0}));
  // Fish's right-side crop belongs to its layer, not the public transpose op.
  StorageView fish_input({1, 1, 2}, 1.f, device), fish_filter({1, 1, 6}, 1.f, device);
  ops::ConvTranspose1D(3)(fish_input, fish_filter, actual);
  expect_storage_eq(actual, StorageView({1, 1, 9}, std::vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1}));
}

TEST_P(OpDeviceTest, ConvTranspose1DValidationPreservesOutput) {
  const Device device = GetParam();
  StorageView x({1, 2, 2}, 1.f, device), w({2, 1, 2}, 1.f, device);
  StorageView bias({1}, .5f, device), actual({1}, 42.f, device);
  ops::ConvTranspose1D op;
  StorageView bad_weight({1, 1, 2}, 1.f, device), bad_bias({2}, 0.f, device);
  StorageView wrong_type(x.shape(), DataType::INT32, device);
  EXPECT_THROW(op(x, bad_weight, actual), std::invalid_argument);
  EXPECT_THROW(op(x, w, bad_bias, actual), std::invalid_argument);
  EXPECT_THROW(op(wrong_type, w, actual), std::invalid_argument);
  EXPECT_THROW(op(x, w, x), std::invalid_argument);
  EXPECT_THROW(op(x, w, bias, bias), std::invalid_argument);
  StorageView retained({16}, 3.f, device);
  StorageView alias({1, 2, 2}, retained.data<float>() + 2, device);
  retained.clear();
  EXPECT_THROW(op(alias, w, retained), std::invalid_argument);
  if (device == Device::CUDA) {
    auto host = x.to(Device::CPU);
    EXPECT_THROW(op(host, w, actual), std::invalid_argument);
  }
  expect_storage_eq(actual, StorageView({1}, 42.f));
}

INSTANTIATE_TEST_SUITE_P(CPU, OpDeviceTest, ::testing::Values(Device::CPU));
INSTANTIATE_TEST_SUITE_P(CPU, OpDeviceFPTest,
                         ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32, 1e-5}),
                         fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, OpDeviceTest, ::testing::Values(Device::CUDA));
INSTANTIATE_TEST_SUITE_P(CUDA, OpDeviceFPTest,
                         ::testing::Values(FloatType{Device::CUDA, DataType::FLOAT32, 1e-5},
                                           FloatType{Device::CUDA, DataType::FLOAT16, 1e-2},
                                           FloatType{Device::CUDA, DataType::BFLOAT16, 4e-2}),
                         fp_test_name);
#endif
