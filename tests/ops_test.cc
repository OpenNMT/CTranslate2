#include <algorithm>
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

TEST_P(OpDeviceFPTest, Conv1DDilation) {
  const Device device = GetParam().device;
  if (device == Device::CPU)
      GTEST_SKIP() << "Dilated convolution is not implemented for CPU.";
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
