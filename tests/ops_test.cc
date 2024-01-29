#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/devices.h"

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

TEST(OpTest, MedianFilter) {
  StorageView x({2, 8}, std::vector<float>{
      0.2556743323802948, 0.8028775453567505, 0.3514494299888611, 0.3542254865169525,
      0.5881291031837463, 0.1458204835653305, 0.6845740675926208, 0.543143630027771,
      0.9039326310157776, 0.38000917434692383, 0.9094009399414062, 0.4063926637172699,
      0.7943458557128906, 0.289182186126709, 0.9932224750518799, 0.01137143187224865});
  StorageView expected({2, 8}, std::vector<float>{
      0.3514494299888611, 0.3542254865169525, 0.3542254865169525, 0.3542254865169525,
      0.3542254865169525, 0.543143630027771, 0.5881291031837463, 0.543143630027771,
      0.9039326310157776, 0.4063926637172699, 0.7943458557128906, 0.4063926637172699,
      0.7943458557128906, 0.4063926637172699, 0.7943458557128906, 0.289182186126709});
  StorageView y;
  ops::MedianFilter(5)(x, y);
  expect_storage_eq(y, expected);
}

class OpDeviceTest : public ::testing::TestWithParam<Device> {
};

class OpDeviceFPTest : public ::testing::TestWithParam<FloatType> {
};


TEST_P(OpDeviceTest, Add) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{3, 5, 7, 9}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddTensors2D) {
  Device device = GetParam();
  StorageView a({4, 2}, std::vector<float>{1.69, 2, 3, 4, 17.42, 2, 3, 4.333}, device);
  StorageView b({4, 2}, std::vector<float>{2, 3, 4, 5, 1.42, 2, 3, 4.232}, device);
  StorageView expected({4, 2}, std::vector<float>{3.69, 5, 7, 9, 18.84, 4, 6, 8.565}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddLargeTensors2D) {
  Device device = GetParam();
  StorageView a({4000, 200}, 42, device);
  StorageView b({4000, 200}, 17, device);
  StorageView expected({4000, 200}, 59, device);
  StorageView c(DataType::INT32, a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddScalarTo1DTensor) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1.17, 2.17, 3.17, 4.17}, device);
  StorageView b(static_cast<float>(3.42));
  StorageView expected({4}, std::vector<float>{4.59, 5.59, 6.59, 7.59}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddScalarTo2DTensor) {
  Device device = GetParam();
  StorageView a({4, 2}, float16_t(42), device);
  StorageView b(float16_t(17));
  StorageView expected({4, 2}, float16_t(59), device);
  StorageView c(DataType::FLOAT16, a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddScalarToLarge2DTensor) {
  Device device = GetParam();
  StorageView a({4000, 200}, 42.f, device);
  StorageView b(17.f);
  StorageView expected({4000, 200}, 59.f, device);
  StorageView c(DataType::FLOAT32, a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, BiasAdd) {
  Device device = GetParam();
  StorageView value({4, 3}, std::vector<float>{1,  2,  3,
                                                          4,  5,  6,
                                                          7,  8,  9,
                                                          10, 11, 12}, device);
  StorageView bias({3}, std::vector<float>{1, 2, 3}, device);
  StorageView expected({4, 3}, std::vector<float>{2,  4,  6,
                                                             5,  7,  9,
                                                             8,  10, 12,
                                                             11, 13, 15}, device);
  StorageView c(device);
  ops::BiasAdd()(value, bias, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, BiasAdd3D) {
  Device device = GetParam();
  StorageView value({4, 3, 2}, std::vector<float>{1,  2,
                                                             3,  4,
                                                             5,  6,

                                                             7,  8,
                                                             9,  10.4265,
                                                             11, 12,

                                                             2,  3,
                                                             4,  5,
                                                             6,  7.917,

                                                             8,  9,
                                                             10, 11,
                                                             12, 13}, device);
  StorageView bias({2}, std::vector<float>{1, 2}, device);
  StorageView expected({4, 3, 2}, std::vector<float>{2,  4,
                                                                4,  6,
                                                                6,  8,

                                                                8,  10,
                                                                10, 12.4265,
                                                                12, 14,

                                                                3,  5,
                                                                5,  7,
                                                                7,  9.917,

                                                                9,  11,
                                                                11, 13,
                                                                13, 15}, device);
  StorageView c(device);
  ops::BiasAdd()(value, bias, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMul) {
    Device device = GetParam();
    StorageView a({4, 2}, std::vector<float>{1, 2,
                                             3, 4,
                                             5, 6,
                                             7, 8}, device);
    StorageView b({2, 4}, std::vector<float>{1, 3, 5, 7,
                                             2, 4, 6, 8}, device);
    StorageView expected({4,4}, std::vector<float>{5, 11, 17, 23,
                                                   11, 25, 39, 53,
                                                   17, 39, 61, 83,
                                                   23, 53, 83, 113}, device);
    StorageView c(a.device());
    ops::MatMul()(a, b, c);
    expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMulBatchLargerThanOne) {
    // Invoke the case of batch_size > 1
    Device device = GetParam();
    StorageView a({2,2,3}, std::vector<float>{1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12}, device);
    StorageView b({2,3,2}, std::vector<float>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, device);
    StorageView expected({2, 2, 2}, std::vector<float>{94, 100, 229, 244, 508, 532, 697, 730}, device);
    StorageView c(DataType::FLOAT32, device);
    ops::MatMul()(a, b, c);
    expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMulBatchWithIntScaling) {
  // Invoke the case of batch_size > 1
  Device device = GetParam();
  StorageView a({2,2,3}, std::vector<float>{1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12}, device);
  StorageView b({2,3,2}, std::vector<float>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, device);
  StorageView expected({2, 2, 2}, std::vector<float>{188, 200, 458, 488, 1016, 1064, 1394, 1460}, device);
  StorageView c(DataType::FLOAT32, device);
  ops::MatMul(false, false, 2)(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMulBatchWithDecimalScaling) {
  // Invoke the case of batch_size > 1
  Device device = GetParam();
  StorageView a({2,2,3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
  StorageView b({2,3,2}, std::vector<float>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, device);
  StorageView expected({2, 2, 2}, std::vector<float>{11.75, 12.5, 28.625, 30.5, 63.5, 66.5, 87.125, 91.25}, device);
  StorageView c(DataType::FLOAT32, device);
  ops::MatMul(false, false, 0.125)(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMulTransposeA) {
    Device device = GetParam();
    StorageView a({2, 4}, std::vector<float>{1, 3, 5, 7, 2, 4, 6, 8}, device);
    StorageView b({2, 4}, std::vector<float>{1, 3, 5, 7, 2, 4, 6, 8}, device);
    StorageView expected({4,4}, std::vector<float>{5, 11, 17, 23, 11, 25, 39, 53, 17, 39, 61, 83, 23, 53, 83, 113}, device);
    StorageView c(a.device());
    ops::MatMul op(true);
    op(a, b, c);
    expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMulTransposeB) {
  Device device = GetParam();
  StorageView a({4, 2}, std::vector<float>{1, 2,
                                           3, 4,
                                           5, 6,
                                           7, 8}, device);
  StorageView b({4, 2}, std::vector<float>{1, 2,
                                           3, 4,
                                           5, 6,
                                           7, 8}, device);
  StorageView expected({4,4}, std::vector<float>{5, 11, 17, 23, 11, 25, 39, 53, 17, 39, 61, 83, 23, 53, 83, 113}, device);
  StorageView c(a.device());
  ops::MatMul op(false, true);
  op(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MatMulTransposeBWithDecimalScaling) {
  Device device = GetParam();
  StorageView a({4, 2}, std::vector<float>{1, 2,
                                           3, 4,
                                           5, 6,
                                           7, 8}, device);
  StorageView b({4, 2}, std::vector<float>{1, 2,
                                           3, 4,
                                           5, 6,
                                           7, 8}, device);
  StorageView expected({4,4}, std::vector<float>{0.625, 1.375,  2.125,  2.875,
                                                 1.375, 3.125,  4.875,  6.625,
                                                 2.125, 4.875,  7.625, 10.375,
                                                 2.875, 6.625, 10.375, 14.125}, device);
  StorageView c(a.device());
  ops::MatMul op(false, true, 0.125);
  op(a, b, c);
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{-1, -1, -1, -1}, device);
  StorageView c(a.device());
  ops::Sub()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, TileFirstDim) {
  Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  StorageView input({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView expected_output({4, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView output(device);
  ops::Tile(0, 2)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, TileLastDim) {
  Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView expected_output({2, 4}, std::vector<float>{1, 2, 1, 2, 3, 4, 3, 4}, device);
  StorageView output(device);
  ops::Tile(1, 2)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, TileMiddleDim) {
  Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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

TEST_P(OpDeviceTest, ConcatBasic) {
  Device device = GetParam();
  StorageView a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView b({2, 3}, std::vector<float>{7, 8, 9, 10, 11, 12}, device);
  StorageView c({4,3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
  StorageView x(device);
  ops::Concat(0)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST_P(OpDeviceTest, ConcatNegativeAxis) {
  Device device = GetParam();
  StorageView a({2, 2, 2}, std::vector<float>{1, 2, 2, 3, 4, 4, 5, 3}, device);
  StorageView b({2, 2, 2}, std::vector<float>{7, 4, 8, 4, 2, 10, 15, 11}, device);
  StorageView c({2, 2, 4}, std::vector<float>{1, 2, 7, 4, 2, 3, 8, 4, 4, 4, 2, 10, 5, 3, 15, 11}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
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

TEST_P(OpDeviceTest, SplitAxis0EqualLengthParts2) {
  Device device = GetParam();
  StorageView input({4, 2}, std::vector<float>{1.42, -2.42,
                                               3.42, 4.42,
                                               5.42, 6.42,
                                               7.42, -8.42}, device);
  StorageView output1(device);
  StorageView output2(device);
  ops::Split(0)(input, output1, output2);
  StorageView expected_output1({2, 2}, std::vector<float>{1.42, -2.42, 3.42, 4.42}, device);
  StorageView expected_output2({2, 2}, std::vector<float>{5.42, 6.42, 7.42, -8.42}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
}

TEST_P(OpDeviceTest, SplitAxis0EqualLengthParts3) {
  Device device = GetParam();
  StorageView input({6, 2}, std::vector<float>{1.42, 17.24,
                                               -2.42, 42.56,
                                               3.42, -101.6,
                                               4.42, -500.543,
                                               5.42, 6.42,
                                               7.42, -8.42}, device);
  StorageView output1(device);
  StorageView output2(device);
  StorageView output3(device);
  ops::Split(0)(input, output1, output2, output3);
  StorageView expected_output1({2, 2}, std::vector<float>{1.42, 17.24, -2.42, 42.56}, device);
  StorageView expected_output2({2, 2}, std::vector<float>{3.42, -101.6, 4.42, -500.543}, device);
  StorageView expected_output3({2, 2}, std::vector<float>{5.42, 6.42, 7.42, -8.42}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
  expect_storage_eq(output3, expected_output3);
}

TEST_P(OpDeviceTest, SplitAxis1EqualLengthParts2) {
  Device device = GetParam();
  StorageView input({4, 2}, std::vector<float>{1, 2,
                                               3, 4,
                                               5, 6,
                                               7, 8}, device);
  StorageView output1(device);
  StorageView output2(device);
  ops::Split(1)(input, output1, output2);
  StorageView expected_output1({4, 1}, std::vector<float>{1, 3, 5, 7}, device);
  StorageView expected_output2({4, 1}, std::vector<float>{2, 4, 6, 8}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
}

TEST_P(OpDeviceTest, SplitAxis1EqualLengthParts3) {
  Device device = GetParam();
  StorageView input({2, 6}, std::vector<float>{1.42, 17.24, -2.42, 42.56, 3.42, -101.6,
                                               4.42, -500.543, 5.42, 6.42, 7.42, -8.42}, device);
  StorageView output1(device);
  StorageView output2(device);
  StorageView output3(device);
  ops::Split(1)(input, output1, output2, output3);
  StorageView expected_output1({2, 2}, std::vector<float>{1.42, 17.24, 4.42, -500.543}, device);
  StorageView expected_output2({2, 2}, std::vector<float>{-2.42, 42.56, 5.42, 6.42}, device);
  StorageView expected_output3({2, 2}, std::vector<float>{3.42, -101.6, 7.42, -8.42}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
  expect_storage_eq(output3, expected_output3);
}

TEST_P(OpDeviceTest, Axis0NonEqualLengthParts2) {
  Device device = GetParam();
  StorageView input({4, 2}, std::vector<float>{1.42, -2.42,
                                               3.42, 4.42,
                                               5.42, 6.42,
                                               7.42, -8.42}, device);
  StorageView output1(device);
  StorageView output2(device);
  ops::Split(0, {3, 1})(input, output1, output2);
  StorageView expected_output1({3, 2}, std::vector<float>{1.42, -2.42, 3.42, 4.42, 5.42, 6.42,}, device);
  StorageView expected_output2({1, 2}, std::vector<float>{7.42, -8.42}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
}

TEST_P(OpDeviceTest, Axis0NonEqualLengthParts3) {
  Device device = GetParam();
  StorageView input({6, 2}, std::vector<float>{1.42, 17.24,
                                               -2.42, 42.56,
                                               3.42, -101.6,
                                               4.42, -500.543,
                                               5.42, 6.42,
                                               7.42, -8.42}, device);
  StorageView output1(device);
  StorageView output2(device);
  StorageView output3(device);
  ops::Split(0, {1, 2, 3})(input, output1, output2, output3);
  StorageView expected_output1({1, 2}, std::vector<float>{1.42, 17.24}, device);
  StorageView expected_output2({2, 2}, std::vector<float>{-2.42, 42.56, 3.42, -101.6}, device);
  StorageView expected_output3({3, 2}, std::vector<float>{4.42, -500.543, 5.42, 6.42, 7.42, -8.42}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
  expect_storage_eq(output3, expected_output3);
}

TEST_P(OpDeviceTest, SplitAxis1NonEqualLengthParts2) {
  Device device = GetParam();
  StorageView input({4, 3}, std::vector<float>{1, 2, 3,
                                               4, 5, 6,
                                               7, 8, -5,
                                               -6, -7, -8}, device);
  StorageView output1(device);
  StorageView output2(device);
  ops::Split(1, {2, 1})(input, output1, output2);
  StorageView expected_output1({4, 2}, std::vector<float>{1, 2, 4, 5, 7, 8, -6, -7}, device);
  StorageView expected_output2({4, 1}, std::vector<float>{3, 6, -5, -8}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
}

TEST_P(OpDeviceTest, SplitAxis1NonEqualLengthParts3) {
  Device device = GetParam();
  StorageView input({2, 6}, std::vector<float>{1.42, 17.24, -2.42, 42.56, 3.42, -101.6,
                                               4.42, -500.543, 5.42, 6.42, 7.42, -8.42}, device);
  StorageView output1(device);
  StorageView output2(device);
  StorageView output3(device);
  ops::Split(1, {1, 2, 3})(input, output1, output2, output3);
  StorageView expected_output1({2, 1}, std::vector<float>{1.42, 4.42}, device);
  StorageView expected_output2({2, 2}, std::vector<float>{17.24, -2.42, -500.543, 5.42}, device);
  StorageView expected_output3({2, 3}, std::vector<float>{42.56, 3.42, -101.6, 6.42, 7.42, -8.42}, device);
  expect_storage_eq(output1, expected_output1);
  expect_storage_eq(output2, expected_output2);
  expect_storage_eq(output3, expected_output3);
}

TEST_P(OpDeviceTest, Mean) {
  const Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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

TEST_P(OpDeviceFPTest, Gemm) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView a(
    {4, 4}, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, device);
  StorageView b(a);
  StorageView y({4, 4}, 2.f, device);
  StorageView expected(
    {4, 4}, std::vector<float>{3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3}, device);
  ops::Gemm op(1.0, 1.0, false, false);

  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  y = y.to(dtype);
  op(a.to(dtype), b.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);

};

TEST_P(OpDeviceTest, GemmFloat16) {
  Device device = GetParam();
  if (!mayiuse_float16(device))
    return;
  StorageView a({8, 8}, float16_t(1.6), device);
  StorageView b({8, 8}, float16_t(1.4), device);
  StorageView c({8, 8}, float16_t(0.75), device);
  StorageView expected({8, 8}, float16_t(20.92), device);
  ops::Gemm op(1.0, 4, false, false);
  op(a, b, c);
  expect_storage_eq(c, expected);
};

TEST_P(OpDeviceTest, GemmFloat32) {
  Device device = GetParam();
  StorageView a(
          {2, 2}, std::vector<float>{1, 1, 1, 1}, device);
  StorageView b(a);
  StorageView expected(
          {2, 2}, std::vector<float>{3, 3, 3, 3}, device);
  StorageView c({2, 2}, std::vector<float>{1, 1, 1, 1}, device);
  ops::Gemm op(1.0, 1.0, false, false);
  op(a, b, c);
  expect_storage_eq(c, expected);
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

TEST_P(OpDeviceFPTest, GemmTransposeB) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView a({2, 3}, std::vector<float>{1, 2, 3,
                                           4, 5, 6}, device);
  StorageView b({4, 3}, std::vector<float>{1, 2, 3,
                                           4, 1, 2,
                                           3, 4, 1,
                                           2, 3, 4}, device);
  // check multiple constructors for c.
  StorageView c({2, 4}, DataType::FLOAT32, device);
  StorageView expected({2, 4}, std::vector<float>{14, 12, 14, 20,
                                                  32, 33, 38, 47}, device);
  ops::Gemm op(1.0, 0, false, true);

  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  c = c.to(dtype);
  op(a.to(dtype), b.to(dtype), c);
  expect_storage_eq(c.to_float32(), expected, error);
};

TEST_P(OpDeviceFPTest, TopKBasic) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  const int k = 3;
  StorageView input({1, 12}, std::vector<float>{1, 2, 98, 1, 1, 99, 3, 1, 3, 96, 4, 1}, device);
  StorageView expected_values({1, 3}, std::vector<float>{99, 98, 96}, device);
  StorageView expected_indices({1, 3}, std::vector<int32_t>{5, 2, 9}, device);
  StorageView values(dtype, device);
  StorageView indices(expected_indices.dtype(), device);
  ops::TopK op(k);
  op(input.to(dtype), values, indices);
  expect_storage_eq(values.to_float32(), expected_values, error);
  expect_storage_eq(indices, expected_indices);
}

TEST_P(OpDeviceFPTest, TopK) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  float error = GetParam().error;
  if(device == Device::CANN) {
    if(dtype == DataType::BFLOAT16)
      GUARD_BFLOAT16_NPU_TEST;
    else if(dtype == DataType::FLOAT32) {
      error = 3.907e-4; // FLOAT32 case does not comply with predefined error value
    }
  }
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
  {
    const float cann_error = 3.907e-4; // FLOAT32 case presents error
    device == Device::CANN ? expect_storage_eq(values, expected_values, cann_error) : expect_storage_eq(values, expected_values);
  }
  expect_storage_eq(indices, expected_indices);
  StorageView input2({2, 4}, std::vector<float>{0.1, 2.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3}, device);
  StorageView expected_values2({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices2({2, 3}, std::vector<int32_t>{1, 3, 2, 1, 0, 3}, device);
  op(input2, values, indices);
  {
    const float cann_error = 3.907e-4; // FLOAT32 case presents error
    device == Device::CANN ? expect_storage_eq(values, expected_values2, cann_error) : expect_storage_eq(values, expected_values2);
  }
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
  {
    const float cann_error = 3.907e-4; // FLOAT32 case presents error
    device == Device::CANN ? expect_storage_eq(values_k2, expected_values_k2, cann_error) : expect_storage_eq(values_k2, expected_values_k2);
  }
  expect_storage_eq(indices_k2, expected_indices_k2);

  const StorageView expected_values_k3({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  const StorageView expected_indices_k3({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values_k3(expected_values_k3.dtype(), device);
  StorageView indices_k3(expected_indices_k3.dtype(), device);
  ops::TopK(3)(input, values_k3, indices_k3);
  {
    const float cann_error = 3.907e-4; // FLOAT32 case presents error
    device == Device::CANN ? expect_storage_eq(values_k3, expected_values_k3, cann_error) : expect_storage_eq(values_k3, expected_values_k3);
  }
  expect_storage_eq(indices_k3, expected_indices_k3);
}

TEST_P(OpDeviceFPTest, TopPMask) {
  const Device device = GetParam().device;
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  x = x.to(dtype);
  StorageView expected({2, 5}, std::vector<float>{
      0.032035, 0.785904, 0.129909, 0.013025, 0.039128,
      0.760941, 0.207381, 0.009342, 0.001544, 0.020792}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x, y);
  expect_storage_eq(y.to_float32(), expected, error);
  ops::SoftMax()(x);
  expect_storage_eq(x.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, SoftMax1D) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x = StorageView({5}, std::vector<float>{
          -0.2, 3.0, 1.2, -1.1, 0.0}, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  x = x.to(dtype);
  StorageView expected({5}, std::vector<float>{
          0.032035, 0.785904, 0.129909, 0.013025, 0.039128}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x, y);
  expect_storage_eq(y.to_float32(), expected, error);
  ops::SoftMax()(x);
  expect_storage_eq(x.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, SoftMax1DWithLength) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x = StorageView({5}, std::vector<float>{
          -0.2, 3.0, 1.2, -1.1, 42.17}, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  x = x.to(dtype);
  StorageView lengths({1}, std::vector<int32_t>{4}, device);
  StorageView expected({5}, std::vector<float>{
          0.0333396, 0.8179057, 0.1351989, 0.013554, 0}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x, lengths, y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LogSoftMax) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x = StorageView({2, 10}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0, 0.2, -3.0, -1.2, 1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0, -4.6, -3.3, -0.2, 1.6, -1.0}, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  x = x.to(dtype);
  StorageView expected({2, 10}, std::vector<float>{
      -3.638294, -0.438294, -2.238294, -4.538294, -3.438294, -3.238294, -6.438294, -4.638294, -2.338294, -3.438294,
      -0.319434, -1.619434, -4.719434, -6.519434, -3.919434, -9.519434, -8.219434, -5.119434, -3.319434, -5.919434}, device);
  StorageView y(dtype, device);
  ops::LogSoftMax()(x, y);
  expect_storage_eq(y.to_float32(), expected, error * 10);
  ops::LogSoftMax()(x);
  expect_storage_eq(x.to_float32(), expected, error * 10);
}

TEST_P(OpDeviceFPTest, MaskedLogSoftMax) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x = StorageView({3, 10}, std::vector<float>{
          -0.2, 3.0, 1.2, -1.1,  0.0,  0.2, -3.0, -1.2,  1.1,  0.0,
           4.6, 3.3, 0.2, -1.6,  1.0, -4.6, -3.3, -0.2,  1.6, -1.0,
          -1.1, 0.0, 0.2, -3.0, -1.2,  4.6,  3.3,  0.2, -1.6,  1.0}, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  x = x.to(dtype);
  StorageView lengths({3}, std::vector<int32_t>{3, 5, 7}, device);
  StorageView expected({3, 10}, std::vector<float>{
          -3.38735985, -0.18735980, -1.98735976,  0,           0,           0,           0,          0,0,0,
          -0.27319955, -1.57319951, -4.67319965, -6.47319936, -3.87319946,  0,           0,          0,0,0,
          -5.96369791, -4.86369800, -4.66369819, -7.86369800, -6.06369781, -0.26369810, -1.56369805, 0,0,0}, device);
  StorageView y(dtype, device);
  ops::LogSoftMax()(x, lengths, y);
  expect_storage_eq(y.to_float32(), expected, error * 10);
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
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  ops::SoftMax()(x.to(dtype), lengths, y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, MaskedSoftMaxWithLengthsEqualToLastDim) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView x({2, 10}, std::vector<float>{
          -0.2, 3.0, 1.2, -1.1, 0.0, 4.6,  3.3, 0.2,  3.0,   1.21,
           4.6, 3.3, 0.2, -1.6, 1.0, 1.2, -1.1, 0.0, 0.17, 0.42}, device);
  StorageView lengths({2}, std::vector<int32_t>{10, 10}, device);
  StorageView expected({2, 10}, std::vector<float>{
          0.0046304, 0.1135965, 0.0187773, 0.0018825, 0.0056556, 0.5626471, 0.15333, 0.0069078, 0.11359, 0.01896,
          0.72038,   0.19632,   0.00884,   0.00146,   0.01968,   0.02404,   0.00241, 0.00724,   0.00858, 0.01102}, device);
  StorageView y(dtype, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  ops::SoftMax()(x.to(dtype), lengths, y);

  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, MaskedSoftMax4D) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
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
  StorageView mask({2, 2, 3}, std::vector<int32_t>{
          1, 2, 3,
          1, 2, 3,
          1, 2, 2,
          1, 2, 2
  }, device);
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

TEST_P(OpDeviceFPTest, MaskedSoftMaxTriangular) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
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
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
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

TEST_P(OpDeviceFPTest, LayerNormZerosAndOnes) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  const float error = GetParam().error;
  StorageView gamma({2}, 0.f, device);
  StorageView beta({2}, 1.f, device);
  StorageView x({5, 2}, std::vector<float>{
          0, 10,
          20, 30,
          40, 50,
          60, 70,
          80, 90}, device);
  StorageView expected({5, 2}, std::vector<float>{
          1, 1,
          1, 1,
          1, 1,
          1, 1,
          1, 1}, device);
  StorageView y(dtype, device);
  ops::LayerNorm()(beta.to(dtype), gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LayerNorm3DEasy) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  const float error = GetParam().error;
  StorageView gamma({2}, std::vector<float>{0.2, 2.1}, device);
  StorageView beta({2}, std::vector<float>{-6.6, -5.7}, device);
  StorageView x({2, 5, 2}, std::vector<float>{
          0, 10,
          20, 30,
          40, 50,
          60, 70,
          80, 90,

          0, 10,
          20, 30,
          40, 50,
          60, 70,
          80, 90}, device);
  StorageView expected({2, 5, 2}, std::vector<float>{
          -6.79999, -3.6,
          -6.79999, -3.6,
          -6.79999, -3.6,
          -6.79999, -3.6,
          -6.79999, -3.6,

          -6.79999, -3.6,
          -6.79999, -3.6,
          -6.79999, -3.6,
          -6.79999, -3.6,
          -6.79999, -3.6}, device);
  StorageView y(dtype, device);
  ops::LayerNorm()(beta.to(dtype), gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}


TEST_P(OpDeviceFPTest, LayerNorm3DHard) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  const float error = GetParam().error;
  StorageView gamma({4}, std::vector<float>{0.2, 2.1, -1.3, -4.2}, device);
  StorageView beta({4}, std::vector<float>{2.2, 4.43, -1.6, -1.7}, device);
  StorageView x({2, 3, 4}, std::vector<float>{
           4.5, 0.6, 0.5,  0.6,
           0.5, 4.6, 5.5,  6.67,
          -5.5, 0.2, 0.5, -7.46,

           -0.5,   0.6, 0.4,  0.6,
            0.5,  17.6, 0.1, -0.62,
          -42.4,  78.6, 0.5,  0.6}, device);
  StorageView expected({2, 3, 4}, std::vector<float>{
           2.546310, 3.259002, -0.798791,  0.641994,
           1.871333, 4.685378, -2.261745, -5.953297,
           2.060307, 6.396746, -2.929379,  3.594856,

           1.859225, 5.930507, -1.957263, -4.701014,
           2.097962, 8.062276, -0.868645,  1.058934,
           1.963113, 7.761240, -1.337295, -0.860878}, device);
  StorageView y(dtype, device);
  ops::LayerNorm()(beta.to(dtype), gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, LayerNormAxis) {
  const Device device = GetParam().device;
  if (device == Device::CUDA || device == Device::CANN) {
    GTEST_SKIP() << "Generalized LayerNorm is not implemented on GPU and NPU";
  }
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView gamma({5}, std::vector<float>{0.2, 2.1, 1.1, -0.6, 0.7}, device);
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView expected({2, 5}, std::vector<float>{
      -0.0262, 4.1202, 0.8633, 0.4316, 0.0000,
      0.3445, 2.5953, 0.0824, 0.3595, 0.2622}, device);
  StorageView y(dtype, device);
  ops::RMSNorm()(gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float32(), expected, error * 10);
}

TEST_P(OpDeviceTest, QuantizeINT8) {
  Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  ops::ReLU()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, ReLULarge) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2, 5, 6}, std::vector<float>{-1.12, 1.55, 2.3, -2.42, 2.17, 4.5, -3.27, 0.12, -1.55, -3.17,
                                                  -1, 1, 2, -2, 2, 4, -3, 0, -1, -32.17,
                                                  -1, 1, 2, -2, 2, 4, -3, 0, -1, -3,
                                                  -5.12, 9.55, 2.3, -2.42, 2.17, 4.5, 3.27, 1.12, -8.55, -33.17,
                                                  -1, 1, 2, -2, 2, 4, -3, 0, -1, -3,
                                                  -1, 1, 2, -2, 2, 4, -3, 0.42, -1, -3.42}, device);
  StorageView expected({2, 5, 6}, std::vector<float>{0, 1.55, 2.3, 0, 2.17, 4.5, 0, 0.12, 0, 0,
                                                  0, 1, 2, 0, 2, 4, 0, 0, 0, 0,
                                                  0, 1, 2, 0, 2, 4, 0, 0, 0, 0,
                                                  0, 9.55, 2.3, 0, 2.17, 4.5, 3.27, 1.12, 0, 0,
                                                  0, 1, 2, 0, 2, 4, 0, 0, 0, 0,
                                                  0, 1, 2, 0, 2, 4, 0, 0.42, 0, 0}, device);

  StorageView output(dtype, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  ops::ReLU()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, GELU) {
  const Device device = GetParam().device;
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11685754358768463, -0.128212109208107}, device);
  StorageView output(dtype, device);
  const ops::GELU gelu_op(ops::GELU::Approximation::Sigmoid);
  gelu_op(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Swish) {
  const Device device = GetParam().device;
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.10996679, -0.27841452}, device);
  StorageView output(dtype, device);
  ops::Swish()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Cos) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  std::vector<float> input_vec({0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4});
  std::vector<float> expected_vec;
  expected_vec.reserve(input_vec.size());
  std::transform(input_vec.begin(), input_vec.end(), std::back_inserter(expected_vec),
                 [](const float& i){return std::cos(i);});
  StorageView input({2, 4}, input_vec, device);
  StorageView expected({2, 4}, expected_vec, device);
  StorageView output(dtype, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  ops::Cos()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Sin) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  std::vector<float > input_vec({0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4});
  std::vector<float > expected_vec;
  expected_vec.reserve(input_vec.size());
  std::transform(input_vec.begin(), input_vec.end(), std::back_inserter(expected_vec),
                 [](const float& i){return std::sin(i);});
  StorageView input({2, 4}, input_vec, device);
  StorageView expected({2, 4}, expected_vec, device);
  StorageView output(dtype, device);
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  ops::Sin()(input.to(dtype), output);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Tanh) {
  const Device device = GetParam().device;
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  auto ops = ops::Min();
  TestMinMax<float>(device, ops, [](float left, float right){
    return left > right? right : left;
  });
}

TEST_P(OpDeviceTest, Max) {
  Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  auto ops = ops::Max();
  TestMinMax<float>(device, ops, [](float left, float right){
    return left > right? left : right;
  });
}

#ifndef CT2_WITH_CUDNN
#  define GUARD_CONV1D_GPU_TEST GTEST_SKIP() << "Conv1D tests on GPU require cuDNN"
#else
#  define GUARD_CONV1D_GPU_TEST do {} while (0)
#endif

static const StorageView conv_input({2, 2, 3}, std::vector<float>{
    0.5728129f, 0.8784890f, 0.2029965f, 0.3689166f, 0.6570600f, 0.9202735f,
    0.7081605f, 0.3570334f, 0.9339380f, 0.8162224f, 0.0597404f, 0.4628246f});

static const StorageView conv_weight({4, 2, 2}, std::vector<float>{
    0.4969918f, 0.3711241f, 0.1489926f, -0.3010672f,
    -0.2055028f, 0.2540314f, 0.3566069f, -0.1201057f,
    -0.0737700f, -0.0630847f, -0.2370351f, -0.0451550f,
    0.0186623f, 0.3600836f, -0.2889268f, -0.4857445f});

static const StorageView conv_bias({4}, std::vector<float>{
    0.4631361f, -0.1047785f, 0.1047658f, -0.3157263f});

TEST_P(OpDeviceFPTest, Conv1D) {
  const Device device = GetParam().device;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  else if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.9309945f, 0.7959076f, 0.0533122f, -0.1099610f,
      -0.1100256f, -0.1701476f, -0.4144599f, -0.8630960f,
      1.0512151f, 0.8567453f, 0.1242856f, 0.0248157f,
      -0.1661695f, -0.0155492f, -0.4387956f, -0.2148425f});
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
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  else if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.4678584f, 0.3327716f, 0.1580907f, -0.005182412f,
      -0.2147914f, -0.2749133f, -0.09873369f, -0.5473697f,
      0.5880789f, 0.3936091f, 0.2290641f, 0.1295942f,
      -0.2709353f, -0.120315f, -0.1230693f, 0.1008837f});
  StorageView output(dtype, device);
  ops::Conv1D()(conv_input.to(device).to(dtype),
                conv_weight.to(device).to(dtype),
                output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, Conv1DPadding) {
  const Device device = GetParam().device;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  else if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 4}, std::vector<float>{
      0.5646521f, 0.9309945f, 0.7959076f, 0.7011377f,
      -0.0035750f, 0.0533122f, -0.1099610f, 0.1816810f,
      0.0519716f, -0.1100256f, -0.1701476f, -0.1283464f,
      -0.2886650f, -0.4144599f, -0.8630960f, -0.5778296f,
      0.4802138f, 1.0512151f, 0.8567453f, 0.9962531f,
      -0.0229165f, 0.1242856f, 0.0248157f, -0.1316590f,
      0.0232352f, -0.1661695f, -0.0155492f, -0.0738365f,
      -0.4572049f, -0.4387956f, -0.2148425f, -0.4320193f});
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
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  else if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 1}, std::vector<float>{
      0.9309945f, 0.0533122f, -0.1100256f, -0.4144599f,
      1.0512151f, 0.1242856f, -0.1661695f, -0.4387956f});
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
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  else if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.5646521f, 0.7959076f, -0.0035750f, -0.1099610f,
      0.0519716f, -0.1701476f, -0.2886650f, -0.8630960f,
      0.4802138f, 0.8567453f, -0.0229165f, 0.0248157f,
      0.0232352f, -0.0155492f, -0.4572049f, -0.2148425f});
  StorageView output(dtype, device);
  ops::Conv1D(2, 1)(conv_input.to(device).to(dtype),
                    conv_weight.to(device).to(dtype),
                    conv_bias.to(device).to(dtype),
                    output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float32(), expected, error);
}

TEST_P(OpDeviceFPTest, SplitAxis0EqualLengthParts) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  if(device == Device::CANN && dtype == DataType::BFLOAT16)
    GUARD_BFLOAT16_NPU_TEST;
  const float error = GetParam().error;
  StorageView input({4, 2}, std::vector<float>{1.42, -2.42,
                                               3.42, 4.42,
                                               5.42, 6.42,
                                               7.42, -8.42}, device);
  StorageView output1(dtype, device);
  StorageView output2(dtype, device);
  ops::Split(0)(input.to(dtype), output1, output2);
  StorageView expected_output1({2, 2}, std::vector<float>{1.42, -2.42, 3.42, 4.42}, device);
  StorageView expected_output2({2, 2}, std::vector<float>{5.42, 6.42, 7.42, -8.42}, device);
  EXPECT_EQ(output1.dtype(), dtype);
  EXPECT_EQ(output2.dtype(), dtype);
  expect_storage_eq(output1.to_float32(), expected_output1, error);
  expect_storage_eq(output2.to_float32(), expected_output2, error);
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
                                           FloatType{Device::CUDA, DataType::BFLOAT16, 1e-2}),
                         fp_test_name);
#elif CT2_WITH_CANN
INSTANTIATE_TEST_SUITE_P(CANN, OpDeviceTest, ::testing::Values(Device::CANN));
INSTANTIATE_TEST_SUITE_P(CANN, OpDeviceFPTest,
                         ::testing::Values(FloatType{Device::CANN, DataType::FLOAT32, 1e-5},
                                           FloatType{Device::CANN, DataType::FLOAT16, 1e-2},
                                           FloatType{Device::CANN, DataType::BFLOAT16, 1e-2}),
                         fp_test_name);
#endif
