#include "test_utils.h"
#include "ctranslate2/primitives.h"
#include "dispatch.h"

class PrimitiveTest : public ::testing::TestWithParam<Device> {
};

TEST_P(PrimitiveTest, FillFloat16) {
  const Device device = GetParam();
  StorageView x({2, 3}, DataType::FLOAT16, device);
  auto fill_value = float16_t(42.23);
  StorageView expected({2, 3}, std::vector<float16_t>{fill_value, fill_value, fill_value,
                                                      fill_value, fill_value, fill_value}, device);
  DEVICE_DISPATCH(device, primitives<D>::fill(x.data<float16_t>(), fill_value, x.size()));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, FillFloat32) {
  const Device device = GetParam();
  StorageView x({2, 3}, DataType::FLOAT32, device);
  auto fill_value = 42.23f;
  StorageView expected({2, 3}, std::vector<float>{fill_value, fill_value, fill_value,
                                                  fill_value, fill_value, fill_value}, device);
  DEVICE_DISPATCH(device, primitives<D>::fill(x.data<float>(), fill_value, x.size()));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, ZeroFloat16) {
  const Device device = GetParam();
  StorageView x({2, 3}, DataType::FLOAT16, device);
  StorageView expected({2, 3}, std::vector<float16_t>{float16_t(0), float16_t(0), float16_t(0),
                                                      float16_t(0), float16_t(0), float16_t(0)}, device);
  DEVICE_DISPATCH(device, primitives<D>::zero(x.data<float16_t>(), x.size()));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, ZeroFloat32) {
  const Device device = GetParam();
  StorageView x({2, 3}, DataType::FLOAT32, device);
  StorageView expected({2, 3}, std::vector<float>{0, 0, 0,
                                                  0, 0, 0}, device);
  DEVICE_DISPATCH(device, primitives<D>::zero(x.data<float>(), x.size()));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, StridedFill) {
  const Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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
    DEVICE_DISPATCH(device, primitives<D>::indexed_fill(x.data<float>(), 1.f, ids.data<int32_t>(), 3, x.size()));
    expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, IndexedFill2D) {
  const Device device = GetParam();
  StorageView x({3, 3}, std::vector<float>{1, 2, 3,
                                           4, 5, 6,
                                           7, 8, 9}, device);
  StorageView ids({6}, std::vector<int32_t>{0, 2, 3, 5, 6, 8}, device);
  StorageView expected({3, 3}, std::vector<float>{-1, 2, -1,
                                                  -1, 5, -1,
                                                  -1, 8, -1}, device);
  DEVICE_DISPATCH(device, primitives<D>::indexed_fill(x.data<float>(), -1.0f, ids.data<int32_t>(), 6, x.size()));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, IndexedFill2DComplexFloats) {
  const Device device = GetParam();
  StorageView x({2, 3}, std::vector<float>{-1.89935, -1.89909, 8.05185,
                                           -1e+10,   -1e+10,  -1e+10}, device);
  StorageView ids({2}, std::vector<int32_t>{2, 5}, device);
  StorageView expected({2, 3}, std::vector<float>{-1.89935, -1.89909, -3.40282e+38,
                                                  -1e+10,   -1e+10,   -3.40282e+38}, device);
  DEVICE_DISPATCH(device, primitives<D>::indexed_fill(x.data<float>(), -3.40282e+38f, ids.data<int32_t>(), 2, x.size()));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, LogSumExp) {
  const Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
  StorageView x({8}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2}, device);
  float result = 0;
  DEVICE_DISPATCH(device, result = primitives<D>::logsumexp(x.data<float>(), x.size()));
  EXPECT_NEAR(result, 2.1908040046691895, 1e-6);
}

TEST_P(PrimitiveTest, PenalizePreviousTokens) {
  const Device device = GetParam();
  if(device == Device::CANN)
    GUARD_OPERATOR_NPU_TEST;
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

TEST_P(PrimitiveTest, AddDepthBroadcast2DInput) {
  const Device device = GetParam();
  StorageView x1({2, 8}, std::vector<float>{12, -20,  31,  0.3, -42.17, 17.42,  40.5, -0.001,
                                            112, 20, -31, 40.3,  -4.17, -7.42, -50.34, 2.031}, device);
  StorageView x2({2}, std::vector<float>{0.1, 0.2}, device);
  StorageView expected({2, 8}, std::vector<float>{12.1, -19.9,  31.1,  0.4, -42.07, 17.52,  40.6, 0.099,
                                                  112.2, 20.2, -30.8, 40.5,  -3.97, -7.22, -50.14, 2.231}, device);
  StorageView output({2, 8}, DataType::FLOAT32, device);
  DEVICE_DISPATCH(device, primitives<D>::add_depth_broadcast(x2.data<float>(), x1.data<float>(), output.data<float>(), x2.size(), x1.size()));
  expect_storage_eq(output, expected);
}

TEST_P(PrimitiveTest, AddDepthBroadcast3DInput) {
  const Device device = GetParam();
  StorageView x1({2, 3, 4}, std::vector<float>{12,   -20.54, 31.1,  0.3,
                                               42.17, 17.42, 40.5, -0.001,
                                               12.2, -20,   -31,    40.3,

                                               7.4, -50.34, 2,  0.12,
                                               1,    20,  -31, 40.3,
                                               2,   -20,   31,  0.3}, device);
  StorageView x2({2}, std::vector<float>{1.1, 2.1}, device);
  StorageView expected({2, 3, 4}, std::vector<float>{13.1, -19.44, 32.2,  1.4,
                                                     43.27, 18.52, 41.6,  1.099,
                                                     13.3, -18.9, -29.9, 41.4,

                                                     9.5, -48.24,   4.1,  2.22,
                                                     3.1,  22.1,  -28.9, 42.4,
                                                     4.1, -17.9,   33.1,  2.4}, device);
  DEVICE_DISPATCH(device, primitives<D>::add_depth_broadcast(x2.data<float>(), x1.data<float>(), x1.data<float>(), x2.size(), x1.size()));
  expect_storage_eq(x1, expected);
}

TEST_P(PrimitiveTest, PrepareLengthMask) {
  const Device device = GetParam();
  StorageView lengths({2}, std::vector<int32_t>{17, 42}, device);
  StorageView mask({2, 3, 4}, DataType::INT32, device);
  DEVICE_DISPATCH(device, primitives<D>::prepare_length_mask(
          lengths.data<int32_t>(), /*batch_size*/ lengths.size(), /*num_heads*/ 3, /*num_queries*/ 4,
          /*mask_future*/ false, /*multi_query*/ false, mask.data<int32_t>()));
  StorageView expected({2, 3, 4}, std::vector<int32_t>{17, 17, 17, 17,
                                                       17, 17, 17, 17,
                                                       17, 17, 17, 17,

                                                       42, 42, 42, 42,
                                                       42, 42, 42, 42,
                                                       42, 42, 42, 42}, device);
  expect_storage_eq(mask, expected);
}

TEST_P(PrimitiveTest, PrepareLengthMaskMultiQuery) {
  const Device device = GetParam();
  StorageView lengths({2}, std::vector<int32_t>{17, 42}, device);
  StorageView mask({2, 4, 3}, DataType::INT32, device);
  DEVICE_DISPATCH(device, primitives<D>::prepare_length_mask(
          lengths.data<int32_t>(), /*batch_size*/ lengths.size(), /*num_heads*/ 3, /*num_queries*/ 4,
          /*mask_future*/ false, /*multi_query*/ true, mask.data<int32_t>()));
  StorageView expected({2, 4, 3}, std::vector<int32_t>{17, 17, 17,
                                                       17, 17, 17,
                                                       17, 17, 17,
                                                       17, 17, 17,

                                                       42, 42, 42,
                                                       42, 42, 42,
                                                       42, 42, 42,
                                                       42, 42, 42}, device);
  expect_storage_eq(mask, expected);
}

TEST_P(PrimitiveTest, PrepareLengthMaskMultiQueryMaskFuture) {
  const Device device = GetParam();
  StorageView lengths({2}, std::vector<int32_t>{17, 42}, device);
  StorageView mask({2, 4, 3}, DataType::INT32, device);
  DEVICE_DISPATCH(device, primitives<D>::prepare_length_mask(
          lengths.data<int32_t>(), /*batch_size*/ lengths.size(), /*num_heads*/ 3, /*num_queries*/ 4,
          /*mask_future*/ true, /*multi_query*/ true, mask.data<int32_t>()));
  StorageView expected({2, 4, 3}, std::vector<int32_t>{1, 1, 1,
                                                       2, 2, 2,
                                                       3, 3, 3,
                                                       4, 4, 4,

                                                       1, 1, 1,
                                                       2, 2, 2,
                                                       3, 3, 3,
                                                       4, 4, 4}, device);
  expect_storage_eq(mask, expected);
}

TEST_P(PrimitiveTest, PrepareLengthMaskMultiQueryMaskFutureSmallLength) {
  const Device device = GetParam();
  StorageView lengths({2}, std::vector<int32_t>{3, 2}, device);
  StorageView mask({2, 4, 3}, DataType::INT32, device);
  DEVICE_DISPATCH(device, primitives<D>::prepare_length_mask(
          lengths.data<int32_t>(), /*batch_size*/ lengths.size(), /*num_heads*/ 3, /*num_queries*/ 4,
          /*mask_future*/ true, /*multi_query*/ true, mask.data<int32_t>()));
  StorageView expected({2, 4, 3}, std::vector<int32_t>{1, 1, 1,
                                                       2, 2, 2,
                                                       3, 3, 3,
                                                       3, 3, 3,

                                                       1, 1, 1,
                                                       2, 2, 2,
                                                       2, 2, 2,
                                                       2, 2, 2}, device);
  expect_storage_eq(mask, expected);
}

TEST_P(PrimitiveTest, PrepareLengthMaskMaskFuture) {
  const Device device = GetParam();
  StorageView lengths({2}, std::vector<int32_t>{17, 42}, device);
  StorageView mask({2, 3, 4}, DataType::INT32, device);
  DEVICE_DISPATCH(device, primitives<D>::prepare_length_mask(
          lengths.data<int32_t>(), /*batch_size*/ lengths.size(), /*num_heads*/ 3, /*num_queries*/ 4,
          /*mask_future*/ true, /*multi_query*/ false, mask.data<int32_t>()));
  StorageView expected({2, 3, 4}, std::vector<int32_t>{1, 2, 3, 4,
                                                       1, 2, 3, 4,
                                                       1, 2, 3, 4,

                                                       1, 2, 3, 4,
                                                       1, 2, 3, 4,
                                                       1, 2, 3, 4}, device);
  expect_storage_eq(mask, expected);
}

TEST_P(PrimitiveTest, PrepareLengthMaskMaskFutureSmallLength) {
  const Device device = GetParam();
  StorageView lengths({2}, std::vector<int32_t>{2, 3}, device);
  StorageView mask({2, 3, 4}, DataType::INT32, device);
  DEVICE_DISPATCH(device, primitives<D>::prepare_length_mask(
          lengths.data<int32_t>(), /*batch_size*/ lengths.size(), /*num_heads*/ 3, /*num_queries*/ 4,
          /*mask_future*/ true, /*multi_query*/ false, mask.data<int32_t>()));
  StorageView expected({2, 3, 4}, std::vector<int32_t>{1, 2, 2, 2,
                                                       1, 2, 2, 2,
                                                       1, 2, 2, 2,

                                                       1, 2, 3, 3,
                                                       1, 2, 3, 3,
                                                       1, 2, 3, 3}, device);
  expect_storage_eq(mask, expected);
}

INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PrimitiveTest, ::testing::Values(Device::CUDA));
#elif CT2_WITH_CANN
INSTANTIATE_TEST_SUITE_P(CANN, PrimitiveTest, ::testing::Values(Device::CANN));
#endif
