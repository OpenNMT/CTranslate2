#include "test_utils.h"
#include "ctranslate2/ops/conv1d.h"

namespace {

  // Independent cross-correlation oracle; accumulate in double precision.
  StorageView scalar_convolution(const StorageView& input, const StorageView& weight,
                                 const StorageView* bias, dim_t stride, dim_t padding,
                                 dim_t dilation, dim_t groups) {
    const dim_t batches = input.dim(0);
    const dim_t channels = input.dim(1);
    const dim_t length = input.dim(2);
    const dim_t outputs = weight.dim(0);
    const dim_t kernel = weight.dim(2);
    const dim_t per_group = channels / groups;
    const dim_t result_length = (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    StorageView result({batches, outputs, result_length});
    for (dim_t b = 0; b < batches; ++b) {
      for (dim_t o = 0; o < outputs; ++o) {
        const dim_t group = o / (outputs / groups);
        for (dim_t t = 0; t < result_length; ++t) {
          double sum = bias ? bias->data<float>()[o] : 0.;
          for (dim_t c = 0; c < per_group; ++c) {
            for (dim_t k = 0; k < kernel; ++k) {
              const dim_t position = t * stride - padding + k * dilation;
              if (position >= 0 && position < length) {
                sum += double(input.data<float>()[(b * channels + group * per_group + c) * length
                                                  + position])
                       * weight.data<float>()[(o * per_group + c) * kernel + k];
              }
            }
          }
          result.data<float>()[(b * outputs + o) * result_length + t] = sum;
        }
      }
    }
    return result;
  }

}

TEST(Conv1DDilationTest, ScalarReferenceWithGroupsPaddingAndStride) {
  // Include depthwise and ordinary grouped cases, both bias paths, and
  // padding larger than the input for the largest dilation.
  for (const dim_t length : {5, 25}) {
    for (const dim_t dilation : {1, 3, 9}) {
      for (const dim_t groups : {1, 2, 4}) {
        for (const dim_t stride : {1, 2}) {
          for (const bool padded : {false, true}) {
            const dim_t padding = padded ? dilation : 0;
            if (length + 2 * padding < dilation * 2 + 1)
              continue;
            for (const bool with_bias : {false, true}) {
              SCOPED_TRACE(::testing::Message() << "length=" << length << " dilation=" << dilation
                                                << " groups=" << groups << " stride=" << stride
                                                << " padding=" << padding << " bias=" << with_bias);
              StorageView input({2, 4, length});
              StorageView weight({4, 4 / groups, 3});
              StorageView bias(Shape{4});
              for (dim_t i = 0; i < input.size(); ++i)
                input.data<float>()[i] = float(i % 17 - 8) / 7;
              for (dim_t i = 0; i < weight.size(); ++i)
                weight.data<float>()[i] = float(i % 11 - 5) / 9;
              for (dim_t i = 0; i < bias.size(); ++i)
                bias.data<float>()[i] = float(i - 2) / 3;
              const auto expected = scalar_convolution(input, weight, with_bias ? &bias : nullptr,
                                                       stride, padding, dilation, groups);
              StorageView output;
              const ops::Conv1D op(stride, padding, dilation, groups);
              if (with_bias)
                op(input, weight, bias, output);
              else
                op(input, weight, output);
              expect_storage_eq(output, expected, 3e-6f);
            }
          }
        }
      }
    }
  }
}

TEST(Conv1DDilationTest, QuantizedWeightsRetainDilatedGroupedBehavior) {
#ifdef CT2_WITH_DNNL
  GTEST_SKIP() << "Quantized convolution is not implemented for oneDNN.";
#else
  for (const dim_t dilation : {1, 3, 9}) {
    for (const dim_t groups : {1, 2, 4}) {
      SCOPED_TRACE(::testing::Message() << "dilation=" << dilation << " groups=" << groups);
      StorageView input({2, 4, 25});
      StorageView weights({4, 4 / groups, 3});
      StorageView quantized_weights(weights.shape(), DataType::INT8);
      StorageView scales({4}, 127.f);
      for (dim_t i = 0; i < input.size(); ++i)
        input.data<float>()[i] = float(i % 3 - 1);
      for (dim_t i = 0; i < weights.size(); ++i) {
        const auto value = i % 3 - 1;
        weights.data<float>()[i] = float(value);
        quantized_weights.data<int8_t>()[i] = int8_t(127 * value);
      }
      // Inputs and weights are exactly representable after int8 quantization.
      const auto expected =
          scalar_convolution(input, weights, nullptr, 2, dilation, dilation, groups);
      StorageView output;
      ops::Conv1D(2, dilation, dilation, groups)(input, quantized_weights, output, &scales);
      expect_storage_eq(output, expected, 1e-5f);
    }
  }
#endif
}
