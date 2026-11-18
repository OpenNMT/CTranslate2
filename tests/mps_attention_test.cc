#include "test_utils.h"
#include "mps/kernels.h"
#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/layers/attention_layer.h"

#include <algorithm>
#include <cmath>
#include <limits>

TEST(MPSBackendTest, DecoderPrefillGemmBetaZeroIgnoresNonfiniteOutput) {
  {
    StorageView a = StorageView({2047, 1024}, .125f).to_float16().to(Device::MPS);
    StorageView b = StorageView({4096, 1024}, .0625f).to_float16().to(Device::MPS);
    StorageView c = StorageView({2047, 4096}, std::numeric_limits<float>::quiet_NaN())
      .to_float16().to(Device::MPS);
    ops::Gemm(1, 0, false, true)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, 8.f) << "exact Qwen prefill shape";
  }
  for (dim_t rows : {dim_t(1), dim_t(33), dim_t(2047)}) {
    StorageView a = StorageView({rows, 128}, .125f).to_float16().to(Device::MPS);
    StorageView b = StorageView({1024, 128}, .0625f).to_float16().to(Device::MPS);
    StorageView c = StorageView({rows, 1024}, std::numeric_limits<float>::quiet_NaN())
      .to_float16().to(Device::MPS);
    ops::Gemm(1, 0, false, true)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, 1.f) << "rows=" << rows;
  }
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16}) {
    StorageView a = StorageView({4096, 1}, .125f).to(dtype).to(Device::MPS);
    StorageView b = StorageView({1, 64}, .0625f).to(dtype).to(Device::MPS);
    StorageView c = StorageView({4096, 64}, std::numeric_limits<float>::quiet_NaN())
      .to(dtype).to(Device::MPS);
    ops::Gemm(1, 0, false, false)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, .0078125f) << "dtype=" << int(dtype);
  }
}

TEST(MPSBackendTest, DecoderPrefillGemmReusesGpuPoisonedStorage) {
  StorageView a = StorageView({2047, 1024}, .125f).to_float16().to(Device::MPS);
  StorageView nan_a = StorageView({2047, 1024}, std::numeric_limits<float>::quiet_NaN())
    .to_float16().to(Device::MPS);
  StorageView b = StorageView({4096, 1024}, .0625f).to_float16().to(Device::MPS);
  StorageView c(DataType::FLOAT16, Device::MPS);
  for (int i = 0; i < 4; ++i) {
    ops::Gemm(1, 0, false, true)(nan_a, b, c);
    ops::Gemm(1, 0, false, true)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, 8.f) << "iteration=" << i;
  }
}

TEST(MPSBackendTest, DecoderPrefillGemmDoesNotReadNanRowsBeyondMatrix) {
  constexpr dim_t rows = 2047, k = 1024, n = 4096;
  std::vector<float> av((rows + 1) * k, .125f);
  std::fill(av.begin() + rows * k, av.end(), std::numeric_limits<float>::quiet_NaN());
  StorageView backing = StorageView({rows + 1, k}, av).to_float16().to(Device::MPS);
  StorageView a(DataType::FLOAT16, Device::MPS);
  a.view(backing.data<float16_t>(), {rows, k});
  StorageView b = StorageView({n, k}, .0625f).to_float16().to(Device::MPS);
  StorageView c(DataType::FLOAT16, Device::MPS);
  ops::Gemm(1, 0, false, true)(a, b, c);
  const auto actual = c.to_float32().to_vector<float>();
  for (size_t i = 0; i < actual.size(); ++i)
    ASSERT_EQ(actual[i], 8.f) << "row=" << i / n << " col=" << i % n;
}

TEST(MPSBackendTest, NativePrefillGemmPreservesStridePaddingAndOddTails) {
  constexpr dim_t m = 1025, n = 1031, k = 1033;
  constexpr dim_t lda = 1040, ldb = 1040, ldc = 1040;
  const float poison = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> av(m * k), bv(n * k);
  for (size_t i = 0; i < av.size(); ++i)
    av[i] = .05f * std::sin(float(i) * .13f);
  for (size_t i = 0; i < bv.size(); ++i)
    bv[i] = .05f * std::cos(float(i) * .07f);
  const auto a_cpu = StorageView({m, k}, av).to_float16();
  const auto b_cpu = StorageView({n, k}, bv).to_float16();
  StorageView reference;
  ops::Gemm(1, 0, false, true)(a_cpu.to_float32(), b_cpu.to_float32(), reference);
  const auto expected = reference.to_float16().to_float32().to_vector<float>();
  // Offset zero exercises native aligned loads. Offset one exercises the
  // generic fallback: it cannot use the native kernel's 16-byte vector loads.
  for (dim_t offset : {dim_t(0), dim_t(1)}) {
    SCOPED_TRACE(offset);
    std::vector<float16_t> a(m * lda + offset, float16_t(poison));
    std::vector<float16_t> b(n * ldb + offset, float16_t(poison));
    for (dim_t row = 0; row < m; ++row)
      std::copy_n(a_cpu.data<float16_t>() + row * k, k, a.data() + offset + row * lda);
    for (dim_t row = 0; row < n; ++row)
      std::copy_n(b_cpu.data<float16_t>() + row * k, k, b.data() + offset + row * ldb);
    StorageView agpu({dim_t(a.size())}, a, Device::MPS);
    StorageView bgpu({dim_t(b.size())}, b, Device::MPS);
    StorageView cgpu({m * ldc + 2}, float16_t(poison), Device::MPS);
    mps::gemm(DataType::FLOAT16, false, true, m, n, k, 1,
              agpu.data<float16_t>() + offset, lda,
              bgpu.data<float16_t>() + offset, ldb,
              0, cgpu.data<float16_t>() + 1, ldc);
    const auto actual = cgpu.to_float32().to_vector<float>();
    ASSERT_TRUE(std::isnan(actual.front()));
    ASSERT_TRUE(std::isnan(actual.back()));
    for (dim_t row = 0; row < m; ++row) {
      for (dim_t col = 0; col < n; ++col)
        ASSERT_NEAR(actual[1 + row * ldc + col], expected[row * n + col], .002f)
          << "row=" << row << " col=" << col;
      for (dim_t col = n; col < ldc; ++col)
        ASSERT_TRUE(std::isnan(actual[1 + row * ldc + col]));
    }
  }
}

TEST(MPSBackendTest, DecoderRotaryCacheGrowthPreservesEarlierPositions) {
  layers::RotaryEmbeddings rotary(128, false);
  StorageView initial = StorageView({1, 16, 2047, 128}, 1.f).to_float16().to(Device::MPS);
  rotary.apply(initial, 0);
  const auto reference = initial.to_float32().to_vector<float>();
  StorageView grow = StorageView({1, 16, 1, 128}, 1.f).to_float16().to(Device::MPS);
  rotary.apply(grow, 2048);
  StorageView repeated = StorageView({1, 16, 2047, 128}, 1.f).to_float16().to(Device::MPS);
  rotary.apply(repeated, 0);
  const auto actual = repeated.to_float32().to_vector<float>();
  for (size_t i = 0; i < actual.size(); ++i)
    ASSERT_FLOAT_EQ(actual[i], reference[i]) << "index=" << i;
}

TEST(MPSBackendTest, LongCausalAttentionIgnoresPoisonedScoreStorage) {
  constexpr dim_t time = 2047, heads = 2, depth = 128;
  StorageView q = StorageView({1, heads, time, depth}, .125f).to_float16().to(Device::MPS);
  StorageView k = StorageView({1, heads, time, depth}, .0625f).to_float16().to(Device::MPS);
  StorageView v = StorageView({1, heads, time, depth}, 1.f).to_float16().to(Device::MPS);
  std::vector<int32_t> lens(heads * time);
  for (dim_t h = 0; h < heads; ++h)
    for (dim_t t = 0; t < time; ++t)
      lens[h * time + t] = t + 1;
  StorageView lengths({heads * time}, lens, Device::MPS);
  for (int iteration = 0; iteration < 3; ++iteration) {
    StorageView scores = StorageView({1, heads, time, time}, std::numeric_limits<float>::quiet_NaN())
      .to_float16().to(Device::MPS);
    ops::MatMul(false, true, 1.f / std::sqrt(float(depth)))(q, k, scores);
    const auto raw = scores.to_float32().to_vector<float>();
    for (float value : raw)
      ASSERT_TRUE(std::isfinite(value)) << "QK iteration=" << iteration;
    ops::SoftMax()(scores, lengths, scores);
    const auto probabilities = scores.to_float32().to_vector<float>();
    for (float value : probabilities)
      ASSERT_TRUE(std::isfinite(value)) << "softmax iteration=" << iteration;
    StorageView out(DataType::FLOAT16, Device::MPS);
    ops::MatMul()(scores, v, out);
    const auto actual = out.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_NEAR(value, 1.f, 0.002f) << "AV iteration=" << iteration;
  }
}
