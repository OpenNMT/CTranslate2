#include "ctranslate2/ops/moe_topk.h"
#include "ctranslate2/ops/ops.h"
#include "mps/kernels.h"
#include "mps/moe.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

TEST(MPSBackendTest, MoeRoutingPrefillShapesUnevenLoadsAndRowOrder) {
  for (dim_t rows : {1, 32, 128, 512, 2048})
    for (dim_t experts : {4, 8, 128}) {
      StorageView input({rows, experts});
      for (dim_t r = 0; r < rows; ++r)
        for (dim_t e = 0; e < experts; ++e)
          input.data<float>()[r * experts + e] =
              r % 10 && e < 2 ? 100.f - float(e)
                              : std::sin(float(r * 7 + e * 13) * .17f);
      for (dim_t k : {dim_t(1), dim_t(2), std::min<dim_t>(8, experts)}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << ' ' << experts << ' ' << k);
        StorageView expected_ids(DataType::INT32), expected_weights;
        ops::MoeTopK route(k);
        route(input, expected_ids, expected_weights);
        auto gpu = input.to(Device::MPS);
        StorageView ids(DataType::INT32, Device::MPS),
            weights(DataType::FLOAT32, Device::MPS);
        route(gpu, ids, weights);
        expect_storage_eq(ids, expected_ids);
        expect_storage_eq(weights, expected_weights, 1e-7f);
        // Reverse token order on GPU and route again with reused allocations.
        // Expert selection and weights must follow tokens exactly.
        std::vector<int32_t> order(rows);
        std::iota(order.rbegin(), order.rend(), 0);
        StorageView permutation({rows}, order, Device::MPS),
            reversed(DataType::FLOAT32, Device::MPS);
        ops::Gather()(gpu, permutation, reversed);
        route(reversed, ids, weights);
        auto host_ids = ids.to(Device::CPU),
             host_weights = weights.to(Device::CPU);
        std::vector<dim_t> counts(experts, 0);
        for (dim_t r = 0; r < rows; ++r) {
          double sum = 0;
          for (dim_t j = 0; j < k; ++j) {
            const auto expert = host_ids.data<int32_t>()[r * k + j];
            ASSERT_GE(expert, 0);
            ASSERT_LT(expert, experts);
            ++counts[expert];
            EXPECT_EQ(expert,
                      expected_ids.data<int32_t>()[(rows - r - 1) * k + j]);
            EXPECT_NEAR(host_weights.data<float>()[r * k + j],
                        expected_weights.data<float>()[(rows - r - 1) * k + j],
                        1e-7f);
            sum += host_weights.data<float>()[r * k + j];
            for (dim_t other = 0; other < j; ++other)
              EXPECT_NE(expert, host_ids.data<int32_t>()[r * k + other]);
          }
          EXPECT_NEAR(sum, 1., 1e-6);
        }
        EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), dim_t(0)),
                  rows * k);
      }
    }
}

namespace {
  float moe_reference_round(float value, DataType dtype) {
    if (dtype == DataType::FLOAT16)
      return float(float16_t(value));
    if (dtype == DataType::BFLOAT16)
      return float(bfloat16_t(value));
    return value;
  }
} // namespace

TEST(MPSBackendTest, MoeRouterProjectionMatchesIndependentReference) {
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
    for (auto shape : {Shape{7, 17, 13}, Shape{33, 4, 1024}}) {
      const dim_t rows = shape[0], experts = shape[1], hidden = shape[2];
      SCOPED_TRACE(::testing::Message() << int(dtype) << ' ' << hidden);
      std::vector<float> inputs(rows * hidden), weights(experts * hidden);
      for (size_t i = 0; i < inputs.size(); ++i)
        inputs[i] = moe_reference_round(.3f * std::sin(float(i) * .17f), dtype);
      for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = .4f * std::cos(float(i) * .23f);
      StorageView expected({rows, experts});
      for (dim_t r = 0; r < rows; ++r)
        for (dim_t e = 0; e < experts; ++e) {
          double sum = 0;
          for (dim_t k = 0; k < hidden; ++k)
            sum += double(inputs[r * hidden + k]) * weights[e * hidden + k];
          expected.data<float>()[r * experts + e] = sum;
        }
      auto x = StorageView({rows, hidden}, inputs).to(dtype).to(Device::MPS);
      StorageView router({experts, hidden}, weights, Device::MPS),
          actual({rows, experts}, DataType::FLOAT32, Device::MPS);
      mps::moe_router(dtype, x.buffer(), router.data<float>(),
                      actual.data<float>(), rows, experts, hidden);
      expect_storage_eq(actual, expected, 2e-6f);
    }
}

TEST(MPSBackendTest, MoeRouterNearCutoffUsesActualInputPrecision) {
  // Rounding the hidden state, rather than a faulty TopK, can change the
  // selected expert. Keep router scores in FP32 even for FP16 activations.
  const std::vector<float> coefficients = {
    .1f, 0.f, -.950008f, 1.f, .05f, 0.f, -.1f, 0.f};
  StorageView router({4, 2}, coefficients, Device::MPS);
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16})
    for (dim_t rows : {7, 32, 33, 64, 65}) {
      SCOPED_TRACE(::testing::Message() << int(dtype) << ' ' << rows);
      std::vector<float> values(rows * 2);
      for (dim_t row = 0; row < rows; ++row) {
        values[row * 2] = 1.f;
        values[row * 2 + 1] = 1.0001f;
      }
      auto host = StorageView({rows, 2}, values).to(dtype).to_float32();
      StorageView expected({rows, 4});
      for (dim_t row = 0; row < rows; ++row)
        for (dim_t expert = 0; expert < 4; ++expert)
          expected.data<float>()[row * 4 + expert] =
            double(host.data<float>()[row * 2]) * coefficients[expert * 2]
            + double(host.data<float>()[row * 2 + 1]) * coefficients[expert * 2 + 1];
      auto input = host.to(dtype).to(Device::MPS);
      StorageView scores({rows, 4}, DataType::FLOAT32, Device::MPS),
        ids(DataType::INT32, Device::MPS), weights(DataType::FLOAT32, Device::MPS),
        expected_ids(DataType::INT32), expected_weights;
      mps::moe_router(dtype, input.buffer(), router.data<float>(),
                       scores.data<float>(), rows, 4, 2);
      ops::MoeTopK{2}(scores, ids, weights);
      ops::MoeTopK{2}(expected, expected_ids, expected_weights);
      expect_storage_eq(scores, expected, 1e-7f);
      expect_storage_eq(ids, expected_ids);
      expect_storage_eq(weights, expected_weights, 1e-7f);
      EXPECT_EQ(expected_ids.data<int32_t>()[1], dtype == DataType::FLOAT32 ? 1 : 2);
      if (dtype == DataType::FLOAT16) {
        StorageView rounded_ids(DataType::INT32), rounded_weights;
        ops::MoeTopK{2}(expected.to_float16().to_float32(), rounded_ids, rounded_weights);
        // Casting scores to half erases the margin and wrongly prefers ID 1.
        EXPECT_EQ(rounded_ids.data<int32_t>()[1], 1);
      }
    }
}

TEST(MPSBackendTest, MoeCompleteExpertExecutionMatchesIndependentReference) {
  constexpr dim_t experts = 4;
  for (const auto& shape : {Shape{13, 17}, Shape{64, 72}}) {
    const dim_t hidden = shape[0], intermediate = shape[1];
    for (auto dtype :
         {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
      for (dim_t rows : {1, 2, 7, 8, 9, 33, 65})
        for (dim_t topk : {1, 2, 4}) {
          // The aligned shape exercises the FP16 vector reduction. Retain the
          // existing FP32/BF16 reference geometries and absolute error bounds.
          if (hidden != 13 && (dtype != DataType::FLOAT16 || rows > 8))
            continue;
          SCOPED_TRACE(::testing::Message() << int(dtype) << ' ' << rows << ' '
                                            << topk << ' ' << hidden);
          auto make_signal = [&](Shape shape, float scale, float phase) {
            dim_t n = std::accumulate(shape.begin(), shape.end(), dim_t(1),
                                      std::multiplies<dim_t>());
            std::vector<float> data(n);
            for (dim_t i = 0; i < n; ++i)
              data[i] = scale * std::sin(float(i) * .17f + phase);
            return StorageView(shape, data).to(dtype);
          };
          auto x = make_signal({rows, hidden}, .5f, .2f);
          std::vector<std::array<StorageView, 3>> host, gpu;
          host.reserve(experts);
          gpu.reserve(experts);
          for (dim_t e = 0; e < experts; ++e) {
            host.push_back({make_signal({intermediate, hidden}, .2f, e * .3f),
                            make_signal({intermediate, hidden}, .3f, e * .7f),
                            make_signal({hidden, intermediate}, .4f, e * .5f)});
            if (topk < experts && e >= 2)
              for (auto& matrix : host.back())
                matrix =
                    StorageView(matrix.shape(),
                                std::vector<float>(
                                    matrix.size(),
                                    std::numeric_limits<float>::quiet_NaN()))
                        .to(dtype);
            gpu.push_back({host.back()[0].to(Device::MPS),
                           host.back()[1].to(Device::MPS),
                           host.back()[2].to(Device::MPS)});
          }
          std::vector<std::array<const StorageView*, 3>> weights;
          for (const auto& matrices : gpu)
            weights.push_back({&matrices[0], &matrices[1], &matrices[2]});
          StorageView logits({rows, experts});
          for (dim_t r = 0; r < rows; ++r)
            for (dim_t e = 0; e < experts; ++e)
              logits.data<float>()[r * experts + e] =
                  e < 2 ? float((r + e) % 3) : -100.f - float(e);
          StorageView ids(DataType::INT32), probabilities;
          ops::MoeTopK{topk}(logits, ids, probabilities);
          const auto xv = x.to_float32().to_vector<float>();
          std::vector<float> reference(rows * hidden, 0);
          for (dim_t e = 0; e < experts; ++e) {
            const auto gate = host[e][0].to_float32().to_vector<float>();
            const auto up = host[e][1].to_float32().to_vector<float>();
            const auto down = host[e][2].to_float32().to_vector<float>();
            for (dim_t r = 0; r < rows; ++r)
              for (dim_t slot = 0; slot < topk; ++slot) {
                if (ids.data<int32_t>()[r * topk + slot] != e)
                  continue;
                std::vector<float> inner(intermediate);
                for (dim_t f = 0; f < intermediate; ++f) {
                  double g = 0, u = 0;
                  for (dim_t d = 0; d < hidden; ++d) {
                    g += double(xv[r * hidden + d]) * gate[f * hidden + d];
                    u += double(xv[r * hidden + d]) * up[f * hidden + d];
                  }
                  const float rounded_gate =
                      moe_reference_round(float(g), dtype);
                  const float activated = moe_reference_round(
                      rounded_gate / (1.f + std::exp(-rounded_gate)), dtype);
                  inner[f] = moe_reference_round(
                      activated * moe_reference_round(float(u), dtype), dtype);
                }
                const float probability = moe_reference_round(
                    probabilities.data<float>()[r * topk + slot], dtype);
                for (dim_t d = 0; d < hidden; ++d) {
                  double sum = 0;
                  for (dim_t f = 0; f < intermediate; ++f)
                    sum += double(inner[f]) * down[d * intermediate + f];
                  const float contribution = moe_reference_round(
                      moe_reference_round(float(sum), dtype) * probability,
                      dtype);
                  reference[r * hidden + d] = moe_reference_round(
                      reference[r * hidden + d] + contribution, dtype);
                }
              }
          }
          auto xgpu = x.to(Device::MPS), igpu = ids.to(Device::MPS),
               pgpu = probabilities.to(Device::MPS);
          for (bool force_gemv : {false, true}) {
            StorageView actual(dtype, Device::MPS);
            mps::moe_forward(xgpu, igpu, pgpu, weights, actual, force_gemv);
            expect_storage_eq(actual.to_float32(),
                              StorageView({rows, hidden}, reference),
                              dtype == DataType::FLOAT32   ? 2e-6f
                              : dtype == DataType::FLOAT16 ? .002f
                                                           : .02f);
            if (rows <= 8) {
              StorageView grouped(dtype, Device::MPS);
              mps::moe_forward(xgpu, igpu, pgpu, weights, grouped, force_gemv,
                               false, nullptr, true);
              // Changing dispatch and workspace order must preserve every
              // dtype rounding boundary, including the expert-order reduction.
              expect_storage_eq(actual, grouped);
            }
          }
        }
  }
}

TEST(MPSBackendTest, MoeDispatchPlanPreservesEveryAssignmentAndEmptyExperts) {
  constexpr dim_t experts = 4, topk = 2;
  for (dim_t rows : {1, 7, 33, 64, 65, 128, 129}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> selected(rows * topk);
    for (dim_t r = 0; r < rows; ++r) {
      selected[r * topk] = r % 3;
      selected[r * topk + 1] = (r + 1) % 3;
    }
    StorageView ids({rows, topk}, selected, Device::MPS),
        offsets({experts + 1}, DataType::INT32, Device::MPS),
        permutation({rows * topk}, DataType::INT32, Device::MPS),
        inverse(permutation.shape(), DataType::INT32, Device::MPS),
        indirect({experts, 6}, DataType::INT32, Device::MPS);
    for (bool decode : {false, true}) {
      mps::moe_dispatch_plan(ids.data<int32_t>(), offsets.data<int32_t>(),
                             permutation.data<int32_t>(),
                             inverse.data<int32_t>(), indirect.data<int32_t>(),
                             rows, experts, topk, 13, 17, decode);
      const auto off = offsets.to_vector<int32_t>(),
                 perm = permutation.to_vector<int32_t>(),
                 inv = inverse.to_vector<int32_t>(),
                 grids = indirect.to_vector<int32_t>();
      EXPECT_EQ(off.front(), 0);
      EXPECT_EQ(off.back(), rows * topk);
      EXPECT_EQ(off[3], off[4]);
      std::vector<int32_t> seen(rows * topk, 0);
      for (dim_t e = 0; e < experts; ++e) {
        const dim_t count = off[e + 1] - off[e];
        for (dim_t stage = 0; stage < 2; ++stage) {
          const dim_t columns = stage ? 13 : 17, tile = decode ? 8 : 32;
          EXPECT_EQ(grids[e * 6 + stage * 3],
                    count ? (columns + tile - 1) / tile : 1);
          EXPECT_EQ(grids[e * 6 + stage * 3 + 1],
                    count ? (decode ? count : (count + 15) / 16) : 1);
          EXPECT_EQ(grids[e * 6 + stage * 3 + 2], 1);
        }
        for (dim_t i = off[e]; i < off[e + 1]; ++i) {
          ASSERT_GE(perm[i], 0);
          ASSERT_LT(perm[i], rows * topk);
          ++seen[perm[i]];
          EXPECT_EQ(selected[perm[i]], e);
          EXPECT_EQ(inv[perm[i]], i);
          if (i > off[e])
            EXPECT_LT(perm[i - 1], perm[i]);
        }
      }
      for (auto count : seen)
        EXPECT_EQ(count, 1);
    }
  }
}

TEST(MPSBackendTest, UnnormalizedMoeRoutingMatchesIndependentFullSoftmax) {
  const float inf = std::numeric_limits<float>::infinity();
  StorageView host({4, 5},
                   std::vector<float>{0,    1,    2,    3,    4,    0,   0,
                                      0,    0,    0,    inf,  inf,  inf, 0,
                                      -inf, -inf, -inf, -inf, -inf, -inf});
  for (auto device : {Device::CPU, Device::MPS})
    for (dim_t k : {1, 2, 5}) {
      auto input = host.to(device);
      StorageView ids(DataType::INT32, device),
          weights(DataType::FLOAT32, device);
      ops::MoeTopK(k, false)(input, ids, weights);
      auto result = weights.to(Device::CPU);
      auto indices = ids.to(Device::CPU);
      double denom = 0;
      for (int e = 0; e < 5; ++e)
        denom += std::exp(double(e) - 4);
      for (dim_t j = 0; j < k; ++j) {
        EXPECT_NEAR(result.data<float>()[j], std::exp(-double(j)) / denom,
                    1e-7);
        EXPECT_NEAR(result.data<float>()[k + j], .2, 1e-7);
        EXPECT_NEAR(result.data<float>()[2 * k + j], j < 3 ? 1. / 3 : 0, 1e-7);
        EXPECT_EQ(result.data<float>()[3 * k + j], 0);
        EXPECT_EQ(indices.data<int32_t>()[j], 4 - j);
      }
    }
}

TEST(MPSBackendTest, PreciseVocabularyProjectionMatchesDoubleAndPreservesTies) {
  for (dim_t rows : {1, 2, 3, 33})
    for (dim_t depth : {13, 64})
      for (bool bias : {false, true}) {
        StorageView x({rows, depth}), w({65, depth}), b({65}),
            expected({rows, 65});
        for (dim_t i = 0; i < x.size(); ++i)
          x.data<float>()[i] =
              float(float16_t(.2f * std::sin(float(i) * .13f)));
        for (dim_t i = 0; i < w.size(); ++i)
          w.data<float>()[i] =
              float(float16_t(.3f * std::cos(float(i) * .17f)));
        for (dim_t i = 0; i < b.size(); ++i)
          b.data<float>()[i] = float(float16_t(float(i) * .01f));
        for (dim_t r = 0; r < rows; ++r)
          for (dim_t c = 0; c < 65; ++c) {
            double sum = bias ? b.data<float>()[c] : 0;
            for (dim_t k = 0; k < depth; ++k)
              sum += double(x.data<float>()[r * depth + k]) *
                     w.data<float>()[c * depth + k];
            expected.data<float>()[r * 65 + c] = sum;
          }
        auto gx = x.to(DataType::FLOAT16).to(Device::MPS),
             gw = w.to(DataType::FLOAT16).to(Device::MPS),
             gb = b.to(DataType::FLOAT16).to(Device::MPS);
        StorageView out(expected.shape(), DataType::FLOAT32, Device::MPS);
        mps::gemm_f16_f32_output(gx.data<float16_t>(), gw.data<float16_t>(),
                                 bias ? gb.data<float16_t>() : nullptr,
                                 out.data<float>(), rows, 65, depth);
        expect_storage_eq(out, expected, 1e-6f);
      }
  StorageView x({1, 2}, std::vector<float16_t>{float16_t(1), float16_t(.0001f)},
                Device::MPS),
      w({2, 2},
        std::vector<float16_t>{float16_t(1), float16_t(0), float16_t(1),
                               float16_t(1)},
        Device::MPS),
      out({1, 2}, DataType::FLOAT32, Device::MPS);
  mps::gemm_f16_f32_output(x.data<float16_t>(), w.data<float16_t>(), nullptr,
                           out.data<float>(), 1, 2, 2);
  auto host = out.to(Device::CPU);
  EXPECT_GT(host.data<float>()[1], host.data<float>()[0]);
  EXPECT_EQ(float16_t(host.data<float>()[1]), float16_t(host.data<float>()[0]));
}

namespace {
  struct Int4Fixture {
    StorageView packed, scales, unpacked;
    Int4Fixture(dim_t n, dim_t k)
        : packed({n, (k + 1) / 2}, int8_t(0x88)), scales({n, (k + 63) / 64}),
          unpacked({n, k}, DataType::FLOAT16) {
      for (dim_t i = 0; i < scales.size(); ++i)
        scales.data<float>()[i] = .003f * float(i % 7 + 1);
      auto* bytes = reinterpret_cast<uint8_t*>(packed.buffer());
      for (dim_t r = 0; r < n; ++r)
        for (dim_t c = 0; c < k; ++c) {
          const int q = int((r * 7 + c * 11) % 15) - 7;
          const auto at = r * packed.dim(1) + c / 2;
          bytes[at] = uint8_t((bytes[at] & (c % 2 ? 15 : 240)) |
                              ((q + 8) << ((c % 2) * 4)));
          unpacked.data<float16_t>()[r * k + c] = float16_t(
              float(q) * scales.data<float>()[r * scales.dim(1) + c / 64]);
        }
    }
  };
} // namespace

TEST(MPSBackendTest, Int4GemvGemmMatchDoubleAcrossGroupsBiasesAndTails) {
  for (dim_t rows : {1, 2, 7, 9, 33, 65})
    for (dim_t depth : {13, 64, 65, 129, 2304, 9216})
      for (bool bias : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << ' ' << depth << ' ' << bias);
        constexpr dim_t n = 65;
        Int4Fixture f(n, depth);
        StorageView x({rows, depth}, DataType::FLOAT16),
            b({n}, DataType::FLOAT16), expected({rows, n});
        for (dim_t i = 0; i < x.size(); ++i)
          x.data<float16_t>()[i] = float16_t(.2f * std::sin(float(i) * .17f));
        for (dim_t i = 0; i < n; ++i)
          b.data<float16_t>()[i] = float16_t(float(i) * .003f);
        for (dim_t r = 0; r < rows; ++r)
          for (dim_t c = 0; c < n; ++c) {
            double sum = bias ? float(b.data<float16_t>()[c]) : 0;
            for (dim_t k = 0; k < depth; ++k)
              sum += double(float(x.data<float16_t>()[r * depth + k])) *
                     float(f.unpacked.data<float16_t>()[c * depth + k]);
            expected.data<float>()[r * n + c] = float(float16_t(float(sum)));
          }
        auto gx = x.to(Device::MPS), gw = f.packed.to(Device::MPS),
             gs = f.scales.to(Device::MPS), gb = b.to(Device::MPS);
        StorageView out({rows, n}, DataType::FLOAT16, Device::MPS);
        mps::gemm_int4(gx.data<float16_t>(), gw.data<int8_t>(),
                       gs.data<float>(), bias ? gb.data<float16_t>() : nullptr,
                       out.data<float16_t>(), rows, n, depth);
        expect_storage_eq(out.to_float32(), expected, .00025f);
      }
}

TEST(MPSBackendTest, Int4FloatActivationsMatchDoubleAcrossGroupsBiasesAndTails) {
  for (dim_t rows : {1, 2, 7, 9, 33, 65})
    for (dim_t depth : {13, 64, 65, 129, 2304, 9216})
      for (bool bias : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << ' ' << depth << ' ' << bias);
        constexpr dim_t n = 65;
        Int4Fixture f(n, depth);
        StorageView x({rows, depth}, DataType::FLOAT32),
            b({n}, DataType::FLOAT16), expected({rows, n});
        for (dim_t i = 0; i < x.size(); ++i)
          x.data<float>()[i] = .21739f * std::sin(float(i) * .17f);
        for (dim_t i = 0; i < n; ++i)
          b.data<float16_t>()[i] = float16_t(float(i) * .003f);
        for (dim_t r = 0; r < rows; ++r)
          for (dim_t c = 0; c < n; ++c) {
            double sum = bias ? float(b.data<float16_t>()[c]) : 0;
            for (dim_t k = 0; k < depth; ++k)
              sum += double(x.data<float>()[r * depth + k]) *
                     float(f.unpacked.data<float16_t>()[c * depth + k]);
            expected.data<float>()[r * n + c] = float(sum);
          }
        auto gx = x.to(Device::MPS), gw = f.packed.to(Device::MPS),
             gs = f.scales.to(Device::MPS), gb = b.to(Device::MPS);
        StorageView out({rows, n}, DataType::FLOAT32, Device::MPS);
        mps::gemm_int4_f32(gx.data<float>(), gw.data<int8_t>(),
                       gs.data<float>(), bias ? gb.data<float16_t>() : nullptr,
                       out.data<float>(), rows, n, depth);
        expect_storage_eq(out.to_float32(), expected, 1e-5f);
      }
}

TEST(MPSBackendTest, Int4GroupedExpertsMatchUnpackedAcrossDispatchModes) {
  constexpr dim_t count = 4;
  for (const auto& shape : {Shape{67, 73}, Shape{64, 72}}) {
    const dim_t hidden = shape[0], intermediate = shape[1];
    std::vector<Int4Fixture> f;
    for (dim_t e = 0; e < count; ++e) {
      f.emplace_back(intermediate, hidden);
      f.emplace_back(intermediate, hidden);
      f.emplace_back(hidden, intermediate);
    }
    std::vector<std::array<StorageView, 3>> packed, scales, unpacked;
    for (dim_t e = 0; e < count; ++e) {
      packed.push_back({f[e * 3].packed.to(Device::MPS),
                        f[e * 3 + 1].packed.to(Device::MPS),
                        f[e * 3 + 2].packed.to(Device::MPS)});
      scales.push_back({f[e * 3].scales.to(Device::MPS),
                        f[e * 3 + 1].scales.to(Device::MPS),
                        f[e * 3 + 2].scales.to(Device::MPS)});
      unpacked.push_back({f[e * 3].unpacked.to(Device::MPS),
                          f[e * 3 + 1].unpacked.to(Device::MPS),
                          f[e * 3 + 2].unpacked.to(Device::MPS)});
    }
    std::vector<std::array<const StorageView*, 3>> pw, sw, uw;
    for (dim_t e = 0; e < count; ++e) {
      pw.push_back({&packed[e][0], &packed[e][1], &packed[e][2]});
      sw.push_back({&scales[e][0], &scales[e][1], &scales[e][2]});
      uw.push_back({&unpacked[e][0], &unpacked[e][1], &unpacked[e][2]});
    }
    for (dim_t rows : {1, 2, 7, 8, 9, 33, 65, 129})
      for (dim_t k : {1, 2, 4}) {
        StorageView x({rows, hidden}, DataType::FLOAT16), router({rows, count});
        for (dim_t i = 0; i < x.size(); ++i)
          x.data<float16_t>()[i] = float16_t(.2f * std::sin(float(i) * .07f));
        for (dim_t r = 0; r < rows; ++r)
          for (dim_t e = 0; e < count; ++e)
            router.data<float>()[r * count + e] =
                e < 2 ? float((r + e) % 3) : -100.f;
        StorageView ids(DataType::INT32), weights;
        ops::MoeTopK{k}(router, ids, weights);
        auto gx = x.to(Device::MPS), gi = ids.to(Device::MPS),
             gw = weights.to(Device::MPS);
        StorageView ref(DataType::FLOAT16, Device::MPS),
            actual(DataType::FLOAT16, Device::MPS);
        for (bool gemv : {false, true})
          for (bool small : {false, true}) {
            mps::moe_forward(gx, gi, gw, uw, ref, gemv, small);
            mps::moe_forward(gx, gi, gw, pw, actual, gemv, small, &sw);
            expect_storage_eq(actual, ref);
            if (rows <= 8) {
              mps::moe_forward(gx, gi, gw, pw, ref, gemv, small, &sw, true);
              expect_storage_eq(actual, ref);
            }
          }
      }
  }
}
