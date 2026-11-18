#include "ctranslate2/ops/fused_attention.h"
#include "ctranslate2/ops/ops.h"
#include "mps/flash_attention.h"
#include "mps/kernels.h"
#include "layers/attention_utils.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
  StorageView quantized_kv_reference(const StorageView& input, unsigned bits) {
    const auto source = input.to_float16().to_float32().to_vector<float>();
    const dim_t depth = input.dim(-1), rows = input.size() / depth;
    std::vector<float> output(source.size());
    const int limit = (1 << (bits - 1)) - 1;
    for (dim_t row = 0; row < rows; ++row)
      for (dim_t start = 0; start < depth; start += 32) {
        const dim_t end = std::min(start + 32, depth);
        float maximum = 0;
        for (dim_t d = start; d < end; ++d)
          maximum = std::max(maximum, std::abs(source[row * depth + d]));
        const float scale = maximum == 0 ? 1.f : maximum * (1.f / limit);
        for (dim_t d = start; d < end; ++d)
          output[row * depth + d] =
              (maximum == 0 ? 0
                            : std::nearbyint(source[row * depth + d] * limit /
                                             maximum)) *
              scale;
      }
    return StorageView(input.shape(), output).to_float16();
  }

  StorageView reference_attention(const StorageView& q, const StorageView& k,
                                  const StorageView& v, dim_t valid,
                                  dim_t offset, dim_t window,
                                  const std::vector<int32_t>* lengths = nullptr,
                                  bool causal = true) {
    const dim_t b = q.dim(0), t = q.dim(1), h = q.dim(2), d = q.dim(3),
                kh = k.dim(2), cap = k.dim(1);
    const auto qv = q.to_float32().to_vector<float>();
    const auto kv = k.to_float32().to_vector<float>(),
               vv = v.to_float32().to_vector<float>();
    std::vector<float> result(q.size(), 0);
    for (dim_t batch = 0; batch < b; ++batch)
      for (dim_t head = 0; head < h; ++head)
        for (dim_t row = 0; row < t; ++row) {
          const dim_t first =
              window ? std::max<dim_t>(0, offset + row + 1 - window) : 0;
          dim_t end = causal ? std::min(valid, offset + row + 1) : valid;
          if (lengths)
            end = std::min(
                end,
                std::max<dim_t>(0, (*lengths)[(batch * h + head) * t + row]));
          if (end <= first)
            continue;
          std::vector<double> scores(end - first);
          double maximum = -std::numeric_limits<double>::infinity();
          for (dim_t key = first; key < end; ++key) {
            double score = 0;
            for (dim_t col = 0; col < d; ++col)
              score +=
                  double(qv[((batch * t + row) * h + head) * d + col]) *
                  kv[((batch * cap + key) * kh + head / (h / kh)) * d + col];
            scores[key - first] = score / std::sqrt(float(d));
            maximum = std::max(maximum, scores[key - first]);
          }
          double denominator = 0;
          for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (dim_t col = 0; col < d; ++col) {
            double value = 0;
            for (dim_t key = first; key < end; ++key)
              value +=
                  scores[key - first] *
                  vv[((batch * cap + key) * kh + head / (h / kh)) * d + col];
            result[((batch * t + row) * h + head) * d + col] =
                value / denominator;
          }
        }
    return StorageView(q.shape(), result);
  }

  StorageView signal(Shape shape, DataType dtype, float frequency) {
    dim_t size = 1;
    for (auto x : shape)
      size *= x;
    std::vector<float> values(size);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = std::sin(float(i) * frequency);
    return StorageView(shape, values).to(dtype);
  }
} // namespace

TEST(MPSBackendTest, WindowAttentionAdapterMatchesReferenceAndFallbacks) {
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {13, 80, 256, 257})
      for (dim_t count : {1, 33, 65}) {
        SCOPED_TRACE(::testing::Message() << int(dtype) << ' ' << depth << ' ' << count);
        constexpr dim_t batch = 2, heads = 2, window = 17, offset = 257;
        const dim_t length = offset + count;
        auto q = signal({batch, count, heads, depth}, dtype, .13f),
          k = signal({batch, length, heads, depth}, dtype, .07f),
          v = signal(k.shape(), dtype, .19f);
        std::vector<int32_t> bounds(batch * heads * count);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t h = 0; h < heads; ++h)
            for (dim_t t = 0; t < count; ++t)
              bounds[(b * heads + h) * count + t] =
                std::min(offset + t + 1, length - b - h);
        StorageView lengths({batch, heads, count}, bounds, Device::MPS),
          qg(dtype, Device::MPS), kg(dtype, Device::MPS), vg(dtype, Device::MPS),
          output(dtype, Device::MPS), result(dtype, Device::MPS),
          position_bias(dtype, Device::MPS);
        ops::Transpose transpose({0, 2, 1, 3});
        transpose(q.to(Device::MPS), qg);
        transpose(k.to(Device::MPS), kg);
        transpose(v.to(Device::MPS), vg);
        // An empty bias object is the normal layer call, not an active bias.
        layers::dot_product_attention(qg, kg, vg, &lengths, nullptr, nullptr,
          nullptr, nullptr, 0, 0, 0, output, nullptr, true,
          1.f / std::sqrt(float(depth)), true, true, 1, nullptr,
          &position_bias, false, window, offset);
        transpose(output, result);
        expect_storage_eq(result.to_float32(),
          reference_attention(q, k, v, length, offset, window, &bounds),
          dtype == DataType::FLOAT32 ? 2e-6f : dtype == DataType::FLOAT16 ? .002f : .02f);
      }
}

TEST(MPSBackendTest, WindowAttentionAdapterPreservesCausalPrefix) {
  constexpr dim_t heads = 4, depth = 8;
  const auto q = signal({1, 65, heads, depth}, DataType::FLOAT16, .13f),
    k = signal(q.shape(), DataType::FLOAT16, .07f),
    v = signal(q.shape(), DataType::FLOAT16, .19f);
  std::vector<float16_t> full;
  for (dim_t count : {65, 64}) {
    auto prefix = [&](const StorageView& x) {
      auto data = x.to_vector<float16_t>();
      data.resize(count * heads * depth);
      return StorageView({1, count, heads, depth}, data, Device::MPS);
    };
    StorageView qg(DataType::FLOAT16, Device::MPS), kg(qg), vg(qg), output(qg), result(qg);
    ops::Transpose transpose({0, 2, 1, 3});
    transpose(prefix(q), qg);
    transpose(prefix(k), kg);
    transpose(prefix(v), vg);
    layers::dot_product_attention(qg, kg, vg, nullptr, nullptr, nullptr,
      nullptr, nullptr, 0, 0, 0, output, nullptr, true,
      1.f / std::sqrt(float(depth)), true, false, 1, nullptr,
      nullptr, false, 8, 0);
    transpose(output, result);
    if (count == 65)
      full = result.to_vector<float16_t>();
    else {
      full.resize(count * heads * depth);
      expect_storage_eq(result, StorageView({1, count, heads, depth}, full));
    }
  }
}

TEST(MPSBackendTest, QuantizedKvPackingMatchesIndependentGroupsAndTails) {
  for (unsigned bits : {4u, 5u, 8u})
    for (dim_t depth : {13, 32, 33, 64, 80, 128, 255, 256}) {
      SCOPED_TRACE(::testing::Message() << bits << " bits, depth " << depth);
      constexpr dim_t batch = 2, time = 7, heads = 3, capacity = 512;
      auto k = signal({batch, time, heads, depth}, DataType::FLOAT16, .13f);
      auto v = signal(k.shape(), DataType::FLOAT16, .07f);
      auto kg = k.to(Device::MPS), vg = v.to(Device::MPS);
      StorageView ck(
          {batch, capacity, heads, mps::packed_kv_row_bytes(depth, bits)},
          DataType::INT8, Device::MPS),
          cv(ck);
      mps::append_quantized_kv(kg.buffer(), vg.buffer(), nullptr, nullptr,
                               ck.buffer(), cv.buffer(), batch, time, heads,
                               depth, 0, 0, capacity, true, bits);
      StorageView decoded(k.shape(), DataType::FLOAT16, Device::MPS);
      mps::unpack_quantized_kv(ck.buffer(), decoded.buffer(), batch, time,
                               heads, depth, capacity, bits);
      expect_storage_eq(decoded.to(Device::CPU),
                        quantized_kv_reference(k, bits), .001f);
      mps::unpack_quantized_kv(cv.buffer(), decoded.buffer(), batch, time,
                               heads, depth, capacity, bits);
      expect_storage_eq(decoded.to(Device::CPU),
                        quantized_kv_reference(v, bits), .001f);
      // CPU inspection checks the actual format, including straddling 5-bit
      // codes and zero padding in the last group. No CT2 unpack helper is used.
      const auto bytes = ck.to(Device::CPU).to_vector<int8_t>();
      const auto values = k.to_float32().to_vector<float>();
      const dim_t width = mps::packed_kv_row_bytes(depth, bits);
      for (dim_t b = 0; b < batch; ++b)
        for (dim_t t = 0; t < time; ++t)
          for (dim_t h = 0; h < heads; ++h)
            for (dim_t start = 0; start < depth; start += 32) {
              const auto* p = reinterpret_cast<const uint8_t*>(bytes.data()) +
                              ((b * capacity + t) * heads + h) * width +
                              (start / 32) * (4 + 4 * bits);
              float scale;
              std::memcpy(&scale, p, sizeof(scale));
              ASSERT_TRUE(std::isfinite(scale));
              ASSERT_GT(scale, 0);
              float maximum = 0;
              for (dim_t d = start; d < std::min(start + 32, depth); ++d)
                maximum = std::max(
                    maximum,
                    std::abs(values[((b * time + t) * heads + h) * depth + d]));
              EXPECT_NEAR(scale,
                          maximum == 0
                              ? 1.f
                              : maximum * (1.f / ((1 << (bits - 1)) - 1)),
                          1e-7f);
              for (unsigned d = 0; d < 32; ++d) {
                const unsigned bit = d * bits;
                unsigned code = p[4 + bit / 8] >> (bit % 8);
                if (bit % 8 + bits > 8)
                  code |= unsigned(p[5 + bit / 8]) << (8 - bit % 8);
                code &= (1u << bits) - 1;
                const float x =
                    start + d < unsigned(depth)
                        ? values[((b * time + t) * heads + h) * depth + start +
                                 d]
                        : 0;
                EXPECT_EQ(int(code) - (1 << (bits - 1)),
                          maximum == 0
                              ? 0
                              : int(std::nearbyint(x * ((1 << (bits - 1)) - 1) /
                                                   maximum)));
              }
            }
    }
}

TEST(MPSBackendTest, QuantizedKvPrefillAndDecodeMatchIndependentAttention) {
  for (unsigned bits : {4u, 5u, 8u})
    for (dim_t depth : {13, 64, 128, 256})
      for (dim_t kvheads : {1, 2})
        for (dim_t time : {7, 65, 2051, 16385}) {
          SCOPED_TRACE(::testing::Message()
                       << bits << '/' << depth << '/' << time);
          constexpr dim_t batch = 2, heads = 4;
          const dim_t qlen = time >= 2051 ? 1 : time, offset = time - qlen;
          const dim_t capacity = ((time + 511) / 512) * 512;
          auto q = signal({batch, qlen, heads, depth}, DataType::FLOAT16, .13f);
          auto k =
              signal({batch, time, kvheads, depth}, DataType::FLOAT16, .07f);
          auto v = signal(k.shape(), DataType::FLOAT16, .11f);
          auto qg = q.to(Device::MPS), kg = k.to(Device::MPS),
               vg = v.to(Device::MPS);
          StorageView ck(DataType::INT8, Device::MPS),
              cv(DataType::INT8, Device::MPS);
          StorageView out(DataType::FLOAT16, Device::MPS);
          const auto rk = quantized_kv_reference(k, bits),
                     rv = quantized_kv_reference(v, bits);
          if (offset) {
            ck = StorageView({batch, capacity, kvheads,
                              mps::packed_kv_row_bytes(depth, bits)},
                             int8_t(-1), Device::MPS);
            cv = StorageView(ck);
            mps::append_quantized_kv(kg.buffer(), vg.buffer(), nullptr, nullptr,
                                     ck.buffer(), cv.buffer(), batch, time,
                                     kvheads, depth, 0, 0, capacity, true,
                                     bits);
            StorageView tail_k(DataType::FLOAT16, Device::MPS),
                tail_v(DataType::FLOAT16, Device::MPS);
            ops::Slide(1, time - 1, 1)(kg, tail_k);
            ops::Slide(1, time - 1, 1)(vg, tail_v);
            kg = std::move(tail_k);
            vg = std::move(tail_v);
          }
          std::vector<int32_t> bounds(batch * heads * qlen);
          for (size_t i = 0; i < bounds.size(); ++i)
            bounds[i] = i % 7 == 0 ? 0 : time - 3;
          StorageView lengths({batch, heads, qlen}, bounds, Device::MPS);
          for (bool masked : {false, true})
            for (dim_t window : {0, 17, 257}) {
              mps::flash_attention(qg, kg, vg, masked ? &lengths : nullptr, out,
                                   &ck, &cv, nullptr, true,
                                   1.f / std::sqrt(float(depth)), offset,
                                   window, true, false, bits);
              expect_storage_eq(out.to_float32(),
                                reference_attention(q, rk, rv, time, offset,
                                                    window,
                                                    masked ? &bounds : nullptr),
                                .003f);
            }
        }
}

TEST(MPSBackendTest, QuantizedKvGrowthAndBeamGatherPreservePackedRows) {
  for (unsigned bits : {4u, 5u, 8u}) {
    constexpr dim_t batch = 2, heads = 2, depth = 33;
    const dim_t width = mps::packed_kv_row_bytes(depth, bits);
    StorageView ck(DataType::INT8, Device::MPS),
        cv(DataType::INT8, Device::MPS);
    StorageView expected(DataType::FLOAT16), expected_v(DataType::FLOAT16),
        out(DataType::FLOAT16, Device::MPS);
    dim_t offset = 0;
    const void* first = nullptr;
    for (dim_t time : {511, 1, 1, 33}) {
      auto q = signal({batch, time, 4, depth}, DataType::FLOAT16, .13f);
      auto k = signal({batch, time, heads, depth}, DataType::FLOAT16, .07f);
      auto v = signal(k.shape(), DataType::FLOAT16, .11f);
      auto qg = q.to(Device::MPS), kg = k.to(Device::MPS),
           vg = v.to(Device::MPS);
      mps::flash_attention(qg, kg, vg, nullptr, out, &ck, &cv, nullptr, true,
                           1.f / std::sqrt(float(depth)), offset, 17, true,
                           false, bits);
      auto ref = quantized_kv_reference(k, bits);
      auto ref_v = quantized_kv_reference(v, bits);
      if (!offset) {
        expected = ref;
        expected_v = ref_v;
      } else {
        StorageView joined(DataType::FLOAT16), joined_v(DataType::FLOAT16);
        ops::Concat(1)({&expected, &ref}, joined);
        expected = std::move(joined);
        ops::Concat(1)({&expected_v, &ref_v}, joined_v);
        expected_v = std::move(joined_v);
      }
      expect_storage_eq(out.to_float32(),
                        reference_attention(q, expected, expected_v,
                                            offset + time, offset, 17),
                        .003f);
      if (!offset)
        first = ck.buffer();
      if (offset == 511)
        EXPECT_EQ(first, ck.buffer());
      if (offset == 512)
        EXPECT_NE(first, ck.buffer());
      offset += time;
    }
    EXPECT_EQ(ck.shape(), (Shape{batch, 1024, heads, width}));
    StorageView ids({3}, std::vector<int32_t>{1, 1, 0});
    auto indices = ids.to(Device::MPS);
    StorageView reordered(DataType::INT8, Device::MPS), ref(DataType::FLOAT16),
        decoded({3, offset, heads, depth}, DataType::FLOAT16, Device::MPS);
    ops::Gather()(ck, indices, reordered);
    ops::Gather()(expected, ids, ref);
    mps::unpack_quantized_kv(reordered.buffer(), decoded.buffer(), 3, offset,
                             heads, depth, 1024, bits);
    expect_storage_eq(decoded.to(Device::CPU), ref, .001f);
    ops::Gather()(cv, indices, reordered);
    ops::Gather()(expected_v, ids, ref);
    mps::unpack_quantized_kv(reordered.buffer(), decoded.buffer(), 3, offset,
                             heads, depth, 1024, bits);
    expect_storage_eq(decoded.to(Device::CPU), ref, .001f);
  }
}

TEST(MPSBackendTest, FlashPrefillMatchesDoubleAcrossMasksWindowsAndTails) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {13, 64, 128, 256})
      for (dim_t qlen : {7, 33, 129}) {
        SCOPED_TRACE(::testing::Message()
                     << int(dtype) << ' ' << depth << ' ' << qlen);
        constexpr dim_t batch = 2, heads = 4, kvheads = 2, offset = 17;
        const dim_t valid = offset + qlen, cap = 512;
        auto q = signal({batch, qlen, heads, depth}, dtype, .13f);
        std::vector<float> kv(batch * cap * kvheads * depth), vv(kv.size());
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t t = 0; t < cap; ++t)
            for (dim_t j = 0; j < kvheads * depth; ++j) {
              const dim_t i = (b * cap + t) * kvheads * depth + j;
              kv[i] = t < valid - b * 5
                          ? std::sin(float(i) * .07f)
                          : std::numeric_limits<float>::quiet_NaN();
              vv[i] = t < valid - b * 5
                          ? std::cos(float(i) * .11f)
                          : std::numeric_limits<float>::quiet_NaN();
            }
        auto k = StorageView({batch, cap, kvheads, depth}, kv).to(dtype);
        auto v = StorageView(k.shape(), vv).to(dtype);
        std::vector<int32_t> lengths(batch * heads * qlen);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t h = 0; h < heads; ++h)
            for (dim_t r = 0; r < qlen; ++r)
              lengths[(b * heads + h) * qlen + r] = h == 0 ? 0 : valid - b * 5;
        auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
             vgpu = v.to(Device::MPS);
        StorageView mask({batch, heads, qlen}, lengths, Device::MPS),
            out(q.shape(), dtype, Device::MPS);
        for (dim_t window : {0, 19}) {
          ASSERT_TRUE(mps::flash_prefill(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              mask.data<int32_t>(), out.buffer(), batch, heads, kvheads, qlen,
              valid, cap, depth, 1.f / std::sqrt(float(depth)), offset, window,
              true, false));
          expect_storage_eq(
              out.to_float32(),
              reference_attention(q, k, v, valid, offset, window, &lengths),
              dtype == DataType::FLOAT16 ? .002f : .016f);
        }
        if (depth == 13 && qlen == 33) {
          std::vector<int32_t> interleaved(lengths.size());
          for (dim_t b = 0; b < batch; ++b)
            for (dim_t h = 0; h < heads; ++h)
              for (dim_t r = 0; r < qlen; ++r)
                interleaved[(b * qlen + r) * heads + h] =
                    lengths[(b * heads + h) * qlen + r];
          auto imask =
              StorageView({batch, qlen, heads}, interleaved, Device::MPS);
          ASSERT_TRUE(mps::flash_prefill(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              imask.data<int32_t>(), out.buffer(), batch, heads, kvheads, qlen,
              valid, cap, depth, 1.f / std::sqrt(float(depth)), offset, 0,
              false, true));
          expect_storage_eq(
              out.to_float32(),
              reference_attention(q, k, v, valid, offset, 0, &lengths, false),
              dtype == DataType::FLOAT16 ? .002f : .016f);
        }
      }
}

TEST(MPSBackendTest, FlashPrefillLongMqaAndLargeScoresAreStable) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t n = 513, depth = 128;
    auto q = StorageView({1, n, 2, depth}, 80.f).to(dtype);
    auto k = StorageView({1, n, 1, depth}, 80.f).to(dtype);
    auto v = signal(k.shape(), dtype, .11f);
    auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
         vgpu = v.to(Device::MPS);
    StorageView out(q.shape(), dtype, Device::MPS);
    ASSERT_TRUE(
        mps::flash_prefill(dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
                           nullptr, out.buffer(), 1, 2, 1, n, n, n, depth,
                           1.f / std::sqrt(float(depth)), 0, 0, true, false));
    expect_storage_eq(out.to_float32(), reference_attention(q, k, v, n, 0, 0),
                      dtype == DataType::FLOAT16 ? .002f : .016f);
  }
}

TEST(MPSBackendTest, FlashDecodeMatchesDoubleReferenceAcrossSplitsGqaAndMasks) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {dim_t(13), dim_t(64), dim_t(65), dim_t(128), dim_t(129),
                        dim_t(255), dim_t(256)})
      for (dim_t valid : {dim_t(65), dim_t(257), dim_t(2049)}) {
        SCOPED_TRACE(::testing::Message()
                     << int(dtype) << ' ' << depth << ' ' << valid);
        constexpr dim_t batch = 2, heads = 6, kvheads = 2;
        const dim_t cap = ((valid + 511) / 512) * 512;
        const auto q = signal({batch, 1, heads, depth}, dtype, .13f);
        const auto k = signal({batch, cap, kvheads, depth}, dtype, .07f);
        const auto v = signal(k.shape(), dtype, .11f);
        auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
             vgpu = v.to(Device::MPS);
        for (dim_t window : {dim_t(0), dim_t(17)}) {
          std::vector<int32_t> lens(batch * heads, valid);
          lens[0] = -1;
          lens[1] = 0;
          lens[2] = int32_t(valid + 3);
          lens[3] = int32_t(valid / 2);
          const auto expected =
              reference_attention(q, k, v, valid, valid - 1, window, &lens);
          StorageView bounds({batch * heads}, lens, Device::MPS),
              out(q.shape(), dtype, Device::MPS);
          const dim_t splits = (valid + 255) / 256;
          StorageView workspace({batch, heads, splits, depth + 2},
                                DataType::FLOAT32, Device::MPS);
          ASSERT_TRUE(mps::flash_decode(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              bounds.data<int32_t>(), out.buffer(), workspace.data<float>(),
              workspace.size(), batch, heads, kvheads, valid, cap, depth,
              1.f / std::sqrt(float(depth)), valid - 1, window));
          expect_storage_eq(out.to_float32(), expected,
                            dtype == DataType::FLOAT16 ? .002f : .016f);
        }
      }
}

TEST(MPSBackendTest, FlashDecodeStableLargeScoresMqaAndPoisonedUnusedCache) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t valid = 513, cap = 1024, depth = 128, heads = 4;
    const auto q = StorageView({1, 1, heads, depth}, 80.f).to(dtype);
    std::vector<float> kv(cap * depth, std::numeric_limits<float>::quiet_NaN()),
        vv(kv);
    for (dim_t key = 0; key < valid; ++key)
      for (dim_t d = 0; d < depth; ++d) {
        kv[key * depth + d] = key % 2 ? 80.f : -80.f;
        vv[key * depth + d] = std::sin(float(key + d) * .1f);
      }
    const auto k = StorageView({1, cap, 1, depth}, kv).to(dtype),
               v = StorageView(k.shape(), vv).to(dtype);
    auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
         vgpu = v.to(Device::MPS);
    StorageView out(q.shape(), dtype, Device::MPS),
        work({heads, 3, depth + 2}, DataType::FLOAT32, Device::MPS);
    ASSERT_TRUE(mps::flash_decode(
        dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(), nullptr,
        out.buffer(), work.data<float>(), work.size(), 1, heads, 1, valid, cap,
        depth, 1.f / std::sqrt(float(depth)), valid - 1));
    expect_storage_eq(out.to_float32(),
                      reference_attention(q, k, v, valid, valid - 1, 0), .016f);
  }
}

TEST(MPSBackendTest, FlashDecodeCooperativeChunksRespectMasksAndTokenOrder) {
  constexpr dim_t batch = 2, queries = 3, heads = 6, kvheads = 2;
  constexpr dim_t valid = 257, capacity = 512, offset = valid - queries;
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {65, 128}) {
      const auto q = signal({batch, queries, heads, depth}, dtype, .13f);
      const auto k = signal({batch, capacity, kvheads, depth}, dtype, .07f);
      const auto v = signal(k.shape(), dtype, .11f);
      auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
           vgpu = v.to(Device::MPS);
      std::vector<int32_t> lengths(batch * heads * queries, valid);
      lengths[0] = 0;
      lengths[3] = valid / 2;
      StorageView bounds({batch, heads, queries}, lengths, Device::MPS),
          out(q.shape(), dtype, Device::MPS),
          workspace({batch, heads, queries, 2, depth + 2}, DataType::FLOAT32,
                    Device::MPS);
      for (bool causal : {false, true})
        for (dim_t window : {0, 17}) {
          if (!causal && window)
            continue;
          ASSERT_TRUE(mps::flash_decode(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              bounds.data<int32_t>(), out.buffer(), workspace.data<float>(),
              workspace.size(), batch, heads, kvheads, valid, capacity, depth,
              1.f / std::sqrt(float(depth)), offset, window, causal, queries,
              true, true));
          expect_storage_eq(out.to_float32(),
                            reference_attention(q, k, v, valid, offset, window,
                                                &lengths, causal),
                            dtype == DataType::FLOAT16 ? .002f : .016f);
        }
    }
}

TEST(MPSBackendTest,
     SharedFlashDecodeBhtdAndInterleavedBeamMatchDoubleReference) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t queries : {dim_t(1), dim_t(5)})
      for (dim_t keys : {dim_t(33), dim_t(1501)}) {
        constexpr dim_t batch = 2, heads = 3, depth = 13;
        auto q = signal({batch, queries, heads, depth}, dtype, .13f);
        auto k = signal({batch, keys, heads, depth}, dtype, .07f),
             v = signal(k.shape(), dtype, .11f);
        StorageView kt(dtype), vt(dtype), qt(dtype);
        ops::Transpose({0, 2, 1, 3})(k, kt);
        ops::Transpose({0, 2, 1, 3})(v, vt);
        const bool interleaved = queries > 1;
        if (!interleaved)
          ops::Transpose({0, 2, 1, 3})(q, qt);
        std::vector<int32_t> mask(batch * heads * queries, keys);
        mask[0] = 0;
        mask[1] = -1;
        mask[2] = keys + 9;
        mask.back() = keys / 2;
        const auto expected =
            reference_attention(q, k, v, keys, keys, 0, &mask);
        auto qgpu = (interleaved ? q : qt).to(Device::MPS);
        auto kgpu = kt.to(Device::MPS), vgpu = vt.to(Device::MPS);
        StorageView lengths({batch, heads, queries}, mask, Device::MPS),
            out(dtype, Device::MPS), converted(dtype, Device::MPS);
        ASSERT_TRUE(ops::FusedAttention()(qgpu, kgpu, vgpu, &lengths,
                                          1.f / std::sqrt(float(depth)),
                                          interleaved, out));
        if (interleaved)
          converted = std::move(out);
        else
          ops::Transpose({0, 2, 1, 3})(out, converted);
        expect_storage_eq(converted.to_float32(), expected,
                          dtype == DataType::FLOAT16 ? .002f : .016f);
      }
}

TEST(MPSBackendTest,
     SharedFlashKeepsGenericFallbackForPrefillAndLargeHeadDimensions) {
  for (auto shape : {Shape{1, 2, 17, 64}, Shape{1, 2, 1, 257}}) {
    StorageView q(shape, float16_t(.1f), Device::MPS),
        k({1, 2, 33, shape.back()}, float16_t(.2f), Device::MPS),
        v(k.shape(), float16_t(.3f), Device::MPS),
        out(DataType::FLOAT16, Device::MPS);
    EXPECT_FALSE(ops::FusedAttention()(q, k, v, nullptr, .125f, false, out));
  }
}

TEST(MPSBackendTest,
     CompactFlashCacheAppendGrowthAndMultiTokenChunksMatchReference) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t batch = 2, heads = 4, kvheads = 2, depth = 13;
    StorageView ck(dtype, Device::MPS), cv(dtype, Device::MPS);
    StorageView allk(dtype), allv(dtype);
    dim_t offset = 0;
    const void* initial = nullptr;
    for (dim_t count : {dim_t(511), dim_t(1), dim_t(3), dim_t(1)}) {
      const auto q = signal({batch, count, heads, depth}, dtype, .13f);
      const auto k = signal({batch, count, kvheads, depth}, dtype, .07f),
                 v = signal(k.shape(), dtype, .11f);
      if (!offset) {
        allk = k;
        allv = v;
      } else {
        StorageView nk(dtype), nv(dtype);
        ops::Concat(1)({&allk, &k}, nk);
        ops::Concat(1)({&allv, &v}, nv);
        allk = std::move(nk);
        allv = std::move(nv);
      }
      auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
           vgpu = v.to(Device::MPS);
      StorageView out(dtype, Device::MPS);
      mps::flash_attention(qgpu, kgpu, vgpu, nullptr, out, &ck, &cv, nullptr,
                           true, 1.f / std::sqrt(float(depth)), offset, 17);
      expect_storage_eq(
          out.to_float32(),
          reference_attention(q, allk, allv, offset + count, offset, 17),
          dtype == DataType::FLOAT16 ? .003f : .025f);
      if (!offset) {
        initial = ck.buffer();
        EXPECT_EQ(ck.dim(1), 512);
      }
      if (offset == 511)
        EXPECT_EQ(ck.buffer(), initial);
      if (offset == 512) {
        EXPECT_NE(ck.buffer(), initial);
        EXPECT_EQ(ck.dim(1), 1024);
      }
      StorageView prefix(dtype, Device::MPS);
      ops::Slide(1, 0, offset + count)(ck, prefix);
      expect_storage_eq(prefix.to(Device::CPU), allk, 0);
      ops::Slide(1, 0, offset + count)(cv, prefix);
      expect_storage_eq(prefix.to(Device::CPU), allv, 0);
      offset += count;
    }
  }
}

TEST(MPSBackendTest,
     InterleavedRotaryMatchesCpuAcrossHeadsBatchesAndPartialDimensions) {
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
    for (bool interleave : {false, true}) {
      const auto input = signal({2, 5, 3, 10}, dtype, .13f);
      const auto sine = signal({5, 6}, dtype, .07f),
                 cosine = signal({5, 6}, dtype, .11f);
      StorageView reference, actual(dtype, Device::MPS), transposed, rotated;
      ops::Rotary rotary(6, interleave);
      // The CPU primitive's established layout is BHTD. Transpose explicitly
      // so this reference does not inherit its interleaved-layout limitation.
      ops::Transpose({0, 2, 1, 3})(input.to_float32(), transposed);
      rotary(transposed, sine.to_float32(), cosine.to_float32(), rotated, true);
      ops::Transpose({0, 2, 1, 3})(rotated, reference);
      rotary(input.to(Device::MPS), sine.to(Device::MPS),
             cosine.to(Device::MPS), actual, false);
      expect_storage_eq(actual.to_float32(), reference.to_float32(),
                        dtype == DataType::BFLOAT16 ? .016f : .002f);
    }
}

TEST(MPSBackendTest,
     FlashMqaInterleavedMasksAndCacheBeamReorderingPreserveTokenOrder) {
  constexpr dim_t batch = 2, heads = 4, depth = 13, time = 7;
  const auto dtype = DataType::FLOAT16;
  auto q = signal({batch, time, heads, depth}, dtype, .13f);
  auto k = signal({batch, time, 1, depth}, dtype, .07f),
       v = signal(k.shape(), dtype, .11f);
  std::vector<int32_t> mask(batch * heads * time), interleaved(mask.size());
  for (dim_t b = 0; b < batch; ++b)
    for (dim_t h = 0; h < heads; ++h)
      for (dim_t t = 0; t < time; ++t) {
        const int32_t value = std::min<dim_t>(t + 1, b == 0 ? 3 : time);
        mask[(b * heads + h) * time + t] = value;
        interleaved[(b * time + t) * heads + h] = value;
      }
  StorageView lengths({batch, time, heads}, interleaved, Device::MPS);
  auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
       vgpu = v.to(Device::MPS);
  StorageView ck(dtype, Device::MPS), cv(dtype, Device::MPS),
      out(dtype, Device::MPS);
  mps::flash_attention(qgpu, kgpu, vgpu, &lengths, out, &ck, &cv, nullptr, true,
                       1.f / std::sqrt(float(depth)), 0, 0, true, true);
  expect_storage_eq(out.to_float32(),
                    reference_attention(q, k, v, time, 0, 0, &mask), .003f);
  StorageView indices({3}, std::vector<int32_t>{1, 1, 0});
  auto gpuindices = indices.to(Device::MPS);
  StorageView reordered_k(dtype, Device::MPS), reordered_v(dtype, Device::MPS),
      refk(dtype), refv(dtype);
  ops::Gather()(ck, gpuindices, reordered_k);
  ops::Gather()(cv, gpuindices, reordered_v);
  ops::Gather()(k, indices, refk);
  ops::Gather()(v, indices, refv);
  ck = std::move(reordered_k);
  cv = std::move(reordered_v);
  q = signal({3, 1, heads, depth}, dtype, .13f);
  k = signal({3, 1, 1, depth}, dtype, .17f);
  v = signal(k.shape(), dtype, .19f);
  StorageView allk(dtype), allv(dtype);
  ops::Concat(1)({&refk, &k}, allk);
  ops::Concat(1)({&refv, &v}, allv);
  qgpu = q.to(Device::MPS);
  kgpu = k.to(Device::MPS);
  vgpu = v.to(Device::MPS);
  mps::flash_attention(qgpu, kgpu, vgpu, nullptr, out, &ck, &cv, nullptr, true,
                       1.f / std::sqrt(float(depth)), time, 0);
  expect_storage_eq(out.to_float32(),
                    reference_attention(q, allk, allv, time + 1, time, 0),
                    .003f);
}
