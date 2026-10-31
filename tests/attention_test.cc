#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#include "layers/attention_utils.h"

#include <algorithm>
#include <cmath>

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"

TEST(AttentionTest, Float16ScoreAccumulation) {
  const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(true);
  // The unscaled dot product (160000) overflows FP16; the scaled score (20000) fits.
  for (const dim_t heads : {1, 2}) {
    StorageView queries = StorageView({1, heads, 1, 64}, 50.f).to(Device::CUDA).to_float16();
    StorageView values = StorageView({1, heads, 2, 64}, 2.f).to(Device::CUDA).to_float16();
    StorageView relative_keys = StorageView({3, 64}, 50.f).to(Device::CUDA).to_float16();
    for (const bool use_relative_keys : {false, true}) {
      StorageView keys = StorageView({1, heads, 2, 64}, use_relative_keys ? 0.f : 50.f)
        .to(Device::CUDA).to_float16();
      StorageView output(DataType::FLOAT16, Device::CUDA);
      StorageView attention(DataType::FLOAT16, Device::CUDA);
      layers::dot_product_attention(queries, keys, values, nullptr,
                                    use_relative_keys ? &relative_keys : nullptr,
                                    nullptr, nullptr, nullptr,
                                    0, 0, 1, output, &attention, true, 0.125f);
      EXPECT_TRUE(cuda::use_true_fp16_gemm());
      expect_storage_eq(output.to_float32(), StorageView({1, heads, 1, 64}, 2.f));
      expect_storage_eq(attention.to_float32(), StorageView({1, heads, 1, 2}, 0.5f));
    }
  }
}

TEST(AttentionTest, Float16AlignmentWithCachedAlibi) {
  StorageView queries = StorageView({1, 1, 1, 64}, 1.f).to(Device::CUDA).to_float16();
  StorageView keys = StorageView({1, 1, 2, 64}, 1.f).to(Device::CUDA).to_float16();
  StorageView values = StorageView({1, 1, 2, 64}, 2.f).to(Device::CUDA).to_float16();
  layers::Alibi alibi;
  for (const bool return_attention : {false, true, false, true}) {
    StorageView output(DataType::FLOAT16, Device::CUDA);
    StorageView attention(DataType::FLOAT16, Device::CUDA);
    layers::dot_product_attention(queries, keys, values, nullptr,
                                  nullptr, nullptr, nullptr, nullptr,
                                  0, 0, 0, output,
                                  return_attention ? &attention : nullptr,
                                  true, 0.125f, true, true, 1, &alibi);
    expect_storage_eq(output.to_float32(), StorageView({1, 1, 1, 64}, 2.f), 0.002f);
  }
}

#endif

TEST(AttentionTest, AlignmentScorePrecision) {
  std::vector<Device> devices{Device::CPU};
#ifdef CT2_WITH_CUDA
  devices.push_back(Device::CUDA);
#endif
  for (const Device device : devices) {
    const DataType dtype = device == Device::CUDA ? DataType::FLOAT16 : DataType::FLOAT32;
    for (const dim_t heads : {1, 2}) {
      for (const dim_t beam : {1, 2}) {
        // Scores 32768 and 32764 round to the same FP16 number before softmax.
        StorageView queries({1, heads, beam, 64}, 64.f);
        std::vector<float> keys_data(heads * 3 * 64, 64.f);
        std::vector<float> values_data(heads * 3 * 64, 0.f);
        for (dim_t h = 0; h < heads; ++h) {
          keys_data[h * 3 * 64 + 64] = 63.5f;
          std::fill_n(values_data.begin() + h * 3 * 64, 64, 1.f);
        }
        StorageView keys({1, heads, 3, 64}, keys_data);
        StorageView values({1, heads, 3, 64}, values_data);
        StorageView lengths({heads * beam}, std::vector<int32_t>(heads * beam, 2));
        queries = queries.to(device).to(dtype);
        keys = keys.to(device).to(dtype);
        values = values.to(device).to(dtype);
        lengths = lengths.to(device);
        for (const bool normalized : {false, true}) {
          StorageView output(dtype, device), attention(dtype, device);
          layers::dot_product_attention(queries, keys, values, &lengths,
                                        nullptr, nullptr, nullptr, nullptr,
                                        0, 0, 0, output, &attention, normalized,
                                        0.125f, true, true, beam);
          const float winner = 1.f / (1.f + std::exp(-4.f));
          expect_storage_eq(output.to_float32(),
                            StorageView({1, heads, beam, 64}, winner), 0.001f);
          std::vector<float> expected;
          for (dim_t row = 0; row < heads * beam; ++row) {
            const std::vector<float> scores = normalized
              ? std::vector<float>{winner, 1.f - winner, 0.f}
              : std::vector<float>{32768.f, 32764.f, 32768.f};
            expected.insert(expected.end(), scores.begin(), scores.end());
          }
          expect_storage_eq(attention.to_float32(),
                            StorageView({beam, heads, 1, 3}, expected), 0.001f);
          EXPECT_EQ(output.dtype(), dtype);
          EXPECT_EQ(attention.dtype(), normalized ? dtype : DataType::FLOAT32);
        }
      }
    }
  }
}

class MockModel : public models::Model {
public:
  MockModel(dim_t num_heads, dim_t num_heads_kv) {
    const dim_t d_model = 64;
    const dim_t d_head = d_model / num_heads;
    
    std::vector<float> linear_0_data(num_heads * d_head * d_model, 0.01f);
    std::vector<float> linear_1_data(2 * num_heads_kv * d_head * d_model, 0.01f);
    
    register_variable("attn/linear_0/weight",
                      StorageView({num_heads * d_head, d_model}, linear_0_data));
    register_variable("attn/linear_1/weight",
                      StorageView({2 * num_heads_kv * d_head, d_model}, linear_1_data));
    register_variable("attn/linear_2/weight",
                      StorageView({d_model, num_heads * d_head}, DataType::FLOAT32));
    register_variable("attn/q_norm/gamma",
                      StorageView({d_model}, std::vector<float>(d_model, 1.0f)));
    register_variable("attn/k_norm/gamma",
                      StorageView({d_head}, std::vector<float>(d_head, 1.0f)));
    
    register_variable("attn/num_heads_kv",
                      StorageView(static_cast<int32_t>(num_heads_kv)));

    set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
  }
protected:
  std::unique_ptr<Model> clone() const override { return nullptr; }
};

class TestableAttention : public layers::MultiHeadAttention {
public:
  using MultiHeadAttention::MultiHeadAttention;
  using MultiHeadAttention::process_cross_attention;
};

class CrossAttentionTest : public ::testing::Test {
protected:
  static constexpr dim_t NUM_HEADS = 4;
  static constexpr dim_t D_MODEL = 64;
  static constexpr dim_t D_HEAD = D_MODEL / NUM_HEADS;
  static constexpr dim_t BATCH = 2;
  static constexpr dim_t Q_LEN = 6;
  static constexpr dim_t V_LEN = 8;

  float get_4d(const StorageView& view, dim_t b, dim_t h, dim_t t, dim_t d) {
    const auto& shape = view.shape();
    return view.data<float>()[b * shape[1] * shape[2] * shape[3] +
                              h * shape[2] * shape[3] + t * shape[3] + d];
  }

};

// MQA: All heads share same K/V
TEST_F(CrossAttentionTest, MultiQueryAttention) {
  MockModel model(NUM_HEADS, /*num_heads_kv=*/1);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);
  // Use non-uniform values to verify normalization is applied
  std::vector<float> value_data(BATCH * V_LEN * D_MODEL);
  for (size_t i = 0; i < value_data.size(); ++i)
    value_data[i] = static_cast<float>(i % 10 + 1);
  std::vector<float> fused_data(BATCH * Q_LEN * NUM_HEADS * D_HEAD);
  for (size_t i = 0; i < fused_data.size(); ++i)
    fused_data[i] = static_cast<float>(i % 10 + 1);
  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, value_data);
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, fused_data);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;
  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);
  // MQA: K/V are replicated to 4D format [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  // Verify K/V values are consistent across batch and time dimensions
  // (In MQA, there's only one set of K/V, so we just verify the tensor is valid)
  float k0 = get_4d(cached_keys, 0, 0, 0, 0);
  float v0 = get_4d(cached_values, 0, 0, 0, 0);
  EXPECT_NE(k0, 0.0f) << "K values should be non-zero after projection";
  EXPECT_NE(v0, 0.0f) << "V values should be non-zero after projection";
  // Verify q_norm and k_norm are applied (RMSNorm normalizes to ~1.0 magnitude)
  float q_val = q_proj.data<float>()[0];
  float k_val = cached_keys.data<float>()[0];
  EXPECT_GT(std::abs(q_val), 0.1f) << "q_norm should produce non-zero output";
  EXPECT_LT(std::abs(q_val), 2.0f) << "q_norm should normalize values";
  EXPECT_GT(std::abs(k_val), 0.1f) << "k_norm should produce non-zero output";
  EXPECT_LT(std::abs(k_val), 2.0f) << "k_norm should normalize values";
}

// GQA: Heads within same group share K/V
TEST_F(CrossAttentionTest, GroupedQueryAttention) {
  constexpr dim_t NUM_KV_HEADS = 2;
  constexpr dim_t HEADS_PER_GROUP = NUM_HEADS / NUM_KV_HEADS;

  MockModel model(NUM_HEADS, NUM_KV_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // GQA: After head replication, shape is [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));

  // Heads in same group share K/V
  for (dim_t group = 0; group < NUM_KV_HEADS; ++group) {
    dim_t first = group * HEADS_PER_GROUP;
    float k_group = get_4d(cached_keys, 0, first, 0, 0);
    float v_group = get_4d(cached_values, 0, first, 0, 0);
    for (dim_t h = first + 1; h < first + HEADS_PER_GROUP; ++h) {
      EXPECT_EQ(get_4d(cached_keys, 0, h, 0, 0), k_group);
      EXPECT_EQ(get_4d(cached_values, 0, h, 0, 0), v_group);
    }
  }
}

// Merged self+cross attention (T5Gemma2 style): self-attention layer also
// projects encoder memory through a separate `memory_kv` linear and concatenates
// the result onto the self-attention K/V before softmax.
TEST(MergedAttentionTest, ForwardMergedProducesOutput) {
  constexpr dim_t NUM_HEADS = 4, NUM_KV = 1, D_HEAD = 16;
  constexpr dim_t D_MODEL = NUM_HEADS * D_HEAD;
  constexpr dim_t QKV_ROWS = (NUM_HEADS + 2 * NUM_KV) * D_HEAD;
  constexpr dim_t KV_ROWS = 2 * NUM_KV * D_HEAD;

  class MergedMockModel : public models::Model {
  public:
    MergedMockModel() {
      register_variable("attn/linear_0/weight",
                        StorageView({QKV_ROWS, D_MODEL}, std::vector<float>(QKV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/linear_1/weight",
                        StorageView({D_MODEL, NUM_HEADS * D_HEAD},
                                    std::vector<float>(D_MODEL * NUM_HEADS * D_HEAD, 0.01f)));
      register_variable("attn/memory_kv/weight",
                        StorageView({KV_ROWS, D_MODEL}, std::vector<float>(KV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/q_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/k_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/num_heads_kv", StorageView(static_cast<int32_t>(NUM_KV)));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
    }
  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

  MergedMockModel model;
  layers::MultiHeadAttention attention(model, "attn", NUM_HEADS, /*self_attention=*/true);
  ASSERT_TRUE(attention.has_merged_encoder_attention());

  constexpr dim_t B = 1, Q_LEN = 1, MEM_LEN = 5;
  StorageView queries({B, Q_LEN, D_MODEL}, std::vector<float>(B * Q_LEN * D_MODEL, 1.0f));
  StorageView memory({B, MEM_LEN, D_MODEL}, std::vector<float>(B * MEM_LEN * D_MODEL, 1.0f));
  StorageView output(DataType::FLOAT32);
  StorageView self_k(DataType::FLOAT32), self_v(DataType::FLOAT32);
  StorageView mem_k(DataType::FLOAT32), mem_v(DataType::FLOAT32);

  attention.forward_merged(queries, &memory, nullptr, nullptr, output,
                           &self_k, &self_v, &mem_k, &mem_v, nullptr, nullptr, /*offset=*/0);

  EXPECT_EQ(output.shape(), (Shape{B, Q_LEN, D_MODEL}));
  EXPECT_EQ(mem_k.shape(), (Shape{B, NUM_HEADS, MEM_LEN, D_HEAD}));
  EXPECT_EQ(self_k.shape(), (Shape{B, NUM_HEADS, Q_LEN, D_HEAD}));
}

// MHA: Each head has independent K/V
TEST_F(CrossAttentionTest, StandardMultiHeadAttention) {
  MockModel model(NUM_HEADS, NUM_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // Shape: [batch, num_heads, time, d_head] - each head has own K/V
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), cached_keys.shape());
}
