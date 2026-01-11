#include "test_utils.h"
#include "ctranslate2/layers/attention.h"

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
