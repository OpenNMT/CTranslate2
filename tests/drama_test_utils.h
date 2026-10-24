#pragma once

#include "test_utils.h"
#include "layers/drama/drama_block.h"

namespace drama_test {

  // Identical mixer weights used by the block and backbone composition tests.
  template <typename Put> void add_mixer(Put put, bool attention) {
    if (attention) {
      put("self_attn/q_proj/weight", {4, 2}, {1, 0, 0, 1, .2f, 0, 0, -.3f});
      put("self_attn/k_proj/weight", {2, 2}, {1, 0, 0, 1});
      put("self_attn/v_proj/weight", {2, 2}, {.3f, 0, 0, .4f});
      put("self_attn/o_proj/weight", {2, 2}, {1, 0, 0, 1});
      put("self_attn/q_norm/weight", {2}, {0, 0});
      put("self_attn/k_norm/weight", {2}, {0, 0});
    } else {
      put("linear_attn/in_proj_qkv/weight", {3, 2}, {.4f, 0, .3f, 0, 0, .5f});
      put("linear_attn/in_proj_z/weight", {1, 2}, {.5f, -.1f});
      put("linear_attn/in_proj_a/weight", {1, 2}, {0, 0});
      put("linear_attn/in_proj_b/weight", {1, 2}, {0, 0});
      put("linear_attn/out_proj/weight", {2, 1}, {.5f, -.2f});
      put("linear_attn/A_log", {1}, {0});
      put("linear_attn/dt_bias", {1}, {0});
      put("linear_attn/norm/weight", {1}, {1});
      put("linear_attn/conv1d/weight", {3, 1, 2}, {.5f, .5f, .5f, .5f, .5f, .5f});
    }
  }

  inline layers::DramaBlockConfig block_config(bool attention, dim_t num_experts = 2) {
    layers::DramaBlockConfig cfg;
    cfg.kind =
        attention ? layers::DramaMixerKind::FullAttention : layers::DramaMixerKind::LinearAttention;
    cfg.norm_epsilon = 1e-5f;
    cfg.num_experts = num_experts;
    cfg.top_k = 1;
    cfg.key_heads = cfg.value_heads = cfg.key_dim = cfg.value_dim = 1;
    cfg.attention = {1, 1, 2, 2, 100.f, 1e-5f, false, true};
    return cfg;
  }

  inline StorageView inputs() {
    return StorageView({3, 2}, std::vector<float>{1, 2, -.5f, 1, 2, -.25f});
  }

}
