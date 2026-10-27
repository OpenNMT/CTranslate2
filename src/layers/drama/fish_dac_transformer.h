#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    // Configures the codec transformer depth, attention window, and normalization.
    struct FishDACTransformerConfig {
      dim_t heads = 16;
      dim_t layers = 8;
      dim_t window = 128;
      float epsilon = 1e-5f;
      float rope_theta = 10000.f;
    };

    // Refines reconstructed codebook latents before temporal upsampling.
    // Offline CPU float32 codec post-transformer. Input/output [1,C,T].
    // Direct-gamma pre-RMSNorm -> packed contiguous Q,K,V projections -> adjacent
    // full-head RoPE -> causal attention over [t-window+1,t] -> output projection
    // -> per-channel layer scale -> residual, then direct-gamma pre-RMSNorm ->
    // SwiGLU -> layer scale -> residual; final direct-gamma RMSNorm.
    // No Q/K normalization, attention output gate, biases, cache, or zero-centered
    // gamma. Scope is codec/quantizer/post_module, using original slash-mapped
    // wqkv/wo/w1/w2/w3 names. Dimensions derive from weights; heads/layers/window
    // are explicit configuration. This is not the complete waveform codec.
    class FishDACTransformer {
    public:
      FishDACTransformer(const models::Model& model, const std::string& scope,
                         FishDACTransformerConfig config = {});
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      // Groups the attention and feed-forward parameters of one codec transformer block.
      struct Block {
        const StorageView *attention_norm, *ffn_norm, *attention_scale, *ffn_scale;
        std::unique_ptr<Dense> qkv, out, gate, down, up;
      };

      void apply_rotary(StorageView& qkv) const;

      void apply_feed_forward(const Block& block, StorageView& hidden) const;

      void apply_attention(const Block& block, StorageView& hidden) const;

      void validate_input(const StorageView& input, const StorageView& output) const;

      FishDACTransformerConfig _config;
      dim_t _channels;
      const StorageView* _norm;
      std::vector<Block> _blocks;
      std::vector<const StorageView*> _parameters;
    };

  }
}
