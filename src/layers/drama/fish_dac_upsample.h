#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    // Expands the time resolution of codec latents before waveform decoding.
    // Offline CPU float32 latent upsampling after Fish DAC's post transformer.
    // Each block: transpose k2/s2 -> causal ConvNeXt. Input/output [1,C,T].
    // Parameters are materialized float32, held by the model for layer lifetime.
    class FishDACLatentUpsampler {
    public:
      FishDACLatentUpsampler(const models::Model& model, const std::string& scope,
                             dim_t block_count = 2);
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      // Combines temporal upsampling with a causal ConvNeXt block.
      struct Block {
        Block(const models::Model& model, const std::string& scope);
        Dense pw1, pw2;
        const StorageView* transpose_weight;
        const StorageView* transpose_bias;
        const StorageView* depthwise_weight;
        const StorageView* depthwise_bias;
        const StorageView* norm_weight;
        const StorageView* norm_bias;
        const StorageView* gamma;
        dim_t channels;
      };

      void forward_block(const Block& block, const StorageView& input, StorageView& output) const;

      dim_t _input_channels = 0;
      std::vector<std::unique_ptr<Block>> _blocks;
      std::vector<const StorageView*> _parameters;
    };

  }
}
