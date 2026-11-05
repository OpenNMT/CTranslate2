#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    // Changes the time resolution of codec latents before waveform decoding.
    // Offline CPU float32 latent resampling around Fish DAC's quantizers.
    // Each block: transpose (or strided) k2/s2 -> causal ConvNeXt. Input/output [1,C,T].
    // Downsampling requires even lengths at every stage. Parameters are
    // materialized float32, held by the model for layer lifetime.
    class FishDACLatentResampler {
    public:
      FishDACLatentResampler(const models::Model& model, const std::string& scope,
                             dim_t block_count = 2, bool downsample = false);
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      // Combines temporal resampling with a causal ConvNeXt block.
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

      bool _downsample;
      dim_t _input_channels = 0;
      std::vector<std::unique_ptr<Block>> _blocks;
      std::vector<const StorageView*> _parameters;
    };

    using FishDACLatentUpsampler = FishDACLatentResampler;

  }
}
