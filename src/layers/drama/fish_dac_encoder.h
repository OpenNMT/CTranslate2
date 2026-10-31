#pragma once

#include "fish_dac_transformer.h"
#include "fish_dac_upsample.h"

namespace ctranslate2 {
  namespace layers {

    // Offline CPU reference encoder. Input mono FLOAT32 PCM [samples] at 44100 Hz;
    // output INT32 [frames,10]. File decoding and resampling belong to the caller.
    class FishDACReferenceEncoder {
    public:
      explicit FishDACReferenceEncoder(const models::Model& model);
      StorageView operator()(const StorageView& waveform) const;

    private:
      const StorageView& parameter(const std::string& name) const;
      void convolution(const StorageView& input, const std::string& name,
                       StorageView& output, dim_t stride = 1, dim_t dilation = 1) const;
      void snake(const StorageView& input, const std::string& name, StorageView& output) const;
      const models::Model& _model;
      FishDACTransformer _encoder_transformer;
      FishDACTransformer _pre_transformer;
      FishDACLatentResampler _downsample;
    };

  }
}
