#pragma once

#include "fish_dac_codebooks.h"
#include "fish_dac_transformer.h"
#include "fish_dac_upsample.h"
#include "fish_dac.h"

namespace ctranslate2 {
  namespace layers {

    // Decodes discrete audio codebook tokens into a complete waveform.
    // Complete offline decoder-only graph, using independently converted official
    // weights. Output CPU float32 [1,1,2048*T] at 44100 Hz; no delay trimming.
    class FishDACDecoder {
    public:
      explicit FishDACDecoder(const models::Model& model, const std::string& scope = "codec",
                              const DramaExecutionOptions& options = {})
        : _books(model, scope + "/quantizer")
        , _transformer(model, scope + "/quantizer/post_module")
        , _upsample(model, scope + "/quantizer/upsample")
        , _waveform(model, scope + "/decoder", {8, 8, 4, 2}, options) {
      }

      void operator()(const StorageView& codes, StorageView& waveform) const {
        StorageView projected, transformed, latent;
        _books(codes, projected);
        _transformer(projected, transformed);
        _upsample(transformed, latent);
        _waveform(latent, waveform);
      }

    private:
      FishDACCodebooks _books;
      FishDACTransformer _transformer;
      FishDACLatentUpsampler _upsample;
      FishDACWaveformDecoder _waveform;
    };

  }
}
