#pragma once

#include "drama_projection.h"
#include <unordered_map>

namespace ctranslate2 {
  namespace layers {

    // Converts upsampled codec latents into audio waveform samples.
    // Offline waveform portion of Fish DAC. Input is the already upsampled
    // latent [1,C,T]; codebooks/post-transformer/ConvNeXt precede this layer.
    // CPU float32 weights, materialized weight normalization, no packing.
    class FishDACWaveformDecoder {
    public:
      FishDACWaveformDecoder(const models::Model& model, const std::string& scope,
                             std::vector<dim_t> strides = {8, 8, 4, 2},
                             const DramaExecutionOptions& options = {});
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      void forward_cpu(const StorageView& input, StorageView& output) const;

      void validate_input(const StorageView& input, const StorageView& output) const;

      // References the weights and optional bias of a waveform convolution.
      struct Convolution {
        const StorageView* weight;
        const StorageView* bias;
      };

      // Groups the activations and convolutions of a waveform residual unit.
      struct Residual {
        const StorageView* alpha0;
        Convolution conv0;
        const StorageView* alpha1;
        Convolution conv1;
      };

      // Groups one waveform upsampling stage and its residual units.
      struct Upsample {
        const StorageView* alpha;
        Convolution transpose;
        std::vector<Residual> residuals;
        dim_t stride;
      };

      static void causal(const StorageView& input, const Convolution& conv, dim_t dilation,
                         StorageView& output);
      void forward_gpu(const StorageView&, StorageView&) const;
      std::vector<std::shared_ptr<const DramaDeviceWeights>> _shared_gpu_parameters;
      int _gpu_device_index = -1;
      std::unordered_map<const StorageView*, StorageView> _gpu_parameters;
      Convolution _initial, _final;
      const StorageView* _final_alpha;
      std::vector<Upsample> _blocks;
      std::vector<const StorageView*> _parameters;
    };

  }
}
