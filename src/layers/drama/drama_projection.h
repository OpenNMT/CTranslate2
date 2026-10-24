#pragma once

#include "ctranslate2/layers/common.h"
#include "drama_options.h"
#include "drama_device_weights.h"

namespace ctranslate2 {
  namespace layers {

    // Replica-owned projection with shared immutable prepared GPU weights.
    class DramaProjection : public Layer {
    public:
      DramaProjection(const models::Model& model, const std::string& scope,
                      const DramaExecutionOptions& options = {});
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, StorageView& output,
                      const StorageView* residual = nullptr) const;
      void select_weights(const StorageView* index, const StorageView* extra_bias = nullptr);

      struct DeviceWeights {
        const StorageView& weight;
        const StorageView& scale;
      };

      // The expert cache controls residency through this interface. Descriptors
      // remain valid until release_device_weights or select_weights is called.
      DeviceWeights device_weights() const;
      void release_device_weights() const;
      size_t device_weight_bytes() const;
      size_t estimated_device_weight_bytes() const;

    private:
      void prepare_gpu_weight(const StorageView& weight, const StorageView* scale) const;
      void forward_gpu(const StorageView& input, const StorageView& weight,
                       const StorageView* bias, const StorageView* scale,
                       const StorageView* residual, StorageView& output) const;
      const StorageView& _weight;
      const StorageView* _bias;
      const StorageView* _qscale;
      const StorageView* _fp8_scale_inv;
      std::unique_ptr<Dense> _cpu;
      StorageView _partial_weight, _partial_bias, _partial_qscale;
      const int _gpu_device_index;
      const ComputeType _gpu_compute_type;
      const std::shared_ptr<DramaDeviceWeightRegistry> _device_weights;
      const std::string _device_weight_key;
      std::string _selection_key;
      bool _share_selected_weights = true;
      mutable std::shared_ptr<const DramaDeviceWeights> _gpu_weights;
      const ops::Dequantize _dequantize_op;
    };

  }
}
