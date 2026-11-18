#pragma once
#include "ctranslate2/layers/common.h"
#include <array>

namespace ctranslate2 {
  namespace layers {
    // Standard decoder SwiGLU MoE. Routing uses FP32 logits; expert projections
    // retain the model's compute type. CPU provides the grouped reference path.
    class MixtureOfExperts : public Layer {
    public:
      MixtureOfExperts(const models::Model& model, const std::string& scope);
      ~MixtureOfExperts();
      void operator()(const StorageView& input, StorageView& output) const;
      DataType output_type() const override { return _dtype; }
      dim_t output_size() const override { return _hidden; }

    private:
      struct Expert;
      const ops::ActivationType _activation = ops::ActivationType::Swish;
      const DataType _dtype;
      const dim_t _hidden, _topk;
      const bool _normalize_topk;
      const ops::ActivationType _sigmoid = ops::ActivationType::Sigmoid;
      std::unique_ptr<Expert> _shared_expert;
      std::unique_ptr<Dense> _shared_gate;
      void add_shared(const StorageView& input, StorageView& output) const;
      StorageView _router;
      std::vector<std::unique_ptr<Expert>> _experts;
      std::vector<std::array<const StorageView*, 3>> _metal_weights, _metal_scales;
    };
  } // namespace layers
} // namespace ctranslate2
