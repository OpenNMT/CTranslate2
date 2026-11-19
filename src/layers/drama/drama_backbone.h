#pragma once

#include "drama_block.h"

namespace ctranslate2 {
  namespace layers {

    // Configures the backbone blocks, context limit, and output token candidates.
    struct DramaBackboneConfig {
      std::vector<DramaBlockConfig> layers;
      float norm_epsilon = 1e-6f;
      dim_t max_context = 0;
      // Empty computes every LM row. Otherwise compute only these unique ids;
      // returned logits keep full vocabulary width, with zeros elsewhere.
      // Callers must restrict sampling to this exact candidate set.
      std::vector<int32_t> lm_candidates;
    };

    // Retains sequence state across all backbone blocks.
    struct DramaBackboneState {
      std::vector<DramaBlockState> layers;
      dim_t length = 0;
      dim_t capacity = 0;
    };

    // Runs the Drama3 backbone to produce hidden states and next-token logits.
    // CPU float32 batch-one backbone over already embedded [time,hidden] rows.
    // Outputs final normalized hidden rows and full-vocabulary float32 logits.
    // Model variables must outlive this container; loading disables prepacking.
    // On failure during layer execution, reset/discard caller state before retry.
    class DramaBackbone : public Layer {
    public:
      DramaBackbone(const models::Model& model, const std::string& scope,
                    DramaBackboneConfig config, const DramaExecutionOptions& options = {});
      DataType output_type() const override;
      dim_t output_size() const override;
      dim_t vocabulary_size() const;
      void initialize_state(DramaBackboneState& state, dim_t max_tokens = 0) const;
      void operator()(const StorageView& embedded, DramaBackboneState& state, StorageView& hidden,
                      StorageView& logits) const;

    private:
      std::vector<std::shared_ptr<const DramaDeviceWeights>> _shared_gpu_parameters;
      void validate_input(const StorageView& embedded, const DramaBackboneState& state,
                          const StorageView& hidden, const StorageView& logits) const;

      bool _resident_gpu = false;
      int _gpu_device_index = 0;
      StorageView _gpu_norm;
      DramaBackboneConfig _config;
      const StorageView& _norm;
      DramaProjection _lm_head;
      dim_t _vocabulary_size;
      std::vector<std::unique_ptr<DramaBlock>> _layers;
      std::vector<const StorageView*> _head_parameters;
    };

  }
}
