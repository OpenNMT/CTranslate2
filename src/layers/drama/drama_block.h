#pragma once

#include "gated_self_attention.h"
#include "gated_delta.h"
#include "moe.h"

namespace ctranslate2 {
  namespace layers {

    enum class DramaMixerKind { LinearAttention, FullAttention };

    // Selects the attention mixer and expert routing dimensions for one block.
    struct DramaBlockConfig {
      DramaMixerKind kind = DramaMixerKind::LinearAttention;
      float norm_epsilon = 1e-6f;
      dim_t num_experts = 0;
      dim_t top_k = 0;
      dim_t key_heads = 0, value_heads = 0, key_dim = 0, value_dim = 0;
      GatedSelfAttentionConfig attention{0, 0, 0, 0};
    };

    // Retains recurrent or attention state for one backbone block.
    struct DramaBlockState {
      StorageView recurrent;
      StorageView history;
      GatedSelfAttentionCache attention;
    };

    // Combines an attention mixer and sparse experts with normalization and residuals.
    // CPU float32 batch-one [time,hidden] backbone block. Model variables must
    // outlive this layer and remain unchanged; weight prepacking is disabled.
    // Each call advances caller-owned sequence state. On runtime failure after
    // mixer execution, discard/reset that state before retrying the sequence.
    class DramaBlock : public Layer {
    public:
      DramaBlock(const models::Model& model, const std::string& scope, DramaBlockConfig config, const DramaExecutionOptions& options = {});
      DataType output_type() const override;
      dim_t output_size() const override;
      void initialize_state(DramaBlockState& state, dim_t max_tokens = 0) const;
      void operator()(const StorageView& input, DramaBlockState& state, StorageView& output) const;

    private:
      std::vector<std::shared_ptr<const DramaDeviceWeights>> _shared_gpu_parameters;
      void validate_input(const StorageView& input, const DramaBlockState& state,
                          const StorageView& output) const;

      void forward_gpu(const StorageView& input, DramaBlockState& state, StorageView& output) const;

      bool _resident_gpu = false;
      int _gpu_device_index = 0;
      StorageView _gpu_input_norm, _gpu_post_norm;
      DramaBlockConfig _config;
      const StorageView& _input_norm;
      const StorageView& _post_norm;
      SparseMoE _moe;
      std::unique_ptr<GatedDeltaNet> _linear;
      std::unique_ptr<GatedSelfAttention> _attention;
    };

  }
}
