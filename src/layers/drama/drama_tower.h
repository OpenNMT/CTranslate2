#pragma once

#include "gated_self_attention.h"
#include <random>

namespace ctranslate2 {
  namespace layers {

    // Controls temperature and candidate filtering during token sampling.
    struct DramaSamplingOptions {
      float temperature = .8f;
      float top_p = .8f;
      dim_t top_k = 30;
      // Request metadata: callers seed their persistent RNG once with this value.
      uint64_t seed = 0;
    };

    // Returns an index into logits. Implements Fish's exclusion of the token
    // crossing top_p; rank0 always survives. Uses the tested Torch CPU exponential
    // race and seed stream; cross-platform bitwise parity is not guaranteed.
    int32_t sample_drama_token(const std::vector<float>& logits,
                               const DramaSamplingOptions& options, std::mt19937& rng);

    // Configures audio codebooks, tower layers, and semantic token boundaries.
    struct DramaTowerConfig {
      GatedSelfAttentionConfig attention{0, 0, 0, 0};
      dim_t num_codebooks = 10;
      dim_t vocab_size = 4096;
      dim_t shared_layers = 4;
      dim_t vq_backbone_codebooks = 1;
      int32_t semantic_begin = 0, semantic_end = 0, eos_id = 0;
    };

    // Retains attention caches across audio frame generation steps.
    struct DramaTowerState {
      std::vector<GatedSelfAttentionCache> shared;
      std::vector<GatedSelfAttentionCache> residual;
      dim_t length = 0, capacity = 0;
    };

    // Predicts audio codebook tokens from backbone outputs and preceding audio frames.
    // Native batch-one CPU float32 tower with INT32 frames [T,1+num_codebooks].
    // Frame columns are [main token,code0,...]. Weights/model must outlive layer;
    // converted projection matrices remain unpacked (DramaProjection supports INT8).
    // Greedy decoding is a diagnostic, without repetition-aware resampling.
    // Reset/discard state on an exception after layer execution has started.
    class DramaTower : public Layer {
    public:
      DramaTower(const models::Model& model, const std::string& scope, DramaTowerConfig config, const DramaExecutionOptions& options = {});
      ~DramaTower() override;
      DataType output_type() const override;
      dim_t output_size() const override;
      void initialize_state(DramaTowerState& state, dim_t max_tokens) const;
      void embed_backbone(const StorageView& frames, const StorageView& text_embeddings,
                          StorageView& output) const;
      // Hidden row p is teacher-forced with NEXT frame p+1, not current frame p.
      void prefill(const StorageView& hidden, const StorageView& frames,
                   const StorageView& next_frames, DramaTowerState& state) const;
      void decode_greedy(const StorageView& logits, const StorageView& hidden,
                         const StorageView& previous_frame, DramaTowerState& state,
                         StorageView& next_frame) const;
      void decode_sample(const StorageView& logits, const StorageView& hidden,
                         const StorageView& previous_frame, DramaTowerState& state,
                         const std::vector<int32_t>& history_main_tokens,
                         const DramaSamplingOptions& options, std::mt19937& rng,
                         StorageView& next_frame) const;

    private:
      void decode_residual_codes(const StorageView& hidden, const StorageView& previous_frame,
                                 StorageView& frame, DramaTowerState& state,
                                 const DramaSamplingOptions* options, std::mt19937* rng) const;

      int32_t select_main_token(const StorageView& logits,
                                const std::vector<int32_t>* history_main_tokens,
                                const DramaSamplingOptions* options, std::mt19937* rng) const;

      // Implements one gated-attention and feed-forward block in the audio tower.
      struct TowerLayer;
      void validate_frames(const StorageView& frames, dim_t rows) const;
      void validate_hidden(const StorageView& hidden) const;
      void validate_state(const DramaTowerState& state, dim_t rows) const;
      void validate_state_aliases(const DramaTowerState& state,
                                  const std::vector<const StorageView*>& tensors) const;
      StorageView embedding_sum(const StorageView& frames, dim_t first, dim_t end) const;
      void run_shared(const StorageView& hidden, const StorageView& frames,
                      const StorageView& next_frames, DramaTowerState& state,
                      StorageView& output) const;
      bool is_semantic(int32_t token) const;
      void decode_impl(const StorageView& logits, const StorageView& hidden,
                       const StorageView& previous_frame, DramaTowerState& state,
                       StorageView& next_frame, const std::vector<int32_t>* history_main_tokens,
                       const DramaSamplingOptions* options, std::mt19937* rng) const;
      bool _resident_gpu = false;
      int _gpu_device_index = 0;
      DramaTowerConfig _config;
      const StorageView& _embeddings;
      Embeddings _embedding_lookup;
      std::vector<std::unique_ptr<TowerLayer>> _shared;
      std::vector<std::unique_ptr<TowerLayer>> _residual;
      std::vector<std::unique_ptr<DramaProjection>> _heads;
    };

  }
}
