#pragma once

#include "drama_projection.h"
#include "ctranslate2/ops/moe_topk.h"
#include <list>

namespace ctranslate2 {
  namespace layers {

    // Routes each token through selected experts and combines their outputs.
    // CPU float32 activation baseline: [rows, hidden] -> [rows, hidden].
    // Computes only selected routed experts, plus a sigmoid-gated shared expert.
    // Model variables must outlive this layer and remain unchanged. Requires
    // unpacked /weight matrices: model loading must disable weight prepacking.
    // GPU INT8 experts optionally use a bounded cache backed by CPU weights.
    // No residual connection or normalization is applied.
    class SparseMoE : public Layer {
    public:
      SparseMoE(const models::Model& model, const std::string& scope, dim_t num_experts, dim_t k, const DramaExecutionOptions& options = {});
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, StorageView& output) const;
      size_t gpu_expert_cache_bytes() const;

      size_t gpu_expert_cache_misses() const {
        return _cache_misses;
      }

    private:
      void forward_individual_gpu(const StorageView& input, StorageView& output) const;

      void validate_input(const StorageView& input, const StorageView& output) const;

      // Applies a gated feed-forward network for one routed or shared expert.
      struct Expert {
        Expert(const models::Model& model, const std::string& scope,
               const DramaExecutionOptions& options);
        void operator()(const StorageView& input, StorageView& output) const;
        DramaProjection gate;
        DramaProjection up;
        DramaProjection down;
        int gpu_device_index;
      };

      void forward_grouped_gpu(const StorageView& input, StorageView& output) const;
      void forward_resident_gpu(const StorageView& input, StorageView& output) const;
      void cache_experts(const int32_t* selected) const;
      const Expert& cached_expert(dim_t id) const;
      size_t _cache_capacity = 0;
      size_t _cache_expert_bytes = 0;
      mutable size_t _cache_misses = 0;
      mutable std::list<dim_t> _cache_lru;
      mutable StorageView _gpu_experts;
      bool _resident_gpu = false;
      bool _grouped_gpu = false;
      DramaProjection _router;
      DramaProjection _shared_gate;
      Expert _shared;
      std::vector<std::unique_ptr<Expert>> _experts;
      ops::MoeTopK _topk;
      dim_t _input_size;
      dim_t _k;
      std::vector<const StorageView*> _parameters;
    };

  }
}
