#pragma once

#include "model.h"
#include "ctranslate2/replica_pool.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ctranslate2 {
  namespace models {

    // Loads Drama3 checkpoint weights and enforces their inference precision policy.
    // CPU checkpoint with optionally file-backed expert weights.
    class Drama3CheckpointStorage;
    struct Drama3StateSnapshot;

    // Shared, mutex-guarded prefix KV cache. Stores backbone+tower state after
    // prefill of a declared static prefix (first N prompt rows), keyed by the
    // raw int32 bytes of those rows. Mirrors layers::DecoderStateCache.
    class Drama3StateCache {
    public:
      Drama3StateCache();
      ~Drama3StateCache();
      const Drama3StateSnapshot* get(const std::vector<int32_t>& key) const;
      void save(std::vector<int32_t> key, Drama3StateSnapshot snapshot);
      // Counters are observational only, intended for tests and diagnostics.
      size_t hits() const;
      size_t misses() const;
      size_t entries() const;

    private:
      std::map<std::vector<int32_t>, Drama3StateSnapshot> _cache;
      mutable std::mutex _mutex;
      mutable size_t _hits = 0;
      mutable size_t _misses = 0;
    };

    class Drama3Model : public Model {
    public:
      // Select mapped expert storage explicitly; generic Model::load stays independent.
      static std::shared_ptr<const Model> load_checkpoint(const std::string& path,
                                                         ComputeType compute_type,
                                                         bool map_experts);

      // Shared across replicas; survives for the model's lifetime.
      Drama3StateCache& state_cache() const { return *_state_cache; }

    protected:
      bool is_quantizable(const std::string& name) const override;
      bool is_linear_weight(const std::string& name) const override;

      bool is_convertible(const StorageView&, const std::string&) const override {
        return false;
      }

      bool is_packable(const std::string&) const override {
        return false;
      }

      void register_variable(std::string name, StorageView variable) override;
      void initialize(ModelReader&) override;
      std::unique_ptr<Model> clone() const override;

      void prepare_weights(ComputeType& type, Device device, int device_index,
                           bool update_weight) override;
      StorageView load_variable(ModelReader& reader, const std::string& name,
                                Shape shape, DataType dtype, dim_t num_bytes,
                                std::istream& stream) override;

    private:
      std::vector<std::string> _fp8_weights;
      std::shared_ptr<Drama3CheckpointStorage> _checkpoint_storage;
      std::shared_ptr<Drama3StateCache> _state_cache = std::make_shared<Drama3StateCache>();
    };

    // Reports time spent in the native speech generation stages.
    // Wall-clock seconds from steady_clock. Total excludes waiting for the
    // replica queue and Python output copying; stage sums exclude bookkeeping.
    struct Drama3Timings {
      double prefill_seconds = 0;
      double backbone_decode_seconds = 0;
      double tower_decode_seconds = 0;
      double codec_seconds = 0;
      double total_native_seconds = 0;
    };

    // Returns generated audio, codebook tokens, completion status, and timings.
    struct Drama3Result {
      StorageView audio;
      StorageView codes;
      int sample_rate = 44100;
      std::string finish_reason;
      Drama3Timings timings;
    };

    struct Drama3Options {
      dim_t max_frames = 512;
      dim_t prefill_chunk_size = 16;
      float temperature = .8f;
      dim_t top_k = 30;
      float top_p = .8f;
      uint64_t seed = 0;
      bool greedy = false;
      // Length (in rows) of a static prompt prefix inside `prompt`. When
      // nonzero and cache_static_prompt is true, the runtime caches the model
      // state after this prefix and reuses it in future generations with the
      // same static prefix. 0 disables caching for this call.
      dim_t static_prompt_rows = 0;
      // Cache the model state after the static prompt and reuse it for future
      // generations using the same static prompt. Ignored when
      // static_prompt_rows == 0.
      bool cache_static_prompt = true;
    };

    // A replica owns its layers and mutable caches and runs on one pool worker.
    class Drama3Replica : public ModelReplica {
    public:
      static std::unique_ptr<Drama3Replica> create_from_model(const Model& model);
      ~Drama3Replica() override;
      // Execution settings can differ from the CPU checkpoint storage.
      Device device() const;
      int device_index() const;
      ComputeType compute_type() const;
      Drama3Result generate(const StorageView& prompt, const Drama3Options& options);

    private:
      struct Impl;
      explicit Drama3Replica(std::unique_ptr<Impl> impl);
      std::unique_ptr<Impl> _impl;
      friend class Drama3;
    };

    // Schedules independent speech requests on replicas sharing immutable CPU weights.
    class Drama3 : public ReplicaPool<Drama3Replica> {
    public:
      explicit Drama3(const std::string& model_path, size_t intra_threads = 8,
                      const std::string& compute_type = "default",
                      const std::string& device = "cpu", int device_index = 0,
                      size_t inter_threads = 1, long max_queued_batches = 0);
      // Synchronous compatibility entry point. Async and batch calls use the same pool.
      Drama3Result generate(const StorageView& prompt, dim_t max_frames = 512,
                            dim_t prefill_chunk_size = 16, float temperature = .8f,
                            dim_t top_k = 30, float top_p = .8f, uint64_t seed = 0,
                            bool greedy = false, dim_t static_prompt_rows = 0,
                            bool cache_static_prompt = true);
      std::future<Drama3Result> generate_async(const StorageView& prompt,
                                               const Drama3Options& options = {});
      // One independent job per prompt, with results in input order. This does not
      // combine sequences into a single tensor batch inside the inference kernels.
      std::vector<std::future<Drama3Result>>
      generate_batch(const std::vector<StorageView>& prompts, const Drama3Options& options = {});
    };

  }
}
