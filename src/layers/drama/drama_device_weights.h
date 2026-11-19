#pragma once

#include "ctranslate2/storage_view.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "drama_options.h"

namespace ctranslate2 {
  namespace layers {

    struct DramaDeviceWeights {
      StorageView weight, scale, scale_inv, bias;

      size_t bytes() const {
        return weight.reserved_memory() + scale.reserved_memory() + scale_inv.reserved_memory()
               + bias.reserved_memory();
      }
    };

    // One registry per checkpoint execution setup. Weak entries let each replica
    // evict its own references without invalidating another replica's descriptors.
    class DramaDeviceWeightRegistry {
    public:
      template <typename Prepare>
      std::shared_ptr<const DramaDeviceWeights> get(const std::string& key, Prepare prepare) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto& entry = _weights[key];
        if (auto weights = entry.lock())
          return weights;
        auto weights = prepare();
        entry = weights;
        return weights;
      }

    private:
      std::mutex _mutex;
      std::unordered_map<std::string, std::weak_ptr<const DramaDeviceWeights>> _weights;
    };

    // Upload other immutable parameters (norms, convolutions, and biases) using
    // the same registry. The layer retains ownership beside its borrowed view.
    inline StorageView upload_drama_weight(
        const StorageView& source, const DramaExecutionOptions& options,
        std::vector<std::shared_ptr<const DramaDeviceWeights>>& owners) {
      if (!options.device_weights)
        return source.to(Device::CUDA);
      const auto key = "tensor:" + std::to_string(reinterpret_cast<uintptr_t>(source.buffer()))
                       + ":" + std::to_string(options.device_index);
      auto weights = options.device_weights->get(key, [&] {
        auto prepared = std::make_shared<DramaDeviceWeights>();
        prepared->weight = source.to(Device::CUDA);
        synchronize_stream(Device::CUDA);
        return prepared;
      });
      StorageView view;
      view.shallow_copy(const_cast<StorageView&>(weights->weight));
      owners.emplace_back(std::move(weights));
      return view;
    }

  }
}
