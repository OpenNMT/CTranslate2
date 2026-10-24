#pragma once

#include "ctranslate2/types.h"
#include "ctranslate2/devices.h"
#include <cstdint>
#include <memory>

namespace ctranslate2 {
  namespace layers {

    class DramaDeviceWeightRegistry;

    // Execution belongs to a replica, independently of checkpoint storage.
    struct DramaExecutionOptions {
      Device device = Device::CPU;
      int device_index = 0;
      ComputeType compute_type = ComputeType::FLOAT32;
      uint64_t expert_cache_bytes = 0;
      std::shared_ptr<DramaDeviceWeightRegistry> device_weights;
      bool resident_backbone = true;
      bool waveform = true;

      bool resident_gpu() const {
        return device == Device::CUDA && resident_backbone;
      }
    };

  }
}
