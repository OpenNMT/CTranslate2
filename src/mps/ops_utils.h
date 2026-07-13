#pragma once

#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include <memory>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace mps {

    inline StorageView to_cpu(const StorageView& view) {
      return view.to(Device::CPU);
    }

    inline StorageView to_cpu_float32(const StorageView& view) {
      StorageView cpu = to_cpu(view);
      if (cpu.dtype() != DataType::FLOAT32)
        cpu = cpu.to(DataType::FLOAT32);
      return cpu;
    }

    inline std::unique_ptr<StorageView> optional_to_cpu(const StorageView* view) {
      if (!view)
        return nullptr;
      return std::make_unique<StorageView>(to_cpu(*view));
    }

    inline std::unique_ptr<StorageView> optional_to_cpu_float32(const StorageView* view) {
      if (!view)
        return nullptr;
      return std::make_unique<StorageView>(to_cpu_float32(*view));
    }

    inline StorageView cpu_float32_output_like(const StorageView& output) {
      return StorageView(output.shape(), DataType::FLOAT32, Device::CPU);
    }

    inline void copy_cpu_float32_to_output(const StorageView& cpu_float32,
                                           StorageView& output) {
      StorageView converted = cpu_float32;
      if (output.dtype() != DataType::FLOAT32)
        converted = cpu_float32.to(output.dtype());
      output.copy_from(converted);
    }

  }
}

#endif
#endif
