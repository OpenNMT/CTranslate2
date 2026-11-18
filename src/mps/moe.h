#pragma once
#include "ctranslate2/storage_view.h"
#include <array>
#include <vector>
namespace ctranslate2 {
  namespace mps {
    void moe_forward(
        const StorageView& input, const StorageView& ids,
        const StorageView& weights,
        const std::vector<std::array<const StorageView*, 3>>& experts,
        StorageView& output, bool force_gemv = false,
        bool force_small_tiles = false,
        const std::vector<std::array<const StorageView*, 3>>* scales = nullptr,
        bool force_generic_plan = false);
  }
} // namespace ctranslate2
