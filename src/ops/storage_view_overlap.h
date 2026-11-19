#pragma once

#include <cstdint>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace ops {

    inline bool storage_views_overlap(const StorageView& a, dim_t a_bytes,
                                      const StorageView& b, dim_t b_bytes) {
      if (a_bytes <= 0 || b_bytes <= 0 || !a.buffer() || !b.buffer()
          || a.device() != b.device() || a.device_index() != b.device_index())
        return false;

      const auto a_begin = reinterpret_cast<std::uintptr_t>(a.buffer());
      const auto b_begin = reinterpret_cast<std::uintptr_t>(b.buffer());
      // Compare distances to avoid overflowing an end address.
      return a_begin <= b_begin
        ? b_begin - a_begin < static_cast<std::uintptr_t>(a_bytes)
        : a_begin - b_begin < static_cast<std::uintptr_t>(b_bytes);
    }

    inline bool storage_views_overlap(const StorageView& a, const StorageView& b) {
      // Include retained allocations even when a view has been cleared.
      return storage_views_overlap(a, a.reserved_memory(), b, b.reserved_memory());
    }

  }
}
