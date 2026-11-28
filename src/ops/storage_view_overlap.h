#pragma once

#include <cstdint>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace ops {

    // Pure address-range test: it does not look at devices.
    inline bool byte_ranges_overlap(const void* a, dim_t a_bytes, const void* b, dim_t b_bytes) {
      if (a_bytes <= 0 || b_bytes <= 0 || !a || !b)
        return false;

      const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
      const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
      // Compare distances to avoid overflowing an end address.
      return a_begin <= b_begin
        ? b_begin - a_begin < static_cast<std::uintptr_t>(a_bytes)
        : a_begin - b_begin < static_cast<std::uintptr_t>(b_bytes);
    }

    inline bool storage_views_overlap(const StorageView& a, dim_t a_bytes,
                                      const StorageView& b, dim_t b_bytes) {
      if (a.device() != b.device() || a.device_index() != b.device_index())
        return false;
      return byte_ranges_overlap(a.buffer(), a_bytes, b.buffer(), b_bytes);
    }

    inline bool storage_views_overlap(const StorageView& a, const StorageView& b) {
      // Include retained allocations even when a view has been cleared.
      return storage_views_overlap(a, a.reserved_memory(), b, b.reserved_memory());
    }

    // True if `input` (by logical size) overlaps `output` (by its full allocation).
    inline bool input_aliases_output(const StorageView& input, const StorageView& output) {
      return &input == &output
        || storage_views_overlap(input, input.size() * input.item_size(),
                                 output, output.reserved_memory());
    }

  }
}
