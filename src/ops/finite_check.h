#pragma once
#include <atomic>
#include <cmath>
#include "cpu/parallel.h"
namespace ctranslate2 { namespace ops { namespace {
bool all_finite(const float* data, dim_t size) {
  std::atomic<bool> invalid{false};
  cpu::parallel_for(0, size, cpu::GRAIN_SIZE * 8, [&](dim_t begin, dim_t end) {
    bool found_invalid = false;
    for (dim_t i = begin; i < end; ++i)
      found_invalid |= !std::isfinite(data[i]);
    if (found_invalid)
      invalid.store(true, std::memory_order_relaxed);
  });
  return !invalid.load(std::memory_order_relaxed);
}
}}}
