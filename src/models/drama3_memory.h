#pragma once

#include <cstdint>
#include <string>

namespace ctranslate2 {

  // Zero disables the GPU expert cache budget. Values use decimal GB, not GiB.
  uint64_t parse_drama3_memory_budget(const std::string& value,
                                      const char* name = "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
  uint64_t drama3_gpu_expert_cache_budget();

}
