#include "drama3_memory.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "env.h"

namespace ctranslate2 {

  uint64_t parse_drama3_memory_budget(const std::string& value, const char* name) {
    if (value.empty())
      return 0;
    try {
      size_t parsed = 0;
      const long double gb = std::stold(value, &parsed);
      const long double bytes = gb * 1000000000.L;
      if (parsed == value.size() && std::isfinite(bytes) && gb >= 0 && (gb == 0 || bytes >= 1)
          && bytes <= static_cast<long double>(std::numeric_limits<int64_t>::max()))
        return static_cast<uint64_t>(bytes);
    } catch (const std::exception&) {
    }
    throw std::invalid_argument(std::string(name)
                                + " must be 0 (disabled) or a positive "
                                  "number of decimal GB");
  }

  uint64_t drama3_gpu_expert_cache_budget() {
    return parse_drama3_memory_budget(read_string_from_env("CT2_DRAMA3_GPU_EXPERT_CACHE_GB"),
                                      "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
  }

}
