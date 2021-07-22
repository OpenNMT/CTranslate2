#pragma once

#include <string>
#include <vector>

namespace ctranslate2 {

  struct ScoringResult {
    std::vector<std::string> tokens;
    std::vector<float> tokens_score;
  };

}
