#include "mps/ops_utils.h"

#include <cstdlib>
#include <cstdint>
#include <limits>

#include "mps/kernels.h"

namespace ctranslate2 {
  namespace mps {

    int activation_code(const ops::ActivationType* activation) {
      if (!activation)
        return -1;

      switch (*activation) {
      case ops::ActivationType::ReLU:
        return static_cast<int>(UnaryOp::RELU);
      case ops::ActivationType::GELUTanh:
        return static_cast<int>(UnaryOp::GELU_TANH);
      case ops::ActivationType::Swish:
        return static_cast<int>(UnaryOp::SWISH);
      case ops::ActivationType::GELU:
        return static_cast<int>(UnaryOp::GELU);
      case ops::ActivationType::GELUSigmoid:
        return static_cast<int>(UnaryOp::GELU_SIGMOID);
      case ops::ActivationType::Tanh:
        return static_cast<int>(UnaryOp::TANH);
      case ops::ActivationType::Sigmoid:
        return static_cast<int>(UnaryOp::SIGMOID);
      }

      return -1;
    }

    bool env_flag(const char* name, bool default_value) {
      const char* value = std::getenv(name);
      if (!value || value[0] == '\0')
        return default_value;
      return value[0] != '0';
    }

    dim_t copy_size(const StorageView& view, dim_t axis) {
      dim_t size = 1;
      for (dim_t i = axis; i < view.rank(); ++i)
        size *= view.dim(i);
      return size;
    }

    dim_t iteration_size(const StorageView& view, dim_t axis) {
      dim_t size = 1;
      for (dim_t i = 0; i < axis; ++i)
        size *= view.dim(i);
      return size;
    }

    bool ranges_overlap(const void* first,
                        size_t first_size,
                        const void* second,
                        size_t second_size) {
      if (first_size == 0 || second_size == 0)
        return false;

      const auto first_begin = reinterpret_cast<uintptr_t>(first);
      const auto second_begin = reinterpret_cast<uintptr_t>(second);
      if (first_begin > std::numeric_limits<uintptr_t>::max() - first_size
          || second_begin > std::numeric_limits<uintptr_t>::max() - second_size)
        return true;

      const auto first_end = first_begin + first_size;
      const auto second_end = second_begin + second_size;
      return first_begin < second_end && second_begin < first_end;
    }

  }
}
