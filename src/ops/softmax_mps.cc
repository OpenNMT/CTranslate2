#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/primitives.h"
#include "mps/kernels.h"

#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>

namespace ctranslate2 {
  namespace ops {

    static bool use_reference_mps_softmax() {
      const char* all = std::getenv("CT2_MPS_REFERENCE_NORMS");
      const char* softmax = std::getenv("CT2_MPS_REFERENCE_SOFTMAX");
      return (all && all[0] != '\0' && all[0] != '0')
             || (softmax && softmax[0] != '\0' && softmax[0] != '0');
    }

    template <typename T>
    static void reference_softmax(const StorageView& input,
                                  const StorageView* lengths,
                                  StorageView& output,
                                  bool log_output) {
      StorageView input_cpu = input.to(Device::CPU);
      std::unique_ptr<StorageView> lengths_cpu;
      if (lengths)
        lengths_cpu = std::make_unique<StorageView>(lengths->to(Device::CPU));
      StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);
      const dim_t depth = input_cpu.dim(-1);
      const dim_t rows = input_cpu.size() / depth;
      const T* x = input_cpu.data<T>();
      const int32_t* row_lengths = lengths_cpu ? lengths_cpu->data<int32_t>() : nullptr;
      T* y = output_cpu.data<T>();
      for (dim_t row = 0; row < rows; ++row) {
        const dim_t valid = row_lengths
                            ? std::max<dim_t>(0, std::min<dim_t>(row_lengths[row], depth))
                            : depth;
        float maximum = -std::numeric_limits<float>::infinity();
        for (dim_t i = 0; i < valid; ++i)
          maximum = std::max(maximum, static_cast<float>(x[row * depth + i]));
        float sum = 0;
        for (dim_t i = 0; i < valid; ++i)
          sum += std::exp(static_cast<float>(x[row * depth + i]) - maximum);
        const float log_sum = std::log(sum);
        for (dim_t i = 0; i < valid; ++i) {
          const float value = static_cast<float>(x[row * depth + i]) - maximum;
          y[row * depth + i] = static_cast<T>(log_output ? value - log_sum
                                                         : std::exp(value - log_sum));
        }
        for (dim_t i = valid; i < depth; ++i)
          y[row * depth + i] = T(0);
      }
      output.copy_from(output_cpu);
    }

#define DECLARE_IMPL(T, DTYPE)                                          \
    template <>                                                         \
    void SoftMax::compute<Device::MPS, T>(const StorageView& input,     \
                                          const StorageView* lengths,   \
                                          StorageView& output) const {  \
      if (use_reference_mps_softmax()) {                                \
        reference_softmax<T>(input, lengths, output, _log);             \
        return;                                                         \
      }                                                                 \
      const dim_t depth = input.dim(-1);                                \
      const dim_t batch_size = input.size() / depth;                    \
      mps::softmax(DTYPE,                                               \
                   input.data<T>(),                                     \
                   lengths ? lengths->data<int32_t>() : nullptr,        \
                   output.data<T>(),                                    \
                   batch_size,                                          \
                   depth,                                               \
                   _log);                                               \
    }

    DECLARE_IMPL(float, DataType::FLOAT32)
    DECLARE_IMPL(float16_t, DataType::FLOAT16)
    DECLARE_IMPL(bfloat16_t, DataType::BFLOAT16)

#undef DECLARE_IMPL

  }
}

#endif
#endif
