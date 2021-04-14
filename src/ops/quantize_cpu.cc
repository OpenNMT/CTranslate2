#include "ctranslate2/ops/quantize.h"

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Quantize::quantize<Device::CPU, int8_t>(const StorageView& input,
                                                 StorageView& output,
                                                 StorageView& scale) const {
      // INT8 quantization rescales based on the per batch absolute maximum.
      constexpr float int8_max = std::numeric_limits<int8_t>::max();
      constexpr float int8_min = std::numeric_limits<int8_t>::min();

      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);

      const auto* input_data = input.data<float>();
      auto* output_data = output.data<int8_t>();
      auto* scale_data = scale.data<float>();

      const float shift = (_shift_to_uint8 ? -int8_min : 0);

      #pragma omp parallel for
      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t offset = i * depth;
        const auto* row = input_data + offset;
        auto* qrow = output_data + offset;
        const auto amax = primitives<Device::CPU>::amax(row, depth);
        const auto row_scale = (amax != 0.f ? int8_max / amax : 1.f);
        cpu::unary_transform(row, qrow, depth,
                             [row_scale, shift](float v) {
                               return static_cast<int8_t>(v * row_scale + shift);
                             });
        scale_data[i] = row_scale;
      }
    }

    template<>
    void Quantize::quantize<Device::CPU, int16_t>(const StorageView& input,
                                                  StorageView& output,
                                                  StorageView& scale) const {
      // INT16 quantization simply rescales by a constant.
      constexpr float int16_max = std::numeric_limits<int16_t>::max();
      constexpr float int16_min = std::numeric_limits<int16_t>::min();

      const dim_t size = input.size();
      const auto* input_data = input.data<float>();
      auto* output_data = output.data<int16_t>();

      float scale_value = global_int16_scale;
      if (_int16_scale_type == ScaleType::PER_LAYER) {
        // The idea is to use 10 bits for the input so that the multiplication is 20
        // bits which gives 12 bits left for accumulation.
        const float amax = primitives<Device::CPU>::amax(input_data, size);
        scale_value = static_cast<float>(1 << 10) / amax;
      }

      scale = StorageView(scale_value);

      cpu::parallel_unary_transform(
        input_data, output_data, size, /*work_size=*/5,
        [scale_value](float v) {
          return static_cast<int16_t>(std::max(std::min(v * scale_value, int16_max), int16_min));
        });
    }

  }
}
