#include "ctranslate2/ops/causal_depthwise_conv1d.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cpu/parallel.h"
#include "finite_check.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void CausalDepthwiseConv1D::compute(const StorageView& input, const StorageView& weight,
                                      const StorageView* bias, StorageView& history,
                                      StorageView& output) const {
      for (const StorageView* tensor :
           {&input, &weight, bias, static_cast<const StorageView*>(&history)}) {
        if (tensor) {
          const auto* data = tensor->data<T>();
          if (!all_finite(data, tensor->size()))
            throw std::invalid_argument(
                "CausalDepthwiseConv1D inputs and history must be finite");
        }
      }
      const dim_t batch = input.dim(0), steps = input.dim(1), channels = input.dim(2);
      const dim_t kernel = weight.dim(1), past = kernel - 1;
      output.resize_as(input);
      if (steps == 0)
        return;
      const auto* x = input.data<T>();
      const auto* w = weight.data<T>();
      const auto* b = bias ? bias->data<T>() : nullptr;
      auto* state = history.data<T>();
      auto* y = output.data<T>();
      // The tiled path amortizes its separate vector activation on wide, long chunks.
      if (kernel > 4 || channels < 1024 || (steps > 1 && steps < 64)) {
        const dim_t grain = input.size() <= cpu::GRAIN_SIZE / 8 / kernel ? batch * channels : 1;
        cpu::parallel_for(0, batch * channels, grain, [&](dim_t begin, dim_t end) {
          std::vector<float> ring(past);
          for (dim_t bc = begin; bc < end; ++bc) {
            const dim_t batch_index = bc / channels;
            const dim_t c = bc % channels;
            for (dim_t i = 0; i < past; ++i)
              ring[i] = state[bc * past + i];
            dim_t cursor = 0;
            for (dim_t t = 0; t < steps; ++t) {
              const dim_t offset = (batch_index * steps + t) * channels + c;
              float value = b ? b[c] : 0.f;
              for (dim_t k = 0; k < past; ++k)
                value += ring[(cursor + k) % past] * w[c * kernel + k];
              value += x[offset] * w[c * kernel + past];
              y[offset] = _apply_silu ? value / (1.f + std::exp(-value)) : value;
              if (past) {
                ring[cursor] = x[offset];
                cursor = (cursor + 1) % past;
              }
            }
            for (dim_t i = 0; i < past; ++i)
              state[bc * past + i] = ring[(cursor + i) % past];
          }
        });
        return;
      }
      constexpr dim_t channel_tile = 64;
      const dim_t tiles = (channels + channel_tile - 1) / channel_tile;
      const dim_t grain = input.size() <= cpu::GRAIN_SIZE ? batch * tiles : 1;
      cpu::parallel_for(0, batch * tiles, grain, [&](dim_t begin, dim_t end) {
        for (dim_t task = begin; task < end; ++task) {
          const dim_t n = task / tiles;
          const dim_t first = (task % tiles) * channel_tile;
          const dim_t last = std::min(first + channel_tile, channels);
          for (dim_t t = 0; t < steps; ++t) {
            auto* out = y + (n * steps + t) * channels;
            for (dim_t c = first; c < last; ++c)
              out[c] = b ? b[c] : 0.f;
            for (dim_t k = 0; k < kernel; ++k) {
              const dim_t source = t + k - past;
              if (source < 0) {
                for (dim_t c = first; c < last; ++c)
                  out[c] += state[(n * channels + c) * past + source + past] * w[c * kernel + k];
              } else {
                const auto* in = x + (n * steps + source) * channels;
                for (dim_t c = first; c < last; ++c)
                  out[c] += in[c] * w[c * kernel + k];
              }
            }
          }
          for (dim_t c = first; past && c < last; ++c) {
            auto* channel_history = state + (n * channels + c) * past;
            for (dim_t i = 0; i < past; ++i) {
              const dim_t source = steps + i;
              channel_history[i] = source < past ? channel_history[source]
                                                : x[(n * steps + source - past) * channels + c];
            }
          }
        }
      });
      if (_apply_silu)
        primitives<Device::CPU>::swish(y, y, output.size());
    }

    template void CausalDepthwiseConv1D::compute<Device::CPU, float>(
        const StorageView&, const StorageView&, const StorageView*, StorageView&, StorageView&) const;


  }
}
