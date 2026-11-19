#include "ctranslate2/ops/conv_transpose1d.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "cpu/parallel.h"
#include "cpu/backend.h"
#include "ctranslate2/ops/gemm.h"

namespace ctranslate2 {
  namespace ops {

    namespace {

      dim_t checked_product(dim_t a, dim_t b) {
        if (a && b > std::numeric_limits<dim_t>::max() / a)
          throw std::invalid_argument("ConvTranspose1D dimensions overflow");
        return a * b;
      }

      dim_t checked_sum(dim_t a, dim_t b) {
        if (b > std::numeric_limits<dim_t>::max() - a)
          throw std::invalid_argument("ConvTranspose1D dimensions overflow");
        return a + b;
      }


    }

    bool ConvTranspose1D::compute_gemm(const StorageView& input, const StorageView& weight,
                                       const StorageView* bias, StorageView& output) const {
      const dim_t batch = input.dim(0), cin = input.dim(1), time = input.dim(2);
      const dim_t cout_group = weight.dim(1), kernel = weight.dim(2);
      const dim_t cout = output.dim(1), length = output.dim(2), cin_group = cin / _groups;
      const auto* x = input.data<float>();
      const auto* w = weight.data<float>();
      const auto* b = bias ? bias->data<float>() : nullptr;
      auto* y = output.data<float>();
      // W is already [Cin/group, Cout/group * kernel] in each group.
      // Multiplying W^T by X produces the overlapping output contributions;
      // col2im below adds those contributions without reversing the kernel.
      // Bound both the GEMM result and a possible input-tile copy to 16 MiB.
      constexpr dim_t workspace_elements = 16 * 1024 * 1024 / sizeof(float);
      const dim_t contribution_rows = checked_product(cout_group, kernel);
      const dim_t tile_rows = checked_sum(cin_group, contribution_rows);
      if (cin_group >= 8 && time >= 2 && tile_rows <= workspace_elements
          && cpu::has_gemm_backend(ComputeType::FLOAT32)) {
        const dim_t tile_size = std::min(time, workspace_elements / tile_rows);
        StorageView contributions, input_tile;
        const Gemm gemm(1.f, 0.f, true, false);
        cpu::parallel_for(0, batch * cout, 1, [&](dim_t begin, dim_t end) {
          for (dim_t row = begin; row < end; ++row)
            std::fill(y + row * length, y + (row + 1) * length, b ? b[row % cout] : 0.f);
        });
        for (dim_t n = 0; n < batch; ++n) {
          for (dim_t group = 0; group < _groups; ++group) {
            StorageView weight_matrix(
                {cin_group, contribution_rows},
                const_cast<float*>(w + group * cin_group * contribution_rows));
            const auto* group_input = x + (n * cin + group * cin_group) * time;
            for (dim_t start = 0; start < time; start += tile_size) {
              const dim_t count = std::min(tile_size, time - start);
              StorageView input_matrix;
              if (count == time) {
                input_matrix.view(const_cast<float*>(group_input), {cin_group, count});
              } else {
                input_tile.resize({cin_group, count});
                for (dim_t ic = 0; ic < cin_group; ++ic)
                  std::copy_n(group_input + ic * time + start, count,
                              input_tile.data<float>() + ic * count);
                input_matrix.view(input_tile.data<float>(), {cin_group, count});
              }
              gemm(weight_matrix, input_matrix, contributions);
              const auto* values = contributions.data<float>();
              cpu::parallel_for(0, cout_group, 1, [&](dim_t begin, dim_t end) {
                for (dim_t oc = begin; oc < end; ++oc) {
                  auto* out = y + (n * cout + group * cout_group + oc) * length;
                  for (dim_t k = 0; k < kernel; ++k) {
                    const auto* column = values + (oc * kernel + k) * count;
                    for (dim_t t = 0; t < count; ++t) {
                      const dim_t pos = (start + t) * _stride - _padding + k * _dilation;
                      if (pos >= 0 && pos < length)
                        out[pos] += column[t];
                    }
                  }
                }
              });
            }
          }
        }
        return true;
      }
      return false;
    }

    template <Device D, typename T>
    void ConvTranspose1D::compute(const StorageView& input, const StorageView& weight,
                                  const StorageView* bias, StorageView& output) const {
      const dim_t batch = input.dim(0), cin = input.dim(1), time = input.dim(2);
      const dim_t cout_group = weight.dim(1), kernel = weight.dim(2);
      const dim_t cout = output.dim(1), length = output.dim(2);
      const auto* x = input.data<T>();
      const auto* w = weight.data<T>();
      const auto* b = bias ? bias->data<T>() : nullptr;
      auto* y = output.data<T>();
      const dim_t cin_group = cin / _groups;
      if (compute_gemm(input, weight, bias, output))
        return;
      // Each worker owns one output channel; all scatter additions are local.
      cpu::parallel_for(0, batch * cout, 1, [&](dim_t begin, dim_t end) {
        for (dim_t row = begin; row < end; ++row) {
          const dim_t n = row / cout, oc = row % cout;
          const dim_t group = oc / cout_group, local_oc = oc % cout_group;
          auto* out = y + row * length;
          std::fill(out, out + length, b ? b[oc] : 0.f);
          for (dim_t local_ic = 0; local_ic < cin_group; ++local_ic) {
            const auto ic = group * cin_group + local_ic;
            const auto* in = x + (n * cin + ic) * time;
            const auto* filter = w + (ic * cout_group + local_oc) * kernel;
            for (dim_t t = 0; t < time; ++t) {
              for (dim_t k = 0; k < kernel; ++k) {
                const dim_t pos = t * _stride - _padding + k * _dilation;
                if (pos >= 0 && pos < length)
                  out[pos] += in[t] * filter[k];
              }
            }
          }
        }
      });
    }

    template void ConvTranspose1D::compute<Device::CPU, float>(
        const StorageView&, const StorageView&, const StorageView*, StorageView&) const;


  }
}
