#include "ctranslate2/ops/pad.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Pad1D::compute(const StorageView& input, StorageView& output) const {
      // Reflect padding: the padded values mirror the input around its edges,
      // WITHOUT repeating the edge element itself (the edge acts as the mirror).
      //
      //   input = [1, 2, 3, 4], left = 2, right = 2
      //
      //   index:   -2 -1 | 0  1  2  3 | 4  5
      //   output: [ 3  2 | 1  2  3  4 | 3  2 ]
      //             ^^^^              ^^^^
      //             left              right
      //
      // For an output position p in [0, left + n + right) and input length n,
      // let x = p - left (the position relative to the input):
      //   x < 0       -> src index = -x             (mirror around index 0)
      //   x >= n      -> src index = 2*(n - 1) - x  (mirror around index n-1)
      //   otherwise   -> src index = x
      //
      // Reflect requires left < n and right < n, so a single mirror is
      // always enough to land back inside the input.
      // (Edge replication, which repeats the edge value, is a different mode.)

      Shape out_shape = input.shape();
      out_shape.back() += _left + _right;
      output.resize(std::move(out_shape));

/*
      const auto* src = input.data<T>();
      auto* dst = output.data<T>();

       cpu::parallel_for(0, outer_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          for (dim_t j = 0; j < inner_size; ++j) {
            float sum = 0.f;
            for (dim_t k = 0; k < axis_size; ++k) {
              sum += src[i * axis_size * inner_size + k * inner_size + j];
            }
            dst[i * inner_size + j] = sum;
            if (!get_sum)
              dst[i * inner_size + j] /= float(axis_size);
          }
        }
      });
    }
*/
    }

    template void Pad1D::compute<Device::CPU, float>(const StorageView&,
                                                    StorageView&) const;

  }
}
