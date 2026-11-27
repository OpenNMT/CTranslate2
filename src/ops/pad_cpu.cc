#include "ctranslate2/ops/pad.h"

#include "cpu/parallel.h"

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

      const auto* src = input.data<T>();
      auto* dst = output.data<T>();

      const dim_t n = input.dim(-1);
      const dim_t out_len = n + _left + _right;
      const dim_t outer = input.size() / n;
      const dim_t grain_size = cpu::get_minimum_batch_copies_per_thread<T>(out_len);

      // Loop based on the output
      cpu::parallel_for(0, outer, grain_size, [&](dim_t begin, dim_t end) {
        for (dim_t r = begin; r < end; ++r) {
          const T* in = src + r * n;
          T* out = dst + r * out_len;
          for (dim_t p = 0; p < out_len; ++p) {
            if (p < _left) {
              out[p] = in[_left - p];
            } else if (p >= _left + n) {
              out[p] = in[2 * (n - 1) - (p - _left)];
            } else {
              // Middle: copy the input.
              out[p] = in[p - _left];
            }
          }
        }
      });
    }

    template void Pad1D::compute<Device::CPU, float>(const StorageView&,
                                                    StorageView&) const;

  }
}
