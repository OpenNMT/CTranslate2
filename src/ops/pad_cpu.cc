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

 
    }

    template void Pad1D::compute<Device::CPU, float>(const StorageView&,
                                                    StorageView&) const;

  }
}
