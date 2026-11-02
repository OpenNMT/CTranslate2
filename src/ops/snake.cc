#include "ctranslate2/ops/snake.h"

#include <stdexcept>

#include "device_dispatch.h"
#include "storage_view_overlap.h"

namespace ctranslate2 {
  namespace ops {

    void Snake::operator()(const StorageView& x, const StorageView& alpha, StorageView& y) const {
      PROFILE("Snake");
      for (const auto* tensor : {&x, &alpha, static_cast<const StorageView*>(&y)}) {
        if (tensor->device() != x.device() || tensor->device_index() != x.device_index()
            || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("Snake requires float32 tensors on the same device");
      }
      if (x.rank() != 3 || x.dim(0) <= 0 || x.dim(1) <= 0 || x.dim(2) < 0)
        throw std::invalid_argument("Snake expects [batch, channels, time]");
      const dim_t channels = x.dim(1);
      if (alpha.shape() != Shape{channels} && alpha.shape() != Shape{1, channels, 1})
        throw std::invalid_argument("Snake alpha must have shape [channels] or [1, channels, 1]");
      if (&y == &x || &y == &alpha
          || storage_views_overlap(x, x.size() * x.item_size(), y, y.reserved_memory())
          || storage_views_overlap(alpha, alpha.size() * alpha.item_size(), y, y.reserved_memory()))
        throw std::invalid_argument("Snake output must not alias inputs");
      const ScopedDeviceSetter setter(x.device(), x.device_index());
      DEVICE_DISPATCH(x.device(), (compute<D, float>(x, alpha, y)));
    }

  }
}
