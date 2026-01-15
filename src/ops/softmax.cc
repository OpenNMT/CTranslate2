#include "ctranslate2/ops/softmax.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    LogSoftMax::LogSoftMax()
      : SoftMax(/*log=*/true) {
    }

    SoftMax::SoftMax(bool log)
      : _log(log) {
    }

    void SoftMax::operator()(StorageView& x) const {
      operator()(x, nullptr, nullptr, x);
    }

    void SoftMax::operator()(const StorageView& x, StorageView& y) const {
      operator()(x, nullptr, nullptr, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const {
      operator()(x, &lengths, nullptr, y);
    }

    void SoftMax::operator()(const StorageView& x,
                             const StorageView& lengths,
                             const StorageView& offsets,
                             StorageView& y) const {
      operator()(x, &lengths, &offsets, y);
    }

    void SoftMax::operator()(const StorageView& x,
                             const StorageView* lengths,
                             const StorageView* offsets,
                             StorageView& y) const {
      PROFILE(_log ? "LogSoftMax" : "SoftMax");
      y.resize_as(x);

      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      if (depth == 0)
        return;

      if (lengths && lengths->size() != batch_size)
        throw std::invalid_argument("Length mask has size "
                                    + std::to_string(lengths->size())
                                    + " which is different than the current batch size "
                                    + std::to_string(batch_size));

      if (offsets && offsets->size() != batch_size)
        throw std::invalid_argument("Offsets input has size "
                                    + std::to_string(offsets->size())
                                    + " which is different than the current batch size "
                                    + std::to_string(batch_size));

      DEVICE_AND_FLOAT_DISPATCH("SoftMax", x.device(), x.dtype(),
                                (compute<D, T>(x, lengths, offsets, y)));
    }

  }
}
