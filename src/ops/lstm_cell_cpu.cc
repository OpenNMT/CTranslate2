#include "ctranslate2/ops/lstm_cell.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LSTMCell::compute(const StorageView& x,
                           const StorageView& h_prev,
                           const StorageView& c_prev,
                           const StorageView& w_ih,
                           const StorageView& w_hh,
                           const StorageView& b_ih,
                           const StorageView& b_hh,
                           StorageView& h,
                           StorageView& c) const {    // output: new cell state [batch, hidden_size]
      // TODO: implement the LSTM cell computation.
    }

    template void LSTMCell::compute<Device::CPU, float>(const StorageView&,
                                                       const StorageView&,
                                                       const StorageView&,
                                                       const StorageView&,
                                                       const StorageView&,
                                                       const StorageView&,
                                                       const StorageView&,
                                                       StorageView&,
                                                       StorageView&) const;

  }
}
