#include "ctranslate2/ops/lstm_cell.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    // Parameters:
    //   x: input at this time step [batch, input_size]
    //   h_prev: previous hidden state [batch, hidden_size]
    //   c_prev: previous cell state [batch, hidden_size]
    //   w_ih: input-to-hidden weights [4 * hidden_size, input_size]
    //   w_hh: hidden-to-hidden weights [4 * hidden_size, hidden_size]
    //   b_ih: input-to-hidden bias [4 * hidden_size]
    //   b_hh: hidden-to-hidden bias [4 * hidden_size]
    //   h: output: new hidden state [batch, hidden_size]
    //   c: output: new cell state [batch, hidden_size]
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
