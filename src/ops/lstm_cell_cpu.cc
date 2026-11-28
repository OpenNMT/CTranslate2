#include "ctranslate2/ops/lstm_cell.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LSTMCell::compute(const StorageView& x,
                           const StorageView& h_prev, // previous hidden state [batch, hidden_size]
                           const StorageView& c_prev, // previous cell state [batch, hidden_size]
                           const StorageView& w_ih, // input-to-hidden weights [4 * hidden_size, input_size]
                           const StorageView& w_hh, // hidden-to-hidden weights [4 * hidden_size, hidden_size]
                           const StorageView& b_ih, // input-to-hidden bias [4 * hidden_size]
                           const StorageView& b_hh, // hidden-to-hidden bias [4 * hidden_size]
                           StorageView& h, // output: new hidden state [batch, hidden_size]
                           StorageView& c) const { // output: new cell state [batch, hidden_size]
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
