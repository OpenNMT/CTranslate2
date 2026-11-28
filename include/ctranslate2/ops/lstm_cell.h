#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Single LSTM time step (same convention as torch.nn.LSTMCell).
    //
    //   gates = x @ w_ih^T + b_ih + h_prev @ w_hh^T + b_hh     [batch, 4 * hidden]
    //   i, f, g, o = chunks of gates (in this order)
    //   c = sigmoid(f) * c_prev + sigmoid(i) * tanh(g)
    //   h = sigmoid(o) * tanh(c)
    //
    // Shapes:
    //   x       [batch, input_size]
    //   h_prev  [batch, hidden_size]
    //   c_prev  [batch, hidden_size]
    //   w_ih    [4 * hidden_size, input_size]
    //   w_hh    [4 * hidden_size, hidden_size]
    //   b_ih    [4 * hidden_size]
    //   b_hh    [4 * hidden_size]
    //   h, c    [batch, hidden_size] (outputs)
    class LSTMCell : public Op {
    public:
      void operator()(const StorageView& x,
                      const StorageView& h_prev,
                      const StorageView& c_prev,
                      const StorageView& w_ih,
                      const StorageView& w_hh,
                      const StorageView& b_ih,
                      const StorageView& b_hh,
                      StorageView& h,
                      StorageView& c) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x,
                   const StorageView& h_prev,
                   const StorageView& c_prev,
                   const StorageView& w_ih,
                   const StorageView& w_hh,
                   const StorageView& b_ih,
                   const StorageView& b_hh,
                   StorageView& h,
                   StorageView& c) const;
    };

  }
}
