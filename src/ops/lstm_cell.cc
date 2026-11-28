#include "ctranslate2/ops/lstm_cell.h"

#include <stdexcept>

namespace ctranslate2 {
  namespace ops {

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
    void LSTMCell::operator()(const StorageView& x,
                              const StorageView& h_prev,
                              const StorageView& c_prev,
                              const StorageView& w_ih,
                              const StorageView& w_hh,
                              const StorageView& b_ih,
                              const StorageView& b_hh,
                              StorageView& h,
                              StorageView& c) const {

      // TODO: check parameters (dtype, shapes, aliasing).
      if (x.device() != h_prev.device() || x.device() != c_prev.device()
          || x.device() != w_ih.device() || x.device() != w_hh.device()
          || x.device() != b_ih.device() || x.device() != b_hh.device()
          || x.device() != h.device() || x.device() != c.device())
        throw std::invalid_argument("LSTMCell requires all tensors on the same device");

      if (x.device_index() != h_prev.device_index() || x.device_index() != c_prev.device_index()
          || x.device_index() != w_ih.device_index() || x.device_index() != w_hh.device_index()
          || x.device_index() != b_ih.device_index() || x.device_index() != b_hh.device_index()
          || x.device_index() != h.device_index() || x.device() != c.device())
        throw std::invalid_argument("LSTMCell requires all tensors to use the same device_index");

      const dim_t batch = x.dim(0);        // first dim of x [batch, input_size]
      const dim_t input_size = x.dim(1);   // second dim of x

      if (batch != h_prev.dim(0) || batch != c_prev.dim(0))
        throw std::invalid_argument("LSTMCell requires all inputs to use the same batch size");


      // CPU only for now.
      compute<Device::CPU, float>(x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh, h, c);
    }

  }
}
