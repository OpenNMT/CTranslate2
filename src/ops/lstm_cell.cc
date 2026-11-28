#include "ctranslate2/ops/lstm_cell.h"

namespace ctranslate2 {
  namespace ops {

    void LSTMCell::operator()(const StorageView& x,
                              const StorageView& h_prev,
                              const StorageView& c_prev,
                              const StorageView& w_ih,
                              const StorageView& w_hh,
                              const StorageView& b_ih,
                              const StorageView& b_hh,
                              StorageView& h,
                              StorageView& c) const {
    }

  }
}
