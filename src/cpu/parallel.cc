#include "parallel.h"

namespace ctranslate2 {
  namespace cpu {

#ifndef _OPENMP

    static thread_local size_t num_threads = 1;

    void set_num_threads(size_t num) {
      num_threads = num;
    }

    size_t get_num_threads() {
      return num_threads;
    }

    BS::light_thread_pool& get_thread_pool() {
      static thread_local BS::thread_pool thread_pool(num_threads);
      return thread_pool;
    }

#endif

  }
}
