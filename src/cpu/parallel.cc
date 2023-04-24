#include "parallel.h"

namespace ctranslate2 {
  namespace cpu {

#ifndef _OPENMP

    static thread_local size_t num_threads = 4;
    static thread_local bool in_parallel = false;

    void set_num_threads(size_t num) {
      num_threads = num;
    }

    void set_in_parallel_region(bool value) {
      in_parallel = value;
    }

    bool in_parallel_region() {
      return in_parallel;
    }

    BS::thread_pool_light& get_thread_pool() {
      static thread_local BS::thread_pool_light thread_pool(num_threads);
      return thread_pool;
    }

#endif

  }
}
