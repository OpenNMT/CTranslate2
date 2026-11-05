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

#  ifdef _WIN32
    // Joining the pool threads at thread exit deadlocks on the loader lock, so
    // ReplicaWorker::finalize() frees the pool instead.
    static thread_local BS::light_thread_pool* thread_pool = nullptr;

    BS::light_thread_pool& get_thread_pool() {
      if (!thread_pool)
        thread_pool = new BS::light_thread_pool(num_threads);
      return *thread_pool;
    }

    void clear_thread_pool() {
      delete thread_pool;
      thread_pool = nullptr;
    }
#  else
    BS::light_thread_pool& get_thread_pool() {
      static thread_local BS::thread_pool thread_pool(num_threads);
      return thread_pool;
    }
#  endif

#endif

  }
}
