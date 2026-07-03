#include "ctranslate2/thread_pool.h"

#include <chrono>
#include <future>
#include <iostream>

#include "ctranslate2/utils.h"

namespace ctranslate2 {

  Job::~Job() {
    if (_counter)
      *_counter -= 1;
  }

  void Job::set_job_counter(std::atomic<size_t>& counter) {
    _counter = &counter;
    *_counter += 1;
  }


  JobQueue::JobQueue(size_t maximum_size)
    : _maximum_size(maximum_size)
    , _request_end(false)
  {
  }

  JobQueue::~JobQueue() {
    close();
  }

  size_t JobQueue::size() const {
    const std::lock_guard<std::mutex> lock(_mutex);
    return _queue.size();
  }

  bool JobQueue::can_get_job() const {
    return !_queue.empty() || _request_end;
  }

  void JobQueue::put(std::unique_ptr<Job> job) {
    std::unique_lock<std::mutex> lock(_mutex);
    _can_put_job.wait(lock, [this]{ return _queue.size() < _maximum_size; });

    _queue.emplace(std::move(job));
    lock.unlock();
    _can_get_job.notify_one();
  }

  std::unique_ptr<Job> JobQueue::get(const std::function<void()>& before_wait) {
    std::unique_lock<std::mutex> lock(_mutex);

    while (!can_get_job()) {
      // Only call before_wait() if we're not shutting down.
      // before_wait() may call cudaStreamSynchronize() which can block
      // indefinitely if there's pending GPU work, preventing clean shutdown.
      if (before_wait && !_request_end) {
        // Release lock before calling before_wait() to avoid holding mutex
        // during potentially blocking CUDA operations.
        lock.unlock();
        before_wait();
        lock.lock();
        // Re-check condition after re-acquiring lock
        continue;
      }
      _can_get_job.wait(lock, [this]{ return can_get_job(); });
    }

    if (!_queue.empty()) {
      auto job = std::move(_queue.front());
      _queue.pop();
      lock.unlock();
      _can_put_job.notify_one();
      return job;
    }

    return nullptr;
  }

  void JobQueue::close() {
    if (_request_end)
      return;

    {
      const std::lock_guard<std::mutex> lock(_mutex);
      _request_end = true;
    }

    _can_get_job.notify_all();
  }


  static void set_thread_affinity(std::thread& thread, int index) {
#if !defined(__linux__) || defined(_OPENMP)
    (void)thread;
    (void)index;
    throw std::runtime_error("Setting thread affinity is only supported in Linux binaries built "
                             "with -DOPENMP_RUNTIME=NONE");
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(index, &cpuset);
    const int status = pthread_setaffinity_np(thread.native_handle(), sizeof (cpu_set_t), &cpuset);
    if (status != 0) {
      throw std::runtime_error("Error calling pthread_setaffinity_np: "
                               + std::to_string(status));
    }
#endif
  }

  static thread_local Worker* local_worker = nullptr;

  void Worker::start(JobQueue& job_queue, int thread_affinity) {
    _thread = std::thread(&Worker::run, this, std::ref(job_queue));
    if (thread_affinity >= 0)
      set_thread_affinity(_thread, thread_affinity);
  }

  void Worker::join(int timeout_ms) {
    if (!_thread.joinable()) {
      return;
    }

    if (timeout_ms <= 0) {
      _thread.join();
      return;
    }

    // Timed join using a helper thread. If the worker doesn't finish within
    // the timeout (e.g., stuck in CUDA teardown on Windows), detach both
    // threads to prevent blocking the process indefinitely.
    std::promise<void> join_promise;
    std::future<void> join_future = join_promise.get_future();

    std::thread join_helper([this, &join_promise]() {
      if (_thread.joinable()) {
        _thread.join();
      }
      join_promise.set_value();
    });

    auto status = join_future.wait_for(std::chrono::milliseconds(timeout_ms));

    if (status == std::future_status::ready) {
      join_helper.join();
    } else {
      std::cerr << "[CTranslate2] Warning: Worker thread join timed out after "
                << timeout_ms << "ms, detaching to prevent hang\n";
      join_helper.detach();
      if (_thread.joinable()) {
        _thread.detach();
      }
    }
  }

  void Worker::run(JobQueue& job_queue) {
    local_worker = this;
    initialize();

    const std::function<void()> before_wait = [this]{ return idle(); };

    while (true) {
      auto job = job_queue.get(before_wait);
      if (!job)
        break;
      job->run();
    }

    finalize();
    local_worker = nullptr;
  }


  ThreadPool::ThreadPool(size_t num_threads, size_t maximum_queue_size, int core_offset)
    : _queue(maximum_queue_size)
    , _num_active_jobs(0)
  {
    _workers.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
      _workers.emplace_back(std::make_unique<Worker>());

    start_workers(core_offset);
  }

  ThreadPool::ThreadPool(std::vector<std::unique_ptr<Worker>> workers,
                         size_t maximum_queue_size,
                         int core_offset)
    : _queue(maximum_queue_size)
    , _workers(std::move(workers))
    , _num_active_jobs(0)
  {
    start_workers(core_offset);
  }

  ThreadPool::~ThreadPool() {
    // Signal all workers to stop idle synchronization BEFORE closing the queue.
    // This prevents the race where a worker enters synchronize_stream() inside
    // idle() just before close() sets _request_end. Without this, the worker
    // blocks on CUDA sync while the queue is already closed.
    for (auto& worker : _workers)
      worker->prepare_shutdown();
    _queue.close();
    for (auto& worker : _workers)
      worker->join();
  }

  void ThreadPool::start_workers(int core_offset) {
    for (int i = 0; static_cast<size_t>(i) < _workers.size(); ++i)
      _workers[i]->start(_queue, core_offset >= 0 ? core_offset + i : core_offset);
  }

  void ThreadPool::post(std::unique_ptr<Job> job) {
    job->set_job_counter(_num_active_jobs);
    _queue.put(std::move(job));
  }

  size_t ThreadPool::num_threads() const {
    return _workers.size();
  }

  size_t ThreadPool::num_queued_jobs() const {
    return _queue.size();
  }

  size_t ThreadPool::num_active_jobs() const {
    return _num_active_jobs;
  }

  Worker& ThreadPool::get_worker(size_t index) {
    return *_workers.at(index);
  }

  Worker& ThreadPool::get_local_worker() {
    if (!local_worker)
      throw std::runtime_error("No worker is available in this thread");
    return *local_worker;
  }

}
