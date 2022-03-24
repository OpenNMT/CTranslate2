#include "ctranslate2/thread_pool.h"

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

  void JobQueue::put(std::unique_ptr<Job> job, bool throttle) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (throttle)
      _can_add_job.wait(lock, [this]{ return _queue.size() < _maximum_size; });

    _queue.emplace(std::move(job));
    lock.unlock();
    _can_get_job.notify_one();
  }

  std::unique_ptr<Job> JobQueue::get() {
    std::unique_lock<std::mutex> lock(_mutex);
    _can_get_job.wait(lock, [this]{ return !_queue.empty() || _request_end; });

    if (!_queue.empty()) {
      auto job = std::move(_queue.front());
      _queue.pop();
      lock.unlock();
      _can_add_job.notify_one();
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


  void Worker::run(JobQueue& job_queue) {
    initialize();

    while (true) {
      auto job = job_queue.get();
      if (!job)
        break;
      run_job(std::move(job));
    }

    finalize();
  }

  void Worker::run_job(std::unique_ptr<Job> job) {
    job->run();
  }


  ThreadPool::ThreadPool(size_t num_threads, size_t maximum_queue_size, int core_offset)
    : _queue(maximum_queue_size)
    , _core_offset(core_offset)
    , _num_active_jobs(0)
  {
    _workers.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
      _workers.emplace_back(std::make_unique<Worker>());

    start_threads();
  }

  ThreadPool::ThreadPool(std::vector<std::unique_ptr<Worker>> workers,
                         size_t maximum_queue_size,
                         int core_offset)
    : _queue(maximum_queue_size)
    , _core_offset(core_offset)
    , _workers(std::move(workers))
    , _num_active_jobs(0)
  {
    start_threads();
  }

  ThreadPool::~ThreadPool() {
    _queue.close();
    for (auto& thread : _threads)
      thread.join();
  }

  void ThreadPool::start_threads() {
    _threads.reserve(_workers.size());
    for (size_t i = 0; i < _workers.size(); ++i) {
      _threads.emplace_back(&Worker::run, _workers[i].get(), std::ref(_queue));
      if (_core_offset >= 0)
        set_thread_affinity(_threads.back(), _core_offset + i);
    }
  }

  void ThreadPool::post(std::unique_ptr<Job> job, bool throttle) {
    job->set_job_counter(_num_active_jobs);
    _queue.put(std::move(job), throttle);
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

}
