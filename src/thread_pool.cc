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
      _can_put_job.wait(lock, [this]{ return _queue.size() < _maximum_size; });

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


  void Worker::start(JobQueue& job_queue, int thread_affinity) {
    _thread = std::thread(&Worker::run, this, std::ref(job_queue));
    if (thread_affinity >= 0)
      set_thread_affinity(_thread, thread_affinity);
  }

  void Worker::join() {
    _thread.join();
  }

  void Worker::run(JobQueue& job_queue) {
    initialize();

    while (true) {
      auto job = job_queue.get();
      if (!job)
        break;
      job->run();
    }

    finalize();
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
    _queue.close();
    for (auto& worker : _workers)
      worker->join();
  }

  void ThreadPool::start_workers(int core_offset) {
    for (int i = 0; static_cast<size_t>(i) < _workers.size(); ++i)
      _workers[i]->start(_queue, core_offset >= 0 ? core_offset + i : core_offset);
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

  Worker& ThreadPool::get_worker(size_t index) {
    return *_workers.at(index);
  }

}
