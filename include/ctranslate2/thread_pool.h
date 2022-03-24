#pragma once

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ctranslate2 {

  // Base class for asynchronous jobs.
  class Job {
  public:
    virtual ~Job();
    virtual void run() = 0;

    // The job counter is used to track the number of active jobs (queued and currently processed).
    void set_job_counter(std::atomic<size_t>& counter);

  private:
    std::atomic<size_t>* _counter = nullptr;
  };

  // A thread-safe queue of jobs.
  class JobQueue {
  public:
    JobQueue(size_t maximum_size = 0);
    ~JobQueue();

    size_t size() const;

    // Puts a job in the queue.
    // If throttle=true, the method blocks until a free slot is available.
    void put(std::unique_ptr<Job> job, bool throttle = true);

    // Gets a job from the queue. The method block until a job is available.
    // If the queue is closed, the method returns a null pointer.
    std::unique_ptr<Job> get();

    void close();

  private:
    mutable std::mutex _mutex;
    std::queue<std::unique_ptr<Job>> _queue;
    std::condition_variable _can_add_job;
    std::condition_variable _can_get_job;
    size_t _maximum_size;
    bool _request_end;
  };

  // A worker processes jobs in a thread.
  class Worker {
  public:
    virtual ~Worker() = default;

    // Consumes and runs jobs in a loop. The method exits when the queue is closed.
    void run(JobQueue& job_queue);

  protected:
    // Called before the work loop.
    virtual void initialize() {}

    // Runs a job.
    virtual void run_job(std::unique_ptr<Job> job);

    // Called after the work loop.
    virtual void finalize() {}
  };

  // A pool of threads.
  class ThreadPool {
  public:
    // Default thread workers.
    ThreadPool(size_t num_threads,
               size_t maximum_queue_size = 0,
               int core_offset = -1);

    // User-defined thread workers.
    ThreadPool(std::vector<std::unique_ptr<Worker>> workers,
               size_t maximum_queue_size = 0,
               int core_offset = -1);

    ~ThreadPool();

    // Posts a new job.
    // If throttle=true, the method blocks until a free slot is available.
    void post(std::unique_ptr<Job> job, bool throttle = true);

    size_t num_threads() const;

    // Number of jobs in the queue.
    size_t num_queued_jobs() const;

    // Number of jobs in the queue and currently processed by a worker.
    size_t num_active_jobs() const;

  private:
    void start_threads();

    JobQueue _queue;
    const int _core_offset;
    std::vector<std::unique_ptr<Worker>> _workers;
    std::vector<std::thread> _threads;
    std::atomic<size_t> _num_active_jobs;
  };

}
