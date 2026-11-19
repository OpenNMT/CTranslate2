// Tests of how the MPS backend reports a failed command buffer. A failure is
// recorded with a test hook, as the completion handler of a failed command
// buffer would do, so that no GPU fault is needed.

#ifdef CT2_WITH_MPS

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <ctranslate2/devices.h>
#include <ctranslate2/models/model.h>
#include <ctranslate2/replica_pool.h>
#include <ctranslate2/translator.h>

#include "mps/testing.h"
#include "mps/utils.h"
#include "test_utils.h"

namespace {

  constexpr std::chrono::seconds timeout(60);

  // Returns the message of the exception thrown by func, or an empty string.
  template <typename Func>
  std::string error_of(Func&& func) {
    try {
      func();
    } catch (const std::exception& e) {
      return e.what();
    }
    return std::string();
  }

  std::string synchronize_error() {
    return error_of([] { synchronize_stream(Device::MPS); });
  }

  bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
  }

  // A thread that has its own MPS stream, like a worker, and runs the functions
  // it is given one at a time.
  class StreamThread {
  public:
    StreamThread()
      : _thread([this]() { loop(); })
    {
    }

    ~StreamThread() {
      stop();
    }

    // Runs func on the thread and returns the message of the exception it
    // throws, or an empty string.
    template <typename Func>
    std::string run(Func func) {
      std::packaged_task<std::string()> task([func]() mutable { return error_of(func); });
      auto result = task.get_future();
      {
        const std::lock_guard<std::mutex> lock(_mutex);
        _tasks.emplace(std::move(task));
      }
      _wake.notify_one();
      if (result.wait_for(timeout) != std::future_status::ready) {
        ADD_FAILURE() << "The MPS test thread did not answer";
        std::abort();
      }
      return result.get();
    }

    // Ends the thread, and so destroys its stream.
    void stop() {
      if (!_thread.joinable())
        return;
      {
        const std::lock_guard<std::mutex> lock(_mutex);
        _stopped = true;
      }
      _wake.notify_one();
      _thread.join();
    }

  private:
    void loop() {
      while (true) {
        std::packaged_task<std::string()> task;
        {
          std::unique_lock<std::mutex> lock(_mutex);
          _wake.wait(lock, [this]() { return _stopped || !_tasks.empty(); });
          if (_tasks.empty())
            return;
          task = std::move(_tasks.front());
          _tasks.pop();
        }
        task();
      }
    }

    std::mutex _mutex;
    std::condition_variable _wake;
    std::queue<std::packaged_task<std::string()>> _tasks;
    bool _stopped = false;
    std::thread _thread;
  };

  // Gives the thread an MPS stream with some work on it.
  void use_stream() {
    StorageView x({4}, 1.f, Device::MPS);
    x.fill(2.f);
  }

  bool wait_for_warnings(uint64_t count) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (mps::testing::kept_error_warnings() < count) {
      if (std::chrono::steady_clock::now() > deadline)
        return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
  }

  // The smallest replica a worker thread needs.
  struct ErrorTestReplica {
    static std::unique_ptr<ErrorTestReplica> create_from_model(const models::Model&) {
      return std::make_unique<ErrorTestReplica>();
    }
  };

  class ErrorTestPool : public ReplicaPool<ErrorTestReplica> {
  public:
    using ReplicaPool::ReplicaPool;
  };

  std::shared_ptr<const models::Model> load_mps_model() {
    return models::Model::load(default_model_dir(), Device::MPS, 0, ComputeType::FLOAT16);
  }

}

// The error belongs to the stream that submitted the failed command buffer:
// another thread that encodes and synchronizes in the meantime must not get it.
TEST(MPSCommandErrorTest, ErrorIsRaisedOnTheThreadThatOwnsIt) {
  StreamThread owner;
  StreamThread other;

  EXPECT_EQ(owner.run([] {
    use_stream();
    mps::testing::record_command_error("failure of the owner");
  }), "");

  float other_value = 0;
  EXPECT_EQ(other.run([&] {
    StorageView y({4}, 2.f, Device::MPS);
    y.fill(3.f);
    synchronize_stream(Device::MPS);
    other_value = y.to(Device::CPU).data<float>()[0];
  }), "");
  EXPECT_EQ(other_value, 3.f);
  EXPECT_EQ(other.run([] { synchronize_stream(Device::MPS); }), "");

  const std::string error = owner.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(error, "failure of the owner")) << "got: " << error;
}

// The next encoding operation of the owner raises it as well, before the
// stream starts anything new.
TEST(MPSCommandErrorTest, ErrorIsRaisedByTheNextEncodingOperation) {
  StreamThread owner;
  StreamThread other;

  EXPECT_EQ(owner.run([] {
    use_stream();
    mps::testing::record_command_error("failure before encoding");
  }), "");
  EXPECT_EQ(other.run([] { use_stream(); }), "");

  const std::string error = owner.run([] { StorageView y({4}, 5.f, Device::MPS); });
  EXPECT_TRUE(contains(error, "failure before encoding")) << "got: " << error;
  EXPECT_EQ(owner.run([] { use_stream(); }), "");
}

TEST(MPSCommandErrorTest, ErrorIsReportedOnce) {
  StreamThread thread;

  EXPECT_EQ(thread.run([] {
    use_stream();
    mps::testing::record_command_error("first failure");
  }), "");

  const std::string error = thread.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(error, "first failure")) << "got: " << error;
  EXPECT_EQ(thread.run([] { synchronize_stream(Device::MPS); }), "");
  EXPECT_EQ(thread.run([] { use_stream(); synchronize_stream(Device::MPS); }), "");

  // A later failure is a new error.
  EXPECT_EQ(thread.run([] { mps::testing::record_command_error("second failure"); }), "");
  const std::string second = thread.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(second, "second failure")) << "got: " << second;
  EXPECT_EQ(thread.run([] { synchronize_stream(Device::MPS); }), "");
}

// The device-wide synchronization reports the error of any stream, and it is
// not reported again by that stream.
TEST(MPSCommandErrorTest, DeviceSynchronizationReportsAnErrorOnce) {
  StreamThread owner;

  EXPECT_EQ(owner.run([] {
    use_stream();
    mps::testing::record_command_error("failure seen by the device barrier");
  }), "");

  const std::string error = error_of([] { mps::synchronize_all(); });
  EXPECT_TRUE(contains(error, "failure seen by the device barrier")) << "got: " << error;
  EXPECT_EQ(error_of([] { mps::synchronize_all(); }), "");
  EXPECT_EQ(owner.run([] { synchronize_stream(Device::MPS); }), "");
}

// When several streams have an error, each one is reported once: the device
// barrier takes one, the other stays with its owner.
TEST(MPSCommandErrorTest, DeviceSynchronizationReportsEachErrorOnce) {
  StreamThread first;
  StreamThread second;

  EXPECT_EQ(first.run([] {
    use_stream();
    mps::testing::record_command_error("failure of the first");
  }), "");
  EXPECT_EQ(second.run([] {
    use_stream();
    mps::testing::record_command_error("failure of the second");
  }), "");

  const std::string barrier = error_of([] { mps::synchronize_all(); });
  const bool barrier_took_first = contains(barrier, "failure of the first");
  const bool barrier_took_second = contains(barrier, "failure of the second");
  ASSERT_NE(barrier_took_first, barrier_took_second) << "got: " << barrier;

  StreamThread& left = barrier_took_first ? second : first;
  const std::string left_message = barrier_took_first ? "failure of the second"
                                                      : "failure of the first";
  const std::string error = left.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(error, left_message)) << "got: " << error;

  EXPECT_EQ(error_of([] { mps::synchronize_all(); }), "");
  EXPECT_EQ(first.run([] { synchronize_stream(Device::MPS); }), "");
  EXPECT_EQ(second.run([] { synchronize_stream(Device::MPS); }), "");
}

// What a worker does when it has no job: it must not throw, it logs the
// failure once, and the stream raises it for the next real operation.
TEST(MPSCommandErrorTest, IdleSynchronizationKeepsTheError) {
  StreamThread thread;

  EXPECT_EQ(thread.run([] {
    use_stream();
    mps::testing::record_command_error("failure while idle");
  }), "");

  const uint64_t warnings = mps::testing::kept_error_warnings();
  EXPECT_EQ(thread.run([] { synchronize_stream_nothrow(Device::MPS); }), "");
  EXPECT_EQ(thread.run([] { synchronize_stream_nothrow(Device::MPS); }), "");
  EXPECT_EQ(mps::testing::kept_error_warnings(), warnings + 1);

  const std::string error = thread.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(error, "failure while idle")) << "got: " << error;
  EXPECT_EQ(thread.run([] { synchronize_stream_nothrow(Device::MPS); }), "");
  EXPECT_EQ(thread.run([] { synchronize_stream(Device::MPS); }), "");
  EXPECT_EQ(mps::testing::kept_error_warnings(), warnings + 1);

  // A new failure is logged again.
  EXPECT_EQ(thread.run([] {
    mps::testing::record_command_error("another failure while idle");
    synchronize_stream_nothrow(Device::MPS);
  }), "");
  EXPECT_EQ(mps::testing::kept_error_warnings(), warnings + 2);
  const std::string another = thread.run([] { StorageView y({4}, 1.f, Device::MPS); });
  EXPECT_TRUE(contains(another, "another failure while idle")) << "got: " << another;
}

// The real idle hook of a replica worker. An exception that leaves it ends the
// worker thread, and so the process.
TEST(MPSCommandErrorTest, IdleWorkerSurvivesAnErrorAndRaisesItLater) {
  ErrorTestPool pool(load_mps_model());
  const uint64_t warnings = mps::testing::kept_error_warnings();

  auto recorded = pool.post<int>([](ErrorTestReplica&) {
    use_stream();
    mps::testing::record_command_error("failure of a job");
    return 0;
  });
  ASSERT_EQ(recorded.wait_for(timeout), std::future_status::ready);
  (void)recorded.get();

  // The worker has no other job: it synchronizes, logs and goes to sleep.
  ASSERT_TRUE(wait_for_warnings(warnings + 1));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(mps::testing::kept_error_warnings(), warnings + 1);

  // The worker is still alive and its next job gets the error, once.
  auto raised = pool.post<std::string>([](ErrorTestReplica&) {
    return error_of([] { synchronize_stream(Device::MPS); });
  });
  ASSERT_EQ(raised.wait_for(timeout), std::future_status::ready);
  const std::string error = raised.get();
  EXPECT_TRUE(contains(error, "failure of a job")) << "got: " << error;

  auto clean = pool.post<std::string>([](ErrorTestReplica&) {
    use_stream();
    return error_of([] { synchronize_stream(Device::MPS); });
  });
  ASSERT_EQ(clean.wait_for(timeout), std::future_status::ready);
  EXPECT_EQ(clean.get(), "");
}

// Same, with a translator: the batch that follows the failure reports it, the
// one after that is translated.
TEST(MPSCommandErrorTest, TranslatorRaisesAnErrorFoundWhileIdleInItsNextBatch) {
  const std::vector<std::string> input = {"آ", "ت", "ز", "م", "و", "ن"};
  TranslationOptions options;
  options.beam_size = 1;
  options.max_decoding_length = 12;

  Translator cpu(default_model_dir(), Device::CPU, ComputeType::FLOAT32);
  Translator mps_translator(default_model_dir(), Device::MPS, ComputeType::FLOAT16);
  const auto expected = cpu.translate_batch({input}, options)[0].output();
  EXPECT_EQ(mps_translator.translate_batch({input}, options)[0].output(), expected);

  const uint64_t warnings = mps::testing::kept_error_warnings();
  auto recorded = mps_translator.post<int>([](models::SequenceToSequenceReplica&) {
    mps::testing::record_command_error("failure after a batch");
    return 0;
  });
  ASSERT_EQ(recorded.wait_for(timeout), std::future_status::ready);
  (void)recorded.get();
  ASSERT_TRUE(wait_for_warnings(warnings + 1));

  const std::string error = error_of([&] { mps_translator.translate_batch({input}, options); });
  EXPECT_TRUE(contains(error, "failure after a batch")) << "got: " << error;
  EXPECT_EQ(mps_translator.translate_batch({input}, options)[0].output(), expected);
}

// ~Model waits for the device. It cannot throw: that would terminate.
TEST(MPSCommandErrorTest, ModelDestroyedWithAPendingErrorKeepsTheError) {
  StreamThread other;
  StreamThread unloader;

  std::shared_ptr<const models::Model> model;
  EXPECT_EQ(unloader.run([&] { model = load_mps_model(); }), "");
  ASSERT_NE(model, nullptr);

  EXPECT_EQ(other.run([] {
    use_stream();
    mps::testing::record_command_error("failure of another model");
  }), "");
  EXPECT_EQ(unloader.run([] {
    mps::testing::record_command_error("failure of the unloading thread");
  }), "");

  const uint64_t warnings = mps::testing::kept_error_warnings();
  EXPECT_EQ(unloader.run([&] { model.reset(); }), "");
  EXPECT_EQ(mps::testing::kept_error_warnings(), warnings + 2);

  // Each stream still has its own error, and only that one.
  const std::string unloader_error = unloader.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(unloader_error, "failure of the unloading thread"))
    << "got: " << unloader_error;
  EXPECT_EQ(unloader.run([] { synchronize_stream(Device::MPS); }), "");
  const std::string other_error = other.run([] { synchronize_stream(Device::MPS); });
  EXPECT_TRUE(contains(other_error, "failure of another model")) << "got: " << other_error;
  EXPECT_EQ(other.run([] { synchronize_stream(Device::MPS); }), "");
}

// A completion handler can run after the thread that submitted the command
// buffer is gone. It records on the state of the stream, not on the stream.
TEST(MPSCommandErrorTest, ErrorCanBeRecordedAfterItsStreamIsGone) {
  mps::testing::StreamErrorProbe probe;
  {
    StreamThread thread;
    EXPECT_EQ(thread.run([&] {
      use_stream();
      probe = mps::testing::stream_error_probe();
    }), "");
    thread.stop();
  }

  EXPECT_NO_THROW(probe.record("late failure"));
  EXPECT_TRUE(probe.pending());
  EXPECT_EQ(synchronize_error(), "");
  EXPECT_EQ(error_of([] { mps::synchronize_all(); }), "");
}

#endif
