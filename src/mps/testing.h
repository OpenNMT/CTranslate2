#pragma once

#ifdef __APPLE__

#include <cstdint>
#include <memory>
#include <string>

namespace ctranslate2 {
  namespace mps {

    class CommandErrors;

    // Hooks for the tests of the error handling. They make a stream fail the
    // way a command buffer does, without a GPU fault. Not part of the API.
    namespace testing {

      // The error state of the MPS stream of one thread. It keeps that state
      // alive after the thread and its stream are gone, like the completion
      // handler of a command buffer does.
      class StreamErrorProbe {
      public:
        // Records a failed command buffer on the stream, as its completion
        // handler would. Like the handler, it can be called from any thread.
        void record(const std::string& message) const;
        // True if a recorded error was not raised yet.
        bool pending() const;

      private:
        friend StreamErrorProbe stream_error_probe();
        std::shared_ptr<CommandErrors> _errors;
      };

      // The probe of the calling thread's stream.
      StreamErrorProbe stream_error_probe();

      // Records a failed command buffer on the calling thread's stream.
      inline void record_command_error(const std::string& message) {
        stream_error_probe().record(message);
      }

      // Number of warnings logged for errors that were kept instead of thrown.
      uint64_t kept_error_warnings();

    }

  }
}

#endif  // __APPLE__
