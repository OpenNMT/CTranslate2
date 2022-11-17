#include "ctranslate2/logging.h"

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include "env.h"

namespace ctranslate2 {

  static spdlog::level::level_enum to_spdlog_level(const LogLevel level) {
    switch (level) {
    case LogLevel::OFF:
      return spdlog::level::off;
    case LogLevel::CRITICAL:
      return spdlog::level::critical;
    case LogLevel::ERROR:
      return spdlog::level::err;
    case LogLevel::WARNING:
      return spdlog::level::warn;
    case LogLevel::INFO:
      return spdlog::level::info;
    case LogLevel::DEBUG:
      return spdlog::level::debug;
    case LogLevel::TRACE:
      return spdlog::level::trace;
    default:
      throw std::invalid_argument("Invalid log level");
    }
  }

  static LogLevel to_ct2_level(const spdlog::level::level_enum level) {
    switch (level) {
    case spdlog::level::off:
      return LogLevel::OFF;
    case spdlog::level::critical:
      return LogLevel::CRITICAL;
    case spdlog::level::err:
      return LogLevel::ERROR;
    case spdlog::level::warn:
      return LogLevel::WARNING;
    case spdlog::level::info:
      return LogLevel::INFO;
    case spdlog::level::debug:
      return LogLevel::DEBUG;
    case spdlog::level::trace:
      return LogLevel::TRACE;
    default:
      throw std::invalid_argument("Invalid log level");
    }
  }

  static LogLevel get_default_level() {
    const auto level = read_int_from_env("CT2_VERBOSE", 0);

    if (level < -3 || level > 3)
      throw std::invalid_argument("Invalid log level " + std::to_string(level)
                                  + " (should be between -3 and 3)");

    return static_cast<LogLevel>(level);
  }

  static void init_logger() {
    auto logger = spdlog::stderr_logger_mt("ctranslate2");
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%l] %v");
    spdlog::set_default_logger(logger);
    set_log_level(get_default_level());
  }

  // Initialize the global logger on program start.
  static struct LoggerInit {
    LoggerInit() {
      init_logger();
    }
  } logger_init;

  void set_log_level(const LogLevel level) {
    spdlog::set_level(to_spdlog_level(level));
  }

  LogLevel get_log_level() {
    return to_ct2_level(spdlog::get_level());
  }

}
