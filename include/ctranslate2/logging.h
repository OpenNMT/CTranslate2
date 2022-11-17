#pragma once

namespace ctranslate2 {

  enum class LogLevel {
    OFF = -3,
    CRITICAL = -2,
    ERROR = -1,
    WARNING = 0,
    INFO = 1,
    DEBUG = 2,
    TRACE = 3,
  };

  void set_log_level(const LogLevel level);
  LogLevel get_log_level();

}
