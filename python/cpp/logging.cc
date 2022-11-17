#include "module.h"

#include <ctranslate2/logging.h>

namespace ctranslate2 {
  namespace python {

    void register_logging(py::module& m) {
      py::enum_<ctranslate2::LogLevel>(m, "LogLevel")
        .value("OFF", ctranslate2::LogLevel::OFF)
        .value("CRITICAL", ctranslate2::LogLevel::CRITICAL)
        .value("ERROR", ctranslate2::LogLevel::ERROR)
        .value("WARNING", ctranslate2::LogLevel::WARNING)
        .value("INFO", ctranslate2::LogLevel::INFO)
        .value("DEBUG", ctranslate2::LogLevel::DEBUG)
        .value("TRACE", ctranslate2::LogLevel::TRACE)
        .export_values();

      m.def("set_log_level", &ctranslate2::set_log_level);
      m.def("get_log_level", &ctranslate2::get_log_level);
    }

  }
}
