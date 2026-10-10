#include "module.h"
#include <sstream>
#include <ctranslate2/profiler.h>

namespace ctranslate2 {
  namespace python {

    void register_profiling(py::module& m) {

      m.def("init_profiling", &ctranslate2::init_profiling);
      m.def("dump_profiling", []() {
        std::ostringstream oss;
        ctranslate2::dump_profiling(oss);
        return oss.str();
	});
	}

  }
}
