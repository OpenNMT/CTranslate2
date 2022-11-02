#include "module.h"

#include <ctranslate2/translation.h>

#include "utils.h"

namespace ctranslate2 {
  namespace python {

    void register_translation_result(py::module& m) {
      py::class_<TranslationResult>(m, "TranslationResult", "A translation result.")

        .def_readonly("hypotheses", &TranslationResult::hypotheses,
                      "Translation hypotheses.")
        .def_readonly("scores", &TranslationResult::scores,
                      "Score of each translation hypothesis (empty if :obj:`return_scores` was disabled).")
        .def_readonly("attention", &TranslationResult::attention,
                      "Attention matrix of each translation hypothesis (empty if :obj:`return_attention` was disabled).")

        .def("__repr__", [](const TranslationResult& result) {
          return "TranslationResult(hypotheses=" + std::string(py::repr(py::cast(result.hypotheses)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ", attention=" + std::string(py::repr(py::cast(result.attention)))
            + ")";
        })
        ;

      declare_async_wrapper<TranslationResult>(m, "AsyncTranslationResult");
    }

  }
}
