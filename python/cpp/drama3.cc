#include "module.h"
#include "utils.h"
#include <pybind11/numpy.h>
#include <ctranslate2/models/drama3.h>

namespace ctranslate2 {
  namespace python {

    namespace {

      py::array_t<float> generation_audio(const models::Drama3Result& result) {
        py::array_t<float> audio(result.audio.size());
        std::copy_n(result.audio.data<float>(), result.audio.size(), audio.mutable_data());
        return audio;
      }

      py::array_t<int32_t> generation_codes(const models::Drama3Result& result) {
        py::array_t<int32_t> codes(std::vector<py::ssize_t>{result.codes.dim(0), 10});
        std::copy_n(result.codes.data<int32_t>(), result.codes.size(), codes.mutable_data());
        return codes;
      }

      py::dict generation_result_to_dict(const models::Drama3Result& result) {
        py::dict output;
        output["audio"] = generation_audio(result);
        output["codes"] = generation_codes(result);
        output["sample_rate"] = result.sample_rate;
        output["finish_reason"] = result.finish_reason;
        py::dict timings;
        timings["prefill_seconds"] = result.timings.prefill_seconds;
        timings["backbone_decode_seconds"] = result.timings.backbone_decode_seconds;
        timings["tower_decode_seconds"] = result.timings.tower_decode_seconds;
        timings["codec_seconds"] = result.timings.codec_seconds;
        timings["total_native_seconds"] = result.timings.total_native_seconds;
        output["timings"] = std::move(timings);
        return output;
      }

    }

    void register_drama3(py::module& m) {
      py::class_<models::Drama3Timings>(m, "Drama3Timings")
          .def_readonly("prefill_seconds", &models::Drama3Timings::prefill_seconds)
          .def_readonly("backbone_decode_seconds", &models::Drama3Timings::backbone_decode_seconds)
          .def_readonly("tower_decode_seconds", &models::Drama3Timings::tower_decode_seconds)
          .def_readonly("codec_seconds", &models::Drama3Timings::codec_seconds)
          .def_readonly("total_native_seconds", &models::Drama3Timings::total_native_seconds);
      py::class_<models::Drama3Result>(m, "Drama3Result")
          .def_property_readonly("audio", &generation_audio)
          .def_property_readonly("codes", &generation_codes)
          .def_readonly("sample_rate", &models::Drama3Result::sample_rate)
          .def_readonly("finish_reason", &models::Drama3Result::finish_reason)
          .def_readonly("timings", &models::Drama3Result::timings)
          // Preserve existing result["audio"] callers while exposing a typed result.
          .def("__getitem__", [](const models::Drama3Result& result, const std::string& key) {
            return py::object(generation_result_to_dict(result)[py::str(key)]);
          })
          .def("to_dict", &generation_result_to_dict);
      declare_async_wrapper<models::Drama3Result>(m, "Drama3ResultAsync");

      auto copy_prompt = [](py::array_t<int32_t, py::array::c_style> prompt) {
        if (prompt.ndim() != 2 || prompt.shape(1) != 11)
          throw std::invalid_argument("Drama3 prompt must have shape [rows,11]");
        return StorageView({prompt.shape(0), prompt.shape(1)},
                           std::vector<int32_t>(prompt.data(), prompt.data() + prompt.size()));
      };
      py::class_<models::Drama3>(
          m, "Drama3", "Experimental offline text-to-speech synthesis using model replicas.")
          .def(py::init<const std::string&, size_t, const std::string&, const std::string&, int,
                        size_t, long>(),
               py::arg("model_path"), py::arg("intra_threads") = 8,
               py::arg("compute_type") = "default", py::arg("device") = "cpu",
               py::arg("device_index") = 0, py::arg("inter_threads") = 1,
               py::arg("max_queued_batches") = 0, py::call_guard<py::gil_scoped_release>(),
               R"pbdoc(
               Loads a converted Drama3 model with a pool of execution replicas.

               Arguments:
                 model_path: Local CTranslate2 model directory.
                 intra_threads: Positive number of CPU threads per replica.
                 compute_type: CTranslate2 compute type, subject to backend support.
                 device: Execution device: cpu, cuda (NVIDIA or ROCm), or auto.
                 device_index: Zero-based GPU index; use 0 for CPU.
                 inter_threads: Positive number of independent execution replicas.
                   CPU checkpoint weights are shared; GPU caches are private to each replica.
                 max_queued_batches: Maximum queued requests; 0 uses four per replica,
                   a negative value allows an unlimited queue.
           )pbdoc")
          .def_property_readonly("device", [](const models::Drama3& model) {
            return device_to_str(model.get_first_replica().device());
          }, "Device used for execution.")
          .def_property_readonly("device_index", [](const models::Drama3& model) {
            return std::vector<int>{model.get_first_replica().device_index()};
          }, "Device IDs where this model executes.")
          .def_property_readonly("compute_type", [](const models::Drama3& model) {
            return compute_type_to_str(model.get_first_replica().compute_type());
          }, "Resolved computation type used for execution.")
          .def_property_readonly("num_replicas", &models::Drama3::num_replicas)
          .def_property_readonly("num_queued_batches", &models::Drama3::num_queued_batches)
          .def_property_readonly("num_active_batches", &models::Drama3::num_active_batches)
          .def_property_readonly("prefix_cache_stats", [](const models::Drama3& model) {
            const auto& base = *model.get_first_replica().model();
            const auto& cache = dynamic_cast<const models::Drama3Model&>(base).state_cache();
            py::dict stats;
            stats["entries"] = cache.entries();
            stats["hits"] = cache.hits();
            stats["misses"] = cache.misses();
            return stats;
          }, "Static-prefix cache diagnostics: {entries, hits, misses}.")
          .def(
              "generate",
              [copy_prompt](models::Drama3& model,
                 py::array_t<int32_t, py::array::c_style> prompt,
                 dim_t max_frames, dim_t prefill_chunk_size, float temperature, dim_t top_k,
                 float top_p, uint64_t seed, bool greedy, bool asynchronous,
                 dim_t static_prompt_rows, bool cache_static_prompt) -> py::object {
                auto input = copy_prompt(prompt);
                std::future<models::Drama3Result> future;
                {
                  py::gil_scoped_release release;
                  future = model.generate_async(input, {max_frames, prefill_chunk_size, temperature,
                                                        top_k, top_p, seed, greedy,
                                                        static_prompt_rows, cache_static_prompt});
                }
                if (asynchronous)
                  return py::cast(AsyncResult<models::Drama3Result>(std::move(future)));
                models::Drama3Result result;
                {
                  py::gil_scoped_release release;
                  result = future.get();
                }
                return py::cast(std::move(result));
              },
              py::arg("prompt_rows"), py::arg("max_frames") = 512,
              py::arg("prefill_chunk_size") = 128, py::arg("temperature") = .8f,
              py::arg("top_k") = 30, py::arg("top_p") = .8f, py::arg("seed") = 0,
              py::arg("greedy") = false, py::arg("asynchronous") = false,
              py::arg("static_prompt_rows") = 0, py::arg("cache_static_prompt") = true,
              R"pbdoc(
             Generates offline speech on the first available replica.

             Arguments:
               prompt_rows: Contiguous INT32 [rows, 11]: text IDs and ten codec columns.
               max_frames: Maximum generated frames (2048 samples at 44100 Hz each).
               prefill_chunk_size: Prompt processing chunk size.
               temperature: Positive sampling temperature.
               top_k: Positive sampling candidate count.
               top_p: Sampling probability threshold in (0, 1].
               seed: Per-request random seed.
               greedy: Use diagnostic greedy decoding.
               asynchronous: Return a Drama3ResultAsync without waiting for generation.

             Returns:
               Drama3Result, or Drama3ResultAsync when asynchronous=True. Results expose
               mono FLOAT32 audio, INT32 codes [frames, 10], sample_rate, finish_reason
               (eos or length), and timings. Mapping-style access remains supported.
         )pbdoc")
          .def(
              "generate_batch",
              [copy_prompt](models::Drama3& model,
                 const std::vector<py::array_t<int32_t, py::array::c_style>>& prompts,
                 dim_t max_frames, dim_t prefill_chunk_size, float temperature, dim_t top_k,
                 float top_p, uint64_t seed, bool greedy, bool asynchronous,
                 dim_t static_prompt_rows, bool cache_static_prompt) {
                std::vector<StorageView> inputs;
                inputs.reserve(prompts.size());
                for (const auto& prompt : prompts)
                  inputs.emplace_back(copy_prompt(prompt));
                std::vector<std::future<models::Drama3Result>> futures;
                {
                  py::gil_scoped_release release;
                  futures = model.generate_batch(inputs, {max_frames, prefill_chunk_size,
                                                          temperature, top_k, top_p, seed, greedy,
                                                          static_prompt_rows, cache_static_prompt});
                }
                if (asynchronous)
                  return py::cast(std::get<std::vector<AsyncResult<models::Drama3Result>>>(
                      maybe_wait_on_futures(std::move(futures), true)));
                std::vector<models::Drama3Result> results;
                {
                  py::gil_scoped_release release;
                  results = wait_on_futures(std::move(futures));
                }
                return py::cast(std::move(results));
              },
              py::arg("prompts"), py::arg("max_frames") = 512,
              py::arg("prefill_chunk_size") = 128, py::arg("temperature") = .8f,
              py::arg("top_k") = 30, py::arg("top_p") = .8f, py::arg("seed") = 0,
              py::arg("greedy") = false, py::arg("asynchronous") = false,
              py::arg("static_prompt_rows") = 0, py::arg("cache_static_prompt") = true,
              R"pbdoc(
             Schedules independent prompts across the replica pool in input order.
             Options match generate. Returns a list of Drama3Result objects, or
             Drama3ResultAsync objects when asynchronous=True. Each prompt uses its
             own sequence state and the supplied seed; kernels process one sequence.
         )pbdoc");
    }

  }
}
