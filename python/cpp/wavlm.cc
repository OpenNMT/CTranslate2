#include "module.h"

#include <ctranslate2/models/wavlm.h>

#include "replica_pool.h"

#include <iostream>

namespace ctranslate2 {
  namespace python {

    class WavLMWrapper : public ReplicaPoolHelper<models::WavLM> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& features, const bool to_cpu) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->encode(features, to_cpu).get();
      }
    };

    void register_wavlm(py::module& m) {
      py::class_<WavLMWrapper>(
        m, "WavLM",
        R"pbdoc(
            Implements the WavLM speech recognition model published by Microsoft.
        )pbdoc")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a WavLM model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8, int8_float32,
                     int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   flash_attention: run model with flash attention 2 for self-attention layer
                   tensor_parallel: run model with tensor parallel mode
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &WavLMWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &WavLMWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &WavLMWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &WavLMWrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &WavLMWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &WavLMWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &WavLMWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("encode", &WavLMWrapper::encode,
             py::arg("features"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features.

                 Arguments:
                   features: hidden_states (up to v.4.3.1, https://github.com/OpenNMT/CTranslate2/blob/59c7dda738892df7a064aa360d0e45a4c3840b07/python/tests/test_transformers.py#L1028) or
                             raw audio, as a float array with shape (followed by VAD)
                             ``[batch_size, 409, 1024]`` or ``[batch_size, 1, 131200]`` 
                   to_cpu: Copy the encoder output to the CPU before returning the value.

                 Returns:
                   The encoder output.
             )pbdoc")

        .def("unload_model", &WavLMWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this wavlm but keep enough runtime context
                 to quickly resume wavlm on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &WavLMWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &WavLMWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
