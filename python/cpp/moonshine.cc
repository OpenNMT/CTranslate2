#include "module.h"

#include <ctranslate2/models/moonshine.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class MoonshineWrapper : public ReplicaPoolHelper<models::Moonshine> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& features, const bool to_cpu) {
        return _pool->encode(features, to_cpu).get();
      }

      std::variant<std::vector<models::MoonshineGenerationResult>,
                   std::vector<AsyncResult<models::MoonshineGenerationResult>>>
      generate(const StorageView& audio,
               std::variant<BatchTokens, BatchIds> prompts,
               bool asynchronous,
               size_t beam_size,
               float patience,
               size_t num_hypotheses,
               float length_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               bool suppress_blank,
               const std::optional<std::vector<int>>& suppress_tokens,
               size_t sampling_topk,
               float sampling_temperature) {
        std::vector<std::future<models::MoonshineGenerationResult>> futures;

        models::MoonshineOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;
        options.suppress_blank = suppress_blank;

        if (suppress_tokens)
          options.suppress_tokens = suppress_tokens.value();
        else
          options.suppress_tokens.clear();
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        if (prompts.index() == 0)
          futures = _pool->generate(audio, std::get<BatchTokens>(prompts), options);
        else
          futures = _pool->generate(audio, std::get<BatchIds>(prompts), options);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }


    };


    void register_moonshine(py::module& m) {
      py::class_<models::MoonshineGenerationResult>(m, "MoonshineGenerationResult",
                                                  "A generation result from the Moonshine model.")

        .def_readonly("sequences", &models::MoonshineGenerationResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("sequences_ids", &models::MoonshineGenerationResult::sequences_ids,
                      "Generated sequences of token IDs.")
        .def_readonly("scores", &models::MoonshineGenerationResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")

        .def("__repr__", [](const models::MoonshineGenerationResult& result) {
          return "MoonshineGenerationResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            + ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ")";
        })
        ;

      declare_async_wrapper<models::MoonshineGenerationResult>(m, "MoonshineGenerationResultAsync");

      py::class_<MoonshineWrapper>(
        m, "Moonshine",
        R"pbdoc(
            Implements the Moonshine speech recognition model published by Useful Sensors.

            See Also:
               https://github.com/usefulsensors/moonshine
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
                 Initializes a Moonshine model from a converted model.

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

        .def_property_readonly("device", &MoonshineWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &MoonshineWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &MoonshineWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &MoonshineWrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &MoonshineWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &MoonshineWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &MoonshineWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("encode", &MoonshineWrapper::encode,
             py::arg("audio"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input audio.

                 Arguments:
                   audio: Audio, as a float array with shape
                     ``[batch_size, 1, audio_length]``.
                   to_cpu: Copy the encoder output to the CPU before returning the value.

                 Returns:
                   The encoder output.
             )pbdoc")

        .def("generate", &MoonshineWrapper::generate,
             py::arg("audio"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("asynchronous")=false,
             py::arg("beam_size")=1,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("suppress_blank")=true,
             py::arg("suppress_tokens")=std::vector<int>{-1},
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features and generates from the given prompt.

                 Arguments:
                   audio: Audio in [batch_size, audio_len] shaped array.
                   prompts: Prompt for model. Defaults to ['<s'].
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   max_length: Maximum generation length.
                   return_scores: Include the scores in the output.
                   suppress_blank: Suppress blank outputs at the beginning of the sampling.
                   suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                     of symbols as defined in the model ``config.json`` file.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.

                 Returns:
                   A list of generation results.
             )pbdoc")

        .def("unload_model", &MoonshineWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this whisper but keep enough runtime context
                 to quickly resume whisper on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &MoonshineWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &MoonshineWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
