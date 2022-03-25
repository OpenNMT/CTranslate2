#include "ctranslate2/translator_pool.h"

#include <spdlog/spdlog.h>

#include "ctranslate2/utils.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {

  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 const std::string& model_dir,
                                 const Device device,
                                 const int device_index,
                                 const ComputeType compute_type)
  {
    models::ModelFileReader model_reader(model_dir);
    create_translators(num_translators,
                       num_threads_per_translator,
                       model_reader,
                       device,
                       {device_index},
                       compute_type);
  }

  TranslatorPool::TranslatorPool(size_t num_translators,
                                 size_t num_threads_per_translator,
                                 models::ModelReader& model_reader,
                                 const Device device,
                                 const int device_index,
                                 const ComputeType compute_type)
  {
    create_translators(num_translators,
                       num_threads_per_translator,
                       model_reader,
                       device,
                       {device_index},
                       compute_type);
  }

  TranslatorPool::TranslatorPool(size_t num_translators_per_device,
                                 size_t num_threads_per_translator,
                                 const std::string& model_dir,
                                 const Device device,
                                 const std::vector<int>& device_indices,
                                 const ComputeType compute_type)
  {
    models::ModelFileReader model_reader(model_dir);
    create_translators(num_translators_per_device,
                       num_threads_per_translator,
                       model_reader,
                       device,
                       device_indices,
                       compute_type);
  }

  TranslatorPool::TranslatorPool(size_t num_translators_per_device,
                                 size_t num_threads_per_translator,
                                 models::ModelReader& model_reader,
                                 const Device device,
                                 const std::vector<int>& device_indices,
                                 const ComputeType compute_type)
  {
    create_translators(num_translators_per_device,
                       num_threads_per_translator,
                       model_reader,
                       device,
                       device_indices,
                       compute_type);
  }

  std::vector<std::future<TranslationResult>>
  TranslatorPool::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                        const TranslationOptions& options,
                                        const size_t max_batch_size,
                                        const BatchType batch_type) {
    return translate_batch_async(source, {}, options, max_batch_size, batch_type);
  }

  std::vector<std::future<TranslationResult>>
  TranslatorPool::translate_batch_async(const std::vector<std::vector<std::string>>& source,
                                        const std::vector<std::vector<std::string>>& target_prefix,
                                        const TranslationOptions& options,
                                        const size_t max_batch_size,
                                        const BatchType batch_type) {
    return TranslateJobCreator(options).post(*_thread_pool,
                                             load_examples({source, target_prefix}),
                                             max_batch_size,
                                             batch_type,
                                             /*throttle=*/false);
  }

  std::vector<std::future<ScoringResult>>
  TranslatorPool::score_batch_async(const std::vector<std::vector<std::string>>& source,
                                    const std::vector<std::vector<std::string>>& target,
                                    const ScoringOptions& options,
                                    const size_t max_batch_size,
                                    const BatchType batch_type) {
    return ScoreJobCreator(options).post(*_thread_pool,
                                         load_examples({source, target}),
                                         max_batch_size,
                                         batch_type,
                                         /*throttle=*/false);
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const TranslationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return translate_batch(source, {}, options, max_batch_size, batch_type);
  }

  template <typename T>
  std::vector<T> get_results_from_futures(std::vector<std::future<T>> futures) {
    std::vector<T> results;
    results.reserve(futures.size());
    for (auto& future : futures)
      results.emplace_back(future.get());
    return results;
  }

  std::vector<TranslationResult>
  TranslatorPool::translate_batch(const std::vector<std::vector<std::string>>& source,
                                  const std::vector<std::vector<std::string>>& target_prefix,
                                  const TranslationOptions& options,
                                  const size_t max_batch_size,
                                  const BatchType batch_type) {
    return get_results_from_futures(translate_batch_async(source,
                                                          target_prefix,
                                                          options,
                                                          max_batch_size,
                                                          batch_type));
  }

  std::vector<ScoringResult>
  TranslatorPool::score_batch(const std::vector<std::vector<std::string>>& source,
                              const std::vector<std::vector<std::string>>& target,
                              const ScoringOptions& options,
                              const size_t max_batch_size,
                              const BatchType batch_type) {
    return get_results_from_futures(score_batch_async(source, target, options, max_batch_size, batch_type));
  }

  template <typename T>
  static std::vector<T> repeat_elements(const std::vector<T>& v, const size_t repeat) {
    std::vector<int> repeated;
    repeated.reserve(v.size() * repeat);
    for (const T& e : v) {
      for (size_t i = 0; i < repeat; ++i)
        repeated.emplace_back(e);
    }
    return repeated;
  }

  static inline bool have_same_compute_capability(const std::vector<int>& device_indices) {
#ifdef CT2_WITH_CUDA
    if (device_indices.size() > 1) {
      int ref_major = -1;
      int ref_minor = -1;
      for (const int device : device_indices) {
        const cudaDeviceProp& device_prop = cuda::get_device_properties(device);
        const int major = device_prop.major;
        const int minor = device_prop.minor;
        if (ref_major < 0) {
          ref_major = major;
          ref_minor = minor;
        } else if (major != ref_major || minor != ref_minor)
          return false;
      }
    }
#else
    (void)device_indices;
#endif

    return true;
  }

  void TranslatorPool::create_translators(size_t num_translators_per_device,
                                          size_t num_threads_per_translator,
                                          models::ModelReader& model_reader,
                                          const Device device,
                                          std::vector<int> device_indices,
                                          const ComputeType compute_type) {
    if (device_indices.empty())
      throw std::invalid_argument("At least one device index should be set");

    if (device == Device::CUDA) {
      // Most computation will run on GPU so multiple CPU computation threads are not useful.
      num_threads_per_translator = 1;

      if (!have_same_compute_capability(device_indices))
        throw std::invalid_argument("All GPU used in parallel must have the same Compute Capability");
    }

    // Repeat each device index by the number of translators running on each device.
    device_indices = repeat_elements(device_indices, num_translators_per_device);

    // The same number of OpenMP threads should be used for loading and running model.
    set_num_threads(num_threads_per_translator);
    const auto models = models::load_replicas(model_reader, device, device_indices, compute_type);

    static const int core_offset = read_int_from_env("CT2_TRANSLATORS_CORE_OFFSET", -1);

    const size_t num_translators = models.size();
    std::vector<std::unique_ptr<Worker>> workers;
    workers.reserve(num_translators);
    _translators.reserve(num_translators);
    for (size_t i = 0; i < num_translators; ++i) {
      const auto& model = models[i];
      _translators.emplace_back(model);
      workers.emplace_back(std::make_unique<TranslatorWorker>(_translators.back(),
                                                              num_threads_per_translator));
    }

    _thread_pool = std::make_unique<ThreadPool>(std::move(workers),
                                                2 * num_translators,
                                                core_offset);
  }

  TranslatorPool::TranslateJob::TranslateJob(Batch batch,
                                             TranslationOptions options,
                                             std::shared_ptr<JobResultConsumer<TranslationResult>> consumer)
    : BatchJob(std::move(batch), std::move(consumer))
    , _options(options)
  {
  }

  std::vector<TranslationResult>
  TranslatorPool::TranslateJob::get_results() const {
    spdlog::debug("Running batch translation on {} examples", _batch.num_examples());
    auto results = TranslatorPool::get_translator()->translate_batch_with_prefix(_batch.get_stream(0),
                                                                                 _batch.get_stream(1),
                                                                                 _options);
    spdlog::debug("Finished batch translation");
    return results;
  }

  TranslatorPool::ScoreJob::ScoreJob(Batch batch,
                                     ScoringOptions options,
                                     std::shared_ptr<JobResultConsumer<ScoringResult>> consumer)
    : BatchJob(std::move(batch), std::move(consumer))
    , _options(options)
  {
  }

  std::vector<ScoringResult>
  TranslatorPool::ScoreJob::get_results() const {
    spdlog::debug("Running batch scoring on {} examples", _batch.num_examples());
    auto results = TranslatorPool::get_translator()->score_batch(_batch.get_stream(0),
                                                                 _batch.get_stream(1),
                                                                 _options);
    spdlog::debug("Finished batch scoring");
    return results;
  }

  TranslationStats TranslatorPool::consume_text_file(const std::string& source_file,
                                                     const std::string& output_file,
                                                     const TranslationOptions& options,
                                                     size_t max_batch_size,
                                                     size_t read_batch_size,
                                                     BatchType batch_type,
                                                     bool with_scores,
                                                     const std::string* target_file) {
    auto source = open_file<std::ifstream>(source_file);
    auto output = open_file<std::ofstream>(output_file);
    auto target = (target_file
                   ? std::make_unique<std::ifstream>(open_file<std::ifstream>(*target_file))
                   : nullptr);

    return consume_text_file(source,
                             output,
                             options,
                             max_batch_size,
                             read_batch_size,
                             batch_type,
                             with_scores,
                             target.get());
  }

  static std::vector<std::string> split_tokens(const std::string& text) {
    return split_string(text, ' ');
  }

  static std::string join_tokens(const std::vector<std::string>& tokens) {
    std::string text;
    for (const auto& token : tokens) {
      if (!text.empty())
        text += ' ';
      text += token;
    }
    return text;
  }

  TranslationStats TranslatorPool::consume_text_file(std::istream& source,
                                                     std::ostream& output,
                                                     const TranslationOptions& options,
                                                     size_t max_batch_size,
                                                     size_t read_batch_size,
                                                     BatchType batch_type,
                                                     bool with_scores,
                                                     std::istream* target) {
    return consume_raw_text_file(source,
                                 target,
                                 output,
                                 split_tokens,
                                 split_tokens,
                                 join_tokens,
                                 options,
                                 max_batch_size,
                                 read_batch_size,
                                 batch_type,
                                 with_scores);
  }

  TranslationStats TranslatorPool::score_text_file(const std::string& source_file,
                                                   const std::string& target_file,
                                                   const std::string& output_file,
                                                   const ScoringOptions& options,
                                                   size_t max_batch_size,
                                                   size_t read_batch_size,
                                                   BatchType batch_type,
                                                   bool with_tokens_score) {
    auto source = open_file<std::ifstream>(source_file);
    auto target = open_file<std::ifstream>(target_file);
    auto output = open_file<std::ofstream>(output_file);
    return score_text_file(source,
                           target,
                           output,
                           options,
                           max_batch_size,
                           read_batch_size,
                           batch_type,
                           with_tokens_score);
  }

  TranslationStats TranslatorPool::score_text_file(std::istream& source,
                                                   std::istream& target,
                                                   std::ostream& output,
                                                   const ScoringOptions& options,
                                                   size_t max_batch_size,
                                                   size_t read_batch_size,
                                                   BatchType batch_type,
                                                   bool with_tokens_score) {
    return score_raw_text_file(source,
                               target,
                               output,
                               split_tokens,
                               split_tokens,
                               join_tokens,
                               options,
                               max_batch_size,
                               read_batch_size,
                               batch_type,
                               with_tokens_score);
  }

  size_t TranslatorPool::num_queued_batches() {
    return _thread_pool->num_queued_jobs();
  }

  size_t TranslatorPool::num_active_batches() const {
    return _thread_pool->num_active_jobs();
  }

  size_t TranslatorPool::num_translators() const {
    return _translators.size();
  }

  const std::vector<Translator>& TranslatorPool::get_translators() const {
    return _translators;
  }

  static thread_local Translator* local_translator = nullptr;

  Translator* TranslatorPool::get_translator() {
    return local_translator;
  }


  TranslatorPool::TranslatorWorker::TranslatorWorker(Translator& translator, size_t num_threads)
    : _translator(translator)
    , _num_threads(num_threads)
  {
  }

  void TranslatorPool::TranslatorWorker::initialize() {
    // Set the number of OpenMP threads for the current thread.
    set_num_threads(_num_threads);
    local_translator = &_translator;
  }

  void TranslatorPool::TranslatorWorker::finalize() {
    // The CUDA context is destroyed when the thread exits, so we clear the translation
    // resources now when the CUDA context is still active.
    _translator.detach_model();
  }

}
