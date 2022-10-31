#pragma once

#include <fstream>

#include "replica_pool.h"
#include "models/sequence_to_sequence.h"

namespace ctranslate2 {

  struct TranslationStats {
    size_t num_tokens = 0;
    size_t num_examples = 0;
    double total_time_in_ms = 0;
  };

  std::vector<ScoringResult>
  run_scoring(models::SequenceToSequenceReplica& translator,
              const Batch& batch,
              const ScoringOptions& options);
  std::vector<TranslationResult>
  run_translation(models::SequenceToSequenceReplica& translator,
                  const Batch& batch,
                  const TranslationOptions& options);

  // TranslatorPool is the high-level class for running translations. It supports parallel
  // and asynchronous translations.
  class TranslatorPool : public ReplicaPool<models::SequenceToSequenceReplica> {
  public:
    // num_threads_per_translator (a.k.a. intra_threads) is forced to 1 when the translator
    // is running on a CUDA device.
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device = Device::CPU,
                   const int device_index = 0,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    // Constructor with ModelReader.
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   models::ModelReader& model_reader,
                   const Device device,
                   const int device_index,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    // Multi-device constructor.
    TranslatorPool(size_t num_translators_per_device,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device,
                   const std::vector<int>& device_indices,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    // Multi-device constructor with ModelReader.
    TranslatorPool(size_t num_translators_per_device,
                   size_t num_threads_per_translator,
                   models::ModelReader& model_reader,
                   const Device device,
                   const std::vector<int>& device_indices,
                   const ComputeType compute_type = ComputeType::DEFAULT,
                   const long max_queued_batches = 0);

    std::vector<std::future<TranslationResult>>
    translate_batch_async(const std::vector<std::vector<std::string>>& source,
                          const TranslationOptions& options = TranslationOptions(),
                          const size_t max_batch_size = 0,
                          const BatchType batch_type = BatchType::Examples);
    std::vector<std::future<TranslationResult>>
    translate_batch_async(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target_prefix,
                          const TranslationOptions& options = TranslationOptions(),
                          const size_t max_batch_size = 0,
                          const BatchType batch_type = BatchType::Examples);

    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const TranslationOptions& options = TranslationOptions(),
                    const size_t max_batch_size = 0,
                    const BatchType batch_type = BatchType::Examples);
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const std::vector<std::vector<std::string>>& target_prefix,
                    const TranslationOptions& options = TranslationOptions(),
                    const size_t max_batch_size = 0,
                    const BatchType batch_type = BatchType::Examples);

    std::vector<std::future<ScoringResult>>
    score_batch_async(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target,
                      const ScoringOptions& options = ScoringOptions(),
                      const size_t max_batch_size = 0,
                      const BatchType batch_type = BatchType::Examples);
    std::vector<ScoringResult>
    score_batch(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                const ScoringOptions& options = ScoringOptions(),
                const size_t max_batch_size = 0,
                const BatchType batch_type = BatchType::Examples);

    // Translate a stream.
    // The reader and writer functions do not need to be thread-safe.
    template <typename Reader, typename Writer>
    void translate_stream(std::istream& in,
                          std::ostream& out,
                          Reader& reader,
                          Writer& writer,
                          const TranslationOptions& options = TranslationOptions(),
                          size_t max_batch_size = 32,
                          size_t read_batch_size = 0,
                          BatchType batch_type = BatchType::Examples) {
      return translate_stream(in,
                              nullptr,
                              out,
                              reader,
                              nullptr,
                              writer,
                              options,
                              max_batch_size,
                              read_batch_size,
                              batch_type);
    }

    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void translate_stream(std::istream& source,
                          std::istream* target,
                          std::ostream& output,
                          SourceReader& source_reader,
                          TargetReader* target_reader,
                          TargetWriter& target_writer,
                          const TranslationOptions& options = TranslationOptions(),
                          size_t max_batch_size = 32,
                          size_t read_batch_size = 0,
                          BatchType batch_type = BatchType::Examples) {
      consume_stream<TranslationResult>(
        source,
        target,
        output,
        source_reader,
        target_reader,
        target_writer,
        max_batch_size,
        read_batch_size,
        batch_type,
        [options](models::SequenceToSequenceReplica& model, const Batch& batch) {
          return run_translation(model, batch, options);
        });
    }

    // Translate a file.
    // These are wrappers around consume_stream that set the appropriate reader and writer.
    TranslationStats translate_text_file(const std::string& source_file,
                                         const std::string& output_file,
                                         const TranslationOptions& options = TranslationOptions(),
                                         size_t max_batch_size = 32,
                                         size_t read_batch_size = 0,
                                         BatchType batch_type = BatchType::Examples,
                                         bool with_scores = false,
                                         const std::string* target_file = nullptr);

    TranslationStats translate_text_file(std::istream& source,
                                         std::ostream& output,
                                         const TranslationOptions& options = TranslationOptions(),
                                         size_t max_batch_size = 32,
                                         size_t read_batch_size = 0,
                                         BatchType batch_type = BatchType::Examples,
                                         bool with_scores = false,
                                         std::istream* target = nullptr);

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats translate_raw_text_file(const std::string& in_file,
                                             const std::string& out_file,
                                             Tokenizer& tokenizer,
                                             Detokenizer& detokenizer,
                                             const TranslationOptions& options = TranslationOptions(),
                                             const size_t max_batch_size = 32,
                                             const size_t read_batch_size = 0,
                                             const BatchType batch_type = BatchType::Examples,
                                             const bool with_scores = false) {
      auto in = open_file<std::ifstream>(in_file);
      auto out = open_file<std::ofstream>(out_file);
      return translate_raw_text_file(in,
                                     out,
                                     tokenizer,
                                     detokenizer,
                                     options,
                                     max_batch_size,
                                     read_batch_size,
                                     batch_type,
                                     with_scores);

    }

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats translate_raw_text_file(std::istream& in,
                                             std::ostream& out,
                                             Tokenizer& tokenizer,
                                             Detokenizer& detokenizer,
                                             const TranslationOptions& options = TranslationOptions(),
                                             const size_t max_batch_size = 32,
                                             const size_t read_batch_size = 0,
                                             const BatchType batch_type = BatchType::Examples,
                                             const bool with_scores = false) {
      return translate_raw_text_file(in,
                                     nullptr,
                                     out,
                                     tokenizer,
                                     tokenizer,
                                     detokenizer,
                                     options,
                                     max_batch_size,
                                     read_batch_size,
                                     batch_type,
                                     with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats translate_raw_text_file(const std::string& source_file,
                                             const std::string* target_file,
                                             const std::string& output_file,
                                             SourceTokenizer& source_tokenizer,
                                             TargetTokenizer& target_tokenizer,
                                             TargetDetokenizer& detokenizer,
                                             const TranslationOptions& options = TranslationOptions(),
                                             const size_t max_batch_size = 32,
                                             const size_t read_batch_size = 0,
                                             const BatchType batch_type = BatchType::Examples,
                                             const bool with_scores = false) {
      auto source = open_file<std::ifstream>(source_file);
      auto output = open_file<std::ofstream>(output_file);
      auto target = (target_file
                     ? std::make_unique<std::ifstream>(open_file<std::ifstream>(*target_file))
                     : nullptr);

      return translate_raw_text_file(source,
                                     target.get(),
                                     output,
                                     source_tokenizer,
                                     target_tokenizer,
                                     detokenizer,
                                     options,
                                     max_batch_size,
                                     read_batch_size,
                                     batch_type,
                                     with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats translate_raw_text_file(std::istream& source,
                                             std::istream* target,
                                             std::ostream& output,
                                             SourceTokenizer& source_tokenizer,
                                             TargetTokenizer& target_tokenizer,
                                             TargetDetokenizer& detokenizer,
                                             const TranslationOptions& options = TranslationOptions(),
                                             const size_t max_batch_size = 32,
                                             const size_t read_batch_size = 0,
                                             const BatchType batch_type = BatchType::Examples,
                                             const bool with_scores = false) {
      TranslationStats stats;

      TextLineReader<SourceTokenizer> source_reader(source_tokenizer);
      TextLineReader<TargetTokenizer> target_reader(target_tokenizer);

      auto writer = [&detokenizer, &stats, &output, &with_scores](const TranslationResult& result) {
        const auto& hypotheses = result.hypotheses;
        const auto& scores = result.scores;
        stats.num_examples += 1;
        stats.num_tokens += hypotheses[0].size();
        for (size_t n = 0; n < hypotheses.size(); ++n) {
          if (with_scores)
            output << (result.has_scores() ? scores[n] : 0) << " ||| ";
          output << detokenizer(hypotheses[n]) << '\n';
        }
      };

      const auto t1 = std::chrono::high_resolution_clock::now();

      translate_stream(source,
                       target,
                       output,
                       source_reader,
                       &target_reader,
                       writer,
                       options,
                       max_batch_size,
                       read_batch_size,
                       batch_type);

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    // Score a stream.
    // The reader and writer functions do not need to be thread-safe.
    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void score_stream(std::istream& source,
                      std::istream& target,
                      std::ostream& output,
                      SourceReader& source_reader,
                      TargetReader& target_reader,
                      TargetWriter& target_writer,
                      const ScoringOptions& options = ScoringOptions(),
                      size_t max_batch_size = 32,
                      size_t read_batch_size = 0,
                      BatchType batch_type = BatchType::Examples) {
      consume_stream<ScoringResult>(
        source,
        &target,
        output,
        source_reader,
        &target_reader,
        target_writer,
        max_batch_size,
        read_batch_size,
        batch_type,
        [options](models::SequenceToSequenceReplica& model, const Batch& batch) {
          return run_scoring(model, batch, options);
        });
    }

    // Score a file.
    // These are wrappers around score_stream that set the appropriate reader and writer.
    TranslationStats score_text_file(const std::string& source_file,
                                     const std::string& target_file,
                                     const std::string& output_file,
                                     const ScoringOptions& options = ScoringOptions(),
                                     size_t max_batch_size = 32,
                                     size_t read_batch_size = 0,
                                     BatchType batch_type = BatchType::Examples,
                                     bool with_tokens_score = false);
    TranslationStats score_text_file(std::istream& source,
                                     std::istream& target,
                                     std::ostream& output,
                                     const ScoringOptions& options = ScoringOptions(),
                                     size_t max_batch_size = 32,
                                     size_t read_batch_size = 0,
                                     BatchType batch_type = BatchType::Examples,
                                     bool with_tokens_score = false);

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats score_raw_text_file(const std::string& source_file,
                                         const std::string& target_file,
                                         const std::string& output_file,
                                         SourceTokenizer& source_tokenizer,
                                         TargetTokenizer& target_tokenizer,
                                         TargetDetokenizer& target_detokenizer,
                                         const ScoringOptions& options = ScoringOptions(),
                                         const size_t max_batch_size = 32,
                                         const size_t read_batch_size = 0,
                                         const BatchType batch_type = BatchType::Examples,
                                         bool with_tokens_score = false) {
      auto source = open_file<std::ifstream>(source_file);
      auto target = open_file<std::ifstream>(target_file);
      auto output = open_file<std::ofstream>(output_file);
      return score_raw_text_file(source,
                                 target,
                                 output,
                                 source_tokenizer,
                                 target_tokenizer,
                                 target_detokenizer,
                                 options,
                                 max_batch_size,
                                 read_batch_size,
                                 batch_type,
                                 with_tokens_score);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats score_raw_text_file(std::istream& source,
                                         std::istream& target,
                                         std::ostream& output,
                                         SourceTokenizer& source_tokenizer,
                                         TargetTokenizer& target_tokenizer,
                                         TargetDetokenizer& target_detokenizer,
                                         const ScoringOptions& options = ScoringOptions(),
                                         const size_t max_batch_size = 32,
                                         const size_t read_batch_size = 0,
                                         const BatchType batch_type = BatchType::Examples,
                                         bool with_token_scores = false) {
      TextLineReader<SourceTokenizer> source_reader(source_tokenizer);
      TextLineReader<TargetTokenizer> target_reader(target_tokenizer);
      TranslationStats stats;

      auto writer = [&target_detokenizer, &stats, &output, with_token_scores](const ScoringResult& result) {
        stats.num_examples += 1;
        stats.num_tokens += result.tokens_score.size();
        output << result.normalized_score() << " ||| " << target_detokenizer(result.tokens);
        if (with_token_scores) {
          output << " |||";
          for (const auto score : result.tokens_score)
            output << ' ' << score;
        }
        output << '\n';
      };

      const auto t1 = std::chrono::high_resolution_clock::now();
      score_stream(source,
                   target,
                   output,
                   source_reader,
                   target_reader,
                   writer,
                   options,
                   max_batch_size,
                   read_batch_size,
                   batch_type);
      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    size_t num_translators() const;

  private:
    friend class BufferedTranslationWrapper;

    template <typename Result,
              typename SourceReader,
              typename TargetReader,
              typename TargetWriter,
              typename Func>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        SourceReader& source_reader,
                        TargetReader* target_reader,
                        TargetWriter& target_writer,
                        size_t max_batch_size,
                        size_t read_batch_size,
                        BatchType batch_type,
                        const Func& func) {
      std::unique_ptr<BatchReader> batch_reader;
      if (target) {
        auto parallel_reader = std::make_unique<ParallelBatchReader>();
        parallel_reader->add(std::make_unique<StreamReader<SourceReader>>(source, source_reader));
        parallel_reader->add(std::make_unique<StreamReader<TargetReader>>(*target, *target_reader));
        batch_reader = std::move(parallel_reader);
      } else {
        batch_reader = std::make_unique<StreamReader<SourceReader>>(source, source_reader);
      }

      consume_batches<Result>(*batch_reader,
                              target_writer,
                              func,
                              max_batch_size,
                              read_batch_size,
                              batch_type);

      output.flush();
    }
  };

}
