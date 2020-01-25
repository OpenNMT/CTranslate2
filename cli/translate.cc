#include <chrono>
#include <fstream>
#include <iostream>

#include <cxxopts.hpp>

#include <ctranslate2/translator_pool.h>
#include <ctranslate2/utils.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/profiler.h>

int main(int argc, char* argv[]) {
  cxxopts::Options cmd_options("translate", "CTranslate2 translation client");
  cmd_options.add_options()
    ("help", "Display available options.")
    ("model", "Path to the CTranslate2 model directory.", cxxopts::value<std::string>())
    ("compute_type", "Force the model type as \"float\", \"int16\" or \"int8\"",
        cxxopts::value<std::string>()->default_value("default"))
        ("src", "Path to the file to translate (read from the standard input if not set).",
        cxxopts::value<std::string>())
    ("tgt", "Path to the output file (write to the standard output if not set.",
        cxxopts::value<std::string>())
    ("use_vmap", "Use the vocabulary map included in the model to restrict the target candidates.",
        cxxopts::value<bool>()->default_value("false"))
    ("batch_size", "Number of sentences to forward into the model at once.",
        cxxopts::value<size_t>()->default_value("30"))
    ("beam_size", "Beam search size (set 1 for greedy decoding).",
        cxxopts::value<size_t>()->default_value("5"))
    ("sampling_topk", "Sample randomly from the top K candidates.",
        cxxopts::value<size_t>()->default_value("1"))
    ("sampling_temperature", "Sampling temperature.",
        cxxopts::value<float>()->default_value("1"))
     ("n_best", "Also output the n-best hypotheses.",
         cxxopts::value<size_t>()->default_value("1"))
    ("with_score", "Also output translation scores.",
        cxxopts::value<bool>()->default_value("false"))
    ("length_penalty", "Length penalty to apply during beam search",
        cxxopts::value<float>()->default_value("0"))
    ("max_sent_length", "Maximum sentence length to produce.",
        cxxopts::value<size_t>()->default_value("250"))
    ("min_sent_length", "Minimum sentence length to produce.",
        cxxopts::value<size_t>()->default_value("1"))
    ("log_throughput", "Log average tokens per second at the end of the translation.",
        cxxopts::value<bool>()->default_value("false"))
    ("log_profiling", "Log execution profiling.",
        cxxopts::value<bool>()->default_value("false"))
    ("inter_threads", "Maximum number of translations to run in parallel.",
        cxxopts::value<size_t>()->default_value("1"))
    ("intra_threads", "Number of OpenMP threads (set to 0 to use the default value).",
        cxxopts::value<size_t>()->default_value("0"))
    ("device", "Device to use (can be cpu, cuda, auto).",
        cxxopts::value<std::string>()->default_value("cpu"))
    ("device_index", "Index of the device to use.",
        cxxopts::value<int>()->default_value("0"))
    ;

  auto result = cmd_options.parse(argc, argv);

  if (result.count("help")) {
    std::cerr << cmd_options.help() << std::endl;
    return 1;
  }
  if (!result.count("model")) {
    std::cerr << "missing model" << std::endl;
    return 1;
  }

  size_t inter_threads = result["inter_threads"].as<size_t>();
  size_t intra_threads = result["intra_threads"].as<size_t>();

  auto model = ctranslate2::models::Model::load(
    result["model"].as<std::string>(),
    result["device"].as<std::string>(),
    result["device_index"].as<int>(),
    result["compute_type"].as<std::string>());

  ctranslate2::TranslatorPool translator_pool(inter_threads, intra_threads, model);

  auto options = ctranslate2::TranslationOptions();
  options.beam_size = result["beam_size"].as<size_t>();
  options.length_penalty = result["length_penalty"].as<float>();
  options.sampling_topk = result["sampling_topk"].as<size_t>();
  options.sampling_temperature = result["sampling_temperature"].as<float>();
  options.max_decoding_length = result["max_sent_length"].as<size_t>();
  options.min_decoding_length = result["min_sent_length"].as<size_t>();
  options.num_hypotheses = result["n_best"].as<size_t>();
  options.use_vmap = result["use_vmap"].as<bool>();

  std::istream* in = &std::cin;
  std::ostream* out = &std::cout;
  if (result.count("src")) {
    auto path = result["src"].as<std::string>();
    auto src_file = new std::ifstream(path);
    if (!src_file->is_open())
      throw std::runtime_error("Unable to open input file " + path);
    in = src_file;
  }
  if (result.count("tgt")) {
    out = new std::ofstream(result["tgt"].as<std::string>());
  }

  auto log_profiling = result["log_profiling"].as<bool>();
  auto t1 = std::chrono::high_resolution_clock::now();
  if (log_profiling)
    ctranslate2::init_profiling(model->device(), inter_threads);
  auto num_tokens = translator_pool.consume_text_file(*in,
                                                      *out,
                                                      result["batch_size"].as<size_t>(),
                                                      options,
                                                      result["with_score"].as<bool>());
  if (log_profiling)
    ctranslate2::dump_profiling(std::cerr);
  auto t2 = std::chrono::high_resolution_clock::now();

  if (in != &std::cin)
    delete in;
  if (out != &std::cout)
    delete out;

  if (result["log_throughput"].as<bool>()) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  }

  return 0;
}
