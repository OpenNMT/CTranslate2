#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ctranslate2/devices.h"
#include "ctranslate2/ops/ops.h"

using namespace ctranslate2;

namespace {

size_t samples_from_env(size_t default_value) {
  const char* value = std::getenv("CT2_MPS_BENCH_SAMPLES");
  if (!value || value[0] == '\0')
    return default_value;
  return std::max<size_t>(1, std::strtoull(value, nullptr, 10));
}

template <typename Function>
double benchmark(Function&& function, size_t samples) {
  for (size_t i = 0; i < 5; ++i)
    function();
  synchronize_device(Device::MPS, 0);

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < samples; ++i)
    function();
  synchronize_device(Device::MPS, 0);
  const auto end = std::chrono::steady_clock::now();
  const auto microseconds =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  return static_cast<double>(microseconds) / static_cast<double>(samples) / 1000.0;
}

void print_result(const std::string& name, double milliseconds, size_t samples) {
  std::cout << std::left << std::setw(24) << name
            << " avg_ms=" << std::fixed << std::setprecision(6) << milliseconds
            << " samples=" << samples << '\n';
}

void benchmark_decode_gemm() {
  const size_t samples = samples_from_env(200);
  StorageView input({1, 512}, float16_t(0.01f), Device::MPS);
  StorageView weight({2048, 512}, float16_t(0.01f), Device::MPS);
  StorageView output(DataType::FLOAT16, Device::MPS);
  const ops::Gemm gemm(1, 0, false, true);
  print_result("fp16_decode_gemm",
               benchmark([&]() { gemm(input, weight, output); }, samples),
               samples);
}

void benchmark_vocab_gemm() {
  const size_t samples = samples_from_env(50);
  StorageView input({1, 512}, float16_t(0.01f), Device::MPS);
  StorageView weight({64176, 512}, float16_t(0.01f), Device::MPS);
  StorageView output(DataType::FLOAT16, Device::MPS);
  const ops::Gemm gemm(1, 0, false, true);
  print_result("fp16_vocab_gemm",
               benchmark([&]() { gemm(input, weight, output); }, samples),
               samples);
}

void benchmark_whisper_decode_gemms() {
  const size_t samples = samples_from_env(100);
  constexpr dim_t k = 1024;
  for (const dim_t n : {dim_t(1024), dim_t(3072), dim_t(4096), dim_t(51865)}) {
    StorageView input({1, k}, float16_t(0.01f), Device::MPS);
    StorageView weight({n, k}, float16_t(0.01f), Device::MPS);
    StorageView output(DataType::FLOAT16, Device::MPS);
    const ops::Gemm gemm(1, 0, false, true);
    print_result("whisper_gemv_1x" + std::to_string(n),
                 benchmark([&]() { gemm(input, weight, output); }, samples),
                 samples);
  }
}

void benchmark_prefill_gemm() {
  const size_t samples = samples_from_env(30);
  StorageView input({128, 512}, float16_t(0.01f), Device::MPS);
  StorageView weight({2048, 512}, float16_t(0.01f), Device::MPS);
  StorageView output(DataType::FLOAT16, Device::MPS);
  const ops::Gemm gemm(1, 0, false, true);
  print_result("fp16_prefill_gemm",
               benchmark([&]() { gemm(input, weight, output); }, samples),
               samples);
}

void benchmark_marian_batch_gemms() {
  const size_t samples = samples_from_env(50);
  const std::vector<dim_t> output_sizes = {512, 1536, 2048, 58104};
  for (const dim_t m : {dim_t(4), dim_t(128)}) {
    for (const dim_t n : output_sizes) {
      StorageView input({m, 512}, float16_t(0.01f), Device::MPS);
      StorageView weight({n, 512}, float16_t(0.01f), Device::MPS);
      StorageView output(DataType::FLOAT16, Device::MPS);
      const ops::Gemm gemm(1, 0, false, true);
      print_result("fp16_gemm_" + std::to_string(m) + "x" + std::to_string(n),
                   benchmark([&]() { gemm(input, weight, output); }, samples),
                   samples);
    }
  }

  const ops::ActivationType swish = ops::ActivationType::Swish;
  for (const dim_t n : {dim_t(512), dim_t(1536), dim_t(2048)}) {
    StorageView input({128, 512}, float16_t(0.01f), Device::MPS);
    StorageView weight({n, 512}, float16_t(0.01f), Device::MPS);
    StorageView bias({n}, float16_t(0.01f), Device::MPS);
    StorageView output(DataType::FLOAT16, Device::MPS);
    const ops::Gemm gemm(1, 0, false, true, false, false,
                         n == 2048 ? &swish : nullptr);
    print_result("fp16_epilogue_128x" + std::to_string(n),
                 benchmark([&]() { gemm(input, weight, output, nullptr, &bias); }, samples),
                 samples);
  }
}

void benchmark_topk() {
  const size_t samples = samples_from_env(300);
  StorageView input({1, 51865}, float16_t(0.01f), Device::MPS);
  StorageView values(DataType::FLOAT16, Device::MPS);
  StorageView indices(DataType::INT32, Device::MPS);
  const ops::TopK topk(1);
  print_result("fp16_argmax",
               benchmark([&]() { topk(input, values, indices); }, samples),
               samples);
}

void benchmark_beam_topk() {
  const size_t samples = samples_from_env(50);
  constexpr dim_t batch_size = 32;
  constexpr dim_t beam_size = 4;
  constexpr dim_t vocabulary_size = 58104;
  constexpr dim_t candidates = 8;
  StorageView input({batch_size, beam_size * vocabulary_size},
                    float16_t(0.01f),
                    Device::MPS);
  StorageView values(DataType::FLOAT16, Device::MPS);
  StorageView indices(DataType::INT32, Device::MPS);
  const ops::TopK topk(candidates);
  print_result("fp16_beam_topk_b32",
               benchmark([&]() { topk(input, values, indices); }, samples),
               samples);
}

void benchmark_beam_logsoftmax() {
  const size_t samples = samples_from_env(50);
  constexpr dim_t rows = 128;
  constexpr dim_t vocabulary_size = 58104;
  StorageView input({rows, vocabulary_size}, float16_t(0.01f), Device::MPS);
  print_result("fp16_logsoftmax_128x58k",
               benchmark([&]() { ops::LogSoftMax()(input); }, samples),
               samples);
}

void benchmark_batched_vocab_gemm() {
  const size_t samples = samples_from_env(20);
  StorageView input({128, 512}, float16_t(0.01f), Device::MPS);
  StorageView weight({58104, 512}, float16_t(0.01f), Device::MPS);
  StorageView output(DataType::FLOAT16, Device::MPS);
  const ops::Gemm gemm(1, 0, false, true);
  print_result("fp16_vocab_gemm_m128",
               benchmark([&]() { gemm(input, weight, output); }, samples),
               samples);
}

void benchmark_concat() {
  const size_t samples = samples_from_env(500);
  std::vector<StorageView> storage;
  std::vector<const StorageView*> inputs;
  storage.reserve(8);
  inputs.reserve(8);
  for (size_t i = 0; i < 8; ++i) {
    storage.emplace_back(Shape{4, 128}, float16_t(i), Device::MPS);
    inputs.push_back(&storage.back());
  }
  StorageView output(DataType::FLOAT16, Device::MPS);
  const ops::Concat concat(0);
  print_result("fp16_concat_8way",
               benchmark([&]() { concat(inputs, output); }, samples),
               samples);
}

}  // namespace

int main(int argc, char* argv[]) {
  const std::string requested = argc > 1 ? argv[1] : "all";
  if (requested == "all" || requested == "decode")
    benchmark_decode_gemm();
  if (requested == "all" || requested == "vocab")
    benchmark_vocab_gemm();
  if (requested == "all" || requested == "whisper-decode")
    benchmark_whisper_decode_gemms();
  if (requested == "all" || requested == "prefill")
    benchmark_prefill_gemm();
  if (requested == "all" || requested == "marian-batch")
    benchmark_marian_batch_gemms();
  if (requested == "all" || requested == "topk")
    benchmark_topk();
  if (requested == "all" || requested == "beam") {
    benchmark_batched_vocab_gemm();
    benchmark_beam_logsoftmax();
    benchmark_beam_topk();
  }
  if (requested == "all" || requested == "concat")
    benchmark_concat();
  return 0;
}
