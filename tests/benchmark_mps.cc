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
  if (requested == "all" || requested == "prefill")
    benchmark_prefill_gemm();
  if (requested == "all" || requested == "topk")
    benchmark_topk();
  if (requested == "all" || requested == "concat")
    benchmark_concat();
  return 0;
}
