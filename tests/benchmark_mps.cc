#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ctranslate2/devices.h"
#include "ctranslate2/ops/ops.h"
#include "mps/kernels.h"
#include "mps/moe.h"

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

void benchmark_flash_decode() {
  const size_t samples = samples_from_env(100);
  constexpr dim_t heads = 16, kv_heads = 8, depth = 128;
  for (dim_t keys :
       {dim_t(512), dim_t(2048), dim_t(4096), dim_t(8192), dim_t(16384)}) {
    StorageView q({1, heads, 1, depth}, float16_t(.01f), Device::MPS);
    StorageView k({1, heads, keys, depth}, float16_t(.02f), Device::MPS);
    StorageView v(k.shape(), float16_t(.03f), Device::MPS);
    StorageView scores(DataType::FLOAT16, Device::MPS), output(DataType::FLOAT16, Device::MPS);
    const float scale = 1.f / std::sqrt(float(depth));
    print_result("generic_attention_" + std::to_string(keys), benchmark([&]() {
      ops::MatMul(false, true, scale)(q, k, scores);
      ops::SoftMax()(scores, scores);
      ops::MatMul()(scores, v, output);
    }, samples), samples);
    StorageView compact_k({1, keys, kv_heads, depth}, float16_t(.02f), Device::MPS);
    StorageView compact_v(compact_k.shape(), float16_t(.03f), Device::MPS);
    StorageView partials({1, heads, (keys+255)/256, depth+2}, DataType::FLOAT32, Device::MPS);
    output.resize_as(q);
    print_result("native_flash_decode_" + std::to_string(keys), benchmark([&]() {
      if (!mps::flash_decode(DataType::FLOAT16, q.buffer(), compact_k.buffer(), compact_v.buffer(),
                             nullptr, output.buffer(), partials.data<float>(), partials.size(),
                             1, heads, kv_heads, keys, keys, depth, scale, keys-1))
        throw std::runtime_error("Flash decode shape not supported on this device");
    }, samples), samples);
    for (unsigned bits : {8u, 5u, 4u}) {
      StorageView packed_k(
          {1, keys, kv_heads, mps::packed_kv_row_bytes(depth, bits)},
          DataType::INT8, Device::MPS);
      StorageView packed_v(packed_k.shape(), DataType::INT8, Device::MPS);
      mps::append_quantized_kv(compact_k.buffer(), compact_v.buffer(), nullptr,
                               nullptr, packed_k.buffer(), packed_v.buffer(), 1,
                               keys, kv_heads, depth, 0, 0, keys, true, bits);
      print_result(
          "flash_int" + std::to_string(bits) + "_" + std::to_string(keys),
          benchmark(
              [&]() {
                if (!mps::flash_decode(
                        DataType::FLOAT16, q.buffer(), packed_k.buffer(),
                        packed_v.buffer(), nullptr, output.buffer(),
                        partials.data<float>(), partials.size(), 1, heads,
                        kv_heads, keys, keys, depth, scale, keys - 1, 0, true,
                        1, true, true, bits))
                  throw std::runtime_error(
                      "Quantized cache decode unsupported");
              },
              samples),
          samples);
    }
  }
}

void benchmark_moe_topk() {
  const size_t samples = samples_from_env(100);
  for (dim_t experts : {dim_t(4), dim_t(8), dim_t(128)})
    for (dim_t rows : {dim_t(1), dim_t(32), dim_t(128), dim_t(512), dim_t(2048)}) {
      StorageView input({rows, experts});
      for (dim_t i = 0; i < input.size(); ++i)
        input.data<float>()[i] = std::sin(float(i) * .17f);
      auto gpu = input.to(Device::MPS);
      StorageView ids(DataType::INT32, Device::MPS), weights(DataType::FLOAT32, Device::MPS);
      ops::MoeTopK route(2);
      print_result("moe_top2_" + std::to_string(rows) + "x" + std::to_string(experts),
                   benchmark([&]() { route(gpu, ids, weights); }, samples), samples);
    }
}

// Geometry of the pinned Qwen3 MoE used by the end-to-end benchmark. This
// includes planning, both expert projections, SwiGLU and ordered aggregation.
void benchmark_moe_experts() {
  constexpr dim_t experts = 4, hidden = 1024, intermediate = 3072, topk = 2;
  const size_t samples = samples_from_env(10);
  std::vector<std::array<StorageView, 3>> storage;
  storage.reserve(experts);
  for (dim_t e = 0; e < experts; ++e)
    storage.push_back({StorageView({intermediate, hidden}, float16_t(.001f), Device::MPS),
                       StorageView({intermediate, hidden}, float16_t(.002f), Device::MPS),
                       StorageView({hidden, intermediate}, float16_t(.003f), Device::MPS)});
  std::vector<std::array<const StorageView*, 3>> weights;
  for (const auto& expert : storage)
    weights.push_back({&expert[0], &expert[1], &expert[2]});
  for (dim_t rows : {1, 8, 31, 128, 512, 2048}) {
    StorageView x({rows, hidden}, float16_t(.01f), Device::MPS);
    std::vector<int32_t> selected(rows * topk);
    for (dim_t r = 0; r < rows; ++r) {
      selected[r * topk] = r % experts;
      selected[r * topk + 1] = (r + 1) % experts;
    }
    StorageView ids({rows, topk}, selected, Device::MPS),
        probabilities({rows, topk}, .5f, Device::MPS),
        output(DataType::FLOAT16, Device::MPS);
    print_result("moe_experts_" + std::to_string(rows), benchmark([&]() {
      mps::moe_forward(x, ids, probabilities, weights, output);
    }, samples), samples);
    if (rows <= 8)
      print_result("moe_plan_legacy_" + std::to_string(rows), benchmark([&]() {
        mps::moe_forward(x, ids, probabilities, weights, output, false, false, nullptr, true);
      }, samples), samples);
  }
}

void benchmark_moe_fixture(const char* path) {
  std::ifstream file(path, std::ios::binary);
  char magic[8];
  uint32_t dimensions[3];
  file.read(magic, sizeof(magic));
  file.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
  if (!file || std::string(magic, 8) != std::string("CT2MOE1\0", 8))
    throw std::invalid_argument("Invalid MoE router fixture header");
  const dim_t rows = dimensions[0], experts = dimensions[1], k = dimensions[2];
  if (rows <= 0 || experts <= 0 || experts > 4096 || k <= 0 || k > experts
      || uint64_t(rows) * experts > 100000000)
    throw std::invalid_argument("Invalid MoE router fixture dimensions");
  StorageView input({rows, experts}), expected_ids({rows, k}, DataType::INT32),
      expected_weights({rows, k});
  file.read(reinterpret_cast<char*>(input.buffer()), input.size() * sizeof(float));
  file.read(reinterpret_cast<char*>(expected_ids.buffer()), expected_ids.size() * sizeof(int32_t));
  file.read(reinterpret_cast<char*>(expected_weights.buffer()), expected_weights.size() * sizeof(float));
  if (!file || file.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("Invalid MoE router fixture size");
  auto gpu = input.to(Device::MPS);
  StorageView ids(DataType::INT32, Device::MPS), weights(DataType::FLOAT32, Device::MPS);
  ops::MoeTopK route(k);
  route(gpu, ids, weights);
  const auto actual_ids = ids.to_vector<int32_t>();
  const auto actual_weights = weights.to_vector<float>();
  float max_error = 0;
  for (dim_t i = 0; i < expected_ids.size(); ++i) {
    if (actual_ids[i] != expected_ids.data<int32_t>()[i])
      throw std::runtime_error("Native MoE expert indices differ from source reference");
    const float error = std::abs(actual_weights[i] - expected_weights.data<float>()[i]);
    if (!std::isfinite(error))
      throw std::runtime_error("Nonfinite native MoE routing weights");
    max_error = std::max(max_error, error);
  }
  std::cout << "moe_fixture rows=" << rows << " experts=" << experts << " k=" << k
            << " max_weight_error=" << std::scientific << max_error << '\n';
  if (max_error > 1e-7f)
    throw std::runtime_error("Native MoE weights differ from source reference");
  const size_t samples = samples_from_env(100);
  print_result("real_router_topk_" + std::to_string(rows),
               benchmark([&]() { route(gpu, ids, weights); }, samples), samples);
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
  if (requested == "all" || requested == "flash-decode")
    benchmark_flash_decode();
  if (requested == "all" || requested == "marian-batch")
    benchmark_marian_batch_gemms();
  if (requested == "all" || requested == "moe-topk")
    benchmark_moe_topk();
  if (requested == "moe-experts")
    benchmark_moe_experts();
  if (requested == "moe-router-file") {
    if (argc != 3)
      throw std::invalid_argument("Usage: benchmark_mps moe-router-file fixture.bin");
    benchmark_moe_fixture(argv[2]);
  }
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
