// Minimal repro for the Windows Ruy shutdown deadlock (ctranslate2-rs#64 / #2076).
//
// It creates a CPU int8 Translator (Ruy backend) with worker threads, runs one
// translation, then destroys the Translator. On an unpatched build the per-thread
// ruy::Context destructor joins Ruy's internal thread pool from a worker thread
// that is exiting under the Windows loader lock, and that join deadlocks.
//
// The point of this repro: it only hangs when CTranslate2 is built with the STATIC
// CRT (/MT) and linked statically. Built with the dynamic CRT (/MD) as a shared
// library -- the configuration of the official wheels -- the same code shuts down
// cleanly. See README.md.
#include <ctranslate2/translator.h>
#include <ctranslate2/models/model.h>

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: repro <model_dir>\n";
    return 2;
  }

  ctranslate2::ReplicaPoolConfig pool_config;
  pool_config.num_threads_per_replica = 4;   // intra_threads: give Ruy a real thread pool

  ctranslate2::models::ModelLoader model_loader(argv[1]);
  model_loader.device = ctranslate2::Device::CPU;
  model_loader.compute_type = ctranslate2::ComputeType::INT8;
  model_loader.num_replicas_per_device = 2;  // inter_threads: use worker threads

  // A large batch so the int8 GEMM is big enough that Ruy actually spawns its
  // internal thread pool. That is a precondition for the hang: the destructor only
  // deadlocks if there are Ruy worker threads to join. A tiny batch runs
  // single-threaded and shuts down cleanly even unpatched.
  const std::vector<std::string> sentence = {"آ", "ت", "ز", "م", "و", "ن"};
  const std::vector<std::vector<std::string>> source(512, sentence);

  std::cout << "translating a batch of " << source.size() << "..." << std::endl;
  {
    ctranslate2::Translator translator(model_loader, pool_config);
    const auto results = translator.translate_batch(source);
    std::cout << "output[0]:";
    for (const auto& token : results[0].hypotheses[0])
      std::cout << ' ' << token;
    std::cout << "\ndestroying Translator (Ruy thread-pool join happens here)..."
              << std::endl;
  }  // <-- unpatched + static /MT: the join deadlocks here and the process hangs.
  std::cout << "SURVIVED: clean shutdown, no deadlock" << std::endl;
  return 0;
}
