#pragma once

#include "replica_pool.h"
#include "models/language_model.h"

namespace ctranslate2 {

  // Encoder is the high-level class to embed texts with language models.
  // It supports parallel and asynchronous generation.
  class Encoder : public ReplicaPool<models::SequenceEncoderReplica> {
  public:
    using ReplicaPool::ReplicaPool;

    std::future<EncoderForwardOutput>
    forward_batch_async(std::vector<std::vector<std::string>> tokens);

    std::future<EncoderForwardOutput>
    forward_batch_async(std::vector<std::vector<size_t>> ids);

    std::future<EncoderForwardOutput>
    forward_batch_async(StorageView ids, StorageView lengths);
  };

}
