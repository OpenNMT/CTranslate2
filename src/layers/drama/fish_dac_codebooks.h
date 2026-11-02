#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    // Reconstructs codec latents by projecting and summing audio codebook embeddings.
    // Decoder-only lookup: raw vectors -> biased projections -> sum. No encoder
    // nearest-neighbor normalization or residual quantization is applied.
    class FishDACCodebooks {
    public:
      FishDACCodebooks(const models::Model& model, const std::string& scope, dim_t num_books = 10);
      // Input INT32 [1,books,time], output FLOAT32 [1,channels,time], CPU.
      void operator()(const StorageView& codes, StorageView& output) const;

    private:
      void validate_codes(const StorageView& codes, const StorageView& output) const;

      // References one codebook embedding table and its output projection.
      struct Book {
        const StorageView* embedding;
        const StorageView* projection;
        const StorageView* bias;
      };

      std::vector<Book> _books;
      dim_t _channels;
    };

  }
}
