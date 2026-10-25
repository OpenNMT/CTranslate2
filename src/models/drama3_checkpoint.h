#pragma once

#include "ctranslate2/models/model_reader.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace models {

    // Shared with model clones so borrowed byte tensors outlive their file reader.
    class Drama3CheckpointStorage {
    public:
      explicit Drama3CheckpointStorage(const std::string& directory);
      StorageView view(Shape shape, std::istream& stream, dim_t num_bytes) const;
      void release_pages(const StorageView& weight) const;

    private:
      size_t _size;
      std::shared_ptr<void> _mapping;
    };

    class Drama3CheckpointReader : public ModelFileReader {
    public:
      explicit Drama3CheckpointReader(const std::string& directory)
        : ModelFileReader(directory)
        , storage(std::make_shared<Drama3CheckpointStorage>(directory))
      {
      }

      const std::shared_ptr<Drama3CheckpointStorage> storage;
    };

  }
}
