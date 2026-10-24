#include "drama3_checkpoint.h"

#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace ctranslate2 {
  namespace models {

    Drama3CheckpointStorage::Drama3CheckpointStorage(const std::string& directory) {
#if defined(__linux__) || defined(__APPLE__)
      const std::string filename = directory + "/model.bin";
      const int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd < 0)
        throw std::runtime_error("Unable to open Drama3 mapped weights");
      struct stat info;
      if (fstat(fd, &info) != 0 || info.st_size <= 0) {
        close(fd);
        throw std::runtime_error("Unable to stat Drama3 mapped weights");
      }
      _size = static_cast<size_t>(info.st_size);
      void* mapping = mmap(nullptr, _size, PROT_READ, MAP_PRIVATE, fd, 0);
      close(fd);
      if (mapping == MAP_FAILED)
        throw std::runtime_error("Unable to map Drama3 weights");
      const auto size = _size;
      _mapping = std::shared_ptr<void>(mapping, [size](void* p) { munmap(p, size); });
#else
      throw std::invalid_argument("Drama3 mapped expert weights require Linux or macOS");
#endif
    }

    StorageView Drama3CheckpointStorage::view(Shape shape, std::istream& stream,
                                             dim_t num_bytes) const {
      const auto position = stream.tellg();
      if (position == std::streampos(-1))
        throw std::runtime_error("Invalid mapped Drama3 weight offset");
      const auto offset = static_cast<size_t>(static_cast<std::streamoff>(position));
      if (offset > _size || static_cast<size_t>(num_bytes) > _size - offset)
        throw std::runtime_error("Drama3 weight exceeds the mapped file");
      // Binary payloads need not be float-aligned: only byte matrices are mapped.
      auto* data = static_cast<int8_t*>(_mapping.get()) + offset;
      stream.seekg(num_bytes, std::ios::cur);
      if (!stream)
        throw std::runtime_error("Unable to seek past mapped Drama3 expert weights");
      return StorageView(std::move(shape), data, Device::CPU);
    }

    void Drama3CheckpointStorage::release_pages(const StorageView& weight) const {
#if defined(__linux__) || defined(__APPLE__)
      if (!weight.owns_data()) {
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        const auto address = reinterpret_cast<uintptr_t>(weight.buffer());
        const auto start = address - address % page;
        const size_t length = address - start + weight.size() * weight.item_size();
        madvise(reinterpret_cast<void*>(start), length, MADV_DONTNEED);
      }
#endif
    }

  }
}
