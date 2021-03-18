#include "ctranslate2/allocator.h"

#include <cstdlib>
#include <memory>

#ifdef _WIN32
#  include <malloc.h>
#endif

#ifdef CT2_WITH_MKL
#  include <mkl.h>
#endif

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#  include <cub/util_allocator.cuh>
#endif

#include "ctranslate2/utils.h"
#include "device_dispatch.h"

namespace ctranslate2 {

  class AlignedAllocator : public Allocator {
  public:
    AlignedAllocator(size_t alignment)
      : _alignment(alignment)
    {
    }

    void* allocate(size_t size, int) override {
      void* ptr = nullptr;
#ifdef _WIN32
      ptr = _aligned_malloc(size, _alignment);
#else
      if (posix_memalign(&ptr, _alignment, size) != 0)
        ptr = nullptr;
#endif
      if (!ptr)
        throw std::runtime_error("aligned_alloc: failed to allocate memory");
      return ptr;
    }

    void free(void* ptr, int) override {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      std::free(ptr);
#endif
    }

  private:
    size_t _alignment;
  };

#ifdef CT2_WITH_MKL
  class MklAllocator : public Allocator {
  public:
    MklAllocator(size_t alignment)
      : _alignment(alignment)
    {
    }

    void* allocate(size_t size, int) override {
      void* ptr = mkl_malloc(size, _alignment);
      if (!ptr)
        throw std::runtime_error("mkl_malloc: failed to allocate memory");
      return ptr;
    }

    void free(void* ptr, int) override {
      mkl_free(ptr);
    }

    void clear_cache() override {
      mkl_free_buffers();
    }

  private:
    size_t _alignment;
  };
#endif

#ifdef CT2_WITH_CUDA
  // See https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html.
  class CubCachingAllocator : public Allocator {
  public:
    CubCachingAllocator() {
      unsigned int bin_growth = 4;
      unsigned int min_bin = 3;
      unsigned int max_bin = 12;
      size_t max_cached_bytes = 200 * (1 << 20);  // 200MB

      const char* config_env = std::getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG");
      if (config_env) {
        const std::vector<std::string> values = split_string(config_env, ',');
        if (values.size() != 4)
          throw std::invalid_argument("CT2_CUDA_CACHING_ALLOCATOR_CONFIG environment variable "
                                      "should have format: "
                                      "bin_growth,min_bin,max_bin,max_cached_bytes");
        bin_growth = std::stoul(values[0]);
        min_bin = std::stoul(values[1]);
        max_bin = std::stoul(values[2]);
        max_cached_bytes = std::stoull(values[3]);
      }

      _allocator.reset(new cub::CachingDeviceAllocator(bin_growth,
                                                       min_bin,
                                                       max_bin,
                                                       max_cached_bytes));
    }

    void* allocate(size_t size, int device_index) override {
      void* ptr = nullptr;
      CUDA_CHECK(_allocator->DeviceAllocate(device_index, &ptr, size, cuda::get_cuda_stream()));
      return ptr;
    }

    void free(void* ptr, int device_index) override {
      _allocator->DeviceFree(device_index, ptr);
    }

    void clear_cache() override {
      _allocator->FreeAllCached();
    }

  private:
    std::unique_ptr<cub::CachingDeviceAllocator> _allocator;
  };

  template<>
  Allocator& get_allocator<Device::CUDA>() {
    // Use 1 allocator per thread for performance.
    static thread_local CubCachingAllocator allocator;
    return allocator;
  }

#endif

  template<>
  Allocator& get_allocator<Device::CPU>() {
    constexpr size_t alignment = 64;
#ifdef CT2_WITH_MKL
    static MklAllocator allocator(alignment);
#else
    static AlignedAllocator allocator(alignment);
#endif
    return allocator;
  }

  Allocator& get_allocator(Device device) {
    Allocator* allocator = nullptr;
    DEVICE_DISPATCH(device, allocator = &get_allocator<D>());
    if (!allocator)
      throw std::runtime_error("No allocator defined for device " + device_to_str(device));
    return *allocator;
  }

}
