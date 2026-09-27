#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <mutex>
#include <unordered_map>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "mps/utils.h"

namespace ctranslate2 {

  class MPSAllocator : public Allocator {
  public:
    void* allocate(size_t size, int device_index) override {
      (void)device_index;
      void* buf = mps::allocate_buffer(size);
      id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf;
      void* contents = [mtl_buf contents];
      std::lock_guard<std::mutex> lock(_mutex);
      _ptr_to_buffer[contents] = mtl_buf;
      return contents;
    }

    void free(void* ptr, int device_index) override {
      (void)device_index;
      if (!ptr)
        return;
      id<MTLBuffer> mtl_buf = nil;
      {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _ptr_to_buffer.find(ptr);
        if (it != _ptr_to_buffer.end()) {
          mtl_buf = it->second;
          _ptr_to_buffer.erase(it);
        }
      }
      if (mtl_buf)
        mps::free_buffer((__bridge void*)mtl_buf);
    }

  private:
    std::mutex _mutex;
    std::unordered_map<void*, id<MTLBuffer>> _ptr_to_buffer;
  };

  template<>
  Allocator& get_allocator<Device::MPS>() {
    static MPSAllocator allocator;
    return allocator;
  }

}

#endif  // __APPLE__
