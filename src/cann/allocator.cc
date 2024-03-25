#include "ctranslate2/allocator.h"
#include "./utils.h"

namespace ctranslate2 {
  namespace cann {

    class CannAllocator : public Allocator {
    public:
      void* allocate(size_t size, int device_index) override {
        int prev_device_index = -1;
        if (device_index >= 0) {
          ACL_CALL(aclrtGetDevice(&prev_device_index));
          ACL_CALL(aclrtSetDevice(device_index));
        }

        void* ptr = nullptr;
        ACL_CALL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        if (prev_device_index >= 0) {
            ACL_CALL(aclrtSetDevice(prev_device_index));
        }
        return ptr;
      }

      void free(void* ptr, int device_index) override {
        int prev_device_index = -1;
        if (device_index >= 0) {
            ACL_CALL(aclrtGetDevice(&prev_device_index));
            ACL_CALL(aclrtSetDevice(device_index));
        }
        ACL_CALL(aclrtFree(ptr));

        if (prev_device_index >= 0) {
            ACL_CALL(aclrtSetDevice(prev_device_index));
        }
      }
    };
  }

  template<>
  Allocator& get_allocator<Device::CANN>() {
    static cann::CannAllocator allocator;
    return allocator;
  }
}
