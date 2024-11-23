#include "utils.h"
#include <vector>

namespace ctranslate2 {
  namespace metal {

    static id<MTLDevice> current_device = nil;
    static id<MTLCommandQueue> command_queue = nil;
    static std::vector<id<MTLDevice>> available_devices;

    bool has_metal() {
      @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
      }
    }

    int get_metal_device_count() {
      @autoreleasepool {
        if (available_devices.empty()) {
          NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
          for (id<MTLDevice> device in devices) {
            available_devices.push_back(device);
          }
          [devices release];
        }
        return available_devices.size();
      }
    }

    id<MTLDevice> get_metal_device() {
      if (current_device == nil) {
        init_metal();
      }
      return current_device;
    }

    void set_metal_device(int index) {
      @autoreleasepool {
        if (index < 0 || index >= get_metal_device_count()) {
          throw MetalError("Invalid Metal device index: " + std::to_string(index));
        }

        if (command_queue != nil) {
          [command_queue release];
          command_queue = nil;
        }

        current_device = available_devices[index];
      }
    }

    void init_metal() {
      @autoreleasepool {
        if (!has_metal()) {
          throw MetalError("No Metal-capable device found");
        }

        if (current_device == nil) {
          current_device = MTLCreateSystemDefaultDevice();
          if (current_device == nil) {
            throw MetalError("Failed to create Metal device");
          }
        }

        if (command_queue == nil) {
          command_queue = [current_device newCommandQueue];
          if (command_queue == nil) {
            throw MetalError("Failed to create Metal command queue");
          }
        }
      }
    }

    id<MTLCommandQueue> create_command_queue() {
      @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
          throw MetalError("Failed to create Metal command queue");
        }
        return queue;
      }
    }

    void synchronize_device() {
      @autoreleasepool {
        if (command_queue != nil) {
          id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
          [command_buffer commit];
          [command_buffer waitUntilCompleted];
        }
      }
    }

    void* metal_malloc(size_t size) {
      @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        id<MTLBuffer> buffer = [device newBufferWithLength:size
                                                 options:MTLResourceStorageModeShared];
        if (buffer == nil) {
          throw MetalError("Failed to allocate Metal buffer");
        }
        return buffer.contents;
      }
    }

    void metal_free(void* ptr) {
      @autoreleasepool {
        if (ptr != nullptr) {
          id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)ptr;
          [buffer release];
        }
      }
    }

    void metal_memcpy(void* dst, const void* src, size_t size, bool to_device) {
      @autoreleasepool {
        if (to_device) {
          id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst;
          memcpy(dst_buffer.contents, src, size);
        } else {
          id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src;
          memcpy(dst, src_buffer.contents, size);
        }
      }
    }

  }
}
