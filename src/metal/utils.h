#pragma once

#include <string>
#include <stdexcept>
#include <Metal/Metal.h>

namespace ctranslate2 {
  namespace metal {

    class MetalError : public std::runtime_error {
    public:
      explicit MetalError(const std::string& msg) : std::runtime_error(msg) {}
    };

    // Check if Metal is available
    bool has_metal();

    // Get number of Metal devices
    int get_metal_device_count();

    // Get current Metal device
    id<MTLDevice> get_metal_device();

    // Set current Metal device
    void set_metal_device(int index);

    // Initialize Metal device
    void init_metal();

    // Create Metal command queue
    id<MTLCommandQueue> create_command_queue();

    // Synchronize Metal device
    void synchronize_device();

    // Memory management
    void* metal_malloc(size_t size);
    void metal_free(void* ptr);
    void metal_memcpy(void* dst, const void* src, size_t size, bool to_device);

  }
}
