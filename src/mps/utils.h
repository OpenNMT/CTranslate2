#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <cstdint>

namespace ctranslate2 {
  namespace mps {

    bool has_mps();
    int get_device_count();
    int get_device_index();
    void set_device_index(int index);
    void flush();
    void synchronize(const char* reason = nullptr, size_t bytes = 0);

    void* allocate_buffer(size_t size);
    void free_buffer(void* ptr);

    void* get_command_queue();
    void* get_device();
    void* get_buffer(const void* ptr, size_t size, size_t* offset);
    // Atomically looks up and retains a registered Metal buffer for command
    // encoding. This closes the lookup-to-retain race with asynchronous frees.
    void* get_buffer_for_use(const void* ptr, size_t size, size_t* offset);
    void record_metal_buffer_use(void* metal_buffer);
    void record_metal_object_use(void* metal_object);
    bool buffer_in_use(const void* ptr, size_t size);

    // Objective-C objects are intentionally exposed as opaque pointers so this
    // header remains usable from regular C++ translation units.
    void* command_buffer();
    void* compute_encoder();
    void* blit_encoder();
    void end_compute_encoder();
    void end_blit_encoder();
    void end_active_encoder();
    void record_compute_dispatch(const char* kernel_name = nullptr);
    void record_blit_operation();
    void record_copy_bytes(size_t bytes);

    enum class ProfileEvent : uint8_t {
      Gemm,
      Gemv,
      CpuFallback,
      TopKGpu,
      TopKCpu,
    };

    bool profile_enabled();
    void record_profile_event(ProfileEvent event);

  }
}

#endif  // __APPLE__
