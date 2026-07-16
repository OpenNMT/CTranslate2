#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mps/utils.h"
#include "mps/kernels.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace mps {

    static id<MTLDevice> g_device = nil;
    static id<MTLCommandQueue> g_queue = nil;
    static int g_device_index = 0;
    static std::mutex g_buffers_mutex;
    static std::mutex g_error_mutex;
    static std::string g_last_command_error;

    struct BufferInfo {
      id<MTLBuffer> buffer;
      size_t size;
    };

    static std::map<uintptr_t, BufferInfo> g_buffers;
    static std::mutex& inflight_mutex() {
      // Completion handlers can run during process teardown. Leak these small
      // registries intentionally so they always outlive thread-local streams.
      static std::mutex* mutex = new std::mutex;
      return *mutex;
    }

    static std::unordered_map<void*, size_t>& inflight_buffers() {
      static auto* buffers = new std::unordered_map<void*, size_t>;
      return *buffers;
    }

    static std::unordered_set<void*>& pending_buffer_frees() {
      static auto* buffers = new std::unordered_set<void*>;
      return *buffers;
    }

    static void destroy_buffer(void* ptr);

    static std::shared_ptr<std::vector<void*>> mark_inflight(
        std::unordered_set<void*>& active_buffers) {
      auto resources = std::make_shared<std::vector<void*>>(active_buffers.begin(),
                                                            active_buffers.end());
      active_buffers.clear();
      std::lock_guard<std::mutex> lock(inflight_mutex());
      for (void* buffer : *resources)
        ++inflight_buffers()[buffer];
      return resources;
    }

    static void unmark_inflight(const std::shared_ptr<std::vector<void*>>& resources) {
      std::vector<void*> buffers_to_destroy;
      {
        std::lock_guard<std::mutex> lock(inflight_mutex());
        for (void* buffer : *resources) {
          auto& buffers = inflight_buffers();
          const auto found = buffers.find(buffer);
          if (found != buffers.end() && --found->second == 0) {
            buffers.erase(found);
            if (pending_buffer_frees().erase(buffer) != 0)
              buffers_to_destroy.emplace_back(buffer);
          }
          CFRelease(static_cast<CFTypeRef>(buffer));
        }
      }
      for (void* buffer : buffers_to_destroy)
        destroy_buffer(buffer);
    }

    struct ProfileCounters {
      std::atomic<uint64_t> command_buffers_created{0};
      std::atomic<uint64_t> command_buffers_committed{0};
      std::atomic<uint64_t> compute_dispatches{0};
      std::atomic<uint64_t> blit_operations{0};
      std::atomic<uint64_t> synchronizations{0};
      std::atomic<uint64_t> gemms{0};
      std::atomic<uint64_t> gemvs{0};
      std::atomic<uint64_t> cpu_fallbacks{0};
      std::atomic<uint64_t> topk_gpu_calls{0};
      std::atomic<uint64_t> topk_cpu_calls{0};
      std::atomic<uint64_t> buffer_lookups{0};
      std::atomic<uint64_t> buffer_lookup_comparisons{0};
      std::atomic<uint64_t> allocations{0};
      std::atomic<uint64_t> allocated_bytes{0};
      std::atomic<uint64_t> copied_bytes{0};
      std::atomic<uint64_t> gpu_time_ns{0};
    };

    static ProfileCounters g_profile;
    static std::mutex g_kernel_profile_mutex;
    static std::unordered_map<std::string, uint64_t> g_kernel_dispatches;

    static bool env_enabled(const char* name) {
      const char* value = std::getenv(name);
      return value && value[0] != '\0' && std::string(value) != "0";
    }

    bool profile_enabled() {
      static const bool enabled = env_enabled("CT2_MPS_PROFILE");
      return enabled;
    }

    static uint64_t operation_limit(const char* specific_name, uint64_t default_value) {
      const char* legacy = std::getenv("CT2_MPS_MAX_OPS");
      const char* specific = std::getenv(specific_name);
      const char* env = specific && specific[0] != '\0' ? specific : legacy;
      if (!env || env[0] == '\0')
        return default_value;
      errno = 0;
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(env, &end, 10);
      if (errno != 0 || end == env || *end != '\0')
        return default_value;
      return static_cast<uint64_t>(parsed);
    }

    static uint64_t max_compute_operations() {
      static const uint64_t value = []() {
        // The M1 Marian trace reaches useful overlap at 16 compute dispatches;
        // copy-only sequences are cheaper to encode and use a separate limit.
        return operation_limit("CT2_MPS_MAX_COMPUTE_OPS", 16);
      }();
      return value;
    }

    static uint64_t max_blit_operations() {
      static const uint64_t value = []() {
        return operation_limit("CT2_MPS_MAX_BLIT_OPS", 64);
      }();
      return value;
    }

    static void print_profile() {
      if (!profile_enabled())
        return;
      std::fprintf(stderr,
                   "CT2 MPS profile: command_buffers_created=%llu "
                   "command_buffers_committed=%llu compute_dispatches=%llu "
                   "blit_operations=%llu synchronizations=%llu gemms=%llu "
                   "gemvs=%llu cpu_fallbacks=%llu topk_gpu_calls=%llu "
                   "topk_cpu_calls=%llu buffer_lookups=%llu "
                   "buffer_lookup_comparisons=%llu allocations=%llu "
                   "allocated_bytes=%llu copied_bytes=%llu gpu_time_ms=%.3f\n",
                   static_cast<unsigned long long>(g_profile.command_buffers_created.load()),
                   static_cast<unsigned long long>(g_profile.command_buffers_committed.load()),
                   static_cast<unsigned long long>(g_profile.compute_dispatches.load()),
                   static_cast<unsigned long long>(g_profile.blit_operations.load()),
                   static_cast<unsigned long long>(g_profile.synchronizations.load()),
                   static_cast<unsigned long long>(g_profile.gemms.load()),
                   static_cast<unsigned long long>(g_profile.gemvs.load()),
                   static_cast<unsigned long long>(g_profile.cpu_fallbacks.load()),
                   static_cast<unsigned long long>(g_profile.topk_gpu_calls.load()),
                   static_cast<unsigned long long>(g_profile.topk_cpu_calls.load()),
                   static_cast<unsigned long long>(g_profile.buffer_lookups.load()),
                   static_cast<unsigned long long>(g_profile.buffer_lookup_comparisons.load()),
                   static_cast<unsigned long long>(g_profile.allocations.load()),
                   static_cast<unsigned long long>(g_profile.allocated_bytes.load()),
                   static_cast<unsigned long long>(g_profile.copied_bytes.load()),
                   static_cast<double>(g_profile.gpu_time_ns.load()) / 1.0e6);
      std::vector<std::pair<std::string, uint64_t>> kernels;
      {
        std::lock_guard<std::mutex> lock(g_kernel_profile_mutex);
        kernels.assign(g_kernel_dispatches.begin(), g_kernel_dispatches.end());
      }
      std::sort(kernels.begin(), kernels.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
      });
      for (const auto& kernel : kernels)
        std::fprintf(stderr,
                     "CT2 MPS kernel: name=%s dispatches=%llu\n",
                     kernel.first.c_str(),
                     static_cast<unsigned long long>(kernel.second));
    }

    static void record_command_buffer_error(id<MTLCommandBuffer> command_buffer) {
      if (!command_buffer || [command_buffer status] != MTLCommandBufferStatusError)
        return;

      std::string message = "MPS command buffer failed";
      if ([command_buffer label])
        message += " (" + std::string([[command_buffer label] UTF8String]) + ")";
      if ([command_buffer error] && [[command_buffer error] localizedDescription])
        message += ": " + std::string([[[command_buffer error] localizedDescription] UTF8String]);

      std::lock_guard<std::mutex> lock(g_error_mutex);
      if (g_last_command_error.empty())
        g_last_command_error = std::move(message);
    }

    static void throw_last_command_error() {
      std::string error;
      {
        std::lock_guard<std::mutex> lock(g_error_mutex);
        error.swap(g_last_command_error);
      }
      if (!error.empty())
        throw std::runtime_error(error);
    }

    static void ensure_initialized() {
      static std::once_flag once;
      std::call_once(once, []() {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device) {
          g_queue = [g_device newCommandQueue];
          if (profile_enabled())
            std::atexit(print_profile);
        }
      });
    }

    class MPSStream {
    public:
      ~MPSStream() {
        // A worker thread can finish with pending asynchronous work. Wait here
        // so completion handlers cannot outlive process-wide Metal state during
        // thread or process teardown. This is not on the inference hot path.
        try {
          synchronize();
        } catch (...) {
        }
        if (_last_submitted)
          [_last_submitted release];
      }

      id<MTLCommandBuffer> command_buffer() {
        throw_last_command_error();
        if (!_command_buffer) {
          ensure_initialized();
          if (!g_queue)
            throw std::runtime_error("MPS command queue not available");
          _command_buffer = [[g_queue commandBuffer] retain];
          if (!_command_buffer)
            throw std::runtime_error("MPS command buffer creation failed");
          if (profile_enabled())
            ++g_profile.command_buffers_created;
        }
        return _command_buffer;
      }

      id<MTLComputeCommandEncoder> compute_encoder() {
        if (_blit_encoder)
          end_blit_encoder();
        if (!_compute_encoder) {
          _compute_encoder = [[command_buffer() computeCommandEncoder] retain];
          if (!_compute_encoder)
            throw std::runtime_error("MPS compute encoder creation failed");
        }
        return _compute_encoder;
      }

      id<MTLBlitCommandEncoder> blit_encoder() {
        if (_compute_encoder)
          end_compute_encoder();
        if (!_blit_encoder) {
          _blit_encoder = [[command_buffer() blitCommandEncoder] retain];
          if (!_blit_encoder)
            throw std::runtime_error("MPS blit encoder creation failed");
        }
        return _blit_encoder;
      }

      void end_compute_encoder() {
        if (_compute_encoder) {
          [_compute_encoder endEncoding];
          [_compute_encoder release];
          _compute_encoder = nil;
        }
      }

      void end_blit_encoder() {
        if (_blit_encoder) {
          [_blit_encoder endEncoding];
          [_blit_encoder release];
          _blit_encoder = nil;
        }
      }

      void end_active_encoder() {
        end_compute_encoder();
        end_blit_encoder();
      }

      void record_operation(bool compute) {
        if (compute)
          ++_compute_operations;
        else
          ++_blit_operations;
        const uint64_t limit = compute ? max_compute_operations() : max_blit_operations();
        const uint64_t count = compute ? _compute_operations : _blit_operations;
        if (limit != 0 && count >= limit)
          flush();
      }

      void record_buffer(void* buffer) {
        if (buffer && _active_buffers.insert(buffer).second) {
          // StorageView temporaries may be destroyed before this command
          // buffer is committed. Metal command encoding alone does not give
          // the registry ownership, so retain each resource until the GPU
          // completion handler has run.
          CFRetain(static_cast<CFTypeRef>(buffer));
        }
      }

      bool has_active_buffer(void* buffer) const {
        return _active_buffers.find(buffer) != _active_buffers.end();
      }

      void flush() {
        end_active_encoder();
        if (!_command_buffer)
          return;
        id<MTLCommandBuffer> submitted = _command_buffer;
        _command_buffer = nil;
        _compute_operations = 0;
        _blit_operations = 0;
        auto resources = mark_inflight(_active_buffers);
        if (_last_submitted)
          [_last_submitted release];
        _last_submitted = submitted;
        [submitted addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
          record_command_buffer_error(completed_buffer);
          if (profile_enabled()
              && completed_buffer.GPUStartTime > 0
              && completed_buffer.GPUEndTime >= completed_buffer.GPUStartTime) {
            const double duration = completed_buffer.GPUEndTime - completed_buffer.GPUStartTime;
            g_profile.gpu_time_ns.fetch_add(static_cast<uint64_t>(duration * 1.0e9));
          }
          unmark_inflight(resources);
        }];
        [submitted commit];
        if (profile_enabled())
          ++g_profile.command_buffers_committed;
      }

      void synchronize() {
        end_active_encoder();
        id<MTLCommandBuffer> submitted = _command_buffer;
        std::shared_ptr<std::vector<void*>> resources;
        if (submitted) {
          _command_buffer = nil;
          _compute_operations = 0;
          _blit_operations = 0;
          resources = mark_inflight(_active_buffers);
          if (_last_submitted) {
            [_last_submitted release];
            _last_submitted = nil;
          }
          [submitted commit];
          if (profile_enabled())
            ++g_profile.command_buffers_committed;
        } else {
          submitted = _last_submitted;
          _last_submitted = nil;
        }
        if (!submitted) {
          throw_last_command_error();
          return;
        }
        if (profile_enabled())
          ++g_profile.synchronizations;
        [submitted waitUntilCompleted];
        if (resources)
          unmark_inflight(resources);
        record_command_buffer_error(submitted);
        [submitted release];
        throw_last_command_error();
      }

    private:
      id<MTLCommandBuffer> _command_buffer = nil;
      id<MTLCommandBuffer> _last_submitted = nil;
      id<MTLComputeCommandEncoder> _compute_encoder = nil;
      id<MTLBlitCommandEncoder> _blit_encoder = nil;
      uint64_t _compute_operations = 0;
      uint64_t _blit_operations = 0;
      std::unordered_set<void*> _active_buffers;
    };

    static MPSStream& current_stream() {
      static thread_local MPSStream stream;
      return stream;
    }

    bool has_mps() {
      ensure_initialized();
      return g_device != nil;
    }

    int get_device_count() {
      if (!has_mps())
        return 0;
      return 1;
    }

    int get_device_index() {
      return g_device_index;
    }

    void set_device_index(int index) {
      if (index != 0)
        THROW_INVALID_ARGUMENT("MPS device index must be 0 (single device only)");
      g_device_index = index;
    }

    void flush() {
      if (!has_mps())
        return;
      current_stream().flush();
    }

    void synchronize(const char* reason, size_t bytes) {
      if (!has_mps())
        return;
      if (env_enabled("CT2_MPS_LOG_SYNC"))
        std::fprintf(stderr,
                     "CT2 MPS synchronize reason=%s bytes=%zu\n",
                     reason ? reason : "explicit",
                     bytes);
      current_stream().synchronize();
    }

    void* allocate_buffer(size_t size) {
      if (!has_mps())
        throw std::runtime_error("MPS device not available");

      id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
      if (!buf)
        throw std::runtime_error("MPS buffer allocation failed");
      if (profile_enabled()) {
        ++g_profile.allocations;
        g_profile.allocated_bytes.fetch_add(size);
      }
      {
        std::lock_guard<std::mutex> lock(g_buffers_mutex);
        g_buffers[reinterpret_cast<uintptr_t>([buf contents])] = BufferInfo{buf, size};
      }
      return (__bridge void*)buf;
    }

    static void destroy_buffer(void* ptr) {
      id<MTLBuffer> buf = (__bridge id<MTLBuffer>)ptr;
      invalidate_packed_weight_cache(ptr);
      {
        std::lock_guard<std::mutex> lock(g_buffers_mutex);
        g_buffers.erase(reinterpret_cast<uintptr_t>([buf contents]));
      }
#if __has_feature(objc_arc)
      (void)CFBridgingRelease(ptr);
#else
      [buf release];
#endif
      (void)buf;
    }

    void free_buffer(void* ptr) {
      if (!ptr)
        return;

      const bool active = current_stream().has_active_buffer(ptr);
      if (active)
        current_stream().flush();
      bool defer = active;
      {
        std::lock_guard<std::mutex> lock(inflight_mutex());
        if (inflight_buffers().find(ptr) != inflight_buffers().end())
          defer = true;
        if (defer)
          pending_buffer_frees().insert(ptr);
      }
      if (!defer)
        destroy_buffer(ptr);
    }

    void* get_command_queue() {
      if (!has_mps())
        return nullptr;
      return (__bridge void*)g_queue;
    }

    void* get_device() {
      if (!has_mps())
        return nullptr;
      return (__bridge void*)g_device;
    }

    void* get_buffer(const void* ptr, size_t size, size_t* offset) {
      if (!ptr)
        return nullptr;

      const auto address = reinterpret_cast<uintptr_t>(ptr);
      if (size > std::numeric_limits<uintptr_t>::max() - address)
        return nullptr;
      const auto requested_end = address + size;
      if (profile_enabled())
        ++g_profile.buffer_lookups;
      std::lock_guard<std::mutex> lock(g_buffers_mutex);
      auto it = g_buffers.upper_bound(address);
      if (it == g_buffers.begin())
        return nullptr;
      --it;
      if (profile_enabled())
        ++g_profile.buffer_lookup_comparisons;
      const uintptr_t base = it->first;
      if (it->second.size > std::numeric_limits<uintptr_t>::max() - base)
        return nullptr;
      const uintptr_t allocation_end = base + it->second.size;
      if (address >= base && requested_end <= allocation_end) {
        if (offset)
          *offset = static_cast<size_t>(address - base);
        return (__bridge void*)it->second.buffer;
      }
      return nullptr;
    }

    void record_metal_buffer_use(void* metal_buffer) {
      current_stream().record_buffer(metal_buffer);
    }

    void record_metal_object_use(void* metal_object) {
      current_stream().record_buffer(metal_object);
    }

    bool buffer_in_use(const void* ptr, size_t size) {
      size_t offset = 0;
      void* buffer = get_buffer(ptr, size, &offset);
      if (!buffer)
        return false;
      if (current_stream().has_active_buffer(buffer))
        return true;
      std::lock_guard<std::mutex> lock(inflight_mutex());
      const auto& buffers = inflight_buffers();
      return buffers.find(buffer) != buffers.end();
    }

    void* command_buffer() {
      return (__bridge void*)current_stream().command_buffer();
    }

    void* compute_encoder() {
      return (__bridge void*)current_stream().compute_encoder();
    }

    void* blit_encoder() {
      return (__bridge void*)current_stream().blit_encoder();
    }

    void end_compute_encoder() {
      current_stream().end_compute_encoder();
    }

    void end_blit_encoder() {
      current_stream().end_blit_encoder();
    }

    void end_active_encoder() {
      current_stream().end_active_encoder();
    }

    void record_compute_dispatch(const char* kernel_name) {
      if (profile_enabled()) {
        ++g_profile.compute_dispatches;
        std::lock_guard<std::mutex> lock(g_kernel_profile_mutex);
        ++g_kernel_dispatches[kernel_name ? kernel_name : "unknown"];
      }
      current_stream().record_operation(true);
    }

    void record_blit_operation() {
      if (profile_enabled())
        ++g_profile.blit_operations;
      current_stream().record_operation(false);
    }

    void record_copy_bytes(size_t bytes) {
      if (profile_enabled())
        g_profile.copied_bytes.fetch_add(bytes);
    }

    void record_profile_event(ProfileEvent event) {
      if (!profile_enabled())
        return;
      switch (event) {
      case ProfileEvent::Gemm:
        ++g_profile.gemms;
        break;
      case ProfileEvent::Gemv:
        ++g_profile.gemvs;
        break;
      case ProfileEvent::CpuFallback:
        ++g_profile.cpu_fallbacks;
        break;
      case ProfileEvent::TopKGpu:
        ++g_profile.topk_gpu_calls;
        break;
      case ProfileEvent::TopKCpu:
        ++g_profile.topk_cpu_calls;
        break;
      }
    }

  }
}

#endif  // __APPLE__
