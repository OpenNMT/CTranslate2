#include "./utils.h"

#include <array>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "ctranslate2/utils.h"

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    constexpr int max_gpus = 16;

    const char* cublasGetStatusName(cublasStatus_t status)
    {
      switch (status)
      {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "UNKNOWN";
      }
    }

    class CudaStream {
    public:
      CudaStream(int device)
        : _device(device)
      {
        CUDA_CHECK(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
      }
      ~CudaStream() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cudaStreamDestroy(_stream);
      }
      cudaStream_t get() const {
        return _stream;
      }
    private:
      int _device;
      cudaStream_t _stream;
    };

    // Pool of CUDA streams, one per device.
    class CudaStreamPool {
    public:
      CudaStreamPool() {
        if (get_gpu_count() > max_gpus)
          throw std::runtime_error("Number of CUDA devices on the machine is larger than "
                                   "the maximum supported number ("
                                   + std::to_string(max_gpus) + ")");
      }

      cudaStream_t get_device_stream() {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        std::call_once(_init_streams[device], [this, device]() {
          _streams[device] = std::make_unique<CudaStream>(device);
        });

        return _streams[device]->get();
      }

    private:
      std::array<std::unique_ptr<CudaStream>, max_gpus> _streams;
      std::array<std::once_flag, max_gpus> _init_streams;
    };

    class CublasHandle {
    public:
      CublasHandle() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, get_cuda_stream()));
      }
      ~CublasHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cublasDestroy(_handle);
      }
      cublasHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      cublasHandle_t _handle;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    cudaStream_t get_cuda_stream() {
      static thread_local CudaStreamPool cuda_stream_pool;
      return cuda_stream_pool.get_device_stream();
    }

    cublasHandle_t get_cublas_handle() {
      static thread_local CublasHandle cublas_handle;
      return cublas_handle.get();
    }

#ifdef CT2_WITH_CUDNN
    class CudnnHandle {
    public:
      CudnnHandle() {
        CUDA_CHECK(cudaGetDevice(&_device));
        CUDNN_CHECK(cudnnCreate(&_handle));
        CUDNN_CHECK(cudnnSetStream(_handle, get_cuda_stream()));
      }
      ~CudnnHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        cudnnDestroy(_handle);
      }
      cudnnHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      cudnnHandle_t _handle;
    };

    cudnnHandle_t get_cudnn_handle() {
      static thread_local CudnnHandle cudnn_handle;
      return cudnn_handle.get();
    }

    cudnnDataType_t get_cudnn_data_type(DataType dtype) {
      switch (dtype) {
      case DataType::FLOAT32:
        return CUDNN_DATA_FLOAT;
      case DataType::FLOAT16:
        return CUDNN_DATA_HALF;
      case DataType::BFLOAT16:
        return CUDNN_DATA_BFLOAT16;
      case DataType::INT32:
        return CUDNN_DATA_INT32;
      case DataType::INT8:
        return CUDNN_DATA_INT8;
      default:
        throw std::invalid_argument("No cuDNN data type for type " + dtype_name(dtype));
      }
    }
#endif

    int get_gpu_count() {
      int gpu_count = 0;
      cudaError_t status = cudaGetDeviceCount(&gpu_count);
      if (status != cudaSuccess)
        return 0;
      return gpu_count;
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    const cudaDeviceProp& get_device_properties(int device) {
      static thread_local std::vector<std::unique_ptr<cudaDeviceProp>> cache;

      if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
      }
      if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1);
      }

      auto& device_prop = cache[device];
      if (!device_prop) {
        device_prop = std::make_unique<cudaDeviceProp>();
        CUDA_CHECK(cudaGetDeviceProperties(device_prop.get(), device));
      }
      return *device_prop;
    }

    // See docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html
    // for hardware support of reduced precision.

    bool gpu_supports_int8(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      // Disable INT8 for sm120: https://github.com/OpenNMT/CTranslate2/issues/1865
      if (device_prop.major == 12 && device_prop.minor == 0)
        return false;
      return device_prop.major > 6 || (device_prop.major == 6 && device_prop.minor == 1);
    }

    bool gpu_has_int8_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major > 7 || (device_prop.major == 7 && device_prop.minor >= 2);
    }

    bool gpu_has_fp16_tensor_cores(int device) {
      const cudaDeviceProp& device_prop = get_device_properties(device);
      return device_prop.major >= 7;
    }

    bool have_same_compute_capability(const std::vector<int>& devices) {
      if (devices.size() > 1) {
        int ref_major = -1;
        int ref_minor = -1;
        for (const int device : devices) {
          const cudaDeviceProp& device_prop = get_device_properties(device);
          const int major = device_prop.major;
          const int minor = device_prop.minor;
          if (ref_major < 0) {
            ref_major = major;
            ref_minor = minor;
          } else if (major != ref_major || minor != ref_minor)
            return false;
        }
      }

      return true;
    }

    static thread_local bool true_fp16_gemm = read_bool_from_env("CT2_CUDA_TRUE_FP16_GEMM", true);

    bool use_true_fp16_gemm() {
      return true_fp16_gemm;
    }

    void use_true_fp16_gemm(bool use) {
      true_fp16_gemm = use;
    }

  }
}
