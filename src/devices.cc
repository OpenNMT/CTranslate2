#include "ctranslate2/devices.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

#include "device_dispatch.h"

namespace ctranslate2 {

  Device str_to_device(const std::string& device) {
    if (device == "cuda" || device == "CUDA")
#ifdef CT2_WITH_CUDA
      return Device::CUDA;
#else
      throw std::invalid_argument("This CTranslate2 package was not compiled with CUDA support");
#endif
    if (device == "cpu" || device == "CPU")
      return Device::CPU;
    if (device == "auto" || device == "AUTO")
#ifdef CT2_WITH_CUDA
      return cuda::has_gpu() ? Device::CUDA : Device::CPU;
#else
      return Device::CPU;
#endif
    throw std::invalid_argument("unsupported device " + device);
  }

  std::string device_to_str(Device device) {
    switch (device) {
    case Device::CUDA:
      return "cuda";
    case Device::CPU:
      return "cpu";
    }
    return "";
  }

  std::string device_to_str(Device device, int index) {
    return device_to_str(device) + ":" + std::to_string(index);
  }

  int get_device_count(Device device) {
    switch (device) {
    case Device::CUDA:
#ifdef CT2_WITH_CUDA
      return cuda::get_gpu_count();
#else
      return 0;
#endif
    case Device::CPU:
      return 1;
    }
    return 0;
  }

  template <Device D>
  int get_device_index();
  template <Device D>
  void set_device_index(int index);

  template<>
  int get_device_index<Device::CPU>() {
    return 0;
  }

  template<>
  void set_device_index<Device::CPU>(int index) {
    if (index != 0)
      throw std::invalid_argument("Invalid CPU device index: " + std::to_string(index));
  }

#ifdef CT2_WITH_CUDA
  template<>
  int get_device_index<Device::CUDA>() {
    int index = 0;
    CUDA_CHECK(cudaGetDevice(&index));
    return index;
  }

  template<>
  void set_device_index<Device::CUDA>(int index) {
    CUDA_CHECK(cudaSetDevice(index));
  }
#endif

  int get_device_index(Device device) {
    int index = 0;
    DEVICE_DISPATCH(device, index = get_device_index<D>());
    return index;
  }

  void set_device_index(Device device, int index) {
    DEVICE_DISPATCH(device, set_device_index<D>(index));
  }

  void synchronize_device(Device device, int index) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA) {
      const ScopedDeviceSetter scoped_device_setter(device, index);
      cudaDeviceSynchronize();
    }
#else
    (void)device;
    (void)index;
#endif
  }

  void synchronize_stream(Device device) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA) {
      cudaStreamSynchronize(cuda::get_cuda_stream());
    }
#else
    (void)device;
#endif
  }

}
