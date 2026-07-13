#pragma once

#include <stdexcept>

#include "ctranslate2/devices.h"

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    constexpr Device D = DEVICE;                \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#if defined(CT2_WITH_CUDA) && defined(CT2_WITH_MPS)
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    DEVICE_CASE(Device::MPS, SINGLE_ARG(STMTS))         \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#elif defined(CT2_WITH_MPS)
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    DEVICE_CASE(Device::MPS, SINGLE_ARG(STMTS))         \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#elif defined(CT2_WITH_CUDA)
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    UNSUPPORTED_DEVICE_CASE(Device::MPS)                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    UNSUPPORTED_DEVICE_CASE(Device::MPS)                \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif
