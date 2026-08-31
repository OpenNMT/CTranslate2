cmake_minimum_required(VERSION 3.15)
project(ct2_ruy_shutdown_repro CXX)

# Point this at a CTranslate2 source checkout (submodules initialized).
#   cmake -DCT2_DIR=/path/to/CTranslate2 ...
if(NOT DEFINED CT2_DIR)
  message(FATAL_ERROR "Pass -DCT2_DIR=<path to CTranslate2 source checkout>")
endif()

# Ruy-only CPU build: this is the backend that exhibits the shutdown deadlock.
set(WITH_RUY  ON  CACHE BOOL   "" FORCE)
set(WITH_MKL  OFF CACHE BOOL   "" FORCE)
set(WITH_DNNL OFF CACHE BOOL   "" FORCE)
set(WITH_CUDA OFF CACHE BOOL   "" FORCE)
set(BUILD_CLI OFF CACHE BOOL   "" FORCE)
set(OPENMP_RUNTIME "COMP" CACHE STRING "" FORCE)

# Build CTranslate2 as part of this project so the `ctranslate2` target carries all
# transitive static deps (ruy, cpu_features, ...) automatically.
add_subdirectory(${CT2_DIR} ctranslate2_build)

add_executable(repro repro.cpp)
target_link_libraries(repro PRIVATE ctranslate2)
target_compile_features(repro PRIVATE cxx_std_17)
