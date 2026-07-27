enable_testing()

option(BUILD_GMOCK "" OFF)
option(INSTALL_GTEST "" OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest)

add_executable(ctranslate2_test
  attention_test.cc
  batching_test.cc
  decoding_test.cc
  layers_test.cc
  model_test.cc
  storage_view_test.cc
  ops_test.cc
  primitives_test.cc
  translator_test.cc
  test_utils.cc
  test.cc)
target_include_directories(ctranslate2_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  )
target_link_libraries(ctranslate2_test PRIVATE
  ${PROJECT_NAME}
  gtest_main
  )

add_executable(benchmark_ops
  benchmark_ops.cc
  )
target_link_libraries(benchmark_ops PRIVATE
  ${PROJECT_NAME}
  )
if(NOT MSVC)
  target_compile_options(benchmark_ops PRIVATE -Wno-unused-result)
endif()

if(WITH_CUDA)
  target_link_libraries(benchmark_ops PRIVATE ${CUDA_LIBRARIES})

elseif(WITH_HIP)
  enable_language(HIP)
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  link_directories(${ROCM_PATH}/lib)

  find_package(hiprand REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(rocthrust REQUIRED)
  find_package(hipcub REQUIRED)

  add_compile_definitions(__HIP_PLATFORM_AMD__)
  add_compile_definitions(__HIP_PLATFORM_HCC__)
  target_include_directories(benchmark_ops PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include ${ROCM_PATH}/include /include)
  target_link_libraries(benchmark_ops PRIVATE hiprand roc::hipblas roc::rocprim roc::rocthrust hip::hipcub)

endif()
