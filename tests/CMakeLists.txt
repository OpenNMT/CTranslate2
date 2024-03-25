enable_testing()

option(BUILD_GMOCK "" OFF)
option(INSTALL_GTEST "" OFF)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest)

add_executable(ctranslate2_test
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
target_link_libraries(ctranslate2_test
  ${PROJECT_NAME}
  gtest_main
  )

add_executable(benchmark_ops
  benchmark_ops.cc
  )
target_link_libraries(benchmark_ops
  ${PROJECT_NAME}
  )

if(WITH_CUDA)
  target_link_libraries(benchmark_ops ${CUDA_LIBRARIES})
elseif(WITH_CANN)
  set(ASCEND_DIR /usr/local/Ascend)
  set(ASCEND_CL_DIR ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/lib64)
  set(ascendcl_lib ${ASCEND_CL_DIR}/libascendcl.so)
  target_link_libraries(benchmark_ops ${ascendcl_lib})
endif()
