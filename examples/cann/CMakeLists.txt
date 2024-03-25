cmake_minimum_required(VERSION 3.7)
project(cann)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
find_package(Threads)
add_executable(cann_run main.cc)
target_link_libraries(cann_run PRIVATE
  ${CMAKE_THREAD_LIBS_INIT}
  ctranslate2
  )
