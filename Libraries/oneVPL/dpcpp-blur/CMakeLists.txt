cmake_minimum_required(VERSION 3.10)


# This sample must be compiled and executed within the oneAPI product enviroment
# with the dpcpp compiler to use dpcpp features
set(DPCPP_COMPILER_NAME "dpcpp")
find_program(DPCPP_COMPILER 
             NAMES ${DPCPP_COMPILER_NAME} 
             PATHS /opt/intel/oneapi )

if(DPCPP_COMPILER)
  set(CMAKE_CXX_COMPILER ${DPCPP_COMPILER})
  message(STATUS "DPCPP compiler found, blur feature enabled")
else()
  message(STATUS "DPCPP compiler not found, blur feature disabled")
endif()

project(dpcpp-blur LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(TARGET dpcpp-blur)
set(SOURCES src/dpcpp-blur.cpp)
set(RUNARGS ${CMAKE_CURRENT_SOURCE_DIR}/content/cars_128x96.i420 128 96)

include_directories(
  /opt/intel/oneapi/vpl/2021.4.0/include/vpl 
  /opt/intel/oneapi/vpl/2021.4.0/include/
  /usr/include/libdrm/
  /home/jeff/innersource_oneVPL/frameworks.media.onevpl.dispatcher/tools/legacy/media_sdk_compatibility_headers
  )

link_directories(/usr/lib/x86_64-linux-gnu)
link_libraries(va va-drm vpl dl pthread -lsycl -lze_loader )


add_executable(dpcpp-blur  ${CMAKE_CURRENT_SOURCE_DIR}/src/dpcpp-blur.cpp)
target_compile_definitions(dpcpp-blur PUBLIC -DLIBVA_SUPPORT)

