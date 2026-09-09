cmake_minimum_required(VERSION 3.18)
project(SpecFerry LANGUAGES CXX)

# Export compiler flags and include paths for clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LJMICRO_SDK_INCLUDE_DIR "/usr/inc" CACHE PATH "NP101 SDK header root")
set(LJMICRO_SDK_LIBRARY_DIR "/usr/lib/ljmicro" CACHE PATH "NP101 SDK library directory")
find_package(Threads REQUIRED)

add_library(np101_context native/np101/context.cpp)
target_compile_features(np101_context PUBLIC cxx_std_17)
target_compile_definitions(np101_context PUBLIC LINUX)
target_include_directories(np101_context PUBLIC native)
target_include_directories(np101_context SYSTEM PUBLIC
  "${LJMICRO_SDK_INCLUDE_DIR}"
  "${LJMICRO_SDK_INCLUDE_DIR}/HAL"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/utils"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/client"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/ops")
foreach(name ovxlib OpenVX OpenVXU CLC VSC GAL ArchModelSw NNArchPerf)
  find_library(SDK_${name} NAMES ${name} PATHS "${LJMICRO_SDK_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(np101_context PUBLIC "${SDK_${name}}")
endforeach()
target_link_libraries(np101_context PUBLIC Threads::Threads ${CMAKE_DL_LIBS} m)
target_link_options(np101_context PUBLIC "-Wl,-rpath-link,${LJMICRO_SDK_LIBRARY_DIR}")
set(CMAKE_BUILD_RPATH "${LJMICRO_SDK_LIBRARY_DIR}")

add_executable(np101_smoke native/np101/smoke.cpp)
target_link_libraries(np101_smoke PRIVATE np101_context)
foreach(target np101_context np101_smoke)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()
