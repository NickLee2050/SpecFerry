cmake_minimum_required(VERSION 3.18)

set(SPECFERRY_TARGET "" CACHE STRING "Execution platform: ep or soc")
set_property(CACHE SPECFERRY_TARGET PROPERTY STRINGS ep soc)
if(NOT SPECFERRY_TARGET MATCHES "^(ep|soc)$")
  message(FATAL_ERROR "Select -DSPECFERRY_TARGET=ep or soc and a separate build directory")
endif()
if(DEFINED SPECFERRY_CONFIGURED_TARGET AND
   NOT SPECFERRY_CONFIGURED_TARGET STREQUAL SPECFERRY_TARGET)
  message(FATAL_ERROR "Do not change platforms in an existing build directory; use build/ep or build/soc")
endif()
set(SPECFERRY_CONFIGURED_TARGET "${SPECFERRY_TARGET}" CACHE INTERNAL "Platform bound to this build")

project(SpecFerry LANGUAGES CXX)

# Export compiler flags and include paths for clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/devices/np101/${SPECFERRY_TARGET}.cmake)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
include(cmake/devices/np101/api.cmake)
message(STATUS "SpecFerry target: ${SPECFERRY_TARGET} (${CMAKE_SYSTEM_PROCESSOR})")
message(STATUS "Platform implementation: ${SPECFERRY_PLATFORM_SOURCE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "SDK headers: ${LJMICRO_SDK_INCLUDE_DIR}")
message(STATUS "SDK libraries: ${LJMICRO_SDK_LIBRARY_DIR}")

include(cmake/targets.cmake)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
