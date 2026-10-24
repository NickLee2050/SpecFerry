if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
  message(FATAL_ERROR "SoC requires Linux riscv64; select cmake/toolchains/np101-riscv64.cmake")
endif()

# Target libraries must come from the matching board SDK, never the host EP SDK.
set(LJMICRO_SDK_INCLUDE_DIR "" CACHE PATH "SoC SDK header root in the board snapshot")
set(LJMICRO_SDK_LIBRARY_DIR "" CACHE PATH "SoC SDK libraries in the board snapshot")
set(CMAKE_SKIP_RPATH TRUE)
set(SPECFERRY_PLATFORM_SOURCE native/platform/soc/runtime_profile.cpp)
