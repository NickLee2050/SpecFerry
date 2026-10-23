set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(NP101_TOOLCHAIN_ROOT "" CACHE PATH "Extracted NP101 RISC-V GCC toolchain")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES NP101_TOOLCHAIN_ROOT)
if(NOT EXISTS "${NP101_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++")
  message(FATAL_ERROR "Set NP101_TOOLCHAIN_ROOT to the extracted vendor toolchain")
endif()

set(CMAKE_CXX_COMPILER "${NP101_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++")
set(CMAKE_SYSROOT "${NP101_TOOLCHAIN_ROOT}/sysroot")
# Scalar RV64GC avoids depending on the board's vendor vector instruction revision.
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gc -mabi=lp64d")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Link with the supplied target libraries; choose their on-board location at launch.
set(CMAKE_SKIP_RPATH TRUE)
