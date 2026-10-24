if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  message(FATAL_ERROR "EP requires Linux x86_64; use a separate build/soc directory for SoC")
endif()

set(LJMICRO_SDK_INCLUDE_DIR "/usr/inc" CACHE PATH "EP SDK header root")
set(LJMICRO_SDK_LIBRARY_DIR "/usr/lib/ljmicro" CACHE PATH "EP SDK library directory")
set(CMAKE_BUILD_RPATH "${LJMICRO_SDK_LIBRARY_DIR}")
set(SPECFERRY_PLATFORM_SOURCE native/platform/ep/runtime_profile.cpp)
