# Both platforms use the same public SDK interface and required Linux definition.
if(NOT EXISTS "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/vsi_nn_pub.h" OR
   NOT IS_DIRECTORY "${LJMICRO_SDK_LIBRARY_DIR}")
  message(FATAL_ERROR "Set LJMICRO_SDK_INCLUDE_DIR and LJMICRO_SDK_LIBRARY_DIR for ${SPECFERRY_TARGET}")
endif()

set(np101_sdk_includes
  "${LJMICRO_SDK_INCLUDE_DIR}"
  "${LJMICRO_SDK_INCLUDE_DIR}/HAL"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/utils"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/client"
  "${LJMICRO_SDK_INCLUDE_DIR}/acuity-ovxlib-dev/ops")
set(np101_sdk_libraries)
foreach(name ovxlib OpenVX OpenVXU CLC VSC GAL ArchModelSw NNArchPerf)
  # Re-resolve after SDK path changes instead of retaining libraries from an old SDK.
  unset(SDK_${name} CACHE)
  find_library(SDK_${name} NAMES ${name} PATHS "${LJMICRO_SDK_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
  list(APPEND np101_sdk_libraries "${SDK_${name}}")
endforeach()

# Compile/link only: catch incompatible target libraries without executing SDK code.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES ${np101_sdk_includes})
set(CMAKE_REQUIRED_DEFINITIONS -DLINUX)
set(CMAKE_REQUIRED_LIBRARIES ${np101_sdk_libraries} OpenSSL::Crypto Threads::Threads m)
set(CMAKE_REQUIRED_LINK_OPTIONS "-Wl,-rpath-link,${LJMICRO_SDK_LIBRARY_DIR}")
unset(SPECFERRY_SDK_LINKS CACHE)
check_cxx_source_compiles("
#include <vsi_nn_pub.h>
#include <openssl/evp.h>
int main() {
  auto context = vsi_nn_CreateContext();
  if (context) { vsi_nn_ReleaseContext(&context); }
  auto digest = EVP_MD_CTX_new();
  EVP_MD_CTX_free(digest);
  return 0;
}" SPECFERRY_SDK_LINKS)
cmake_pop_check_state()
if(NOT SPECFERRY_SDK_LINKS)
  message(FATAL_ERROR "SDK/OpenSSL cannot link for ${SPECFERRY_TARGET}; check the target libraries")
endif()

add_library(np101_sdk INTERFACE)
target_compile_definitions(np101_sdk INTERFACE LINUX)
target_include_directories(np101_sdk SYSTEM INTERFACE ${np101_sdk_includes})
target_link_libraries(np101_sdk INTERFACE ${np101_sdk_libraries} Threads::Threads m)
target_link_options(np101_sdk INTERFACE "-Wl,-rpath-link,${LJMICRO_SDK_LIBRARY_DIR}")
