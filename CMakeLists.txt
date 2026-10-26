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

include(cmake/platforms/${SPECFERRY_TARGET}.cmake)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
include(cmake/np101-sdk.cmake)
message(STATUS "SpecFerry target: ${SPECFERRY_TARGET} (${CMAKE_SYSTEM_PROCESSOR})")
message(STATUS "Platform implementation: ${SPECFERRY_PLATFORM_SOURCE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "SDK headers: ${LJMICRO_SDK_INCLUDE_DIR}")
message(STATUS "SDK libraries: ${LJMICRO_SDK_LIBRARY_DIR}")

# Shared data contracts and integrity checks do not depend on the device SDK.
add_library(np101_data native/np101/tensor_spec.cpp native/np101/weights.cpp native/np101/component_spec.cpp
  native/np101/memory_budget.cpp)
target_compile_features(np101_data PUBLIC cxx_std_17)
target_include_directories(np101_data PUBLIC native)
target_link_libraries(np101_data PUBLIC OpenSSL::Crypto Threads::Threads)
target_compile_options(np101_data PRIVATE -Wall -Wextra -Wpedantic)

# Link one concrete platform implementation; no EP/SoC preprocessor definitions.
add_library(specferry_platform_${SPECFERRY_TARGET}
  native/platform/runtime_profile.cpp ${SPECFERRY_PLATFORM_SOURCE})
target_link_libraries(specferry_platform_${SPECFERRY_TARGET} PUBLIC np101_data)
target_compile_options(specferry_platform_${SPECFERRY_TARGET} PRIVATE -Wall -Wextra -Wpedantic)
add_library(specferry_platform ALIAS specferry_platform_${SPECFERRY_TARGET})

add_library(np101_sdk_identity native/np101/sdk_identity.cpp)
target_link_libraries(np101_sdk_identity PUBLIC specferry_platform)
target_compile_options(np101_sdk_identity PRIVATE -Wall -Wextra -Wpedantic)

add_library(specferry_diagnostics native/np101/diagnostics.cpp)
target_compile_features(specferry_diagnostics PUBLIC cxx_std_17)
target_include_directories(specferry_diagnostics PUBLIC native)
target_compile_options(specferry_diagnostics PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_context native/np101/context.cpp native/np101/tensor.cpp native/np101/weight_bank.cpp)
target_compile_features(np101_context PUBLIC cxx_std_17)
target_include_directories(np101_context PUBLIC native)
target_link_libraries(np101_context PUBLIC np101_sdk np101_sdk_identity specferry_diagnostics)
target_compile_options(np101_context PRIVATE -Wall -Wextra -Wpedantic)

add_library(qwen_data native/models/qwen3_5/config.cpp)
target_link_libraries(qwen_data PUBLIC np101_data)
target_compile_options(qwen_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_ops native/np101/ops/graph_builder.cpp native/np101/ops/mixers.cpp
  native/np101/ops/vocabulary.cpp native/np101/ops/cache_update.cpp)
target_link_libraries(np101_ops PUBLIC np101_context)
target_compile_options(np101_ops PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_kv_cache native/np101/kv_cache.cpp)
target_link_libraries(np101_kv_cache PUBLIC np101_context)
target_compile_options(np101_kv_cache PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_delta_net native/models/qwen3_5/delta_net.cpp native/models/qwen3_5/compiled.cpp)
target_link_libraries(np101_delta_net PUBLIC np101_ops qwen_data)
target_compile_options(np101_delta_net PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_attention native/models/qwen3_5/attention.cpp)
target_link_libraries(np101_attention PUBLIC np101_ops np101_kv_cache qwen_data)
target_compile_options(np101_attention PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_decoder native/models/qwen3_5/decoder.cpp native/models/qwen3_5/model.cpp)
target_link_libraries(np101_decoder PUBLIC np101_delta_net np101_attention)
target_compile_options(np101_decoder PRIVATE -Wall -Wextra -Wpedantic)

add_library(opt_data native/models/opt/config.cpp)
target_link_libraries(opt_data PUBLIC np101_data)
target_compile_options(opt_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(specferry_generation native/inference/generation.cpp)
target_compile_features(specferry_generation PUBLIC cxx_std_17)
target_include_directories(specferry_generation PUBLIC native)
target_link_libraries(specferry_generation PUBLIC specferry_diagnostics)
target_compile_options(specferry_generation PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(np101_decoder PUBLIC specferry_generation)

add_executable(specferry_qwen_generate native/cli/qwen_generate.cpp)
target_link_libraries(specferry_qwen_generate PRIVATE np101_decoder)
target_compile_options(specferry_qwen_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_qwen_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(np101_opt_decoder native/models/opt/decoder.cpp native/models/opt/model.cpp
  native/models/opt/layer.cpp native/models/opt/compiled_model.cpp
  native/models/opt/partitioned_graph_model.cpp native/models/opt/block_prefill.cpp)
target_link_libraries(np101_opt_decoder PUBLIC np101_ops np101_kv_cache opt_data specferry_generation)
target_compile_options(np101_opt_decoder PRIVATE -Wall -Wextra -Wpedantic)

add_executable(specferry_opt_generate native/cli/opt_generate.cpp)
target_link_libraries(specferry_opt_generate PRIVATE np101_opt_decoder)
target_compile_options(specferry_opt_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_opt_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(specferry_opt_graph_generate native/cli/opt_graph_generate.cpp)
target_link_libraries(specferry_opt_graph_generate PRIVATE np101_opt_decoder)
target_compile_options(specferry_opt_graph_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_opt_graph_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests/native)
endif()
