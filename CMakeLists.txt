cmake_minimum_required(VERSION 3.18)
project(SpecFerry LANGUAGES CXX)

# Export compiler flags and include paths for clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LJMICRO_SDK_INCLUDE_DIR "/usr/inc" CACHE PATH "NP101 SDK header root")
set(LJMICRO_SDK_LIBRARY_DIR "/usr/lib/ljmicro" CACHE PATH "NP101 SDK library directory")
find_package(Threads REQUIRED)

find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_library(np101_data native/np101/tensor_spec.cpp native/np101/weights.cpp native/np101/component_spec.cpp)
target_compile_features(np101_data PUBLIC cxx_std_17)
target_include_directories(np101_data PUBLIC native)
target_link_libraries(np101_data PUBLIC OpenSSL::Crypto)
target_compile_options(np101_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_context native/np101/context.cpp native/np101/tensor.cpp)
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
target_link_libraries(np101_context PUBLIC np101_data Threads::Threads m)
target_link_options(np101_context PUBLIC "-Wl,-rpath-link,${LJMICRO_SDK_LIBRARY_DIR}")
set(CMAKE_BUILD_RPATH "${LJMICRO_SDK_LIBRARY_DIR}")

target_compile_options(np101_context PRIVATE -Wall -Wextra -Wpedantic)

add_library(qwen_data native/models/qwen3_5/config.cpp)
target_link_libraries(qwen_data PUBLIC np101_data)
target_compile_options(qwen_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_ops native/np101/ops/graph_builder.cpp native/np101/ops/mixers.cpp
  native/np101/ops/vocabulary.cpp)
target_link_libraries(np101_ops PUBLIC np101_context)
target_compile_options(np101_ops PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_kv_cache native/np101/kv_cache.cpp)
target_link_libraries(np101_kv_cache PUBLIC np101_context)
target_compile_options(np101_kv_cache PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_delta_net native/models/qwen3_5/delta_net.cpp)
target_link_libraries(np101_delta_net PUBLIC np101_ops qwen_data)
target_compile_options(np101_delta_net PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_attention native/models/qwen3_5/attention.cpp)
target_link_libraries(np101_attention PUBLIC np101_ops np101_kv_cache qwen_data)
target_compile_options(np101_attention PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_decoder native/models/qwen3_5/decoder.cpp)
target_link_libraries(np101_decoder PUBLIC np101_delta_net np101_attention)
target_compile_options(np101_decoder PRIVATE -Wall -Wextra -Wpedantic)

add_library(opt_data native/models/opt/config.cpp)
target_link_libraries(opt_data PUBLIC np101_data)
target_compile_options(opt_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(specferry_generation native/inference/generation.cpp)
target_compile_features(specferry_generation PUBLIC cxx_std_17)
target_include_directories(specferry_generation PUBLIC native)
target_compile_options(specferry_generation PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_opt_decoder native/models/opt/decoder.cpp native/models/opt/model.cpp)
target_link_libraries(np101_opt_decoder PUBLIC np101_ops np101_kv_cache opt_data specferry_generation)
target_compile_options(np101_opt_decoder PRIVATE -Wall -Wextra -Wpedantic)

add_executable(specferry_opt_generate native/cli/opt_generate.cpp)
target_link_libraries(specferry_opt_generate PRIVATE np101_opt_decoder)
target_compile_options(specferry_opt_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_opt_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests/native)
endif()
