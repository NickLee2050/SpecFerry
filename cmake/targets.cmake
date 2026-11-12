# Link one concrete platform implementation; no EP/SoC preprocessor definitions.
add_library(specferry_platform_${SPECFERRY_TARGET}
  src/devices/np101/api/runtime_profile.cpp ${SPECFERRY_PLATFORM_SOURCE})
target_link_libraries(specferry_platform_${SPECFERRY_TARGET} PUBLIC np101_data)
target_compile_options(specferry_platform_${SPECFERRY_TARGET} PRIVATE -Wall -Wextra -Wpedantic)
add_library(specferry_platform ALIAS specferry_platform_${SPECFERRY_TARGET})

add_library(np101_sdk_identity src/devices/np101/api/sdk_identity.cpp)
target_link_libraries(np101_sdk_identity PUBLIC specferry_platform)
target_compile_options(np101_sdk_identity PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_context src/devices/np101/api/context.cpp src/devices/np101/api/tensor.cpp src/devices/np101/api/weight_bank.cpp)
target_compile_features(np101_context PUBLIC cxx_std_17)
target_include_directories(np101_context PUBLIC src)
target_link_libraries(np101_context PUBLIC np101_sdk np101_sdk_identity specferry_diagnostics)
target_compile_options(np101_context PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_ops src/devices/np101/api/ops/graph_builder.cpp src/devices/np101/api/ops/mixers.cpp
  src/devices/np101/api/ops/vocabulary.cpp src/devices/np101/api/ops/cache_update.cpp)
target_link_libraries(np101_ops PUBLIC np101_context)
target_compile_options(np101_ops PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_kv_cache src/devices/np101/api/kv_cache.cpp)
target_link_libraries(np101_kv_cache PUBLIC np101_context)
target_compile_options(np101_kv_cache PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_delta_net src/models/qwen3_5/delta_net.cpp src/models/qwen3_5/compiled.cpp)
target_link_libraries(np101_delta_net PUBLIC np101_ops qwen_data)
target_compile_options(np101_delta_net PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_attention src/models/qwen3_5/attention.cpp)
target_link_libraries(np101_attention PUBLIC np101_ops np101_kv_cache qwen_data)
target_compile_options(np101_attention PRIVATE -Wall -Wextra -Wpedantic)

add_library(np101_decoder src/models/qwen3_5/decoder.cpp src/models/qwen3_5/model.cpp
  src/models/qwen3_5/model_build.cpp src/models/qwen3_5/request.cpp src/models/qwen3_5/execution.cpp)
target_link_libraries(np101_decoder PUBLIC np101_delta_net np101_attention)
target_compile_options(np101_decoder PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(np101_decoder PUBLIC specferry_generation)

add_executable(specferry_qwen_generate src/cli/qwen_generate.cpp)
target_link_libraries(specferry_qwen_generate PRIVATE np101_decoder)
target_compile_options(specferry_qwen_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_qwen_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_library(np101_opt_decoder src/models/opt/decoder.cpp src/models/opt/model.cpp
  src/models/opt/layer.cpp src/models/opt/compiled_model.cpp
  src/models/opt/partitioned_graph_model.cpp src/models/opt/partitioned_graph_build.cpp
  src/models/opt/execution.cpp src/models/opt/graph_execution.cpp src/models/opt/request.cpp src/models/opt/block_prefill.cpp)
target_link_libraries(np101_opt_decoder PUBLIC np101_ops np101_kv_cache opt_data specferry_generation)
target_compile_options(np101_opt_decoder PRIVATE -Wall -Wextra -Wpedantic)

add_executable(specferry_opt_generate src/cli/opt_generate.cpp)
target_link_libraries(specferry_opt_generate PRIVATE np101_opt_decoder)
target_compile_options(specferry_opt_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_opt_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(specferry_opt_graph_generate src/cli/opt_graph_generate.cpp)
target_link_libraries(specferry_opt_graph_generate PRIVATE np101_opt_decoder)
target_compile_options(specferry_opt_graph_generate PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_opt_graph_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
