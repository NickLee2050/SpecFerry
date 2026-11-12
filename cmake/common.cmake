# Shared data contracts and integrity checks do not depend on the device SDK.
add_library(np101_data src/devices/np101/api/tensor_spec.cpp src/devices/np101/api/weights.cpp src/devices/np101/api/component_spec.cpp
  src/devices/np101/api/memory_budget.cpp)
target_compile_features(np101_data PUBLIC cxx_std_17)
target_include_directories(np101_data PUBLIC src)
target_link_libraries(np101_data PUBLIC OpenSSL::Crypto Threads::Threads)
target_compile_options(np101_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(specferry_diagnostics src/devices/np101/api/diagnostics.cpp)
target_compile_features(specferry_diagnostics PUBLIC cxx_std_17)
target_include_directories(specferry_diagnostics PUBLIC src)
target_compile_options(specferry_diagnostics PRIVATE -Wall -Wextra -Wpedantic)

add_library(qwen_data src/models/qwen3_5/config.cpp)
target_link_libraries(qwen_data PUBLIC np101_data)
target_compile_options(qwen_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(opt_data src/models/opt/config.cpp)
target_link_libraries(opt_data PUBLIC np101_data)
target_compile_options(opt_data PRIVATE -Wall -Wextra -Wpedantic)

add_library(specferry_generation src/inference/generation.cpp
  src/inference/request.cpp src/inference/report.cpp src/io/files.cpp)
target_compile_features(specferry_generation PUBLIC cxx_std_17)
target_include_directories(specferry_generation PUBLIC src)
target_link_libraries(specferry_generation PUBLIC specferry_diagnostics)
target_compile_options(specferry_generation PRIVATE -Wall -Wextra -Wpedantic)

# Independent process supervision has no SDK or model dependency.
add_library(specferry_supervision src/supervision/process.cpp)
target_compile_features(specferry_supervision PUBLIC cxx_std_17)
target_include_directories(specferry_supervision PUBLIC src)
target_link_libraries(specferry_supervision PUBLIC Threads::Threads)
target_compile_options(specferry_supervision PRIVATE -Wall -Wextra -Wpedantic)
add_executable(specferry_supervise src/cli/supervise.cpp)
target_link_libraries(specferry_supervise PRIVATE specferry_supervision)
target_compile_options(specferry_supervise PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(specferry_supervise PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
