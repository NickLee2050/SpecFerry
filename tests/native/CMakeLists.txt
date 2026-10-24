add_library(np101_test_support np101/test_support.cpp)
target_link_libraries(np101_test_support PUBLIC np101_context)
target_compile_options(np101_test_support PRIVATE -Wall -Wextra -Wpedantic)

# This diagnostic is tied to the accepted x86 EP SDK, not validated on SoC.
if(SPECFERRY_TARGET STREQUAL "ep")
  add_library(np101_verify_trace SHARED np101/verify_trace.cpp)
  target_compile_features(np101_verify_trace PRIVATE cxx_std_17)
  target_link_libraries(np101_verify_trace PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
  target_compile_options(np101_verify_trace PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(np101_verify_trace PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endif()

function(specferry_test_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE ${ARGN})
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endfunction()

function(specferry_native_test name source)
  specferry_test_executable(${name} ${source} np101_test_support ${ARGN})
endfunction()

# Device diagnostics remain explicit commands; ordinary CTest must not open NP101.
function(specferry_cpu_test name source contract)
  specferry_test_executable(${name} ${source} ${ARGN})
  if(NOT CMAKE_CROSSCOMPILING)
    add_test(NAME ${contract} COMMAND ${name})
    set_tests_properties(${contract} PROPERTIES LABELS "unit;cpu;${SPECFERRY_TARGET}" TIMEOUT 30)
  endif()
endfunction()


specferry_native_test(np101_kv_cache_check np101/kv_cache_check.cpp np101_kv_cache)
specferry_native_test(np101_cache_append_check np101/cache_append_check.cpp np101_ops)
specferry_native_test(np101_decoder_check np101/decoder_check.cpp np101_opt_decoder)

specferry_native_test(np101_opt_io_check np101/opt_io_check.cpp np101_opt_decoder)
specferry_native_test(np101_sampling_check np101/sampling_check.cpp np101_ops)
specferry_native_test(np101_lookup_check np101/lookup_check.cpp np101_ops)
specferry_native_test(np101_projection_check np101/projection_check.cpp np101_ops)
specferry_native_test(np101_fp16_preservation_check np101/fp16_preservation_check.cpp np101_ops)
specferry_native_test(np101_generation_check np101/generation_check.cpp np101_opt_decoder)
specferry_native_test(np101_compiled_model_check np101/compiled_model_check.cpp np101_opt_decoder)
specferry_test_executable(np101_compiled_model_transfer_check
  np101/compiled_model_transfer_check.cpp np101_ops specferry_generation)
specferry_test_executable(np101_memory_check np101/memory_check.cpp np101_context)

specferry_cpu_test(np101_data_test np101/data_test.cpp np101_data_contract np101_context opt_data qwen_data)
specferry_cpu_test(generation_test generation_test.cpp autoregressive_generation_contract specferry_generation)
specferry_cpu_test(diagnostics_test diagnostics_test.cpp sdk_timing_contract specferry_generation)
specferry_cpu_test(memory_budget_test memory_budget_test.cpp segmented_memory_budget np101_data)
