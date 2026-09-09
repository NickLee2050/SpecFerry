function(specferry_native_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE np101_context)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endfunction()

specferry_native_test(np101_conv_relu_pool_test np101/conv_relu_pool_test.cpp)
specferry_native_test(np101_op_check np101/op_check.cpp np101/operators.cpp np101/case_file.cpp)
specferry_native_test(np101_weight_allocation_check np101/weight_allocation_check.cpp)
specferry_native_test(np101_graph_sharing_check np101/graph_sharing_check.cpp)

add_executable(np101_data_test np101/data_test.cpp np101/case_file.cpp)
target_link_libraries(np101_data_test PRIVATE np101_data)
target_compile_options(np101_data_test PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(np101_data_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME np101_data_contract COMMAND np101_data_test)
set_tests_properties(np101_data_contract PROPERTIES LABELS "unit;cpu" TIMEOUT 30)

# Device diagnostics are explicit commands: ordinary CTest must not open NP101
# or spend minutes waiting for a device graph to complete.
