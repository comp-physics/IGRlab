add_executable(dg_igr_hdg_runtime_abi_test test_runtime_abi.cpp)
target_link_libraries(dg_igr_hdg_runtime_abi_test
                      PRIVATE dg_igr_hdg_runtime_common)
target_compile_definitions(dg_igr_hdg_runtime_abi_test PRIVATE
  DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS=201703L)
target_compile_options(dg_igr_hdg_runtime_abi_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_runtime_abi_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

add_test(NAME dg_igr_hdg_runtime_abi_cpu
         COMMAND dg_igr_hdg_runtime_abi_test)
set_tests_properties(dg_igr_hdg_runtime_abi_cpu PROPERTIES
  LABELS "hdg;runtime;common;cpu"
  TIMEOUT 30)
