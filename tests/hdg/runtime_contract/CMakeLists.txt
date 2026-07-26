add_executable(dg_igr_hdg_runtime_contract_test test_run_contract.cpp)
target_include_directories(dg_igr_hdg_runtime_contract_test PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
target_compile_options(dg_igr_hdg_runtime_contract_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
target_compile_definitions(dg_igr_hdg_runtime_contract_test PRIVATE
  DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS=201703L)
set_target_properties(dg_igr_hdg_runtime_contract_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME dg_igr_hdg_runtime_contract_cpu
    COMMAND dg_igr_hdg_runtime_contract_test)
  set_tests_properties(dg_igr_hdg_runtime_contract_cpu PROPERTIES
    LABELS "hdg;cpu;runtime;contract;cxx17"
    TIMEOUT 30)
endif()
