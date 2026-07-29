foreach(_dg_igr_hdg_app_test IN ITEMS
        application_contract
        checkpoint_codec
        command_line
        durable_io
        scientific_contract)
  set(_dg_igr_hdg_app_target
      "dg_igr_hdg_runtime_app_${_dg_igr_hdg_app_test}_test")
  add_executable("${_dg_igr_hdg_app_target}"
                 "test_${_dg_igr_hdg_app_test}.cpp")
  target_link_libraries("${_dg_igr_hdg_app_target}"
                        PRIVATE dg_igr_hdg_runtime_app_contract)
  target_compile_definitions("${_dg_igr_hdg_app_target}" PRIVATE
    DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS=201703L)
  target_compile_options("${_dg_igr_hdg_app_target}" PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties("${_dg_igr_hdg_app_target}" PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

  add_test(NAME "dg_igr_hdg_runtime_app_${_dg_igr_hdg_app_test}_cpu"
           COMMAND "${_dg_igr_hdg_app_target}")
  set_tests_properties(
      "dg_igr_hdg_runtime_app_${_dg_igr_hdg_app_test}_cpu" PROPERTIES
      LABELS "hdg;runtime;app;contract;cpu"
      TIMEOUT 30)
endforeach()

add_test(NAME dg_igr_hdg_runtime_app_cli_process_cpu
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-process-output"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
set_tests_properties(dg_igr_hdg_runtime_app_cli_process_cpu PROPERTIES
  LABELS "hdg;runtime;app;cli;cpu"
  TIMEOUT 30)
