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
  TIMEOUT 180)

add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cpu_serial
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cpu-serial"
    -DHDG_RUNTIME_FUNCTIONAL=TRUE
    -DHDG_RUNTIME_RANKS=1
    -DHDG_RUNTIME_BACKEND=cpu
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cpu_rank2
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cpu-rank2"
    -DHDG_RUNTIME_FUNCTIONAL=TRUE
    -DHDG_RUNTIME_RANKS=2
    -DHDG_RUNTIME_BACKEND=cpu
    "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
set_tests_properties(
  dg_igr_hdg_runtime_app_cli_functional_cpu_serial
  dg_igr_hdg_runtime_app_cli_functional_cpu_rank2
  PROPERTIES
    LABELS "hdg;runtime;app;cli;functional;cpu"
    RESOURCE_LOCK hdg_runtime_cpu
    TIMEOUT 1200)
set_tests_properties(
  dg_igr_hdg_runtime_app_cli_functional_cpu_rank2
  PROPERTIES PROCESSORS 2)

if(MFEM_USE_CUDA)
  add_test(NAME dg_igr_hdg_runtime_app_cli_process_cuda_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-process-cuda-rank1"
      -DHDG_RUNTIME_RANKS=1
      -DHDG_RUNTIME_BACKEND=cuda
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cuda_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cuda-rank1"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_RANKS=1
      -DHDG_RUNTIME_BACKEND=cuda
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cuda_rank2
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cuda-rank2"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_RANKS=2
      -DHDG_RUNTIME_BACKEND=cuda
      "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_process_cuda_rank1
    dg_igr_hdg_runtime_app_cli_functional_cuda_rank1
    dg_igr_hdg_runtime_app_cli_functional_cuda_rank2
    PROPERTIES
      LABELS "hdg;runtime;app;cli;functional;cuda"
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_functional_cuda_rank2
    PROPERTIES PROCESSORS 2)
endif()
