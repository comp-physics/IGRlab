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
  if(_dg_igr_hdg_app_test STREQUAL "command_line")
    target_include_directories("${_dg_igr_hdg_app_target}" PRIVATE
      "${PROJECT_SOURCE_DIR}/tests/hdg/runtime/common")
    target_link_libraries("${_dg_igr_hdg_app_target}" PRIVATE mfem MPI::MPI_CXX)
  endif()
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

add_executable(dg_igr_hdg_runtime_app_w8c_shared_contract_test
  test_w8c_shared_contract.cpp)
target_include_directories(
  dg_igr_hdg_runtime_app_w8c_shared_contract_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu")
target_link_libraries(
  dg_igr_hdg_runtime_app_w8c_shared_contract_test PRIVATE
  dg_igr_hdg_runtime_app_contract)
target_compile_definitions(
  dg_igr_hdg_runtime_app_w8c_shared_contract_test PRIVATE
  DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS=201703L)
target_compile_options(
  dg_igr_hdg_runtime_app_w8c_shared_contract_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(
  dg_igr_hdg_runtime_app_w8c_shared_contract_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
add_test(NAME dg_igr_hdg_runtime_app_w8c_shared_contract_cpu
  COMMAND dg_igr_hdg_runtime_app_w8c_shared_contract_test)
set_tests_properties(
  dg_igr_hdg_runtime_app_w8c_shared_contract_cpu PROPERTIES
  LABELS "hdg;runtime;app;contract;cpu;w8c"
  TIMEOUT 30)

if(MFEM_USE_CUDA)
  add_executable(dg_igr_hdg_runtime_app_static_entry_selector_test
    test_static_entry_selector.cpp
    $<TARGET_OBJECTS:dg_igr_hdg_runtime_app_entry>)
  target_include_directories(
    dg_igr_hdg_runtime_app_static_entry_selector_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/app")
  target_link_libraries(
    dg_igr_hdg_runtime_app_static_entry_selector_test PRIVATE
    dg_igr_hdg_runtime_app_contract MPI::MPI_CXX)
  target_compile_definitions(
    dg_igr_hdg_runtime_app_static_entry_selector_test PRIVATE
    DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS=201703L)
  target_compile_options(
    dg_igr_hdg_runtime_app_static_entry_selector_test PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(
    dg_igr_hdg_runtime_app_static_entry_selector_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  add_test(NAME dg_igr_hdg_runtime_app_static_entry_selector_host
    COMMAND dg_igr_hdg_runtime_app_static_entry_selector_test)
  set_tests_properties(
    dg_igr_hdg_runtime_app_static_entry_selector_host PROPERTIES
    LABELS "hdg;runtime;app;contract;host-only"
    TIMEOUT 30)
endif()

add_test(NAME dg_igr_hdg_runtime_app_cli_process_cpu
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_BUILD_DRIVER=${CMAKE_SOURCE_DIR}/build.sh"
    "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-process-output"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
add_test(NAME dg_igr_hdg_runtime_app_cli_process_cpu_rank2
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-process-cpu-rank2"
    -DHDG_RUNTIME_RANKS=2
    -DHDG_RUNTIME_BACKEND=cpu
    "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
set_tests_properties(
  dg_igr_hdg_runtime_app_cli_process_cpu
  dg_igr_hdg_runtime_app_cli_process_cpu_rank2
  PROPERTIES LABELS "hdg;runtime;app;cli;cpu" TIMEOUT 300)
set_tests_properties(dg_igr_hdg_runtime_app_cli_process_cpu_rank2
  PROPERTIES PROCESSORS 2)

add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cpu_serial
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_SCIENTIFIC_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_scientific_contract_test>"
    "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cpu-serial"
    -DHDG_RUNTIME_FUNCTIONAL=TRUE
    -DHDG_RUNTIME_RANKS=1
    -DHDG_RUNTIME_BACKEND=cpu
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cpu_rank2
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
    "-DHDG_RUNTIME_SCIENTIFIC_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_scientific_contract_test>"
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

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  add_test(NAME dg_igr_hdg_runtime_app_cli_w8c_smooth_scalar_cpu_rank2
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCALAR_VTK_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_command_line_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-w8c-smooth-scalar-cpu-rank2"
      -DHDG_RUNTIME_W8C_SMOOTH_RANK2=TRUE
      -DHDG_RUNTIME_RANKS=2
      "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  set_tests_properties(dg_igr_hdg_runtime_app_cli_w8c_smooth_scalar_cpu_rank2
    PROPERTIES LABELS "hdg;runtime;app;cli;cpu;rank2;w8c;smooth-transition"
               PROCESSORS 2 RESOURCE_LOCK hdg_runtime_cpu TIMEOUT 7200)
  add_test(NAME dg_igr_hdg_runtime_app_cli_scalar_igr_cpu_serial
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCALAR_VTK_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_command_line_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-scalar-igr-cpu-serial"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_SCALAR_IGR=TRUE
      -DHDG_RUNTIME_RANKS=1
      -DHDG_RUNTIME_BACKEND=cpu
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_scalar_igr_cpu_rank2
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCALAR_VTK_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_command_line_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-scalar-igr-cpu-rank2"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_SCALAR_IGR=TRUE
      -DHDG_RUNTIME_RANKS=2
      -DHDG_RUNTIME_BACKEND=cpu
      "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_scalar_igr_cpu_serial
    dg_igr_hdg_runtime_app_cli_scalar_igr_cpu_rank2
    PROPERTIES
      LABELS "hdg;runtime;app;cli;functional;cpu;scalar-igr"
      RESOURCE_LOCK hdg_runtime_cpu
      TIMEOUT 1200)
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_scalar_igr_cpu_rank2
    PROPERTIES PROCESSORS 2)
endif()

if(MFEM_USE_CUDA)
  add_test(NAME dg_igr_hdg_runtime_app_cli_process_cuda_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_BUILD_DRIVER=${CMAKE_SOURCE_DIR}/build.sh"
      "-DHDG_RUNTIME_SCIENTIFIC_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_scientific_contract_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-process-cuda-rank1"
      -DHDG_RUNTIME_RANKS=1
      -DHDG_RUNTIME_BACKEND=cuda
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_process_cuda_rank2
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCIENTIFIC_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_scientific_contract_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-process-cuda-rank2"
      -DHDG_RUNTIME_RANKS=2
      -DHDG_RUNTIME_BACKEND=cuda
      "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cuda_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCIENTIFIC_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_scientific_contract_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cuda-rank1"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_RANKS=1
      -DHDG_RUNTIME_BACKEND=cuda
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_functional_cuda_rank2
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCIENTIFIC_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_scientific_contract_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-functional-cuda-rank2"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_RANKS=2
      -DHDG_RUNTIME_BACKEND=cuda
      "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCALAR_VTK_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_command_line_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-scalar-igr-cuda-rank1"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_SCALAR_IGR=TRUE
      -DHDG_RUNTIME_RANKS=1
      -DHDG_RUNTIME_BACKEND=cuda
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  add_test(NAME dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank2
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_SCALAR_VTK_TEST=$<TARGET_FILE:dg_igr_hdg_runtime_app_command_line_test>"
      "-DHDG_RUNTIME_SCALAR_RANK2_MANIFEST_CONSUMER=$<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cli-scalar-igr-cuda-rank2"
      -DHDG_RUNTIME_FUNCTIONAL=TRUE
      -DHDG_RUNTIME_SCALAR_IGR=TRUE
      -DHDG_RUNTIME_RANKS=2
      -DHDG_RUNTIME_BACKEND=cuda
      "-DHDG_RUNTIME_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_RUNTIME_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_RUNTIME_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_RUNTIME_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_process.cmake")
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_process_cuda_rank1
    dg_igr_hdg_runtime_app_cli_process_cuda_rank2
    dg_igr_hdg_runtime_app_cli_functional_cuda_rank1
    dg_igr_hdg_runtime_app_cli_functional_cuda_rank2
    dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank1
    dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank2
    PROPERTIES
      LABELS "hdg;runtime;app;cli;functional;cuda"
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_process_cuda_rank2
    dg_igr_hdg_runtime_app_cli_functional_cuda_rank2
    dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank2
    PROPERTIES PROCESSORS 2)
  # The measured pre-repair scalar tests were 277.58s (rank 1) and 916.58s
  # (rank 2) for about three full-trajectory equivalents. The exact added
  # campaigns raise those totals to 7.45 and 11.9 equivalents respectively:
  # 3/6 full scheduled retries plus 1/2 terminal 19-step runs and 1/2
  # 10-step restart continuations. These rounded bounds retain roughly 2x the
  # proportional measured time without changing registration or resource use.
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank1
    PROPERTIES TIMEOUT 1800)
  set_tests_properties(
    dg_igr_hdg_runtime_app_cli_scalar_igr_cuda_rank2
    PROPERTIES TIMEOUT 7200)
endif()
