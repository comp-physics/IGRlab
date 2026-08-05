add_executable(dg_igr_hdg_runtime_cpu_test
  independent_oracle.cpp
  test_persistent_smooth_euler.cpp)

target_include_directories(dg_igr_hdg_runtime_cpu_test
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(dg_igr_hdg_runtime_cpu_test
  PRIVATE
    dg_igr_hdg_runtime_cpu
    mfem
    MPI::MPI_CXX)

target_compile_options(dg_igr_hdg_runtime_cpu_test
  PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)

set_target_properties(dg_igr_hdg_runtime_cpu_test
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

add_test(
  NAME hdg_runtime_cpu_serial
  COMMAND dg_igr_hdg_runtime_cpu_test)
set_tests_properties(hdg_runtime_cpu_serial
  PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK hdg_runtime_cpu
    TIMEOUT 1800)

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  add_executable(dg_igr_hdg_runtime_w8c_cpu_serial_test
    independent_oracle.cpp
    test_w8c_cpu_serial.cpp)
  target_include_directories(dg_igr_hdg_runtime_w8c_cpu_serial_test PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(dg_igr_hdg_runtime_w8c_cpu_serial_test PRIVATE
    dg_igr_hdg_runtime_cpu
    dg_igr_hdg_auxiliary_cpu
    mfem
    MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_runtime_w8c_cpu_serial_test PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(dg_igr_hdg_runtime_w8c_cpu_serial_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  add_test(NAME hdg_runtime_w8c_cpu_serial
    COMMAND dg_igr_hdg_runtime_w8c_cpu_serial_test)
  set_tests_properties(hdg_runtime_w8c_cpu_serial PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK hdg_runtime_cpu
    LABELS "hdg;runtime;cpu;w8c"
    TIMEOUT 1800)

  add_executable(dg_igr_hdg_runtime_w8c_smooth_transition_cpu_serial_test
    independent_oracle.cpp
    sc25_scalar_igr_independent_oracle.cpp
    test_w8c_smooth_transition_cpu_serial.cpp)
  target_include_directories(
    dg_igr_hdg_runtime_w8c_smooth_transition_cpu_serial_test PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(
    dg_igr_hdg_runtime_w8c_smooth_transition_cpu_serial_test PRIVATE
    dg_igr_hdg_runtime_cpu
    dg_igr_hdg_auxiliary_cpu
    mfem
    MPI::MPI_CXX)
  target_compile_options(
    dg_igr_hdg_runtime_w8c_smooth_transition_cpu_serial_test PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(
    dg_igr_hdg_runtime_w8c_smooth_transition_cpu_serial_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  add_test(NAME hdg_runtime_w8c_smooth_transition_cpu_serial
    COMMAND dg_igr_hdg_runtime_w8c_smooth_transition_cpu_serial_test)
  set_tests_properties(hdg_runtime_w8c_smooth_transition_cpu_serial PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK hdg_runtime_cpu
    LABELS "hdg;runtime;cpu;w8c;smooth-transition"
    TIMEOUT 1800)

  add_test(NAME hdg_runtime_w8c_smooth_transition_cli_cpu_serial
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_RUNTIME_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime>"
      "-DHDG_RUNTIME_TEST_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/w8c-smooth-transition-cli"
      -DHDG_RUNTIME_W8C_SMOOTH=TRUE
      -DHDG_RUNTIME_RANKS=1
      -P "${PROJECT_SOURCE_DIR}/tests/hdg/runtime/app/test_cli_process.cmake")
  set_tests_properties(hdg_runtime_w8c_smooth_transition_cli_cpu_serial
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK hdg_runtime_cpu
      LABELS "hdg;runtime;app;cli;cpu;w8c;smooth-transition"
      TIMEOUT 3600)
endif()

add_test(
  NAME hdg_runtime_cpu_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cpu_test>
    ${MPIEXEC_POSTFLAGS})
set_tests_properties(hdg_runtime_cpu_rank2
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK hdg_runtime_cpu
    TIMEOUT 1800)

foreach(_n IN ITEMS 3 6 12)
  add_test(
    NAME hdg_runtime_cpu_functional_serial_n${_n}
    COMMAND dg_igr_hdg_runtime_cpu_test --functional-n ${_n})
  set_tests_properties(hdg_runtime_cpu_functional_serial_n${_n}
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK hdg_runtime_cpu
      TIMEOUT 3600)
endforeach()

add_test(
  NAME hdg_runtime_cpu_solver_linked_convergence
  COMMAND dg_igr_hdg_runtime_cpu_test --scientific-convergence)
set_tests_properties(hdg_runtime_cpu_solver_linked_convergence
  PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK hdg_runtime_cpu
    TIMEOUT 3600)

add_test(
  NAME hdg_runtime_cpu_functional_rank2_n6
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cpu_test>
    --functional-n 6
    ${MPIEXEC_POSTFLAGS})
set_tests_properties(hdg_runtime_cpu_functional_rank2_n6
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK hdg_runtime_cpu
    TIMEOUT 3600)

if(DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  add_library(dg_igr_hdg_sc25_scalar_igr_strict_cxx17 OBJECT
    "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.cpp")
  target_include_directories(dg_igr_hdg_sc25_scalar_igr_strict_cxx17
    PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_hdg_sc25_scalar_igr_strict_cxx17
    PRIVATE mfem MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_sc25_scalar_igr_strict_cxx17 PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror;-pedantic-errors>)
  set_target_properties(dg_igr_hdg_sc25_scalar_igr_strict_cxx17 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

  add_executable(dg_igr_hdg_sc25_scalar_igr_cpu_test
    sc25_scalar_igr_independent_oracle.cpp
    test_persistent_sc25_scalar_igr.cpp)

  target_include_directories(dg_igr_hdg_sc25_scalar_igr_cpu_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")

  target_link_libraries(dg_igr_hdg_sc25_scalar_igr_cpu_test
    PRIVATE
      dg_igr_hdg_runtime_cpu
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)

  target_compile_options(dg_igr_hdg_sc25_scalar_igr_cpu_test
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)

  set_target_properties(dg_igr_hdg_sc25_scalar_igr_cpu_test
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF)

  add_test(
    NAME hdg_sc25_scalar_igr_cpu_rank1_foundation
    COMMAND dg_igr_hdg_sc25_scalar_igr_cpu_test)
  set_tests_properties(hdg_sc25_scalar_igr_cpu_rank1_foundation
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK hdg_sc25_scalar_igr_cpu
      TIMEOUT 1800)

  add_test(
    NAME hdg_sc25_scalar_igr_cpu_rank1_trajectory
    COMMAND dg_igr_hdg_sc25_scalar_igr_cpu_test --trajectory)
  set_tests_properties(hdg_sc25_scalar_igr_cpu_rank1_trajectory
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK hdg_sc25_scalar_igr_cpu
      TIMEOUT 3600)

  add_test(
    NAME hdg_sc25_scalar_igr_cpu_rank2_trajectory
    COMMAND
      "${MPIEXEC_EXECUTABLE}"
      "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cpu_test>
      --rank2-trajectory
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(hdg_sc25_scalar_igr_cpu_rank2_trajectory
    PROPERTIES
      PROCESSORS 2
      RESOURCE_LOCK hdg_sc25_scalar_igr_cpu
      TIMEOUT 3600)

  add_executable(dg_igr_hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    emit_sc25_scalar_igr_rank2_manifest.cpp)

  target_include_directories(
    dg_igr_hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/tests/hdg/runtime/common")

  target_link_libraries(
    dg_igr_hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    PRIVATE
      dg_igr_hdg_runtime_cpu
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)

  target_compile_options(
    dg_igr_hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)

  set_target_properties(
    dg_igr_hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF)

  add_test(
    NAME hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    COMMAND
      "${MPIEXEC_EXECUTABLE}"
      "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cpu_rank2_manifest_producer>
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(hdg_sc25_scalar_igr_cpu_rank2_manifest_producer
    PROPERTIES
      PROCESSORS 2
      RESOURCE_LOCK hdg_sc25_scalar_igr_cpu
      TIMEOUT 3600)
endif()
