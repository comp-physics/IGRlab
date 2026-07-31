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
endif()
