add_executable(dg_igr_hdg_runtime_cpu_test
  independent_oracle.cpp
  test_persistent_smooth_euler.cpp)

if(MFEM_USE_CUDA)
  set_source_files_properties(independent_oracle.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(dg_igr_hdg_runtime_cpu_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
endif()

target_include_directories(dg_igr_hdg_runtime_cpu_test
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(dg_igr_hdg_runtime_cpu_test
  PRIVATE
    dg_igr_hdg_generic_cpu
    mfem
    MPI::MPI_CXX)

target_compile_options(dg_igr_hdg_runtime_cpu_test
  PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)

set_target_properties(dg_igr_hdg_runtime_cpu_test
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

if(MFEM_USE_CUDA)
  set_target_properties(dg_igr_hdg_runtime_cpu_test PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  get_target_property(_dg_igr_hdg_runtime_cpu_test_resolve
                      dg_igr_hdg_runtime_cpu_test
                      CUDA_RESOLVE_DEVICE_SYMBOLS)
  if(NOT _dg_igr_hdg_runtime_cpu_test_resolve STREQUAL "ON")
    message(FATAL_ERROR
            "dg_igr_hdg_runtime_cpu_test must own CUDA device-symbol resolution in CUDA profiles.")
  endif()
endif()

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
    dg_igr_hdg_generic_cpu
    dg_igr_hdg_sc25_oracle_cpu
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
    dg_igr_hdg_generic_cpu
    dg_igr_hdg_sc25_oracle_cpu
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

  # The retired scalar card is never exercised through the production CLI.
  # Its frozen W8C implementation remains reachable only by the direct oracle
  # target above; comparative scientific-efficacy matrices are held until the
  # general-purpose reference solver is promoted.

  add_executable(dg_igr_hdg_n8_generic_reference_transfer_test
    euler_committed_observable_oracle.cpp
    test_n8_generic_reference_transfer.cpp)
  target_include_directories(dg_igr_hdg_n8_generic_reference_transfer_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu"
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(dg_igr_hdg_n8_generic_reference_transfer_test
    PRIVATE
      dg_igr_hdg_generic_cpu
      dg_igr_hdg_sc25_oracle_cpu
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_n8_generic_reference_transfer_test
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(dg_igr_hdg_n8_generic_reference_transfer_test
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF)

  add_test(NAME hdg_n8_generic_reference_transfer_rank1
    COMMAND dg_igr_hdg_n8_generic_reference_transfer_test)
  add_test(NAME hdg_n8_generic_reference_transfer_rank2
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_n8_generic_reference_transfer_test>
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    hdg_n8_generic_reference_transfer_rank1
    hdg_n8_generic_reference_transfer_rank2
    PROPERTIES
      LABELS "hdg;generic;sc25;reference-transfer;n8;cpu;igr"
      RESOURCE_LOCK hdg_reference_transfer_cpu
      TIMEOUT 21600)
  set_tests_properties(hdg_n8_generic_reference_transfer_rank2
    PROPERTIES PROCESSORS 2)
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

add_executable(dg_igr_hdg_generic_euler_cpu_test
  test_generic_euler.cpp)
target_include_directories(dg_igr_hdg_generic_euler_cpu_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_hdg_generic_euler_cpu_test PRIVATE
  dg_igr_hdg_generic_cpu mfem MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_generic_euler_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_generic_euler_cpu_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

foreach(_ranks IN ITEMS 1 2 3)
  if(_ranks EQUAL 1)
    add_test(NAME hdg_generic_euler_cpu_rank1
      COMMAND dg_igr_hdg_generic_euler_cpu_test)
  else()
    add_test(NAME hdg_generic_euler_cpu_rank${_ranks}
      COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" ${_ranks}
              ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:dg_igr_hdg_generic_euler_cpu_test>
              ${MPIEXEC_POSTFLAGS})
  endif()
  set_tests_properties(hdg_generic_euler_cpu_rank${_ranks} PROPERTIES
    PROCESSORS ${_ranks}
    RESOURCE_LOCK hdg_generic_euler_cpu
    LABELS "hdg;generic;cpu;mpi"
    TIMEOUT 180)
endforeach()

add_executable(dg_igr_hdg_generic_euler_convergence_cpu_test
  test_generic_euler_convergence.cpp)
target_include_directories(dg_igr_hdg_generic_euler_convergence_cpu_test
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/tests/hdg/runtime")
target_link_libraries(dg_igr_hdg_generic_euler_convergence_cpu_test PRIVATE
  dg_igr_hdg_generic_cpu mfem MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_generic_euler_convergence_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_generic_euler_convergence_cpu_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
foreach(_regularization IN ITEMS conventional igr)
  foreach(_order IN ITEMS 1 2 3)
    add_test(
      NAME hdg_generic_euler_cpu_convergence_${_regularization}_p${_order}
      COMMAND dg_igr_hdg_generic_euler_convergence_cpu_test
        --order ${_order} --regularization ${_regularization})
    set_tests_properties(
      hdg_generic_euler_cpu_convergence_${_regularization}_p${_order}
      PROPERTIES
        PROCESSORS 1
        RESOURCE_LOCK hdg_generic_euler_cpu
        LABELS "hdg;generic;cpu;convergence;spatial;backward-euler;${_regularization};p${_order}"
        TIMEOUT 3600)
  endforeach()
endforeach()

# Reuse the production convergence fixture to qualify the bounded IGR
# quadrature candidate: one extra polynomial degree of volume overintegration
# beyond the conventional face rule.  Keeping this as a parameterized case
# makes the q12 default an explicit control until the candidate is promoted.
foreach(_order IN ITEMS 1 2 3)
  math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
  math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
  add_test(
    NAME hdg_generic_euler_cpu_convergence_igr_reduced_quadrature_p${_order}
    COMMAND dg_igr_hdg_generic_euler_convergence_cpu_test
      --order ${_order} --regularization igr
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order})
  set_tests_properties(
    hdg_generic_euler_cpu_convergence_igr_reduced_quadrature_p${_order}
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK hdg_generic_euler_cpu
      LABELS "hdg;generic;cpu;convergence;spatial;backward-euler;igr;quadrature-candidate;p${_order}"
      TIMEOUT 3600)
endforeach()

# Directly compare the reduced IGR quadrature schedule to q12/q12 on the
# same active-IGR one-step and multi-step trajectories.  This remains a
# test-only numerical qualification and publishes no runtime evidence API.
foreach(_order IN ITEMS 1 2 3)
  math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
  math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
  add_test(
    NAME hdg_generic_euler_cpu_igr_quadrature_equivalence_p${_order}
    COMMAND dg_igr_hdg_generic_euler_convergence_cpu_test
      --order ${_order} --regularization igr
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order}
      --quadrature-equivalence)
  set_tests_properties(
    hdg_generic_euler_cpu_igr_quadrature_equivalence_p${_order}
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK hdg_generic_euler_cpu
      LABELS "hdg;generic;cpu;igr;quadrature-equivalence;p${_order}"
      TIMEOUT 3600)
endforeach()

add_executable(dg_igr_hdg_traced_scalar_igr_p1_local_test
  test_traced_scalar_igr_p1_local.cpp
  "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.cpp")
target_include_directories(dg_igr_hdg_traced_scalar_igr_p1_local_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_hdg_traced_scalar_igr_p1_local_test PRIVATE
  dg_igr_hdg_generic_cpu mfem MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_traced_scalar_igr_p1_local_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_traced_scalar_igr_p1_local_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
add_test(NAME hdg_traced_scalar_igr_p1_local
  COMMAND dg_igr_hdg_traced_scalar_igr_p1_local_test)
set_tests_properties(hdg_traced_scalar_igr_p1_local PROPERTIES
  PROCESSORS 1 RESOURCE_LOCK hdg_traced_scalar_igr_p1_local TIMEOUT 180
  LABELS "hdg;traced-scalar-igr;local;cpu")

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
      dg_igr_hdg_sc25_oracle_cpu
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

endif()
