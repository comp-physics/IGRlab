add_executable(dg_igr_hdg_runtime_cuda_w8a_leaf
  test_persistent_smooth_euler.cu)

target_include_directories(dg_igr_hdg_runtime_cuda_w8a_leaf
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(dg_igr_hdg_runtime_cuda_w8a_leaf
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    MPI::MPI_CXX)

target_compile_options(dg_igr_hdg_runtime_cuda_w8a_leaf
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(dg_igr_hdg_runtime_cuda_w8a_leaf
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES 120
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(dg_igr_hdg_runtime_cuda_generic_external_topology
  test_generic_external_topology.cu)
target_include_directories(dg_igr_hdg_runtime_cuda_generic_external_topology
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_hdg_runtime_cuda_generic_external_topology
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_runtime_cuda_generic_external_topology
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_hdg_runtime_cuda_generic_external_topology
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES 120
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_external_topology_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_external_topology)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_external_topology_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_external_topology>
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_external_topology_negatives_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_external_topology --negatives)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_external_topology_negatives_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_external_topology>
    --negatives
    ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_external_topology_rank1
  dg_igr_hdg_runtime_cuda_generic_external_topology_negatives_rank1
  PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_external_topology_rank2
  dg_igr_hdg_runtime_cuda_generic_external_topology_negatives_rank2
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)

set(_dg_igr_hdg_runtime_cuda_generic_euler_session_sources
  test_generic_euler_session.cu)
add_executable(dg_igr_hdg_runtime_cuda_generic_euler_session
  ${_dg_igr_hdg_runtime_cuda_generic_euler_session_sources})
target_include_directories(dg_igr_hdg_runtime_cuda_generic_euler_session
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_hdg_runtime_cuda_generic_euler_session
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_runtime_cuda_generic_euler_session
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_hdg_runtime_cuda_generic_euler_session
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES 120
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_session)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_session>
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_p2_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_session --p2)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_p2_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_session>
    --p2
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_negatives_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_session --negatives)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_negatives_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_session>
    --negatives
    ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_session_rank1
  dg_igr_hdg_runtime_cuda_generic_euler_session_p2_rank1
  dg_igr_hdg_runtime_cuda_generic_euler_session_negatives_rank1
  PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_session_rank2
  dg_igr_hdg_runtime_cuda_generic_euler_session_p2_rank2
  dg_igr_hdg_runtime_cuda_generic_euler_session_negatives_rank2
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)

get_target_property(_dg_igr_hdg_cuda_leaf_resolve
  dg_igr_hdg_runtime_cuda_w8a_leaf CUDA_RESOLVE_DEVICE_SYMBOLS)
if(NOT _dg_igr_hdg_cuda_leaf_resolve STREQUAL "ON")
  message(FATAL_ERROR
    "dg_igr_hdg_runtime_cuda_w8a_leaf must own final CUDA device linking.")
endif()

add_test(
  NAME dg_igr_hdg_runtime_cuda_w8a_rank1
  COMMAND dg_igr_hdg_runtime_cuda_w8a_leaf)
add_test(
  NAME dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_w8a_leaf>
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_w8a_rank1_token_negatives
  COMMAND dg_igr_hdg_runtime_cuda_w8a_leaf --token-negatives)
add_test(
  NAME dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged_token_negatives
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_w8a_leaf>
    --token-negatives
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_w8a_functional_rank1
  COMMAND dg_igr_hdg_runtime_cuda_w8a_leaf --functional)
add_test(
  NAME dg_igr_hdg_runtime_cuda_w8a_functional_rank2_host_staged
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_w8a_leaf>
    --functional
    ${MPIEXEC_POSTFLAGS})

set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank1
  dg_igr_hdg_runtime_cuda_w8a_rank1_token_negatives
  dg_igr_hdg_runtime_cuda_w8a_functional_rank1
  PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)

if(DG_IGR_ENABLE_HDG_RUNTIME AND MFEM_USE_CUDA)
  add_executable(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    test_traced_scalar_igr_p1_private.cu)
  target_include_directories(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    PRIVATE
      dg_igr_hdg_runtime_cuda
      dg_igr_hdg_runtime_cpu
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(
    NAME hdg_traced_scalar_igr_p1_private_cuda
    COMMAND dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test)
  set_tests_properties(hdg_traced_scalar_igr_p1_private_cuda
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1800
      LABELS "hdg;runtime;cuda;traced-igr;private;packed-batch")

  add_executable(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    test_traced_scalar_igr_p1_local.cu)
  target_include_directories(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    PRIVATE
      dg_igr_hdg_runtime_cuda
      dg_igr_hdg_runtime_cpu
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(
    NAME hdg_traced_scalar_igr_p1_local_cuda
    COMMAND dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test)
  set_tests_properties(hdg_traced_scalar_igr_p1_local_cuda
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1800
      LABELS "hdg;runtime;cuda;traced-igr;local-algebra")

  add_executable(dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test
    "${PROJECT_SOURCE_DIR}/tests/hdg/runtime/cpu/sc25_scalar_igr_independent_oracle.cpp"
    test_persistent_sc25_scalar_igr.cu)

  target_include_directories(dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/tests/hdg/runtime/cpu"
      "${CMAKE_CURRENT_SOURCE_DIR}")

  target_link_libraries(dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test
    PRIVATE
      dg_igr_hdg_runtime_cuda
      dg_igr_hdg_runtime_cpu
      dg_igr_hdg_auxiliary_cpu
      mfem
      MPI::MPI_CXX)

  target_compile_options(dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  set_target_properties(dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  add_test(
    NAME hdg_sc25_scalar_igr_cuda_rank1_c1
    COMMAND dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test --c1)

  add_test(
    NAME hdg_sc25_scalar_igr_cuda_rank1_c2
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/run_sc25_scalar_igr_rank1.cmake")

  find_program(DG_IGR_CUOBJDUMP_EXECUTABLE
    NAMES cuobjdump
    HINTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin"
    REQUIRED)
  find_program(DG_IGR_NM_EXECUTABLE NAMES nm llvm-nm REQUIRED)
  add_test(
    NAME hdg_sc25_scalar_igr_cuda_rank1_device_object
    COMMAND
      "${CMAKE_COMMAND}"
      "-DOBJECT_ROOT=${PROJECT_BINARY_DIR}"
      "-DRUNTIME_ARCHIVE=$<TARGET_FILE:dg_igr_hdg_runtime_cuda>"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cuda_rank1_test>"
      "-DCUOBJDUMP_EXECUTABLE=${DG_IGR_CUOBJDUMP_EXECUTABLE}"
      "-DNM_EXECUTABLE=${DG_IGR_NM_EXECUTABLE}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_sc25_scalar_igr_device_object.cmake")

  set_tests_properties(
    hdg_sc25_scalar_igr_cuda_rank1_c1
    hdg_sc25_scalar_igr_cuda_rank1_c2
    hdg_sc25_scalar_igr_cuda_rank1_device_object
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)

  add_executable(dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test
    test_persistent_sc25_scalar_igr_rank2.cu)

  target_include_directories(dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")

  target_link_libraries(dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test
    PRIVATE
      dg_igr_hdg_runtime_cuda
      mfem
      MPI::MPI_CXX)

  target_compile_options(dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  set_target_properties(dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  add_test(
    NAME hdg_sc25_scalar_igr_cuda_rank2_c3
    COMMAND
      "${MPIEXEC_EXECUTABLE}"
      "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test>
      ${MPIEXEC_POSTFLAGS})

  add_test(
    NAME hdg_sc25_scalar_igr_cuda_rank2_device_object
    COMMAND
      "${CMAKE_COMMAND}"
      "-DOBJECT_ROOT=${PROJECT_BINARY_DIR}"
      "-DRUNTIME_ARCHIVE=$<TARGET_FILE:dg_igr_hdg_runtime_cuda>"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cuda_rank2_test>"
      "-DCUOBJDUMP_EXECUTABLE=${DG_IGR_CUOBJDUMP_EXECUTABLE}"
      "-DNM_EXECUTABLE=${DG_IGR_NM_EXECUTABLE}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_sc25_scalar_igr_rank2_device_object.cmake")

  set_tests_properties(hdg_sc25_scalar_igr_cuda_rank2_c3
    PROPERTIES
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(hdg_sc25_scalar_igr_cuda_rank2_device_object
    PROPERTIES
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)

  add_executable(dg_igr_hdg_sc25_scalar_igr_cuda_rank2_smooth_transition_test
    test_persistent_sc25_scalar_igr_rank2_smooth_transition.cu)
  target_include_directories(
    dg_igr_hdg_sc25_scalar_igr_cuda_rank2_smooth_transition_test PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(
    dg_igr_hdg_sc25_scalar_igr_cuda_rank2_smooth_transition_test PRIVATE
    dg_igr_hdg_runtime_cuda mfem MPI::MPI_CXX)
  target_compile_options(
    dg_igr_hdg_sc25_scalar_igr_cuda_rank2_smooth_transition_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(
    dg_igr_hdg_sc25_scalar_igr_cuda_rank2_smooth_transition_test PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES 120 CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(
    NAME hdg_sc25_scalar_igr_cuda_rank2_w8c_smooth_transition
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_sc25_scalar_igr_cuda_rank2_smooth_transition_test>
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(hdg_sc25_scalar_igr_cuda_rank2_w8c_smooth_transition
    PROPERTIES PROCESSORS 2 RESOURCE_LOCK dg_igr_sm120 TIMEOUT 7200
    LABELS "hdg;runtime;cuda;rank2;one-gpu;host-staged;w8c;smooth-transition")

  add_executable(dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test
    test_traced_scalar_igr_p1_private_numerics.cu)
  target_include_directories(dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test
    PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test
    PRIVATE dg_igr_hdg_runtime_cuda mfem MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test
    PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120 CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME hdg_traced_scalar_igr_p1_private_numerics_cuda_rank1
    COMMAND dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test)
  add_test(NAME hdg_traced_scalar_igr_p1_private_numerics_cuda_rank2
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test>
      ${MPIEXEC_POSTFLAGS})
  add_test(NAME hdg_traced_scalar_igr_p1_private_numerics_cuda_rank2_zero_local
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_traced_scalar_igr_p1_private_numerics_cuda_test>
      --rank0-only ${MPIEXEC_POSTFLAGS})
  set_tests_properties(hdg_traced_scalar_igr_p1_private_numerics_cuda_rank1
    PROPERTIES PROCESSORS 1 RESOURCE_LOCK dg_igr_sm120 TIMEOUT 1800
      LABELS "hdg;runtime;cuda;traced-scalar-igr;p1;identity-prime;rank1")
  set_tests_properties(hdg_traced_scalar_igr_p1_private_numerics_cuda_rank2
    hdg_traced_scalar_igr_p1_private_numerics_cuda_rank2_zero_local
    PROPERTIES PROCESSORS 2 RESOURCE_LOCK dg_igr_sm120 TIMEOUT 1800
      LABELS "hdg;runtime;cuda;traced-scalar-igr;p1;identity-prime;rank2")
endif()
set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged_token_negatives
  dg_igr_hdg_runtime_cuda_w8a_functional_rank2_host_staged
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
