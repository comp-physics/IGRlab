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
endif()
set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged_token_negatives
  dg_igr_hdg_runtime_cuda_w8a_functional_rank2_host_staged
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
