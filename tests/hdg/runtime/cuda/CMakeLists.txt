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

set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank1
  dg_igr_hdg_runtime_cuda_w8a_rank1_token_negatives
  PROPERTIES
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged_token_negatives
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
