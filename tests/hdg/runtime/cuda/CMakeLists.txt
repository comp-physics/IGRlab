if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "persistent CUDA W8A tests are parent-build-only")
endif()
if(NOT MFEM_USE_CUDA OR NOT MFEM_USE_MPI OR NOT MFEM_USE_UMPIRE OR
   NOT TARGET umpire)
  message(FATAL_ERROR
    "persistent CUDA W8A tests require CUDA+MPI+MFEM/HYPRE Umpire device policy")
endif()

set(_dg_igr_cuda_w8a_source
  ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/runtime/cuda/persistent_smooth_euler.cu)
add_library(dg_igr_hdg_runtime_cuda_w8a_candidate STATIC
  ${_dg_igr_cuda_w8a_source})
add_executable(dg_igr_hdg_runtime_cuda_w8a_leaf
  test_persistent_smooth_euler.cu)
foreach(_target IN ITEMS
    dg_igr_hdg_runtime_cuda_w8a_candidate
    dg_igr_hdg_runtime_cuda_w8a_leaf)
  target_include_directories(${_target} PRIVATE
    ${DG_IGR_HDG_REPOSITORY_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${_target} PRIVATE igr_mfem mfem umpire)
  target_compile_options(${_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-Werror>)
  set_target_properties(${_target} PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES 120 CUDA_SEPARABLE_COMPILATION ON)
endforeach()
target_link_libraries(dg_igr_hdg_runtime_cuda_w8a_leaf PRIVATE
  dg_igr_hdg_runtime_cuda_w8a_candidate)

add_test(NAME dg_igr_hdg_runtime_cuda_w8a_rank1
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:dg_igr_hdg_runtime_cuda_w8a_leaf>
          ${MPIEXEC_POSTFLAGS})
add_test(NAME dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:dg_igr_hdg_runtime_cuda_w8a_leaf>
          ${MPIEXEC_POSTFLAGS})
add_test(NAME dg_igr_hdg_runtime_cuda_w8a_token_negatives
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:dg_igr_hdg_runtime_cuda_w8a_leaf>
          --token-negatives ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank1
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  dg_igr_hdg_runtime_cuda_w8a_token_negatives
  PROPERTIES
    LABELS "hdg;runtime;cuda;sm120;p1;n3;persistent;source-candidate"
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
set_tests_properties(dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  PROPERTIES PROCESSORS 2)

# Parent add_subdirectory/test registration is intentionally coordinator-owned.
