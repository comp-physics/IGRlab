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

add_executable(dg_igr_hdg_runtime_cuda_hash_exact
  test_hash_exact.cu)
target_include_directories(dg_igr_hdg_runtime_cuda_hash_exact
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_hdg_runtime_cuda_hash_exact
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_runtime_cuda_hash_exact
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_hdg_runtime_cuda_hash_exact
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
  NAME dg_igr_hdg_runtime_cuda_hash_exact
  COMMAND dg_igr_hdg_runtime_cuda_hash_exact)
set_tests_properties(dg_igr_hdg_runtime_cuda_hash_exact
  PROPERTIES
    LABELS "hdg;runtime;cuda;hash;exact"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 300)

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
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_tensor_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_session --engine tensor)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_tensor_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_session>
    --engine tensor
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_p2_tensor_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_session
    --p2 --engine tensor)
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_session_p2_tensor_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_session>
    --p2 --engine tensor
    ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_session_rank1
  dg_igr_hdg_runtime_cuda_generic_euler_session_p2_rank1
  dg_igr_hdg_runtime_cuda_generic_euler_session_negatives_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;ordinary;session"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_session_tensor_rank1
  dg_igr_hdg_runtime_cuda_generic_euler_session_p2_tensor_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;session"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_session_rank2
  dg_igr_hdg_runtime_cuda_generic_euler_session_p2_rank2
  dg_igr_hdg_runtime_cuda_generic_euler_session_negatives_rank2
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;ordinary;session;mpi;host-staged"
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_session_tensor_rank2
  dg_igr_hdg_runtime_cuda_generic_euler_session_p2_tensor_rank2
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;session;mpi;host-staged"
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)

# Qualification of IGR as an optional augmentation of the production generic
# CUDA Euler session.  This target deliberately links no SC25/P1 oracle.
add_executable(dg_igr_hdg_runtime_cuda_generic_euler_extension
  test_generic_euler_extension.cu)
target_include_directories(dg_igr_hdg_runtime_cuda_generic_euler_extension
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_hdg_runtime_cuda_generic_euler_extension
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_runtime_cuda_generic_euler_extension
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_hdg_runtime_cuda_generic_euler_extension
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

foreach(_order IN ITEMS 1 2 3)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_extension_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_extension
      --order ${_order})
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_extension_p${_order}_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}"
      "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_extension>
      --order ${_order}
      ${MPIEXEC_POSTFLAGS})
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_extension_tensor_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_extension
      --order ${_order} --engine tensor)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_extension_tensor_p${_order}_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}"
      "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_extension>
      --order ${_order} --engine tensor
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_extension_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;extension;ordinary;igr;conventional;p${_order};physical;periodic;source;initial-trace;lifecycle"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_extension_p${_order}_rank2
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;extension;ordinary;igr;conventional;p${_order};physical;periodic;source;initial-trace;lifecycle;mpi;host-staged"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_extension_tensor_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;extension;tensor;igr;conventional;p${_order};physical;periodic;source;initial-trace;lifecycle"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_extension_tensor_p${_order}_rank2
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;extension;tensor;igr;conventional;p${_order};physical;periodic;source;initial-trace;lifecycle;mpi;host-staged"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
endforeach()

# Direct rank-one CPU/ordinary-CUDA/tensor-CUDA field agreement reuses one
# trajectory fixture for all pairwise comparisons at p1-p3.  Rank-two engine
# correctness remains in the generic extension matrix above.
add_executable(dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
  test_cpu_cuda_numerical_alignment.cu)
target_include_directories(
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
  PRIVATE
    dg_igr_hdg_runtime_cuda
    dg_igr_hdg_generic_cpu
    mfem
    MPI::MPI_CXX)
target_compile_options(
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
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
foreach(_order IN ITEMS 1 2 3)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
      --order ${_order})
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;ordinary;tensor;igr;contact;physical;periodic;source;initial-trace;conservation;trajectory;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)
endforeach()

# The original entropy-wave convergence fixture is shared by CPU, ordinary
# CUDA, and tensor CUDA.  Each rank-one case forms the complete CPU reference
# first, selects MFEM CUDA once, then forms both CUDA engine ladders and
# compares backend-neutral global error/norm sequences at every h and dt.
add_executable(dg_igr_hdg_runtime_cuda_generic_euler_convergence
  test_generic_euler_convergence.cu)
target_include_directories(
  dg_igr_hdg_runtime_cuda_generic_euler_convergence
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/tests/hdg/runtime")
target_link_libraries(
  dg_igr_hdg_runtime_cuda_generic_euler_convergence
  PRIVATE
    dg_igr_hdg_runtime_cuda
    dg_igr_hdg_generic_cpu
    mfem
    MPI::MPI_CXX)
target_compile_options(
  dg_igr_hdg_runtime_cuda_generic_euler_convergence
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_convergence
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
foreach(_regularization IN ITEMS conventional igr)
  foreach(_order IN ITEMS 1 2 3)
    add_test(
      NAME dg_igr_hdg_runtime_cuda_generic_euler_convergence_${_regularization}_p${_order}_rank1
      COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
        --order ${_order} --regularization ${_regularization})
    set_tests_properties(
      dg_igr_hdg_runtime_cuda_generic_euler_convergence_${_regularization}_p${_order}_rank1
      PROPERTIES
        LABELS "hdg;runtime;cpu;cuda;generic-euler;convergence;spatial;backward-euler;ordinary;tensor;${_regularization};p${_order};alignment"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 7200)
  endforeach()
endforeach()

# CPU/ordinary-CUDA/tensor-CUDA alignment for the same bounded IGR
# quadrature candidate used by the CPU-only convergence registration.
foreach(_order IN ITEMS 1 2 3)
  math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
  math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_convergence_igr_reduced_quadrature_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
      --order ${_order} --regularization igr
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order})
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_convergence_igr_reduced_quadrature_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;convergence;spatial;backward-euler;ordinary;tensor;igr;quadrature-candidate;p${_order};alignment"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)
endforeach()

# Standalone test-only performance observability for ordinary and tensor FP64
# CUDA accepted-step lifecycles at p1-p3.  This is intentionally a build
# target, not a CTest: its default no-export 10,816-element workload performs
# one warmup and five measured samples.
add_executable(dg_igr_hdg_runtime_cuda_accepted_step_benchmark
  benchmark_generic_euler_accepted_step.cu)
find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" --no-optional-locks -C "${PROJECT_SOURCE_DIR}"
          rev-parse HEAD
  RESULT_VARIABLE _benchmark_git_revision_result
  OUTPUT_VARIABLE _benchmark_git_revision
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" --no-optional-locks -C "${PROJECT_SOURCE_DIR}"
          rev-parse HEAD^{tree}
  RESULT_VARIABLE _benchmark_git_tree_result
  OUTPUT_VARIABLE _benchmark_git_tree
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" --no-optional-locks -C "${PROJECT_SOURCE_DIR}"
          status --porcelain=v1 --untracked-files=all
  RESULT_VARIABLE _benchmark_git_status_result
  OUTPUT_VARIABLE _benchmark_git_status
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _benchmark_git_revision_result EQUAL 0 OR
   NOT _benchmark_git_tree_result EQUAL 0 OR
   NOT _benchmark_git_status_result EQUAL 0 OR
   _benchmark_git_revision STREQUAL "" OR _benchmark_git_tree STREQUAL "")
  message(FATAL_ERROR
    "The accepted-step benchmark requires an exact Git source identity")
endif()
if(_benchmark_git_status STREQUAL "")
  set(_benchmark_git_clean_state "clean")
else()
  set(_benchmark_git_clean_state "dirty")
endif()
target_compile_definitions(
  dg_igr_hdg_runtime_cuda_accepted_step_benchmark
  PRIVATE
    DG_IGR_BENCHMARK_SOURCE_REVISION="${_benchmark_git_revision}"
    DG_IGR_BENCHMARK_SOURCE_TREE="${_benchmark_git_tree}"
    DG_IGR_BENCHMARK_SOURCE_STATUS="${_benchmark_git_clean_state}"
    DG_IGR_BENCHMARK_BUILD_DIRECTORY="${CMAKE_BINARY_DIR}"
    DG_IGR_BENCHMARK_CMAKE_CACHE="${CMAKE_BINARY_DIR}/CMakeCache.txt")
target_include_directories(
  dg_igr_hdg_runtime_cuda_accepted_step_benchmark
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(
  dg_igr_hdg_runtime_cuda_accepted_step_benchmark
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    umpire
    MPI::MPI_CXX)
target_compile_options(
  dg_igr_hdg_runtime_cuda_accepted_step_benchmark
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(
  dg_igr_hdg_runtime_cuda_accepted_step_benchmark
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

# Opt-in static evidence gate.  It is not part of the default correctness
# matrix: invoke this custom target only after the production-linked benchmark
# binary has been built by the canonical build.sh CUDA route.
find_program(DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE
  NAMES cuobjdump
  HINTS
    "${CUDAToolkit_BIN_DIR}"
    "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin")
if(DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE)
  add_custom_target(dg_igr_hdg_runtime_cuda_tensor_sass_evidence
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime_cuda_accepted_step_benchmark>"
      "-DCUOBJDUMP_EXECUTABLE=${DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE}"
      "-DEXPECTED_KERNEL_SUBSTRING=TensorBorrowedCondenseKernel"
      "-DEVIDENCE_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/tensor-sass-evidence"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_generic_tensor_sass.cmake"
    DEPENDS dg_igr_hdg_runtime_cuda_accepted_step_benchmark
    COMMENT "Checking production fused tensor local-algebra FP64 SASS"
    VERBATIM)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_tensor_sass_evidence
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime_cuda_accepted_step_benchmark>"
      "-DCUOBJDUMP_EXECUTABLE=${DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE}"
      "-DEXPECTED_KERNEL_SUBSTRING=TensorBorrowedCondenseKernel"
      "-DEVIDENCE_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/tensor-sass-evidence"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_generic_tensor_sass.cmake")
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_tensor_sass_evidence
    PROPERTIES
      LABELS "hdg;runtime;cuda;tensor;sass"
      TIMEOUT 300)
else()
  message(STATUS
    "Generic tensor SASS evidence target disabled: cuobjdump was not found.")
endif()

# Borrowed generic sessions must use their request/mesh communicator.  Each
# invocation leaves one MPI_COMM_WORLD rank passive while the active subgroup
# performs a complete CUDA IGR transaction; the rank-2 case also proves an
# asymmetric-null mesh rejection before device setup.
add_executable(dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator
  test_generic_euler_subcommunicator.cu)
target_include_directories(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator
  PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator
  PRIVATE
    dg_igr_hdg_runtime_cuda
    mfem
    MPI::MPI_CXX)
target_compile_options(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator
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
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_rank1
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 1
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 3
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 2
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_tensor_rank1
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 1
    --engine tensor
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_tensor_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 3
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 2
    --engine tensor
    ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;igr;subcommunicator;rank1;mpi"
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_rank2
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;igr;subcommunicator;rank2;mpi;negative"
    PROCESSORS 3
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_tensor_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;subcommunicator;rank1;mpi;no-fallback"
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_tensor_rank2
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;subcommunicator;rank2;mpi;negative;host-staged;no-fallback"
    PROCESSORS 3
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)

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
    test_traced_scalar_igr_p1_private.cu
    "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.cpp")
  target_include_directories(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(dg_igr_hdg_traced_scalar_igr_p1_private_cuda_test
    PRIVATE
      dg_igr_hdg_sc25_oracle_cuda
      dg_igr_hdg_generic_cpu
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
    test_traced_scalar_igr_p1_local.cu
    "${PROJECT_SOURCE_DIR}/src/numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.cpp")
  target_include_directories(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(dg_igr_hdg_traced_scalar_igr_p1_local_cuda_test
    PRIVATE
      dg_igr_hdg_sc25_oracle_cuda
      dg_igr_hdg_generic_cpu
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
      dg_igr_hdg_sc25_oracle_cuda
      dg_igr_hdg_sc25_oracle_cpu
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
      "-DRUNTIME_ARCHIVE=$<TARGET_FILE:dg_igr_hdg_sc25_oracle_cuda>"
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
      dg_igr_hdg_sc25_oracle_cuda
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
      "-DRUNTIME_ARCHIVE=$<TARGET_FILE:dg_igr_hdg_sc25_oracle_cuda>"
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
    dg_igr_hdg_sc25_oracle_cuda mfem MPI::MPI_CXX)
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

endif()
set_tests_properties(
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged
  dg_igr_hdg_runtime_cuda_w8a_rank2_host_staged_token_negatives
  dg_igr_hdg_runtime_cuda_w8a_functional_rank2_host_staged
  PROPERTIES
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)
