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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Bounded direct numerical/architecture screens for the typed common-shift ROW
# path. Rank one covers native tensor and ordinary CUDA. The two rank-two tests
# independently require the established host-staged HYPRE transport for both
# engines while exercising conventional and IGR candidates transactionally.
add_test(
  NAME dg_igr_hdg_runtime_cuda_rodas3_tensor_rank1_architecture_screen
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_extension
    --order 1 --engine tensor --time-treatment rodas3 --rodas3-screen)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_rodas3_tensor_rank1_architecture_screen
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;rodas3;tensor;rank1;conventional;igr;common-shift;architecture-screen;zhang-shu;limiter;admissibility;closure"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1800)

foreach(_engine IN ITEMS ordinary tensor)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_rodas3_${_engine}_rank2_architecture_screen
    COMMAND
      "${MPIEXEC_EXECUTABLE}"
      "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_extension>
      --order 1 --engine ${_engine} --time-treatment rodas3
      --rodas3-rank2-screen
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_rodas3_${_engine}_rank2_architecture_screen
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;rodas3;${_engine};rank2;mpi;host-staged;conventional;igr;common-shift;architecture-screen;rollback"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1800)
endforeach()

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

# Focused productionization gate for the typed partitioned Backward-Euler
# lifecycle.  Reuse the same conventional-degeneration, periodic/physical IGR,
# and transactional fixtures at every supported order, engine, and rank
# topology without duplicating backend-specific test code.
foreach(_order IN ITEMS 1 2 3)
  foreach(_engine IN ITEMS ordinary tensor)
    set(_engine_args)
    if(_engine STREQUAL "tensor")
      list(APPEND _engine_args --engine tensor)
    endif()
    add_test(
      NAME dg_igr_hdg_runtime_cuda_partitioned_backward_euler_p${_order}_${_engine}_rank1
      COMMAND dg_igr_hdg_runtime_cuda_generic_euler_extension
        --order ${_order} ${_engine_args}
        --time-treatment partitioned-backward-euler --partitioned-smoke)
    add_test(
      NAME dg_igr_hdg_runtime_cuda_partitioned_backward_euler_p${_order}_${_engine}_rank2
      COMMAND
        "${MPIEXEC_EXECUTABLE}"
        "${MPIEXEC_NUMPROC_FLAG}" 2
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_extension>
        --order ${_order} ${_engine_args}
        --time-treatment partitioned-backward-euler --partitioned-smoke
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
      dg_igr_hdg_runtime_cuda_partitioned_backward_euler_p${_order}_${_engine}_rank1
      PROPERTIES
        LABELS "hdg;runtime;cuda;generic-euler;partitioned-backward-euler;${_engine};igr;conventional;p${_order};physical;periodic;lifecycle"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 3600)
    set_tests_properties(
      dg_igr_hdg_runtime_cuda_partitioned_backward_euler_p${_order}_${_engine}_rank2
      PROPERTIES
        LABELS "hdg;runtime;cuda;generic-euler;partitioned-backward-euler;${_engine};igr;conventional;p${_order};physical;periodic;lifecycle;mpi;host-staged"
        PROCESSORS 2
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 3600)
  endforeach()
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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

# RODAS3 reuses the complete CPU/ordinary-CUDA/tensor-CUDA alignment fixture,
# including conventional and IGR trajectories, prepared assessment, and exact
# committed-state agreement at every supported polynomial order.
foreach(_order IN ITEMS 1 2 3)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rodas3_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
      --order ${_order} --time-treatment rodas3)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rodas3_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;rodas3;ordinary;tensor;igr;conventional;physical;periodic;source;initial-trace;conservation;assessment;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
endforeach()

# Winner-specific cross-backend qualification reuses the complete bounded
# six-case trajectory fixture.  This covers CPU, ordinary CUDA, and tensor
# CUDA pairwise agreement at every supported polynomial order while retaining
# the registrations above as the fully coupled Backward-Euler controls.
foreach(_order IN ITEMS 1 2 3)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_partitioned_backward_euler_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
      --order ${_order}
      --time-treatment partitioned-backward-euler)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_partitioned_backward_euler_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;partitioned-backward-euler;ordinary;tensor;igr;contact;physical;periodic;source;initial-trace;conservation;trajectory;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)
endforeach()

# Exact tensor-IGR production candidate: retain all six conventional/IGR,
# periodic/physical, supplied/source, and localized-contact cases in the same
# shared CPU/ordinary-CUDA/tensor-CUDA fixture as the legacy controls above.
# The requested engine-default policy must resolve to the qualified rank-one
# tensor-IGR trace leaf; explicit policy selectors remain ablation controls.
foreach(_order IN ITEMS 1 2 3)
  math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
  math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
  # Conventional cells use the same explicit role tolerances as the campaign
  # candidate (and no IGR quadrature override).  Bind that complete policy to
  # the shared spatial and Backward-Euler temporal fixture as well.
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_convergence_conventional_candidate_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
      --order ${_order} --regularization conventional
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_convergence_conventional_candidate_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;convergence;spatial;backward-euler;ordinary;tensor;conventional;role-tolerance-candidate;production-trace-policy;p${_order};alignment"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)

  add_test(
    NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_candidate_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
      --order ${_order}
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order}
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12
      --tensor-igr-trace-policy engine-default)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_candidate_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;ordinary;tensor;igr;contact;physical;periodic;source;initial-trace;conservation;trajectory;quadrature-candidate;role-tolerance-candidate;production-trace-policy;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)
endforeach()

# Explicit coupled full-patch ablations reuse the complete six-case
# CPU/ordinary-CUDA/tensor-CUDA alignment fixture.  These are qualification
# registrations only: application engine-default routing is unchanged.
set(_dg_igr_coupled_trace_policies
  native-element-schwarz-matrix-free-warp-coupled-fp64
  native-element-schwarz-matrix-free-warp-coupled-sigma-jacobi2-fp64
  native-element-schwarz-matrix-free-warp-coupled-sigma-jacobi10-fp64)
foreach(_coupled_policy IN LISTS _dg_igr_coupled_trace_policies)
  string(REPLACE "native-element-schwarz-matrix-free-warp-coupled-" ""
    _coupled_policy_suffix "${_coupled_policy}")
  string(REPLACE "-" "_" _coupled_policy_suffix
    "${_coupled_policy_suffix}")
  foreach(_order IN ITEMS 1 2 3)
    math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
    math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
    add_test(
      NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_coupled_${_coupled_policy_suffix}_p${_order}_rank1
      COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
        --order ${_order}
        --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
        --igr-face-quadrature-order ${_igr_face_quadrature_order}
        --newton-linear-rtol 1e-10
        --final-linear-rtol 3e-11
        --true-residual-tightening-linear-rtol 1e-12
        --tensor-igr-trace-policy ${_coupled_policy})
    set_tests_properties(
      dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_coupled_${_coupled_policy_suffix}_p${_order}_rank1
      PROPERTIES
        LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;ordinary;tensor;igr;contact;physical;periodic;source;initial-trace;conservation;trajectory;quadrature-candidate;role-tolerance-candidate;coupled-full-patch;ablation;p${_order}"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 7200)
  endforeach()
endforeach()

# Explicit robust J10 mirrors the promoted engine-default policy.  Reuse the
# full active six-case alignment fixture and require its fast lower branch at
# p1-p3 with the bounded quadrature/tolerance candidate.
set(_dg_igr_robust_trace_policy
  native-element-schwarz-matrix-free-warp-robust-coupled-sigma-jacobi10-fp64)
foreach(_order IN ITEMS 1 2 3)
  math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
  math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
  add_test(
    NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_robust_jacobi10_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
      --order ${_order}
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order}
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12
      --tensor-igr-trace-policy ${_dg_igr_robust_trace_policy})
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_robust_jacobi10_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;ordinary;tensor;igr;contact;physical;periodic;source;initial-trace;conservation;trajectory;quadrature-candidate;role-tolerance-candidate;robust-adaptive;lower-branch;production-trace-policy;explicit-policy-control;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)
endforeach()

# Parse and validate candidate policies without selecting a CUDA device.  The
# explicit warp selector below is a private ablation/parser control, separate
# from the engine-default production candidate above.  These fast checks cover
# malformed input before the shared numerical suite starts.
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_options_default
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_options_candidate
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --order 3
    --igr-volume-quadrature-order 10
    --igr-face-quadrature-order 8
    --newton-linear-rtol 1e-10
    --final-linear-rtol 3e-11
    --true-residual-tightening-linear-rtol 1e-12
    --tensor-igr-trace-policy
      native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64
    --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_nonfinite_tolerance
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --newton-linear-rtol nan --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_nonpositive_tolerance
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --final-linear-rtol 0 --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_duplicate_tolerance
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --newton-linear-rtol 1e-10 --newton-linear-rtol 1e-10
    --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_invalid_tolerance_order
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --newton-linear-rtol 1e-10 --final-linear-rtol 3e-11
    --true-residual-tightening-linear-rtol 1e-9
    --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_incomplete_quadrature
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --igr-volume-quadrature-order 10 --validate-options-only)
add_test(
  NAME dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_duplicate_trace_policy
  COMMAND dg_igr_hdg_runtime_cuda_cpu_numerical_alignment
    --tensor-igr-trace-policy engine-default
    --tensor-igr-trace-policy common-fast --validate-options-only)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_options_default
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_options_candidate
  PROPERTIES
    LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;argument-validation"
    PROCESSORS 1
    TIMEOUT 30)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_nonfinite_tolerance
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_nonpositive_tolerance
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_duplicate_tolerance
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_invalid_tolerance_order
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_incomplete_quadrature
  dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_rejects_duplicate_trace_policy
  PROPERTIES
    LABELS "hdg;runtime;cpu;cuda;generic-euler;alignment;argument-validation"
    PROCESSORS 1
    TIMEOUT 30
    WILL_FAIL TRUE)

# The original entropy-wave convergence fixture is shared by CPU, ordinary
# CUDA, and tensor CUDA.  Each rank-one case forms the complete CPU reference
# first, selects MFEM CUDA once, then forms both CUDA engine ladders and
# compares backend-neutral global error/norm sequences at every h and dt.
add_executable(dg_igr_hdg_runtime_cuda_generic_euler_convergence
  test_generic_euler_convergence.cu
  "${PROJECT_SOURCE_DIR}/src/sim/hdg/parallel_vtk.cpp")
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Test-only, rank-one tensor decision harness.  It keeps the established
# convergence registrations below unchanged, executes no output path, and
# compares fixed-horizon RODAS3, monolithic BE, and partitioned BE on an
# evolving smooth state with active IGR closure.  U4 error is measured against
# a bounded same-mesh RODAS3 reference and kept above its Richardson
# uncertainty.  The production C_alpha=1 average-length lowering is mandatory;
# the engine-default trace policy remains the default and explicit selectors
# remain available only for bounded ablations.
add_test(
  NAME dg_igr_hdg_runtime_cuda_rodas3_equal_error_probe_igr_p3_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
    --order 3 --regularization igr
    --architecture-selection-screen
    --screen-probe-only
    --screen-elements 12
    --screen-final-time 0.08)
add_test(
  NAME dg_igr_hdg_runtime_cuda_rodas3_equal_error_igr_p3_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
    --order 3 --regularization igr
    --architecture-selection-screen
    --screen-elements 12
    --screen-final-time 0.08)
add_test(
  NAME dg_igr_hdg_runtime_cuda_rodas3_equal_error_with_output_igr_p3_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
    --order 3 --regularization igr
    --architecture-selection-screen
    --screen-elements 12
    --screen-final-time 0.08
    --screen-output-root
      "${CMAKE_CURRENT_BINARY_DIR}/rodas3-equal-error-output")
add_test(
  NAME dg_igr_hdg_runtime_cuda_rodas3_adaptive_equal_error_with_output_igr_p3_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
    --order 3 --regularization igr
    --architecture-selection-screen
    --adaptive-rodas3-screen
    --screen-elements 12
    --screen-final-time 0.08
    --screen-time-atol 1.0e-10
    --screen-time-rtol 1.0e-6
    --screen-initial-dt 0.08
    --screen-dt-min 1.0e-6
    --screen-output-root
      "${CMAKE_CURRENT_BINARY_DIR}/rodas3-adaptive-equal-error-output")
set_tests_properties(
  dg_igr_hdg_runtime_cuda_rodas3_equal_error_probe_igr_p3_rank1
  dg_igr_hdg_runtime_cuda_rodas3_equal_error_igr_p3_rank1
  dg_igr_hdg_runtime_cuda_rodas3_equal_error_with_output_igr_p3_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;active-igr;corrected-alpha;same-mesh-reference;rodas3;backward-euler;partitioned-backward-euler;equal-error;architecture-selection;performance;manual;p3"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_rodas3_adaptive_equal_error_with_output_igr_p3_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;active-igr;corrected-alpha;same-mesh-reference;rodas3;adaptive;backward-euler;partitioned-backward-euler;equal-error;architecture-selection;performance;p3"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    ENVIRONMENT "DG_IGR_RUN_ARCHITECTURE_SELECTION_SCREEN=1"
    TIMEOUT 3600)

# Final-selection whole-device memory controls.  The driver returns CTest skip
# status 77 unless the exact selected steps/controller tuple, selected GPU UUID,
# and CUDA_VISIBLE_DEVICES are supplied by the tracked build.sh invocation.
add_test(
  NAME dg_igr_hdg_runtime_cuda_adaptive_trace_memory_matrix_igr_p3_rank1
  COMMAND /usr/bin/env python3
    "${CMAKE_SOURCE_DIR}/scripts/perf/adaptive_trace_memory_matrix.py"
    --executable
      "$<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_convergence>"
    --nvidia-smi nvidia-smi)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_adaptive_trace_memory_matrix_igr_p3_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;rodas3;adaptive;backward-euler;partitioned-backward-euler;architecture-selection;memory;performance;manual;p3"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)

# Startup-inclusive timing is a separate no-poller matrix.  Each method runs
# in three fresh processes with initial/midpoint/final output and the same
# externally supplied equal-error controls used by the memory qualification.
add_test(
  NAME dg_igr_hdg_runtime_cuda_adaptive_trace_fresh_process_timing_igr_p3_rank1
  COMMAND /usr/bin/env python3
    "${CMAKE_SOURCE_DIR}/scripts/perf/adaptive_trace_memory_matrix.py"
    --mode fresh-process-timing
    --executable
      "$<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_convergence>")
set_tests_properties(
  dg_igr_hdg_runtime_cuda_adaptive_trace_fresh_process_timing_igr_p3_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;active-igr;corrected-alpha;same-mesh-reference;rodas3;adaptive;backward-euler;partitioned-backward-euler;architecture-selection;fresh-process;startup;output;performance;manual;p3"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)

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

# One bounded winner-specific ladder reuses the shared CPU/ordinary-CUDA/
# tensor-CUDA spatial and temporal convergence oracle.  The p1-p3 backend
# surface is covered by the focused alignment matrix above, so p2 is enough to
# bind the selected time treatment without duplicating the legacy matrix.
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_convergence_partitioned_backward_euler_igr_p2_rank1
  COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
    --order 2 --regularization igr
    --time-treatment partitioned-backward-euler)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_convergence_partitioned_backward_euler_igr_p2_rank1
  PROPERTIES
    LABELS "hdg;runtime;cpu;cuda;generic-euler;convergence;spatial;temporal-order;partitioned-backward-euler;ordinary;tensor;igr;p2;alignment"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)

# The original failing entropy-wave corner is isolated as one bounded p1,
# mesh-3, one-step tensor-IGR gate for each fully coupled specialization.
# It has identically zero velocity gradient and therefore directly guards the
# coupled full-patch solve on that corner without repeating either convergence
# ladder.
foreach(_coupled_policy IN LISTS _dg_igr_coupled_trace_policies)
  string(REPLACE "native-element-schwarz-matrix-free-warp-coupled-" ""
    _coupled_policy_suffix "${_coupled_policy}")
  string(REPLACE "-" "_" _coupled_policy_suffix
    "${_coupled_policy_suffix}")
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_zero_gradient_coupled_${_coupled_policy_suffix}_p1_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
      --order 1 --regularization igr
      --igr-volume-quadrature-order 6
      --igr-face-quadrature-order 4
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12
      --tensor-igr-trace-policy ${_coupled_policy}
      --zero-gradient-only)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_zero_gradient_coupled_${_coupled_policy_suffix}_p1_rank1
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;zero-gradient;coupled-full-patch;ablation;p1"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1800)
endforeach()

# The same bounded zero-gradient corner must select the robust policy's
# cooperative coupled branch after the exact first-application retained-patch
# rejection at every supported order.  The W8A rank-one device fixture
# separately qualifies setup-time structural classification, coupled admission,
# and invalid/nonfinite fail-closed classification for p1-p3.
foreach(_order IN ITEMS 1 2 3)
  math(EXPR _igr_volume_quadrature_order "2 * ${_order} + 4")
  math(EXPR _igr_face_quadrature_order "2 * ${_order} + 2")
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_zero_gradient_robust_jacobi10_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
      --order ${_order} --regularization igr
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order}
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12
      --tensor-igr-trace-policy ${_dg_igr_robust_trace_policy}
      --zero-gradient-only)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_zero_gradient_robust_jacobi10_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;zero-gradient;robust-adaptive;coupled-branch;ablation;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1800)

  # Production routing must exercise the same mixed lower/coupled path without
  # an explicit private-policy override.
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_zero_gradient_engine_default_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
      --order ${_order} --regularization igr
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order}
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12
      --zero-gradient-only)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_zero_gradient_engine_default_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cuda;generic-euler;tensor;igr;zero-gradient;coupled-branch;robust-adaptive;production-trace-policy;p${_order}"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1800)
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

  # Bind the complete explicit candidate policy—not just its quadrature—to
  # the shared spatial and Backward-Euler temporal convergence fixture.
  add_test(
    NAME dg_igr_hdg_runtime_cuda_generic_euler_convergence_igr_candidate_p${_order}_rank1
    COMMAND dg_igr_hdg_runtime_cuda_generic_euler_convergence
      --order ${_order} --regularization igr
      --igr-volume-quadrature-order ${_igr_volume_quadrature_order}
      --igr-face-quadrature-order ${_igr_face_quadrature_order}
      --newton-linear-rtol 1e-10
      --final-linear-rtol 3e-11
      --true-residual-tightening-linear-rtol 1e-12)
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_generic_euler_convergence_igr_candidate_p${_order}_rank1
    PROPERTIES
      LABELS "hdg;runtime;cpu;cuda;generic-euler;convergence;spatial;backward-euler;ordinary;tensor;igr;quadrature-candidate;role-tolerance-candidate;production-trace-policy;p${_order};alignment"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 7200)
endforeach()

# Standalone test-only performance observability for ordinary and tensor FP64
# CUDA accepted-step lifecycles at p1-p3.  This is intentionally a build
# target, not a CTest: its default no-export 10,816-element workload performs
# one warmup and five measured samples.
add_executable(dg_igr_igr_screened_patch_cuda_test
  "${PROJECT_SOURCE_DIR}/tests/test_igr_screened_patch_cuda.cu")
target_include_directories(dg_igr_igr_screened_patch_cuda_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_igr_screened_patch_cuda_test PRIVATE
  igr_mfem mfem umpire MPI::MPI_CXX)
target_compile_options(dg_igr_igr_screened_patch_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_igr_screened_patch_cuda_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME dg_igr_igr_screened_patch_cuda
  COMMAND dg_igr_igr_screened_patch_cuda_test)
set_tests_properties(dg_igr_igr_screened_patch_cuda PROPERTIES
  LABELS "cuda;igr;screening" TIMEOUT 90 RUN_SERIAL TRUE)

if(IGRLAB_BUILD_RESEARCH_TESTS)
  add_executable(dg_igr_condensation_panel_cuda
    "${PROJECT_SOURCE_DIR}/tests/research/igr_condensation_panel_cuda.cu")
  target_include_directories(dg_igr_condensation_panel_cuda PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(dg_igr_condensation_panel_cuda PRIVATE igr_mfem mfem umpire MPI::MPI_CXX CUDA::cublas)
  target_compile_options(dg_igr_condensation_panel_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_condensation_panel_cuda PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME dg_igr_condensation_panel_cuda
    COMMAND dg_igr_condensation_panel_cuda --order 3 --nx 2 --repeats 2)
  set_tests_properties(dg_igr_condensation_panel_cuda PROPERTIES
    LABELS "cuda;igr;research" TIMEOUT 90 RUN_SERIAL TRUE)

  # Manual saved-state mechanism study; input fields come from the shared DG
  # research solver. No end-to-end application claim follows from this target.
  add_executable(dg_igr_dg_source_tensor_cuda
    "${PROJECT_SOURCE_DIR}/tests/research/igr_dg_source_tensor_cuda.cu")
  target_link_libraries(dg_igr_dg_source_tensor_cuda PRIVATE CUDA::cudart)
  set_target_properties(dg_igr_dg_source_tensor_cuda PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

endif()

add_executable(dg_igr_igr_stationary_correction_cuda_test
  "${PROJECT_SOURCE_DIR}/tests/test_igr_stationary_correction_cuda.cu")
target_include_directories(dg_igr_igr_stationary_correction_cuda_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_igr_stationary_correction_cuda_test PRIVATE
  igr_mfem mfem umpire MPI::MPI_CXX CUDA::cublas)
target_compile_options(dg_igr_igr_stationary_correction_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_igr_stationary_correction_cuda_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME dg_igr_igr_stationary_correction_cuda
  COMMAND dg_igr_igr_stationary_correction_cuda_test)
set_tests_properties(dg_igr_igr_stationary_correction_cuda PROPERTIES
  LABELS "cuda;igr;two-level" TIMEOUT 90 RUN_SERIAL TRUE)

add_executable(dg_igr_igr_two_level_inverse_cuda_test
  "${PROJECT_SOURCE_DIR}/tests/test_igr_two_level_inverse_cuda.cu")
target_include_directories(dg_igr_igr_two_level_inverse_cuda_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_igr_two_level_inverse_cuda_test PRIVATE
  igr_mfem mfem umpire MPI::MPI_CXX CUDA::cublas)
target_compile_options(dg_igr_igr_two_level_inverse_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_igr_two_level_inverse_cuda_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME dg_igr_igr_two_level_inverse_cuda
  COMMAND dg_igr_igr_two_level_inverse_cuda_test)
set_tests_properties(dg_igr_igr_two_level_inverse_cuda PROPERTIES
  LABELS "cuda;igr;two-level" TIMEOUT 90 RUN_SERIAL TRUE)

add_executable(dg_igr_igr_dense_inverse_cuda_test
  "${PROJECT_SOURCE_DIR}/tests/test_igr_dense_inverse_cuda.cu")
target_include_directories(dg_igr_igr_dense_inverse_cuda_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_igr_dense_inverse_cuda_test PRIVATE
  igr_mfem mfem umpire MPI::MPI_CXX CUDA::cublas)
target_compile_options(dg_igr_igr_dense_inverse_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_igr_dense_inverse_cuda_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME dg_igr_igr_dense_inverse_cuda
  COMMAND dg_igr_igr_dense_inverse_cuda_test)
set_tests_properties(dg_igr_igr_dense_inverse_cuda PROPERTIES
  LABELS "cuda;igr;mixed-precision" TIMEOUT 60 RUN_SERIAL TRUE)

add_executable(dg_igr_dgsem_tensor_volume_benchmark
  benchmark_dgsem_tensor_volume.cu)
target_include_directories(dg_igr_dgsem_tensor_volume_benchmark PRIVATE
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(dg_igr_dgsem_tensor_volume_benchmark PRIVATE
  igr_mfem mfem umpire MPI::MPI_CXX CUDA::cublas)
target_compile_options(dg_igr_dgsem_tensor_volume_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(dg_igr_dgsem_tensor_volume_benchmark PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
# This disassembly qualifier records CUDA 13 SM120 instruction spellings.
# SM80 numerical tests are enabled below; SM80 SASS needs its own qualification.
if(DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE AND
   DG_IGR_HDG_NATIVE_CUDA_ARCH STREQUAL "120")
  add_custom_target(dg_igr_hdg_runtime_cuda_tensor_sass_evidence
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime_cuda_accepted_step_benchmark>"
      "-DCUOBJDUMP_EXECUTABLE=${DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE}"
      # Select UseTensor=true beside the separately retained SIMT specialization.
      "-DEXPECTED_KERNEL_SUBSTRING=TensorBorrowedCondenseKernelILb1E"
      "-DEXPECTED_VOLUME_KERNEL_SUBSTRING=BorrowedEulerTensorVolumeContractionKernel"
      "-DEVIDENCE_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/tensor-sass-evidence"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_generic_tensor_sass.cmake"
    DEPENDS dg_igr_hdg_runtime_cuda_accepted_step_benchmark
    COMMENT "Checking production tensor condensation and volume FP64 SASS"
    VERBATIM)
  add_test(
    NAME dg_igr_hdg_runtime_cuda_tensor_sass_evidence
    COMMAND
      "${CMAKE_COMMAND}"
      "-DTEST_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_runtime_cuda_accepted_step_benchmark>"
      "-DCUOBJDUMP_EXECUTABLE=${DG_IGR_GENERIC_TENSOR_CUOBJDUMP_EXECUTABLE}"
      # Select UseTensor=true beside the separately retained SIMT specialization.
      "-DEXPECTED_KERNEL_SUBSTRING=TensorBorrowedCondenseKernelILb1E"
      "-DEXPECTED_VOLUME_KERNEL_SUBSTRING=BorrowedEulerTensorVolumeContractionKernel"
      "-DEVIDENCE_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/tensor-sass-evidence"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_generic_tensor_sass.cmake")
  set_tests_properties(
    dg_igr_hdg_runtime_cuda_tensor_sass_evidence
    PROPERTIES
      LABELS "hdg;runtime;cuda;tensor;sass"
      TIMEOUT 300)
else()
  message(STATUS
    "Generic tensor SASS evidence target requires cuobjdump and the qualified SM120 profile.")
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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_ordinary_rank1
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 1
    --time-treatment partitioned-backward-euler
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_ordinary_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 3
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 2
    --time-treatment partitioned-backward-euler
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_tensor_rank1
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 1
    --engine tensor
    --time-treatment partitioned-backward-euler
    ${MPIEXEC_POSTFLAGS})
add_test(
  NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_tensor_rank2
  COMMAND
    "${MPIEXEC_EXECUTABLE}"
    "${MPIEXEC_NUMPROC_FLAG}" 3
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
    --active-ranks 2
    --engine tensor
    --time-treatment partitioned-backward-euler
    ${MPIEXEC_POSTFLAGS})

# RODAS3 communicator qualification keeps one MPI_COMM_WORLD rank passive and
# runs the complete active-subcommunicator transaction. Rank two additionally
# proves the same-GPU host-staged HYPRE route for each CUDA engine.
foreach(_engine IN ITEMS ordinary tensor)
  foreach(_active_ranks IN ITEMS 1 2)
    math(EXPR _world_ranks "${_active_ranks} + 1")
    set(_rank_labels "rank${_active_ranks}")
    if(_active_ranks EQUAL 2)
      string(APPEND _rank_labels ";host-staged")
    endif()
    add_test(
      NAME dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_rodas3_${_engine}_rank${_active_ranks}
      COMMAND
        "${MPIEXEC_EXECUTABLE}"
        "${MPIEXEC_NUMPROC_FLAG}" ${_world_ranks}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator>
        --active-ranks ${_active_ranks}
        --engine ${_engine}
        --time-treatment rodas3
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
      dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_rodas3_${_engine}_rank${_active_ranks}
      PROPERTIES
        LABELS "hdg;runtime;cuda;generic-euler;rodas3;${_engine};igr;subcommunicator;mpi;assessment;no-fallback;${_rank_labels}"
        PROCESSORS ${_world_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1800)
  endforeach()
endforeach()
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
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_ordinary_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;partitioned-backward-euler;ordinary;igr;subcommunicator;rank1;mpi"
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_ordinary_rank2
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;partitioned-backward-euler;ordinary;igr;subcommunicator;rank2;mpi;negative;host-staged"
    PROCESSORS 3
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_tensor_rank1
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;partitioned-backward-euler;tensor;igr;subcommunicator;rank1;mpi;no-fallback"
    PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 3600)
set_tests_properties(
  dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator_partitioned_backward_euler_tensor_rank2
  PROPERTIES
    LABELS "hdg;runtime;cuda;generic-euler;partitioned-backward-euler;tensor;igr;subcommunicator;rank2;mpi;negative;host-staged;no-fallback"
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
      CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
      CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
      CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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

  # Frozen object checks distinguish native SM120 from MFEM's SM80 PTX.
  # That distinction does not apply to the fully native SM80 profile.
  if(DG_IGR_HDG_NATIVE_CUDA_ARCH STREQUAL "120")
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
  endif()

  set_tests_properties(
    hdg_sc25_scalar_igr_cuda_rank1_c1
    hdg_sc25_scalar_igr_cuda_rank1_c2
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
      CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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

  if(DG_IGR_HDG_NATIVE_CUDA_ARCH STREQUAL "120")
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
  endif()

  set_tests_properties(hdg_sc25_scalar_igr_cuda_rank2_c3
    PROPERTIES
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  if(DG_IGR_HDG_NATIVE_CUDA_ARCH STREQUAL "120")
    set_tests_properties(
      hdg_sc25_scalar_igr_cuda_rank1_device_object
      hdg_sc25_scalar_igr_cuda_rank2_device_object
      PROPERTIES
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 3600)
  endif()

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
    CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}" CUDA_SEPARABLE_COMPILATION ON
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
