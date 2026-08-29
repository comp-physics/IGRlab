if(NOT DEFINED HDG_MAIN_EXECUTABLE OR
   NOT DEFINED HDG_MAIN_MESH OR
   NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR
   NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR
    "CUDA HDG executable, mesh, ranks, and MPI launcher are required.")
endif()

if(NOT DEFINED HDG_MAIN_CUDA_ENGINE)
  set(HDG_MAIN_CUDA_ENGINE ordinary)
endif()
if(NOT HDG_MAIN_CUDA_ENGINE STREQUAL "ordinary" AND
   NOT HDG_MAIN_CUDA_ENGINE STREQUAL "tensor")
  message(FATAL_ERROR
    "CUDA HDG engine must be ordinary or tensor, got '${HDG_MAIN_CUDA_ENGINE}'.")
endif()
set(_expected_trace_preconditioner hypre_ilu0)
set(_expected_requested_trace_policy engine-default)
set(_expected_resolved_trace_policy engine-default)
set(_expected_trace_solver hypre-parcsr-gmres)
set(_expected_ilu_type 0)
set(_expected_ilu_fill 0)
set(_expected_ilu_reordering 1)
set(_expected_ilu_iterations 1)
set(_expected_ilu_accounting_kind derived_bounds_v1)
if(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor" AND HDG_MAIN_RANKS EQUAL 1)
  set(_expected_trace_preconditioner
    native_element_schwarz_warp_equilibrated_field_split_fp64)
  set(_expected_resolved_trace_policy
    native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64)
  set(_expected_trace_solver native-rank1-fp64)
  set(_expected_ilu_type -1)
  set(_expected_ilu_fill -1)
  set(_expected_ilu_reordering -1)
  set(_expected_ilu_iterations 0)
  set(_expected_ilu_accounting_kind unavailable)
  set(_expected_ilu_factor_reuse_policy not_applicable)
  set(_expected_ilu_triangular_solve_kind not_applicable)
  set(_expected_ilu_lower_jacobi_iterations 0)
  set(_expected_ilu_upper_jacobi_iterations 0)
elseif(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor")
  set(_expected_trace_solver hypre-host-staged-rank2)
  set(_expected_ilu_factor_reuse_policy first_trace_solve_per_attempt)
  set(_expected_ilu_triangular_solve_kind iterative_jacobi)
  set(_expected_ilu_lower_jacobi_iterations 5)
  set(_expected_ilu_upper_jacobi_iterations 5)
else()
  set(_expected_ilu_factor_reuse_policy refresh_each_trace_solve)
  set(_expected_ilu_triangular_solve_kind direct)
  set(_expected_ilu_lower_jacobi_iterations 0)
  set(_expected_ilu_upper_jacobi_iterations 0)
endif()
if(HDG_MAIN_RANKS EQUAL 2)
  set(_expected_trace_solver hypre-host-staged-rank2)
endif()

set(_base_arguments
  --discretization hdg
  --device cuda
  --hdg-cuda-engine "${HDG_MAIN_CUDA_ENGINE}"
  --regularization igr
  --serial-refine 0
  --parallel-refine 0
  --mesh "${HDG_MAIN_MESH}"
  --time-step 0.00001
  --t-final 0.00001
  --no-vtk_save
  --no-glvis)

function(run_cuda_igr_success order expected_alpha expected_beta)
  set(_arguments ${_base_arguments} --order ${order} ${ARGN})
  if(HDG_MAIN_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}"
        "${HDG_MAIN_RANKS}"
        ${HDG_MAIN_MPIEXEC_PREFLAGS}
        "${HDG_MAIN_EXECUTABLE}" ${_arguments}
        ${HDG_MAIN_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  if(NOT _status EQUAL 0 OR
     NOT _stdout MATCHES
       "\\[hdg-main\\] status=Success.*cuda_engine=${HDG_MAIN_CUDA_ENGINE}.*requested_trace_policy=${_expected_requested_trace_policy}.*resolved_trace_policy=${_expected_resolved_trace_policy}.*trace_solver=${_expected_trace_solver}.*fallback_count=0.*trace_preconditioner=${_expected_trace_preconditioner}.*trace_ilu_type=${_expected_ilu_type}.*trace_ilu_level_of_fill=${_expected_ilu_fill}.*trace_ilu_local_reordering=${_expected_ilu_reordering}.*trace_ilu_application_iterations=${_expected_ilu_iterations}.*trace_ilu_accounting_kind=${_expected_ilu_accounting_kind}.*trace_ilu_factor_reuse_policy=${_expected_ilu_factor_reuse_policy}.*trace_ilu_triangular_solve_kind=${_expected_ilu_triangular_solve_kind}.*trace_ilu_lower_jacobi_iterations=${_expected_ilu_lower_jacobi_iterations}.*trace_ilu_upper_jacobi_iterations=${_expected_ilu_upper_jacobi_iterations}.*regularization=igr.*alpha=${expected_alpha}.*beta=${expected_beta}" OR
     NOT _stderr STREQUAL "")
    message(FATAL_ERROR
      "CUDA HDG p${order} IGR execution failed: ${_status}\n${_stdout}\n${_stderr}")
  endif()
endfunction()

function(expect_cuda_igr_rejection expected_diagnostic)
  set(_arguments ${_base_arguments} ${ARGN})
  if(HDG_MAIN_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}"
        "${HDG_MAIN_RANKS}"
        ${HDG_MAIN_MPIEXEC_PREFLAGS}
        "${HDG_MAIN_EXECUTABLE}" ${_arguments}
        ${HDG_MAIN_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  set(_output "${_stdout}\n${_stderr}")
  if("${_status}" STREQUAL "0" OR NOT _output MATCHES "${expected_diagnostic}")
    message(FATAL_ERROR
      "CUDA HDG IGR rejection failed: ${_status}\n${_output}")
  endif()
endfunction()

# This is the production main-backend route, not the frozen SC25/P1 oracle.
# All supported orders share the same generic IC/trace projection, source,
# time adapter, nonlinear lifecycle, condensation, and optional IGR fields.
run_cuda_igr_success(1 1 1)
run_cuda_igr_success(1 0.2 0.8
  --hdg-igr-alpha 0.2 --hdg-igr-beta 0.8)
run_cuda_igr_success(1 1 1 --hdg-static-source 0.001)
run_cuda_igr_success(2 1 1)
run_cuda_igr_success(3 1 1)
expect_cuda_igr_rejection(
  "HDG IGR alpha and beta must be finite and positive"
  --order 1 --hdg-igr-alpha 0)
expect_cuda_igr_rejection(
  "HDG IGR alpha and beta must be finite and positive"
  --order 1 --hdg-igr-beta 0)
