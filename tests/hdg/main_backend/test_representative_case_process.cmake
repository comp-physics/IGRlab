if(NOT DEFINED HDG_PERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_NONPERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_REPRESENTATIVE_WORK)
  message(FATAL_ERROR
    "Periodic/nonperiodic generic HDG executables and a work directory are required.")
endif()

if(NOT DEFINED HDG_REPRESENTATIVE_DEVICE)
  set(HDG_REPRESENTATIVE_DEVICE cpu)
endif()
if(NOT HDG_REPRESENTATIVE_DEVICE STREQUAL "cpu" AND
   NOT HDG_REPRESENTATIVE_DEVICE STREQUAL "cuda")
  message(FATAL_ERROR
    "Representative HDG device must be cpu or cuda, got '${HDG_REPRESENTATIVE_DEVICE}'.")
endif()
if(NOT DEFINED HDG_MAIN_CUDA_ENGINE)
  set(HDG_MAIN_CUDA_ENGINE ordinary)
endif()
if(NOT HDG_MAIN_CUDA_ENGINE STREQUAL "ordinary" AND
   NOT HDG_MAIN_CUDA_ENGINE STREQUAL "tensor")
  message(FATAL_ERROR
    "Representative CUDA engine must be ordinary or tensor, got '${HDG_MAIN_CUDA_ENGINE}'.")
endif()
if(HDG_REPRESENTATIVE_DEVICE STREQUAL "cpu" AND
   NOT HDG_MAIN_CUDA_ENGINE STREQUAL "ordinary")
  message(FATAL_ERROR
    "CPU representative coverage requires the ordinary CUDA selector.")
endif()
if(NOT DEFINED HDG_REPRESENTATIVE_RANKS)
  set(HDG_REPRESENTATIVE_RANKS 1)
endif()
if(HDG_REPRESENTATIVE_RANKS LESS 1)
  message(FATAL_ERROR "Representative HDG rank count must be positive.")
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
if(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor" AND
   HDG_REPRESENTATIVE_RANKS EQUAL 1)
  set(_expected_trace_preconditioner
    native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64)
  set(_expected_resolved_trace_policy
    native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi10-fp64)
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
if(HDG_REPRESENTATIVE_RANKS EQUAL 2)
  set(_expected_trace_solver hypre-host-staged-rank2)
endif()

set(_periodic_command "${HDG_PERIODIC_EXECUTABLE}")
set(_nonperiodic_command "${HDG_NONPERIODIC_EXECUTABLE}")
set(_command_postflags)
if(HDG_REPRESENTATIVE_RANKS GREATER 1)
  if(NOT DEFINED HDG_REPRESENTATIVE_MPIEXEC OR
     NOT DEFINED HDG_REPRESENTATIVE_NUMPROC_FLAG)
    message(FATAL_ERROR
      "Representative rank-two HDG execution requires an MPI launcher.")
  endif()
  set(_periodic_command
    "${HDG_REPRESENTATIVE_MPIEXEC}"
    "${HDG_REPRESENTATIVE_NUMPROC_FLAG}"
    "${HDG_REPRESENTATIVE_RANKS}"
    ${HDG_REPRESENTATIVE_MPIEXEC_PREFLAGS}
    "${HDG_PERIODIC_EXECUTABLE}")
  set(_nonperiodic_command
    "${HDG_REPRESENTATIVE_MPIEXEC}"
    "${HDG_REPRESENTATIVE_NUMPROC_FLAG}"
    "${HDG_REPRESENTATIVE_RANKS}"
    ${HDG_REPRESENTATIVE_MPIEXEC_PREFLAGS}
    "${HDG_NONPERIODIC_EXECUTABLE}")
  set(_command_postflags ${HDG_REPRESENTATIVE_MPIEXEC_POSTFLAGS})
endif()

set(_periodic_mesh_arguments)
if(DEFINED HDG_REPRESENTATIVE_PERIODIC_MESH)
  if(NOT EXISTS "${HDG_REPRESENTATIVE_PERIODIC_MESH}")
    message(FATAL_ERROR
      "Representative periodic mesh does not exist: ${HDG_REPRESENTATIVE_PERIODIC_MESH}")
  endif()
  list(APPEND _periodic_mesh_arguments
    --mesh "${HDG_REPRESENTATIVE_PERIODIC_MESH}")
endif()
set(_nonperiodic_mesh_arguments)
if(DEFINED HDG_REPRESENTATIVE_NONPERIODIC_MESH)
  if(NOT EXISTS "${HDG_REPRESENTATIVE_NONPERIODIC_MESH}")
    message(FATAL_ERROR
      "Representative physical mesh does not exist: ${HDG_REPRESENTATIVE_NONPERIODIC_MESH}")
  endif()
  list(APPEND _nonperiodic_mesh_arguments
    --mesh "${HDG_REPRESENTATIVE_NONPERIODIC_MESH}")
endif()
if(HDG_REPRESENTATIVE_DEVICE STREQUAL "cuda" AND
   (NOT DEFINED HDG_REPRESENTATIVE_PERIODIC_MESH OR
    NOT DEFINED HDG_REPRESENTATIVE_NONPERIODIC_MESH))
  message(FATAL_ERROR
    "CUDA representative coverage requires explicitly bounded periodic and physical meshes.")
endif()

file(REMOVE_RECURSE "${HDG_REPRESENTATIVE_WORK}")
file(MAKE_DIRECTORY "${HDG_REPRESENTATIVE_WORK}")

set(_common_arguments
  --discretization hdg
  --device "${HDG_REPRESENTATIVE_DEVICE}"
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00001
  --t-final 0.00001
  --no-glvis)
if(HDG_REPRESENTATIVE_DEVICE STREQUAL "cuda")
  list(APPEND _common_arguments
    --hdg-cuda-engine "${HDG_MAIN_CUDA_ENGINE}")
endif()

# The CUDA gate deliberately uses bounded meshes and no VTK.  It reaches both
# real application entry points through Solver::solve, exercises the periodic
# source path with IGR, and covers every physical-boundary mode at p1.  The
# separate CUDA qualification matrix carries the full p1-p3 order surface.
if(HDG_REPRESENTATIVE_DEVICE STREQUAL "cuda")
  execute_process(
    COMMAND ${_periodic_command} ${_common_arguments}
      ${_periodic_mesh_arguments}
      --order 1
      --regularization igr
      --hdg-static-source 0.01
      --no-vtk_save
      ${_command_postflags}
    RESULT_VARIABLE _periodic_status
    OUTPUT_VARIABLE _periodic_stdout
    ERROR_VARIABLE _periodic_stderr)
  if(NOT _periodic_status EQUAL 0 OR
     NOT _periodic_stdout MATCHES
       "\\[hdg-main\\] status=Success.*cuda_engine=${HDG_MAIN_CUDA_ENGINE}.*requested_trace_policy=${_expected_requested_trace_policy}.*resolved_trace_policy=${_expected_resolved_trace_policy}.*trace_solver=${_expected_trace_solver}.*fallback_count=0.*trace_preconditioner=${_expected_trace_preconditioner}.*trace_ilu_type=${_expected_ilu_type}.*trace_ilu_level_of_fill=${_expected_ilu_fill}.*trace_ilu_local_reordering=${_expected_ilu_reordering}.*trace_ilu_application_iterations=${_expected_ilu_iterations}.*trace_ilu_accounting_kind=${_expected_ilu_accounting_kind}.*trace_ilu_factor_reuse_policy=${_expected_ilu_factor_reuse_policy}.*trace_ilu_triangular_solve_kind=${_expected_ilu_triangular_solve_kind}.*trace_ilu_lower_jacobi_iterations=${_expected_ilu_lower_jacobi_iterations}.*trace_ilu_upper_jacobi_iterations=${_expected_ilu_upper_jacobi_iterations}.*regularization=igr" OR
     NOT _periodic_stderr STREQUAL "")
    message(FATAL_ERROR
      "CUDA periodic p1 HDG IGR execution failed: ${_periodic_status}\\n${_periodic_stdout}\\n${_periodic_stderr}")
  endif()

  foreach(_boundary_mode IN ITEMS inlet-outlet all-outlet all-slip)
    execute_process(
      COMMAND ${_nonperiodic_command} ${_common_arguments}
        ${_nonperiodic_mesh_arguments}
        --order 1
        --bc-mode ${_boundary_mode}
        --regularization igr
        --no-vtk_save
        ${_command_postflags}
      RESULT_VARIABLE _physical_status
      OUTPUT_VARIABLE _physical_stdout
      ERROR_VARIABLE _physical_stderr)
    if(NOT _physical_status EQUAL 0 OR
       NOT _physical_stdout MATCHES
         "\\[hdg-main\\] status=Success.*cuda_engine=${HDG_MAIN_CUDA_ENGINE}.*requested_trace_policy=${_expected_requested_trace_policy}.*resolved_trace_policy=${_expected_resolved_trace_policy}.*trace_solver=${_expected_trace_solver}.*fallback_count=0.*trace_preconditioner=${_expected_trace_preconditioner}.*trace_ilu_type=${_expected_ilu_type}.*trace_ilu_level_of_fill=${_expected_ilu_fill}.*trace_ilu_local_reordering=${_expected_ilu_reordering}.*trace_ilu_application_iterations=${_expected_ilu_iterations}.*trace_ilu_accounting_kind=${_expected_ilu_accounting_kind}.*trace_ilu_factor_reuse_policy=${_expected_ilu_factor_reuse_policy}.*trace_ilu_triangular_solve_kind=${_expected_ilu_triangular_solve_kind}.*trace_ilu_lower_jacobi_iterations=${_expected_ilu_lower_jacobi_iterations}.*trace_ilu_upper_jacobi_iterations=${_expected_ilu_upper_jacobi_iterations}.*regularization=igr" OR
       NOT _physical_stderr STREQUAL "")
      message(FATAL_ERROR
        "CUDA physical-boundary ${_boundary_mode} p1 HDG IGR execution failed: ${_physical_status}\\n${_physical_stdout}\\n${_physical_stderr}")
    endif()
  endforeach()
  return()
endif()

# Exercise the conventional source path while IGR augments the same solver.
# Distinct final VTK payloads prove that the projected source reaches the
# numerical request; a success-only smoke would not detect an accidentally
# dropped source view.
set(_zero_output "${HDG_REPRESENTATIVE_WORK}/igr-zero-source")
set(_source_output "${HDG_REPRESENTATIVE_WORK}/igr-nonzero-source")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DG_IGR_VTK_PREFIX_PATH=${_zero_output}"
    ${_periodic_command} ${_common_arguments} ${_periodic_mesh_arguments}
    --order 1
    --regularization igr
    --time-step-save 0.00001
    --vtk_save
    ${_command_postflags}
  RESULT_VARIABLE _zero_status
  OUTPUT_VARIABLE _zero_stdout
  ERROR_VARIABLE _zero_stderr)
if(NOT _zero_status EQUAL 0 OR
   NOT _zero_stdout MATCHES
     "\\[hdg-main\\] status=Success.*regularization=igr" OR
   NOT _zero_stderr STREQUAL "")
  message(FATAL_ERROR
    "periodic zero-source HDG execution failed: ${_zero_status}\\n${_zero_stdout}\\n${_zero_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DG_IGR_VTK_PREFIX_PATH=${_source_output}"
    ${_periodic_command} ${_common_arguments} ${_periodic_mesh_arguments}
    --order 1
    --regularization igr
    --time-step-save 0.00001
    --vtk_save
    --hdg-static-source 0.01
    ${_command_postflags}
  RESULT_VARIABLE _source_status
  OUTPUT_VARIABLE _source_stdout
  ERROR_VARIABLE _source_stderr)
if(NOT _source_status EQUAL 0 OR
   NOT _source_stdout MATCHES
     "\\[hdg-main\\] status=Success.*regularization=igr" OR
   NOT _source_stderr STREQUAL "")
  message(FATAL_ERROR
    "periodic HDG static-source execution failed: ${_source_status}\\n${_source_stdout}\\n${_source_stderr}")
endif()

set(_collection "dgsem_smooth_advection_periodic_hdg_p_1_ref_0")
set(_zero_final
  "${_zero_output}/${_collection}/Cycle000001/proc000000.vtu")
set(_source_final
  "${_source_output}/${_collection}/Cycle000001/proc000000.vtu")
if(NOT EXISTS "${_zero_final}" OR NOT EXISTS "${_source_final}")
  message(FATAL_ERROR "source comparison is missing a final VTU payload")
endif()
file(READ "${_zero_final}" _zero_payload)
file(READ "${_source_final}" _source_payload)
if(_zero_payload STREQUAL _source_payload)
  message(FATAL_ERROR
    "nonzero static source did not change the committed final IGR-augmented HDG field")
endif()

execute_process(
  COMMAND ${_nonperiodic_command} ${_common_arguments}
    ${_nonperiodic_mesh_arguments}
    --order 1
    --bc-mode inlet-outlet
    --no-vtk_save
    ${_command_postflags}
  RESULT_VARIABLE _boundary_status
  OUTPUT_VARIABLE _boundary_stdout
  ERROR_VARIABLE _boundary_stderr)
if(NOT _boundary_status EQUAL 0 OR
   NOT _boundary_stdout MATCHES "\\[hdg-main\\] status=Success" OR
   NOT _boundary_stderr STREQUAL "")
  message(FATAL_ERROR
    "nonperiodic conventional HDG execution failed: ${_boundary_status}\\n${_boundary_stdout}\\n${_boundary_stderr}")
endif()

# IGR is an augmentation of the same physical-boundary Euler lifecycle at
# every supported order.  The selectable plans cover a mixed supersonic
# inlet/outlet/slip boundary, every-attribute supersonic outlet, and
# every-attribute slip wall.  One short step keeps the full p1-p3 matrix fast.
foreach(_boundary_mode IN ITEMS inlet-outlet all-outlet all-slip)
  foreach(_order IN ITEMS 1 2 3)
    execute_process(
      COMMAND ${_nonperiodic_command} ${_common_arguments}
        ${_nonperiodic_mesh_arguments}
        --order ${_order}
        --bc-mode ${_boundary_mode}
        --regularization igr
        --no-vtk_save
        ${_command_postflags}
      RESULT_VARIABLE _igr_status
      OUTPUT_VARIABLE _igr_stdout
      ERROR_VARIABLE _igr_stderr)
    if(NOT _igr_status EQUAL 0 OR
       NOT _igr_stdout MATCHES
         "\\[hdg-main\\] status=Success.*regularization=igr" OR
       NOT _igr_stderr STREQUAL "")
      message(FATAL_ERROR
        "physical-boundary ${_boundary_mode} p${_order} HDG IGR execution failed: ${_igr_status}\\n${_igr_stdout}\\n${_igr_stderr}")
    endif()
  endforeach()
endforeach()
