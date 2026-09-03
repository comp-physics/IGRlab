if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_OUTPUT OR
   NOT DEFINED HDG_MAIN_MESH OR NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR "Main executable, output, mesh, ranks, and MPI launcher are required.")
endif()

if(NOT DEFINED HDG_MAIN_DEVICE)
  set(HDG_MAIN_DEVICE cuda)
endif()
if(NOT HDG_MAIN_DEVICE STREQUAL "cpu" AND
   NOT HDG_MAIN_DEVICE STREQUAL "cuda")
  message(FATAL_ERROR
    "HDG main device must be cpu or cuda, got '${HDG_MAIN_DEVICE}'.")
endif()
if(NOT DEFINED HDG_MAIN_CUDA_ENGINE)
  set(HDG_MAIN_CUDA_ENGINE ordinary)
endif()
if(NOT DEFINED HDG_MAIN_OMIT_CUDA_ENGINE)
  set(HDG_MAIN_OMIT_CUDA_ENGINE OFF)
endif()
if(NOT HDG_MAIN_CUDA_ENGINE STREQUAL "ordinary" AND
   NOT HDG_MAIN_CUDA_ENGINE STREQUAL "tensor")
  message(FATAL_ERROR
    "HDG CUDA engine must be ordinary or tensor, got '${HDG_MAIN_CUDA_ENGINE}'.")
endif()
if(HDG_MAIN_DEVICE STREQUAL "cpu" AND
   NOT HDG_MAIN_CUDA_ENGINE STREQUAL "ordinary")
  message(FATAL_ERROR "CPU HDG process tests require the ordinary CUDA selector.")
endif()
if(HDG_MAIN_OMIT_CUDA_ENGINE AND
   NOT HDG_MAIN_CUDA_ENGINE STREQUAL "ordinary")
  message(FATAL_ERROR
    "Only the ordinary-default process gate may omit --hdg-cuda-engine.")
endif()
if(NOT DEFINED HDG_MAIN_REGULARIZATION)
  set(HDG_MAIN_REGULARIZATION none)
endif()
if(NOT HDG_MAIN_REGULARIZATION STREQUAL "none" AND
   NOT HDG_MAIN_REGULARIZATION STREQUAL "igr")
  message(FATAL_ERROR
    "HDG regularization must be none or igr, got '${HDG_MAIN_REGULARIZATION}'.")
endif()
if(NOT DEFINED HDG_MAIN_TIME_INTEGRATOR)
  set(HDG_MAIN_TIME_INTEGRATOR backward-euler)
endif()
if(NOT DEFINED HDG_MAIN_OMIT_TIME_INTEGRATOR)
  set(HDG_MAIN_OMIT_TIME_INTEGRATOR OFF)
endif()
if(NOT HDG_MAIN_TIME_INTEGRATOR STREQUAL "backward-euler" AND
   NOT HDG_MAIN_TIME_INTEGRATOR STREQUAL "partitioned-backward-euler" AND
   NOT HDG_MAIN_TIME_INTEGRATOR STREQUAL "rodas3")
  message(FATAL_ERROR
    "HDG time integrator must be backward-euler, partitioned-backward-euler, or rodas3, got '${HDG_MAIN_TIME_INTEGRATOR}'.")
endif()
if(HDG_MAIN_OMIT_TIME_INTEGRATOR AND
   NOT HDG_MAIN_TIME_INTEGRATOR STREQUAL "backward-euler")
  message(FATAL_ERROR
    "Only the backward-Euler-default process gate may omit --time-integrator.")
endif()
if(DEFINED HDG_MAIN_POSITIVITY_LIMITER AND
   NOT HDG_MAIN_POSITIVITY_LIMITER STREQUAL "case" AND
   NOT HDG_MAIN_POSITIVITY_LIMITER STREQUAL "none" AND
   NOT HDG_MAIN_POSITIVITY_LIMITER STREQUAL "zhang-shu")
  message(FATAL_ERROR
    "HDG positivity limiter must be case, none, or zhang-shu, got '${HDG_MAIN_POSITIVITY_LIMITER}'.")
endif()
if(DEFINED HDG_MAIN_ZHANG_SHU_POINTS AND
   NOT HDG_MAIN_ZHANG_SHU_POINTS STREQUAL "consumer" AND
   NOT HDG_MAIN_ZHANG_SHU_POINTS STREQUAL "quadrature")
  message(FATAL_ERROR
    "HDG Zhang-Shu point set must be consumer or quadrature, got '${HDG_MAIN_ZHANG_SHU_POINTS}'.")
endif()
if(NOT DEFINED HDG_MAIN_EXPECT_ZHANG_SHU_FIELDS)
  set(HDG_MAIN_EXPECT_ZHANG_SHU_FIELDS OFF)
endif()
if(NOT HDG_MAIN_EXPECT_ZHANG_SHU_FIELDS STREQUAL "ON" AND
   NOT HDG_MAIN_EXPECT_ZHANG_SHU_FIELDS STREQUAL "OFF")
  message(FATAL_ERROR
    "HDG Zhang-Shu field expectation must be ON or OFF.")
endif()
if(DEFINED HDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS AND
   NOT HDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS STREQUAL "ON" AND
   NOT HDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS STREQUAL "OFF")
  message(FATAL_ERROR
    "HDG Zhang-Shu VTK diagnostics selector must be ON or OFF.")
endif()
if(DEFINED HDG_MAIN_C_ALPHA AND HDG_MAIN_C_ALPHA STREQUAL "")
  message(FATAL_ERROR "HDG --c-alpha process value must not be empty.")
endif()
if(DEFINED HDG_MAIN_HDG_IGR_ALPHA AND HDG_MAIN_HDG_IGR_ALPHA STREQUAL "")
  message(FATAL_ERROR "HDG --hdg-igr-alpha process value must not be empty.")
endif()

function(require_euler_field text_variable tag field components context)
  set(_text "${${text_variable}}")
  if(NOT _text MATCHES
      "<${tag}[^>]*(Name=\"${field}\"[^>]*NumberOfComponents=\"${components}\"|NumberOfComponents=\"${components}\"[^>]*Name=\"${field}\")[^>]*>")
    message(FATAL_ERROR
      "${context} misses ${components}-component Euler field '${field}'")
  endif()
endfunction()

if(NOT DEFINED HDG_MAIN_ORDER)
  set(HDG_MAIN_ORDER 1)
endif()
if(NOT DEFINED HDG_MAIN_DT)
  set(HDG_MAIN_DT 0.05)
endif()
if(NOT DEFINED HDG_MAIN_T_FINAL)
  set(HDG_MAIN_T_FINAL 0.10)
endif()
if(NOT DEFINED HDG_MAIN_DT_SAVE)
  set(HDG_MAIN_DT_SAVE 0.05)
endif()
if(NOT DEFINED HDG_MAIN_EXPECTED_STEPS)
  set(HDG_MAIN_EXPECTED_STEPS 2)
endif()
if(NOT DEFINED HDG_MAIN_EXPECTED_FINAL_TIME)
  set(HDG_MAIN_EXPECTED_FINAL_TIME "0\\.1")
endif()
if(NOT DEFINED HDG_MAIN_EXPECTED_CYCLES)
  set(HDG_MAIN_EXPECTED_CYCLES "000000,000001,000002")
endif()
if(NOT DEFINED HDG_MAIN_EXPECTED_TIMES)
  set(HDG_MAIN_EXPECTED_TIMES "0,0.05,0.1")
endif()
if(NOT DEFINED HDG_MAIN_N)
  set(HDG_MAIN_N 3)
endif()
if(NOT DEFINED HDG_MAIN_SERIAL_REFINE)
  set(HDG_MAIN_SERIAL_REFINE 0)
endif()
if(NOT DEFINED HDG_MAIN_PARALLEL_REFINE)
  set(HDG_MAIN_PARALLEL_REFINE 0)
endif()
if(HDG_MAIN_N LESS 1 OR HDG_MAIN_N GREATER 100000 OR
   HDG_MAIN_SERIAL_REFINE LESS 0 OR HDG_MAIN_PARALLEL_REFINE LESS 0)
  message(FATAL_ERROR "HDG cell census requires bounded nonnegative mesh/refinement inputs.")
endif()
math(EXPR _refinement_levels
  "${HDG_MAIN_SERIAL_REFINE} + ${HDG_MAIN_PARALLEL_REFINE}")
if(_refinement_levels GREATER 10)
  message(FATAL_ERROR "HDG cell census refinement depth exceeds checked process-test bounds.")
endif()
math(EXPR _expected_global_cells "${HDG_MAIN_N} * ${HDG_MAIN_N}")
if(_refinement_levels GREATER 0)
  foreach(_level RANGE 1 ${_refinement_levels})
    math(EXPR _expected_global_cells "${_expected_global_cells} * 4")
  endforeach()
endif()

file(REMOVE_RECURSE "${HDG_MAIN_OUTPUT}")
set(_arguments
  --discretization hdg
  --device "${HDG_MAIN_DEVICE}"
  --regularization "${HDG_MAIN_REGULARIZATION}"
  --mesh "${HDG_MAIN_MESH}"
  --order "${HDG_MAIN_ORDER}"
  --serial-refine "${HDG_MAIN_SERIAL_REFINE}"
  --parallel-refine "${HDG_MAIN_PARALLEL_REFINE}"
  --time-step "${HDG_MAIN_DT}"
  --t-final "${HDG_MAIN_T_FINAL}"
  --no-glvis)
if(NOT HDG_MAIN_OMIT_CUDA_ENGINE)
  list(APPEND _arguments --hdg-cuda-engine "${HDG_MAIN_CUDA_ENGINE}")
endif()
if(NOT HDG_MAIN_OMIT_TIME_INTEGRATOR)
  list(APPEND _arguments --time-integrator "${HDG_MAIN_TIME_INTEGRATOR}")
endif()
if(DEFINED HDG_MAIN_POSITIVITY_LIMITER)
  list(APPEND _arguments
    --positivity-limiter "${HDG_MAIN_POSITIVITY_LIMITER}")
endif()
if(DEFINED HDG_MAIN_ZHANG_SHU_POINTS)
  list(APPEND _arguments
    --zhang-shu-points "${HDG_MAIN_ZHANG_SHU_POINTS}")
endif()
if(DEFINED HDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS)
  if(HDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS)
    list(APPEND _arguments --vtk-zhang-shu-diagnostics)
  else()
    list(APPEND _arguments --no-vtk-zhang-shu-diagnostics)
  endif()
endif()
if(DEFINED HDG_MAIN_C_ALPHA)
  list(APPEND _arguments --c-alpha "${HDG_MAIN_C_ALPHA}")
endif()
if(DEFINED HDG_MAIN_HDG_IGR_ALPHA)
  list(APPEND _arguments --hdg-igr-alpha "${HDG_MAIN_HDG_IGR_ALPHA}")
endif()
set(_success_pattern
  "\\[hdg-main\\] status=Success.*step=${HDG_MAIN_EXPECTED_STEPS}.*time=${HDG_MAIN_EXPECTED_FINAL_TIME}")
if(HDG_MAIN_DEVICE STREQUAL "cuda")
  set(_expected_requested_trace_policy engine-default)
  set(_expected_resolved_trace_policy engine-default)
  if(HDG_MAIN_RANKS EQUAL 2)
    set(_expected_trace_solver hypre-host-staged-rank2)
  elseif(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor")
    set(_expected_trace_solver native-rank1-fp64)
  else()
    set(_expected_trace_solver hypre-parcsr-gmres)
  endif()
  if(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor" AND
     HDG_MAIN_REGULARIZATION STREQUAL "none" AND
     HDG_MAIN_TIME_INTEGRATOR STREQUAL "rodas3" AND
     HDG_MAIN_RANKS EQUAL 1)
    # The generic facade lowers this integrator-specific default to the exact
    # admitted engine policy before device initialization.  Engine telemetry
    # therefore reports the typed native request as well as its identical
    # resolved selection; Backward Euler continues to report engine-default.
    set(_expected_requested_trace_policy
      native-element-schwarz-matrix-free-fp64)
    set(_expected_trace_preconditioner native_element_schwarz_fp64)
    set(_expected_resolved_trace_policy
      native-element-schwarz-matrix-free-fp64)
    set(_expected_ilu_type -1)
    set(_expected_ilu_fill -1)
    set(_expected_ilu_iterations 0)
    set(_expected_ilu_reordering -1)
    set(_expected_ilu_accounting unavailable)
    set(_expected_ilu_factor_reuse_policy not_applicable)
    set(_expected_ilu_triangular_solve_kind not_applicable)
    set(_expected_ilu_lower_jacobi_iterations 0)
    set(_expected_ilu_upper_jacobi_iterations 0)
  elseif(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor" AND
     HDG_MAIN_REGULARIZATION STREQUAL "none")
    set(_expected_trace_preconditioner hypre_diag_scale)
    set(_expected_ilu_type -1)
    set(_expected_ilu_fill -1)
    set(_expected_ilu_iterations 0)
    set(_expected_ilu_reordering -1)
    set(_expected_ilu_accounting unavailable)
    set(_expected_ilu_factor_reuse_policy not_applicable)
    set(_expected_ilu_triangular_solve_kind not_applicable)
    set(_expected_ilu_lower_jacobi_iterations 0)
    set(_expected_ilu_upper_jacobi_iterations 0)
  elseif(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor" AND
         HDG_MAIN_REGULARIZATION STREQUAL "igr" AND
         HDG_MAIN_RANKS EQUAL 1)
    set(_expected_trace_preconditioner
      native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64)
    set(_expected_resolved_trace_policy
      native-element-schwarz-matrix-free-warp-robust-coupled-sigma-jacobi10-fp64)
    set(_expected_ilu_type -1)
    set(_expected_ilu_fill -1)
    set(_expected_ilu_iterations 0)
    set(_expected_ilu_reordering -1)
    set(_expected_ilu_accounting unavailable)
    set(_expected_ilu_factor_reuse_policy not_applicable)
    set(_expected_ilu_triangular_solve_kind not_applicable)
    set(_expected_ilu_lower_jacobi_iterations 0)
    set(_expected_ilu_upper_jacobi_iterations 0)
  elseif(HDG_MAIN_CUDA_ENGINE STREQUAL "tensor" AND
         HDG_MAIN_REGULARIZATION STREQUAL "igr")
    set(_expected_trace_preconditioner hypre_ilu0)
    set(_expected_ilu_type 0)
    set(_expected_ilu_fill 0)
    set(_expected_ilu_iterations 1)
    set(_expected_ilu_reordering 1)
    set(_expected_ilu_accounting derived_bounds_v1)
    set(_expected_ilu_factor_reuse_policy first_trace_solve_per_attempt)
    set(_expected_ilu_triangular_solve_kind iterative_jacobi)
    set(_expected_ilu_lower_jacobi_iterations 5)
    set(_expected_ilu_upper_jacobi_iterations 5)
  else()
    set(_expected_trace_preconditioner hypre_ilu0)
    set(_expected_ilu_type 0)
    set(_expected_ilu_fill 0)
    set(_expected_ilu_iterations 1)
    set(_expected_ilu_accounting derived_bounds_v1)
    set(_expected_ilu_factor_reuse_policy refresh_each_trace_solve)
    set(_expected_ilu_triangular_solve_kind direct)
    set(_expected_ilu_lower_jacobi_iterations 0)
    set(_expected_ilu_upper_jacobi_iterations 0)
    if(HDG_MAIN_REGULARIZATION STREQUAL "igr")
      set(_expected_ilu_reordering 1)
    else()
      set(_expected_ilu_reordering 0)
    endif()
  endif()
  string(APPEND _success_pattern
    ".*cuda_engine=${HDG_MAIN_CUDA_ENGINE}.*time_treatment=${HDG_MAIN_TIME_INTEGRATOR}.*requested_trace_policy=${_expected_requested_trace_policy}.*resolved_trace_policy=${_expected_resolved_trace_policy}.*trace_solver=${_expected_trace_solver}.*fallback_count=0.*trace_preconditioner=${_expected_trace_preconditioner}.*trace_ilu_type=${_expected_ilu_type}.*trace_ilu_level_of_fill=${_expected_ilu_fill}.*trace_ilu_local_reordering=${_expected_ilu_reordering}.*trace_ilu_application_iterations=${_expected_ilu_iterations}.*trace_ilu_accounting_kind=${_expected_ilu_accounting}.*trace_ilu_factor_reuse_policy=${_expected_ilu_factor_reuse_policy}.*trace_ilu_triangular_solve_kind=${_expected_ilu_triangular_solve_kind}.*trace_ilu_lower_jacobi_iterations=${_expected_ilu_lower_jacobi_iterations}.*trace_ilu_upper_jacobi_iterations=${_expected_ilu_upper_jacobi_iterations}")
else()
  string(APPEND _success_pattern
    ".*time_treatment=${HDG_MAIN_TIME_INTEGRATOR}")
endif()
if(DEFINED HDG_MAIN_EXPECT_POSITIVITY_LIMITER)
  string(APPEND _success_pattern
    ".*positivity_limiter=${HDG_MAIN_EXPECT_POSITIVITY_LIMITER}")
endif()
if(DEFINED HDG_MAIN_EXPECT_ZHANG_SHU_POINTS)
  string(APPEND _success_pattern
    ".*zhang_shu_points=${HDG_MAIN_EXPECT_ZHANG_SHU_POINTS}")
endif()
if(DEFINED HDG_MAIN_EXPECT_C_ALPHA)
  string(APPEND _success_pattern
    ".*c_alpha=${HDG_MAIN_EXPECT_C_ALPHA}")
endif()
if(DEFINED HDG_MAIN_CFL)
  list(APPEND _arguments --cfl-number "${HDG_MAIN_CFL}")
endif()
if(DEFINED HDG_MAIN_TIME_ATOL)
  list(APPEND _arguments --time-atol "${HDG_MAIN_TIME_ATOL}")
endif()
if(DEFINED HDG_MAIN_TIME_RTOL)
  list(APPEND _arguments --time-rtol "${HDG_MAIN_TIME_RTOL}")
endif()
if(DEFINED HDG_MAIN_DT_MIN)
  list(APPEND _arguments --dt-min "${HDG_MAIN_DT_MIN}")
endif()
if(DEFINED HDG_MAIN_EXPECTED_REJECTIONS)
  string(APPEND _success_pattern
    ".*rejected_attempts=${HDG_MAIN_EXPECTED_REJECTIONS}")
endif()
if(HDG_MAIN_NO_VTK)
  list(APPEND _arguments --no-vtk_save)
else()
  list(APPEND _arguments --time-step-save "${HDG_MAIN_DT_SAVE}" --vtk_save)
endif()

if(HDG_MAIN_RANKS EQUAL 1)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${HDG_MAIN_OUTPUT}"
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
    RESULT_VARIABLE _status OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
else()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${HDG_MAIN_OUTPUT}"
      "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}" "${HDG_MAIN_RANKS}"
      ${HDG_MAIN_MPIEXEC_PREFLAGS}
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      ${HDG_MAIN_MPIEXEC_POSTFLAGS}
    RESULT_VARIABLE _status OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
endif()

if(DEFINED HDG_MAIN_EXPECT_REJECTION_PATTERN)
  set(_combined_output "${_stdout}\n${_stderr}")
  if(_status EQUAL 0 OR
     NOT _combined_output MATCHES "${HDG_MAIN_EXPECT_REJECTION_PATTERN}")
    message(FATAL_ERROR
      "generic HDG main did not produce the selected rejection: ${_status}\n${_stdout}\n${_stderr}")
  endif()
  file(REMOVE_RECURSE "${HDG_MAIN_OUTPUT}")
  return()
endif()

if(NOT _status EQUAL 0 OR
   NOT _stdout MATCHES "${_success_pattern}" OR
   NOT _stderr STREQUAL "")
  message(FATAL_ERROR
    "generic HDG main execution failed: ${_status}\n${_stdout}\n${_stderr}")
endif()

if(HDG_MAIN_VERIFY_REPEATABLE_OUTPUT AND NOT HDG_MAIN_NO_VTK)
  set(_repeat_output "${HDG_MAIN_OUTPUT}-exact-repeat")
  file(REMOVE_RECURSE "${_repeat_output}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${_repeat_output}"
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
    RESULT_VARIABLE _repeat_status
    OUTPUT_VARIABLE _repeat_stdout
    ERROR_VARIABLE _repeat_stderr)
  if(NOT _repeat_status EQUAL 0 OR
     NOT _repeat_stdout MATCHES "${_success_pattern}" OR
     NOT _repeat_stderr STREQUAL "")
    message(FATAL_ERROR
      "repeated HDG main execution failed: ${_repeat_status}\n${_repeat_stdout}\n${_repeat_stderr}")
  endif()
endif()

if(HDG_MAIN_DEVICE STREQUAL "cuda")
  set(_collection
    "dgsem_smooth_advection_periodic_hdg_cuda_${HDG_MAIN_CUDA_ENGINE}_p_${HDG_MAIN_ORDER}_ref_${_refinement_levels}")
else()
  set(_collection
    "dgsem_smooth_advection_periodic_hdg_p_${HDG_MAIN_ORDER}_ref_${_refinement_levels}")
endif()
set(_root "${HDG_MAIN_OUTPUT}/${_collection}")
if(HDG_MAIN_NO_VTK)
  file(GLOB_RECURSE _artifacts LIST_DIRECTORIES FALSE
    "${HDG_MAIN_OUTPUT}/*")
  if(_artifacts)
    message(FATAL_ERROR "HDG published output after --no-vtk_save: ${_artifacts}")
  endif()
  return()
endif()
set(_pvd "${_root}/${_collection}.pvd")
if(NOT EXISTS "${_pvd}")
  message(FATAL_ERROR "missing CUDA HDG PVD series: ${_pvd}")
endif()
file(READ "${_pvd}" _pvd_text)
string(REPLACE "," ";" _cycles "${HDG_MAIN_EXPECTED_CYCLES}")
string(REPLACE "," ";" _times "${HDG_MAIN_EXPECTED_TIMES}")
list(LENGTH _cycles _cycle_count)
list(LENGTH _times _time_count)
if(NOT _cycle_count EQUAL _time_count OR _cycle_count EQUAL 0)
  message(FATAL_ERROR "CUDA HDG expected output cycles/times are inconsistent")
endif()
math(EXPR _last_output_index "${_cycle_count} - 1")
foreach(_index RANGE 0 ${_last_output_index})
  list(GET _cycles ${_index} _cycle)
  list(GET _times ${_index} _time)
  if(NOT EXISTS "${_root}/Cycle${_cycle}/data.pvtu")
    message(FATAL_ERROR "missing committed CUDA HDG output cycle ${_cycle}")
  endif()
  if(NOT _pvd_text MATCHES
      "timestep=\\\"${_time}\\\"[^<]*Cycle${_cycle}/data.pvtu")
    message(FATAL_ERROR "CUDA HDG PVD misses committed time ${_time} cycle ${_cycle}")
  endif()
  file(READ "${_root}/Cycle${_cycle}/data.pvtu" _pvtu_text)
  require_euler_field(_pvtu_text PDataArray density 1
    "HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray momentum 2
    "HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray energy 1
    "HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray velocity 2
    "HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray pressure 1
    "HDG PVTU cycle ${_cycle}")
  if(HDG_MAIN_REGULARIZATION STREQUAL "igr")
    require_euler_field(_pvtu_text PDataArray sigma 1
      "IGR HDG PVTU cycle ${_cycle}")
  elseif(_pvtu_text MATCHES "Name=\\\"sigma\\\"")
    message(FATAL_ERROR
      "conventional HDG PVTU cycle ${_cycle} unexpectedly contains sigma")
  endif()
  foreach(_field IN ITEMS
      zhang_shu_ever_active zhang_shu_theta_rho_min_cumulative
      zhang_shu_theta_p_min_cumulative)
    if(HDG_MAIN_EXPECT_ZHANG_SHU_FIELDS)
      require_euler_field(_pvtu_text PDataArray "${_field}" 1
        "HDG PVTU cycle ${_cycle}")
    elseif(_pvtu_text MATCHES "Name=\\\"${_field}\\\"")
      message(FATAL_ERROR
        "HDG PVTU cycle ${_cycle} unexpectedly contains ${_field}")
    endif()
  endforeach()
  if(_pvtu_text MATCHES "Name=\\\"conservative\\\"")
    message(FATAL_ERROR
      "HDG PVTU cycle ${_cycle} exposes obsolete packed conservative data")
  endif()
  string(REGEX MATCHALL
    "<Piece[^>]*>"
    _pvtu_piece_references "${_pvtu_text}")
  list(LENGTH _pvtu_piece_references _pvtu_piece_count)
  if(NOT _pvtu_piece_count EQUAL HDG_MAIN_RANKS)
    message(FATAL_ERROR
      "CUDA HDG PVTU cycle ${_cycle} has ${_pvtu_piece_count} piece references; expected ${HDG_MAIN_RANKS}")
  endif()
  math(EXPR _last_rank "${HDG_MAIN_RANKS} - 1")
  set(_cycle_cell_total 0)
  foreach(_rank RANGE 0 ${_last_rank})
    set(_rank_text "000000${_rank}")
    string(LENGTH "${_rank_text}" _rank_text_length)
    math(EXPR _rank_suffix_start "${_rank_text_length} - 6")
    string(SUBSTRING "${_rank_text}" ${_rank_suffix_start} 6 _rank_suffix)
    set(_vtu "${_root}/Cycle${_cycle}/proc${_rank_suffix}.vtu")
    if(NOT _pvtu_text MATCHES
        "<Piece Source=\\\"proc${_rank_suffix}\\.vtu\\\"/>")
      message(FATAL_ERROR
        "CUDA HDG PVTU cycle ${_cycle} omits rank ${_rank} piece proc${_rank_suffix}.vtu")
    endif()
    if(NOT EXISTS "${_vtu}")
      message(FATAL_ERROR "missing CUDA HDG VTU piece for rank ${_rank}: ${_vtu}")
    endif()
    file(READ "${_vtu}" _vtu_text)
    string(REGEX MATCHALL "NumberOfCells=\"[0-9]+\""
      _piece_cell_fields "${_vtu_text}")
    list(LENGTH _piece_cell_fields _piece_cell_field_count)
    if(NOT _piece_cell_field_count EQUAL 1)
      message(FATAL_ERROR
        "CUDA HDG VTU piece for rank ${_rank} cycle ${_cycle} must contain exactly one numeric NumberOfCells attribute")
    endif()
    list(GET _piece_cell_fields 0 _piece_cell_field)
    string(REGEX REPLACE "NumberOfCells=\"([0-9]+)\"" "\\1"
      _piece_cells "${_piece_cell_field}")
    math(EXPR _cycle_cell_total "${_cycle_cell_total} + ${_piece_cells}")
    require_euler_field(_vtu_text DataArray density 1
      "HDG VTU rank ${_rank} cycle ${_cycle}")
    require_euler_field(_vtu_text DataArray momentum 2
      "HDG VTU rank ${_rank} cycle ${_cycle}")
    require_euler_field(_vtu_text DataArray energy 1
      "HDG VTU rank ${_rank} cycle ${_cycle}")
    require_euler_field(_vtu_text DataArray velocity 2
      "HDG VTU rank ${_rank} cycle ${_cycle}")
    require_euler_field(_vtu_text DataArray pressure 1
      "HDG VTU rank ${_rank} cycle ${_cycle}")
    if(HDG_MAIN_REGULARIZATION STREQUAL "igr")
      require_euler_field(_vtu_text DataArray sigma 1
        "IGR HDG VTU rank ${_rank} cycle ${_cycle}")
    elseif(_vtu_text MATCHES "Name=\\\"sigma\\\"")
      message(FATAL_ERROR
        "conventional HDG VTU rank ${_rank} cycle ${_cycle} unexpectedly contains sigma")
    endif()
    foreach(_field IN ITEMS
        zhang_shu_ever_active zhang_shu_theta_rho_min_cumulative
        zhang_shu_theta_p_min_cumulative)
      if(HDG_MAIN_EXPECT_ZHANG_SHU_FIELDS)
        require_euler_field(_vtu_text DataArray "${_field}" 1
          "HDG VTU rank ${_rank} cycle ${_cycle}")
      elseif(_vtu_text MATCHES "Name=\\\"${_field}\\\"")
        message(FATAL_ERROR
          "HDG VTU rank ${_rank} cycle ${_cycle} unexpectedly contains ${_field}")
      endif()
    endforeach()
    if(_vtu_text MATCHES "Name=\\\"conservative\\\"")
      message(FATAL_ERROR
        "HDG VTU rank ${_rank} cycle ${_cycle} exposes obsolete conservative data")
    endif()
  endforeach()
  if(NOT _cycle_cell_total EQUAL _expected_global_cells)
    message(FATAL_ERROR
      "CUDA HDG VTU cycle ${_cycle} has ${_cycle_cell_total} global cells; expected ${_expected_global_cells} after serial+parallel refinement")
  endif()
endforeach()
string(REGEX MATCHALL "Cycle[0-9][0-9][0-9][0-9][0-9][0-9]/data.pvtu"
  _pvd_entries "${_pvd_text}")
list(LENGTH _pvd_entries _count)
if(NOT _count EQUAL _cycle_count)
  message(FATAL_ERROR "CUDA HDG PVD has ${_count} publications; expected ${_cycle_count}")
endif()

if(HDG_MAIN_VERIFY_REPEATABLE_OUTPUT)
  file(GLOB_RECURSE _artifacts RELATIVE "${HDG_MAIN_OUTPUT}"
    LIST_DIRECTORIES FALSE "${HDG_MAIN_OUTPUT}/*")
  file(GLOB_RECURSE _repeat_artifacts RELATIVE "${_repeat_output}"
    LIST_DIRECTORIES FALSE "${_repeat_output}/*")
  list(SORT _artifacts)
  list(SORT _repeat_artifacts)
  if(NOT "${_artifacts}" STREQUAL "${_repeat_artifacts}")
    message(FATAL_ERROR
      "repeated CUDA HDG publication set differs:\n${_artifacts}\n${_repeat_artifacts}")
  endif()
  foreach(_artifact IN LISTS _artifacts)
    file(SHA256 "${HDG_MAIN_OUTPUT}/${_artifact}" _hash)
    file(SHA256 "${_repeat_output}/${_artifact}" _repeat_hash)
    if(NOT _hash STREQUAL _repeat_hash)
      message(FATAL_ERROR
        "repeated CUDA HDG output is not byte-identical: ${_artifact}")
    endif()
  endforeach()
endif()
