if(NOT DEFINED HDG_MAIN_EXECUTABLE OR
   NOT DEFINED HDG_MAIN_MESH OR
   NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR
   NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR
    "HDG main executable, mesh, ranks, and MPI launcher are required.")
endif()

set(_base_arguments
  --discretization hdg
  --device cpu
  --regularization igr
  --serial-refine 0
  --parallel-refine 0
  --mesh "${HDG_MAIN_MESH}"
  --time-step 0.00001
  --t-final 0.00001
  --no-vtk_save
  --no-glvis)

function(run_igr_success order expected_c_alpha expected_alpha expected_penalty)
  set(_arguments ${_base_arguments} --order ${order} ${ARGN})
  if(HDG_MAIN_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}" "${HDG_MAIN_RANKS}"
        ${HDG_MAIN_MPIEXEC_PREFLAGS}
        "${HDG_MAIN_EXECUTABLE}" ${_arguments}
        ${HDG_MAIN_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  if(NOT _status EQUAL 0 OR
     NOT _stdout MATCHES
       "\\[hdg-main\\] status=Success.*regularization=igr.*c_alpha=${expected_c_alpha}.*alpha_effective=${expected_alpha}.*alpha_length_policy=average.*h_eff=0.333333.*euler_trace_penalty=${expected_penalty}" OR
     NOT _stderr STREQUAL "")
    message(FATAL_ERROR
      "HDG p${order} IGR execution failed: ${_status}\\n${_stdout}\\n${_stderr}")
  endif()
endfunction()

function(expect_igr_rejection expected_diagnostic)
  set(_arguments ${_base_arguments} ${ARGN})
  if(HDG_MAIN_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}" "${HDG_MAIN_RANKS}"
        ${HDG_MAIN_MPIEXEC_PREFLAGS}
        "${HDG_MAIN_EXECUTABLE}" ${_arguments}
        ${HDG_MAIN_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  set(_output "${_stdout}\\n${_stderr}")
  if("${_status}" STREQUAL "0" OR NOT _output MATCHES "${expected_diagnostic}")
    message(FATAL_ERROR
      "HDG IGR rejection failed: ${_status}\\n${_output}")
  endif()
endfunction()

# Every supported order uses the same periodic no-VTK Euler lifecycle.  The
# explicit p1 tuple confirms that the dimensionless C_alpha is scaled by the
# case's Average policy while the independent Euler trace penalty is not.
run_igr_success(1 1 0.111111 1)
run_igr_success(1 1.25 0.138889 0.75
  --hdg-igr-alpha 1.25 --hdg-igr-beta 0.75)
run_igr_success(2 1 0.111111 1)
run_igr_success(3 1 0.111111 1)
expect_igr_rejection(
  "HDG IGR c_alpha and Euler trace penalty must be finite and positive"
  --order 1 --hdg-igr-alpha 0)
expect_igr_rejection(
  "HDG IGR c_alpha and Euler trace penalty must be finite and positive"
  --order 1 --hdg-igr-beta 0)
expect_igr_rejection(
  "HDG IGR does not support the element-local alpha length policy"
  --order 1 --igr-alpha-length local)
