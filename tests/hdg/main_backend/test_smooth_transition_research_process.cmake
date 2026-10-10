if(NOT DEFINED HDG_W8C_EXECUTABLE OR
   NOT DEFINED HDG_W8C_WORK OR
   NOT DEFINED HDG_W8C_RANKS OR
   NOT DEFINED HDG_W8C_MPIEXEC OR
   NOT DEFINED HDG_W8C_NUMPROC_FLAG)
  message(FATAL_ERROR
    "W8C research executable, work directory, ranks, and MPI launcher are required.")
endif()

file(REMOVE_RECURSE "${HDG_W8C_WORK}")
file(MAKE_DIRECTORY "${HDG_W8C_WORK}")

set(_base_arguments
  --discretization hdg
  --device cpu
  --order 1
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.0001
  --t-final 0.0002
  --no-vtk_save
  --no-glvis)

function(run_w8c_smoke expected_regularization)
  set(_arguments ${_base_arguments} --regularization "${expected_regularization}" ${ARGN})
  set(_expected_status
    "\\[hdg-main\\] status=Success.*step=2.*regularization=${expected_regularization}")
  if(expected_regularization STREQUAL "igr")
    set(_expected_status
      "${_expected_status}.*c_alpha=0.1.*alpha_effective=0.0015625.*alpha_length_policy=average.*h_eff=0.125.*euler_trace_penalty=2.5")
  endif()
  if(HDG_W8C_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_W8C_EXECUTABLE}" ${_arguments}
      WORKING_DIRECTORY "${HDG_W8C_WORK}"
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_W8C_MPIEXEC}" "${HDG_W8C_NUMPROC_FLAG}" "${HDG_W8C_RANKS}"
        ${HDG_W8C_MPIEXEC_PREFLAGS}
        "${HDG_W8C_EXECUTABLE}" ${_arguments}
        ${HDG_W8C_MPIEXEC_POSTFLAGS}
      WORKING_DIRECTORY "${HDG_W8C_WORK}"
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  if(NOT _status EQUAL 0 OR
     NOT _stdout MATCHES "${_expected_status}" OR
     NOT _stderr STREQUAL "")
    message(FATAL_ERROR
      "generic W8C research ${expected_regularization} smoke failed: ${_status}\\n${_stdout}\\n${_stderr}")
  endif()
endfunction()

# This is engineering coverage only.  It deliberately excludes W8C q14/q11
# observables, full-Newton policy, and frozen-runtime scientific parity.
run_w8c_smoke(none)
run_w8c_smoke(igr --hdg-igr-alpha 0.1 --hdg-igr-beta 2.5)
