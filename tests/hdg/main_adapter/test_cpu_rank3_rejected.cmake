if(NOT DEFINED HDG_SMOOTH_PERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_OUTPUT OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_MPIEXEC OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_NUMPROC_FLAG)
  message(FATAL_ERROR "adapter executable, output, and MPI launcher are required")
endif()

file(REMOVE_RECURSE "${HDG_SMOOTH_PERIODIC_OUTPUT}")
file(REMOVE "${HDG_SMOOTH_PERIODIC_OUTPUT}.checkpoint")
execute_process(
  COMMAND "${HDG_SMOOTH_PERIODIC_MPIEXEC}"
    "${HDG_SMOOTH_PERIODIC_NUMPROC_FLAG}" 3
    ${HDG_SMOOTH_PERIODIC_MPIEXEC_PREFLAGS}
    "${HDG_SMOOTH_PERIODIC_EXECUTABLE}"
      --discretization hdg
      --hdg-output-dir "${HDG_SMOOTH_PERIODIC_OUTPUT}"
    ${HDG_SMOOTH_PERIODIC_MPIEXEC_POSTFLAGS}
  RESULT_VARIABLE status
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
set(combined_output "${output}${error}")
if(status EQUAL 0 OR
   NOT combined_output MATCHES "CPU HDG requires exactly one or two MPI ranks" OR
   EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}" OR
   EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}.checkpoint")
  message(FATAL_ERROR
    "rank-three adapter execution did not reject cleanly: ${status};${output};${error}")
endif()
