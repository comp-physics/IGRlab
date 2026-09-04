if(NOT DEFINED TEST_EXE OR NOT DEFINED OUTPUT_ROOT OR
   NOT DEFINED MPIEXEC OR NOT DEFINED MPI_NUMPROC_FLAG)
  message(FATAL_ERROR "missing static-source mismatch test arguments")
endif()

function(require_rejected_restart mode diagnosis)
  execute_process(
    COMMAND "${MPIEXEC}" "${MPI_NUMPROC_FLAG}" 4 "${TEST_EXE}"
            "${OUTPUT_ROOT}/${mode}" "${mode}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  set(output "${stdout}\n${stderr}")
  if(status EQUAL 0)
    message(FATAL_ERROR "${mode} restart was accepted\n${output}")
  endif()
  if(NOT output MATCHES "${diagnosis}")
    message(FATAL_ERROR
      "${mode} restart failed without its compatibility diagnosis\n${output}")
  endif()
endfunction()

require_rejected_restart(
  changed-source
  "restart manifest static source is incompatible with the active projected source"
)
require_rejected_restart(
  schema2-source
  "restart manifest schema 2 cannot certify the active static source"
)

message(STATUS "vtk-restart-static-source-mismatch: PASS")
