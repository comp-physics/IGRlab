if(NOT DEFINED TEST_EXECUTABLE OR NOT EXISTS "${TEST_EXECUTABLE}")
  message(FATAL_ERROR "TEST_EXECUTABLE must name the scalar-IGR CUDA rank-one test")
endif()

execute_process(
  COMMAND "${TEST_EXECUTABLE}" --c2
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)

if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "scalar-IGR CUDA rank-one C2 failed (${_result})\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
endif()

if(NOT _stdout MATCHES "SC25_CUDA_RANK1_C1_PASS" OR
   NOT _stdout MATCHES "SC25_CUDA_RANK1_C2_PASS")
  message(FATAL_ERROR
    "scalar-IGR CUDA rank-one C2 omitted mandatory same-executable C1/C2 evidence markers\n${_stdout}")
endif()
