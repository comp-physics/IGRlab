if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_WORK)
  message(FATAL_ERROR "HDG main executable and isolated work directory are required.")
endif()
if(NOT DEFINED HDG_MAIN_DEVICE)
  set(HDG_MAIN_DEVICE cpu)
endif()

file(REMOVE_RECURSE "${HDG_MAIN_WORK}")
file(MAKE_DIRECTORY "${HDG_MAIN_WORK}")
execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}"
    --discretization dgsem
    --device "${HDG_MAIN_DEVICE}"
    --order 1
    --serial-refine 0
    --parallel-refine 0
    --time-step 0.00001
    --t-final 0.00001
    --analysis-steps 0
    --no-vtk_save
    --no-glvis
  WORKING_DIRECTORY "${HDG_MAIN_WORK}"
  RESULT_VARIABLE _status
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)

if(NOT _status EQUAL 0 OR
   _stdout MATCHES "\\[hdg-main\\]" OR
   NOT _stderr STREQUAL "")
  message(FATAL_ERROR
    "feature-enabled legacy DGSEM route failed: ${_status}\n${_stdout}\n${_stderr}")
endif()
