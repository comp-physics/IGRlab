if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_MESH)
  message(FATAL_ERROR "CUDA main executable and periodic N3 mesh are required.")
endif()

execute_process(
  COMMAND "${HDG_MAIN_EXECUTABLE}"
    --discretization hdg
    --device cpu
    --mesh "${HDG_MAIN_MESH}"
    --order 1
    --serial-refine 0
    --parallel-refine 0
    --time-step 0.05
    --t-final 0.10
    --no-vtk_save
    --no-glvis
  RESULT_VARIABLE _status OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
if(_status EQUAL 0 OR
   NOT _stderr MATCHES "\\[hdg-main\\] status=Rejected")
  message(FATAL_ERROR
    "CUDA main backend accepted an excluded CPU profile: ${_status}\n${_stdout}\n${_stderr}")
endif()
