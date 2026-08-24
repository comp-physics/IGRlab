if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_MESH)
  message(FATAL_ERROR "CUDA HDG executable and bounded mesh are required.")
endif()

set(_base_arguments
  --discretization hdg
  --device cuda
  --regularization none
  --mesh "${HDG_MAIN_MESH}"
  --order 1
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00001
  --t-final 0.00001
  --no-vtk_save
  --no-glvis)

foreach(_engine IN ITEMS ordinary tensor)
  foreach(_route IN ITEMS restart state-file)
    if(_route STREQUAL "restart")
      set(_route_arguments --restart-vtk unsupported-state.pvd)
    else()
      set(_route_arguments --save-state)
    endif()
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_base_arguments}
        --hdg-cuda-engine "${_engine}" ${_route_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
    if(_status EQUAL 0 OR
       NOT _stderr MATCHES
         "\\[hdg-main\\] status=Rejected reason=generic CUDA HDG restart/state-file routes are unsupported cuda_engine=${_engine}")
      message(FATAL_ERROR
        "CUDA HDG ${_engine} ${_route} route was not rejected atomically: "
        "${_status}\n${_stdout}\n${_stderr}")
    endif()
  endforeach()
endforeach()
