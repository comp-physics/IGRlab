if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_MESH OR
   NOT DEFINED HDG_MAIN_DEVICE)
  message(FATAL_ERROR
    "HDG executable, bounded mesh, and backend device are required.")
endif()

string(TOLOWER "${HDG_MAIN_DEVICE}" _device)
if(_device STREQUAL "cpu")
  set(_engines cpu)
  set(_expected_reason
    "\\[hdg-main\\] status=Rejected reason=generic HDG restart/state-file routes are unsupported")
elseif(_device STREQUAL "cuda")
  set(_engines ordinary tensor)
else()
  message(FATAL_ERROR "Unsupported HDG test device '${HDG_MAIN_DEVICE}'.")
endif()

set(_base_arguments
  --discretization hdg
  --device "${_device}"
  --regularization none
  --mesh "${HDG_MAIN_MESH}"
  --order 1
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00001
  --t-final 0.00001
  --no-vtk_save
  --no-glvis)

foreach(_engine IN LISTS _engines)
  if(_device STREQUAL "cuda")
    set(_engine_arguments --hdg-cuda-engine "${_engine}")
    set(_route_expected_reason
      "\\[hdg-main\\] status=Rejected reason=generic CUDA HDG restart/state-file routes are unsupported cuda_engine=${_engine}")
  else()
    set(_engine_arguments)
    set(_route_expected_reason "${_expected_reason}")
  endif()
  if(_device STREQUAL "cpu")
    set(_routes restart state-file hdg-checkpoint hdg-restart-checkpoint)
  else()
    set(_routes restart state-file)
  endif()
  foreach(_route IN LISTS _routes)
    if(_route STREQUAL "restart")
      set(_route_arguments --restart-vtk unsupported-state.pvd)
      set(_selected_expected_reason "${_route_expected_reason}")
    elseif(_route STREQUAL "state-file")
      set(_route_arguments --save-state)
      set(_selected_expected_reason "${_route_expected_reason}")
    elseif(_route STREQUAL "hdg-checkpoint")
      set(_route_arguments --hdg-checkpoint unsupported-state.chk)
      set(_selected_expected_reason
        "\\[hdg-main\\] status=Rejected reason=generic CPU HDG checkpoint routes are unsupported")
    else()
      set(_route_arguments --hdg-restart-checkpoint unsupported-state.chk)
      set(_selected_expected_reason
        "\\[hdg-main\\] status=Rejected reason=generic CPU HDG checkpoint routes are unsupported")
    endif()
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_base_arguments}
        ${_engine_arguments} ${_route_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
    if(_status EQUAL 0 OR
       NOT _stderr MATCHES "${_selected_expected_reason}")
      message(FATAL_ERROR
        "HDG ${_device}/${_engine} ${_route} route was not rejected atomically: "
        "${_status}\n${_stdout}\n${_stderr}")
    endif()
  endforeach()
endforeach()
