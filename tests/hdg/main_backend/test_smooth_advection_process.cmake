if(NOT DEFINED HDG_MAIN_EXECUTABLE OR
   NOT DEFINED HDG_MAIN_OUTPUT OR
   NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR
   NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR
    "HDG main executable, output, ranks, and MPI launcher are required.")
endif()

function(require_euler_field text_variable tag field components context)
  set(_text "${${text_variable}}")
  if(NOT _text MATCHES
      "<${tag}[^>]*(Name=\"${field}\"[^>]*NumberOfComponents=\"${components}\"|NumberOfComponents=\"${components}\"[^>]*Name=\"${field}\")[^>]*>")
    message(FATAL_ERROR
      "${context} misses ${components}-component Euler field '${field}'")
  endif()
endfunction()

set(_arguments
  --discretization hdg
  --device cpu
  --order 1
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00006
  --t-final 0.00013
  --time-step-save 0.00005
  --vtk_save
  --no-glvis)

file(REMOVE_RECURSE "${HDG_MAIN_OUTPUT}")
if(HDG_MAIN_RANKS EQUAL 1)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${HDG_MAIN_OUTPUT}"
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
else()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${HDG_MAIN_OUTPUT}"
      "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}" "${HDG_MAIN_RANKS}"
      ${HDG_MAIN_MPIEXEC_PREFLAGS}
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      ${HDG_MAIN_MPIEXEC_POSTFLAGS}
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
endif()

if(NOT _status EQUAL 0 OR
   NOT _stdout MATCHES "\\[hdg-main\\] status=Success.*step=3" OR
   NOT _stderr STREQUAL "")
  message(FATAL_ERROR
    "generic HDG main execution failed: ${_status}\n${_stdout}\n${_stderr}")
endif()

set(_collection "dgsem_smooth_advection_periodic_hdg_p_1_ref_0")
set(_root "${HDG_MAIN_OUTPUT}/${_collection}")
set(_pvd "${_root}/${_collection}.pvd")
if(NOT EXISTS "${_pvd}")
  message(FATAL_ERROR "missing HDG PVD series: ${_pvd}")
endif()
file(READ "${_pvd}" _pvd_text)
set(_expected_cycles 000000 000001 000002 000003)
set(_expected_times 0 5e-05 0.0001 0.00013)
foreach(_index RANGE 0 3)
  list(GET _expected_cycles ${_index} _cycle)
  list(GET _expected_times ${_index} _time)
  if(NOT _pvd_text MATCHES
       "timestep=\\\"${_time}\\\"[^<]*Cycle${_cycle}/data.pvtu" OR
     NOT EXISTS "${_root}/Cycle${_cycle}/data.pvtu" OR
     NOT EXISTS "${_root}/Cycle${_cycle}/proc000000.vtu")
    message(FATAL_ERROR "missing complete HDG ParaView cycle ${_cycle}")
  endif()
  if(HDG_MAIN_RANKS EQUAL 2 AND
     NOT EXISTS "${_root}/Cycle${_cycle}/proc000001.vtu")
    message(FATAL_ERROR "missing rank-one VTU for HDG cycle ${_cycle}")
  endif()
  file(READ "${_root}/Cycle${_cycle}/data.pvtu" _pvtu_text)
  file(READ "${_root}/Cycle${_cycle}/proc000000.vtu" _vtu_text)
  require_euler_field(_pvtu_text PDataArray density 1
    "CPU HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray momentum 2
    "CPU HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray energy 1
    "CPU HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray velocity 2
    "CPU HDG PVTU cycle ${_cycle}")
  require_euler_field(_pvtu_text PDataArray pressure 1
    "CPU HDG PVTU cycle ${_cycle}")
  require_euler_field(_vtu_text DataArray density 1
    "CPU HDG VTU cycle ${_cycle}")
  require_euler_field(_vtu_text DataArray momentum 2
    "CPU HDG VTU cycle ${_cycle}")
  require_euler_field(_vtu_text DataArray energy 1
    "CPU HDG VTU cycle ${_cycle}")
  require_euler_field(_vtu_text DataArray velocity 2
    "CPU HDG VTU cycle ${_cycle}")
  require_euler_field(_vtu_text DataArray pressure 1
    "CPU HDG VTU cycle ${_cycle}")
  if(_pvtu_text MATCHES "Name=\\\"(sigma|conservative)\\\"" OR
     _vtu_text MATCHES "Name=\\\"(sigma|conservative)\\\"")
    message(FATAL_ERROR
      "conventional CPU HDG cycle ${_cycle} exposes a nonstandard point field")
  endif()
endforeach()

string(REGEX MATCHALL "Cycle[0-9][0-9][0-9][0-9][0-9][0-9]/data.pvtu"
  _pvd_entries "${_pvd_text}")
list(LENGTH _pvd_entries _pvd_entry_count)
if(NOT _pvd_entry_count EQUAL 4)
  message(FATAL_ERROR
    "HDG PVD has ${_pvd_entry_count} publications; expected exactly 4")
endif()

file(GLOB _cycles LIST_DIRECTORIES TRUE "${_root}/Cycle*")
list(LENGTH _cycles _cycle_count)
if(NOT _cycle_count EQUAL 4)
  message(FATAL_ERROR
    "HDG output has ${_cycle_count} cycle directories; expected exactly 4")
endif()
