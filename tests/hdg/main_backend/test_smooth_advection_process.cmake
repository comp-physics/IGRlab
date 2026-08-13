if(NOT DEFINED HDG_MAIN_EXECUTABLE OR
   NOT DEFINED HDG_MAIN_OUTPUT OR
   NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR
   NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR
    "HDG main executable, output, ranks, and MPI launcher are required.")
endif()

set(_arguments
  --discretization hdg
  --device cpu
  --order 1
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00001
  --t-final 0.00001
  --time-step-save 0.00001
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
   NOT _stdout MATCHES "\\[hdg-main\\] status=Success.*step=1" OR
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
foreach(_cycle IN ITEMS 000000 000001)
  if(NOT _pvd_text MATCHES "Cycle${_cycle}/data.pvtu" OR
     NOT EXISTS "${_root}/Cycle${_cycle}/data.pvtu" OR
     NOT EXISTS "${_root}/Cycle${_cycle}/proc000000.vtu")
    message(FATAL_ERROR "missing complete HDG ParaView cycle ${_cycle}")
  endif()
  if(HDG_MAIN_RANKS EQUAL 2 AND
     NOT EXISTS "${_root}/Cycle${_cycle}/proc000001.vtu")
    message(FATAL_ERROR "missing rank-one VTU for HDG cycle ${_cycle}")
  endif()
endforeach()

file(GLOB _cycles LIST_DIRECTORIES TRUE "${_root}/Cycle*")
list(LENGTH _cycles _cycle_count)
if(NOT _cycle_count EQUAL 2)
  message(FATAL_ERROR
    "HDG output has ${_cycle_count} cycle directories; expected exactly 2")
endif()
