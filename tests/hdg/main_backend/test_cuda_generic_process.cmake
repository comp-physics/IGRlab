if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_OUTPUT OR
   NOT DEFINED HDG_MAIN_MESH OR NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR "CUDA main executable, output, mesh, ranks, and MPI launcher are required.")
endif()

if(NOT DEFINED HDG_MAIN_ORDER)
  set(HDG_MAIN_ORDER 1)
endif()
if(NOT DEFINED HDG_MAIN_N)
  set(HDG_MAIN_N 3)
endif()

file(REMOVE_RECURSE "${HDG_MAIN_OUTPUT}")
set(_arguments
  --discretization hdg
  --device cuda
  --mesh "${HDG_MAIN_MESH}"
  --order "${HDG_MAIN_ORDER}"
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.05
  --t-final 0.10
  --no-glvis)
if(HDG_MAIN_NO_VTK)
  list(APPEND _arguments --no-vtk_save)
else()
  list(APPEND _arguments --time-step-save 0.05 --vtk_save)
endif()

if(HDG_MAIN_RANKS EQUAL 1)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${HDG_MAIN_OUTPUT}"
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
    RESULT_VARIABLE _status OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
else()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "DG_IGR_VTK_PREFIX_PATH=${HDG_MAIN_OUTPUT}"
      "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}" "${HDG_MAIN_RANKS}"
      ${HDG_MAIN_MPIEXEC_PREFLAGS}
      "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      ${HDG_MAIN_MPIEXEC_POSTFLAGS}
    RESULT_VARIABLE _status OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
endif()

if(NOT _status EQUAL 0 OR
   NOT _stdout MATCHES "\\[hdg-main\\] status=Success.*step=2.*time=0\\.1" OR
   NOT _stderr STREQUAL "")
  message(FATAL_ERROR
    "generic CUDA HDG main execution failed: ${_status}\n${_stdout}\n${_stderr}")
endif()

set(_collection
  "dgsem_smooth_advection_periodic_hdg_cuda_p_${HDG_MAIN_ORDER}_n_${HDG_MAIN_N}")
set(_root "${HDG_MAIN_OUTPUT}/${_collection}")
if(HDG_MAIN_NO_VTK)
  file(GLOB_RECURSE _artifacts LIST_DIRECTORIES FALSE
    "${HDG_MAIN_OUTPUT}/*")
  if(_artifacts)
    message(FATAL_ERROR "CUDA HDG published output after --no-vtk_save: ${_artifacts}")
  endif()
  return()
endif()
set(_pvd "${_root}/${_collection}.pvd")
if(NOT EXISTS "${_pvd}")
  message(FATAL_ERROR "missing CUDA HDG PVD series: ${_pvd}")
endif()
file(READ "${_pvd}" _pvd_text)
set(_cycles 000000 000001 000002)
set(_times 0 0.05 0.1)
foreach(_index RANGE 0 2)
  list(GET _cycles ${_index} _cycle)
  list(GET _times ${_index} _time)
  if(NOT EXISTS "${_root}/Cycle${_cycle}/data.pvtu")
    message(FATAL_ERROR "missing committed CUDA HDG output cycle ${_cycle}")
  endif()
  if(NOT _pvd_text MATCHES
      "timestep=\\\"${_time}\\\"[^<]*Cycle${_cycle}/data.pvtu")
    message(FATAL_ERROR "CUDA HDG PVD misses committed time ${_time} cycle ${_cycle}")
  endif()
  file(READ "${_root}/Cycle${_cycle}/data.pvtu" _pvtu_text)
  if(NOT _pvtu_text MATCHES "Name=\\\"conservative\\\"" OR
     NOT _pvtu_text MATCHES "NumberOfComponents=\\\"4\\\"")
    message(FATAL_ERROR "CUDA HDG PVTU cycle ${_cycle} misses four-component conservative data")
  endif()
  string(REGEX MATCHALL
    "<Piece Source=\\\"proc[0-9][0-9][0-9][0-9][0-9][0-9]\\.vtu\\\"/>"
    _pvtu_piece_references "${_pvtu_text}")
  list(LENGTH _pvtu_piece_references _pvtu_piece_count)
  if(NOT _pvtu_piece_count EQUAL HDG_MAIN_RANKS)
    message(FATAL_ERROR
      "CUDA HDG PVTU cycle ${_cycle} has ${_pvtu_piece_count} piece references; expected ${HDG_MAIN_RANKS}")
  endif()
  math(EXPR _last_rank "${HDG_MAIN_RANKS} - 1")
  foreach(_rank RANGE 0 ${_last_rank})
    set(_rank_text "000000${_rank}")
    string(LENGTH "${_rank_text}" _rank_text_length)
    math(EXPR _rank_suffix_start "${_rank_text_length} - 6")
    string(SUBSTRING "${_rank_text}" ${_rank_suffix_start} 6 _rank_suffix)
    set(_vtu "${_root}/Cycle${_cycle}/proc${_rank_suffix}.vtu")
    if(NOT _pvtu_text MATCHES
        "<Piece Source=\\\"proc${_rank_suffix}\\.vtu\\\"/>")
      message(FATAL_ERROR
        "CUDA HDG PVTU cycle ${_cycle} omits rank ${_rank} piece proc${_rank_suffix}.vtu")
    endif()
    if(NOT EXISTS "${_vtu}")
      message(FATAL_ERROR "missing CUDA HDG VTU piece for rank ${_rank}: ${_vtu}")
    endif()
    file(READ "${_vtu}" _vtu_text)
    if(NOT _vtu_text MATCHES "Name=\\\"conservative\\\"" OR
       NOT _vtu_text MATCHES "NumberOfComponents=\\\"4\\\"")
      message(FATAL_ERROR
        "CUDA HDG VTU piece for rank ${_rank} cycle ${_cycle} misses four-component conservative data")
    endif()
  endforeach()
endforeach()
string(REGEX MATCHALL "Cycle[0-9][0-9][0-9][0-9][0-9][0-9]/data.pvtu"
  _pvd_entries "${_pvd_text}")
list(LENGTH _pvd_entries _count)
if(NOT _count EQUAL 3)
  message(FATAL_ERROR "CUDA HDG PVD has ${_count} publications; expected 3")
endif()
