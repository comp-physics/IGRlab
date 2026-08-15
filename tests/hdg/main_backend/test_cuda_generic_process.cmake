if(NOT DEFINED HDG_MAIN_EXECUTABLE OR NOT DEFINED HDG_MAIN_OUTPUT OR
   NOT DEFINED HDG_MAIN_MESH OR NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR "CUDA main executable, output, mesh, ranks, and MPI launcher are required.")
endif()

if(NOT DEFINED HDG_MAIN_ORDER)
  set(HDG_MAIN_ORDER 1)
endif()
if(NOT DEFINED HDG_MAIN_DT)
  set(HDG_MAIN_DT 0.05)
endif()
if(NOT DEFINED HDG_MAIN_T_FINAL)
  set(HDG_MAIN_T_FINAL 0.10)
endif()
if(NOT DEFINED HDG_MAIN_DT_SAVE)
  set(HDG_MAIN_DT_SAVE 0.05)
endif()
if(NOT DEFINED HDG_MAIN_EXPECTED_STEPS)
  set(HDG_MAIN_EXPECTED_STEPS 2)
endif()
if(HDG_MAIN_DT STREQUAL "0.03" AND HDG_MAIN_T_FINAL STREQUAL "0.10" AND
   HDG_MAIN_DT_SAVE STREQUAL "0.05")
  set(HDG_MAIN_EXPECTED_STEPS 4)
endif()
if(NOT DEFINED HDG_MAIN_N)
  set(HDG_MAIN_N 3)
endif()
if(NOT DEFINED HDG_MAIN_SERIAL_REFINE)
  set(HDG_MAIN_SERIAL_REFINE 0)
endif()
if(NOT DEFINED HDG_MAIN_PARALLEL_REFINE)
  set(HDG_MAIN_PARALLEL_REFINE 0)
endif()
if(HDG_MAIN_N LESS 1 OR HDG_MAIN_N GREATER 100000 OR
   HDG_MAIN_SERIAL_REFINE LESS 0 OR HDG_MAIN_PARALLEL_REFINE LESS 0)
  message(FATAL_ERROR "CUDA HDG cell census requires bounded nonnegative mesh/refinement inputs.")
endif()
math(EXPR _refinement_levels
  "${HDG_MAIN_SERIAL_REFINE} + ${HDG_MAIN_PARALLEL_REFINE}")
if(_refinement_levels GREATER 10)
  message(FATAL_ERROR "CUDA HDG cell census refinement depth exceeds checked process-test bounds.")
endif()
math(EXPR _expected_global_cells "${HDG_MAIN_N} * ${HDG_MAIN_N}")
if(_refinement_levels GREATER 0)
  foreach(_level RANGE 1 ${_refinement_levels})
    math(EXPR _expected_global_cells "${_expected_global_cells} * 4")
  endforeach()
endif()

file(REMOVE_RECURSE "${HDG_MAIN_OUTPUT}")
set(_arguments
  --discretization hdg
  --device cuda
  --mesh "${HDG_MAIN_MESH}"
  --order "${HDG_MAIN_ORDER}"
  --serial-refine "${HDG_MAIN_SERIAL_REFINE}"
  --parallel-refine "${HDG_MAIN_PARALLEL_REFINE}"
  --time-step "${HDG_MAIN_DT}"
  --t-final "${HDG_MAIN_T_FINAL}"
  --no-glvis)
if(DEFINED HDG_MAIN_CFL)
  list(APPEND _arguments --cfl-number "${HDG_MAIN_CFL}")
endif()
if(HDG_MAIN_NO_VTK)
  list(APPEND _arguments --no-vtk_save)
else()
  list(APPEND _arguments --time-step-save "${HDG_MAIN_DT_SAVE}" --vtk_save)
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
   NOT _stdout MATCHES "\\[hdg-main\\] status=Success.*step=${HDG_MAIN_EXPECTED_STEPS}" OR
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
  if(NOT _pvtu_text MATCHES
      "<PDataArray[^>]*(Name=\\\"conservative\\\"[^>]*NumberOfComponents=\\\"4\\\"|NumberOfComponents=\\\"4\\\"[^>]*Name=\\\"conservative\\\")[^>]*>")
    message(FATAL_ERROR "CUDA HDG PVTU cycle ${_cycle} misses four-component conservative data")
  endif()
  string(REGEX MATCHALL
    "<Piece[^>]*>"
    _pvtu_piece_references "${_pvtu_text}")
  list(LENGTH _pvtu_piece_references _pvtu_piece_count)
  if(NOT _pvtu_piece_count EQUAL HDG_MAIN_RANKS)
    message(FATAL_ERROR
      "CUDA HDG PVTU cycle ${_cycle} has ${_pvtu_piece_count} piece references; expected ${HDG_MAIN_RANKS}")
  endif()
  math(EXPR _last_rank "${HDG_MAIN_RANKS} - 1")
  set(_cycle_cell_total 0)
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
    string(REGEX MATCHALL "NumberOfCells=\"[0-9]+\""
      _piece_cell_fields "${_vtu_text}")
    list(LENGTH _piece_cell_fields _piece_cell_field_count)
    if(NOT _piece_cell_field_count EQUAL 1)
      message(FATAL_ERROR
        "CUDA HDG VTU piece for rank ${_rank} cycle ${_cycle} must contain exactly one numeric NumberOfCells attribute")
    endif()
    list(GET _piece_cell_fields 0 _piece_cell_field)
    string(REGEX REPLACE "NumberOfCells=\"([0-9]+)\"" "\\1"
      _piece_cells "${_piece_cell_field}")
    math(EXPR _cycle_cell_total "${_cycle_cell_total} + ${_piece_cells}")
    if(NOT _vtu_text MATCHES
        "<DataArray[^>]*(Name=\\\"conservative\\\"[^>]*NumberOfComponents=\\\"4\\\"|NumberOfComponents=\\\"4\\\"[^>]*Name=\\\"conservative\\\")[^>]*>")
      message(FATAL_ERROR
        "CUDA HDG VTU piece for rank ${_rank} cycle ${_cycle} misses four-component conservative data")
    endif()
  endforeach()
  if(NOT _cycle_cell_total EQUAL _expected_global_cells)
    message(FATAL_ERROR
      "CUDA HDG VTU cycle ${_cycle} has ${_cycle_cell_total} global cells; expected ${_expected_global_cells} after serial+parallel refinement")
  endif()
endforeach()
string(REGEX MATCHALL "Cycle[0-9][0-9][0-9][0-9][0-9][0-9]/data.pvtu"
  _pvd_entries "${_pvd_text}")
list(LENGTH _pvd_entries _count)
if(NOT _count EQUAL 3)
  message(FATAL_ERROR "CUDA HDG PVD has ${_count} publications; expected 3")
endif()
