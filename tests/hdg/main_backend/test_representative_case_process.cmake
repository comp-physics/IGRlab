if(NOT DEFINED HDG_PERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_NONPERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_REPRESENTATIVE_WORK)
  message(FATAL_ERROR
    "Periodic/nonperiodic generic HDG executables and a work directory are required.")
endif()

file(REMOVE_RECURSE "${HDG_REPRESENTATIVE_WORK}")
file(MAKE_DIRECTORY "${HDG_REPRESENTATIVE_WORK}")

set(_common_arguments
  --discretization hdg
  --device cpu
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00001
  --t-final 0.00001
  --no-glvis)

# Exercise the conventional source path while IGR augments the same solver.
# Distinct final VTK payloads prove that the projected source reaches the
# numerical request; a success-only smoke would not detect an accidentally
# dropped source view.
set(_zero_output "${HDG_REPRESENTATIVE_WORK}/igr-zero-source")
set(_source_output "${HDG_REPRESENTATIVE_WORK}/igr-nonzero-source")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DG_IGR_VTK_PREFIX_PATH=${_zero_output}"
    "${HDG_PERIODIC_EXECUTABLE}" ${_common_arguments}
    --order 1
    --regularization igr
    --time-step-save 0.00001
    --vtk_save
  RESULT_VARIABLE _zero_status
  OUTPUT_VARIABLE _zero_stdout
  ERROR_VARIABLE _zero_stderr)
if(NOT _zero_status EQUAL 0 OR
   NOT _zero_stdout MATCHES
     "\\[hdg-main\\] status=Success.*regularization=igr" OR
   NOT _zero_stderr STREQUAL "")
  message(FATAL_ERROR
    "periodic zero-source HDG execution failed: ${_zero_status}\\n${_zero_stdout}\\n${_zero_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DG_IGR_VTK_PREFIX_PATH=${_source_output}"
    "${HDG_PERIODIC_EXECUTABLE}" ${_common_arguments}
    --order 1
    --regularization igr
    --time-step-save 0.00001
    --vtk_save
    --hdg-static-source 0.01
  RESULT_VARIABLE _source_status
  OUTPUT_VARIABLE _source_stdout
  ERROR_VARIABLE _source_stderr)
if(NOT _source_status EQUAL 0 OR
   NOT _source_stdout MATCHES
     "\\[hdg-main\\] status=Success.*regularization=igr" OR
   NOT _source_stderr STREQUAL "")
  message(FATAL_ERROR
    "periodic HDG static-source execution failed: ${_source_status}\\n${_source_stdout}\\n${_source_stderr}")
endif()

set(_collection "dgsem_smooth_advection_periodic_hdg_p_1_ref_0")
set(_zero_final
  "${_zero_output}/${_collection}/Cycle000001/proc000000.vtu")
set(_source_final
  "${_source_output}/${_collection}/Cycle000001/proc000000.vtu")
if(NOT EXISTS "${_zero_final}" OR NOT EXISTS "${_source_final}")
  message(FATAL_ERROR "source comparison is missing a final VTU payload")
endif()
file(READ "${_zero_final}" _zero_payload)
file(READ "${_source_final}" _source_payload)
if(_zero_payload STREQUAL _source_payload)
  message(FATAL_ERROR
    "nonzero static source did not change the committed final IGR-augmented HDG field")
endif()

execute_process(
  COMMAND "${HDG_NONPERIODIC_EXECUTABLE}" ${_common_arguments}
    --order 1
    --bc-mode inlet-outlet
    --no-vtk_save
  RESULT_VARIABLE _boundary_status
  OUTPUT_VARIABLE _boundary_stdout
  ERROR_VARIABLE _boundary_stderr)
if(NOT _boundary_status EQUAL 0 OR
   NOT _boundary_stdout MATCHES "\\[hdg-main\\] status=Success" OR
   NOT _boundary_stderr STREQUAL "")
  message(FATAL_ERROR
    "nonperiodic conventional HDG execution failed: ${_boundary_status}\\n${_boundary_stdout}\\n${_boundary_stderr}")
endif()

# IGR is an augmentation of the same physical-boundary Euler lifecycle at
# every supported order.  The selectable plans cover a mixed supersonic
# inlet/outlet/slip boundary, every-attribute supersonic outlet, and
# every-attribute slip wall.  One short step keeps the full p1-p3 matrix fast.
foreach(_boundary_mode IN ITEMS inlet-outlet all-outlet all-slip)
  foreach(_order IN ITEMS 1 2 3)
    execute_process(
      COMMAND "${HDG_NONPERIODIC_EXECUTABLE}" ${_common_arguments}
        --order ${_order}
        --bc-mode ${_boundary_mode}
        --regularization igr
        --no-vtk_save
      RESULT_VARIABLE _igr_status
      OUTPUT_VARIABLE _igr_stdout
      ERROR_VARIABLE _igr_stderr)
    if(NOT _igr_status EQUAL 0 OR
       NOT _igr_stdout MATCHES
         "\\[hdg-main\\] status=Success.*regularization=igr" OR
       NOT _igr_stderr STREQUAL "")
      message(FATAL_ERROR
        "physical-boundary ${_boundary_mode} p${_order} HDG IGR execution failed: ${_igr_status}\\n${_igr_stdout}\\n${_igr_stderr}")
    endif()
  endforeach()
endforeach()
