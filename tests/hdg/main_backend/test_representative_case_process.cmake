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
  --order 1
  --serial-refine 0
  --parallel-refine 0
  --time-step 0.00001
  --t-final 0.00001
  --no-glvis)

# Exercise the optional source path without changing the source-free default
# trajectory used by the existing periodic case tests.  Distinct final VTK
# payloads prove that the projected source reaches the numerical request; a
# success-only smoke would not detect an accidentally dropped source view.
set(_zero_output "${HDG_REPRESENTATIVE_WORK}/zero-source")
set(_source_output "${HDG_REPRESENTATIVE_WORK}/nonzero-source")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DG_IGR_VTK_PREFIX_PATH=${_zero_output}"
    "${HDG_PERIODIC_EXECUTABLE}" ${_common_arguments}
    --time-step-save 0.00001
    --vtk_save
  RESULT_VARIABLE _zero_status
  OUTPUT_VARIABLE _zero_stdout
  ERROR_VARIABLE _zero_stderr)
if(NOT _zero_status EQUAL 0 OR
   NOT _zero_stdout MATCHES "\\[hdg-main\\] status=Success" OR
   NOT _zero_stderr STREQUAL "")
  message(FATAL_ERROR
    "periodic zero-source HDG execution failed: ${_zero_status}\\n${_zero_stdout}\\n${_zero_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "DG_IGR_VTK_PREFIX_PATH=${_source_output}"
    "${HDG_PERIODIC_EXECUTABLE}" ${_common_arguments}
    --time-step-save 0.00001
    --vtk_save
    --hdg-static-source 0.01
  RESULT_VARIABLE _source_status
  OUTPUT_VARIABLE _source_stdout
  ERROR_VARIABLE _source_stderr)
if(NOT _source_status EQUAL 0 OR
   NOT _source_stdout MATCHES "\\[hdg-main\\] status=Success" OR
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
    "nonzero static source did not change the committed final HDG field")
endif()

execute_process(
  COMMAND "${HDG_NONPERIODIC_EXECUTABLE}" ${_common_arguments}
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

# Physical-boundary IGR is rejected by generic Euler-HDG admission.  The case
# supplies ordinary semantic data only; it does not carry a special rejection.
execute_process(
  COMMAND "${HDG_NONPERIODIC_EXECUTABLE}" ${_common_arguments}
    --bc-mode inlet-outlet
    --regularization igr
    --no-vtk_save
  RESULT_VARIABLE _igr_status
  OUTPUT_VARIABLE _igr_stdout
  ERROR_VARIABLE _igr_stderr)
set(_igr_output "${_igr_stdout}\\n${_igr_stderr}")
if("${_igr_status}" STREQUAL "0" OR
   NOT _igr_output MATCHES "generic CPU session initialization failed")
  message(FATAL_ERROR
    "physical-boundary HDG IGR was not rejected by generic admission: ${_igr_status}\\n${_igr_output}")
endif()
