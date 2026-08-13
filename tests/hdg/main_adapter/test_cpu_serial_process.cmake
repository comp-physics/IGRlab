if(NOT DEFINED HDG_SMOOTH_PERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_OUTPUT)
  message(FATAL_ERROR "adapter executable and output path are required")
endif()

file(REMOVE_RECURSE "${HDG_SMOOTH_PERIODIC_OUTPUT}")
file(REMOVE "${HDG_SMOOTH_PERIODIC_OUTPUT}.checkpoint")

function(expect_early_rejection label expected_text)
  execute_process(
    COMMAND "${HDG_SMOOTH_PERIODIC_EXECUTABLE}"
      --discretization hdg
      --hdg-output-dir "${HDG_SMOOTH_PERIODIC_OUTPUT}"
      ${ARGN}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT error MATCHES "${expected_text}" OR
     EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}" OR
     EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}.checkpoint")
    message(FATAL_ERROR
      "${label} was not rejected before resource setup: ${status};${output};${error}")
  endif()
endfunction()

expect_early_rejection(cuda "device selection is unsupported" --device cuda)
expect_early_rejection(legacy_dg "DG/DGSEM-only legacy selector"
                       --dg-backend dg)
expect_early_rejection(restart "unsupported HDG adapter option" --restart-vtk state.pvd)
expect_early_rejection(generic_mesh "unsupported HDG adapter option" --mesh mesh.mesh)

execute_process(
  COMMAND "${HDG_SMOOTH_PERIODIC_EXECUTABLE}"
    --discretization hdg
    --hdg-output-dir "${HDG_SMOOTH_PERIODIC_OUTPUT}"
  RESULT_VARIABLE status
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT status EQUAL 0 OR
   NOT output MATCHES "status=Success.*step=20" OR
   NOT error STREQUAL "")
  message(FATAL_ERROR "CPU serial adapter run failed: ${status};${output};${error}")
endif()

foreach(step IN ITEMS 000000 000010 000020)
  if(NOT EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}/state-step-${step}.vtk")
    message(FATAL_ERROR "missing VTK output for step ${step}")
  endif()
endforeach()
