if(NOT DEFINED HDG_SMOOTH_PERIODIC_EXECUTABLE OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_OUTPUT OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_MPIEXEC OR
   NOT DEFINED HDG_SMOOTH_PERIODIC_NUMPROC_FLAG)
  message(FATAL_ERROR "adapter executable, output, and MPI launcher are required")
endif()

file(REMOVE_RECURSE "${HDG_SMOOTH_PERIODIC_OUTPUT}")
file(REMOVE "${HDG_SMOOTH_PERIODIC_OUTPUT}.checkpoint")
execute_process(
  COMMAND "${HDG_SMOOTH_PERIODIC_MPIEXEC}"
    "${HDG_SMOOTH_PERIODIC_NUMPROC_FLAG}" 2
    ${HDG_SMOOTH_PERIODIC_MPIEXEC_PREFLAGS}
    "${HDG_SMOOTH_PERIODIC_EXECUTABLE}"
      --discretization hdg
      --hdg-output-dir "${HDG_SMOOTH_PERIODIC_OUTPUT}"
      --device cuda
    ${HDG_SMOOTH_PERIODIC_MPIEXEC_POSTFLAGS}
  RESULT_VARIABLE status
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT status EQUAL 0 OR
   NOT output MATCHES "status=Success.*step=20" OR
   NOT error STREQUAL "")
  message(FATAL_ERROR "CUDA rank-two adapter run failed: ${status};${output};${error}")
endif()

file(GLOB vtk_outputs "${HDG_SMOOTH_PERIODIC_OUTPUT}/*.vtk")
list(LENGTH vtk_outputs vtk_output_count)
if(NOT vtk_output_count EQUAL 3)
  message(FATAL_ERROR
    "CUDA rank-two adapter produced ${vtk_output_count} VTK files, expected 3")
endif()
foreach(step IN ITEMS 000000 000010 000020)
  if(NOT EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}/state-step-${step}.vtk")
    message(FATAL_ERROR "missing VTK output for step ${step}")
  endif()
endforeach()
if(NOT EXISTS "${HDG_SMOOTH_PERIODIC_OUTPUT}.checkpoint")
  message(FATAL_ERROR "missing checkpoint alias")
endif()
