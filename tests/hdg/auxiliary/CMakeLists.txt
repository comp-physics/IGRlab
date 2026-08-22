if(NOT DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT)
  return()
endif()
if(NOT TARGET dg_igr_hdg_auxiliary_cpu)
  message(FATAL_ERROR
          "The HDG auxiliary test requires dg_igr_hdg_auxiliary_cpu.")
endif()
if(NOT MFEM_USE_MPI OR NOT TARGET MPI::MPI_CXX)
  message(FATAL_ERROR
          "The HDG auxiliary test requires the MPI/HYPRE CPU profile.")
endif()
if(NOT MPIEXEC_EXECUTABLE OR NOT MPIEXEC_NUMPROC_FLAG)
  message(FATAL_ERROR
          "The HDG auxiliary rank-two test requires an MPI launcher.")
endif()
add_executable(dg_igr_hdg_auxiliary_cpu_test
  test_screened_poisson_cpu.cpp
  screened_poisson_h1_reference.cpp)
if(MFEM_USE_CUDA)
  set_source_files_properties(screened_poisson_h1_reference.cpp
    PROPERTIES LANGUAGE CUDA)
  set_target_properties(dg_igr_hdg_auxiliary_cpu_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF)
endif()
target_link_libraries(dg_igr_hdg_auxiliary_cpu_test PRIVATE
  dg_igr_hdg_auxiliary_cpu
  MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_auxiliary_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror;-pedantic-errors>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
set_target_properties(dg_igr_hdg_auxiliary_cpu_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

set(hdg_auxiliary_isolated_driver
    "${CMAKE_CURRENT_BINARY_DIR}/run_hdg_auxiliary_isolated.cmake")
file(WRITE "${hdg_auxiliary_isolated_driver}" [=[
if(NOT DEFINED HDG_AUX_EXECUTABLE OR HDG_AUX_EXECUTABLE STREQUAL "")
  message(FATAL_ERROR "HDG auxiliary executable was not provided")
endif()

function(run_hdg_auxiliary_child mode marker)
  execute_process(
    COMMAND ${HDG_AUX_LAUNCHER} "${HDG_AUX_EXECUTABLE}"
            ${HDG_AUX_POSTFLAGS} "${mode}"
    RESULT_VARIABLE child_result
    OUTPUT_VARIABLE child_output
    ERROR_VARIABLE child_error)
  if(NOT child_output STREQUAL "")
    message("${child_output}")
  endif()
  if(NOT child_error STREQUAL "")
    message("${child_error}")
  endif()
  string(FIND "${child_output}" "${marker}" marker_offset)
  if(NOT child_result EQUAL 0 OR marker_offset EQUAL -1)
    message(FATAL_ERROR
            "HDG auxiliary child ${mode} failed: exit=${child_result}")
  endif()
endfunction()

# Stateful HYPRE failures run to process termination.  The passing census then
# starts in a third process with a clean process-global HYPRE error state.
run_hdg_auxiliary_child(
  "--isolated-negative-nonzero"
  "PASS HDG-AUX-CPU-03 isolated nonzero nonconvergence rollback")
run_hdg_auxiliary_child(
  "--isolated-negative-zero"
  "PASS HDG-AUX-CPU-03 isolated zero-RHS nonconvergence rollback")
run_hdg_auxiliary_child(
  "--positive-with-isolated-negatives"
  "PASS HDG-AUX-CPU-03 source/runtime census")
]=])

add_test(NAME dg_igr_hdg_auxiliary_cpu_serial
  COMMAND ${CMAKE_COMMAND}
          "-DHDG_AUX_EXECUTABLE:FILEPATH=$<TARGET_FILE:dg_igr_hdg_auxiliary_cpu_test>"
          "-DHDG_AUX_LAUNCHER:STRING="
          "-DHDG_AUX_POSTFLAGS:STRING="
          -P "${hdg_auxiliary_isolated_driver}")
set_tests_properties(dg_igr_hdg_auxiliary_cpu_serial PROPERTIES
  LABELS "hdg;auxiliary;cpu;experimental;serial"
  TIMEOUT 1800)
add_test(NAME dg_igr_hdg_auxiliary_cpu_mpi_2
  COMMAND ${CMAKE_COMMAND}
          "-DHDG_AUX_EXECUTABLE:FILEPATH=$<TARGET_FILE:dg_igr_hdg_auxiliary_cpu_test>"
          "-DHDG_AUX_LAUNCHER:STRING=${MPIEXEC_EXECUTABLE};${MPIEXEC_NUMPROC_FLAG};2;${MPIEXEC_PREFLAGS}"
          "-DHDG_AUX_POSTFLAGS:STRING=${MPIEXEC_POSTFLAGS}"
          -P "${hdg_auxiliary_isolated_driver}")
set_tests_properties(dg_igr_hdg_auxiliary_cpu_mpi_2 PROPERTIES
  LABELS "hdg;auxiliary;cpu;experimental;mpi"
  PROCESSORS 2
  SKIP_RETURN_CODE 75
  TIMEOUT 1800)
