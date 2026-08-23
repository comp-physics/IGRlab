if(NOT DEFINED HDG_MAIN_EXECUTABLE OR
   NOT DEFINED HDG_MAIN_MESH OR
   NOT DEFINED HDG_MAIN_RANKS OR
   NOT DEFINED HDG_MAIN_MPIEXEC OR
   NOT DEFINED HDG_MAIN_NUMPROC_FLAG)
  message(FATAL_ERROR
    "CUDA HDG executable, mesh, ranks, and MPI launcher are required.")
endif()

set(_base_arguments
  --discretization hdg
  --device cuda
  --regularization igr
  --serial-refine 0
  --parallel-refine 0
  --mesh "${HDG_MAIN_MESH}"
  --time-step 0.00001
  --t-final 0.00001
  --no-vtk_save
  --no-glvis)

function(run_cuda_igr_success order expected_alpha expected_beta)
  set(_arguments ${_base_arguments} --order ${order} ${ARGN})
  if(HDG_MAIN_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}"
        "${HDG_MAIN_RANKS}"
        ${HDG_MAIN_MPIEXEC_PREFLAGS}
        "${HDG_MAIN_EXECUTABLE}" ${_arguments}
        ${HDG_MAIN_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  if(NOT _status EQUAL 0 OR
     NOT _stdout MATCHES
       "\\[hdg-main\\] status=Success.*regularization=igr.*alpha=${expected_alpha}.*beta=${expected_beta}" OR
     NOT _stderr STREQUAL "")
    message(FATAL_ERROR
      "CUDA HDG p${order} IGR execution failed: ${_status}\n${_stdout}\n${_stderr}")
  endif()
endfunction()

function(expect_cuda_igr_rejection expected_diagnostic)
  set(_arguments ${_base_arguments} ${ARGN})
  if(HDG_MAIN_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_MAIN_EXECUTABLE}" ${_arguments}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  else()
    execute_process(
      COMMAND "${HDG_MAIN_MPIEXEC}" "${HDG_MAIN_NUMPROC_FLAG}"
        "${HDG_MAIN_RANKS}"
        ${HDG_MAIN_MPIEXEC_PREFLAGS}
        "${HDG_MAIN_EXECUTABLE}" ${_arguments}
        ${HDG_MAIN_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE _status
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
  endif()
  set(_output "${_stdout}\n${_stderr}")
  if("${_status}" STREQUAL "0" OR NOT _output MATCHES "${expected_diagnostic}")
    message(FATAL_ERROR
      "CUDA HDG IGR rejection failed: ${_status}\n${_output}")
  endif()
endfunction()

# This is the production main-backend route, not the frozen SC25/P1 oracle.
# All supported orders share the same generic IC/trace projection, source,
# time adapter, nonlinear lifecycle, condensation, and optional IGR fields.
run_cuda_igr_success(1 1 1)
run_cuda_igr_success(1 0.2 0.8
  --hdg-igr-alpha 0.2 --hdg-igr-beta 0.8)
run_cuda_igr_success(1 1 1 --hdg-static-source 0.001)
run_cuda_igr_success(2 1 1)
run_cuda_igr_success(3 1 1)
expect_cuda_igr_rejection(
  "HDG IGR alpha and beta must be finite and positive"
  --order 1 --hdg-igr-alpha 0)
expect_cuda_igr_rejection(
  "HDG IGR alpha and beta must be finite and positive"
  --order 1 --hdg-igr-beta 0)
