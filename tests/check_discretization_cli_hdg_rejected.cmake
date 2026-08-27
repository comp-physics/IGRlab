if(NOT DEFINED TEST_EXE OR NOT DEFINED SENTINEL)
  message(FATAL_ERROR "TEST_EXE and SENTINEL are required.")
endif()

function(expect_early_rejection expected_diagnostic)
  file(REMOVE "${SENTINEL}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "DG_IGR_CONFIG_BOUNDARY_SENTINEL=${SENTINEL}"
            "${TEST_EXE}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  set(output "${stdout}\n${stderr}")

  if("${result}" STREQUAL "0")
    message(FATAL_ERROR "Rejected request unexpectedly returned zero.\n${output}")
  endif()
  if(NOT output MATCHES "${expected_diagnostic}")
    message(FATAL_ERROR
      "Expected early rejection diagnostic was not emitted.\n${output}")
  endif()
  if(EXISTS "${SENTINEL}")
    message(FATAL_ERROR "Rejected request reached the post-setup make_ic callback.")
  endif()
endfunction()

expect_early_rejection(
  "Invalid HDG CUDA engine 'invalid'\\. Use 'ordinary' or 'tensor'\\."
  --hdg-cuda-engine invalid)
expect_early_rejection(
  "HDG CUDA engine 'tensor' requires '--discretization hdg --device cuda'\\."
  --discretization hdg --device cpu --hdg-cuda-engine tensor)
expect_early_rejection(
  "HDG CUDA engine 'tensor' requires '--discretization hdg --device cuda'\\."
  --discretization dg --device cuda --hdg-cuda-engine tensor)
expect_early_rejection(
  "Case 'discretization-cli-boundary' does not provide an HDG backend adapter\\."
  --discretization hdg)
expect_early_rejection(
  "Time integrator 'backward-euler' is unavailable for DG/DGSEM"
  --time-integrator backward-euler)
expect_early_rejection(
  "Time integrator 'rk3ssp' is unavailable for HDG"
  --discretization hdg --time-integrator rk3ssp)
expect_early_rejection(
  "Case 'discretization-cli-boundary' does not provide an HDG backend adapter\\."
  --discretization hdg --regularization igr --device cuda)
