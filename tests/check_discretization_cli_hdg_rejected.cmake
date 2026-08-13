if(NOT DEFINED TEST_EXE OR NOT DEFINED SENTINEL)
  message(FATAL_ERROR "TEST_EXE and SENTINEL are required.")
endif()

file(REMOVE "${SENTINEL}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
          "DG_IGR_CONFIG_BOUNDARY_SENTINEL=${SENTINEL}"
          "${TEST_EXE}" --discretization hdg
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
set(output "${stdout}\n${stderr}")

if("${result}" STREQUAL "0")
  message(FATAL_ERROR "--discretization hdg unexpectedly returned zero.\n${output}")
endif()
if(NOT output MATCHES "Case 'discretization-cli-boundary' does not provide an HDG backend adapter\\.")
  message(FATAL_ERROR "Expected case-level HDG diagnostic was not emitted.\n${output}")
endif()
if(EXISTS "${SENTINEL}")
  message(FATAL_ERROR "HDG rejection reached the post-setup make_ic callback.")
endif()
