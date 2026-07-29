if(NOT DEFINED HDG_RUNTIME_EXECUTABLE OR
   NOT DEFINED HDG_RUNTIME_TEST_OUTPUT)
  message(FATAL_ERROR "executable and test output path are required")
endif()

file(REMOVE_RECURSE "${HDG_RUNTIME_TEST_OUTPUT}")
execute_process(
  COMMAND "${HDG_RUNTIME_EXECUTABLE}" --help
  RESULT_VARIABLE help_status
  OUTPUT_VARIABLE help_output
  ERROR_VARIABLE help_error)
if(NOT help_status EQUAL 0 OR
   NOT help_output MATCHES "--validation-capture" OR
   NOT help_output MATCHES "AfterRecoveryBeforeCommit" OR
   NOT help_error STREQUAL "")
  message(FATAL_ERROR "HelpOnly contract failed: ${help_status};${help_error}")
endif()

set(common_arguments
  --card exp.hdg.euler.conventional.smooth_persistent.w8a.v1
  --case conventional-smooth-persistent-v1
  --mesh-family periodic-cartesian-quad --mesh-n 3 --order 1
  --dt 0.05 --steps 3 --final-time 0.15
  --backend cpu --device-id none
  --output-dir "${HDG_RUNTIME_TEST_OUTPUT}"
  --output-steps 0,1,3 --evidence-dir none --validation-capture none
  --checkpoint-out "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint"
  --checkpoint-step 1 --stop-after-checkpoint false
  --restart-from none --max-retries 0 --inject-failure off
  --seed 0x573841534d4f4b31)

execute_process(
  COMMAND "${HDG_RUNTIME_EXECUTABLE}" --space h1 ${common_arguments}
  RESULT_VARIABLE h1_status
  OUTPUT_VARIABLE h1_output
  ERROR_VARIABLE h1_error)
if(NOT h1_status EQUAL 65 OR
   NOT h1_error MATCHES "unsupported_h1" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  message(FATAL_ERROR "side-effect-free H1 contract failed: ${h1_status};${h1_output};${h1_error}")
endif()

set(invalid_arguments ${common_arguments})
list(REMOVE_ITEM invalid_arguments --mesh-n 3)
list(APPEND invalid_arguments --mesh-n 6)
execute_process(
  COMMAND "${HDG_RUNTIME_EXECUTABLE}" --space l2 ${invalid_arguments}
  RESULT_VARIABLE invalid_status
  OUTPUT_VARIABLE invalid_output
  ERROR_VARIABLE invalid_error)
if(NOT invalid_status EQUAL 64 OR
   NOT invalid_error MATCHES "invalid_polynomial_mesh_pair" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  message(FATAL_ERROR "invalid tuple no-action contract failed: ${invalid_status};${invalid_output};${invalid_error}")
endif()

file(REMOVE_RECURSE "${HDG_RUNTIME_TEST_OUTPUT}")
file(REMOVE "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
