if(NOT DEFINED HDG_RUNTIME_EXECUTABLE OR
   NOT DEFINED HDG_RUNTIME_TEST_OUTPUT)
  message(FATAL_ERROR "executable and test output path are required")
endif()

if(NOT DEFINED HDG_RUNTIME_RANKS)
  set(HDG_RUNTIME_RANKS 1)
endif()

function(run_runtime prefix)
  if(HDG_RUNTIME_RANKS EQUAL 1)
    execute_process(
      COMMAND "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  elseif(HDG_RUNTIME_RANKS EQUAL 2)
    if(NOT DEFINED HDG_RUNTIME_MPIEXEC OR
       NOT DEFINED HDG_RUNTIME_NUMPROC_FLAG)
      message(FATAL_ERROR "rank-2 runtime launcher is required")
    endif()
    execute_process(
      COMMAND "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
        ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
        ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  else()
    message(FATAL_ERROR "only rank-1 and rank-2 CLI tests are admitted")
  endif()
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(require_regular_file path label)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
    message(FATAL_ERROR "missing ${label}: ${path}")
  endif()
endfunction()

if(HDG_RUNTIME_FUNCTIONAL)
  if(NOT DEFINED HDG_RUNTIME_BACKEND)
    set(HDG_RUNTIME_BACKEND cpu)
  endif()
  if(HDG_RUNTIME_BACKEND STREQUAL "cuda")
    set(runtime_device 0)
  else()
    set(runtime_device none)
  endif()
  math(EXPR injection_rank "${HDG_RUNTIME_RANKS} - 1")
  set(reference_directory "${HDG_RUNTIME_TEST_OUTPUT}-reference")
  # The frozen same-tuple restart contract binds a split trajectory to one
  # durable output identity. The reader resumes publication in that directory.
  set(writer_directory "${HDG_RUNTIME_TEST_OUTPUT}-split")
  set(reader_directory "${writer_directory}")
  set(reference_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}-reference.checkpoint")
  set(split_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  file(REMOVE_RECURSE "${reference_directory}" "${writer_directory}"
                      "${reader_directory}")
  file(REMOVE "${reference_checkpoint}" "${split_checkpoint}")

  set(functional_common
    --card exp.hdg.euler.conventional.smooth_persistent.w8a.v1
    --case conventional-smooth-persistent-v1
    --space l2 --mesh-family periodic-cartesian-quad --mesh-n 6 --order 2
    --dt 0.05 --steps 20 --final-time 1.0
    --backend "${HDG_RUNTIME_BACKEND}" --device-id "${runtime_device}"
    --output-steps 0,10,20 --seed 0x57384146554e4331)

  run_runtime(reference ${functional_common}
    --output-dir "${reference_directory}"
    --checkpoint-out "${reference_checkpoint}" --checkpoint-step 10
    --stop-after-checkpoint false --restart-from none
    --max-retries 0 --inject-failure off)
  if(NOT reference_status EQUAL 0 OR
     NOT reference_output MATCHES "status=Success.*step=20.*retries=0" OR
     NOT reference_error STREQUAL "")
    message(FATAL_ERROR
      "functional uninterrupted run failed: ${reference_status};${reference_output};${reference_error}")
  endif()

  run_runtime(writer ${functional_common}
    --output-dir "${writer_directory}"
    --checkpoint-out "${split_checkpoint}" --checkpoint-step 10
    --stop-after-checkpoint true --restart-from none
    --max-retries 0 --inject-failure off)
  if(NOT writer_status EQUAL 0 OR
     NOT writer_output MATCHES "status=Success.*step=10.*retries=0" OR
     NOT writer_error STREQUAL "")
    message(FATAL_ERROR
      "functional split writer failed: ${writer_status};${writer_output};${writer_error}")
  endif()

  run_runtime(reader ${functional_common}
    --output-dir "${reader_directory}"
    --checkpoint-out none --checkpoint-step none
    --stop-after-checkpoint false --restart-from "${split_checkpoint}"
    --max-retries 1
    --inject-failure "AfterRecoveryBeforeCommit:14:${injection_rank}")
  if(NOT reader_status EQUAL 0 OR
     NOT reader_output MATCHES "status=Success.*step=20.*retries=1" OR
     NOT reader_error STREQUAL "")
    message(FATAL_ERROR
      "functional split reader/retry failed: ${reader_status};${reader_output};${reader_error}")
  endif()

  set(functional_publications
      "${reference_directory}/state-step-000000.vtk"
      "${reference_directory}/state-step-000010.vtk"
      "${reference_directory}/state-step-000020.vtk"
      "${reference_directory}/run-metadata.json"
      "${reference_checkpoint}"
      "${writer_directory}/state-step-000000.vtk"
      "${writer_directory}/state-step-000010.vtk"
      "${writer_directory}/run-metadata-checkpoint.json"
      "${split_checkpoint}"
      "${reader_directory}/state-step-000020.vtk"
      "${reader_directory}/run-metadata.json")
  foreach(path IN LISTS functional_publications)
    require_regular_file("${path}" "functional publication")
  endforeach()

  file(SHA256 "${reference_directory}/state-step-000020.vtk" reference_vtk_hash)
  file(SHA256 "${reader_directory}/state-step-000020.vtk" reader_vtk_hash)
  file(READ "${reference_directory}/run-metadata.json" reference_metadata)
  file(READ "${reader_directory}/run-metadata.json" reader_metadata)
  set(state_parity FALSE)
  if(HDG_RUNTIME_BACKEND STREQUAL "cuda")
    string(REGEX MATCH
      "\"final_interior_sha256\": \"([0-9a-f]+)\""
      reference_interior_match "${reference_metadata}")
    set(reference_interior_hash "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "\"final_trace_sha256\": \"([0-9a-f]+)\""
      reference_trace_match "${reference_metadata}")
    set(reference_trace_hash "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "\"final_interior_sha256\": \"([0-9a-f]+)\""
      reader_interior_match "${reader_metadata}")
    set(reader_interior_hash "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "\"final_trace_sha256\": \"([0-9a-f]+)\""
      reader_trace_match "${reader_metadata}")
    set(reader_trace_hash "${CMAKE_MATCH_1}")
    if(NOT reference_interior_hash STREQUAL "" AND
       reference_interior_hash STREQUAL reader_interior_hash AND
       NOT reference_trace_hash STREQUAL "" AND
       reference_trace_hash STREQUAL reader_trace_hash)
      set(state_parity TRUE)
    endif()
  else()
    string(REGEX MATCH
      "\"final_state_sha256\": \"([0-9a-f]+)\""
      reference_hash_match "${reference_metadata}")
    set(reference_state_hash "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "\"final_state_sha256\": \"([0-9a-f]+)\""
      reader_hash_match "${reader_metadata}")
    set(reader_state_hash "${CMAKE_MATCH_1}")
    if(NOT reference_state_hash STREQUAL "" AND
       reference_state_hash STREQUAL reader_state_hash)
      set(state_parity TRUE)
    endif()
  endif()
  if(reference_vtk_hash STREQUAL "" OR reader_vtk_hash STREQUAL "" OR
     NOT state_parity OR NOT reader_metadata MATCHES "\"status\": \"Success\"" OR
     NOT reader_metadata MATCHES "\"retries\": 1")
    message(FATAL_ERROR "functional restart/retry parity contract failed")
  endif()
  return()
endif()

file(REMOVE_RECURSE "${HDG_RUNTIME_TEST_OUTPUT}")
file(REMOVE "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
if(NOT DEFINED HDG_RUNTIME_BACKEND)
  set(HDG_RUNTIME_BACKEND cpu)
endif()
if(HDG_RUNTIME_BACKEND STREQUAL "cuda")
  set(runtime_device 0)
else()
  set(runtime_device none)
endif()
execute_process(
  COMMAND "${HDG_RUNTIME_EXECUTABLE}" --help
  RESULT_VARIABLE help_status
  OUTPUT_VARIABLE help_output
  ERROR_VARIABLE help_error)
if(NOT help_status EQUAL 0 OR
   help_output MATCHES "--validation-capture" OR
   help_output MATCHES "--evidence-dir" OR
   NOT help_output MATCHES "AfterRecoveryBeforeCommit" OR
   NOT help_error STREQUAL "")
  message(FATAL_ERROR "HelpOnly contract failed: ${help_status};${help_error}")
endif()

set(common_arguments
  --card exp.hdg.euler.conventional.smooth_persistent.w8a.v1
  --case conventional-smooth-persistent-v1
  --mesh-family periodic-cartesian-quad --mesh-n 3 --order 1
  --dt 0.05 --steps 3 --final-time 0.15
  --backend "${HDG_RUNTIME_BACKEND}" --device-id "${runtime_device}"
  --output-dir "${HDG_RUNTIME_TEST_OUTPUT}"
  --output-steps 0,1,3
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
list(FIND invalid_arguments --mesh-n mesh_n_option_index)
if(mesh_n_option_index LESS 0)
  message(FATAL_ERROR "common argument list has no --mesh-n option")
endif()
math(EXPR mesh_n_value_index "${mesh_n_option_index} + 1")
list(REMOVE_AT invalid_arguments ${mesh_n_value_index})
list(INSERT invalid_arguments ${mesh_n_value_index} 6)
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

run_runtime(smoke "--space" "l2" ${common_arguments})
if(NOT smoke_status EQUAL 0 OR
   NOT smoke_output MATCHES "status=Success.*step=3.*retries=0" OR
   NOT smoke_error STREQUAL "")
  message(FATAL_ERROR
    "three-step CLI smoke failed: ${smoke_status};${smoke_output};${smoke_error}")
endif()
foreach(path IN ITEMS
    "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000000.vtk"
    "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000001.vtk"
    "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000003.vtk"
    "${HDG_RUNTIME_TEST_OUTPUT}/run-metadata.json"
    "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  require_regular_file("${path}" "smoke publication")
endforeach()
file(READ "${HDG_RUNTIME_TEST_OUTPUT}/run-metadata.json" smoke_metadata)
if(NOT smoke_metadata MATCHES "\"status\": \"Success\"" OR
   NOT smoke_metadata MATCHES "\"committed_step\": 3")
  message(FATAL_ERROR "three-step CLI smoke metadata is not truthful")
endif()
