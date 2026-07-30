if(NOT DEFINED HDG_RUNTIME_EXECUTABLE OR
   NOT DEFINED HDG_RUNTIME_TEST_OUTPUT)
  message(FATAL_ERROR "executable and test output path are required")
endif()

if(NOT DEFINED HDG_RUNTIME_RANKS)
  set(HDG_RUNTIME_RANKS 1)
endif()

if(HDG_RUNTIME_RANKS EQUAL 1 AND
   NOT HDG_RUNTIME_FUNCTIONAL AND
   DEFINED HDG_RUNTIME_BUILD_DRIVER AND
   NOT "$ENV{DG_IGR_TEST_REUSE_MFEM_ROOT}" STREQUAL "" AND
   NOT "$ENV{DG_IGR_TEST_REUSE_SPACK_ENV}" STREQUAL "" AND
   NOT "$ENV{DG_IGR_TEST_REUSE_MODE}" STREQUAL "")
  if("$ENV{DG_IGR_TEST_REUSE_MODE}" STREQUAL "gpu")
    set(reuse_mode_args --gpu --cuda-arch 120)
  elseif("$ENV{DG_IGR_TEST_REUSE_MODE}" STREQUAL "cpu")
    set(reuse_mode_args --cpu-only)
  else()
    message(FATAL_ERROR "invalid reuse-fixture mode")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      DG_IGR_RUN_REUSE_FIXTURES=1
      "DG_IGR_SPACK_ENV=$ENV{DG_IGR_TEST_REUSE_SPACK_ENV}"
      bash "${HDG_RUNTIME_BUILD_DRIVER}" ${reuse_mode_args}
      --reuse-mfem-root "$ENV{DG_IGR_TEST_REUSE_MFEM_ROOT}"
      --no-deps --no-tests --print-plan
    RESULT_VARIABLE reuse_fixture_status
    OUTPUT_VARIABLE reuse_fixture_output
    ERROR_VARIABLE reuse_fixture_error)
  if(NOT reuse_fixture_status EQUAL 0 OR
     NOT reuse_fixture_output MATCHES
         "reuse-fixtures.*compatibility cases passed" OR
     NOT reuse_fixture_output MATCHES
         "mfem_reuse_state=validated-read-only" OR
     NOT reuse_fixture_error STREQUAL "")
    message(FATAL_ERROR
      "MFEM/Spack reuse compatibility fixtures failed: ${reuse_fixture_status};${reuse_fixture_output};${reuse_fixture_error}")
  endif()
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

function(run_private_scientific_gate prefix)
  if(HDG_RUNTIME_RANKS EQUAL 1)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        DG_IGR_HDG_TEST_SCIENTIFIC_GATE_FAILURE=1
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  else()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        DG_IGR_HDG_TEST_SCIENTIFIC_GATE_FAILURE=1
        "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
        ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
        ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  endif()
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(run_private_final_sync_failure prefix)
  if(HDG_RUNTIME_RANKS EQUAL 1)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        DG_IGR_HDG_TEST_FINAL_DIRECTORY_SYNC_FAILURE=1
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  else()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        DG_IGR_HDG_TEST_FINAL_DIRECTORY_SYNC_FAILURE=1
        "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
        ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
        ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  endif()
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(run_private_post_publication_failure prefix fault)
  if(HDG_RUNTIME_RANKS EQUAL 1)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        "DG_IGR_HDG_TEST_POST_PUBLICATION_FAILURE=${fault}"
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  else()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        "DG_IGR_HDG_TEST_POST_PUBLICATION_FAILURE=${fault}"
        "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
        ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
        ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  endif()
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(run_private_post_publication_failure_no_retry prefix fault)
  if(HDG_RUNTIME_RANKS EQUAL 1)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        "DG_IGR_HDG_TEST_POST_PUBLICATION_FAILURE=${fault}"
        DG_IGR_HDG_TEST_POST_PUBLICATION_RETRY=0
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  else()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        "DG_IGR_HDG_TEST_POST_PUBLICATION_FAILURE=${fault}"
        DG_IGR_HDG_TEST_POST_PUBLICATION_RETRY=0
        "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
        ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
        "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
        ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
      RESULT_VARIABLE status
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error)
  endif()
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(replace_cli_option output input option value)
  set(arguments ${${input}})
  list(FIND arguments "${option}" option_index)
  if(option_index LESS 0)
    message(FATAL_ERROR "CLI option is unavailable: ${option}")
  endif()
  math(EXPR value_index "${option_index} + 1")
  list(REMOVE_AT arguments ${value_index})
  list(INSERT arguments ${value_index} "${value}")
  set(${output} ${arguments} PARENT_SCOPE)
endfunction()

function(require_regular_file path label)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
    message(FATAL_ERROR "missing ${label}: ${path}")
  endif()
endfunction()

function(validate_physical_vtk path mesh_n label)
  file(READ "${path}" vtk)
  file(STRINGS "${path}" vtk_lines)
  math(EXPR elements "${mesh_n} * ${mesh_n}")
  math(EXPR points "4 * ${elements}")
  string(FIND "${vtk}" "0 0 0" origin_position)
  string(FIND "${vtk}" "1 1 0" corner_position)
  if(NOT vtk MATCHES "DATASET UNSTRUCTURED_GRID" OR
     NOT vtk MATCHES "POINTS ${points} double" OR
     NOT vtk MATCHES "CELLS ${elements} " OR
     NOT vtk MATCHES "SCALARS conservative0 double 1" OR
     NOT vtk MATCHES "SCALARS conservative1 double 1" OR
     NOT vtk MATCHES "SCALARS conservative2 double 1" OR
     NOT vtk MATCHES "SCALARS conservative3 double 1" OR
     origin_position LESS 0 OR corner_position LESS 0)
    message(FATAL_ERROR "${label} is not a physical four-component FE VTK")
  endif()
  list(FIND vtk_lines "SCALARS conservative0 double 1" density_header)
  list(FIND vtk_lines "SCALARS conservative3 double 1" energy_header)
  math(EXPR density_index "${density_header} + 2")
  math(EXPR energy_index "${energy_header} + 2")
  list(GET vtk_lines ${density_index} density)
  list(GET vtk_lines ${energy_index} energy)
  if(density STREQUAL "" OR energy STREQUAL "" OR
     density LESS_EQUAL 0.0 OR energy LESS_EQUAL 0.0)
    message(FATAL_ERROR "${label} has invalid selected conservative values")
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
  validate_physical_vtk(
    "${reference_directory}/state-step-000000.vtk" 6
    "functional initial generation")
  validate_physical_vtk(
    "${reference_directory}/state-step-000010.vtk" 6
    "functional intermediate generation")
  validate_physical_vtk(
    "${reference_directory}/state-step-000020.vtk" 6
    "functional final generation")

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
  if(HDG_RUNTIME_BACKEND STREQUAL "cuda")
    if(NOT DEFINED HDG_RUNTIME_SCIENTIFIC_TEST)
      message(FATAL_ERROR "CUDA field parity comparator is unavailable")
    endif()
    set(cpu_directory "${HDG_RUNTIME_TEST_OUTPUT}-cpu-parity")
    set(cpu_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}-cpu-parity.checkpoint")
    file(REMOVE_RECURSE "${cpu_directory}")
    file(REMOVE "${cpu_checkpoint}")
    set(cpu_functional_common ${functional_common})
    list(FIND cpu_functional_common --backend backend_option_index)
    list(FIND cpu_functional_common --device-id device_option_index)
    math(EXPR backend_value_index "${backend_option_index} + 1")
    math(EXPR device_value_index "${device_option_index} + 1")
    list(REMOVE_AT cpu_functional_common ${backend_value_index})
    list(INSERT cpu_functional_common ${backend_value_index} cpu)
    list(REMOVE_AT cpu_functional_common ${device_value_index})
    list(INSERT cpu_functional_common ${device_value_index} none)
    run_runtime(cpu_reference ${cpu_functional_common}
      --output-dir "${cpu_directory}"
      --checkpoint-out "${cpu_checkpoint}" --checkpoint-step 10
      --stop-after-checkpoint false --restart-from none
      --max-retries 0 --inject-failure off)
    if(NOT cpu_reference_status EQUAL 0 OR
       NOT cpu_reference_output MATCHES "status=Success.*step=20" OR
       NOT cpu_reference_error STREQUAL "")
      message(FATAL_ERROR
        "CPU parity trajectory failed: ${cpu_reference_status};${cpu_reference_output};${cpu_reference_error}")
    endif()
    foreach(step IN ITEMS 000000 000010 000020)
      execute_process(
        COMMAND "${HDG_RUNTIME_SCIENTIFIC_TEST}" --compare-vtk
          "${cpu_directory}/state-step-${step}.vtk"
          "${reference_directory}/state-step-${step}.vtk" 2e-8
        RESULT_VARIABLE compare_status
        OUTPUT_VARIABLE compare_output
        ERROR_VARIABLE compare_error)
      if(NOT compare_status EQUAL 0 OR NOT compare_error STREQUAL "")
        message(FATAL_ERROR
          "CPU-CUDA globally ordered field parity failed at ${step}: ${compare_output};${compare_error}")
      endif()
    endforeach()
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

run_private_scientific_gate(scientific "--space" "l2" ${common_arguments})
if(NOT scientific_status EQUAL 71 OR
   NOT scientific_error MATCHES "reason=true_residual_gate")
  message(FATAL_ERROR
    "private scientific gate did not produce exit 71: ${scientific_status};${scientific_output};${scientific_error}")
endif()

file(REMOVE_RECURSE "${HDG_RUNTIME_TEST_OUTPUT}")
file(REMOVE "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")

run_private_final_sync_failure(
  final_sync "--space" "l2" ${common_arguments})
if(NOT final_sync_status EQUAL 70 OR
   NOT final_sync_error MATCHES
       "status=RuntimeFailure.*reason=backend_failure.*step=0" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}/.dg-igr-current" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000000.vtk" OR
   EXISTS "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  message(FATAL_ERROR
    "final directory-sync failure committed logical state: ${final_sync_status};${final_sync_output};${final_sync_error}")
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
validate_physical_vtk(
  "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000000.vtk" 3
  "smoke initial generation")
validate_physical_vtk(
  "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000001.vtk" 3
  "smoke intermediate generation")
validate_physical_vtk(
  "${HDG_RUNTIME_TEST_OUTPUT}/state-step-000003.vtk" 3
  "smoke final generation")
file(READ "${HDG_RUNTIME_TEST_OUTPUT}/run-metadata.json" smoke_metadata)
if(NOT smoke_metadata MATCHES "\"status\": \"Success\"" OR
   NOT smoke_metadata MATCHES "\"committed_step\": 3")
  message(FATAL_ERROR "three-step CLI smoke metadata is not truthful")
endif()

# A restart is authorized by the current generation marker and its complete
# manifest, not merely by bytes that happen to decode as a checkpoint.
math(EXPR required_reader_source "${HDG_RUNTIME_RANKS} - 1")
set(manifest_failure_cases corrupt malformed missing extra)
foreach(case IN LISTS manifest_failure_cases)
  set(case_output "${HDG_RUNTIME_TEST_OUTPUT}-manifest-${case}")
  set(case_checkpoint "${case_output}.checkpoint")
  file(REMOVE_RECURSE "${case_output}")
  file(REMOVE "${case_checkpoint}")
  replace_cli_option(case_writer common_arguments --output-dir
                     "${case_output}")
  replace_cli_option(case_writer case_writer --checkpoint-out
                     "${case_checkpoint}")
  replace_cli_option(case_writer case_writer --stop-after-checkpoint true)
  run_runtime(case_writer_run "--space" "l2" ${case_writer})
  if(NOT case_writer_run_status EQUAL 0 OR
     NOT case_writer_run_output MATCHES
         "status=Success.*step=1.*retries=0" OR
     NOT case_writer_run_error STREQUAL "")
    message(FATAL_ERROR
      "manifest-authority writer failed for ${case}: ${case_writer_run_status};${case_writer_run_output};${case_writer_run_error}")
  endif()
  file(READ_SYMLINK "${case_output}/.dg-igr-current" current_target)
  set(case_generation "${case_output}/${current_target}")
  if(case STREQUAL "corrupt")
    file(APPEND "${case_generation}/checkpoint.v2" "corrupt")
  elseif(case STREQUAL "malformed")
    file(WRITE "${case_generation}/manifest.v1" "not-a-generation-manifest\n")
  elseif(case STREQUAL "missing")
    file(REMOVE "${case_generation}/manifest.v1")
  elseif(case STREQUAL "extra")
    file(WRITE "${case_generation}/unlisted.bin" "extra")
  endif()
  replace_cli_option(case_reader case_writer --stop-after-checkpoint false)
  replace_cli_option(case_reader case_reader --checkpoint-out none)
  replace_cli_option(case_reader case_reader --checkpoint-step none)
  replace_cli_option(case_reader case_reader --restart-from
                     "${case_checkpoint}")
  replace_cli_option(case_reader case_reader --max-retries 1)
  replace_cli_option(case_reader case_reader --inject-failure
                     "AfterRecoveryBeforeCommit:2:${required_reader_source}")
  run_runtime(case_reader_run "--space" "l2" ${case_reader})
  if(NOT case_reader_run_status EQUAL 66 OR
     NOT case_reader_run_error MATCHES "reason=restart_incompatible")
    message(FATAL_ERROR
      "restart accepted ${case} generation authority: args=${case_reader};${case_reader_run_status};${case_reader_run_output};${case_reader_run_error}")
  endif()
endforeach()

set(authority_output "${HDG_RUNTIME_TEST_OUTPUT}-restart-authority")
set(authority_checkpoint "${authority_output}.checkpoint")
set(ordinary_checkpoint "${authority_output}.ordinary")
set(stale_checkpoint "${authority_output}.stale")
file(REMOVE_RECURSE "${authority_output}")
file(REMOVE "${authority_checkpoint}" "${ordinary_checkpoint}"
            "${stale_checkpoint}")
replace_cli_option(authority_writer common_arguments --output-dir
                   "${authority_output}")
replace_cli_option(authority_writer authority_writer --checkpoint-out
                   "${authority_checkpoint}")
replace_cli_option(authority_writer authority_writer --stop-after-checkpoint
                   true)
run_runtime(authority_writer_run "--space" "l2" ${authority_writer})
if(NOT authority_writer_run_status EQUAL 0 OR
   NOT authority_writer_run_output MATCHES "status=Success.*step=1")
  message(FATAL_ERROR
    "restart-authority writer failed: ${authority_writer_run_status};${authority_writer_run_output};${authority_writer_run_error}")
endif()
file(READ_SYMLINK "${authority_output}/.dg-igr-current"
                  authority_old_target)
set(authority_old_generation
    "${authority_output}/${authority_old_target}")
file(COPY_FILE "${authority_checkpoint}" "${ordinary_checkpoint}")
replace_cli_option(unauthorized_reader authority_writer
                   --stop-after-checkpoint false)
replace_cli_option(unauthorized_reader unauthorized_reader
                   --checkpoint-out none)
replace_cli_option(unauthorized_reader unauthorized_reader
                   --checkpoint-step none)
replace_cli_option(unauthorized_reader unauthorized_reader --restart-from
                   "${ordinary_checkpoint}")
replace_cli_option(unauthorized_reader unauthorized_reader --max-retries 1)
replace_cli_option(unauthorized_reader unauthorized_reader --inject-failure
                   "AfterRecoveryBeforeCommit:2:${required_reader_source}")
run_runtime(unauthorized_run "--space" "l2" ${unauthorized_reader})
if(NOT unauthorized_run_status EQUAL 66 OR
   NOT unauthorized_run_error MATCHES "reason=restart_incompatible")
  message(FATAL_ERROR
    "ordinary checkpoint copy bypassed generation authority: ${unauthorized_run_status};${unauthorized_run_output};${unauthorized_run_error}")
endif()

replace_cli_option(valid_reader unauthorized_reader --restart-from
                   "${authority_checkpoint}")
run_runtime(valid_reader_run "--space" "l2" ${valid_reader})
if(NOT valid_reader_run_status EQUAL 0 OR
   NOT valid_reader_run_output MATCHES "status=Success.*step=3.*retries=1" OR
   NOT valid_reader_run_error STREQUAL "")
  message(FATAL_ERROR
    "current manifest-valid checkpoint was rejected: ${valid_reader_run_status};${valid_reader_run_output};${valid_reader_run_error}")
endif()
file(CREATE_LINK "${authority_old_generation}/checkpoint.v2"
                 "${stale_checkpoint}" SYMBOLIC)
foreach(path IN ITEMS
    "${authority_old_generation}/checkpoint.v2" "${stale_checkpoint}")
  replace_cli_option(stale_reader valid_reader --restart-from "${path}")
  run_runtime(stale_reader_run "--space" "l2" ${stale_reader})
  if(NOT stale_reader_run_status EQUAL 66 OR
     NOT stale_reader_run_error MATCHES "reason=restart_incompatible")
    message(FATAL_ERROR
      "stale/direct checkpoint bypassed current authority: ${path};${stale_reader_run_status};${stale_reader_run_output};${stale_reader_run_error}")
  endif()
endforeach()

# Publication remains provisional through the final collective readiness
# boundary. Every admitted source rank must restore disk and memory, then retry
# exactly once without duplicating or losing a logical commit.
math(EXPR maximum_post_publication_source "${HDG_RUNTIME_RANKS} - 1")
foreach(phase IN ITEMS after-marker before-finalize)
  foreach(source RANGE 0 ${maximum_post_publication_source})
    set(post_output
        "${HDG_RUNTIME_TEST_OUTPUT}-post-${phase}-rank${source}")
    set(post_checkpoint "${post_output}.checkpoint")
    file(REMOVE_RECURSE "${post_output}")
    file(REMOVE "${post_checkpoint}")
    replace_cli_option(post_arguments common_arguments --output-dir
                       "${post_output}")
    replace_cli_option(post_arguments post_arguments --checkpoint-out
                       "${post_checkpoint}")
    run_private_post_publication_failure(
      post_run "${phase}:3:${source}" "--space" "l2" ${post_arguments})
    if(NOT post_run_status EQUAL 0 OR
       NOT post_run_output MATCHES "status=Success.*step=3.*retries=1" OR
       NOT post_run_error STREQUAL "")
      message(FATAL_ERROR
        "post-publication ${phase}/rank${source} retry failed: ${post_run_status};${post_run_output};${post_run_error}")
    endif()
    file(GLOB post_states "${post_output}/state-step-*.vtk")
    list(LENGTH post_states post_state_count)
    file(READ "${post_output}/run-metadata.json" post_metadata)
    if(NOT post_state_count EQUAL 3 OR
       NOT post_metadata MATCHES "\"committed_step\": 3" OR
       NOT post_metadata MATCHES "\"retries\": 1")
      message(FATAL_ERROR
        "post-publication ${phase}/rank${source} duplicated or lost a commit")
    endif()
  endforeach()
endforeach()

# With retry disabled, the previous marker must remain restartable by a later
# process. This observes the restored authority before any retry publishes.
set(no_retry_output "${HDG_RUNTIME_TEST_OUTPUT}-post-no-retry")
set(no_retry_checkpoint "${no_retry_output}.checkpoint")
file(REMOVE_RECURSE "${no_retry_output}")
file(REMOVE "${no_retry_checkpoint}")
replace_cli_option(no_retry_arguments common_arguments --output-dir
                   "${no_retry_output}")
replace_cli_option(no_retry_arguments no_retry_arguments --checkpoint-out
                   "${no_retry_checkpoint}")
run_private_post_publication_failure_no_retry(
  no_retry_run "after-marker:3:${maximum_post_publication_source}"
  "--space" "l2" ${no_retry_arguments})
if(NOT no_retry_run_status EQUAL 70 OR
   NOT no_retry_run_error MATCHES "step=2" OR
   NOT EXISTS "${no_retry_output}/.dg-igr-current" OR
   EXISTS "${no_retry_output}/state-step-000003.vtk")
  message(FATAL_ERROR
    "post-publication failure did not restore prior marker: ${no_retry_run_status};${no_retry_run_output};${no_retry_run_error}")
endif()
replace_cli_option(recovery_reader no_retry_arguments --checkpoint-out none)
replace_cli_option(recovery_reader recovery_reader --checkpoint-step none)
replace_cli_option(recovery_reader recovery_reader --restart-from
                   "${no_retry_checkpoint}")
replace_cli_option(recovery_reader recovery_reader --max-retries 1)
replace_cli_option(recovery_reader recovery_reader --inject-failure
                   "AfterRecoveryBeforeCommit:2:${required_reader_source}")
run_runtime(recovery_reader_run "--space" "l2" ${recovery_reader})
if(NOT recovery_reader_run_status EQUAL 0 OR
   NOT recovery_reader_run_output MATCHES
       "status=Success.*step=3.*retries=1" OR
   NOT recovery_reader_run_error STREQUAL "")
  message(FATAL_ERROR
    "restored prior generation was not restartable: ${recovery_reader_run_status};${recovery_reader_run_output};${recovery_reader_run_error}")
endif()

set(injection_phases Assembly Condensation TraceSolve Recovery Diagnostics
                     CheckpointStaging OutputStaging
                     AfterRecoveryBeforeCommit)
math(EXPR maximum_injection_source "${HDG_RUNTIME_RANKS} - 1")
foreach(phase IN LISTS injection_phases)
  if(phase STREQUAL "CheckpointStaging" OR phase STREQUAL "OutputStaging")
    set(injection_step 1)
  else()
    set(injection_step 2)
  endif()
  foreach(source RANGE 0 ${maximum_injection_source})
    file(REMOVE_RECURSE "${HDG_RUNTIME_TEST_OUTPUT}")
    file(REMOVE "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
    set(injected_arguments ${common_arguments})
    list(FIND injected_arguments --max-retries retry_option_index)
    list(FIND injected_arguments --inject-failure injection_option_index)
    if(retry_option_index LESS 0 OR injection_option_index LESS 0)
      message(FATAL_ERROR "focused injection controls are unavailable")
    endif()
    math(EXPR retry_value_index "${retry_option_index} + 1")
    math(EXPR injection_value_index "${injection_option_index} + 1")
    list(REMOVE_AT injected_arguments ${retry_value_index})
    list(INSERT injected_arguments ${retry_value_index} 1)
    list(REMOVE_AT injected_arguments ${injection_value_index})
    list(INSERT injected_arguments ${injection_value_index}
         "${phase}:${injection_step}:${source}")
    run_runtime(injected "--space" "l2" ${injected_arguments})
    if(NOT injected_status EQUAL 0 OR
       NOT injected_output MATCHES
           "status=Success.*step=3.*retries=1.*phase=${phase} source_rank=${source}" OR
       NOT injected_error STREQUAL "")
      message(FATAL_ERROR
        "focused ${phase}/rank${source} retry failed: ${injected_status};${injected_output};${injected_error}")
    endif()
    file(GLOB injected_states
         "${HDG_RUNTIME_TEST_OUTPUT}/state-step-*.vtk")
    list(LENGTH injected_states injected_state_count)
    file(READ "${HDG_RUNTIME_TEST_OUTPUT}/run-metadata.json"
         injected_metadata)
    if(NOT injected_state_count EQUAL 3 OR
       NOT injected_metadata MATCHES "\"committed_step\": 3" OR
       NOT injected_metadata MATCHES "\"retries\": 1")
      message(FATAL_ERROR
        "focused ${phase}/rank${source} duplicated or lost a commit")
    endif()
  endforeach()
endforeach()
