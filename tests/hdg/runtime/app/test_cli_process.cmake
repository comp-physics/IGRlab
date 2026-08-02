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
   NOT "$ENV{DG_IGR_TEST_REUSE_MODE}" STREQUAL "" AND
   NOT "$ENV{DG_IGR_TEST_REUSE_PROFILE}" STREQUAL "" AND
   NOT "$ENV{DG_IGR_TEST_REUSE_CALIPER}" STREQUAL "" AND
   NOT "$ENV{DG_IGR_TEST_REUSE_CUDA_ARCH}" STREQUAL "")
  if("$ENV{DG_IGR_TEST_REUSE_MODE}" STREQUAL "gpu")
    if(NOT "$ENV{DG_IGR_TEST_REUSE_CALIPER}" STREQUAL "0" OR
       NOT "$ENV{DG_IGR_TEST_REUSE_CUDA_ARCH}" MATCHES "^[0-9]+$" OR
       NOT "$ENV{DG_IGR_TEST_REUSE_PROFILE}" STREQUAL
           "cuda-sm$ENV{DG_IGR_TEST_REUSE_CUDA_ARCH}")
      message(FATAL_ERROR "inconsistent GPU reuse-fixture profile")
    endif()
    set(reuse_mode_args --gpu --cuda-arch
      "$ENV{DG_IGR_TEST_REUSE_CUDA_ARCH}")
  elseif("$ENV{DG_IGR_TEST_REUSE_MODE}" STREQUAL "cpu")
    if(NOT "$ENV{DG_IGR_TEST_REUSE_CUDA_ARCH}" STREQUAL "none")
      message(FATAL_ERROR "inconsistent CPU reuse-fixture CUDA profile")
    elseif("$ENV{DG_IGR_TEST_REUSE_PROFILE}" STREQUAL "cpu" AND
           "$ENV{DG_IGR_TEST_REUSE_CALIPER}" STREQUAL "0")
      set(reuse_mode_args --cpu-only)
    elseif("$ENV{DG_IGR_TEST_REUSE_PROFILE}" STREQUAL "cpu-caliper" AND
           "$ENV{DG_IGR_TEST_REUSE_CALIPER}" STREQUAL "1")
      set(reuse_mode_args --cpu-only --caliper)
    else()
      message(FATAL_ERROR "inconsistent CPU reuse-fixture profile")
    endif()
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

function(run_serial_runtime prefix)
  execute_process(
    COMMAND "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(run_private_scalar_finalize_preparation_failure prefix)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      DG_IGR_HDG_TEST_SCALAR_FINALIZE_PREPARATION_FAILURE=8:1
      "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
      ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
      "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
    TIMEOUT 180
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(run_private_scalar_restart_preparation_failure prefix)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      DG_IGR_HDG_TEST_SCALAR_RESTART_PREPARATION_FAILURE=10:1
      "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
      ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
      "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
    TIMEOUT 180
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  set(${prefix}_status "${status}" PARENT_SCOPE)
  set(${prefix}_output "${output}" PARENT_SCOPE)
  set(${prefix}_error "${error}" PARENT_SCOPE)
endfunction()

function(run_private_scalar_preparation_failure prefix environment value)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "${environment}=${value}"
      "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
      ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
      "${HDG_RUNTIME_EXECUTABLE}" ${ARGN}
      ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
    TIMEOUT 180
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
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

function(require_manifest_artifact generation artifact label)
  set(artifact_path "${generation}/${artifact}")
  set(manifest_path "${generation}/manifest.v1")
  require_regular_file("${artifact_path}" "${label} artifact")
  require_regular_file("${manifest_path}" "${label} manifest")
  file(READ "${manifest_path}" manifest)
  string(ASCII 9 tab)
  string(FIND "${manifest}" "artifact${tab}${artifact}${tab}"
              artifact_record)
  if(artifact_record LESS 0)
    message(FATAL_ERROR
      "${label} is not authorized by its generation manifest: ${artifact_path}")
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

function(validate_scalar_igr_vtk path mesh_n label)
  file(READ "${path}" vtk)
  math(EXPR elements "${mesh_n} * ${mesh_n}")
  math(EXPR points "4 * ${elements}")
  if(NOT vtk MATCHES "DATASET UNSTRUCTURED_GRID" OR
     NOT vtk MATCHES "POINTS ${points} double" OR
     NOT vtk MATCHES "CELLS ${elements} " OR
     NOT vtk MATCHES "SCALARS U0 double 1" OR
     NOT vtk MATCHES "SCALARS U3 double 1" OR
     NOT vtk MATCHES "SCALARS Sigma double 1" OR
     NOT vtk MATCHES "VECTORS z double")
    message(FATAL_ERROR
      "${label} is not a physical scalar-IGR U/Sigma/z FE VTK")
  endif()
endfunction()

function(validate_scalar_cuda_metadata metadata label expected_step)
  if(HDG_RUNTIME_RANKS EQUAL 1)
    set(expected_backend "cuda.rank1.one_gpu")
  else()
    set(expected_backend "cuda.rank2.one_gpu.host_staged")
  endif()
  string(FIND "${metadata}" "\"fields\": [\"U\", \"Sigma\", \"z\"]"
       fields_offset)
  if(fields_offset EQUAL -1 OR
     NOT metadata MATCHES
         "\"card\": \"exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3\"" OR
     NOT metadata MATCHES
         "\"scenario\": \"W8B-SC25-N8-K1-20STEP-v1\"" OR
     NOT metadata MATCHES "\"backend\": \"${expected_backend}\"" OR
     NOT metadata MATCHES
         "\"layout\": \"mfem.l2.byNODES.U4.Sigma1.z2.separate.v3\"" OR
     NOT metadata MATCHES "\"gpu_uuid\": \"GPU-[0-9a-f]+\"" OR
     NOT metadata MATCHES "\"committed_step\": ${expected_step}" OR
     NOT metadata MATCHES "\"accepted_steps\": ${expected_step}" OR
     NOT metadata MATCHES "\"status\": \"Success\"" OR
     NOT metadata MATCHES "\"reason\": \"none\"" OR
     NOT metadata MATCHES "\"trace_residual_gate\": true" OR
     NOT metadata MATCHES "\"solver_contract\": true" OR
     NOT metadata MATCHES "\"persistent_reuse\": true" OR
     NOT metadata MATCHES "\"zero_fallback\": true" OR
     NOT metadata MATCHES "\"zero_post_baseline_allocation\": true" OR
     NOT metadata MATCHES "\"device_residency\": true" OR
     NOT metadata MATCHES "\"host_staged_same_gpu\": true")
    message(FATAL_ERROR "${label} CUDA scalar-IGR evidence is incomplete")
  endif()
  foreach(scientific_field IN ITEMS maximum_conservation_error minimum_density
                                    minimum_pressure minimum_speed_margin)
    string(REGEX MATCH
      "\"${scientific_field}\": ([0-9.eE+-]+)"
      scientific_match "${metadata}")
    if(CMAKE_MATCH_1 STREQUAL "")
      message(FATAL_ERROR
        "${label} CUDA scalar-IGR ${scientific_field} evidence is missing")
    endif()
    set(${scientific_field} "${CMAKE_MATCH_1}")
  endforeach()
  if(maximum_conservation_error GREATER 2e-10 OR
     minimum_density LESS 1e-8 OR minimum_pressure LESS 1e-8 OR
     minimum_speed_margin LESS 0)
    message(FATAL_ERROR
      "${label} CUDA scalar-IGR scientific gate failed: conservation=${maximum_conservation_error};density=${minimum_density};pressure=${minimum_pressure};speed_margin=${minimum_speed_margin}")
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

  if(HDG_RUNTIME_SCALAR_IGR)
    set(functional_common
      --card exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3
      --case sc25-periodic-smooth-persistent-v3
      --space l2 --mesh-family periodic-cartesian-quad --mesh-n 8 --order 1
      --dt 0.000244140625 --steps 20 --final-time 0.0048828125
      --backend "${HDG_RUNTIME_BACKEND}" --device-id "${runtime_device}"
      --output-steps 0,10,20 --seed 0x5738425343323553)
    set(functional_mesh_n 8)
    set(functional_injection_step 7)
    set(reference_retry_limit 1)
    set(reference_injection
        "AfterRecoveryBeforeCommit:${functional_injection_step}:${injection_rank}")
    set(writer_retry_limit 1)
    set(writer_injection "${reference_injection}")
    set(reader_retry_limit 0)
    set(reader_injection off)
    set(reference_expected_retries 1)
    set(writer_expected_retries 1)
    set(reader_expected_retries 1)
  else()
    set(functional_common
      --card exp.hdg.euler.conventional.smooth_persistent.w8a.v1
      --case conventional-smooth-persistent-v1
      --space l2 --mesh-family periodic-cartesian-quad --mesh-n 6 --order 2
      --dt 0.05 --steps 20 --final-time 1.0
      --backend "${HDG_RUNTIME_BACKEND}" --device-id "${runtime_device}"
      --output-steps 0,10,20 --seed 0x57384146554e4331)
    set(functional_mesh_n 6)
    set(functional_injection_step 14)
    set(reference_retry_limit 0)
    set(reference_injection off)
    set(writer_retry_limit 0)
    set(writer_injection off)
    set(reader_retry_limit 1)
    set(reader_injection
        "AfterRecoveryBeforeCommit:${functional_injection_step}:${injection_rank}")
    set(reference_expected_retries 0)
    set(writer_expected_retries 0)
    set(reader_expected_retries 1)
  endif()

  if(HDG_RUNTIME_SCALAR_IGR)
    set(h1_directory "${HDG_RUNTIME_TEST_OUTPUT}-h1")
    set(h1_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}-h1.checkpoint")
    file(REMOVE_RECURSE "${h1_directory}")
    file(REMOVE "${h1_checkpoint}")
    replace_cli_option(h1_arguments functional_common --space h1)
    run_runtime(h1 ${h1_arguments}
      --output-dir "${h1_directory}"
      --checkpoint-out "${h1_checkpoint}" --checkpoint-step 10
      --stop-after-checkpoint false --restart-from none
      --max-retries "${reference_retry_limit}"
      --inject-failure "${reference_injection}")
    if(NOT h1_status EQUAL 65 OR
       NOT h1_error MATCHES "unsupported_h1" OR
       EXISTS "${h1_directory}" OR EXISTS "${h1_checkpoint}")
      message(FATAL_ERROR
        "scalar-IGR H1 rejection had side effects: ${h1_status};${h1_output};${h1_error}")
    endif()
  endif()

  run_runtime(reference ${functional_common}
    --output-dir "${reference_directory}"
    --checkpoint-out "${reference_checkpoint}" --checkpoint-step 10
    --stop-after-checkpoint false --restart-from none
    --max-retries "${reference_retry_limit}"
    --inject-failure "${reference_injection}")
  if(NOT reference_status EQUAL 0 OR
     NOT reference_output MATCHES
         "status=Success.*step=20.*retries=${reference_expected_retries}" OR
     NOT reference_error STREQUAL "")
    message(FATAL_ERROR
      "functional uninterrupted run failed: ${reference_status};${reference_output};${reference_error}")
  endif()

  run_runtime(writer ${functional_common}
    --output-dir "${writer_directory}"
    --checkpoint-out "${split_checkpoint}" --checkpoint-step 10
    --stop-after-checkpoint true --restart-from none
    --max-retries "${writer_retry_limit}"
    --inject-failure "${writer_injection}")
  if(NOT writer_status EQUAL 0 OR
     NOT writer_output MATCHES
         "status=Success.*step=10.*retries=${writer_expected_retries}" OR
     NOT writer_error STREQUAL "")
    message(FATAL_ERROR
      "functional split writer failed: ${writer_status};${writer_output};${writer_error}")
  endif()

  run_runtime(reader ${functional_common}
    --output-dir "${reader_directory}"
    --checkpoint-out none --checkpoint-step none
    --stop-after-checkpoint false --restart-from "${split_checkpoint}"
    --max-retries "${reader_retry_limit}"
    --inject-failure "${reader_injection}")
  if(NOT reader_status EQUAL 0 OR
     NOT reader_output MATCHES
         "status=Success.*step=20.*retries=${reader_expected_retries}" OR
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
  if(HDG_RUNTIME_SCALAR_IGR)
    validate_scalar_igr_vtk(
      "${reference_directory}/state-step-000000.vtk" ${functional_mesh_n}
      "functional initial generation")
    validate_scalar_igr_vtk(
      "${reference_directory}/state-step-000010.vtk" ${functional_mesh_n}
      "functional intermediate generation")
    validate_scalar_igr_vtk(
      "${reference_directory}/state-step-000020.vtk" ${functional_mesh_n}
      "functional final generation")
  else()
    validate_physical_vtk(
      "${reference_directory}/state-step-000000.vtk" ${functional_mesh_n}
      "functional initial generation")
    validate_physical_vtk(
      "${reference_directory}/state-step-000010.vtk" ${functional_mesh_n}
      "functional intermediate generation")
    validate_physical_vtk(
      "${reference_directory}/state-step-000020.vtk" ${functional_mesh_n}
      "functional final generation")
  endif()

  file(READ "${reference_directory}/run-metadata.json" reference_metadata)
  file(READ "${writer_directory}/run-metadata-checkpoint.json"
       writer_metadata)
  file(READ "${reader_directory}/run-metadata.json" reader_metadata)
  if(HDG_RUNTIME_SCALAR_IGR AND
     HDG_RUNTIME_BACKEND STREQUAL "cuda")
    validate_scalar_cuda_metadata("${reference_metadata}" "reference" 20)
    validate_scalar_cuda_metadata("${writer_metadata}" "split writer" 10)
    validate_scalar_cuda_metadata("${reader_metadata}" "split reader" 20)
  endif()
  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_RANKS EQUAL 2 AND
     HDG_RUNTIME_BACKEND STREQUAL "cpu")
    foreach(injection_metadata IN ITEMS reference_metadata writer_metadata)
      if(NOT "${${injection_metadata}}" MATCHES
             "\"injection_configured_source_rank\": 1" OR
         NOT "${${injection_metadata}}" MATCHES
             "\"injection_observed_origin_rank\": 1" OR
         NOT "${${injection_metadata}}" MATCHES
             "\"injection_local_count\": 1" OR
         NOT "${${injection_metadata}}" MATCHES
             "\"injection_non_origin_local_count\": 0" OR
         NOT "${${injection_metadata}}" MATCHES
             "\"injection_collective_decisions\": 1")
        message(FATAL_ERROR
          "scalar-IGR source-rank injection audit is invalid")
      endif()
    endforeach()
  endif()
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
  if(HDG_RUNTIME_SCALAR_IGR)
    if(NOT DEFINED HDG_RUNTIME_SCALAR_VTK_TEST)
      message(FATAL_ERROR "scalar-IGR physical VTK comparator is unavailable")
    endif()
    # MFEM's element-coloring diagnostic is not stable across independent
    # processes. Compare exact global coordinates and every U/Sigma/z value;
    # the metadata state digest below separately requires byte identity.
    execute_process(
      COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk
        "${reference_directory}/state-step-000020.vtk"
        "${reader_directory}/state-step-000020.vtk" 1e-15
      RESULT_VARIABLE restart_vtk_compare_status)
    set(restart_vtk_identity TRUE)
  else()
    if(NOT DEFINED HDG_RUNTIME_SCIENTIFIC_TEST)
      message(FATAL_ERROR "physical VTK comparator is unavailable")
    endif()
    execute_process(
      COMMAND "${HDG_RUNTIME_SCIENTIFIC_TEST}" --compare-vtk
        "${reference_directory}/state-step-000020.vtk"
        "${reader_directory}/state-step-000020.vtk" 1e-15
      RESULT_VARIABLE restart_vtk_compare_status)
    set(restart_vtk_identity TRUE)
  endif()
  if(NOT restart_vtk_identity OR NOT restart_vtk_compare_status EQUAL 0 OR
     NOT state_parity OR NOT reader_metadata MATCHES "\"status\": \"Success\"" OR
     NOT reader_metadata MATCHES "\"retries\": 1")
    message(FATAL_ERROR "functional restart/retry parity contract failed")
  endif()
  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_RANKS EQUAL 2 AND
     HDG_RUNTIME_BACKEND STREQUAL "cpu")
    if(NOT DEFINED HDG_RUNTIME_SCALAR_VTK_TEST)
      message(FATAL_ERROR "scalar-IGR physical VTK comparator is unavailable")
    endif()
    set(serial_oracle_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-independent-serial-oracle")
    set(serial_oracle_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-independent-serial-oracle.checkpoint")
    file(REMOVE_RECURSE "${serial_oracle_directory}")
    file(REMOVE "${serial_oracle_checkpoint}")
    run_serial_runtime(serial_oracle ${functional_common}
      --output-dir "${serial_oracle_directory}"
      --checkpoint-out "${serial_oracle_checkpoint}" --checkpoint-step 10
      --stop-after-checkpoint false --restart-from none
      --max-retries 1 --inject-failure "AfterRecoveryBeforeCommit:7:0")
    if(NOT serial_oracle_status EQUAL 0 OR
       NOT serial_oracle_output MATCHES "status=Success.*step=20.*retries=1" OR
       NOT serial_oracle_error STREQUAL "")
      message(FATAL_ERROR
        "scalar-IGR independent serial oracle failed: ${serial_oracle_status};${serial_oracle_output};${serial_oracle_error}")
    endif()
    foreach(initial_vtk IN ITEMS
        "${serial_oracle_directory}/state-step-000000.vtk"
        "${reference_directory}/state-step-000000.vtk")
      execute_process(
        COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}"
          --validate-scalar-initial-vtk "${initial_vtk}" 2e-11
        RESULT_VARIABLE scalar_initial_oracle_status
        OUTPUT_VARIABLE scalar_initial_oracle_output
        ERROR_VARIABLE scalar_initial_oracle_error)
      if(NOT scalar_initial_oracle_status EQUAL 0)
        message(FATAL_ERROR
          "scalar-IGR independent topology/physics oracle failed for ${initial_vtk}: ${scalar_initial_oracle_output};${scalar_initial_oracle_error}")
      endif()
    endforeach()
    foreach(step IN ITEMS 000000 000010 000020)
      execute_process(
        COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk
          "${serial_oracle_directory}/state-step-${step}.vtk"
          "${reference_directory}/state-step-${step}.vtk"
          2e-11
        RESULT_VARIABLE scalar_element_oracle_status
        OUTPUT_VARIABLE scalar_element_oracle_output
        ERROR_VARIABLE scalar_element_oracle_error)
      if(NOT scalar_element_oracle_status EQUAL 0)
        message(FATAL_ERROR
          "scalar-IGR rank2 x-slab element/value oracle failed at ${step}: ${scalar_element_oracle_output};${scalar_element_oracle_error}")
      endif()
    endforeach()

    set(initial_pack_negative_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-initial-pack-negative")
    set(initial_pack_negative_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-initial-pack-negative.checkpoint")
    file(REMOVE_RECURSE "${initial_pack_negative_directory}")
    file(REMOVE "${initial_pack_negative_checkpoint}")
    run_private_scalar_preparation_failure(initial_pack_negative
      DG_IGR_HDG_TEST_SCALAR_INITIAL_PACK_FAILURE 0:1
      ${functional_common}
      --output-dir "${initial_pack_negative_directory}"
      --checkpoint-out "${initial_pack_negative_checkpoint}"
      --checkpoint-step 10 --stop-after-checkpoint false --restart-from none
      --max-retries 1 --inject-failure "AfterRecoveryBeforeCommit:7:1")
    file(GLOB initial_pack_negative_states
         "${initial_pack_negative_directory}/state-step-*.vtk")
    file(GLOB initial_pack_negative_generations
         "${initial_pack_negative_directory}/.dg-igr-generations/generation-*")
    if(NOT initial_pack_negative_status EQUAL 72 OR
       NOT initial_pack_negative_error MATCHES
           "initial pack failure source_rank=1" OR
       initial_pack_negative_states OR initial_pack_negative_generations OR
       EXISTS "${initial_pack_negative_directory}/.dg-igr-current" OR
       EXISTS "${initial_pack_negative_directory}/run-metadata.json" OR
       EXISTS "${initial_pack_negative_checkpoint}")
      message(FATAL_ERROR
        "scalar-IGR asymmetric initial-pack consensus failed: ${initial_pack_negative_status};${initial_pack_negative_output};${initial_pack_negative_error}")
    endif()

    set(candidate_pack_negative_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-candidate-pack-negative")
    set(candidate_pack_negative_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-candidate-pack-negative.checkpoint")
    file(REMOVE_RECURSE "${candidate_pack_negative_directory}")
    file(REMOVE "${candidate_pack_negative_checkpoint}")
    run_private_scalar_preparation_failure(candidate_pack_negative
      DG_IGR_HDG_TEST_SCALAR_CANDIDATE_PACK_FAILURE 8:1
      ${functional_common}
      --output-dir "${candidate_pack_negative_directory}"
      --checkpoint-out "${candidate_pack_negative_checkpoint}"
      --checkpoint-step 10 --stop-after-checkpoint false --restart-from none
      --max-retries 1 --inject-failure "AfterRecoveryBeforeCommit:7:1")
    file(GLOB candidate_pack_negative_states
         "${candidate_pack_negative_directory}/state-step-*.vtk")
    list(LENGTH candidate_pack_negative_states
         candidate_pack_negative_state_count)
    file(READ_SYMLINK
         "${candidate_pack_negative_directory}/.dg-igr-current"
         candidate_pack_negative_current)
    if(NOT candidate_pack_negative_status EQUAL 72 OR
       NOT candidate_pack_negative_error MATCHES
           "candidate pack failure source_rank=1" OR
       NOT candidate_pack_negative_state_count EQUAL 1 OR
       NOT candidate_pack_negative_current MATCHES "generation-0-" OR
       NOT EXISTS
           "${candidate_pack_negative_directory}/state-step-000000.vtk" OR
       EXISTS "${candidate_pack_negative_directory}/state-step-000010.vtk" OR
       EXISTS "${candidate_pack_negative_directory}/run-metadata.json" OR
       EXISTS "${candidate_pack_negative_checkpoint}")
      message(FATAL_ERROR
        "scalar-IGR asymmetric candidate-pack rollback failed: ${candidate_pack_negative_status};${candidate_pack_negative_output};${candidate_pack_negative_error}")
    endif()

    set(metadata_negative_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-metadata-preparation-negative")
    set(metadata_negative_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-metadata-preparation-negative.checkpoint")
    file(REMOVE_RECURSE "${metadata_negative_directory}")
    file(REMOVE "${metadata_negative_checkpoint}")
    run_private_scalar_preparation_failure(metadata_negative
      DG_IGR_HDG_TEST_SCALAR_METADATA_PREPARATION_FAILURE 20:0
      ${functional_common}
      --output-dir "${metadata_negative_directory}"
      --checkpoint-out "${metadata_negative_checkpoint}"
      --checkpoint-step 10 --stop-after-checkpoint false --restart-from none
      --max-retries 1 --inject-failure "AfterRecoveryBeforeCommit:7:1")
    file(GLOB metadata_negative_states
         "${metadata_negative_directory}/state-step-*.vtk")
    list(LENGTH metadata_negative_states metadata_negative_state_count)
    file(READ_SYMLINK "${metadata_negative_directory}/.dg-igr-current"
         metadata_negative_current)
    if(NOT metadata_negative_status EQUAL 72 OR
       NOT metadata_negative_error MATCHES
           "metadata preparation failure source_rank=0" OR
       NOT metadata_negative_state_count EQUAL 2 OR
       NOT metadata_negative_current MATCHES "generation-10-" OR
       NOT EXISTS "${metadata_negative_directory}/state-step-000000.vtk" OR
       NOT EXISTS "${metadata_negative_directory}/state-step-000010.vtk" OR
       NOT EXISTS "${metadata_negative_checkpoint}" OR
       EXISTS "${metadata_negative_directory}/state-step-000020.vtk" OR
       EXISTS "${metadata_negative_directory}/run-metadata.json")
      message(FATAL_ERROR
        "scalar-IGR root metadata rollback failed: ${metadata_negative_status};${metadata_negative_output};${metadata_negative_error}")
    endif()

    set(finalize_negative_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-finalize-preparation-negative")
    set(finalize_negative_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-finalize-preparation-negative.checkpoint")
    file(REMOVE_RECURSE "${finalize_negative_directory}")
    file(REMOVE "${finalize_negative_checkpoint}")
    run_private_scalar_finalize_preparation_failure(finalize_negative
      ${functional_common}
      --output-dir "${finalize_negative_directory}"
      --checkpoint-out "${finalize_negative_checkpoint}" --checkpoint-step 10
      --stop-after-checkpoint false --restart-from none
      --max-retries 1 --inject-failure "AfterRecoveryBeforeCommit:7:1")
    file(GLOB finalize_negative_states
         "${finalize_negative_directory}/state-step-*.vtk")
    list(LENGTH finalize_negative_states finalize_negative_state_count)
    if(NOT finalize_negative_status EQUAL 72 OR
       NOT finalize_negative_error MATCHES
           "finalize preparation failure source_rank=1" OR
       NOT finalize_negative_state_count EQUAL 1 OR
       NOT EXISTS "${finalize_negative_directory}/state-step-000000.vtk" OR
       EXISTS "${finalize_negative_directory}/state-step-000010.vtk" OR
       EXISTS "${finalize_negative_directory}/run-metadata.json" OR
       EXISTS "${finalize_negative_checkpoint}")
      message(FATAL_ERROR
        "scalar-IGR asymmetric finalize-preparation rollback failed: ${finalize_negative_status};${finalize_negative_output};${finalize_negative_error}")
    endif()

    set(restart_negative_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-restart-preparation-negative")
    set(restart_negative_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-restart-preparation-negative.checkpoint")
    file(REMOVE_RECURSE "${restart_negative_directory}")
    file(REMOVE "${restart_negative_checkpoint}")
    run_runtime(restart_negative_writer ${functional_common}
      --output-dir "${restart_negative_directory}"
      --checkpoint-out "${restart_negative_checkpoint}" --checkpoint-step 10
      --stop-after-checkpoint true --restart-from none
      --max-retries 1 --inject-failure "AfterRecoveryBeforeCommit:7:1")
    if(NOT restart_negative_writer_status EQUAL 0)
      message(FATAL_ERROR
        "scalar-IGR restart-preparation fixture writer failed")
    endif()
    file(SHA256 "${restart_negative_checkpoint}" restart_checkpoint_before)
    file(SHA256 "${restart_negative_directory}/state-step-000010.vtk"
         restart_state_before)
    file(READ_SYMLINK "${restart_negative_directory}/.dg-igr-current"
         restart_current_before)
    file(GLOB restart_generations_before
         "${restart_negative_directory}/.dg-igr-generations/generation-*")
    list(LENGTH restart_generations_before restart_generation_count_before)
    run_private_scalar_restart_preparation_failure(restart_negative
      ${functional_common}
      --output-dir "${restart_negative_directory}"
      --checkpoint-out none --checkpoint-step none
      --stop-after-checkpoint false
      --restart-from "${restart_negative_checkpoint}"
      --max-retries 0 --inject-failure off)
    file(SHA256 "${restart_negative_checkpoint}" restart_checkpoint_after)
    file(SHA256 "${restart_negative_directory}/state-step-000010.vtk"
         restart_state_after)
    file(READ_SYMLINK "${restart_negative_directory}/.dg-igr-current"
         restart_current_after)
    file(GLOB restart_generations_after
         "${restart_negative_directory}/.dg-igr-generations/generation-*")
    list(LENGTH restart_generations_after restart_generation_count_after)
    if(NOT restart_negative_status EQUAL 66 OR
       NOT restart_negative_error MATCHES
           "restart preparation failure source_rank=1" OR
       NOT restart_checkpoint_before STREQUAL restart_checkpoint_after OR
       NOT restart_state_before STREQUAL restart_state_after OR
       NOT restart_current_before STREQUAL restart_current_after OR
       NOT restart_generation_count_before EQUAL restart_generation_count_after OR
       EXISTS "${restart_negative_directory}/state-step-000020.vtk" OR
       EXISTS "${restart_negative_directory}/run-metadata.json")
      message(FATAL_ERROR
        "scalar-IGR asymmetric restart-preparation rollback failed: ${restart_negative_status};${restart_negative_output};${restart_negative_error}")
    endif()
  endif()
  if(HDG_RUNTIME_BACKEND STREQUAL "cuda")
    if(HDG_RUNTIME_SCALAR_IGR)
      if(NOT DEFINED HDG_RUNTIME_SCALAR_VTK_TEST)
        message(FATAL_ERROR "CUDA scalar field parity comparator is unavailable")
      endif()
    elseif(NOT DEFINED HDG_RUNTIME_SCIENTIFIC_TEST)
      message(FATAL_ERROR "CUDA field parity comparator is unavailable")
    endif()
    if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_RANKS EQUAL 2)
      if(NOT DEFINED HDG_RUNTIME_SCALAR_RANK2_MANIFEST_CONSUMER OR
         "$ENV{DG_IGR_SC25_R2_HANDOFF_DIR}" STREQUAL "" OR
         "$ENV{DG_IGR_SC25_R2_RUN_ID}" STREQUAL "" OR
         "$ENV{DG_IGR_SC25_R2_COMPLETE_SHA256}" STREQUAL "" OR
         "$ENV{DG_IGR_SC25_R2_RANK0_SHA256}" STREQUAL "" OR
         "$ENV{DG_IGR_SC25_R2_RANK1_SHA256}" STREQUAL "")
        message(FATAL_ERROR
          "rank-2 CUDA scalar parity requires the immutable CPU handoff locator and exact hashes")
      endif()
      set(handoff_directory "$ENV{DG_IGR_SC25_R2_HANDOFF_DIR}")
      set(handoff_complete "${handoff_directory}/COMPLETE.c3m")
      set(handoff_rank0 "${handoff_directory}/rank-0.c3m")
      set(handoff_rank1 "${handoff_directory}/rank-1.c3m")
      foreach(handoff_file IN ITEMS handoff_complete handoff_rank0 handoff_rank1)
        require_regular_file("${${handoff_file}}" "immutable CPU handoff")
      endforeach()
      file(SHA256 "${handoff_complete}" observed_complete_sha256)
      file(SHA256 "${handoff_rank0}" observed_rank0_sha256)
      file(SHA256 "${handoff_rank1}" observed_rank1_sha256)
      if(NOT observed_complete_sha256 STREQUAL
             "$ENV{DG_IGR_SC25_R2_COMPLETE_SHA256}" OR
         NOT observed_rank0_sha256 STREQUAL
             "$ENV{DG_IGR_SC25_R2_RANK0_SHA256}" OR
         NOT observed_rank1_sha256 STREQUAL
             "$ENV{DG_IGR_SC25_R2_RANK1_SHA256}")
        message(FATAL_ERROR "immutable CPU handoff digest mismatch")
      endif()
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
          "DG_IGR_SC25_R2_HANDOFF_DIR=${handoff_directory}"
          "DG_IGR_SC25_R2_RUN_ID=$ENV{DG_IGR_SC25_R2_RUN_ID}"
          "DG_IGR_SC25_R2_COMPLETE_SHA256=$ENV{DG_IGR_SC25_R2_COMPLETE_SHA256}"
          "${HDG_RUNTIME_MPIEXEC}" "${HDG_RUNTIME_NUMPROC_FLAG}" 2
          ${HDG_RUNTIME_MPIEXEC_PREFLAGS}
          "${HDG_RUNTIME_SCALAR_RANK2_MANIFEST_CONSUMER}"
          ${HDG_RUNTIME_MPIEXEC_POSTFLAGS}
        RESULT_VARIABLE manifest_consumer_status
        OUTPUT_VARIABLE manifest_consumer_output
        ERROR_VARIABLE manifest_consumer_error)
      if(NOT manifest_consumer_status EQUAL 0 OR
         NOT manifest_consumer_output MATCHES
             "SC25_CUDA_RANK2_C3_PASS.*states=21 steps=20.*host_staged=1.*manifest_unchanged=1" OR
         NOT manifest_consumer_output MATCHES
             "SC25_CUDA_RANK2_MANIFEST_CONSUMED" OR
         NOT manifest_consumer_error STREQUAL "")
        message(FATAL_ERROR
          "immutable CPU producer/CUDA consumer parity failed: ${manifest_consumer_status};${manifest_consumer_output};${manifest_consumer_error}")
      endif()
      file(SHA256 "${handoff_complete}" final_complete_sha256)
      file(SHA256 "${handoff_rank0}" final_rank0_sha256)
      file(SHA256 "${handoff_rank1}" final_rank1_sha256)
      if(NOT final_complete_sha256 STREQUAL observed_complete_sha256 OR
         NOT final_rank0_sha256 STREQUAL observed_rank0_sha256 OR
         NOT final_rank1_sha256 STREQUAL observed_rank1_sha256)
        message(FATAL_ERROR "immutable CPU handoff changed during consumption")
      endif()
    else()
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
      if(HDG_RUNTIME_SCALAR_IGR)
        set(cpu_parity_retry_limit 1)
        set(cpu_parity_injection "${reference_injection}")
      else()
        set(cpu_parity_retry_limit 0)
        set(cpu_parity_injection off)
      endif()
      run_runtime(cpu_reference ${cpu_functional_common}
        --output-dir "${cpu_directory}"
        --checkpoint-out "${cpu_checkpoint}" --checkpoint-step 10
        --stop-after-checkpoint false --restart-from none
        --max-retries "${cpu_parity_retry_limit}"
        --inject-failure "${cpu_parity_injection}")
      if(NOT cpu_reference_status EQUAL 0 OR
         NOT cpu_reference_output MATCHES "status=Success.*step=20" OR
         NOT cpu_reference_error STREQUAL "")
        message(FATAL_ERROR
          "CPU parity trajectory failed: ${cpu_reference_status};${cpu_reference_output};${cpu_reference_error}")
      endif()
      foreach(step IN ITEMS 000000 000010 000020)
        if(HDG_RUNTIME_SCALAR_IGR)
          set(parity_command "${HDG_RUNTIME_SCALAR_VTK_TEST}"
              --compare-scalar-vtk)
          set(parity_tolerance 1e-10)
        else()
          set(parity_command "${HDG_RUNTIME_SCIENTIFIC_TEST}" --compare-vtk)
          set(parity_tolerance 2e-8)
        endif()
        execute_process(
          COMMAND ${parity_command}
            "${cpu_directory}/state-step-${step}.vtk"
            "${reference_directory}/state-step-${step}.vtk"
            "${parity_tolerance}"
          RESULT_VARIABLE compare_status
          OUTPUT_VARIABLE compare_output
          ERROR_VARIABLE compare_error)
        if(NOT compare_status EQUAL 0 OR NOT compare_error STREQUAL "")
          message(FATAL_ERROR
            "CPU-CUDA globally ordered field parity failed at ${step}: ${compare_output};${compare_error}")
        endif()
      endforeach()
    endif()
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
  set(case_generation_checkpoint
      "${case_generation}/restart-checkpoint.v2")
  require_manifest_artifact(
    "${case_generation}" "restart-checkpoint.v2"
    "manifest-authority ${case} checkpoint")
  if(case STREQUAL "corrupt")
    file(SIZE "${case_generation_checkpoint}" checkpoint_size_before)
    file(SHA256 "${case_generation_checkpoint}" checkpoint_sha_before)
    file(APPEND "${case_generation_checkpoint}" "corrupt")
    file(SIZE "${case_generation_checkpoint}" checkpoint_size_after)
    file(SHA256 "${case_generation_checkpoint}" checkpoint_sha_after)
    if(checkpoint_size_after LESS_EQUAL checkpoint_size_before OR
       checkpoint_sha_after STREQUAL checkpoint_sha_before)
      message(FATAL_ERROR
        "real manifest-listed checkpoint was not corrupted fail-closed")
    endif()
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
set(authority_old_checkpoint
    "${authority_old_generation}/restart-checkpoint.v2")
require_manifest_artifact(
  "${authority_old_generation}" "restart-checkpoint.v2"
  "formerly current restart checkpoint")
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
file(READ_SYMLINK "${authority_output}/.dg-igr-current"
                  authority_new_target)
if(authority_new_target STREQUAL authority_old_target)
  message(FATAL_ERROR
    "restart-authority fixture did not make the prior generation stale")
endif()
require_regular_file("${authority_old_checkpoint}"
                     "formerly current restart checkpoint")
file(CREATE_LINK "${authority_old_checkpoint}"
                 "${stale_checkpoint}" SYMBOLIC)
foreach(path IN ITEMS
    "${authority_old_checkpoint}" "${stale_checkpoint}")
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
