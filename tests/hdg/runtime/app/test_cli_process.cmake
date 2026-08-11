cmake_policy(SET CMP0054 NEW)

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

function(require_pre_resource_rejection label expected_rejection
         output_path checkpoint_path)
  file(REMOVE_RECURSE "${output_path}")
  file(REMOVE "${checkpoint_path}")
  execute_process(
    COMMAND "${HDG_RUNTIME_EXECUTABLE}" --space l2 ${ARGN}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT status EQUAL 64 OR
     NOT error MATCHES "${expected_rejection}" OR
     EXISTS "${output_path}" OR EXISTS "${checkpoint_path}")
    message(FATAL_ERROR
      "${label} did not reject without artifacts: ${status};${output};${error}")
  endif()
endfunction()

function(require_regular_file path label)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
    message(FATAL_ERROR "missing ${label}: ${path}")
  endif()
endfunction()

function(require_exact_generation generation_root label expected_generation)
  set(expected ${ARGN})
  list(SORT expected)
  set(expected_members ${expected} manifest.v1)
  list(SORT expected_members)
  file(GLOB generation_members RELATIVE "${generation_root}"
       LIST_DIRECTORIES TRUE "${generation_root}/*" "${generation_root}/.*")
  list(SORT generation_members)
  if(NOT "${generation_members}" STREQUAL "${expected_members}")
    message(FATAL_ERROR
      "${label} generation member census mismatch: expected=${expected_members};observed=${generation_members}")
  endif()
  foreach(member IN LISTS generation_members)
    if(IS_DIRECTORY "${generation_root}/${member}" OR
       IS_SYMLINK "${generation_root}/${member}")
      message(FATAL_ERROR
        "${label} generation member is not a regular file: ${member}")
    endif()
  endforeach()

  set(manifest_path "${generation_root}/manifest.v1")
  file(STRINGS "${manifest_path}" manifest_lines ENCODING UTF-8)
  list(LENGTH manifest_lines manifest_line_count)
  list(LENGTH expected expected_count)
  math(EXPR expected_line_count "${expected_count} + 2")
  if(NOT manifest_line_count EQUAL expected_line_count)
    message(FATAL_ERROR
      "${label} manifest record census mismatch: expected=${expected_line_count};observed=${manifest_line_count}")
  endif()
  list(GET manifest_lines 0 manifest_header)
  string(ASCII 9 tab)
  list(GET manifest_lines 1 manifest_generation)
  if(NOT manifest_header STREQUAL "DG_IGR_DURABLE_GENERATION_V1" OR
     NOT manifest_generation STREQUAL
         "generation${tab}${expected_generation}")
    message(FATAL_ERROR "${label} manifest header/generation is malformed")
  endif()

  set(manifest_artifacts)
  math(EXPR last_manifest_index "${manifest_line_count} - 1")
  foreach(index RANGE 2 ${last_manifest_index})
    list(GET manifest_lines ${index} manifest_line)
    string(REPLACE "${tab}" ";" manifest_fields "${manifest_line}")
    list(LENGTH manifest_fields manifest_field_count)
    if(NOT manifest_field_count EQUAL 4)
      message(FATAL_ERROR "${label} manifest record is malformed")
    endif()
    list(GET manifest_fields 0 manifest_kind)
    list(GET manifest_fields 1 manifest_name)
    list(GET manifest_fields 2 manifest_bytes)
    list(GET manifest_fields 3 manifest_sha256)
    list(FIND expected "${manifest_name}" expected_index)
    list(FIND manifest_artifacts "${manifest_name}" duplicate_index)
    string(LENGTH "${manifest_sha256}" manifest_sha256_length)
    if(NOT manifest_kind STREQUAL "artifact" OR expected_index LESS 0 OR
       duplicate_index GREATER_EQUAL 0 OR
       NOT manifest_bytes MATCHES "^(0|[1-9][0-9]*)$" OR
       NOT manifest_sha256_length EQUAL 64 OR
       NOT manifest_sha256 MATCHES "^[0-9a-f]+$")
      message(FATAL_ERROR
        "${label} manifest has an unauthorized, duplicate, or malformed record: ${manifest_line}")
    endif()
    set(artifact_path "${generation_root}/${manifest_name}")
    file(SIZE "${artifact_path}" artifact_bytes)
    file(SHA256 "${artifact_path}" artifact_sha256)
    if(NOT "${manifest_bytes}" STREQUAL "${artifact_bytes}" OR
       NOT "${manifest_sha256}" STREQUAL "${artifact_sha256}")
      message(FATAL_ERROR
        "${label} manifest length/digest mismatch: ${manifest_name}")
    endif()
    list(APPEND manifest_artifacts "${manifest_name}")
  endforeach()
  list(SORT manifest_artifacts)
  if(NOT "${manifest_artifacts}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${label} manifest is missing an exact generation member")
  endif()
endfunction()

function(require_manifest_artifact generation_root artifact label)
  set(manifest_path "${generation_root}/manifest.v1")
  set(artifact_path "${generation_root}/${artifact}")
  require_regular_file("${manifest_path}" "${label} manifest")
  require_regular_file("${artifact_path}" "${label} artifact")
  file(STRINGS "${manifest_path}" manifest_lines ENCODING UTF-8)
  string(ASCII 9 tab)
  set(matches 0)
  foreach(manifest_line IN LISTS manifest_lines)
    string(REPLACE "${tab}" ";" manifest_fields "${manifest_line}")
    list(LENGTH manifest_fields manifest_field_count)
    if(manifest_field_count EQUAL 4)
      list(GET manifest_fields 0 manifest_kind)
      list(GET manifest_fields 1 manifest_name)
      if(manifest_kind STREQUAL "artifact" AND
         manifest_name STREQUAL artifact)
        list(GET manifest_fields 2 manifest_bytes)
        list(GET manifest_fields 3 manifest_sha256)
        file(SIZE "${artifact_path}" artifact_bytes)
        file(SHA256 "${artifact_path}" artifact_sha256)
        string(LENGTH "${manifest_sha256}" manifest_sha256_length)
        if(NOT manifest_bytes MATCHES "^(0|[1-9][0-9]*)$" OR
           NOT manifest_sha256_length EQUAL 64 OR
           NOT manifest_sha256 MATCHES "^[0-9a-f]+$" OR
           NOT "${manifest_bytes}" STREQUAL "${artifact_bytes}" OR
           NOT "${manifest_sha256}" STREQUAL "${artifact_sha256}")
          message(FATAL_ERROR "${label} manifest record is invalid")
        endif()
        math(EXPR matches "${matches} + 1")
      endif()
    endif()
  endforeach()
  if(NOT matches EQUAL 1)
    message(FATAL_ERROR
      "${label} requires exactly one structurally valid manifest record")
  endif()
endfunction()

function(require_exact_logical_artifacts root label expected_step0
                                           expected_step10)
  if(NOT ARGC EQUAL 4 AND NOT ARGC EQUAL 5)
    message(FATAL_ERROR
      "${label} durable oracle requires step-0/10 and optional step-20 sets")
  endif()
  if(ARGC EQUAL 5)
    set(expected "${ARGV4}")
    set(expected_generation_count 3)
  else()
    set(expected "${expected_step10}")
    set(expected_generation_count 2)
  endif()
  list(SORT expected)
  file(GLOB observed RELATIVE "${root}" LIST_DIRECTORIES FALSE "${root}/*")
  list(SORT observed)
  if(NOT "${observed}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${label} logical artifact census mismatch: expected=${expected};observed=${observed}")
  endif()
  foreach(artifact IN LISTS observed)
    if(NOT IS_SYMLINK "${root}/${artifact}")
      message(FATAL_ERROR "${label} logical artifact is not an alias: ${artifact}")
    endif()
    file(READ_SYMLINK "${root}/${artifact}" artifact_target)
    if(NOT artifact_target STREQUAL ".dg-igr-current/${artifact}")
      message(FATAL_ERROR
        "${label} logical alias target is invalid: ${artifact_target}")
    endif()
  endforeach()
  file(GLOB hidden RELATIVE "${root}" LIST_DIRECTORIES TRUE "${root}/.*")
  list(SORT hidden)
  set(expected_hidden .dg-igr-current .dg-igr-generations)
  if(NOT "${hidden}" STREQUAL "${expected_hidden}" OR
     NOT IS_SYMLINK "${root}/.dg-igr-current" OR
     NOT IS_DIRECTORY "${root}/.dg-igr-generations" OR
     IS_SYMLINK "${root}/.dg-igr-generations")
    message(FATAL_ERROR
      "${label} hidden durable census mismatch: expected=${expected_hidden};observed=${hidden}")
  endif()

  set(generations_root "${root}/.dg-igr-generations")
  file(GLOB generations RELATIVE "${generations_root}"
       LIST_DIRECTORIES TRUE "${generations_root}/*" "${generations_root}/.*")
  list(SORT generations)
  list(LENGTH generations generation_count)
  if(NOT generation_count EQUAL expected_generation_count)
    message(FATAL_ERROR
      "${label} immutable generation census mismatch: expected=${expected_generation_count};observed=${generation_count}")
  endif()
  set(expected_steps 0 10 20)
  foreach(index RANGE 0 2)
    if(index GREATER_EQUAL expected_generation_count)
      break()
    endif()
    list(GET generations ${index} generation)
    list(GET expected_steps ${index} expected_generation)
    if(NOT generation MATCHES
        "^generation-${expected_generation}-[0-9]+-[0-9]+$")
      message(FATAL_ERROR
        "${label} generation identity/order is invalid: ${generation}")
    endif()
    set(generation_root "${generations_root}/${generation}")
    if(NOT IS_DIRECTORY "${generation_root}" OR IS_SYMLINK "${generation_root}")
      message(FATAL_ERROR "${label} generation is not a directory: ${generation}")
    endif()
    if(index EQUAL 0)
      set(expected_prefix "${expected_step0}")
    elseif(index EQUAL 1)
      set(expected_prefix "${expected_step10}")
    else()
      set(expected_prefix "${ARGV4}")
    endif()
    require_exact_generation("${generation_root}" "${label} ${generation}"
                             ${expected_generation} ${expected_prefix})
  endforeach()

  math(EXPR current_index "${expected_generation_count} - 1")
  list(GET generations ${current_index} current_generation_name)
  file(READ_SYMLINK "${root}/.dg-igr-current" current_generation)
  set(expected_current ".dg-igr-generations/${current_generation_name}")
  file(REAL_PATH "${root}/.dg-igr-current" resolved_current)
  file(REAL_PATH "${generations_root}/${current_generation_name}"
       resolved_expected_current)
  if(NOT current_generation STREQUAL expected_current OR
     NOT resolved_current STREQUAL resolved_expected_current)
    message(FATAL_ERROR
      "${label} current marker does not resolve to the final enumerated generation")
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

function(validate_scalar_cuda_metadata metadata label expected_step scenario)
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
     NOT metadata MATCHES "\"scenario\": \"${scenario}\"" OR
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
     NOT metadata MATCHES
         "\"zero_post_baseline_cuda_umpire_allocation\": true" OR
     NOT metadata MATCHES "\"device_residency\": true" OR
     NOT metadata MATCHES "\"host_staged_same_gpu\": true")
    message(FATAL_ERROR "${label} CUDA scalar-IGR evidence is incomplete")
  endif()
  string(REGEX MATCH "\"gpu_uuid\": \"GPU-([0-9a-f]+)\""
         uuid_match "${metadata}")
  string(LENGTH "${CMAKE_MATCH_1}" uuid_hex_length)
  if(NOT uuid_hex_length EQUAL 32)
    message(FATAL_ERROR "${label} CUDA scalar-IGR GPU UUID is not 32 hex digits")
  endif()
  foreach(hash_field IN ITEMS initial_state_sha256 final_interior_sha256
                              final_trace_sha256)
    string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
           hash_match "${metadata}")
    string(LENGTH "${CMAKE_MATCH_1}" hash_length)
    if(NOT hash_length EQUAL 64)
      message(FATAL_ERROR
        "${label} CUDA scalar-IGR ${hash_field} is not 64 hex digits")
    endif()
  endforeach()
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

if(HDG_RUNTIME_W8C_SMOOTH_RANK2)
  if(NOT HDG_RUNTIME_RANKS EQUAL 2 OR
     NOT DEFINED HDG_RUNTIME_SCALAR_VTK_TEST)
    message(FATAL_ERROR "W8C scalar rank-two test requires its comparator")
  endif()
  if(NOT DEFINED HDG_RUNTIME_BACKEND)
    set(HDG_RUNTIME_BACKEND cpu)
  endif()
  if(HDG_RUNTIME_BACKEND STREQUAL "cpu")
    set(w8c_device none)
  elseif(HDG_RUNTIME_BACKEND STREQUAL "cuda")
    if(NOT DEFINED HDG_RUNTIME_DEVICE_ID OR
       NOT HDG_RUNTIME_DEVICE_ID EQUAL 0)
      message(FATAL_ERROR "W8C CUDA rank-two test requires device 0")
    endif()
    set(w8c_device 0)
  else()
    message(FATAL_ERROR "W8C rank-two test admits only CPU or CUDA")
  endif()
  set(w8c_common
    --card exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3
    --case periodic-smooth-transition-v2
    --space l2 --mesh-family periodic-cartesian-quad --mesh-n 8 --order 1
    --dt 0.0009765625 --steps 64 --final-time 0.0625
    --backend "${HDG_RUNTIME_BACKEND}" --device-id "${w8c_device}"
    --output-steps 0,32,64
    --seed 0x573843534d4f5632)
  set(rank2_directory "${HDG_RUNTIME_TEST_OUTPUT}-rank2")
  set(split_directory "${HDG_RUNTIME_TEST_OUTPUT}-split")
  set(split_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  if(HDG_RUNTIME_BACKEND STREQUAL "cpu")
    set(serial_directory "${HDG_RUNTIME_TEST_OUTPUT}-serial")
    file(REMOVE_RECURSE "${serial_directory}")
  endif()
  file(REMOVE_RECURSE "${rank2_directory}" "${split_directory}")
  file(REMOVE "${split_checkpoint}")
  if(HDG_RUNTIME_BACKEND STREQUAL "cpu")
    run_serial_runtime(serial ${w8c_common}
      --output-dir "${serial_directory}" --checkpoint-out "${serial_directory}.checkpoint"
      --checkpoint-step 32 --stop-after-checkpoint false --restart-from none
      --max-retries 0 --inject-failure off)
    if(NOT serial_status EQUAL 0 OR NOT serial_output MATCHES "status=Success.*step=64.*retries=0" OR NOT serial_error STREQUAL "")
      message(FATAL_ERROR "W8C scalar serial reference failed: ${serial_status};${serial_output};${serial_error}")
    endif()
  endif()
  run_runtime(rank2 ${w8c_common}
    --output-dir "${rank2_directory}" --checkpoint-out "${rank2_directory}.checkpoint"
    --checkpoint-step 32 --stop-after-checkpoint false --restart-from none
    --max-retries 0 --inject-failure off)
  if(NOT rank2_status EQUAL 0 OR NOT rank2_output MATCHES "status=Success.*step=64.*retries=0" OR NOT rank2_error STREQUAL "")
    message(FATAL_ERROR "W8C scalar rank-two reference failed: ${rank2_status};${rank2_output};${rank2_error}")
  endif()
  run_runtime(writer ${w8c_common}
    --output-dir "${split_directory}" --checkpoint-out "${split_checkpoint}"
    --checkpoint-step 32 --stop-after-checkpoint true --restart-from none
    --max-retries 1 --inject-failure AfterRecoveryBeforeCommit:7:1)
  if(NOT writer_status EQUAL 0 OR NOT writer_output MATCHES "status=Success.*step=32.*retries=1" OR NOT writer_error STREQUAL "")
    message(FATAL_ERROR "W8C scalar rank-two retry writer failed: ${writer_status};${writer_output};${writer_error}")
  endif()
  run_runtime(reader ${w8c_common}
    --output-dir "${split_directory}" --checkpoint-out none --checkpoint-step none
    --stop-after-checkpoint false --restart-from "${split_checkpoint}"
    --max-retries 0 --inject-failure off)
  if(NOT reader_status EQUAL 0 OR NOT reader_output MATCHES "status=Success.*step=64.*retries=1" OR NOT reader_error STREQUAL "")
    message(FATAL_ERROR "W8C scalar rank-two restart reader failed: ${reader_status};${reader_output};${reader_error}")
  endif()
  foreach(directory IN ITEMS rank2_directory split_directory)
    foreach(step IN ITEMS 000000 000032 000064)
      validate_scalar_igr_vtk("${${directory}}/state-step-${step}.vtk" 8
        "W8C scalar ${HDG_RUNTIME_BACKEND} ${directory} step ${step}")
    endforeach()
  endforeach()
  foreach(step IN ITEMS 000000 000032 000064)
    if(HDG_RUNTIME_BACKEND STREQUAL "cpu")
      execute_process(COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk-w8c-rank2
                      "${serial_directory}/state-step-${step}.vtk"
                      "${rank2_directory}/state-step-${step}.vtk" 5e-11
                      RESULT_VARIABLE compare_status OUTPUT_VARIABLE compare_output ERROR_VARIABLE compare_error)
      set(parity_label "serial/rank-two")
    else()
      execute_process(COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk-w8c-rank2
                      "${rank2_directory}/state-step-${step}.vtk"
                      "${split_directory}/state-step-${step}.vtk" 5e-11
                      RESULT_VARIABLE compare_status OUTPUT_VARIABLE compare_output ERROR_VARIABLE compare_error)
      set(parity_label "uninterrupted/restart")
    endif()
    if(NOT compare_status EQUAL 0 OR NOT compare_output MATCHES "HDG scalar-IGR W8C rank-two VTK parity PASS" OR NOT compare_error STREQUAL "")
      message(FATAL_ERROR "W8C ${parity_label} field-topology comparator failed at ${step}: ${compare_status};${compare_output};${compare_error}")
    endif()
  endforeach()
  if(HDG_RUNTIME_BACKEND STREQUAL "cpu")
    execute_process(COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk-w8c-rank2
                    "${rank2_directory}/state-step-000064.vtk"
                    "${split_directory}/state-step-000064.vtk" 5e-11
                    RESULT_VARIABLE split_compare_status
                    OUTPUT_VARIABLE split_compare_output
                    ERROR_VARIABLE split_compare_error)
    if(NOT split_compare_status EQUAL 0 OR
       NOT split_compare_output MATCHES "HDG scalar-IGR W8C rank-two VTK parity PASS" OR
       NOT split_compare_error STREQUAL "")
      message(FATAL_ERROR "W8C rank-two split-reader field-topology comparator failed: ${split_compare_status};${split_compare_output};${split_compare_error}")
    endif()
  endif()
  if(HDG_RUNTIME_BACKEND STREQUAL "cuda")
    foreach(path IN ITEMS
        "${rank2_directory}/run-metadata.json"
        "${split_directory}/run-metadata-checkpoint.json"
        "${split_directory}/run-metadata.json"
        "${split_checkpoint}")
      require_regular_file("${path}" "W8C CUDA publication")
    endforeach()
    file(READ "${rank2_directory}/run-metadata.json" w8c_reference_metadata)
    file(READ "${split_directory}/run-metadata-checkpoint.json" w8c_writer_metadata)
    file(READ "${split_directory}/run-metadata.json" w8c_reader_metadata)
    validate_scalar_cuda_metadata("${w8c_reference_metadata}" "W8C CUDA uninterrupted" 64 "W8C-V2-N8-DT10")
    validate_scalar_cuda_metadata("${w8c_writer_metadata}" "W8C CUDA split writer" 32 "W8C-V2-N8-DT10")
    validate_scalar_cuda_metadata("${w8c_reader_metadata}" "W8C CUDA split reader" 64 "W8C-V2-N8-DT10")
    if(NOT w8c_writer_metadata MATCHES "\"retries\": 1" OR
       NOT w8c_reader_metadata MATCHES "\"retries\": 1")
      message(FATAL_ERROR "W8C CUDA retry evidence is incomplete")
    endif()
    foreach(hash_field IN ITEMS final_interior_sha256 final_trace_sha256)
      string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
             w8c_reference_hash_match "${w8c_reference_metadata}")
      set(w8c_reference_hash "${CMAKE_MATCH_1}")
      string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
             w8c_reader_hash_match "${w8c_reader_metadata}")
      if(w8c_reference_hash STREQUAL "" OR
         NOT w8c_reference_hash STREQUAL "${CMAKE_MATCH_1}")
        message(FATAL_ERROR "W8C CUDA ${hash_field} restart parity failed")
      endif()
    endforeach()
  endif()
  return()
endif()

if(HDG_RUNTIME_W8C_MATRIX)
  if(NOT HDG_RUNTIME_RANKS EQUAL 1)
    message(FATAL_ERROR "W8C smooth-transition matrix is CPU serial")
  endif()
  if(NOT DEFINED HDG_RUNTIME_MATRIX_COMPARATOR)
    message(FATAL_ERROR "W8C smooth-transition matrix comparator is required")
  endif()
  set(matrix_comparator_arguments)
  foreach(row IN ITEMS n16dt11 n16dt12 n32dt12)
    if(row STREQUAL "n16dt11")
      set(matrix_n 16)
      set(matrix_dt 0.00048828125)
      set(matrix_steps 128)
      set(matrix_checkpoint 64)
      set(matrix_outputs 0,64,128)
      set(matrix_steps_padded 000000 000016 000032 000048 000064 000080 000096 000112 000128)
    elseif(row STREQUAL "n16dt12")
      set(matrix_n 16)
      set(matrix_dt 0.000244140625)
      set(matrix_steps 256)
      set(matrix_checkpoint 128)
      set(matrix_outputs 0,128,256)
      set(matrix_steps_padded 000000 000032 000064 000096 000128 000160 000192 000224 000256)
    else()
      set(matrix_n 32)
      set(matrix_dt 0.000244140625)
      set(matrix_steps 256)
      set(matrix_checkpoint 128)
      set(matrix_outputs 0,128,256)
      set(matrix_steps_padded 000000 000032 000064 000096 000128 000160 000192 000224 000256)
    endif()
    foreach(policy IN ITEMS conventional scalar)
      set(matrix_directory "${HDG_RUNTIME_TEST_OUTPUT}-${row}-${policy}")
      set(matrix_checkpoint_path "${matrix_directory}.checkpoint")
      file(REMOVE_RECURSE "${matrix_directory}")
      file(REMOVE "${matrix_checkpoint_path}")
      if(policy STREQUAL "scalar")
        set(matrix_card exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3)
      else()
        set(matrix_card exp.hdg.euler.conventional.smooth_persistent.w8a.v1)
      endif()
      run_serial_runtime(matrix_run
        --card "${matrix_card}" --case periodic-smooth-transition-v2
        --space l2 --mesh-family periodic-cartesian-quad --mesh-n "${matrix_n}" --order 1
        --dt "${matrix_dt}" --steps "${matrix_steps}" --final-time 0.0625
        --backend cpu --device-id none --output-steps "${matrix_outputs}"
        --output-dir "${matrix_directory}" --checkpoint-out "${matrix_checkpoint_path}"
        --checkpoint-step "${matrix_checkpoint}" --stop-after-checkpoint false
        --restart-from none --max-retries 0 --inject-failure off
        --seed 0x573843534d4f5632)
      if(NOT matrix_run_status EQUAL 0 OR
         NOT matrix_run_output MATCHES "status=Success.*step=${matrix_steps}.*retries=0" OR
         NOT matrix_run_error STREQUAL "")
        message(FATAL_ERROR "W8C matrix ${row}/${policy} failed: ${matrix_run_status};${matrix_run_output};${matrix_run_error}")
      endif()
      foreach(step IN LISTS matrix_steps_padded)
        require_regular_file("${matrix_directory}/state-step-${step}.vtk"
          "W8C matrix ${row}/${policy} output ${step}")
        if(policy STREQUAL "scalar")
          validate_scalar_igr_vtk("${matrix_directory}/state-step-${step}.vtk" "${matrix_n}"
            "W8C matrix ${row}/${policy} output ${step}")
        else()
          validate_physical_vtk("${matrix_directory}/state-step-${step}.vtk" "${matrix_n}"
            "W8C matrix ${row}/${policy} output ${step}")
        endif()
      endforeach()
      file(GLOB matrix_states "${matrix_directory}/state-step-*.vtk")
      list(LENGTH matrix_states matrix_state_count)
      if(NOT matrix_state_count EQUAL 9)
        message(FATAL_ERROR "W8C matrix ${row}/${policy} output census is ${matrix_state_count}, expected 9")
      endif()
      require_regular_file("${matrix_checkpoint_path}" "W8C matrix checkpoint")
      list(APPEND matrix_comparator_arguments "${matrix_directory}")
    endforeach()
  endforeach()
  execute_process(
    COMMAND "${HDG_RUNTIME_MATRIX_COMPARATOR}" --matrix ${matrix_comparator_arguments}
    RESULT_VARIABLE matrix_comparator_status
    OUTPUT_VARIABLE matrix_comparator_output
    ERROR_VARIABLE matrix_comparator_error)
  if(NOT matrix_comparator_status EQUAL 0 OR
     NOT matrix_comparator_error STREQUAL "" OR
     NOT matrix_comparator_output MATCHES "q18-matrix-disposition=")
    message(FATAL_ERROR
      "W8C matrix q18 comparator failed: ${matrix_comparator_status};${matrix_comparator_output};${matrix_comparator_error}")
  endif()
  message(STATUS "W8C matrix q18 comparator: ${matrix_comparator_output}")
  return()
endif()

if(HDG_RUNTIME_W8C_SMOOTH)
  if(NOT HDG_RUNTIME_RANKS EQUAL 1)
    message(FATAL_ERROR "W8C smooth-transition CLI coverage is CPU serial")
  endif()
  foreach(policy IN ITEMS conventional scalar)
    set(reference_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-${policy}-reference")
    set(reference_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-${policy}-reference.checkpoint")
    set(split_directory "${HDG_RUNTIME_TEST_OUTPUT}-${policy}-split")
    set(split_checkpoint
        "${HDG_RUNTIME_TEST_OUTPUT}-${policy}-split.checkpoint")
    file(REMOVE_RECURSE "${reference_directory}" "${split_directory}")
    file(REMOVE "${reference_checkpoint}" "${split_checkpoint}")

    if(policy STREQUAL "scalar")
      set(card exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3)
    else()
      set(card exp.hdg.euler.conventional.smooth_persistent.w8a.v1)
    endif()
    set(common
      --card "${card}" --case periodic-smooth-transition-v2
      --space l2 --mesh-family periodic-cartesian-quad --mesh-n 8 --order 1
      --dt 0.0009765625 --steps 64 --final-time 0.0625
      --backend cpu --device-id none --output-steps 0,32,64
      --max-retries 0 --inject-failure off --seed 0x573843534d4f5632)

    run_serial_runtime(reference ${common}
      --output-dir "${reference_directory}"
      --checkpoint-out "${reference_checkpoint}" --checkpoint-step 32
      --stop-after-checkpoint false --restart-from none)
    if(NOT reference_status EQUAL 0 OR
       NOT reference_output MATCHES "status=Success.*step=64.*retries=0" OR
       NOT reference_error STREQUAL "")
      message(FATAL_ERROR
        "W8C smooth ${policy} uninterrupted launch failed: ${reference_status};${reference_output};${reference_error}")
    endif()

    run_serial_runtime(writer ${common}
      --output-dir "${split_directory}"
      --checkpoint-out "${split_checkpoint}" --checkpoint-step 32
      --stop-after-checkpoint true --restart-from none)
    if(NOT writer_status EQUAL 0 OR
       NOT writer_output MATCHES "status=Success.*step=32.*retries=0" OR
       NOT writer_error STREQUAL "")
      message(FATAL_ERROR
        "W8C smooth ${policy} split-writer launch failed: ${writer_status};${writer_output};${writer_error}")
    endif()

    run_serial_runtime(reader ${common}
      --output-dir "${split_directory}"
      --checkpoint-out none --checkpoint-step none
      --stop-after-checkpoint false --restart-from "${split_checkpoint}")
    if(NOT reader_status EQUAL 0 OR
       NOT reader_output MATCHES "status=Success.*step=64.*retries=0" OR
       NOT reader_error STREQUAL "")
      message(FATAL_ERROR
        "W8C smooth ${policy} split-reader launch failed: ${reader_status};${reader_output};${reader_error}")
    endif()

    foreach(directory IN ITEMS reference_directory split_directory)
      foreach(step IN ITEMS 000000 000032 000064)
        require_regular_file(
          "${${directory}}/state-step-${step}.vtk"
          "W8C smooth ${policy} ${directory} step ${step}")
      endforeach()
      file(GLOB states "${${directory}}/state-step-*.vtk")
      list(LENGTH states state_count)
      if(NOT state_count EQUAL 3)
        message(FATAL_ERROR
          "W8C smooth ${policy} ${directory} output census is ${state_count}, expected 3")
      endif()
    endforeach()
    foreach(path IN ITEMS
        "${reference_directory}/run-metadata.json"
        "${reference_checkpoint}"
        "${split_directory}/run-metadata-checkpoint.json"
        "${split_directory}/run-metadata.json"
        "${split_checkpoint}")
      require_regular_file("${path}" "W8C smooth ${policy} publication")
    endforeach()

    if(policy STREQUAL "scalar")
      foreach(directory IN ITEMS reference_directory split_directory)
        foreach(step IN ITEMS 000000 000032 000064)
          validate_scalar_igr_vtk(
            "${${directory}}/state-step-${step}.vtk" 8
            "W8C smooth scalar ${directory} step ${step}")
        endforeach()
      endforeach()
    else()
      foreach(directory IN ITEMS reference_directory split_directory)
        foreach(step IN ITEMS 000000 000032 000064)
          validate_physical_vtk(
            "${${directory}}/state-step-${step}.vtk" 8
            "W8C smooth conventional ${directory} step ${step}")
        endforeach()
      endforeach()
    endif()

    file(READ "${reference_directory}/run-metadata.json"
         reference_metadata)
    file(READ "${split_directory}/run-metadata-checkpoint.json"
         writer_metadata)
    file(READ "${split_directory}/run-metadata.json" reader_metadata)
    foreach(metadata IN ITEMS reference_metadata writer_metadata
                              reader_metadata)
      if(NOT "${${metadata}}" MATCHES
             "\"case\": \"periodic-smooth-transition-v2\"" OR
         NOT "${${metadata}}" MATCHES "\"scenario\": \"W8C-V2-N8-DT10\"" OR
         NOT "${${metadata}}" MATCHES "\"status\": \"Success\"" OR
         NOT "${${metadata}}" MATCHES "\"retries\": 0")
        message(FATAL_ERROR
          "W8C smooth ${policy} ${metadata} identity/status is incomplete")
      endif()
    endforeach()
    if(NOT writer_metadata MATCHES "\"committed_step\": 32" OR
       NOT reference_metadata MATCHES "\"committed_step\": 64" OR
       NOT reader_metadata MATCHES "\"committed_step\": 64")
      message(FATAL_ERROR
        "W8C smooth ${policy} metadata does not bind steps 32/64")
    endif()
    string(REGEX MATCH
      "\"final_state_sha256\": \"([0-9a-f]+)\""
      reference_hash_match "${reference_metadata}")
    set(reference_hash "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "\"final_state_sha256\": \"([0-9a-f]+)\""
      reader_hash_match "${reader_metadata}")
    set(reader_hash "${CMAKE_MATCH_1}")
    string(LENGTH "${reference_hash}" reference_hash_length)
    if(NOT reference_hash_length EQUAL 64 OR
       NOT reference_hash STREQUAL reader_hash)
      message(FATAL_ERROR
        "W8C smooth ${policy} uninterrupted/restart final-state identity differs")
    endif()
  endforeach()
  return()
endif()

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
  set(baseline_directory "${HDG_RUNTIME_TEST_OUTPUT}-baseline")
  set(baseline_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}-baseline.checkpoint")
  set(reference_directory "${HDG_RUNTIME_TEST_OUTPUT}-reference")
  # The frozen same-tuple restart contract binds a split trajectory to one
  # durable output identity. The reader resumes publication in that directory.
  set(writer_directory "${HDG_RUNTIME_TEST_OUTPUT}-split")
  set(reader_directory "${writer_directory}")
  set(reference_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}-reference.checkpoint")
  set(split_checkpoint "${HDG_RUNTIME_TEST_OUTPUT}.checkpoint")
  file(REMOVE_RECURSE "${baseline_directory}" "${reference_directory}" "${writer_directory}"
                      "${reader_directory}")
  file(REMOVE "${baseline_checkpoint}" "${reference_checkpoint}"
              "${split_checkpoint}")

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

  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_BACKEND STREQUAL "cuda")
    run_runtime(baseline ${functional_common}
      --output-dir "${baseline_directory}"
      --checkpoint-out "${baseline_checkpoint}" --checkpoint-step 10
      --stop-after-checkpoint false --restart-from none
      --max-retries 0 --inject-failure off)
    if(NOT baseline_status EQUAL 0 OR
       NOT baseline_output MATCHES "status=Success.*step=20.*retries=0" OR
       NOT baseline_error STREQUAL "")
      message(FATAL_ERROR
        "fault-free CUDA scalar baseline failed: ${baseline_status};${baseline_output};${baseline_error}")
    endif()
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

  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_RANKS EQUAL 2 AND
     HDG_RUNTIME_BACKEND STREQUAL "cuda")
    if(NOT DEFINED HDG_RUNTIME_SCALAR_VTK_TEST OR
       "$ENV{DG_IGR_SC25_R2_HANDOFF_DIR}" STREQUAL "" OR
       "$ENV{DG_IGR_SC25_R2_RUN_ID}" STREQUAL "" OR
       "$ENV{DG_IGR_SC25_R2_COMPLETE_SHA256}" STREQUAL "" OR
       "$ENV{DG_IGR_SC25_R2_RANK0_SHA256}" STREQUAL "" OR
       "$ENV{DG_IGR_SC25_R2_RANK1_SHA256}" STREQUAL "")
      message(FATAL_ERROR
        "rank-2 CUDA scalar artifacts require the hardened CPU handoff consumer inputs")
    endif()
    execute_process(
      COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}"
        --validate-scalar-rank2-artifacts
        "$ENV{DG_IGR_SC25_R2_HANDOFF_DIR}"
        "$ENV{DG_IGR_SC25_R2_RUN_ID}"
        "$ENV{DG_IGR_SC25_R2_COMPLETE_SHA256}"
        "$ENV{DG_IGR_SC25_R2_RANK0_SHA256}"
        "$ENV{DG_IGR_SC25_R2_RANK1_SHA256}"
        "${writer_directory}" "${split_checkpoint}"
      RESULT_VARIABLE artifact_binding_status
      OUTPUT_VARIABLE artifact_binding_output
      ERROR_VARIABLE artifact_binding_error)
    if(NOT artifact_binding_status EQUAL 0 OR
       NOT artifact_binding_output MATCHES
           "SCALAR_RANK2_ARTIFACT_BINDING_PASS.*vtk0_sha256=[0-9a-f][0-9a-f]+.*vtk10_sha256=[0-9a-f][0-9a-f]+.*checkpoint_sha256=[0-9a-f][0-9a-f]+.*sidecar_sha256=[0-9a-f][0-9a-f]+.*metadata_sha256=[0-9a-f][0-9a-f]+" OR
       NOT artifact_binding_error STREQUAL "")
      message(FATAL_ERROR
        "rank-2 CUDA CLI artifacts were not bound to the CPU manifest: ${artifact_binding_status};${artifact_binding_output};${artifact_binding_error}")
    endif()
    foreach(binding_field IN ITEMS vtk0_sha256 vtk10_sha256
                                   checkpoint_sha256 sidecar_sha256
                                   metadata_sha256)
      string(REGEX MATCH "${binding_field}=([0-9a-f]+)"
             binding_match "${artifact_binding_output}")
      string(LENGTH "${CMAKE_MATCH_1}" binding_length)
      if(NOT binding_length EQUAL 64)
        message(FATAL_ERROR
          "rank-2 CUDA ${binding_field} binding is not exactly 64 hex digits")
      endif()
    endforeach()

    foreach(sidecar_case IN ITEMS count-zero count-three count-maximum
                                  checksum shape)
      set(sidecar_negative_directory
          "${HDG_RUNTIME_TEST_OUTPUT}-sidecar-${sidecar_case}")
      file(REMOVE_RECURSE "${sidecar_negative_directory}")
      execute_process(
        COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}"
          --make-scalar-sidecar-negative "${writer_directory}"
          "${sidecar_negative_directory}" "${sidecar_case}"
        RESULT_VARIABLE sidecar_builder_status
        OUTPUT_VARIABLE sidecar_builder_output
        ERROR_VARIABLE sidecar_builder_error)
      if(NOT sidecar_builder_status EQUAL 0 OR
         NOT sidecar_builder_error STREQUAL "")
        message(FATAL_ERROR
          "malformed sidecar ${sidecar_case} fixture failed: ${sidecar_builder_output};${sidecar_builder_error}")
      endif()
      set(sidecar_negative_checkpoint
          "${sidecar_negative_directory}/restart-checkpoint.v2")
      file(READ_SYMLINK "${sidecar_negative_directory}/.dg-igr-current"
                        sidecar_current_before)
      file(SHA256 "${sidecar_negative_checkpoint}"
                  sidecar_checkpoint_hash_before)
      file(SHA256
        "${sidecar_negative_directory}/checkpoint-runtime-identity.v1"
        sidecar_identity_hash_before)
      file(GLOB sidecar_generations_before
        "${sidecar_negative_directory}/.dg-igr-generations/generation-*")
      list(LENGTH sidecar_generations_before sidecar_generation_count_before)
      run_runtime(sidecar_negative ${functional_common}
        --output-dir "${sidecar_negative_directory}"
        --checkpoint-out none --checkpoint-step none
        --stop-after-checkpoint false
        --restart-from "${sidecar_negative_checkpoint}"
        --max-retries 0 --inject-failure off)
      file(READ_SYMLINK "${sidecar_negative_directory}/.dg-igr-current"
                        sidecar_current_after)
      file(SHA256 "${sidecar_negative_checkpoint}"
                  sidecar_checkpoint_hash_after)
      file(SHA256
        "${sidecar_negative_directory}/checkpoint-runtime-identity.v1"
        sidecar_identity_hash_after)
      file(GLOB sidecar_generations_after
        "${sidecar_negative_directory}/.dg-igr-generations/generation-*")
      list(LENGTH sidecar_generations_after sidecar_generation_count_after)
      if(NOT sidecar_negative_status EQUAL 66 OR
         NOT sidecar_negative_error MATCHES "reason=restart_incompatible" OR
         NOT sidecar_current_before STREQUAL sidecar_current_after OR
         NOT sidecar_checkpoint_hash_before STREQUAL
             sidecar_checkpoint_hash_after OR
         NOT sidecar_identity_hash_before STREQUAL sidecar_identity_hash_after OR
         NOT sidecar_generation_count_before EQUAL
             sidecar_generation_count_after)
        message(FATAL_ERROR
          "malformed sidecar ${sidecar_case} did not fail cleanly without drift: ${sidecar_negative_status};${sidecar_negative_output};${sidecar_negative_error}")
      endif()
    endforeach()
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

  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_RANKS EQUAL 2 AND
     HDG_RUNTIME_BACKEND STREQUAL "cuda")
    foreach(trajectory_directory IN ITEMS baseline_directory
                                          reference_directory writer_directory)
      execute_process(
        COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}"
          --validate-scalar-rank2-artifacts
          "$ENV{DG_IGR_SC25_R2_HANDOFF_DIR}"
          "$ENV{DG_IGR_SC25_R2_RUN_ID}"
          "$ENV{DG_IGR_SC25_R2_COMPLETE_SHA256}"
          "$ENV{DG_IGR_SC25_R2_RANK0_SHA256}"
          "$ENV{DG_IGR_SC25_R2_RANK1_SHA256}"
          "${${trajectory_directory}}" none
        RESULT_VARIABLE trajectory_binding_status
        OUTPUT_VARIABLE trajectory_binding_output
        ERROR_VARIABLE trajectory_binding_error)
      if(NOT trajectory_binding_status EQUAL 0 OR
         NOT trajectory_binding_output MATCHES
             "SCALAR_RANK2_TRAJECTORY_BINDING_PASS.*vtk0_sha256=[0-9a-f]+.*vtk10_sha256=[0-9a-f]+.*vtk20_sha256=[0-9a-f]+.*metadata_sha256=[0-9a-f]+" OR
         NOT trajectory_binding_error STREQUAL "")
        message(FATAL_ERROR
          "rank-2 CUDA trajectory was not bound to CPU steps 0/10/20: ${trajectory_directory};${trajectory_binding_output};${trajectory_binding_error}")
      endif()
      foreach(binding_field IN ITEMS vtk0_sha256 vtk10_sha256 vtk20_sha256
                                     metadata_sha256)
        string(REGEX MATCH "${binding_field}=([0-9a-f]+)"
               trajectory_hash_match "${trajectory_binding_output}")
        string(LENGTH "${CMAKE_MATCH_1}" trajectory_hash_length)
        if(NOT trajectory_hash_length EQUAL 64)
          message(FATAL_ERROR
            "rank-2 CUDA trajectory ${binding_field} is not 64 hex digits")
        endif()
      endforeach()
    endforeach()
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
    file(READ "${baseline_directory}/run-metadata.json" baseline_metadata)
    validate_scalar_cuda_metadata("${baseline_metadata}" "baseline" 20
      "W8B-SC25-N8-K1-20STEP-v1")
    validate_scalar_cuda_metadata("${reference_metadata}" "reference" 20
      "W8B-SC25-N8-K1-20STEP-v1")
    validate_scalar_cuda_metadata("${writer_metadata}" "split writer" 10
      "W8B-SC25-N8-K1-20STEP-v1")
    validate_scalar_cuda_metadata("${reader_metadata}" "split reader" 20
      "W8B-SC25-N8-K1-20STEP-v1")
  endif()
  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_RANKS EQUAL 2)
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
  if(HDG_RUNTIME_SCALAR_IGR AND HDG_RUNTIME_BACKEND STREQUAL "cuda")
    foreach(step IN ITEMS 000000 000010 000020)
      foreach(candidate_directory IN ITEMS reference_directory reader_directory)
        execute_process(
          COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk
            "${baseline_directory}/state-step-${step}.vtk"
            "${${candidate_directory}}/state-step-${step}.vtk" 1e-15
          RESULT_VARIABLE baseline_compare_status
          OUTPUT_VARIABLE baseline_compare_output
          ERROR_VARIABLE baseline_compare_error)
        if(NOT baseline_compare_status EQUAL 0 OR
           NOT baseline_compare_error STREQUAL "")
          message(FATAL_ERROR
            "CUDA scalar injected/split trajectory did not return to the fault-free baseline at ${step}: ${baseline_compare_output};${baseline_compare_error}")
        endif()
      endforeach()
    endforeach()
    execute_process(
      COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk
        "${reference_directory}/state-step-000010.vtk"
        "${writer_directory}/state-step-000010.vtk" 1e-15
      RESULT_VARIABLE writer_step10_status
      OUTPUT_VARIABLE writer_step10_output
      ERROR_VARIABLE writer_step10_error)
    if(NOT writer_step10_status EQUAL 0 OR
       NOT writer_step10_error STREQUAL "")
      message(FATAL_ERROR
        "CUDA scalar reference/writer step-10 identity failed: ${writer_step10_output};${writer_step10_error}")
    endif()
    foreach(hash_field IN ITEMS final_interior_sha256 final_trace_sha256)
      string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
             baseline_hash_match "${baseline_metadata}")
      set(baseline_hash "${CMAKE_MATCH_1}")
      string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
             reference_hash_match "${reference_metadata}")
      set(reference_hash "${CMAKE_MATCH_1}")
      string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
             reader_hash_match "${reader_metadata}")
      set(reader_hash "${CMAKE_MATCH_1}")
      if(NOT baseline_hash STREQUAL reference_hash OR
         NOT baseline_hash STREQUAL reader_hash)
        message(FATAL_ERROR
          "CUDA scalar ${hash_field} differs from the fault-free baseline")
      endif()
    endforeach()
    set(scalar_step0_artifacts "state-step-000000.vtk")
    set(scalar_step10_artifacts
      "checkpoint-runtime-identity.v1;restart-checkpoint.v2;state-step-000000.vtk;state-step-000010.vtk")
    set(scalar_step20_artifacts
      "checkpoint-runtime-identity.v1;restart-checkpoint.v2;run-metadata.json;state-step-000000.vtk;state-step-000010.vtk;state-step-000020.vtk")
    require_exact_logical_artifacts("${baseline_directory}" "baseline"
      "${scalar_step0_artifacts}" "${scalar_step10_artifacts}"
      "${scalar_step20_artifacts}")
    require_exact_logical_artifacts("${reference_directory}" "reference"
      "${scalar_step0_artifacts}" "${scalar_step10_artifacts}"
      "${scalar_step20_artifacts}")
    require_exact_logical_artifacts("${writer_directory}" "split"
      "${scalar_step0_artifacts}"
      "checkpoint-runtime-identity.v1;restart-checkpoint.v2;run-metadata-checkpoint.json;state-step-000000.vtk;state-step-000010.vtk"
      "checkpoint-runtime-identity.v1;restart-checkpoint.v2;run-metadata-checkpoint.json;run-metadata.json;state-step-000000.vtk;state-step-000010.vtk;state-step-000020.vtk")

    foreach(physical_field IN ITEMS physical_cuda_launches
                                    physical_cuda_transfers
                                    physical_cuda_owned_allocations
                                    physical_cuda_observed_allocations)
      string(REGEX MATCH "\"${physical_field}\": ([0-9]+)"
             baseline_physical_match "${baseline_metadata}")
      set(baseline_${physical_field} "${CMAKE_MATCH_1}")
      if("${baseline_${physical_field}}" STREQUAL "")
        message(FATAL_ERROR
          "fault-free baseline lacks ${physical_field} evidence")
      endif()
    endforeach()

    # Every admitted scalar-CUDA source rank consumes each scheduled step-10
    # boundary once. The failed logical attempt is erased, while physical work
    # remains monotone and the bounded retry returns to the fault-free state.
    math(EXPR scalar_maximum_source "${HDG_RUNTIME_RANKS} - 1")
    foreach(campaign_phase IN ITEMS CheckpointStaging OutputStaging
                                    AfterRecoveryBeforeCommit)
      if(campaign_phase STREQUAL "CheckpointStaging")
        set(campaign_boundary checkpoint-staging)
      elseif(campaign_phase STREQUAL "OutputStaging")
        set(campaign_boundary output-staging)
      else()
        set(campaign_boundary post-publication-pre-finalize)
      endif()
      foreach(campaign_source RANGE 0 ${scalar_maximum_source})
        set(campaign_directory
          "${HDG_RUNTIME_TEST_OUTPUT}-strong-step10-${campaign_phase}-rank${campaign_source}")
        set(campaign_checkpoint "${campaign_directory}.checkpoint")
        file(REMOVE_RECURSE "${campaign_directory}")
        file(REMOVE "${campaign_checkpoint}")
        run_runtime(campaign ${functional_common}
          --output-dir "${campaign_directory}"
          --checkpoint-out "${campaign_checkpoint}" --checkpoint-step 10
          --stop-after-checkpoint false --restart-from none
          --max-retries 1
          --inject-failure "${campaign_phase}:10:${campaign_source}")
        if(NOT campaign_status EQUAL 0 OR
           NOT campaign_output MATCHES
             "status=Success.*step=20.*retries=1" OR
           NOT campaign_error STREQUAL "")
          message(FATAL_ERROR
            "scalar CUDA ${campaign_phase}/rank${campaign_source} bounded retry failed: ${campaign_status};${campaign_output};${campaign_error}")
        endif()
        file(READ "${campaign_directory}/run-metadata.json"
                  campaign_metadata)
        if(NOT campaign_metadata MATCHES
             "\"injection_configured_source_rank\": ${campaign_source}" OR
           NOT campaign_metadata MATCHES
             "\"injection_observed_origin_rank\": ${campaign_source}" OR
           NOT campaign_metadata MATCHES "\"injection_local_count\": 1" OR
           NOT campaign_metadata MATCHES
             "\"injection_non_origin_local_count\": 0" OR
           NOT campaign_metadata MATCHES
             "\"injection_collective_decisions\": 1" OR
           NOT campaign_metadata MATCHES
             "\"injection_consumed_boundary_count\": 1" OR
           NOT campaign_metadata MATCHES "\"accepted_steps\": 20" OR
           NOT campaign_metadata MATCHES "\"attempted_steps\": 20" OR
           NOT campaign_metadata MATCHES "\"retries\": 1" OR
           NOT campaign_metadata MATCHES
             "boundary=${campaign_boundary} phase=${campaign_phase} target_step=10 consumed_boundary_count=1 source_rank=${campaign_source}")
          message(FATAL_ERROR
            "scalar CUDA ${campaign_phase}/rank${campaign_source} consumption/logical audit is invalid")
        endif()
        foreach(physical_field IN ITEMS physical_cuda_launches
                                        physical_cuda_transfers
                                        physical_cuda_owned_allocations
                                        physical_cuda_observed_allocations)
          string(REGEX MATCH "\"${physical_field}\": ([0-9]+)"
                 campaign_physical_match "${campaign_metadata}")
          set(campaign_${physical_field} "${CMAKE_MATCH_1}")
          if("${campaign_${physical_field}}" STREQUAL "")
            message(FATAL_ERROR
              "scalar CUDA campaign lacks ${physical_field} evidence")
          endif()
        endforeach()
        if(NOT campaign_physical_cuda_launches GREATER
               baseline_physical_cuda_launches OR
           NOT campaign_physical_cuda_transfers GREATER
               baseline_physical_cuda_transfers OR
           NOT campaign_physical_cuda_owned_allocations EQUAL
               baseline_physical_cuda_owned_allocations OR
           NOT campaign_physical_cuda_observed_allocations EQUAL
               baseline_physical_cuda_observed_allocations)
          message(FATAL_ERROR
            "scalar CUDA ${campaign_phase}/rank${campaign_source} physical ledger/plateau audit is invalid")
        endif()
        foreach(step IN ITEMS 000000 000010 000020)
          execute_process(
            COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk
              "${baseline_directory}/state-step-${step}.vtk"
              "${campaign_directory}/state-step-${step}.vtk" 1e-15
            RESULT_VARIABLE campaign_compare_status
            OUTPUT_VARIABLE campaign_compare_output
            ERROR_VARIABLE campaign_compare_error)
          if(NOT campaign_compare_status EQUAL 0 OR
             NOT campaign_compare_error STREQUAL "")
            message(FATAL_ERROR
              "scalar CUDA ${campaign_phase}/rank${campaign_source} state-${step} differs from baseline: ${campaign_compare_output};${campaign_compare_error}")
          endif()
        endforeach()
        foreach(hash_field IN ITEMS final_interior_sha256 final_trace_sha256)
          string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
                 baseline_campaign_hash_match "${baseline_metadata}")
          set(baseline_campaign_hash "${CMAKE_MATCH_1}")
          string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
                 campaign_hash_match "${campaign_metadata}")
          if(NOT baseline_campaign_hash STREQUAL "${CMAKE_MATCH_1}")
            message(FATAL_ERROR
              "scalar CUDA ${campaign_phase}/rank${campaign_source} ${hash_field} differs from baseline")
          endif()
        endforeach()
        require_exact_logical_artifacts(
          "${campaign_directory}"
          "${campaign_phase}/rank${campaign_source} retry"
          "${scalar_step0_artifacts}" "${scalar_step10_artifacts}"
          "${scalar_step20_artifacts}")
      endforeach()
    endforeach()

    # A terminal step-20 post-publication fault has no in-process
    # compensation. It must leave the exact step-10 authority restartable, and
    # a later split reader alone may publish the final generation.
    foreach(campaign_source RANGE 0 ${scalar_maximum_source})
      set(terminal_directory
        "${HDG_RUNTIME_TEST_OUTPUT}-strong-step20-rank${campaign_source}")
      set(terminal_checkpoint "${terminal_directory}.checkpoint")
      file(REMOVE_RECURSE "${terminal_directory}")
      file(REMOVE "${terminal_checkpoint}")
      run_runtime(terminal ${functional_common}
        --output-dir "${terminal_directory}"
        --checkpoint-out "${terminal_checkpoint}" --checkpoint-step 10
        --stop-after-checkpoint false --restart-from none
        --max-retries 0
        --inject-failure
          "AfterRecoveryBeforeCommit:20:${campaign_source}")
      if(NOT terminal_status EQUAL 72 OR
         NOT terminal_error MATCHES
           "status=RuntimeFailure.*step=19.*retries=0.*boundary=post-publication-pre-finalize phase=AfterRecoveryBeforeCommit target_step=20 consumed_boundary_count=1 source_rank=${campaign_source}")
        message(FATAL_ERROR
          "scalar CUDA terminal post-publication rank${campaign_source} campaign failed: ${terminal_status};${terminal_output};${terminal_error}")
      endif()
      require_exact_logical_artifacts(
        "${terminal_directory}"
        "terminal post-publication rank${campaign_source} rollback"
        "${scalar_step0_artifacts}" "${scalar_step10_artifacts}")
      file(SHA256 "${terminal_checkpoint}" terminal_checkpoint_before)
      run_runtime(terminal_reader ${functional_common}
        --output-dir "${terminal_directory}"
        --checkpoint-out none --checkpoint-step none
        --stop-after-checkpoint false
        --restart-from "${terminal_checkpoint}"
        --max-retries 0 --inject-failure off)
      file(SHA256 "${terminal_checkpoint}" terminal_checkpoint_after)
      if(NOT terminal_reader_status EQUAL 0 OR
         NOT terminal_reader_output MATCHES
           "status=Success.*step=20.*retries=1" OR
         NOT terminal_reader_error STREQUAL "" OR
         NOT terminal_checkpoint_before STREQUAL terminal_checkpoint_after)
        message(FATAL_ERROR
          "scalar CUDA terminal rank${campaign_source} restart continuation failed: ${terminal_reader_status};${terminal_reader_output};${terminal_reader_error}")
      endif()
      file(READ "${terminal_directory}/run-metadata.json"
                terminal_reader_metadata)
      foreach(step IN ITEMS 000000 000010 000020)
        execute_process(
          COMMAND "${HDG_RUNTIME_SCALAR_VTK_TEST}" --compare-scalar-vtk
            "${baseline_directory}/state-step-${step}.vtk"
            "${terminal_directory}/state-step-${step}.vtk" 1e-15
          RESULT_VARIABLE terminal_compare_status
          OUTPUT_VARIABLE terminal_compare_output
          ERROR_VARIABLE terminal_compare_error)
        if(NOT terminal_compare_status EQUAL 0 OR
           NOT terminal_compare_error STREQUAL "")
          message(FATAL_ERROR
            "scalar CUDA terminal rank${campaign_source} restart state-${step} differs from baseline: ${terminal_compare_output};${terminal_compare_error}")
        endif()
      endforeach()
      foreach(hash_field IN ITEMS final_interior_sha256 final_trace_sha256)
        string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
               baseline_terminal_hash_match "${baseline_metadata}")
        set(baseline_terminal_hash "${CMAKE_MATCH_1}")
        string(REGEX MATCH "\"${hash_field}\": \"([0-9a-f]+)\""
               terminal_hash_match "${terminal_reader_metadata}")
        if(NOT baseline_terminal_hash STREQUAL "${CMAKE_MATCH_1}")
          message(FATAL_ERROR
            "scalar CUDA terminal rank${campaign_source} ${hash_field} differs from baseline")
        endif()
      endforeach()
      require_exact_logical_artifacts(
        "${terminal_directory}"
        "terminal post-publication rank${campaign_source} restart"
        "${scalar_step0_artifacts}" "${scalar_step10_artifacts}"
        "${scalar_step20_artifacts}")
    endforeach()
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

set(crossed_conventional_arguments ${common_arguments})
replace_cli_option(crossed_conventional_arguments
                   crossed_conventional_arguments --case
                   sc25-periodic-smooth-persistent-v3)
replace_cli_option(crossed_conventional_arguments
                   crossed_conventional_arguments --output-dir
                   "${HDG_RUNTIME_TEST_OUTPUT}-crossed-conventional")
replace_cli_option(crossed_conventional_arguments
                   crossed_conventional_arguments --checkpoint-out
                   "${HDG_RUNTIME_TEST_OUTPUT}-crossed-conventional.checkpoint")
require_pre_resource_rejection(
  "conventional-card/scalar-case" unknown_case
  "${HDG_RUNTIME_TEST_OUTPUT}-crossed-conventional"
  "${HDG_RUNTIME_TEST_OUTPUT}-crossed-conventional.checkpoint"
  ${crossed_conventional_arguments})

set(scalar_negative_arguments
  --card exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3
  --case sc25-periodic-smooth-persistent-v3
  --mesh-family periodic-cartesian-quad --mesh-n 8 --order 1
  --dt 0.000244140625 --steps 20 --final-time 0.0048828125
  --backend "${HDG_RUNTIME_BACKEND}" --device-id "${runtime_device}"
  --output-dir "${HDG_RUNTIME_TEST_OUTPUT}-scalar-negative"
  --output-steps 0,10,20
  --checkpoint-out "${HDG_RUNTIME_TEST_OUTPUT}-scalar-negative.checkpoint"
  --checkpoint-step 10 --stop-after-checkpoint false
  --restart-from none --max-retries 1
  --inject-failure "AfterRecoveryBeforeCommit:7:0"
  --seed 0x5738425343323553)

set(crossed_scalar_arguments ${scalar_negative_arguments})
replace_cli_option(crossed_scalar_arguments crossed_scalar_arguments --case
                   conventional-smooth-persistent-v1)
replace_cli_option(crossed_scalar_arguments crossed_scalar_arguments
                   --output-dir
                   "${HDG_RUNTIME_TEST_OUTPUT}-crossed-scalar")
replace_cli_option(crossed_scalar_arguments crossed_scalar_arguments
                   --checkpoint-out
                   "${HDG_RUNTIME_TEST_OUTPUT}-crossed-scalar.checkpoint")
require_pre_resource_rejection(
  "scalar-card/conventional-case" unknown_case
  "${HDG_RUNTIME_TEST_OUTPUT}-crossed-scalar"
  "${HDG_RUNTIME_TEST_OUTPUT}-crossed-scalar.checkpoint"
  ${crossed_scalar_arguments})

set(scalar_short_arguments ${scalar_negative_arguments})
replace_cli_option(scalar_short_arguments scalar_short_arguments --steps 3)
replace_cli_option(scalar_short_arguments scalar_short_arguments --output-dir
                   "${HDG_RUNTIME_TEST_OUTPUT}-scalar-three-step")
replace_cli_option(scalar_short_arguments scalar_short_arguments
                   --checkpoint-out
                   "${HDG_RUNTIME_TEST_OUTPUT}-scalar-three-step.checkpoint")
require_pre_resource_rejection(
  "scalar-card/three-step" unknown_scenario
  "${HDG_RUNTIME_TEST_OUTPUT}-scalar-three-step"
  "${HDG_RUNTIME_TEST_OUTPUT}-scalar-three-step.checkpoint"
  ${scalar_short_arguments})

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
