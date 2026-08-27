foreach(_required IN ITEMS TEST_EXECUTABLE CUOBJDUMP_EXECUTABLE
                           EXPECTED_KERNEL_SUBSTRING EVIDENCE_OUTPUT_DIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "missing generic tensor SASS argument ${_required}")
  endif()
endforeach()

if(NOT EXISTS "${TEST_EXECUTABLE}")
  message(FATAL_ERROR
    "production-linked tensor evidence binary does not exist: ${TEST_EXECUTABLE}")
endif()

get_filename_component(_test_executable_real_path
  "${TEST_EXECUTABLE}" REALPATH)
get_filename_component(_cuobjdump_real_path
  "${CUOBJDUMP_EXECUTABLE}" REALPATH)
file(MAKE_DIRECTORY "${EVIDENCE_OUTPUT_DIR}")
get_filename_component(_evidence_output_real_path
  "${EVIDENCE_OUTPUT_DIR}" REALPATH)
file(SHA256 "${_test_executable_real_path}" _test_executable_sha256)
file(SHA256 "${_cuobjdump_real_path}" _cuobjdump_sha256)

execute_process(
  COMMAND "${_cuobjdump_real_path}" --version
  RESULT_VARIABLE _version_result
  OUTPUT_VARIABLE _version_output
  ERROR_VARIABLE _version_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE)
if(NOT _version_result EQUAL 0)
  message(FATAL_ERROR
    "cuobjdump --version failed for ${_cuobjdump_real_path}: ${_version_error}")
endif()
set(_cuobjdump_version "${_version_output}")
if(NOT _version_error STREQUAL "")
  if(NOT _cuobjdump_version STREQUAL "")
    string(APPEND _cuobjdump_version "\n")
  endif()
  string(APPEND _cuobjdump_version "${_version_error}")
endif()
if(_cuobjdump_version STREQUAL "")
  message(FATAL_ERROR "cuobjdump --version returned no identity")
endif()

execute_process(
  COMMAND "${_cuobjdump_real_path}" --dump-elf-symbols
          --gpu-architecture sm_120 "${_test_executable_real_path}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _symbols
  ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "cuobjdump --dump-elf-symbols failed for ${TEST_EXECUTABLE}: ${_error}")
endif()

# Resolve one exact native entry before asking cuobjdump for its SASS.  This
# binds the instruction count to the production condensation leaf and avoids
# accepting tensor instructions from another function in the linked binary.
string(REGEX MATCHALL
  "[^\n]*STO_ENTRY[^\n]*${EXPECTED_KERNEL_SUBSTRING}[^\n]*"
  _kernel_symbol_lines "${_symbols}")
list(LENGTH _kernel_symbol_lines _kernel_symbol_count)
if(NOT _kernel_symbol_count EQUAL 1)
  message(FATAL_ERROR
    "expected exactly one native sm_120 entry containing "
    "${EXPECTED_KERNEL_SUBSTRING}, found ${_kernel_symbol_count}")
endif()
list(GET _kernel_symbol_lines 0 _kernel_symbol_line)
string(STRIP "${_kernel_symbol_line}" _kernel_symbol_line)
string(REGEX REPLACE ".*[ \t]" "" _kernel_symbol "${_kernel_symbol_line}")
if(_kernel_symbol STREQUAL "")
  message(FATAL_ERROR
    "could not extract the exact ${EXPECTED_KERNEL_SUBSTRING} symbol")
endif()

execute_process(
  COMMAND "${_cuobjdump_real_path}" --dump-sass
          --gpu-architecture sm_120 --function "${_kernel_symbol}"
          "${_test_executable_real_path}"
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _function_sass
  ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "cuobjdump --dump-sass failed for ${_kernel_symbol}: ${_error}")
endif()

string(FIND "${_function_sass}" "${_kernel_symbol}" _kernel_marker_offset)
string(FIND "${_function_sass}" "code for sm_120" _sm120_code_offset)
string(FIND "${_function_sass}" "arch = sm_120" _sm120_arch_offset)
if(_kernel_marker_offset LESS 0 OR
   (_sm120_code_offset LESS 0 AND _sm120_arch_offset LESS 0))
  message(FATAL_ERROR
    "exact ${EXPECTED_KERNEL_SUBSTRING} dump is not native sm_120 SASS")
endif()

# CUDA 13.3 lowers the requested FP64 WMMA m8n8k4 operation to the definitive
# SM120 DMMA.8x8x4 spelling.  Deliberately do not accept HMMA, an unrelated MMA
# family, or a tensor-capable library name as evidence of this binary64 leaf.
string(REGEX MATCHALL
  "(^|\n)[^\n]*[ \t]DMMA[.]8x8x4([ \t]|$)"
  _dmma_instructions "${_function_sass}")
list(LENGTH _dmma_instructions _dmma_count)
if(_dmma_count LESS 1)
  message(FATAL_ERROR
    "${EXPECTED_KERNEL_SUBSTRING} has no DMMA.8x8x4 instruction")
endif()

# Retain one finite, static exact-function artifact in the canonical build
# tree.  This is post-build disassembly, not application instrumentation or an
# event stream.  The companion manifest binds the production executable,
# cuobjdump identity/commands, exact native symbol, and artifact hash.
set(_sass_artifact
  "${_evidence_output_real_path}/tensor-fused-local-algebra-sm120.sass")
set(_evidence_manifest
  "${_evidence_output_real_path}/tensor-sass-evidence-v1.txt")
file(WRITE "${_sass_artifact}" "${_function_sass}")
file(SHA256 "${_sass_artifact}" _sass_artifact_sha256)
file(SHA256 "${_test_executable_real_path}" _test_executable_sha256_after)
file(SHA256 "${_cuobjdump_real_path}" _cuobjdump_sha256_after)
if(NOT "${_test_executable_sha256_after}" STREQUAL
       "${_test_executable_sha256}" OR
   NOT "${_cuobjdump_sha256_after}" STREQUAL "${_cuobjdump_sha256}")
  message(FATAL_ERROR
    "tensor SASS executable or cuobjdump changed during disassembly")
endif()
string(REPLACE "\n" "\\n" _cuobjdump_version_field
  "${_cuobjdump_version}")
set(_symbols_command
  "${_cuobjdump_real_path} --dump-elf-symbols --gpu-architecture sm_120 ${_test_executable_real_path}")
set(_sass_command
  "${_cuobjdump_real_path} --dump-sass --gpu-architecture sm_120 --function ${_kernel_symbol} ${_test_executable_real_path}")
file(WRITE "${_evidence_manifest}"
  "schema=dg-igr-hdg-tensor-sass-evidence\n"
  "version=1\n"
  "scope=fused-local-algebra-condensation\n"
  "architecture=sm_120\n"
  "executable=${_test_executable_real_path}\n"
  "executable_sha256=${_test_executable_sha256}\n"
  "cuobjdump=${_cuobjdump_real_path}\n"
  "cuobjdump_sha256=${_cuobjdump_sha256}\n"
  "cuobjdump_version=${_cuobjdump_version_field}\n"
  "symbols_command=${_symbols_command}\n"
  "sass_command=${_sass_command}\n"
  "kernel=${_kernel_symbol}\n"
  "fp64_tensor_instruction=DMMA.8x8x4\n"
  "fp64_tensor_static_instruction_sites=${_dmma_count}\n"
  "sass_artifact=${_sass_artifact}\n"
  "sass_artifact_sha256=${_sass_artifact_sha256}\n"
  "source_contract_intended_fused_stages=blocked-lu-trailing-update,blocked-forward-rhs-update,blocked-backward-rhs-update,schur-contraction\n"
  "assembly_contract=fused-simt-no-tensor-instruction-claim\n")
file(SHA256 "${_evidence_manifest}" _evidence_manifest_sha256)

message(STATUS
  "GENERIC_TENSOR_SASS_PASS executable=${_test_executable_real_path} executable_sha256=${_test_executable_sha256} cuobjdump=${_cuobjdump_real_path} cuobjdump_sha256=${_cuobjdump_sha256} kernel=${_kernel_symbol} fp64_tensor_static_instruction_sites=${_dmma_count} sass_artifact=${_sass_artifact} sass_artifact_sha256=${_sass_artifact_sha256} manifest=${_evidence_manifest} manifest_sha256=${_evidence_manifest_sha256}")
