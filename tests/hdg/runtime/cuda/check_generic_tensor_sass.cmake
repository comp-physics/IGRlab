foreach(_required IN ITEMS TEST_EXECUTABLE CUOBJDUMP_EXECUTABLE
                           EXPECTED_KERNEL_SUBSTRING)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "missing generic tensor SASS argument ${_required}")
  endif()
endforeach()

if(NOT EXISTS "${TEST_EXECUTABLE}")
  message(FATAL_ERROR
    "production-linked tensor evidence binary does not exist: ${TEST_EXECUTABLE}")
endif()

execute_process(
  COMMAND "${CUOBJDUMP_EXECUTABLE}" --dump-elf-symbols
          --gpu-architecture sm_120 "${TEST_EXECUTABLE}"
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
  COMMAND "${CUOBJDUMP_EXECUTABLE}" --dump-sass
          --gpu-architecture sm_120 --function "${_kernel_symbol}"
          "${TEST_EXECUTABLE}"
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

message(STATUS
  "GENERIC_TENSOR_SASS_PASS executable=${TEST_EXECUTABLE} kernel=${_kernel_symbol} fp64_tensor_instructions=${_dmma_count}")
