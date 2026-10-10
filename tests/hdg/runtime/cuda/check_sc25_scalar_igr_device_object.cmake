cmake_policy(SET CMP0009 NEW)

foreach(_required IN ITEMS OBJECT_ROOT RUNTIME_ARCHIVE TEST_EXECUTABLE
                           CUOBJDUMP_EXECUTABLE NM_EXECUTABLE)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "missing required device-object argument ${_required}")
  endif()
endforeach()

foreach(_artifact IN ITEMS RUNTIME_ARCHIVE TEST_EXECUTABLE)
  if(NOT EXISTS "${${_artifact}}")
    message(FATAL_ERROR "${_artifact} does not exist: ${${_artifact}}")
  endif()
endforeach()

file(GLOB_RECURSE _objects
  "${OBJECT_ROOT}/*hdg_sc25_periodic_v3_device.cu.o")
list(LENGTH _objects _object_count)
if(NOT _object_count EQUAL 1)
  message(FATAL_ERROR
    "expected exactly one hdg_sc25_periodic_v3_device.cu.o, found ${_object_count}: ${_objects}")
endif()
list(GET _objects 0 _object)
file(SIZE "${_object}" _object_size)
if(_object_size EQUAL 0)
  message(FATAL_ERROR "scalar-IGR CUDA device object is empty: ${_object}")
endif()

set(_rank1_kernels
  HdgSc25AssembleLocalFp64Kernel
  HdgSc25CondenseFp64Kernel
  HdgSc25RecoverFp64Kernel)
set(_rank2_kernels
  HdgSc25Rank2AssembleLocalFp64Kernel
  HdgSc25Rank2CondenseFp64Kernel
  HdgSc25Rank2RecoverFp64Kernel)
set(_executable_kernels ${_rank1_kernels} ${_rank2_kernels})

function(_read_definitions _artifact _output)
  execute_process(
    COMMAND "${NM_EXECUTABLE}" -A -C --defined-only "${_artifact}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "nm failed for ${_artifact}: ${_error}")
  endif()
  set(${_output} "${_symbols}" PARENT_SCOPE)
endfunction()

_read_definitions("${_object}" _object_symbols)
_read_definitions("${RUNTIME_ARCHIVE}" _archive_symbols)
_read_definitions("${TEST_EXECUTABLE}" _executable_symbols)
get_filename_component(_object_name "${_object}" NAME)
get_filename_component(_archive_name "${RUNTIME_ARCHIVE}" NAME)

string(REGEX MATCHALL
  "(^|\n)[^\n]*[ \t][TW][ \t]+HdgSc25[A-Za-z0-9_]*Kernel(\n|$)"
  _object_kernel_definitions "${_object_symbols}")
list(LENGTH _object_kernel_definitions _object_kernel_count)
if(NOT _object_kernel_count EQUAL 3)
  message(FATAL_ERROR
    "the scalar device object must define exactly the three frozen kernels; found ${_object_kernel_count}\n${_object_symbols}")
endif()

foreach(_symbol IN LISTS _rank1_kernels)
  foreach(_scope IN ITEMS object archive)
    if(_scope STREQUAL "object")
      set(_symbols "${_object_symbols}")
    elseif(_scope STREQUAL "archive")
      set(_symbols "${_archive_symbols}")
    else()
      set(_symbols "${_executable_symbols}")
    endif()
    string(REGEX MATCHALL
      "(^|\n)[^\n]*[ \t][TW][ \t]+${_symbol}(\n|$)"
      _definitions "${_symbols}")
    list(LENGTH _definitions _definition_count)
    if(NOT _definition_count EQUAL 1)
      message(FATAL_ERROR
        "${_scope} must contain exactly one definition of ${_symbol}; found ${_definition_count}")
    endif()
  endforeach()
  string(REGEX MATCH
    "(^|\n)[^\n]*${_archive_name}:${_object_name}:[^\n]*[ \t][TW][ \t]+${_symbol}(\n|$)"
    _archive_origin "${_archive_symbols}")
  if(NOT _archive_origin)
    message(FATAL_ERROR
      "${_symbol} is not bound to ${_object_name} inside ${_archive_name}")
  endif()
endforeach()

string(REGEX MATCHALL
  "(^|\n)[^\n]*[ \t][TW][ \t]+HdgSc25[A-Za-z0-9_]*Kernel(\n|$)"
  _executable_kernel_definitions "${_executable_symbols}")
list(LENGTH _executable_kernel_definitions _executable_kernel_count)
if(NOT _executable_kernel_count EQUAL 6)
  message(FATAL_ERROR
    "the final executable must define exactly the six rank-one/rank-two kernels; found ${_executable_kernel_count}\n${_executable_symbols}")
endif()
foreach(_symbol IN LISTS _executable_kernels)
  string(REGEX MATCHALL
    "(^|\n)[^\n]*[ \t][TW][ \t]+${_symbol}(\n|$)"
    _definitions "${_executable_symbols}")
  list(LENGTH _definitions _definition_count)
  if(NOT _definition_count EQUAL 1)
    message(FATAL_ERROR
      "executable must contain exactly one definition of ${_symbol}; found ${_definition_count}")
  endif()
endforeach()

function(_check_native_sass _artifact _label _expected_count)
  set(_expected_kernels ${ARGN})
  execute_process(
    COMMAND "${CUOBJDUMP_EXECUTABLE}" --dump-sass "${_artifact}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _sass
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "cuobjdump --dump-sass failed for ${_label}: ${_error}")
  endif()
  string(REGEX MATCHALL
    "(^|\n)[ \t]*Function[ \t]*:[ \t]*HdgSc25[A-Za-z0-9_]*Kernel[ \t]*(\n|$)"
    _all_scalar_function_markers "${_sass}")
  list(LENGTH _all_scalar_function_markers _all_scalar_function_count)
  if(NOT _all_scalar_function_count EQUAL ${_expected_count})
    message(FATAL_ERROR
      "${_label} must contain exactly ${_expected_count} exact-boundary scalar SASS functions; found ${_all_scalar_function_count}")
  endif()
  foreach(_symbol IN LISTS _expected_kernels)
    string(REGEX MATCHALL
      "(^|\n)[ \t]*Function[ \t]*:[ \t]*${_symbol}[ \t]*(\n|$)"
      _function_markers "${_sass}")
    list(LENGTH _function_markers _function_marker_count)
    if(NOT _function_marker_count EQUAL 1)
      message(FATAL_ERROR
        "${_label} must contain exactly one exact-boundary SASS function for ${_symbol}; found ${_function_marker_count}")
    endif()
    list(GET _function_markers 0 _function_marker)
    string(REGEX REPLACE "^\n" "" _function_marker "${_function_marker}")
    string(FIND "${_sass}" "${_function_marker}" _function_offset)
    string(SUBSTRING "${_sass}" 0 ${_function_offset} _prefix)
    string(FIND "${_prefix}" "code for sm_120" _sm120_code REVERSE)
    string(FIND "${_prefix}" "arch = sm_120" _sm120_arch REVERSE)
    if(_sm120_code GREATER _sm120_arch)
      set(_sm120_offset ${_sm120_code})
    else()
      set(_sm120_offset ${_sm120_arch})
    endif()
    string(FIND "${_prefix}" "code for sm_80" _sm80_code REVERSE)
    string(FIND "${_prefix}" "arch = sm_80" _sm80_arch REVERSE)
    if(_sm80_code GREATER _sm80_arch)
      set(_sm80_offset ${_sm80_code})
    else()
      set(_sm80_offset ${_sm80_arch})
    endif()
    if(_sm120_offset LESS 0 OR _sm120_offset LESS _sm80_offset)
      message(FATAL_ERROR
        "${_label} binds ${_symbol} to PTX/MFEM80 rather than native sm_120")
    endif()
    string(SUBSTRING "${_sass}" ${_function_offset} -1 _function_tail)
    string(FIND "${_function_tail}" "\nFunction" _next_function)
    if(_next_function GREATER 0)
      string(SUBSTRING "${_function_tail}" 0 ${_next_function}
             _function_body)
    else()
      set(_function_body "${_function_tail}")
    endif()
    if(NOT _function_body MATCHES
       "/\\*[0-9A-Fa-f]+\\*/[^\n]*[A-Z][A-Z0-9_.]+")
      message(FATAL_ERROR
        "${_label} has no nonempty native instruction stream for ${_symbol}")
    endif()
  endforeach()
endfunction()

_check_native_sass("${_object}" "device object" 3 ${_rank1_kernels})
_check_native_sass("${TEST_EXECUTABLE}" "final executable" 6
                   ${_executable_kernels})

message(STATUS
  "SC25_CUDA_DEVICE_OBJECT_PASS object=${_object} bytes=${_object_size} archive=${RUNTIME_ARCHIVE} executable=${TEST_EXECUTABLE}")
