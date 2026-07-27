if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(dg_igr_hdg_cpu_oracle LANGUAGES CXX)
  find_package(MFEM 4.9 CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

get_filename_component(DG_IGR_HDG_REPOSITORY_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(dg_igr_hdg_contract_test test_hdg_contract.cpp)
target_include_directories(dg_igr_hdg_contract_test PRIVATE ../../include)
target_compile_options(dg_igr_hdg_contract_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_contract_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

# Compile controls and rejection cases through one isolated C++17 harness.
# A negative is accepted only when compilation fails with its unique marker;
# missing files/includes and unrelated configure or syntax failures fail closed.
function(dg_igr_hdg_contract_try_compile name source result_var output_var)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/negative/${source}")
    message(FATAL_ERROR
      "HDG contract compile case '${name}' source is missing: ${source}")
  endif()
  set(dg_igr_hdg_result "dg_igr_hdg_contract_${name}")
  try_compile(${dg_igr_hdg_result}
    "${CMAKE_CURRENT_BINARY_DIR}/contract-compile/${name}"
    "${CMAKE_CURRENT_SOURCE_DIR}/negative"
    dg_igr_hdg_contract_negative
    dg_igr_hdg_contract_negative
    CMAKE_FLAGS
      "-DHDG_NEGATIVE_SOURCE:FILEPATH=${CMAKE_CURRENT_SOURCE_DIR}/negative/${source}"
      "-DDG_IGR_REPOSITORY_ROOT:PATH=${DG_IGR_HDG_REPOSITORY_ROOT}"
      "-DCMAKE_TRY_COMPILE_TARGET_TYPE:STRING=STATIC_LIBRARY"
    OUTPUT_VARIABLE dg_igr_hdg_output)
  set(${result_var} "${${dg_igr_hdg_result}}" PARENT_SCOPE)
  set(${output_var} "${dg_igr_hdg_output}" PARENT_SCOPE)
endfunction()

function(dg_igr_expect_hdg_contract_compile_success name source)
  dg_igr_hdg_contract_try_compile(
    "${name}" "${source}" dg_igr_hdg_compiled dg_igr_hdg_output)
  if(NOT dg_igr_hdg_compiled)
    message(FATAL_ERROR
      "HDG contract positive control '${name}' failed to compile:\n"
      "${dg_igr_hdg_output}")
  endif()
endfunction()

function(dg_igr_expect_hdg_contract_compile_failure name source marker)
  get_property(dg_igr_hdg_markers GLOBAL PROPERTY DG_IGR_HDG_MARKERS)
  list(FIND dg_igr_hdg_markers "${marker}" dg_igr_hdg_duplicate_marker)
  if(NOT dg_igr_hdg_duplicate_marker EQUAL -1)
    message(FATAL_ERROR "duplicate HDG negative marker: ${marker}")
  endif()
  set_property(GLOBAL APPEND PROPERTY DG_IGR_HDG_MARKERS "${marker}")

  dg_igr_hdg_contract_try_compile(
    "${name}" "${source}" dg_igr_hdg_compiled dg_igr_hdg_output)
  if(dg_igr_hdg_compiled)
    message(FATAL_ERROR
      "HDG contract negative case '${name}' unexpectedly compiled:\n"
      "${dg_igr_hdg_output}")
  endif()
  string(FIND "${dg_igr_hdg_output}" "${marker}" dg_igr_hdg_marker_offset)
  if(dg_igr_hdg_marker_offset EQUAL -1)
    message(FATAL_ERROR
      "HDG contract negative case '${name}' failed without marker '${marker}':\n"
      "${dg_igr_hdg_output}")
  endif()
endfunction()

dg_igr_expect_hdg_contract_compile_success(
  positive_control positive_control.cpp)

dg_igr_expect_hdg_contract_compile_failure(
  invalid_trace invalid_trace.cpp HDG_CONTRACT_TRACE_POTENTIAL)
dg_igr_expect_hdg_contract_compile_failure(
  duplicate_trace duplicate_trace.cpp HDG_CONTRACT_TRACE_UNIQUE_NAMES)
dg_igr_expect_hdg_contract_compile_failure(
  non_trace_tuple_spec non_trace_tuple_spec.cpp HDG_CONTRACT_TRACE_TUPLE)
dg_igr_expect_hdg_contract_compile_failure(
  invalid_model invalid_model.cpp HDG_CONTRACT_MODEL_TAG)
dg_igr_expect_hdg_contract_compile_failure(
  invalid_stabilization invalid_stabilization.cpp
  HDG_CONTRACT_STABILIZATION_TAG)
dg_igr_expect_hdg_contract_compile_failure(
  invalid_time invalid_time.cpp HDG_CONTRACT_TIME_TAG)
dg_igr_expect_hdg_contract_compile_failure(
  invalid_backend invalid_backend.cpp HDG_CONTRACT_BACKEND_POLICY)
dg_igr_expect_hdg_contract_compile_failure(
  invalid_precision invalid_precision.cpp HDG_CONTRACT_PRECISION_HOMOGENEOUS)

dg_igr_expect_hdg_contract_compile_failure(
  nontrivial_device nontrivial_device.cpp HDG_CONTRACT_DEVICE_TRIVIAL)
dg_igr_expect_hdg_contract_compile_failure(
  nonstandard_layout_device nonstandard_layout_device.cpp
  HDG_CONTRACT_DEVICE_STANDARD_LAYOUT)
dg_igr_expect_hdg_contract_compile_failure(
  polymorphic_device polymorphic_device.cpp
  HDG_CONTRACT_DEVICE_NONPOLYMORPHIC)

dg_igr_expect_hdg_contract_compile_failure(
  polymorphic_local polymorphic_local.cpp
  HDG_CONTRACT_LOCAL_NONPOLYMORPHIC)
dg_igr_expect_hdg_contract_compile_failure(
  polymorphic_condensation polymorphic_condensation.cpp
  HDG_CONTRACT_CONDENSATION_NONPOLYMORPHIC)
dg_igr_expect_hdg_contract_compile_failure(
  polymorphic_assembler polymorphic_assembler.cpp
  HDG_CONTRACT_ASSEMBLER_NONPOLYMORPHIC)
dg_igr_expect_hdg_contract_compile_failure(
  polymorphic_solver polymorphic_solver.cpp
  HDG_CONTRACT_SOLVER_NONPOLYMORPHIC)
dg_igr_expect_hdg_contract_compile_failure(
  polymorphic_recovery polymorphic_recovery.cpp
  HDG_CONTRACT_RECOVERY_NONPOLYMORPHIC)

dg_igr_expect_hdg_contract_compile_failure(
  wrong_evaluate_residual wrong_evaluate_residual.cpp
  HDG_CONTRACT_EXACT_EVALUATE_RESIDUAL)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_evaluate_jacobian wrong_evaluate_jacobian.cpp
  HDG_CONTRACT_EXACT_EVALUATE_JACOBIAN)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_condense wrong_condense.cpp HDG_CONTRACT_EXACT_CONDENSE)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_apply_schur wrong_apply_schur.cpp HDG_CONTRACT_EXACT_APPLY_SCHUR)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_assemble wrong_assemble.cpp HDG_CONTRACT_EXACT_ASSEMBLE)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_solve wrong_solve.cpp HDG_CONTRACT_EXACT_SOLVE)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_recover wrong_recover.cpp HDG_CONTRACT_EXACT_RECOVER)

dg_igr_expect_hdg_contract_compile_failure(
  generic_phase_method generic_phase_method.cpp
  HDG_NEGATIVE_GENERIC_PHASE_METHOD)
dg_igr_expect_hdg_contract_compile_failure(
  variadic_phase_method variadic_phase_method.cpp
  HDG_NEGATIVE_VARIADIC_PHASE_METHOD)
dg_igr_expect_hdg_contract_compile_failure(
  overloaded_phase_method overloaded_phase_method.cpp
  HDG_NEGATIVE_OVERLOADED_PHASE_METHOD)
dg_igr_expect_hdg_contract_compile_failure(
  by_value_request by_value_request.cpp HDG_NEGATIVE_BY_VALUE_REQUEST)
dg_igr_expect_hdg_contract_compile_failure(
  const_result const_result.cpp HDG_NEGATIVE_CONST_RESULT)
dg_igr_expect_hdg_contract_compile_failure(
  convertible_request convertible_request.cpp
  HDG_NEGATIVE_CONVERTIBLE_REQUEST)
dg_igr_expect_hdg_contract_compile_failure(
  wrong_return wrong_return.cpp HDG_NEGATIVE_WRONG_RETURN)
dg_igr_expect_hdg_contract_compile_failure(
  const_qualified_method const_qualified_method.cpp
  HDG_NEGATIVE_CONST_QUALIFIED_METHOD)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME dg_igr_hdg_contract_cpu COMMAND dg_igr_hdg_contract_test)
  set_tests_properties(dg_igr_hdg_contract_cpu PROPERTIES
    LABELS "hdg;cpu;contract"
    TIMEOUT 30)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/runtime_contract/CMakeLists.txt")
  add_subdirectory(runtime_contract)
endif()

if((DG_IGR_ENABLE_HDG_RUNTIME OR
    DG_IGR_ENABLE_HDG_AUXILIARY_EXPERIMENT) AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/CMakeLists.txt")
  add_subdirectory(runtime)
endif()

# CPU core, trace, device, and formulation tests remain independently
# registerable even when the accelerator-only oracle below is unavailable.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/CMakeLists.txt")
  add_subdirectory(core)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/trace/CMakeLists.txt")
  add_subdirectory(trace)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/device/CMakeLists.txt")
  add_subdirectory(device)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/euler/CMakeLists.txt")
  add_subdirectory(euler)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/euler_device/CMakeLists.txt")
  add_subdirectory(euler_device)
endif()

if(NOT MFEM_VERSION_INT EQUAL 40900)
  message(FATAL_ERROR "HDG oracle requires immutable MFEM v4.9, got ${MFEM_VERSION}")
endif()
if(NOT MFEM_GIT_STRING STREQUAL
   "tags/v4.9-0-gd9d6526cc1749980a2ba1da16e2c1ca1e07d82ec")
  message(FATAL_ERROR "HDG oracle requires pinned MFEM commit d9d6526")
endif()
if(NOT MFEM_USE_MPI)
  message(STATUS "Skipping HDG CPU oracle: MFEM MPI support is unavailable")
  return()
endif()
if(MFEM_USE_CUDA OR MFEM_USE_HIP OR MFEM_USE_RAJA OR MFEM_USE_UMPIRE)
  message(STATUS "Skipping HDG CPU oracle for accelerator-enabled MFEM")
  return()
endif()

add_executable(hdg_oracle
  hdg_oracle_main.cpp
  ../../src/numerics/hdg/scalar_hdg_oracle.cpp)
target_include_directories(hdg_oracle PRIVATE ../../include)
target_link_libraries(hdg_oracle PRIVATE mfem MPI::MPI_CXX)
target_compile_options(hdg_oracle PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(hdg_oracle PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(hdg_registered_run_root
      "${CMAKE_CURRENT_BINARY_DIR}/registered-runs")
  add_test(NAME dg_igr_hdg_oracle_serial_cpu
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run_registered_oracle.sh"
            serial 1 $<TARGET_FILE:hdg_oracle>
            "${hdg_registered_run_root}")
  set_tests_properties(dg_igr_hdg_oracle_serial_cpu PROPERTIES
    LABELS "hdg;cpu;oracle"
    TIMEOUT 180)

  foreach(hdg_mpi_ranks IN ITEMS 1 2)
    add_test(NAME dg_igr_hdg_oracle_mpi_${hdg_mpi_ranks}_cpu
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run_registered_oracle.sh"
              mpi ${hdg_mpi_ranks} $<TARGET_FILE:hdg_oracle>
              "${hdg_registered_run_root}"
              "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}")
    set_tests_properties(dg_igr_hdg_oracle_mpi_${hdg_mpi_ranks}_cpu PROPERTIES
      LABELS "hdg;cpu;oracle;mpi"
      PROCESSORS ${hdg_mpi_ranks}
      RESOURCE_LOCK dg_igr_sm120
      SKIP_RETURN_CODE 75
      TIMEOUT 180)
  endforeach()
endif()
