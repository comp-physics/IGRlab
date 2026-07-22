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

# These checks intentionally compile invalid translation units at configure
# time.  A CTest WILL_FAIL command can pass for an unrelated command failure;
# try_compile proves that the contract's static assertions reject each case.
function(dg_igr_expect_hdg_contract_compile_failure name source)
  try_compile(dg_igr_hdg_contract_rejected
    "${CMAKE_CURRENT_BINARY_DIR}/negative-contract/${name}"
    "${CMAKE_CURRENT_SOURCE_DIR}/negative"
    dg_igr_hdg_contract_negative
    dg_igr_hdg_contract_negative
    CMAKE_FLAGS
      "-DHDG_NEGATIVE_SOURCE:FILEPATH=${CMAKE_CURRENT_SOURCE_DIR}/negative/${source}"
      "-DDG_IGR_REPOSITORY_ROOT:PATH=${DG_IGR_HDG_REPOSITORY_ROOT}"
      "-DCMAKE_TRY_COMPILE_TARGET_TYPE:STRING=STATIC_LIBRARY"
    OUTPUT_VARIABLE dg_igr_hdg_contract_negative_output)
  if(dg_igr_hdg_contract_rejected)
    message(FATAL_ERROR
      "HDG contract negative case '${name}' unexpectedly compiled:\n"
      "${dg_igr_hdg_contract_negative_output}")
  endif()
endfunction()

foreach(dg_igr_hdg_negative_case IN ITEMS
    invalid_trace.cpp
    duplicate_trace.cpp
    invalid_model.cpp
    invalid_stabilization.cpp
    invalid_time.cpp
    invalid_backend.cpp
    invalid_precision.cpp
    polymorphic_device.cpp
    polymorphic_phase.cpp
    wrong_phase_signature.cpp)
  get_filename_component(dg_igr_hdg_negative_name
    "${dg_igr_hdg_negative_case}" NAME_WE)
  dg_igr_expect_hdg_contract_compile_failure(
    "${dg_igr_hdg_negative_name}" "${dg_igr_hdg_negative_case}")
endforeach()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME dg_igr_hdg_contract_cpu COMMAND dg_igr_hdg_contract_test)
  set_tests_properties(dg_igr_hdg_contract_cpu PROPERTIES
    LABELS "hdg;cpu;contract"
    TIMEOUT 30)
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
