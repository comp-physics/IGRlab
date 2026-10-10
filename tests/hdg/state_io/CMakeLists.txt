if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(dg_igr_hdg_state_io_tests LANGUAGES CXX)

  find_package(MFEM 4.9 CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

get_filename_component(DG_IGR_HDG_REPOSITORY_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

add_executable(dg_igr_hdg_state_io_test test_state_io.cpp)
target_include_directories(dg_igr_hdg_state_io_test PRIVATE
  "${DG_IGR_HDG_REPOSITORY_ROOT}/include")
target_link_libraries(dg_igr_hdg_state_io_test PRIVATE mfem MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_state_io_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_state_io_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME dg_igr_hdg_state_io_rank1 COMMAND dg_igr_hdg_state_io_test)
  set_tests_properties(dg_igr_hdg_state_io_rank1 PROPERTIES
    LABELS "hdg;state-io;cpu;rank1"
    TIMEOUT 30)
  add_test(NAME dg_igr_hdg_state_io_rank2
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_hdg_state_io_test>
            ${MPIEXEC_POSTFLAGS})
  set_tests_properties(dg_igr_hdg_state_io_rank2 PROPERTIES
    LABELS "hdg;state-io;cpu;rank2;mpi"
    PROCESSORS 2
    TIMEOUT 30)
endif()
