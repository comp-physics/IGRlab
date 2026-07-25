add_executable(dg_igr_hdg_trace_mpi_test test_trace_mpi.cpp)
target_include_directories(dg_igr_hdg_trace_mpi_test PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  target_sources(dg_igr_hdg_trace_mpi_test PRIVATE
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/trace/trace_mpi.cpp)
  target_link_libraries(dg_igr_hdg_trace_mpi_test PRIVATE mfem MPI::MPI_CXX)
else()
  target_link_libraries(dg_igr_hdg_trace_mpi_test PRIVATE igr_mfem mfem MPI::MPI_CXX)
endif()
target_compile_options(dg_igr_hdg_trace_mpi_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_trace_mpi_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_executable(dg_igr_hdg_trace_dg_mpi_test test_trace_dg_mpi.cpp)
target_include_directories(dg_igr_hdg_trace_dg_mpi_test PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  target_sources(dg_igr_hdg_trace_dg_mpi_test PRIVATE
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/trace/trace_mpi.cpp
    ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/trace/trace_dg_mpi.cpp)
  target_link_libraries(dg_igr_hdg_trace_dg_mpi_test PRIVATE mfem MPI::MPI_CXX)
else()
  target_link_libraries(dg_igr_hdg_trace_dg_mpi_test PRIVATE
    igr_mfem mfem MPI::MPI_CXX)
endif()
target_compile_options(dg_igr_hdg_trace_dg_mpi_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_trace_dg_mpi_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_library(dg_igr_hdg_trace_dg_mpi_cxx17_compile OBJECT
  ${DG_IGR_HDG_REPOSITORY_ROOT}/src/numerics/hdg/trace/trace_dg_mpi.cpp)
target_include_directories(dg_igr_hdg_trace_dg_mpi_cxx17_compile PRIVATE
  ${DG_IGR_HDG_REPOSITORY_ROOT}/include)
target_link_libraries(dg_igr_hdg_trace_dg_mpi_cxx17_compile PRIVATE
  mfem MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_trace_dg_mpi_cxx17_compile PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_trace_dg_mpi_cxx17_compile PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_test(NAME dg_igr_hdg_trace_mpi_serial_cpu COMMAND dg_igr_hdg_trace_mpi_test)
  set_tests_properties(dg_igr_hdg_trace_mpi_serial_cpu PROPERTIES
    LABELS "hdg;trace;cpu" TIMEOUT 30)

  add_test(NAME dg_igr_hdg_trace_mpi_2_cpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:dg_igr_hdg_trace_mpi_test>)
  set_tests_properties(dg_igr_hdg_trace_mpi_2_cpu PROPERTIES
    LABELS "hdg;trace;cpu;mpi" PROCESSORS 2 RESOURCE_LOCK dg_igr_sm120
    SKIP_RETURN_CODE 75 TIMEOUT 60)

  add_test(NAME dg_igr_hdg_trace_dg_mpi_serial_cpu
    COMMAND dg_igr_hdg_trace_dg_mpi_test)
  set_tests_properties(dg_igr_hdg_trace_dg_mpi_serial_cpu PROPERTIES
    LABELS "hdg;trace;dg-interface;cpu" TIMEOUT 30)

  add_test(NAME dg_igr_hdg_trace_dg_mpi_2_cpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:dg_igr_hdg_trace_dg_mpi_test>)
  set_tests_properties(dg_igr_hdg_trace_dg_mpi_2_cpu PROPERTIES
    LABELS "hdg;trace;dg-interface;cpu;mpi" PROCESSORS 2
    RESOURCE_LOCK dg_igr_sm120 SKIP_RETURN_CODE 75 TIMEOUT 60)
endif()
