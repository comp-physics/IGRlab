get_filename_component(DG_IGR_HDG_EULER_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

option(DG_IGR_ENABLE_HDG_EULER_EXPERIMENT
  "Build the unregistered serial smooth-Euler HDG experiment" OFF)
if(NOT DG_IGR_ENABLE_HDG_EULER_EXPERIMENT)
  return()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MFEM 4.9 CONFIG REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)
elseif(NOT TARGET mfem)
  message(FATAL_ERROR "smooth-Euler HDG requires the parent MFEM discovery")
endif()

if(MFEM_USE_CUDA OR MFEM_USE_HIP)
  message(FATAL_ERROR
    "DG_IGR_ENABLE_HDG_EULER_EXPERIMENT=ON requires a non-CUDA, non-HIP MFEM")
endif()

add_executable(dg_igr_hdg_smooth_euler_cpu_test
  test_smooth_euler.cpp
  ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/euler/smooth_euler.cpp)
target_include_directories(dg_igr_hdg_smooth_euler_cpu_test PRIVATE
  ${DG_IGR_HDG_EULER_ROOT}/include)
target_compile_options(dg_igr_hdg_smooth_euler_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_smooth_euler_cpu_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  target_sources(dg_igr_hdg_smooth_euler_cpu_test PRIVATE
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/core/host_reference.cpp)
  target_link_libraries(dg_igr_hdg_smooth_euler_cpu_test PRIVATE mfem)
else()
  target_link_libraries(dg_igr_hdg_smooth_euler_cpu_test PRIVATE igr_mfem mfem)
  add_test(NAME dg_igr_hdg_smooth_euler_serial_cpu
    COMMAND dg_igr_hdg_smooth_euler_cpu_test)
  set_tests_properties(dg_igr_hdg_smooth_euler_serial_cpu PROPERTIES
    LABELS "hdg;euler;cpu;experimental;serial" TIMEOUT 600)
endif()

add_executable(dg_igr_hdg_smooth_euler_mpi_cpu_test
  test_smooth_euler_mpi.cpp)
target_include_directories(dg_igr_hdg_smooth_euler_mpi_cpu_test PRIVATE
  ${DG_IGR_HDG_EULER_ROOT}/include)
target_compile_options(dg_igr_hdg_smooth_euler_mpi_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_smooth_euler_mpi_cpu_test PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  target_sources(dg_igr_hdg_smooth_euler_mpi_cpu_test PRIVATE
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/core/host_reference.cpp
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/euler/smooth_euler.cpp
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/euler/smooth_euler_mpi.cpp
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/trace/trace_mpi.cpp
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/trace/trace_dg_mpi.cpp)
  target_link_libraries(dg_igr_hdg_smooth_euler_mpi_cpu_test PRIVATE
    mfem MPI::MPI_CXX)
else()
  target_link_libraries(dg_igr_hdg_smooth_euler_mpi_cpu_test PRIVATE
    igr_mfem mfem MPI::MPI_CXX)

  add_library(dg_igr_hdg_smooth_euler_mpi_cxx17_compile OBJECT
    ${DG_IGR_HDG_EULER_ROOT}/src/numerics/hdg/euler/smooth_euler_mpi.cpp)
  target_include_directories(dg_igr_hdg_smooth_euler_mpi_cxx17_compile PRIVATE
    ${DG_IGR_HDG_EULER_ROOT}/include)
  target_link_libraries(dg_igr_hdg_smooth_euler_mpi_cxx17_compile PRIVATE
    mfem MPI::MPI_CXX)
  target_compile_options(dg_igr_hdg_smooth_euler_mpi_cxx17_compile PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
  set_target_properties(dg_igr_hdg_smooth_euler_mpi_cxx17_compile PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

  add_test(NAME dg_igr_hdg_smooth_euler_mpi_serial_cpu
    COMMAND dg_igr_hdg_smooth_euler_mpi_cpu_test)
  set_tests_properties(dg_igr_hdg_smooth_euler_mpi_serial_cpu PROPERTIES
    LABELS "hdg;euler;cpu;experimental;serial;mpi-transaction"
    TIMEOUT 600)

  add_test(NAME dg_igr_hdg_smooth_euler_mpi_2_cpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:dg_igr_hdg_smooth_euler_mpi_cpu_test>)
  set_tests_properties(dg_igr_hdg_smooth_euler_mpi_2_cpu PROPERTIES
    LABELS "hdg;euler;cpu;experimental;mpi;rank2"
    PROCESSORS 2 RESOURCE_LOCK dg_igr_sm120 SKIP_RETURN_CODE 75 TIMEOUT 600)
endif()

# The production source glob and root build.sh default-off seam are
# coordinator-owned; this leaf only isolates its own registration.
