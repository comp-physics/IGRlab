add_executable(dg_igr_hdg_runtime_cpu_test
  independent_oracle.cpp
  test_persistent_smooth_euler.cpp)

target_compile_features(dg_igr_hdg_runtime_cpu_test PRIVATE cxx_std_17)
target_compile_options(dg_igr_hdg_runtime_cpu_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_runtime_cpu_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
target_include_directories(dg_igr_hdg_runtime_cpu_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(dg_igr_hdg_runtime_cpu_test PRIVATE
  dg_igr_hdg_runtime_cpu
  mfem
  MPI::MPI_CXX)

add_test(NAME hdg_runtime_cpu_serial
  COMMAND $<TARGET_FILE:dg_igr_hdg_runtime_cpu_test>)
set_tests_properties(hdg_runtime_cpu_serial PROPERTIES
  LABELS "hdg;runtime;cpu;serial"
  PROCESSORS 1
  RESOURCE_LOCK hdg_runtime_cpu
  TIMEOUT 1800)

add_test(NAME hdg_runtime_cpu_rank2
  COMMAND ${MPIEXEC_EXECUTABLE}
          ${MPIEXEC_NUMPROC_FLAG} 2
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:dg_igr_hdg_runtime_cpu_test>
          ${MPIEXEC_POSTFLAGS})
set_tests_properties(hdg_runtime_cpu_rank2 PROPERTIES
  LABELS "hdg;runtime;cpu;rank2"
  PROCESSORS 2
  RESOURCE_LOCK hdg_runtime_cpu
  TIMEOUT 1800)
