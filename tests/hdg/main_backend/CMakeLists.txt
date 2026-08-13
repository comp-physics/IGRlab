if(NOT TARGET dg_igr_hdg_main_backend_cpu OR
   NOT TARGET dgsem_smooth_advection_periodic)
  message(FATAL_ERROR
    "The generic CPU HDG main backend and opted-in example are required.")
endif()

foreach(_ranks IN ITEMS 1 2)
  add_test(NAME dg_igr_hdg_main_backend_smooth_advection_cpu_rank${_ranks}
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/rank${_ranks}-output"
      "-DHDG_MAIN_RANKS=${_ranks}"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_smooth_advection_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_smooth_advection_cpu_rank${_ranks} PROPERTIES
      LABELS "hdg;main-backend;cpu;mpi;process"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK hdg_generic_euler_cpu
      TIMEOUT 1200)
endforeach()

add_test(NAME dg_igr_hdg_main_backend_preserves_dgsem_cpu_serial
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_WORK=${CMAKE_CURRENT_BINARY_DIR}/dgsem-compat"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_dgsem_compat_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_preserves_dgsem_cpu_serial PROPERTIES
    LABELS "hdg;main-backend;cpu;compatibility;dgsem;process"
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 300)
