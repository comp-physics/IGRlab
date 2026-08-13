if(NOT TARGET hdg_smooth_periodic)
  message(FATAL_ERROR
          "The CPU-serial hdg_smooth_periodic application target is required.")
endif()

add_test(NAME dg_igr_hdg_smooth_periodic_cpu_serial
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-serial-output"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_serial_process.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cpu_serial PROPERTIES
  LABELS "hdg;main-adapter;cpu;serial;process"
  RESOURCE_LOCK hdg_runtime_cpu
  TIMEOUT 1200)

add_test(NAME dg_igr_hdg_smooth_periodic_cpu_rank2_rejected
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rank2-output"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_SMOOTH_PERIODIC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_rank2_rejected.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cpu_rank2_rejected PROPERTIES
  LABELS "hdg;main-adapter;cpu;mpi;process"
  PROCESSORS 2
  TIMEOUT 60)
