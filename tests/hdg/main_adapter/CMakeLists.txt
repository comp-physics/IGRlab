if(NOT TARGET hdg_smooth_periodic)
  message(FATAL_ERROR
          "The hdg_smooth_periodic application target is required.")
endif()

if(NOT MFEM_USE_CUDA)
add_test(NAME dg_igr_hdg_smooth_periodic_cpu_serial
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-serial-output"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_serial_process.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cpu_serial PROPERTIES
  LABELS "hdg;main-adapter;cpu;serial;process"
  RESOURCE_LOCK hdg_runtime_cpu
  TIMEOUT 1200)

add_test(NAME dg_igr_hdg_smooth_periodic_cpu_rank2
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rank2-output"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_SMOOTH_PERIODIC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_rank2_process.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cpu_rank2 PROPERTIES
  LABELS "hdg;main-adapter;cpu;mpi;process"
  PROCESSORS 2
  RESOURCE_LOCK hdg_runtime_cpu
  TIMEOUT 1200)

add_test(NAME dg_igr_hdg_smooth_periodic_cpu_rank3_rejected
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rank3-output"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_SMOOTH_PERIODIC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_rank3_rejected.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cpu_rank3_rejected PROPERTIES
  LABELS "hdg;main-adapter;cpu;mpi;process;negative"
  PROCESSORS 3
  TIMEOUT 60)
else()
add_test(NAME dg_igr_hdg_smooth_periodic_cuda_rank1
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank1-output"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_rank1_process.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cuda_rank1 PROPERTIES
  LABELS "hdg;main-adapter;cuda;rank1;process"
  PROCESSORS 1
  RESOURCE_LOCK dg_igr_sm120
  TIMEOUT 1200)

add_test(NAME dg_igr_hdg_smooth_periodic_cuda_rank2
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_SMOOTH_PERIODIC_EXECUTABLE=$<TARGET_FILE:hdg_smooth_periodic>"
    "-DHDG_SMOOTH_PERIODIC_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank2-output"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_SMOOTH_PERIODIC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_SMOOTH_PERIODIC_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_rank2_process.cmake")
set_tests_properties(dg_igr_hdg_smooth_periodic_cuda_rank2 PROPERTIES
  LABELS "hdg;main-adapter;cuda;rank2;process"
  PROCESSORS 2
  RESOURCE_LOCK dg_igr_sm120
  TIMEOUT 1200)
endif()
