if(NOT TARGET dgsem_smooth_advection_periodic)
  message(FATAL_ERROR
    "The opted-in generic HDG main-backend example is required.")
endif()

if(NOT MFEM_USE_CUDA)
if(NOT TARGET dg_igr_hdg_main_backend_cpu)
  message(FATAL_ERROR "The generic CPU HDG main backend is required.")
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
endif()

if(MFEM_USE_CUDA)
  if(NOT TARGET dg_igr_hdg_main_backend_cuda OR
     NOT TARGET dgsem_smooth_advection_periodic)
    message(FATAL_ERROR
      "The CUDA generic HDG main backend and opted-in example are required.")
  endif()

  foreach(_ranks IN ITEMS 1 2)
    add_test(NAME dg_igr_hdg_main_backend_cuda_rank${_ranks}_n3
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank${_ranks}-n3-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(dg_igr_hdg_main_backend_cuda_rank${_ranks}_n3 PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
  endforeach()

  # Arbitrary fixed dt must clip both output and final boundaries without the
  # old .05/integral-grid admission restriction.
  add_test(NAME dg_igr_hdg_main_backend_cuda_rank1_nonintegral_dt
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank1-nonintegral-dt-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_DT=0.03"
      "-DHDG_MAIN_T_FINAL=0.10"
      "-DHDG_MAIN_DT_SAVE=0.05"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(dg_igr_hdg_main_backend_cuda_rank1_nonintegral_dt PROPERTIES
    LABELS "hdg;main-backend;cuda;time;clipping" RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1200)

  foreach(_ranks IN ITEMS 1 2)
    add_test(NAME dg_igr_hdg_main_backend_cuda_rank${_ranks}_cfl
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank${_ranks}-cfl-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_DT=0"
        "-DHDG_MAIN_CFL=0.25"
        "-DHDG_MAIN_T_FINAL=0.001"
        "-DHDG_MAIN_EXPECTED_STEPS=1"
        "-DHDG_MAIN_NO_VTK=ON"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(dg_igr_hdg_main_backend_cuda_rank${_ranks}_cfl PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed;cfl"
      PROCESSORS ${_ranks} RESOURCE_LOCK dg_igr_sm120 TIMEOUT 1200)
  endforeach()

  foreach(_ranks IN ITEMS 1 2)
    add_test(NAME dg_igr_hdg_main_backend_cuda_rank${_ranks}_n6
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank${_ranks}-n6-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh"
        "-DHDG_MAIN_ORDER=2"
        "-DHDG_MAIN_N=6"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(dg_igr_hdg_main_backend_cuda_rank${_ranks}_n6 PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed;p2;n6"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
  endforeach()

  foreach(_refine_mode IN ITEMS serial parallel)
    if(_refine_mode STREQUAL "serial")
      set(_serial_refine 1)
      set(_parallel_refine 0)
      set(_ranks 1)
    else()
      set(_serial_refine 0)
      set(_parallel_refine 1)
      set(_ranks 2)
    endif()
    add_test(NAME dg_igr_hdg_main_backend_cuda_${_refine_mode}_refine
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-${_refine_mode}-refine-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_SERIAL_REFINE=${_serial_refine}"
        "-DHDG_MAIN_PARALLEL_REFINE=${_parallel_refine}"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(dg_igr_hdg_main_backend_cuda_${_refine_mode}_refine PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed;refinement"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
  endforeach()

  add_test(NAME dg_igr_hdg_main_backend_cuda_rank1_n3_no_vtk
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-rank1-n3-no-vtk-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_MAIN_NO_VTK=ON"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(dg_igr_hdg_main_backend_cuda_rank1_n3_no_vtk PROPERTIES
    LABELS "hdg;main-backend;cuda;rank1;process;generic-borrowed;no-vtk"
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1200)

  add_test(NAME dg_igr_hdg_main_backend_cuda_rejects_cpu_profile
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_reject_process.cmake")
  set_tests_properties(dg_igr_hdg_main_backend_cuda_rejects_cpu_profile PROPERTIES
    LABELS "hdg;main-backend;cuda;negative;profile"
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 300)

  add_test(NAME dg_igr_hdg_main_backend_preserves_dgsem_cuda_profile
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_WORK=${CMAKE_CURRENT_BINARY_DIR}/dgsem-cuda-compat"
      "-DHDG_MAIN_DEVICE=cuda"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_dgsem_compat_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_preserves_dgsem_cuda_profile PROPERTIES
      LABELS "hdg;main-backend;cuda;compatibility;dgsem;process"
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 300)
endif()
