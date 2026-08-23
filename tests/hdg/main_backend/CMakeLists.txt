if(NOT TARGET dgsem_smooth_advection_periodic)
  message(FATAL_ERROR
    "The opted-in generic HDG main-backend example is required.")
endif()

if(NOT MFEM_USE_CUDA)
if(NOT TARGET dg_igr_hdg_main_backend_cpu)
  message(FATAL_ERROR "The generic CPU HDG main backend is required.")
endif()

add_executable(dg_igr_hdg_main_backend_collective_projection_test
  test_collective_projection.cpp)
target_compile_definitions(dg_igr_hdg_main_backend_collective_projection_test PRIVATE
  DG_IGR_HDG_REPOSITORY_ROOT="${PROJECT_SOURCE_DIR}")
target_link_libraries(dg_igr_hdg_main_backend_collective_projection_test PRIVATE
  dg_igr_hdg_main_backend_cpu MPI::MPI_CXX)
target_compile_options(dg_igr_hdg_main_backend_collective_projection_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
set_target_properties(dg_igr_hdg_main_backend_collective_projection_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)

add_test(NAME dg_igr_hdg_main_backend_collective_projection_rejected_rank2
  COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:dg_igr_hdg_main_backend_collective_projection_test>
          ${MPIEXEC_POSTFLAGS})
set_tests_properties(
  dg_igr_hdg_main_backend_collective_projection_rejected_rank2 PROPERTIES
    LABELS "hdg;main-backend;cpu;mpi;negative;collective;projection"
    PROCESSORS 2
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 60)

# Exercise the CPU application entry's committed-state CFL proposal, shared
# time planner, clipped final step, and transactional no-output path.  The
# matching CUDA gates use this same process fixture below.
add_test(NAME dg_igr_hdg_main_backend_cpu_rank2_cfl
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rank2-cfl-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=2"
    "-DHDG_MAIN_DT=0"
    "-DHDG_MAIN_CFL=0.25"
    "-DHDG_MAIN_T_FINAL=0.02"
    "-DHDG_MAIN_EXPECTED_STEPS=3"
    "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.02"
    "-DHDG_MAIN_NO_VTK=ON"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(dg_igr_hdg_main_backend_cpu_rank2_cfl PROPERTIES
  LABELS "hdg;main-backend;cpu;mpi;process;generic-borrowed;cfl"
  PROCESSORS 2
  RESOURCE_LOCK hdg_generic_euler_cpu
  TIMEOUT 1200)

if(NOT TARGET dgsem_smooth_advection OR
   NOT TARGET hdg_smooth_transition_research)
  message(FATAL_ERROR
    "The physical-boundary and smooth-transition research cases are required.")
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

add_test(NAME dg_igr_hdg_main_backend_igr_vtk_cpu_rank1
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-igr-vtk-rank1-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_REGULARIZATION=igr"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=1"
    "-DHDG_MAIN_DT=0.00001"
    "-DHDG_MAIN_T_FINAL=0.00001"
    "-DHDG_MAIN_DT_SAVE=0.00001"
    "-DHDG_MAIN_EXPECTED_STEPS=1"
    "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-05"
    "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001"
    "-DHDG_MAIN_EXPECTED_TIMES=0,1e-05"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(dg_igr_hdg_main_backend_igr_vtk_cpu_rank1 PROPERTIES
  LABELS "hdg;main-backend;cpu;igr;vtk;fields;process"
  PROCESSORS 1
  RESOURCE_LOCK hdg_generic_euler_cpu
  TIMEOUT 1200)

foreach(_ranks IN ITEMS 1 2)
  add_test(NAME dg_igr_hdg_main_backend_igr_cpu_rank${_ranks}_n3
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_RANKS=${_ranks}"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_igr_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_igr_cpu_rank${_ranks}_n3 PROPERTIES
      LABELS "hdg;main-backend;cpu;igr;mpi;process;periodic;n3;no-vtk"
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

add_test(NAME dg_igr_hdg_main_backend_representative_cases_cpu_serial
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_PERIODIC_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_NONPERIODIC_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection>"
    "-DHDG_REPRESENTATIVE_WORK=${CMAKE_CURRENT_BINARY_DIR}/representative-cases"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_representative_case_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_representative_cases_cpu_serial PROPERTIES
    LABELS "hdg;main-backend;cpu;source;boundary;igr;process"
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 1200)

foreach(_ranks IN ITEMS 1 2)
  add_test(NAME dg_igr_hdg_main_backend_w8c_research_cpu_rank${_ranks}
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_W8C_EXECUTABLE=$<TARGET_FILE:hdg_smooth_transition_research>"
      "-DHDG_W8C_WORK=${CMAKE_CURRENT_BINARY_DIR}/w8c-smooth-transition-rank${_ranks}"
      "-DHDG_W8C_RANKS=${_ranks}"
      "-DHDG_W8C_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_W8C_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_W8C_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_W8C_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_smooth_transition_research_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_w8c_research_cpu_rank${_ranks} PROPERTIES
      LABELS "hdg;main-backend;cpu;igr;w8c;research;process;periodic;n8;smoke"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK hdg_generic_euler_cpu
      TIMEOUT 1200)
endforeach()
endif()

if(MFEM_USE_CUDA)
  if(NOT TARGET dg_igr_hdg_main_backend_cuda OR
     NOT TARGET dgsem_smooth_advection_periodic OR
     NOT TARGET dgsem_smooth_advection)
    message(FATAL_ERROR
      "The CUDA generic HDG main backend and periodic/physical examples are required.")
  endif()

  # These are real application-entry gates, not direct backend fixtures.  The
  # periodic case carries an IGR-augmented source and the physical case covers
  # each of its selectable Euler boundary plans on the same CUDA HDG lifecycle.
  # Explicit small meshes keep every rank within the bounded CUDA arena.
  foreach(_ranks IN ITEMS 1 2)
    add_test(NAME dg_igr_hdg_main_backend_representative_cases_cuda_rank${_ranks}_p1
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_PERIODIC_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_NONPERIODIC_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection>"
        "-DHDG_REPRESENTATIVE_WORK=${CMAKE_CURRENT_BINARY_DIR}/cuda-representative-rank${_ranks}"
        "-DHDG_REPRESENTATIVE_DEVICE=cuda"
        "-DHDG_REPRESENTATIVE_RANKS=${_ranks}"
        "-DHDG_REPRESENTATIVE_PERIODIC_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_REPRESENTATIVE_NONPERIODIC_MESH=${CMAKE_CURRENT_SOURCE_DIR}/smooth_advection_physical_n2.mesh"
        "-DHDG_REPRESENTATIVE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_REPRESENTATIVE_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_REPRESENTATIVE_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_REPRESENTATIVE_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_representative_case_process.cmake")
    set_tests_properties(
      dg_igr_hdg_main_backend_representative_cases_cuda_rank${_ranks}_p1 PROPERTIES
        LABELS "hdg;main-backend;cuda;real-application;igr;source;periodic;physical;boundary;p1;no-vtk;mpi"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 3600)
  endforeach()

  # A production-main fixture over a seven-attribute mesh covers every
  # conventional semantic Euler boundary policy in one solve.  IGR remains a
  # request option on that same main backend; no SC25/P1 oracle is linked.
  add_executable(dg_igr_hdg_main_backend_cuda_physical_extension_test
    test_cuda_physical_extension.cu)
  target_include_directories(
    dg_igr_hdg_main_backend_cuda_physical_extension_test PRIVATE
      "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(
    dg_igr_hdg_main_backend_cuda_physical_extension_test PRIVATE
      DG_IGR_HDG_REPOSITORY_ROOT="${PROJECT_SOURCE_DIR}"
      DG_IGR_HDG_CUDA_PHYSICAL_MESH="${CMAKE_CURRENT_SOURCE_DIR}/cuda_physical_boundaries.mesh")
  target_link_libraries(
    dg_igr_hdg_main_backend_cuda_physical_extension_test PRIVATE
      dg_igr_hdg_main_backend_cuda
      igr_mfem
      mfem
      MPI::MPI_CXX)
  target_compile_options(
    dg_igr_hdg_main_backend_cuda_physical_extension_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_test PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_rank1
    COMMAND dg_igr_hdg_main_backend_cuda_physical_extension_test)
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_main_backend_cuda_physical_extension_test>
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_rank1 PROPERTIES
      LABELS "hdg;main-backend;cuda;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_rank2 PROPERTIES
      LABELS "hdg;main-backend;cuda;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace;mpi"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)

  foreach(_ranks IN ITEMS 1 2)
    add_test(NAME dg_igr_hdg_main_backend_cuda_igr_rank${_ranks}_p1_p3
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_igr_process.cmake")
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_igr_rank${_ranks}_p1_p3 PROPERTIES
        LABELS "hdg;main-backend;cuda;igr;periodic;p1-p3;source;initial-trace;mpi;process"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 3600)
  endforeach()

  foreach(_ranks IN ITEMS 1 2)
    if(_ranks EQUAL 1)
      set(_verify_repeatable_output ON)
    else()
      set(_verify_repeatable_output OFF)
    endif()
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
        "-DHDG_MAIN_VERIFY_REPEATABLE_OUTPUT=${_verify_repeatable_output}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(dg_igr_hdg_main_backend_cuda_rank${_ranks}_n3 PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
  endforeach()

  add_test(NAME dg_igr_hdg_main_backend_cuda_igr_vtk_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-igr-vtk-rank1-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_REGULARIZATION=igr"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_DT=0.00001"
      "-DHDG_MAIN_T_FINAL=0.00001"
      "-DHDG_MAIN_DT_SAVE=0.00001"
      "-DHDG_MAIN_EXPECTED_STEPS=1"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-05"
      "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001"
      "-DHDG_MAIN_EXPECTED_TIMES=0,1e-05"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(dg_igr_hdg_main_backend_cuda_igr_vtk_rank1 PROPERTIES
    LABELS "hdg;main-backend;cuda;igr;vtk;fields;process"
    PROCESSORS 1
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1200)

  # The runtime session gate proves exact prepared/committed byte identity and
  # abort/retry preservation.  The dependent main gates prove that those
  # acknowledged bytes are published only after commit; rank 1 also repeats
  # the complete process and compares every serialized output byte.
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_rank1_n3 PROPERTIES
      DEPENDS dg_igr_hdg_runtime_cuda_generic_euler_session_rank1)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_rank2_n3 PROPERTIES
      DEPENDS dg_igr_hdg_runtime_cuda_generic_euler_session_rank2)

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
      "-DHDG_MAIN_DT_SAVE=0.04"
      "-DHDG_MAIN_EXPECTED_STEPS=5"
      "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001,000002,000003"
      "-DHDG_MAIN_EXPECTED_TIMES=0,0.04,0.08,0.1"
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
        "-DHDG_MAIN_T_FINAL=0.02"
        "-DHDG_MAIN_EXPECTED_STEPS=3"
        "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.02"
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

  # Keep p2 serial and parallel refinement as independent process gates.
  # They validate each refined trajectory's own output, never field equality.
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
    add_test(NAME dg_igr_hdg_main_backend_cuda_p2_${_refine_mode}_refine_small_dt
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-p2-${_refine_mode}-refine-small-dt-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh"
        "-DHDG_MAIN_ORDER=2"
        "-DHDG_MAIN_N=6"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_SERIAL_REFINE=${_serial_refine}"
        "-DHDG_MAIN_PARALLEL_REFINE=${_parallel_refine}"
        "-DHDG_MAIN_DT=0.005"
        "-DHDG_MAIN_T_FINAL=0.01"
        "-DHDG_MAIN_DT_SAVE=0.005"
        "-DHDG_MAIN_EXPECTED_STEPS=2"
        "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.01"
        "-DHDG_MAIN_EXPECTED_TIMES=0,0.005,0.01"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_p2_${_refine_mode}_refine_small_dt PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed;p2;refinement"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
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
