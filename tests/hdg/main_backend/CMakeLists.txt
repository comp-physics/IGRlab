if(NOT TARGET dgsem_smooth_advection_periodic)
  message(FATAL_ERROR
    "The opted-in generic HDG main-backend example is required.")
endif()

# Keep the production-main CFL matrix on the shared process fixture.  Its
# arguments remain explicit so the selected device, CUDA engine,
# regularization mode, polynomial order, and rank count are visible in CTest.
function(add_hdg_main_cfl_process_test test_name output_tag device
         cuda_engine regularization order ranks expected_steps)
  set(_time_integrator backward-euler)
  if(ARGC GREATER 8)
    set(_time_integrator "${ARGV8}")
  endif()
  if(NOT _time_integrator STREQUAL "backward-euler" AND
     NOT _time_integrator STREQUAL "partitioned-backward-euler")
    message(FATAL_ERROR
      "Unknown HDG production-main time integrator '${_time_integrator}'.")
  endif()
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${output_tag}-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=${device}"
      "-DHDG_MAIN_CUDA_ENGINE=${cuda_engine}"
      "-DHDG_MAIN_REGULARIZATION=${regularization}"
      "-DHDG_MAIN_ORDER=${order}"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=${ranks}"
      "-DHDG_MAIN_TIME_INTEGRATOR=${_time_integrator}"
      "-DHDG_MAIN_DT=0"
      "-DHDG_MAIN_CFL=0.25"
      "-DHDG_MAIN_T_FINAL=0.02"
      "-DHDG_MAIN_EXPECTED_STEPS=${expected_steps}"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.02"
      "-DHDG_MAIN_NO_VTK=ON"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")

  if(regularization STREQUAL "igr")
    set(_mode_label igr)
  else()
    set(_mode_label conventional)
  endif()
  if(device STREQUAL "cuda")
    set(_device_labels "cuda;${cuda_engine}")
    set(_resource_lock dg_igr_sm120)
  else()
    set(_device_labels cpu)
    set(_resource_lock hdg_generic_euler_cpu)
  endif()
  set(_labels
    hdg main-backend ${_device_labels} ${_mode_label} p${order}
    rank${ranks} mpi process generic-borrowed cfl no-vtk)
  if(_time_integrator STREQUAL "partitioned-backward-euler")
    list(APPEND _labels partitioned-backward-euler)
  endif()
  set_tests_properties(${test_name} PROPERTIES
    LABELS "${_labels}"
    PROCESSORS ${ranks}
    RESOURCE_LOCK ${_resource_lock}
    TIMEOUT 1200)
endfunction()

# Refinement stays in the same application process fixture and intentionally
# publishes VTK: the fixture validates the global refined-cell census as well
# as the committed trajectory and mode-specific Euler/IGR field surface.
function(add_hdg_main_cuda_refinement_process_test test_name output_tag
         cuda_engine regularization order refinement_mode)
  set(_time_integrator backward-euler)
  if(ARGC GREATER 6)
    set(_time_integrator "${ARGV6}")
  endif()
  if(NOT _time_integrator STREQUAL "backward-euler" AND
     NOT _time_integrator STREQUAL "partitioned-backward-euler")
    message(FATAL_ERROR
      "Unknown HDG production-main time integrator '${_time_integrator}'.")
  endif()
  if(refinement_mode STREQUAL "serial")
    set(_serial_refine 1)
    set(_parallel_refine 0)
    set(_ranks 1)
  elseif(refinement_mode STREQUAL "parallel")
    set(_serial_refine 0)
    set(_parallel_refine 1)
    set(_ranks 2)
  else()
    message(FATAL_ERROR
      "Unknown HDG production-main refinement mode '${refinement_mode}'.")
  endif()
  if(regularization STREQUAL "igr")
    set(_mode_label igr)
  else()
    set(_mode_label conventional)
  endif()

  add_test(NAME ${test_name}
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${output_tag}-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=${cuda_engine}"
      "-DHDG_MAIN_REGULARIZATION=${regularization}"
      "-DHDG_MAIN_ORDER=${order}"
      "-DHDG_MAIN_N=6"
      "-DHDG_MAIN_RANKS=${_ranks}"
      "-DHDG_MAIN_SERIAL_REFINE=${_serial_refine}"
      "-DHDG_MAIN_PARALLEL_REFINE=${_parallel_refine}"
      "-DHDG_MAIN_TIME_INTEGRATOR=${_time_integrator}"
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
  set(_labels
    hdg main-backend cuda ${cuda_engine} ${_mode_label} p${order}
    rank${_ranks} ${refinement_mode}-refinement mpi process generic-borrowed
    fixed-step vtk refinement)
  if(_time_integrator STREQUAL "partitioned-backward-euler")
    list(APPEND _labels partitioned-backward-euler)
  endif()
  set_tests_properties(${test_name} PROPERTIES
    LABELS "${_labels}"
    PROCESSORS ${_ranks}
    RESOURCE_LOCK dg_igr_sm120
    TIMEOUT 1200)
endfunction()

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
# matching CUDA gates use this same process fixture below.  Rank one spans the
# supported p1-p3 selection; the established rank-two p1 representatives keep
# their original CTest names.  For this n3 fixture, the production proposal
# cfl*hmin/((2*p+1)*maximum_wave_speed) gives 3/4/6 accepted steps at p1/p2/p3,
# including the planner's clipped final step.
foreach(_order IN ITEMS 1 2 3)
  if(_order EQUAL 1)
    set(_expected_cfl_steps 3)
  elseif(_order EQUAL 2)
    set(_expected_cfl_steps 4)
  else()
    set(_expected_cfl_steps 6)
  endif()
  add_hdg_main_cfl_process_test(
    dg_igr_hdg_main_backend_cpu_conventional_p${_order}_rank1_cfl
    cpu-conventional-p${_order}-rank1-cfl
    cpu ordinary none ${_order} 1 ${_expected_cfl_steps})
  add_hdg_main_cfl_process_test(
    dg_igr_hdg_main_backend_cpu_igr_p${_order}_rank1_cfl
    cpu-igr-p${_order}-rank1-cfl
    cpu ordinary igr ${_order} 1 ${_expected_cfl_steps})
endforeach()

add_hdg_main_cfl_process_test(
  dg_igr_hdg_main_backend_cpu_rank2_cfl
  cpu-rank2-cfl
  cpu ordinary none 1 2 3)
add_hdg_main_cfl_process_test(
  dg_igr_hdg_main_backend_cpu_igr_rank2_cfl
  cpu-igr-rank2-cfl
  cpu ordinary igr 1 2 3)

# The selected partitioned Backward-Euler architecture uses this CPU p2
# application gate as the numerical-reference counterpart to the bounded CUDA
# winner gates below.  Existing Backward-Euler registrations remain unchanged.
add_hdg_main_cfl_process_test(
  dg_igr_hdg_main_backend_cpu_partitioned_backward_euler_igr_p2_rank1_cfl
  cpu-partitioned-backward-euler-igr-p2-rank1-cfl
  cpu ordinary igr 2 1 4 partitioned-backward-euler)

add_test(NAME dg_igr_hdg_main_backend_cpu_rodas3_fixed_rank1
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rodas3-fixed-rank1-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_REGULARIZATION=none"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=1"
    "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
    "-DHDG_MAIN_DT=0.00001"
    "-DHDG_MAIN_T_FINAL=0.00001"
    "-DHDG_MAIN_EXPECTED_STEPS=1"
    "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-05"
    "-DHDG_MAIN_NO_VTK=ON"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(dg_igr_hdg_main_backend_cpu_rodas3_fixed_rank1 PROPERTIES
  LABELS "hdg;main-backend;cpu;rodas3;fixed-step;process"
  PROCESSORS 1
  RESOURCE_LOCK hdg_generic_euler_cpu
  TIMEOUT 1200)

add_test(NAME dg_igr_hdg_main_backend_cpu_backward_euler_rejects_zhang_shu_rank1
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-backward-euler-rejects-zhang-shu-rank1-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_REGULARIZATION=none"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=1"
    "-DHDG_MAIN_TIME_INTEGRATOR=backward-euler"
    "-DHDG_MAIN_POSITIVITY_LIMITER=zhang-shu"
    "-DHDG_MAIN_DT=0.00001"
    "-DHDG_MAIN_T_FINAL=0.00001"
    "-DHDG_MAIN_NO_VTK=ON"
    "-DHDG_MAIN_EXPECT_REJECTION_PATTERN=RODAS3-only typed opt-in"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_cpu_backward_euler_rejects_zhang_shu_rank1
  PROPERTIES
    LABELS "hdg;main-backend;cpu;backward-euler;zhang-shu;negative;process"
    PROCESSORS 1
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 1200)

add_test(NAME dg_igr_hdg_main_backend_cpu_rejects_conflicting_c_alpha_rank1
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rejects-conflicting-c-alpha-rank1-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_REGULARIZATION=igr"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=1"
    "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
    "-DHDG_MAIN_C_ALPHA=0.5"
    "-DHDG_MAIN_HDG_IGR_ALPHA=1.0"
    "-DHDG_MAIN_DT=0.00001"
    "-DHDG_MAIN_T_FINAL=0.00001"
    "-DHDG_MAIN_NO_VTK=ON"
    "-DHDG_MAIN_EXPECT_REJECTION_PATTERN=Conflicting HDG IGR C_alpha values"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_cpu_rejects_conflicting_c_alpha_rank1
  PROPERTIES
    LABELS "hdg;main-backend;cpu;igr;c-alpha;negative;process"
    PROCESSORS 1
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 1200)

add_test(NAME dg_igr_hdg_main_backend_cpu_rodas3_adaptive_rejection_rank1
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rodas3-adaptive-rejection-rank1-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_REGULARIZATION=none"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=1"
    "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
    "-DHDG_MAIN_POSITIVITY_LIMITER=zhang-shu"
    "-DHDG_MAIN_ZHANG_SHU_POINTS=quadrature"
    "-DHDG_MAIN_EXPECT_POSITIVITY_LIMITER=zhang-shu"
    "-DHDG_MAIN_EXPECT_ZHANG_SHU_POINTS=quadrature"
    "-DHDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS=ON"
    "-DHDG_MAIN_EXPECT_ZHANG_SHU_FIELDS=ON"
    "-DHDG_MAIN_DT=-0.001"
    "-DHDG_MAIN_DT_MIN=1e-8"
    "-DHDG_MAIN_TIME_ATOL=1e-10"
    "-DHDG_MAIN_TIME_RTOL=1e-10"
    "-DHDG_MAIN_T_FINAL=0.001"
    "-DHDG_MAIN_DT_SAVE=0.001"
    "-DHDG_MAIN_EXPECTED_STEPS=[1-9][0-9]*"
    "-DHDG_MAIN_EXPECTED_REJECTIONS=[1-9][0-9]*"
    "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.001"
    "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001"
    "-DHDG_MAIN_EXPECTED_TIMES=0,0.001"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_cpu_rodas3_adaptive_rejection_rank1 PROPERTIES
    LABELS "hdg;main-backend;cpu;rodas3;adaptive;rejection;rollback;output-landing;zhang-shu;limiter-diagnostics;process"
    PROCESSORS 1
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 1200)

# Exercise the production CPU output-presence consensus inside a genuine
# adaptive two-rank transaction.  The first proposal is clipped to the
# intermediate publication boundary and rejected before a smaller accepted
# retry; exact output cycles prove that rollback did not publish or advance.
add_test(NAME dg_igr_hdg_main_backend_cpu_rodas3_adaptive_intermediate_output_rank2
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cpu-rodas3-adaptive-intermediate-output-rank2-output"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    "-DHDG_MAIN_REGULARIZATION=none"
    "-DHDG_MAIN_ORDER=1"
    "-DHDG_MAIN_N=3"
    "-DHDG_MAIN_RANKS=2"
    "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
    "-DHDG_MAIN_DT=-0.001"
    "-DHDG_MAIN_DT_MIN=1e-8"
    "-DHDG_MAIN_TIME_ATOL=1e-10"
    "-DHDG_MAIN_TIME_RTOL=1e-10"
    "-DHDG_MAIN_T_FINAL=0.001"
    "-DHDG_MAIN_DT_SAVE=0.00099"
    "-DHDG_MAIN_EXPECTED_STEPS=[1-9][0-9]*"
    "-DHDG_MAIN_EXPECTED_REJECTIONS=[1-9][0-9]*"
    "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.001"
    "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001,000002"
    "-DHDG_MAIN_EXPECTED_TIMES=0,0.00099,0.001"
    "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
    "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_cpu_rodas3_adaptive_intermediate_output_rank2
  PROPERTIES
    LABELS "hdg;main-backend;cpu;rodas3;rank2;mpi;adaptive;rejection;rollback;output-clipping;intermediate-output;process"
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

add_test(NAME dg_igr_hdg_main_backend_cpu_rejects_unsupported_state_routes
  COMMAND "${CMAKE_COMMAND}"
    "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
    "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
    "-DHDG_MAIN_DEVICE=cpu"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_unsupported_state_process.cmake")
set_tests_properties(
  dg_igr_hdg_main_backend_cpu_rejects_unsupported_state_routes PROPERTIES
    LABELS "hdg;main-backend;cpu;negative;restart;state-file;process"
    PROCESSORS 1
    RESOURCE_LOCK hdg_generic_euler_cpu
    TIMEOUT 300)

if(TARGET sedov_taylor_blast)
  add_test(NAME dg_igr_hdg_main_backend_cpu_rodas3_igr_sedov_p1_rank1_fixed
    COMMAND $<TARGET_FILE:sedov_taylor_blast>
      --discretization hdg
      --device cpu
      --regularization igr
      --time-integrator rodas3
      --positivity-limiter none
      --hdg-igr-alpha 1
      --mesh ${PROJECT_SOURCE_DIR}/examples/sedov_taylor_blast/sedov_taylor.msh
      --order 1
      --serial-refine 0
      --parallel-refine 0
      --time-step 0.000001
      --t-final 0.000001
      --no-vtk_save
      --no-glvis)
  set_tests_properties(
    dg_igr_hdg_main_backend_cpu_rodas3_igr_sedov_p1_rank1_fixed PROPERTIES
      LABELS "hdg;main-backend;cpu;igr;rodas3;sedov;p1;rank1;slip;fixed-step;limiter-off;no-vtk;application"
      PASS_REGULAR_EXPRESSION
        "\\[hdg-main\\] status=Success step=1.*time=1e-06.*time_treatment=rodas3.*regularization=igr.*c_alpha=1"
      PROCESSORS 1
      RESOURCE_LOCK hdg_generic_euler_cpu
      TIMEOUT 1200)
endif()
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
    add_test(NAME dg_igr_hdg_main_backend_representative_cases_cuda_tensor_rank${_ranks}_p1
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_PERIODIC_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_NONPERIODIC_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection>"
        "-DHDG_REPRESENTATIVE_WORK=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-representative-rank${_ranks}"
        "-DHDG_REPRESENTATIVE_DEVICE=cuda"
        "-DHDG_MAIN_CUDA_ENGINE=tensor"
        "-DHDG_REPRESENTATIVE_RANKS=${_ranks}"
        "-DHDG_REPRESENTATIVE_PERIODIC_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_REPRESENTATIVE_NONPERIODIC_MESH=${CMAKE_CURRENT_SOURCE_DIR}/smooth_advection_physical_n2.mesh"
        "-DHDG_REPRESENTATIVE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_REPRESENTATIVE_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_REPRESENTATIVE_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_REPRESENTATIVE_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_representative_case_process.cmake")
    set_tests_properties(
      dg_igr_hdg_main_backend_representative_cases_cuda_tensor_rank${_ranks}_p1 PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;real-application;igr;source;periodic;physical;boundary;p1;no-vtk;mpi"
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
      CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
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
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_tensor_rank1
    COMMAND dg_igr_hdg_main_backend_cuda_physical_extension_test
      --engine tensor)
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_tensor_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_main_backend_cuda_physical_extension_test>
      --engine tensor
      ${MPIEXEC_POSTFLAGS})
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_candidate_rank1
    COMMAND dg_igr_hdg_main_backend_cuda_physical_extension_test
      --candidate-numerical-policy)
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_candidate_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_main_backend_cuda_physical_extension_test>
      --candidate-numerical-policy
      ${MPIEXEC_POSTFLAGS})
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_candidate_tensor_rank1
    COMMAND dg_igr_hdg_main_backend_cuda_physical_extension_test
      --engine tensor --candidate-numerical-policy)
  add_test(NAME dg_igr_hdg_main_backend_cuda_physical_extension_candidate_tensor_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_main_backend_cuda_physical_extension_test>
      --engine tensor --candidate-numerical-policy
      ${MPIEXEC_POSTFLAGS})
  add_test(NAME dg_igr_hdg_main_backend_cuda_engine_mismatch_rank2
    COMMAND
      "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_hdg_main_backend_cuda_physical_extension_test>
      --rank-engine-mismatch
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
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_tensor_rank1 PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_tensor_rank2 PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace;mpi;host-staged"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_candidate_rank1 PROPERTIES
      LABELS "hdg;main-backend;cuda;ordinary;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace;quadrature-candidate;role-tolerance-candidate"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_candidate_rank2 PROPERTIES
      LABELS "hdg;main-backend;cuda;ordinary;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace;quadrature-candidate;role-tolerance-candidate;mpi;host-staged"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_candidate_tensor_rank1 PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace;quadrature-candidate;role-tolerance-candidate;production-trace-policy"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_physical_extension_candidate_tensor_rank2 PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;physical;all-euler-boundaries;igr;conventional;p1-p3;source;initial-trace;quadrature-candidate;role-tolerance-candidate;mpi;host-staged"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 3600)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_engine_mismatch_rank2 PROPERTIES
      LABELS "hdg;main-backend;cuda;negative;consensus;ordinary;tensor;mpi"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 300)

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
    add_test(NAME dg_igr_hdg_main_backend_cuda_igr_tensor_rank${_ranks}_p1_p3
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_CUDA_ENGINE=tensor"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_igr_process.cmake")
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_igr_tensor_rank${_ranks}_p1_p3 PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;igr;periodic;p1-p3;source;initial-trace;mpi;process"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 3600)
  endforeach()

  foreach(_ranks IN ITEMS 1 2)
    if(_ranks EQUAL 1)
      set(_verify_repeatable_output ON)
      set(_omit_cuda_engine ON)
    else()
      set(_verify_repeatable_output OFF)
      set(_omit_cuda_engine OFF)
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
        "-DHDG_MAIN_OMIT_CUDA_ENGINE=${_omit_cuda_engine}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(dg_igr_hdg_main_backend_cuda_rank${_ranks}_n3 PROPERTIES
      LABELS "hdg;main-backend;cuda;mpi;process;generic-borrowed"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
    add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rank${_ranks}_n3
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rank${_ranks}-n3-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_CUDA_ENGINE=tensor"
        "-DHDG_MAIN_RANKS=${_ranks}"
        "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
        "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        "-DHDG_MAIN_VERIFY_REPEATABLE_OUTPUT=${_verify_repeatable_output}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_tensor_rank${_ranks}_n3 PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;conventional;fixed-step;vtk;mpi;process;generic-borrowed"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
  endforeach()

  # Exercise the application-level default rather than only parser/config
  # defaults: the process fixture expects Backward Euler in status telemetry
  # while deliberately omitting --time-integrator from the launched command.
  add_test(NAME dg_igr_hdg_main_backend_cuda_default_backward_euler_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-default-backward-euler-rank1-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=ordinary"
      "-DHDG_MAIN_REGULARIZATION=none"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_TIME_INTEGRATOR=backward-euler"
      "-DHDG_MAIN_OMIT_TIME_INTEGRATOR=ON"
      "-DHDG_MAIN_DT=0.00001"
      "-DHDG_MAIN_T_FINAL=0.00001"
      "-DHDG_MAIN_EXPECTED_STEPS=1"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-05"
      "-DHDG_MAIN_NO_VTK=ON"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_default_backward_euler_rank1 PROPERTIES
      LABELS "hdg;main-backend;cuda;ordinary;conventional;backward-euler;default;rank1;fixed-step;process;no-vtk"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

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
  add_test(NAME dg_igr_hdg_main_backend_cuda_igr_vtk_tensor_rank1
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-igr-vtk-tensor-rank1-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=tensor"
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
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_igr_vtk_tensor_rank1 PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;igr;vtk;sigma;fields;process"
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
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rank1_n3 PROPERTIES
      DEPENDS dg_igr_hdg_runtime_cuda_generic_euler_session_tensor_rank1)
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rank2_n3 PROPERTIES
      DEPENDS dg_igr_hdg_runtime_cuda_generic_euler_session_tensor_rank2)

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
      LABELS "hdg;main-backend;cuda;ordinary;conventional;p1;rank${_ranks};mpi;process;generic-borrowed;cfl;no-vtk"
      PROCESSORS ${_ranks} RESOURCE_LOCK dg_igr_sm120 TIMEOUT 1200)
    add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rank${_ranks}_cfl
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rank${_ranks}-cfl-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_CUDA_ENGINE=tensor"
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
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_tensor_rank${_ranks}_cfl PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;conventional;p1;rank${_ranks};mpi;process;generic-borrowed;cfl;no-vtk"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
    add_test(NAME dg_igr_hdg_main_backend_cuda_igr_rank${_ranks}_cfl
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-igr-rank${_ranks}-cfl-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_REGULARIZATION=igr"
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
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_igr_rank${_ranks}_cfl PROPERTIES
        LABELS "hdg;main-backend;cuda;ordinary;igr;p1;rank${_ranks};mpi;process;generic-borrowed;cfl;no-vtk"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
    add_test(NAME dg_igr_hdg_main_backend_cuda_igr_tensor_rank${_ranks}_cfl
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-igr-tensor-rank${_ranks}-cfl-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
        "-DHDG_MAIN_CUDA_ENGINE=tensor"
        "-DHDG_MAIN_REGULARIZATION=igr"
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
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_igr_tensor_rank${_ranks}_cfl PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;igr;p1;rank${_ranks};mpi;process;generic-borrowed;cfl;no-vtk"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
  endforeach()

  # The established names above remain the rank-one/rank-two p1
  # representatives.  Higher-order CFL gates are rank-one and explicitly
  # encode engine, mathematical mode, and order in both name and labels.
  foreach(_order IN ITEMS 2 3)
    if(_order EQUAL 2)
      set(_expected_cfl_steps 4)
    else()
      set(_expected_cfl_steps 6)
    endif()
    add_hdg_main_cfl_process_test(
      dg_igr_hdg_main_backend_cuda_ordinary_conventional_p${_order}_rank1_cfl
      cuda-ordinary-conventional-p${_order}-rank1-cfl
      cuda ordinary none ${_order} 1 ${_expected_cfl_steps})
    add_hdg_main_cfl_process_test(
      dg_igr_hdg_main_backend_cuda_tensor_conventional_p${_order}_rank1_cfl
      cuda-tensor-conventional-p${_order}-rank1-cfl
      cuda tensor none ${_order} 1 ${_expected_cfl_steps})
    add_hdg_main_cfl_process_test(
      dg_igr_hdg_main_backend_cuda_ordinary_igr_p${_order}_rank1_cfl
      cuda-ordinary-igr-p${_order}-rank1-cfl
      cuda ordinary igr ${_order} 1 ${_expected_cfl_steps})
    add_hdg_main_cfl_process_test(
      dg_igr_hdg_main_backend_cuda_tensor_igr_p${_order}_rank1_cfl
      cuda-tensor-igr-p${_order}-rank1-cfl
      cuda tensor igr ${_order} 1 ${_expected_cfl_steps})
  endforeach()

  # Winner-specific application qualification is deliberately bounded: p2
  # rank-one CFL gates cover both CUDA engines, while the two fixed-step VTK
  # gates split serial and parallel refinement across ordinary and tensor CUDA.
  add_hdg_main_cfl_process_test(
    dg_igr_hdg_main_backend_cuda_ordinary_partitioned_backward_euler_igr_p2_rank1_cfl
    cuda-ordinary-partitioned-backward-euler-igr-p2-rank1-cfl
    cuda ordinary igr 2 1 4 partitioned-backward-euler)
  add_hdg_main_cfl_process_test(
    dg_igr_hdg_main_backend_cuda_tensor_partitioned_backward_euler_igr_p2_rank1_cfl
    cuda-tensor-partitioned-backward-euler-igr-p2-rank1-cfl
    cuda tensor igr 2 1 4 partitioned-backward-euler)

  add_hdg_main_cuda_refinement_process_test(
    dg_igr_hdg_main_backend_cuda_ordinary_partitioned_backward_euler_igr_p2_rank1_serial_refine
    cuda-ordinary-partitioned-backward-euler-igr-p2-rank1-serial-refine
    ordinary igr 2 serial partitioned-backward-euler)
  add_hdg_main_cuda_refinement_process_test(
    dg_igr_hdg_main_backend_cuda_tensor_partitioned_backward_euler_igr_p2_rank2_parallel_refine
    cuda-tensor-partitioned-backward-euler-igr-p2-rank2-parallel-refine
    tensor igr 2 parallel partitioned-backward-euler)

  # RODAS3 remains an explicit opt-in.  Keep its production-entry matrix
  # compact because the direct runtime qualification below this application
  # seam owns the p1--p3, boundary/source, and engine-alignment coverage.
  add_test(NAME dg_igr_hdg_main_backend_cuda_ordinary_rodas3_conventional_p1_rank1_fixed
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-ordinary-rodas3-conventional-p1-rank1-fixed-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=ordinary"
      "-DHDG_MAIN_REGULARIZATION=none"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
      "-DHDG_MAIN_DT=0.000001"
      "-DHDG_MAIN_T_FINAL=0.000001"
      "-DHDG_MAIN_EXPECTED_STEPS=1"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-06"
      "-DHDG_MAIN_NO_VTK=ON"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_ordinary_rodas3_conventional_p1_rank1_fixed
    PROPERTIES
      LABELS "hdg;main-backend;cuda;ordinary;conventional;rodas3;p1;rank1;fixed-step;process;no-vtk"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

  add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_p2_rank1_fixed
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rodas3-igr-p2-rank1-fixed-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=tensor"
      "-DHDG_MAIN_REGULARIZATION=igr"
      "-DHDG_MAIN_ORDER=2"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
      "-DHDG_MAIN_C_ALPHA=0.5"
      "-DHDG_MAIN_EXPECT_C_ALPHA=0.5"
      "-DHDG_MAIN_POSITIVITY_LIMITER=zhang-shu"
      "-DHDG_MAIN_ZHANG_SHU_POINTS=quadrature"
      "-DHDG_MAIN_EXPECT_POSITIVITY_LIMITER=zhang-shu"
      "-DHDG_MAIN_EXPECT_ZHANG_SHU_POINTS=quadrature"
      "-DHDG_MAIN_VTK_ZHANG_SHU_DIAGNOSTICS=ON"
      "-DHDG_MAIN_EXPECT_ZHANG_SHU_FIELDS=ON"
      "-DHDG_MAIN_DT=0.000001"
      "-DHDG_MAIN_T_FINAL=0.000001"
      "-DHDG_MAIN_DT_SAVE=0.000001"
      "-DHDG_MAIN_EXPECTED_STEPS=1"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-06"
      "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001"
      "-DHDG_MAIN_EXPECTED_TIMES=0,1e-06"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_p2_rank1_fixed
    PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;igr;rodas3;p2;rank1;fixed-step;vtk;zhang-shu;limiter-diagnostics;process"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

  add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_p1_rank2_fixed
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rodas3-igr-p1-rank2-fixed-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=tensor"
      "-DHDG_MAIN_REGULARIZATION=igr"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=2"
      "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
      "-DHDG_MAIN_DT=0.000001"
      "-DHDG_MAIN_T_FINAL=0.000001"
      "-DHDG_MAIN_EXPECTED_STEPS=1"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=1e-06"
      "-DHDG_MAIN_NO_VTK=ON"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_p1_rank2_fixed
    PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;igr;rodas3;p1;rank2;mpi;host-staged;fixed-step;process;no-vtk"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

  # Keep the adaptive rank-two production entry independent of the fixed-step
  # host-staged gate above.  The strict embedded tolerance makes the initial
  # horizon-sized proposal reject before a smaller proposal is accepted, while
  # --no-vtk_save keeps this MPI lifecycle check bounded.
  add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_p1_rank2_adaptive_rejection
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rodas3-igr-p1-rank2-adaptive-rejection-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=tensor"
      "-DHDG_MAIN_REGULARIZATION=igr"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=2"
      "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
      "-DHDG_MAIN_DT=-0.001"
      "-DHDG_MAIN_DT_MIN=1e-8"
      "-DHDG_MAIN_TIME_ATOL=1e-10"
      "-DHDG_MAIN_TIME_RTOL=1e-10"
      "-DHDG_MAIN_T_FINAL=0.001"
      "-DHDG_MAIN_EXPECTED_STEPS=[1-9][0-9]*"
      "-DHDG_MAIN_EXPECTED_REJECTIONS=[1-9][0-9]*"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.001"
      "-DHDG_MAIN_NO_VTK=ON"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DHDG_MAIN_MPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
      "-DHDG_MAIN_MPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_p1_rank2_adaptive_rejection
    PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;igr;rodas3;p1;rank2;mpi;host-staged;adaptive;rejection;rollback;process;no-vtk"
      PROCESSORS 2
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

  add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_conventional_p1_rank1_adaptive_rejection
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rodas3-conventional-p1-rank1-adaptive-rejection-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=tensor"
      "-DHDG_MAIN_REGULARIZATION=none"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
      "-DHDG_MAIN_DT=-0.001"
      "-DHDG_MAIN_DT_MIN=1e-8"
      "-DHDG_MAIN_TIME_ATOL=1e-10"
      "-DHDG_MAIN_TIME_RTOL=1e-10"
      "-DHDG_MAIN_T_FINAL=0.001"
      "-DHDG_MAIN_DT_SAVE=0.001"
      "-DHDG_MAIN_EXPECTED_STEPS=[1-9][0-9]*"
      "-DHDG_MAIN_EXPECTED_REJECTIONS=[1-9][0-9]*"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.001"
      "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001"
      "-DHDG_MAIN_EXPECTED_TIMES=0,0.001"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rodas3_conventional_p1_rank1_adaptive_rejection
    PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;conventional;rodas3;p1;rank1;adaptive;rejection;rollback;output-landing;process"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

  # The nominal 1e-3 controller proposal is first clipped to the 9.9e-4
  # intermediate publication boundary.  A strict embedded-error gate forces
  # rejection/retry around that clipped plan; exact three-cycle publication
  # checks then prove that rollback did not advance or duplicate output.
  add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_conventional_p1_rank1_adaptive_intermediate_output_rejection
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-rodas3-conventional-p1-rank1-adaptive-intermediate-output-rejection-output"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      "-DHDG_MAIN_CUDA_ENGINE=tensor"
      "-DHDG_MAIN_REGULARIZATION=none"
      "-DHDG_MAIN_ORDER=1"
      "-DHDG_MAIN_N=3"
      "-DHDG_MAIN_RANKS=1"
      "-DHDG_MAIN_TIME_INTEGRATOR=rodas3"
      "-DHDG_MAIN_DT=-0.001"
      "-DHDG_MAIN_DT_MIN=1e-8"
      "-DHDG_MAIN_TIME_ATOL=1e-10"
      "-DHDG_MAIN_TIME_RTOL=1e-10"
      "-DHDG_MAIN_T_FINAL=0.001"
      "-DHDG_MAIN_DT_SAVE=0.00099"
      "-DHDG_MAIN_EXPECTED_STEPS=[1-9][0-9]*"
      "-DHDG_MAIN_EXPECTED_REJECTIONS=[1-9][0-9]*"
      "-DHDG_MAIN_EXPECTED_FINAL_TIME=0\\.001"
      "-DHDG_MAIN_EXPECTED_CYCLES=000000,000001,000002"
      "-DHDG_MAIN_EXPECTED_TIMES=0,0.00099,0.001"
      "-DHDG_MAIN_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DHDG_MAIN_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_generic_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_tensor_rodas3_conventional_p1_rank1_adaptive_intermediate_output_rejection
    PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;conventional;rodas3;p1;rank1;adaptive;rejection;rollback;output-clipping;intermediate-output;process"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

  # Five fixed steps are the common admissible short cylinder interval used by
  # the architecture bake-off.  This is the production application and mesh,
  # with output disabled so the gate remains a bounded evolution check.
  if(TARGET supersonic_cylinder_w_igr)
    add_test(
      NAME dg_igr_hdg_main_backend_cuda_tensor_backward_euler_cylinder_p2_rank1
      COMMAND $<TARGET_FILE:supersonic_cylinder_w_igr>
        --discretization hdg
        --device cuda
        --hdg-cuda-engine tensor
        --regularization igr
        --time-integrator backward-euler
        --mesh ${PROJECT_SOURCE_DIR}/examples/supersonic_example_w_igr/supersonic_cylinder.msh
        --order 2
        --serial-refine 0
        --parallel-refine 0
        --time-step 0.00001
        --t-final 0.00005
        --no-vtk_save
        --no-glvis)
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_tensor_backward_euler_cylinder_p2_rank1
      PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;igr;backward-euler;monolithic;p2;rank1;physical;cylinder;fixed-step;no-vtk;application"
        PASS_REGULAR_EXPRESSION
          "\\[hdg-main\\] status=Success step=5.*time=5e-05.*cuda_engine=tensor.*time_treatment=backward-euler.*fallback_count=0.*regularization=igr.*c_alpha=1.*alpha_effective=0.0104992.*alpha_length_policy=max.*h_eff=0.102466"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)

    add_test(
      NAME dg_igr_hdg_main_backend_cuda_tensor_partitioned_backward_euler_cylinder_p2_rank1
      COMMAND $<TARGET_FILE:supersonic_cylinder_w_igr>
        --discretization hdg
        --device cuda
        --hdg-cuda-engine tensor
        --regularization igr
        --time-integrator partitioned-backward-euler
        --mesh ${PROJECT_SOURCE_DIR}/examples/supersonic_example_w_igr/supersonic_cylinder.msh
        --order 2
        --serial-refine 0
        --parallel-refine 0
        --time-step 0.00001
        --t-final 0.00005
        --no-vtk_save
        --no-glvis)
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_tensor_partitioned_backward_euler_cylinder_p2_rank1
      PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;igr;partitioned-backward-euler;p2;rank1;physical;cylinder;fixed-step;no-vtk;application"
        PASS_REGULAR_EXPRESSION
          "\\[hdg-main\\] status=Success step=5.*time=5e-05.*cuda_engine=tensor.*time_treatment=partitioned-backward-euler.*fallback_count=0.*regularization=igr.*c_alpha=1.*alpha_effective=0.0104992.*alpha_length_policy=max.*h_eff=0.102466"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)

    add_test(
      NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_cylinder_p2_rank1
      COMMAND $<TARGET_FILE:supersonic_cylinder_w_igr>
        --discretization hdg
        --device cuda
        --hdg-cuda-engine tensor
        --regularization igr
        --time-integrator rodas3
        --mesh ${PROJECT_SOURCE_DIR}/examples/supersonic_example_w_igr/supersonic_cylinder.msh
        --order 2
        --serial-refine 0
        --parallel-refine 0
        --time-step 0.00001
        --t-final 0.00005
        --no-vtk_save
        --no-glvis)
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_tensor_rodas3_cylinder_p2_rank1
      PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;igr;rodas3;p2;rank1;physical;cylinder;fixed-step;no-vtk;application"
        PASS_REGULAR_EXPRESSION
          "\\[hdg-main\\] status=Success step=5.*time=5e-05.*cuda_engine=tensor.*time_treatment=rodas3.*fallback_count=0.*regularization=igr.*c_alpha=1.*alpha_effective=0.0104992.*alpha_length_policy=max.*h_eff=0.102466"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
  endif()

  if(TARGET sedov_taylor_blast)
    add_test(
      NAME dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_sedov_p1_rank1_fixed
      COMMAND $<TARGET_FILE:sedov_taylor_blast>
        --discretization hdg
        --device cuda
        --hdg-cuda-engine tensor
        --regularization igr
        --time-integrator rodas3
        --positivity-limiter none
        --hdg-igr-alpha 1
        --mesh ${PROJECT_SOURCE_DIR}/examples/sedov_taylor_blast/sedov_taylor.msh
        --order 1
        --serial-refine 0
        --parallel-refine 0
        --time-step 0.000001
        --t-final 0.000001
        --no-vtk_save
        --no-glvis)
    set_tests_properties(
      dg_igr_hdg_main_backend_cuda_tensor_rodas3_igr_sedov_p1_rank1_fixed
      PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;igr;rodas3;sedov;p1;rank1;slip;fixed-step;limiter-off;no-vtk;application"
        PASS_REGULAR_EXPRESSION
          "\\[hdg-main\\] status=Success step=1.*time=1e-06.*cuda_engine=tensor.*time_treatment=rodas3.*fallback_count=0.*regularization=igr.*c_alpha=1"
        PROCESSORS 1
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
  endif()

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
      LABELS "hdg;main-backend;cuda;ordinary;conventional;p2;rank${_ranks};${_refine_mode}-refinement;mpi;process;generic-borrowed;fixed-step;vtk;refinement"
      PROCESSORS ${_ranks}
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)
    add_test(NAME dg_igr_hdg_main_backend_cuda_tensor_p2_${_refine_mode}_refine_small_dt
      COMMAND "${CMAKE_COMMAND}"
        "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
        "-DHDG_MAIN_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/cuda-tensor-p2-${_refine_mode}-refine-small-dt-output"
        "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh"
        "-DHDG_MAIN_CUDA_ENGINE=tensor"
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
      dg_igr_hdg_main_backend_cuda_tensor_p2_${_refine_mode}_refine_small_dt
      PROPERTIES
        LABELS "hdg;main-backend;cuda;tensor;conventional;p2;rank${_ranks};${_refine_mode}-refinement;mpi;process;generic-borrowed;fixed-step;vtk;refinement"
        PROCESSORS ${_ranks}
        RESOURCE_LOCK dg_igr_sm120
        TIMEOUT 1200)
  endforeach()

  # Fill the remaining rank-one serial p1-p3 cross-product.  The conventional
  # p2 ordinary/tensor cells retain their established names above.
  foreach(_order IN ITEMS 1 2 3)
    foreach(_cuda_engine IN ITEMS ordinary tensor)
      foreach(_regularization IN ITEMS none igr)
        if(_order EQUAL 2 AND _regularization STREQUAL "none")
          continue()
        endif()
        if(_regularization STREQUAL "igr")
          set(_mode_tag igr)
        else()
          set(_mode_tag conventional)
        endif()
        add_hdg_main_cuda_refinement_process_test(
          dg_igr_hdg_main_backend_cuda_${_cuda_engine}_${_mode_tag}_p${_order}_rank1_serial_refine_small_dt
          cuda-${_cuda_engine}-${_mode_tag}-p${_order}-rank1-serial-refine-small-dt
          ${_cuda_engine} ${_regularization} ${_order} serial)
      endforeach()
    endforeach()
  endforeach()

  # Rank two retains a single p2 parallel-refinement representative per
  # engine and mathematical mode.  Conventional names are preserved above.
  foreach(_cuda_engine IN ITEMS ordinary tensor)
    add_hdg_main_cuda_refinement_process_test(
      dg_igr_hdg_main_backend_cuda_${_cuda_engine}_igr_p2_rank2_parallel_refine_small_dt
      cuda-${_cuda_engine}-igr-p2-rank2-parallel-refine-small-dt
      ${_cuda_engine} igr 2 parallel)
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

  add_test(NAME dg_igr_hdg_main_backend_cuda_rejects_unsupported_state_routes
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dgsem_smooth_advection_periodic>"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n3.mesh"
      "-DHDG_MAIN_DEVICE=cuda"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_unsupported_state_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_rejects_unsupported_state_routes PROPERTIES
      LABELS "hdg;main-backend;cuda;ordinary;tensor;negative;restart;state-file;process"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 300)

  add_executable(dg_igr_hdg_main_backend_cuda_checkpoint_split_test
    test_cuda_checkpoint_split_driver.cu)
  target_include_directories(
    dg_igr_hdg_main_backend_cuda_checkpoint_split_test PRIVATE
      "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(
    dg_igr_hdg_main_backend_cuda_checkpoint_split_test PRIVATE
      DG_IGR_HDG_REPOSITORY_ROOT="${PROJECT_SOURCE_DIR}")
  target_link_libraries(
    dg_igr_hdg_main_backend_cuda_checkpoint_split_test PRIVATE
      dg_igr_hdg_main_backend_cuda
      dg_igr_hdg_runtime_app_contract
      igr_mfem
      mfem
      MPI::MPI_CXX)
  target_compile_options(
    dg_igr_hdg_main_backend_cuda_checkpoint_split_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_target_properties(
    dg_igr_hdg_main_backend_cuda_checkpoint_split_test PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_ARCHITECTURES "${DG_IGR_HDG_NATIVE_CUDA_ARCH}"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  add_test(NAME dg_igr_hdg_main_backend_cuda_checkpoint_process
    COMMAND "${CMAKE_COMMAND}"
      "-DHDG_MAIN_EXECUTABLE=$<TARGET_FILE:dg_igr_hdg_main_backend_cuda_checkpoint_split_test>"
      "-DHDG_MAIN_MESH=${PROJECT_SOURCE_DIR}/examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh"
      "-DHDG_MAIN_WORK=${CMAKE_CURRENT_BINARY_DIR}/cuda-checkpoint-process"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_checkpoint_process.cmake")
  set_tests_properties(
    dg_igr_hdg_main_backend_cuda_checkpoint_process PROPERTIES
      LABELS "hdg;main-backend;cuda;tensor;rodas3;igr;p2;dynamic-checkpoint;restart;corruption;process"
      PROCESSORS 1
      RESOURCE_LOCK dg_igr_sm120
      TIMEOUT 1200)

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
