# IGRlab correctness tests. Historical dg_igr target/CTest names remain
# transitional interfaces for existing scripts and retained evidence.

set(_igrlab_serial_mpi_args "")
if(IGRLAB_MPI_TEST_LAUNCHER)
  set(_igrlab_serial_mpi_args --mpi-single-rank --mpi-ranks 1
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}")
endif()

add_test(NAME dg_igr_cuda_build_policy
  COMMAND /usr/bin/env bash
          ${CMAKE_SOURCE_DIR}/tests/scripts/check_cuda_build_policy.sh
          ${CMAKE_SOURCE_DIR}
)
set_tests_properties(dg_igr_cuda_build_policy PROPERTIES
  LABELS "build;cuda;contract"
  TIMEOUT 30
)

foreach(build_check pace_submission pace_manifest pace_discovery delta_site dependency_bindings
                    prebuilt_validation mfem_cache app_iteration)
  add_test(NAME igrlab_${build_check}
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/test_${build_check}.py)
  set_tests_properties(igrlab_${build_check} PROPERTIES
    LABELS "build;contract;python;no-gpu"
    TIMEOUT 30)
endforeach()

foreach(study_check gpu_study study_build study_workloads study_profiles study_retry)
  add_test(NAME igrlab_${study_check}_python
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/scripts/perf/test_${study_check}.py)
  set_tests_properties(igrlab_${study_check}_python PROPERTIES
    LABELS "profiling;study;contract;python;no-gpu" TIMEOUT 120)
endforeach()

add_test(NAME igrlab_mpi_test_launcher
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/tests/scripts/test_mpi_test_launcher.py
          --cmake ${CMAKE_COMMAND})
set_tests_properties(igrlab_mpi_test_launcher PROPERTIES
  LABELS "build;contract;python;no-gpu" TIMEOUT 30)

add_test(NAME dg_igr_hdg_tensor_campaign_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_hdg_tensor_campaign.py
)
set_tests_properties(dg_igr_hdg_tensor_campaign_python PROPERTIES
  LABELS "hdg;tensor;campaign;python;no-gpu"
  TIMEOUT 180
)

add_test(NAME dg_igr_profile_harness_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_profile_harness.py
)
set_tests_properties(dg_igr_profile_harness_python PROPERTIES
  LABELS "profiling;provenance;python;no-gpu"
  TIMEOUT 180
)

add_test(NAME dg_igr_gpu_memory_control_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_gpu_memory_control.py
)
set_tests_properties(dg_igr_gpu_memory_control_python PROPERTIES
  LABELS "hdg;tensor;memory;campaign;python;no-gpu"
  TIMEOUT 180
)

add_test(NAME dg_igr_adaptive_trace_memory_matrix_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_adaptive_trace_memory_matrix.py
)
set_tests_properties(dg_igr_adaptive_trace_memory_matrix_python PROPERTIES
  LABELS "hdg;tensor;memory;architecture-selection;python;no-gpu"
  TIMEOUT 180
)

add_test(NAME dg_igr_shock_architecture_matrix_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_shock_architecture_matrix.py
)
set_tests_properties(dg_igr_shock_architecture_matrix_python PROPERTIES
  LABELS "shock;architecture-selection;python;no-gpu"
  TIMEOUT 60
)

add_test(NAME dg_igr_shock_qoi_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_shock_qoi.py
)
set_tests_properties(dg_igr_shock_qoi_python PROPERTIES
  LABELS "shock;qoi;architecture-selection;python;no-gpu"
  TIMEOUT 60
)

add_test(NAME dg_igr_hdg_tensor_memory_campaign_python
  COMMAND /usr/bin/env python3
          ${CMAKE_SOURCE_DIR}/scripts/perf/test_hdg_tensor_memory_campaign.py
)
set_tests_properties(dg_igr_hdg_tensor_memory_campaign_python PROPERTIES
  LABELS "hdg;tensor;memory;campaign;python;no-gpu"
  TIMEOUT 180
)

# Compiler-free checks of the canonical dependency admission and name aliases.
add_test(NAME igrlab_spack_external_artifacts
  COMMAND /usr/bin/env bash
          ${CMAKE_SOURCE_DIR}/tests/scripts/check_spack_external_artifacts.sh
          ${CMAKE_SOURCE_DIR})
set_tests_properties(igrlab_spack_external_artifacts PROPERTIES
  LABELS "build;spack;contract;no-gpu" TIMEOUT 60)

add_test(NAME igrlab_cmake_compatibility
  COMMAND ${CMAKE_COMMAND}
          "-DIGRLAB_COMPATIBILITY_HELPER=${CMAKE_SOURCE_DIR}/cmake/IGRlabCompatibility.cmake"
          "-DIGRLAB_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/igrlab-cmake-compatibility"
          -P ${CMAKE_SOURCE_DIR}/tests/scripts/check_igrlab_cmake_aliases.cmake)
set_tests_properties(igrlab_cmake_compatibility PROPERTIES
  LABELS "build;compatibility;contract;no-gpu" TIMEOUT 30)

function(dg_igr_add_observability_test target_name compile_enabled caliper_enabled)
  add_executable(${target_name} test_observability.cpp)
  set_target_properties(${target_name} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
  )
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(${target_name} PRIVATE
    DG_IGR_ENABLE_OBSERVABILITY=${compile_enabled}
    DG_IGR_OBSERVABILITY_USE_NVTX=0
    DG_IGR_OBSERVABILITY_USE_CALIPER=${caliper_enabled}
  )
  if(caliper_enabled)
    target_link_libraries(${target_name} PRIVATE caliper)
  endif()
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>
  )
  add_test(NAME ${target_name} COMMAND ${target_name})
  set_tests_properties(${target_name} PROPERTIES
    LABELS "observability;contract"
    TIMEOUT 30
  )
endfunction()

dg_igr_add_observability_test(dg_igr_observability_noop_test 0 0)
dg_igr_add_observability_test(dg_igr_observability_recording_test 1 0)
if(DG_IGR_ENABLE_OBSERVABILITY AND DG_IGR_OBSERVABILITY_USE_CALIPER)
  dg_igr_add_observability_test(dg_igr_observability_caliper_test 1 1)
endif()

add_executable(dg_igr_frozen_scalar_igr_algebra_test
  test_frozen_scalar_igr_algebra.cpp
)
set_target_properties(dg_igr_frozen_scalar_igr_algebra_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(dg_igr_frozen_scalar_igr_algebra_test PRIVATE mfem)
target_include_directories(dg_igr_frozen_scalar_igr_algebra_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME dg_igr_frozen_scalar_igr_algebra_tests
  COMMAND dg_igr_frozen_scalar_igr_algebra_test
)
set_tests_properties(dg_igr_frozen_scalar_igr_algebra_tests PROPERTIES
  LABELS "igr;frozen-scalar;algebra"
  TIMEOUT 30
)

add_executable(dg_igr_unit_tests
  unit_tests_main.cpp
  test_euler.cpp
  test_fluxes.cpp
  test_limiters.cpp
  test_dg_operator.cpp
  test_dgsem_basis.cpp
  test_dgsem_flux.cpp
  test_dgsem_operator.cpp
  test_dgsem_advection.cpp
  test_dgsem_igr.cpp
  test_bc.cpp
  test_igr.cpp
  test_time_defaults.cpp
  test_diagnostics.cpp
)

add_executable(dg_igr_exact_face_test test_igr_exact_face.cpp)
target_link_libraries(dg_igr_exact_face_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_exact_face_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
set(_igrlab_exact_face_command dg_igr_exact_face_test)
if(MPIEXEC_EXECUTABLE)
  if(NOT MPIEXEC_NUMPROC_FLAG)
    message(FATAL_ERROR "Exact face MPI test launcher requires MPIEXEC_NUMPROC_FLAG.")
  endif()
  set(_igrlab_exact_face_command
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_exact_face_test> ${MPIEXEC_POSTFLAGS})
endif()
add_test(NAME dg_igr_exact_face_cpu COMMAND ${_igrlab_exact_face_command} cpu)
set_tests_properties(dg_igr_exact_face_cpu PROPERTIES
  LABELS "igr;exact-face;parity;cpu" TIMEOUT 120)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  set_source_files_properties(test_igr_exact_face.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_options(dg_igr_exact_face_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  add_test(NAME dg_igr_exact_face_cuda COMMAND ${_igrlab_exact_face_command} cuda)
  set_tests_properties(dg_igr_exact_face_cuda PROPERTIES
    LABELS "igr;exact-face;parity;cuda" TIMEOUT 120 RUN_SERIAL TRUE)
endif()

add_executable(dg_igr_dgsem_maximum_test test_dgsem_maximum.cpp)
target_link_libraries(dg_igr_dgsem_maximum_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_dgsem_maximum_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
set(_igrlab_dgsem_maximum_command dg_igr_dgsem_maximum_test)
if(MPIEXEC_EXECUTABLE)
  if(NOT MPIEXEC_NUMPROC_FLAG)
    message(FATAL_ERROR "Maximum parity MPI launcher requires MPIEXEC_NUMPROC_FLAG.")
  endif()
  set(_igrlab_dgsem_maximum_command
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_dgsem_maximum_test> ${MPIEXEC_POSTFLAGS})
endif()
add_test(NAME dg_igr_dgsem_maximum_cpu
  COMMAND ${_igrlab_dgsem_maximum_command} cpu)
set_tests_properties(dg_igr_dgsem_maximum_cpu PROPERTIES
  LABELS "dgsem;maximum;parity;cpu" TIMEOUT 120 RUN_SERIAL TRUE)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  set_source_files_properties(test_dgsem_maximum.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_options(dg_igr_dgsem_maximum_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  add_test(NAME dg_igr_dgsem_maximum_cuda
    COMMAND ${_igrlab_dgsem_maximum_command} cuda)
  set_tests_properties(dg_igr_dgsem_maximum_cuda PROPERTIES
    LABELS "dgsem;maximum;parity;cuda" TIMEOUT 120 RUN_SERIAL TRUE)
endif()

add_executable(dg_igr_dg_device_faces_test test_dg_device_faces.cpp)
target_link_libraries(dg_igr_dg_device_faces_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_dg_device_faces_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
set_source_files_properties(test_dg_device_faces.cpp PROPERTIES LANGUAGE CXX)
set(_igrlab_dg_device_faces_command dg_igr_dg_device_faces_test)
if(MPIEXEC_EXECUTABLE)
  set(_igrlab_dg_device_faces_command
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_dg_device_faces_test> ${MPIEXEC_POSTFLAGS})
endif()
add_test(NAME dg_igr_dg_device_faces_cpu
  COMMAND ${_igrlab_dg_device_faces_command} cpu)
set_tests_properties(dg_igr_dg_device_faces_cpu PROPERTIES
  LABELS "dg;face;parity;cpu" TIMEOUT 120)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  add_test(NAME dg_igr_dg_device_faces_cuda
    COMMAND ${_igrlab_dg_device_faces_command} cuda)
  set_tests_properties(dg_igr_dg_device_faces_cuda PROPERTIES
    LABELS "dg;face;parity;cuda" TIMEOUT 180 RUN_SERIAL TRUE)
endif()

add_executable(dg_igr_face_density_host_cache_test
  test_igr_face_density_host_cache.cpp)
target_link_libraries(dg_igr_face_density_host_cache_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_face_density_host_cache_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
# Keep CXX in CUDA builds: __CUDACC__ would change the original host oracle.
set_source_files_properties(test_igr_face_density_host_cache.cpp PROPERTIES
  LANGUAGE CXX)
set(_igrlab_host_face_cache_command dg_igr_face_density_host_cache_test)
if(MPIEXEC_EXECUTABLE)
  if(NOT MPIEXEC_NUMPROC_FLAG)
    message(FATAL_ERROR "Host face-density MPI launcher requires MPIEXEC_NUMPROC_FLAG.")
  endif()
  set(_igrlab_host_face_cache_command
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_face_density_host_cache_test> ${MPIEXEC_POSTFLAGS})
  add_test(NAME dg_igr_face_density_host_cache_cpu_mpi4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_face_density_host_cache_test> ${MPIEXEC_POSTFLAGS} cpu)
  set_tests_properties(dg_igr_face_density_host_cache_cpu_mpi4 PROPERTIES
    LABELS "igr;exact-face;cache;parity;cpu;mpi" TIMEOUT 180
    RUN_SERIAL TRUE PROCESSORS 4)
endif()
add_test(NAME dg_igr_face_density_host_cache_cpu
  COMMAND ${_igrlab_host_face_cache_command} cpu)
set_tests_properties(dg_igr_face_density_host_cache_cpu PROPERTIES
  LABELS "igr;exact-face;cache;parity;cpu" TIMEOUT 120)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  add_test(NAME dg_igr_face_density_host_cache_cuda
    COMMAND ${_igrlab_host_face_cache_command} cuda)
  set_tests_properties(dg_igr_face_density_host_cache_cuda PROPERTIES
    LABELS "igr;exact-face;cache;parity;cuda" TIMEOUT 180 RUN_SERIAL TRUE)
endif()

add_executable(dg_igr_euler_admissibility_cache_test
  test_euler_admissibility_cache.cpp)
# Keep the original MFEM/EOS oracle and the actual helper caller in CXX.
set_source_files_properties(test_euler_admissibility_cache.cpp PROPERTIES
  LANGUAGE CXX)
target_link_libraries(dg_igr_euler_admissibility_cache_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_euler_admissibility_cache_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
set(_igrlab_admissibility_command dg_igr_euler_admissibility_cache_test)
if(MPIEXEC_EXECUTABLE)
  if(NOT MPIEXEC_NUMPROC_FLAG)
    message(FATAL_ERROR "Admissibility MPI test launcher requires MPIEXEC_NUMPROC_FLAG.")
  endif()
  set(_igrlab_admissibility_command
    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:dg_igr_euler_admissibility_cache_test> ${MPIEXEC_POSTFLAGS})
  add_test(NAME dg_igr_euler_admissibility_cache_cpu_mpi4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:dg_igr_euler_admissibility_cache_test> ${MPIEXEC_POSTFLAGS} cpu)
  set_tests_properties(dg_igr_euler_admissibility_cache_cpu_mpi4 PROPERTIES
    LABELS "admissibility;cache;parity;cpu;mpi;subcommunicator"
    TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 4)
endif()
add_test(NAME dg_igr_euler_admissibility_cache_cpu
  COMMAND ${_igrlab_admissibility_command} cpu)
set_tests_properties(dg_igr_euler_admissibility_cache_cpu PROPERTIES
  LABELS "admissibility;cache;parity;cpu" TIMEOUT 120)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  target_sources(dg_igr_euler_admissibility_cache_test PRIVATE
    test_euler_admissibility_cache_cuda.cu)
  target_compile_definitions(dg_igr_euler_admissibility_cache_test PRIVATE
    IGRLAB_ADMISSIBILITY_CUDA_PROBE=1)
  target_compile_options(dg_igr_euler_admissibility_cache_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  add_test(NAME dg_igr_euler_admissibility_cache_cuda
    COMMAND ${_igrlab_admissibility_command} cuda)
  set_tests_properties(dg_igr_euler_admissibility_cache_cuda PROPERTIES
    LABELS "admissibility;cache;parity;cuda" TIMEOUT 180 RUN_SERIAL TRUE)
endif()

add_executable(dg_igr_nodal_source_test test_igr_nodal_source.cpp)
target_link_libraries(dg_igr_nodal_source_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_nodal_source_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME dg_igr_nodal_source_cpu COMMAND dg_igr_nodal_source_test cpu)
set_tests_properties(dg_igr_nodal_source_cpu PROPERTIES LABELS "igr;source;research;cpu" TIMEOUT 90)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  set_source_files_properties(test_igr_nodal_source.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_options(dg_igr_nodal_source_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  add_test(NAME dg_igr_nodal_source_cuda COMMAND dg_igr_nodal_source_test cuda)
  set_tests_properties(dg_igr_nodal_source_cuda PROPERTIES
    LABELS "igr;source;research;cuda" TIMEOUT 90 RUN_SERIAL TRUE)
endif()

# Bounded research controls for volume quadrature and physical slip-wall lifts.
add_executable(dg_igr_overintegrated_volume_test test_dgsem_overintegrated_volume.cpp)
add_executable(dg_igr_wall_source_probe research/igr_wall_source_probe.cpp)
if(IGRLAB_BUILD_RESEARCH_TESTS)
  add_executable(dg_igr_wall_compression research/igr_wall_compression.cpp)
  target_link_libraries(dg_igr_wall_compression PRIVATE igr_mfem mfem)
  target_include_directories(dg_igr_wall_compression PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()
foreach(target IN ITEMS dg_igr_overintegrated_volume_test dg_igr_wall_source_probe)
  target_link_libraries(${target} PRIVATE igr_mfem mfem)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endforeach()
foreach(pair IN ITEMS overintegrated_volume wall_source)
  if(pair STREQUAL "overintegrated_volume")
    set(probe dg_igr_overintegrated_volume_test)
  else()
    set(probe dg_igr_wall_source_probe)
  endif()
  add_test(NAME dg_igr_${pair}_cpu COMMAND ${probe} cpu)
  set_tests_properties(dg_igr_${pair}_cpu PROPERTIES LABELS "igr;research;cpu" TIMEOUT 120)
  if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
    add_test(NAME dg_igr_${pair}_cuda COMMAND ${probe} cuda)
    set_tests_properties(dg_igr_${pair}_cuda PROPERTIES LABELS "igr;research;cuda" TIMEOUT 120 RUN_SERIAL TRUE)
  endif()
endforeach()
add_test(NAME dg_igr_wall_annulus_source_cpu COMMAND dg_igr_wall_source_probe cpu --annulus)
set_tests_properties(dg_igr_wall_annulus_source_cpu PROPERTIES LABELS "igr;research;cpu" TIMEOUT 120)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  add_test(NAME dg_igr_wall_annulus_source_cuda COMMAND dg_igr_wall_source_probe cuda --annulus)
  set_tests_properties(dg_igr_wall_annulus_source_cuda PROPERTIES LABELS "igr;research;cuda" TIMEOUT 120 RUN_SERIAL TRUE)
endif()
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  set_source_files_properties(test_dgsem_overintegrated_volume.cpp PROPERTIES LANGUAGE CUDA)
  target_compile_options(dg_igr_overintegrated_volume_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
endif()

add_executable(dg_igr_modal_damping_test test_modal_damping.cpp)
target_link_libraries(dg_igr_modal_damping_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_modal_damping_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME dg_igr_modal_damping_cpu COMMAND dg_igr_modal_damping_test cpu)
set_tests_properties(dg_igr_modal_damping_cpu PROPERTIES
  LABELS "modal-damping;conservation;timestepping;cpu"
  TIMEOUT 90
)
if(MFEM_USE_CUDA)
  add_test(NAME dg_igr_modal_damping_cuda COMMAND dg_igr_modal_damping_test cuda)
  set_tests_properties(dg_igr_modal_damping_cuda PROPERTIES
    LABELS "modal-damping;conservation;timestepping;cuda"
    TIMEOUT 90
    RUN_SERIAL TRUE
  )
endif()

add_executable(dg_igr_explicit_ssprk_test
  test_explicit_ssprk.cpp
)
target_link_libraries(dg_igr_explicit_ssprk_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_explicit_ssprk_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME dg_igr_explicit_ssprk_cpu
  COMMAND dg_igr_explicit_ssprk_test cpu
)
set_tests_properties(dg_igr_explicit_ssprk_cpu PROPERTIES
  LABELS "timestepping;ssprk;lifecycle;cpu"
  TIMEOUT 60
)
if(MFEM_USE_CUDA)
  add_test(NAME dg_igr_explicit_ssprk_cuda
    COMMAND dg_igr_explicit_ssprk_test cuda
  )
  set_tests_properties(dg_igr_explicit_ssprk_cuda PROPERTIES
    LABELS "timestepping;ssprk;lifecycle;cuda"
    TIMEOUT 60
    RUN_SERIAL TRUE
  )
endif()

target_link_libraries(dg_igr_unit_tests PRIVATE igr_mfem mfem)

target_include_directories(dg_igr_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Small evolution gates use the production DG/DGSEM Mult and SSPRK(3,3)
# implementations.  Longer convergence ladders remain separately selectable.
add_executable(igrlab_numerical_evolution_test test_numerical_evolution.cpp)
target_link_libraries(igrlab_numerical_evolution_test PRIVATE igr_mfem mfem)
target_include_directories(igrlab_numerical_evolution_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include)
target_compile_features(igrlab_numerical_evolution_test PRIVATE cxx_std_17)
target_compile_options(igrlab_numerical_evolution_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
add_test(NAME igrlab_numerical_conservation_cpu
  COMMAND igrlab_numerical_evolution_test --conservation cpu)
set_tests_properties(igrlab_numerical_conservation_cpu PROPERTIES
  LABELS "numerical-acceptance;numerical-smoke;conservation;dg;dgsem;production;cpu"
  PROCESSORS 1 TIMEOUT 180)
add_test(NAME igrlab_numerical_dg_convergence_cpu
  COMMAND igrlab_numerical_evolution_test --convergence cpu)
set_tests_properties(igrlab_numerical_dg_convergence_cpu PROPERTIES
  LABELS "numerical-acceptance;numerical-convergence;convergence;spatial;dg;production;cpu"
  PROCESSORS 1 TIMEOUT 300)
if(MFEM_USE_CUDA)
  # The source stays ordinary C++; final device-symbol resolution reaches the
  # production CUDA kernels in igr_mfem through the same link closure as units.
  set_target_properties(igrlab_numerical_evolution_test PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME igrlab_numerical_conservation_cuda
    COMMAND igrlab_numerical_evolution_test --conservation cuda)
  add_test(NAME igrlab_numerical_backend_parity_cuda
    COMMAND igrlab_numerical_evolution_test --backend-parity cuda)
  set_tests_properties(igrlab_numerical_conservation_cuda
                       igrlab_numerical_backend_parity_cuda PROPERTIES
    LABELS "numerical-acceptance;numerical-smoke;conservation;dg;dgsem;production;cuda"
    PROCESSORS 1 TIMEOUT 240 RUN_SERIAL TRUE)
  set_property(TEST igrlab_numerical_backend_parity_cuda APPEND PROPERTY
    LABELS "backend-parity;cpu")
endif()

if(MFEM_USE_MPI AND MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
  if(IGRLAB_BUILD_RESEARCH_TESTS)
    add_executable(dg_igr_screened_locality_probe igr_screened_locality_probe.cpp)
    target_link_libraries(dg_igr_screened_locality_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_screened_locality_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)

    add_executable(dg_igr_local_operator_probe research/igr_local_operator_probe.cpp)
    target_link_libraries(dg_igr_local_operator_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_local_operator_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_local_operator_cpu
      COMMAND dg_igr_local_operator_probe --nx 3 --ny 2 --order 3
        --output ${CMAKE_CURRENT_BINARY_DIR}/igr_local_operator_smoke.json)
    set_tests_properties(dg_igr_local_operator_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_screened_spacetime_probe research/screened_spacetime_probe.cpp)
    target_link_libraries(dg_igr_screened_spacetime_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_screened_spacetime_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_screened_spacetime_cpu
      COMMAND dg_igr_screened_spacetime_probe --self-test)
    set_tests_properties(dg_igr_screened_spacetime_cpu PROPERTIES
      LABELS "cpu-reference;igr;timestepping;research" TIMEOUT 120)

    add_executable(dg_igr_local_predictor_probe research/igr_local_predictor_probe.cpp)
    target_link_libraries(dg_igr_local_predictor_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_local_predictor_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_local_predictor_cpu COMMAND dg_igr_local_predictor_probe --self-test)
    set_tests_properties(dg_igr_local_predictor_cpu PROPERTIES
      LABELS "cpu-reference;igr;timestepping;research" TIMEOUT 120)

    add_executable(dg_igr_high_order_hybrid_probe research/igr_high_order_hybrid_probe.cpp)
    target_link_libraries(dg_igr_high_order_hybrid_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_high_order_hybrid_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_high_order_hybrid_cpu COMMAND dg_igr_high_order_hybrid_probe --self-test)
    set_tests_properties(dg_igr_high_order_hybrid_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_hp_causal_probe research/igr_hp_causal_probe.cpp)
    target_link_libraries(dg_igr_hp_causal_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_hp_causal_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_hp_causal_cpu COMMAND dg_igr_hp_causal_probe --self-test)
    set_tests_properties(dg_igr_hp_causal_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_hp_front_tracking_probe research/igr_hp_front_tracking_probe.cpp)
    target_link_libraries(dg_igr_hp_front_tracking_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_hp_front_tracking_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_hp_front_tracking_cpu COMMAND dg_igr_hp_front_tracking_probe --self-test)
    set_tests_properties(dg_igr_hp_front_tracking_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_hp_enriched_probe research/igr_hp_enriched_probe.cpp)
    target_link_libraries(dg_igr_hp_enriched_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_hp_enriched_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_hp_enriched_cpu COMMAND dg_igr_hp_enriched_probe --self-test)
    set_tests_properties(dg_igr_hp_enriched_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_hp_moving_time_probe research/igr_hp_moving_time_probe.cpp)
    target_link_libraries(dg_igr_hp_moving_time_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_hp_moving_time_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_hp_moving_time_cpu COMMAND dg_igr_hp_moving_time_probe --self-test)
    set_tests_properties(dg_igr_hp_moving_time_cpu PROPERTIES
      LABELS "cpu-reference;igr;timestepping;research" TIMEOUT 120)

    foreach(compatibility_probe IN ITEMS work evolution)
      add_executable(dg_igr_dg_compatibility_${compatibility_probe}_probe
        research/igr_dg_compatibility_${compatibility_probe}_probe.cpp)
      target_link_libraries(dg_igr_dg_compatibility_${compatibility_probe}_probe PRIVATE igr_mfem mfem)
      target_include_directories(dg_igr_dg_compatibility_${compatibility_probe}_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
      add_test(NAME dg_igr_dg_compatibility_${compatibility_probe}_cpu
        COMMAND dg_igr_dg_compatibility_${compatibility_probe}_probe --self-test)
      set_tests_properties(dg_igr_dg_compatibility_${compatibility_probe}_cpu PROPERTIES
        LABELS "cpu-reference;igr;research" TIMEOUT 120)
    endforeach()

    foreach(boundary_probe IN ITEMS boundary_evolution wall_work recovery_evolution source_tangent moment_memory primitive_moments kinetic_product primitive_wall slope_penalty radial_wall ordinary_slope curvature_penalty hp_patch front_tangent commuting_evolution gradient_evolution entropy_coupled signed_enrichment signed_predictor entropy_coupled_functional predictor_curvature predictor_curvature_picard)
      add_executable(dg_igr_dg_${boundary_probe}_probe
        research/igr_dg_${boundary_probe}_probe.cpp)
      target_link_libraries(dg_igr_dg_${boundary_probe}_probe PRIVATE igr_mfem mfem)
      target_include_directories(dg_igr_dg_${boundary_probe}_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
      add_test(NAME dg_igr_dg_${boundary_probe}_cpu
        COMMAND dg_igr_dg_${boundary_probe}_probe --self-test)
      set_tests_properties(dg_igr_dg_${boundary_probe}_cpu PROPERTIES
        LABELS "cpu-reference;igr;research" TIMEOUT 120)
    endforeach()

    add_executable(dg_igr_high_order_condensation_probe research/igr_high_order_condensation_probe.cpp)
    target_link_libraries(dg_igr_high_order_condensation_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_high_order_condensation_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_high_order_condensation_cpu COMMAND dg_igr_high_order_condensation_probe --self-test)
    set_tests_properties(dg_igr_high_order_condensation_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_high_order_composite_probe research/igr_high_order_composite_probe.cpp)
    target_link_libraries(dg_igr_high_order_composite_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_high_order_composite_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_high_order_composite_cpu COMMAND dg_igr_high_order_composite_probe --self-test)
    set_tests_properties(dg_igr_high_order_composite_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_subcell_evolution_probe research/igr_subcell_evolution_probe.cpp)
    target_link_libraries(dg_igr_subcell_evolution_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_subcell_evolution_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_subcell_evolution_cpu COMMAND dg_igr_subcell_evolution_probe --self-test)
    set_tests_properties(dg_igr_subcell_evolution_cpu PROPERTIES
      LABELS "cpu-reference;igr;research" TIMEOUT 120)

    add_executable(dg_igr_subcell_predictor_probe research/igr_subcell_predictor_probe.cpp)
    target_link_libraries(dg_igr_subcell_predictor_probe PRIVATE igr_mfem mfem)
    target_include_directories(dg_igr_subcell_predictor_probe PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME dg_igr_subcell_predictor_cpu COMMAND dg_igr_subcell_predictor_probe --self-test)
    set_tests_properties(dg_igr_subcell_predictor_cpu PROPERTIES
      LABELS "cpu-reference;igr;timestepping;research" TIMEOUT 120)

  endif()

  add_executable(dg_igr_research_rk_test test_research_rk.cpp)
  target_link_libraries(dg_igr_research_rk_test PRIVATE igr_mfem mfem)
  target_include_directories(dg_igr_research_rk_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
  add_test(NAME dg_igr_research_rk_cpu COMMAND dg_igr_research_rk_test)
  set_tests_properties(dg_igr_research_rk_cpu PROPERTIES
    LABELS "cpu-reference;timestepping;research" TIMEOUT 120)

  add_executable(dg_igr_igr_extrapolation_history_test
    test_igr_extrapolation_history.cpp)
  target_link_libraries(dg_igr_igr_extrapolation_history_test PRIVATE igr_mfem mfem)
  target_include_directories(dg_igr_igr_extrapolation_history_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
  add_test(NAME dg_igr_igr_extrapolation_history
    COMMAND dg_igr_igr_extrapolation_history_test)
  set_tests_properties(dg_igr_igr_extrapolation_history PROPERTIES
    LABELS "cpu-reference;igr;timestepping;extrapolated" TIMEOUT 180)

  add_executable(dg_igr_q_sigma_subcommunicator_test
    test_q_sigma_subcommunicator.cpp
  )
  target_link_libraries(dg_igr_q_sigma_subcommunicator_test PRIVATE
    igr_mfem
    mfem
  )
  target_include_directories(dg_igr_q_sigma_subcommunicator_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  add_test(NAME dg_igr_q_sigma_subcommunicator_cpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:dg_igr_q_sigma_subcommunicator_test>
  )
  set_tests_properties(dg_igr_q_sigma_subcommunicator_cpu PROPERTIES
    LABELS "cpu-reference;mpi;subcommunicator;igr;extrapolated;admission;rollback"
    TIMEOUT 180
    RUN_SERIAL TRUE
    PROCESSORS 4
  )

  add_executable(dg_igr_vtk_restart_subcommunicator_test
    test_vtk_restart_subcommunicator.cpp
  )
  target_link_libraries(dg_igr_vtk_restart_subcommunicator_test PRIVATE
    igr_mfem
    mfem
  )
  target_include_directories(dg_igr_vtk_restart_subcommunicator_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  add_test(NAME dg_igr_vtk_restart_subcommunicator_cpu
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            $<TARGET_FILE:dg_igr_vtk_restart_subcommunicator_test>
            ${CMAKE_BINARY_DIR}/tests/vtk-restart-subcommunicator-cpu
  )
  set_tests_properties(dg_igr_vtk_restart_subcommunicator_cpu PROPERTIES
    LABELS "mpi;subcommunicator;dg;vtk;restart"
    TIMEOUT 180
    RUN_SERIAL TRUE
    PROCESSORS 4
  )
  add_test(NAME dg_igr_vtk_restart_static_source_mismatch_cpu
    COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:dg_igr_vtk_restart_subcommunicator_test>
            -DOUTPUT_ROOT=${CMAKE_BINARY_DIR}/tests/vtk-restart-static-source-mismatch-cpu
            -DMPIEXEC=${MPIEXEC_EXECUTABLE}
            -DMPI_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_vtk_restart_source_mismatch.cmake
  )
  set_tests_properties(dg_igr_vtk_restart_static_source_mismatch_cpu PROPERTIES
    LABELS "mpi;subcommunicator;dg;vtk;restart;static-source"
    TIMEOUT 180
    RUN_SERIAL TRUE
    PROCESSORS 4
  )
endif()

add_executable(dg_igr_discretization_cli_boundary_test
  test_discretization_cli_boundary.cpp
)
target_link_libraries(dg_igr_discretization_cli_boundary_test PRIVATE
  igr_mfem
  mfem
)
target_include_directories(dg_igr_discretization_cli_boundary_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
add_test(NAME dg_igr_discretization_cli_hdg_rejected
  COMMAND ${CMAKE_COMMAND}
          -DTEST_EXE=$<TARGET_FILE:dg_igr_discretization_cli_boundary_test>
          -DSENTINEL=${CMAKE_CURRENT_BINARY_DIR}/discretization-cli-boundary.sentinel
          -P ${CMAKE_CURRENT_SOURCE_DIR}/check_discretization_cli_hdg_rejected.cmake
)
set_tests_properties(dg_igr_discretization_cli_hdg_rejected PROPERTIES
  LABELS "config;cli;discretization"
  TIMEOUT 30
)

add_executable(dg_igr_zhang_shu_sequential_test
  test_zhang_shu_sequential.cpp
)
target_link_libraries(dg_igr_zhang_shu_sequential_test PRIVATE igr_mfem mfem)
target_include_directories(dg_igr_zhang_shu_sequential_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_test(NAME dg_igr_zhang_shu_sequential_cpu
  COMMAND dg_igr_zhang_shu_sequential_test cpu
)
set_tests_properties(dg_igr_zhang_shu_sequential_cpu PROPERTIES
  LABELS "limiter;admissibility;conservation"
  TIMEOUT 120
  RUN_SERIAL TRUE
)

if (MFEM_USE_CUDA)
  add_test(NAME dg_igr_zhang_shu_sequential_cuda
    COMMAND dg_igr_zhang_shu_sequential_test cuda
  )
  set_tests_properties(dg_igr_zhang_shu_sequential_cuda PROPERTIES
    LABELS "cuda;limiter;admissibility;conservation"
    TIMEOUT 120
    RUN_SERIAL TRUE
  )

  set_source_files_properties(test_analysis_device_alias.cpp PROPERTIES
    LANGUAGE CUDA
  )
  add_executable(dg_igr_analysis_device_alias_test
    test_analysis_device_alias.cpp
  )
  set_target_properties(dg_igr_analysis_device_alias_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_options(dg_igr_analysis_device_alias_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  target_link_libraries(dg_igr_analysis_device_alias_test PRIVATE igr_mfem mfem)
  target_include_directories(dg_igr_analysis_device_alias_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  add_test(NAME dg_igr_analysis_device_alias_cuda
    COMMAND dg_igr_analysis_device_alias_test cuda
  )
  set_tests_properties(dg_igr_analysis_device_alias_cuda PROPERTIES
    LABELS "cuda;analysis;diagnostics"
    TIMEOUT 120
    RUN_SERIAL TRUE
  )

  add_executable(dg_igr_limiter_device_alias_test
    test_limiter_device_alias.cpp
  )
  target_link_libraries(dg_igr_limiter_device_alias_test PRIVATE igr_mfem mfem)
  target_include_directories(dg_igr_limiter_device_alias_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  add_test(NAME dg_igr_limiter_device_alias_cuda
    COMMAND dg_igr_limiter_device_alias_test cuda
  )
  set_tests_properties(dg_igr_limiter_device_alias_cuda PROPERTIES
    LABELS "cuda;limiter;admissibility"
    TIMEOUT 120
    RUN_SERIAL TRUE
  )

  add_executable(dg_igr_exact_iid_device_parity_test
    test_exact_iid_device_parity.cpp
  )
  target_link_libraries(dg_igr_exact_iid_device_parity_test PRIVATE
    igr_mfem
    mfem
  )
  target_include_directories(dg_igr_exact_iid_device_parity_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  )
  set(DG_IGR_EXACT_IID_DEVICE_PARITY_REF
      "${CMAKE_CURRENT_BINARY_DIR}/exact-iid-device-parity-reference.txt")
  add_test(NAME dg_igr_exact_iid_device_parity_cpu_reference
    COMMAND dg_igr_exact_iid_device_parity_test
            cpu --write-reference ${DG_IGR_EXACT_IID_DEVICE_PARITY_REF}
  )
  set_tests_properties(dg_igr_exact_iid_device_parity_cpu_reference PROPERTIES
    FIXTURES_SETUP dg_igr_exact_iid_device_parity_reference
    LABELS "cpu-reference;dg;igr;invariant-domain-flux;parity"
    TIMEOUT 120
    RUN_SERIAL TRUE
  )
  add_test(NAME dg_igr_exact_iid_device_parity_cuda
    COMMAND dg_igr_exact_iid_device_parity_test
            cuda --compare ${DG_IGR_EXACT_IID_DEVICE_PARITY_REF}
  )
  set_tests_properties(dg_igr_exact_iid_device_parity_cuda PROPERTIES
    FIXTURES_REQUIRED dg_igr_exact_iid_device_parity_reference
    LABELS "cuda;dg;igr;invariant-domain-flux;parity"
    TIMEOUT 120
    RUN_SERIAL TRUE
  )
endif()

add_test(NAME dg_igr_euler_tests COMMAND dg_igr_unit_tests --euler)
add_test(NAME dg_igr_flux_tests COMMAND dg_igr_unit_tests --fluxes)
add_test(NAME dg_igr_limiter_tests COMMAND dg_igr_unit_tests --limiters)
add_test(NAME dg_igr_dg_operator_tests COMMAND dg_igr_unit_tests --dg-operator)
add_test(NAME dg_igr_dgsem_basis_tests COMMAND dg_igr_unit_tests --dgsem-basis)
add_test(NAME dg_igr_dgsem_flux_tests COMMAND dg_igr_unit_tests --dgsem-flux)
add_test(NAME dg_igr_dgsem_operator_tests COMMAND dg_igr_unit_tests --dgsem-operator)
if(MFEM_USE_CUDA)
  add_test(NAME dg_igr_dgsem_operator_cuda
    COMMAND dg_igr_unit_tests --dgsem-operator
  )
  set_tests_properties(dg_igr_dgsem_operator_cuda PROPERTIES
    ENVIRONMENT "DG_IGR_TEST_DEVICE=cuda;DG_IGR_DGSEM_DEVICE=cuda"
    LABELS "cuda;dgsem;production;llf"
    TIMEOUT 180
    RUN_SERIAL TRUE
  )
endif()
add_test(NAME dg_igr_dgsem_advection_tests COMMAND dg_igr_unit_tests --dgsem-advection)
set_tests_properties(dg_igr_dgsem_advection_tests PROPERTIES
  LABELS "numerical-acceptance;numerical-convergence;convergence;spatial;dgsem;production;cpu")
add_test(NAME dg_igr_dgsem_igr_tests COMMAND dg_igr_unit_tests --dgsem-igr)
add_test(NAME dg_igr_dgsem_portability_refactor_tests
         COMMAND dg_igr_unit_tests --dgsem-operator --dgsem-igr)
add_test(NAME dg_igr_bc_tests COMMAND dg_igr_unit_tests --bc)
add_test(NAME dg_igr_igr_tests COMMAND dg_igr_unit_tests --igr)
add_test(NAME dg_igr_time_defaults_tests COMMAND dg_igr_unit_tests --time-defaults)
add_test(NAME dg_igr_diagnostics_tests COMMAND dg_igr_unit_tests --diagnostics)
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA)
  add_test(NAME dg_igr_diagnostics_cuda COMMAND dg_igr_unit_tests --diagnostics)
  set_tests_properties(dg_igr_diagnostics_cuda PROPERTIES
    ENVIRONMENT "DG_IGR_TEST_DEVICE=cuda"
    LABELS "cuda;diagnostics;parity" TIMEOUT 180 RUN_SERIAL TRUE)
endif()

if(TARGET dgsem_smooth_advection_periodic)
  add_test(NAME dg_igr_hdg_static_source_legacy_ignored_by_dgsem
    COMMAND dgsem_smooth_advection_periodic
            -dev cpu -rs 0 -rp 0 -o 1 -tf 1.0e-4 -dt 1.0e-4 -c 0.0
            -no-glvis -no-vtk -no-save-state -iprint 0
            --hdg-static-source 0.01
  )
  set_tests_properties(dg_igr_hdg_static_source_legacy_ignored_by_dgsem
    PROPERTIES
      LABELS "config;compatibility;dgsem;static-source"
      TIMEOUT 120
  )

  add_test(NAME dg_igr_static_source_ordinary_dg_smoke
    COMMAND dgsem_smooth_advection_periodic
            -dev cpu -rs 0 -rp 0 -o 1 -tf 1.0e-4 -dt 1.0e-4 -c 0.0
            -no-glvis -no-vtk -no-save-state -iprint 0
            --dg-backend dg --static-source 0.01
  )
  set_tests_properties(dg_igr_static_source_ordinary_dg_smoke
    PROPERTIES
      LABELS "dg;static-source;smoke"
      TIMEOUT 120
  )
endif()

if (TARGET supersonic_cylinder_w_igr)
  add_test(NAME dg_igr_sigma_vtk_lifecycle_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_sigma_vtk_lifecycle.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --device cpu
            --out-dir ${CMAKE_BINARY_DIR}/tests/sigma-vtk-lifecycle-cpu
            --failure-control
            --igr-stage-rtol-scale 10
  )
  set_tests_properties(dg_igr_sigma_vtk_lifecycle_cpu PROPERTIES
    LABELS "igr;vtk;lifecycle"
    TIMEOUT 180
    RUN_SERIAL TRUE
  )

  add_test(NAME dg_igr_vtk_restart_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --device cpu
            --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-cpu
            --format-controls
            --negative-controls
  )
  set_tests_properties(dg_igr_vtk_restart_cpu PROPERTIES
    LABELS "igr;vtk;restart"
    TIMEOUT 300
    RUN_SERIAL TRUE
  )

  add_test(NAME dg_igr_vtk_restart_ssprk104_iidflux_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --device cpu
            --time-integrator ssprk104
            --invariant-domain-flux
            --zhang-shu-vtk-points
            --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-ssprk104-iidflux-cpu
            --negative-controls
  )
  set_tests_properties(dg_igr_vtk_restart_ssprk104_iidflux_cpu PROPERTIES
    LABELS "igr;vtk;restart;ssprk104;invariant-domain-flux"
    TIMEOUT 360
    RUN_SERIAL TRUE
  )

  foreach(restart_order IN ITEMS 2 3)
    add_test(NAME dg_igr_vtk_restart_order${restart_order}_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cpu
              --order ${restart_order}
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-order${restart_order}-cpu
    )
    set_tests_properties(dg_igr_vtk_restart_order${restart_order}_cpu PROPERTIES
      LABELS "igr;vtk;restart;high-order"
      TIMEOUT 300
      RUN_SERIAL TRUE
    )
  endforeach()

  add_test(NAME dg_igr_vtk_restart_cfl_save_boundary_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --device cpu
            --cfl-driven
            --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-cfl-save-boundary-cpu
  )
  set_tests_properties(dg_igr_vtk_restart_cfl_save_boundary_cpu PROPERTIES
    LABELS "igr;vtk;restart;cfl"
    TIMEOUT 240
    RUN_SERIAL TRUE
  )

  if (TARGET supersonic_cylinder)
    add_test(NAME dg_vtk_restart_no_igr_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:supersonic_cylinder>
              --device cpu
              --no-igr
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-no-igr-cpu
    )
    set_tests_properties(dg_vtk_restart_no_igr_cpu PROPERTIES
      LABELS "dg;vtk;restart"
      TIMEOUT 180
      RUN_SERIAL TRUE
    )

    add_test(NAME dg_vtk_restart_ssprk104_no_igr_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:supersonic_cylinder>
              --device cpu
              --no-igr
              --time-integrator ssprk104
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-ssprk104-no-igr-cpu
    )
    set_tests_properties(dg_vtk_restart_ssprk104_no_igr_cpu PROPERTIES
      LABELS "dg;vtk;restart;ssprk104;conventional"
      TIMEOUT 240
      RUN_SERIAL TRUE
    )
  endif()

  if (MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
    add_test(NAME dg_igr_vtk_restart_mpi_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cpu
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-mpi-cpu
              --mpi-ranks 2
              --mpiexec ${MPIEXEC_EXECUTABLE}
              --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
    )
    set_tests_properties(dg_igr_vtk_restart_mpi_cpu PROPERTIES
      LABELS "mpi;igr;vtk;restart;exact-face"
      TIMEOUT 300
      RUN_SERIAL TRUE
      PROCESSORS 2
    )

    add_test(NAME dg_igr_vtk_restart_ssprk104_iidflux_mpi_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cpu
              --time-integrator ssprk104
              --invariant-domain-flux
              --zhang-shu-vtk-points
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-ssprk104-iidflux-mpi-cpu
              --mpi-ranks 2
              --mpiexec ${MPIEXEC_EXECUTABLE}
              --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
    )
    set_tests_properties(dg_igr_vtk_restart_ssprk104_iidflux_mpi_cpu PROPERTIES
      LABELS "mpi;igr;vtk;restart;ssprk104;invariant-domain-flux"
      TIMEOUT 360
      RUN_SERIAL TRUE
      PROCESSORS 2
    )
  endif()

  if (MFEM_USE_CUDA)
    add_test(NAME dg_igr_sigma_vtk_lifecycle_cuda
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_sigma_vtk_lifecycle.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cuda
              --out-dir ${CMAKE_BINARY_DIR}/tests/sigma-vtk-lifecycle-cuda
    )
    set_tests_properties(dg_igr_sigma_vtk_lifecycle_cuda PROPERTIES
      LABELS "cuda;igr;vtk;lifecycle"
      TIMEOUT 180
      RUN_SERIAL TRUE
    )

    add_test(NAME dg_igr_vtk_restart_cuda
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cuda
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-cuda
    )
    set_tests_properties(dg_igr_vtk_restart_cuda PROPERTIES
      LABELS "cuda;igr;vtk;restart;exact-face"
      TIMEOUT 300
      RUN_SERIAL TRUE
    )

    add_test(NAME dg_igr_vtk_restart_ssprk104_iidflux_cuda
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cuda
              --time-integrator ssprk104
              --invariant-domain-flux
              --zhang-shu-vtk-points
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-ssprk104-iidflux-cuda
    )
    set_tests_properties(dg_igr_vtk_restart_ssprk104_iidflux_cuda PROPERTIES
      LABELS "cuda;igr;vtk;restart;ssprk104;invariant-domain-flux"
      TIMEOUT 360
      RUN_SERIAL TRUE
    )

    if (TARGET supersonic_cylinder)
      add_test(NAME dg_vtk_restart_ssprk104_no_igr_cuda
        COMMAND /usr/bin/env python3
                ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
                ${_igrlab_serial_mpi_args}
                --exe $<TARGET_FILE:supersonic_cylinder>
                --device cuda
                --no-igr
                --time-integrator ssprk104
                --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-ssprk104-no-igr-cuda
      )
      set_tests_properties(dg_vtk_restart_ssprk104_no_igr_cuda PROPERTIES
        LABELS "cuda;dg;vtk;restart;ssprk104;conventional"
        TIMEOUT 300
        RUN_SERIAL TRUE
      )
    endif()

    if (MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
      add_test(NAME dg_igr_sigma_vtk_lifecycle_mpi_cuda
        COMMAND /usr/bin/env python3
                ${CMAKE_SOURCE_DIR}/tests/scripts/check_sigma_vtk_lifecycle.py
                --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
                --device cuda
                --out-dir ${CMAKE_BINARY_DIR}/tests/sigma-vtk-lifecycle-mpi-cuda
                --mpi-ranks 2
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
      )
      set_tests_properties(dg_igr_sigma_vtk_lifecycle_mpi_cuda PROPERTIES
        LABELS "cuda;mpi;igr;vtk;lifecycle;exact-face"
        TIMEOUT 240
        RUN_SERIAL TRUE
        PROCESSORS 2
      )

      add_test(NAME dg_igr_vtk_restart_mpi_cuda
        COMMAND /usr/bin/env python3
                ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
                --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
                --device cuda
                --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-mpi-cuda
                --mpi-ranks 2
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
      )
      set_tests_properties(dg_igr_vtk_restart_mpi_cuda PROPERTIES
        LABELS "cuda;mpi;igr;vtk;restart;exact-face"
        TIMEOUT 360
        RUN_SERIAL TRUE
        PROCESSORS 2
      )

      add_test(NAME dg_igr_vtk_restart_ssprk104_iidflux_mpi_cuda
        COMMAND /usr/bin/env python3
                ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
                --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
                --device cuda
                --time-integrator ssprk104
                --invariant-domain-flux
                --zhang-shu-vtk-points
                --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-ssprk104-iidflux-mpi-cuda
                --mpi-ranks 2
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
      )
      set_tests_properties(dg_igr_vtk_restart_ssprk104_iidflux_mpi_cuda PROPERTIES
        LABELS "cuda;mpi;igr;vtk;restart;ssprk104;invariant-domain-flux"
        TIMEOUT 420
        RUN_SERIAL TRUE
        PROCESSORS 2
      )
    endif()
  endif()
endif()

if (TARGET dgsem_sedov_taylor_blast)
  add_test(NAME dgsem_vtk_restart_p1_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
            --device cpu
            --order 1
            --negative-controls
            --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-p1-cpu
  )
  set_tests_properties(dgsem_vtk_restart_p1_cpu PROPERTIES
    LABELS "dgsem;vtk;restart;cold;gauss-lobatto;host-route;negative"
    TIMEOUT 600
    RUN_SERIAL TRUE
  )

  add_test(NAME dgsem_vtk_restart_p4_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
            --device cpu
            --order 4
            --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-p4-cpu
  )
  set_tests_properties(dgsem_vtk_restart_p4_cpu PROPERTIES
    LABELS "dgsem;vtk;restart;cold;gauss-lobatto;host-route;high-order"
    TIMEOUT 300
    RUN_SERIAL TRUE
  )

  add_test(NAME dgsem_vtk_restart_p3_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
            --device cpu
            --order 3
            --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-p3-cpu
  )
  set_tests_properties(dgsem_vtk_restart_p3_cpu PROPERTIES
    LABELS "dgsem;vtk;restart;cold;gauss-lobatto;host-route;interior-order"
    TIMEOUT 300
    RUN_SERIAL TRUE
  )

  add_test(NAME dgsem_vtk_restart_extrapolated_llf_p1_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
            ${_igrlab_serial_mpi_args}
            --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
            --device cpu
            --order 1
            --extrapolated-llf
            --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-extrapolated-llf-p1-cpu
  )
  set_tests_properties(dgsem_vtk_restart_extrapolated_llf_p1_cpu PROPERTIES
    LABELS "cpu-reference;dgsem;vtk;restart;cold;llf;igr;extrapolated;packed-production"
    TIMEOUT 600
    RUN_SERIAL TRUE
  )

  if (MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
    add_test(NAME dgsem_vtk_restart_p2_mpi_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
              --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
              --device cpu
              --order 2
              --mpi-ranks 2
              --mpiexec ${MPIEXEC_EXECUTABLE}
              --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
              --rank-piece-reorder-negative
              --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-p2-mpi-cpu
    )
    set_tests_properties(dgsem_vtk_restart_p2_mpi_cpu PROPERTIES
      LABELS "mpi;dgsem;vtk;restart;cold;gauss-lobatto;host-route"
      TIMEOUT 300
      RUN_SERIAL TRUE
      PROCESSORS 2
      SKIP_RETURN_CODE 75
    )

    add_test(NAME dgsem_vtk_restart_extrapolated_llf_p1_mpi_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
              --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
              --device cpu
              --order 1
              --extrapolated-llf
              --mpi-ranks 2
              --mpiexec ${MPIEXEC_EXECUTABLE}
              --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
              --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-extrapolated-llf-p1-mpi-cpu
    )
    set_tests_properties(
      dgsem_vtk_restart_extrapolated_llf_p1_mpi_cpu PROPERTIES
      LABELS "cpu-reference;mpi;dgsem;vtk;restart;cold;llf;igr;extrapolated;packed-production"
      TIMEOUT 600
      RUN_SERIAL TRUE
      PROCESSORS 2
      SKIP_RETURN_CODE 75
    )
  endif()

  if (MFEM_USE_CUDA)
    add_test(NAME dgsem_vtk_restart_p2_cuda
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
              --device cuda
              --order 2
              --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-p2-cuda
    )
    set_tests_properties(dgsem_vtk_restart_p2_cuda PROPERTIES
      LABELS "cuda;dgsem;vtk;restart;cold;gauss-lobatto;host-route;pre-composition"
      TIMEOUT 600
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 75
    )

    add_test(NAME dgsem_vtk_restart_extrapolated_llf_p1_cuda
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
              ${_igrlab_serial_mpi_args}
              --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
              --device cuda
              --order 1
              --extrapolated-llf
              --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-extrapolated-llf-p1-cuda
    )
    set_tests_properties(dgsem_vtk_restart_extrapolated_llf_p1_cuda PROPERTIES
      LABELS "cuda;dgsem;vtk;restart;cold;llf;igr;extrapolated;packed-production"
      TIMEOUT 600
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 75
    )

    if (MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
      add_test(NAME dgsem_vtk_restart_extrapolated_llf_p1_mpi_cuda
        COMMAND /usr/bin/env python3
                ${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_vtk_restart.py
                --exe $<TARGET_FILE:dgsem_sedov_taylor_blast>
                --device cuda
                --order 1
                --extrapolated-llf
                --mpi-ranks 2
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
                --out-dir ${CMAKE_BINARY_DIR}/tests/dgsem-vtk-restart-extrapolated-llf-p1-mpi-cuda
      )
      set_tests_properties(
        dgsem_vtk_restart_extrapolated_llf_p1_mpi_cuda PROPERTIES
        LABELS "cuda;mpi;dgsem;vtk;restart;cold;llf;igr;extrapolated;packed-production"
        TIMEOUT 600
        RUN_SERIAL TRUE
        PROCESSORS 2
        SKIP_RETURN_CODE 75
      )
    endif()
  endif()
endif()

set_tests_properties(dg_igr_dgsem_portability_refactor_tests PROPERTIES
  LABELS "dgsem;portability"
  TIMEOUT 180
)

if (MFEM_USE_CUDA AND TARGET supersonic_cylinder_w_igr)
  add_test(NAME dg_igr_cylinder_l2_cpu_cuda_pvd_drift_exact_face
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_cpu_cuda_pvd_drift.py
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --out-dir ${CMAKE_BINARY_DIR}/tests/cylinder-l2-cpu-cuda-pvd-drift-exact-face
            --order 1
            --serial-ref 0
            --parallel-ref 0
            --tf 2.0e-4
            --dt 1.0e-4
            --dt-save 1.0e-4
            --igr-stage-rtol-scale 10
            --igr-lin-rtol 1.0e-8
            --cpu-igr-lin-atol 0.0
            --cuda-igr-lin-atol 0.0
            --cpu-igr-linear-solver minres
            --cuda-igr-linear-solver minres
            --cpu-igr-preconditioner chebyshev
            --cuda-igr-preconditioner chebyshev
            --default-rel-rms-limit 1.0e-5
            --cpu-igr-exact-dg-face
            --cuda-igr-exact-dg-face
            --cpu-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
            --cpu-env DG_IGR_IGR_DEVICE_QF=1
            --cpu-env DG_IGR_DEBUG_QF=1
            --cpu-env DG_IGR_IGR_CHEB_ORDER=3
            --cpu-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
            --cuda-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
            --cuda-env DG_IGR_IGR_DEVICE_QF=1
            --cuda-env DG_IGR_DEBUG_QF=1
            --cuda-env DG_IGR_IGR_CHEB_ORDER=3
            --cuda-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
            --require-cpu-igr-converged
            --require-cuda-igr-converged
            --require-cpu-log "event=exact-dg-face-operator"
            --require-cpu-log "exact_face_effective=1"
            --require-cuda-log "event=exact-dg-face-operator"
            --require-cuda-log "exact_face_effective=1"
  )
  set_tests_properties(dg_igr_cylinder_l2_cpu_cuda_pvd_drift_exact_face
    PROPERTIES
      LABELS "cuda;parity;igr;vtk;exact-face"
      TIMEOUT 360
  )

  add_test(NAME dg_igr_cylinder_l2_cuda_direct_identity_parity_exact_face
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_cuda_direct_identity_parity.py
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --out-dir ${CMAKE_BINARY_DIR}/tests/cylinder-l2-cuda-direct-identity-parity-exact-face
            --order 1
            --serial-ref 0
            --parallel-ref 0
            --tf 2.0e-4
            --dt 1.0e-4
            --dt-save 1.0e-4
            --igr-stage-rtol-scale 10
            --igr-lin-rtol 1.0e-8
            --default-rel-rms-limit 1.0e-8
            --intermediate-sigma-rel-rms-limit 1.0e-7
  )
  set_tests_properties(dg_igr_cylinder_l2_cuda_direct_identity_parity_exact_face
    PROPERTIES
      LABELS "cuda;parity;igr;vtk;exact-face;direct-identity"
      TIMEOUT 360
  )

  add_test(NAME dg_igr_cylinder_l2_cpu_cuda_pvd_drift_exact_face_production
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_cpu_cuda_pvd_drift.py
            --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
            --out-dir ${CMAKE_BINARY_DIR}/tests/cylinder-l2-cpu-cuda-pvd-drift-exact-face-production
            --order 2
            --serial-ref 0
            --parallel-ref 1
            --igr-space L2
            --igr-grad BrokenL2
            --tf 1.0e-3
            --dt 1.0e-4
            --dt-save 2.0e-4
            --disable-vtk-diagnostics
            --arrays density pressure sigma energy momentum velocity
            --igr-stage-rtol-scale 1
            --cpu-igr-lin-rtol 1.0e-8
            --cuda-igr-lin-rtol 1.0e-5
            --cpu-igr-lin-maxit 1000
            --cpu-igr-lin-atol 0.0
            --cuda-igr-lin-atol 0.0
            --cpu-igr-linear-solver minres
            --cuda-igr-linear-solver minres
            --cpu-igr-preconditioner chebyshev
            --cuda-igr-preconditioner chebyshev
            --cpu-igr-exact-dg-face
            --cuda-igr-exact-dg-face
            --default-rel-rms-limit 1.0e-5
            --nan-scan
            --admissibility-scan
            --cpu-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
            --cpu-env DG_IGR_IGR_DEVICE_QF=1
            --cpu-env DG_IGR_DEBUG_QF=1
            --cpu-env DG_IGR_IGR_CHEB_ORDER=3
            --cpu-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
            --cuda-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
            --cuda-env DG_IGR_IGR_DEVICE_QF=1
            --cuda-env DG_IGR_DEBUG_QF=1
            --cuda-env DG_IGR_IGR_CHEB_ORDER=3
            --cuda-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
            --require-cpu-igr-converged
            --require-cuda-igr-converged
            --require-cpu-log "event=exact-dg-face-operator"
            --require-cpu-log "last_solver=minres"
            --require-cpu-log "last_precond=operatorchebyshev-base"
            --require-cpu-log "exact_face_effective=1"
            --require-cuda-log "event=exact-dg-face-operator"
            --require-cuda-log "last_solver=minres"
            --require-cuda-log "last_precond=operatorchebyshev-base"
            --require-cuda-log "exact_face_effective=1"
  )
  set_tests_properties(dg_igr_cylinder_l2_cpu_cuda_pvd_drift_exact_face_production
    PROPERTIES
      LABELS "cuda;parity;igr;vtk;exact-face;production"
      TIMEOUT 900
      RUN_SERIAL TRUE
  )

  if (MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
    add_test(NAME dg_igr_cylinder_l2_mpi_cpu_cuda_pvd_drift_exact_face
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_cpu_cuda_pvd_drift.py
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --out-dir ${CMAKE_BINARY_DIR}/tests/cylinder-l2-mpi-cpu-cuda-pvd-drift-exact-face
              --mpi-ranks 2
              --mpiexec ${MPIEXEC_EXECUTABLE}
              --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
              --order 1
              --serial-ref 0
              --parallel-ref 0
              --tf 2.0e-4
              --dt 1.0e-4
              --dt-save 1.0e-4
              --igr-stage-rtol-scale 10
              --igr-lin-rtol 1.0e-8
              --cpu-igr-lin-atol 0.0
              --cuda-igr-lin-atol 0.0
              --cpu-igr-linear-solver minres
              --cuda-igr-linear-solver minres
              --cpu-igr-preconditioner chebyshev
              --cuda-igr-preconditioner chebyshev
              --default-rel-rms-limit 1.0e-5
              --cpu-igr-exact-dg-face
              --cuda-igr-exact-dg-face
              --cpu-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
              --cpu-env DG_IGR_IGR_DEVICE_QF=1
              --cpu-env DG_IGR_DEBUG_QF=1
              --cpu-env DG_IGR_IGR_CHEB_ORDER=3
              --cpu-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
              --cuda-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
              --cuda-env DG_IGR_IGR_DEVICE_QF=1
              --cuda-env DG_IGR_DEBUG_QF=1
              --cuda-env DG_IGR_IGR_CHEB_ORDER=3
              --cuda-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
              --require-cpu-igr-converged
              --require-cuda-igr-converged
              --require-cpu-log "event=exact-dg-face-operator"
              --require-cpu-log "exact_face_effective=1"
              --require-cuda-log "event=exact-dg-face-operator"
              --require-cuda-log "mode=local-and-shared-2d"
              --require-cuda-log "exact_face_effective=1"
    )
    set_tests_properties(dg_igr_cylinder_l2_mpi_cpu_cuda_pvd_drift_exact_face
      PROPERTIES
        LABELS "cuda;mpi;parity;igr;vtk;exact-face"
        TIMEOUT 480
    )

    add_test(NAME dg_igr_cylinder_l2_mpi_cpu_cuda_pvd_drift_exact_face_production
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_cpu_cuda_pvd_drift.py
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --out-dir ${CMAKE_BINARY_DIR}/tests/cylinder-l2-mpi-cpu-cuda-pvd-drift-exact-face-production
              --mpi-ranks 2
              --mpiexec ${MPIEXEC_EXECUTABLE}
              --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
              --order 2
              --serial-ref 0
              --parallel-ref 1
              --igr-space L2
              --igr-grad BrokenL2
              --tf 1.0e-3
              --dt 1.0e-4
              --dt-save 2.0e-4
              --disable-vtk-diagnostics
              --arrays density pressure sigma energy momentum velocity
              --igr-stage-rtol-scale 1
              --cpu-igr-lin-rtol 1.0e-8
              --cuda-igr-lin-rtol 1.0e-5
              --cpu-igr-lin-maxit 1000
              --cpu-igr-lin-atol 0.0
              --cuda-igr-lin-atol 0.0
              --cpu-igr-linear-solver minres
              --cuda-igr-linear-solver minres
              --cpu-igr-preconditioner chebyshev
              --cuda-igr-preconditioner chebyshev
              --cpu-igr-exact-dg-face
              --cuda-igr-exact-dg-face
              --default-rel-rms-limit 1.0e-5
              --nan-scan
              --admissibility-scan
              --cpu-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
              --cpu-env DG_IGR_IGR_DEVICE_QF=1
              --cpu-env DG_IGR_DEBUG_QF=1
              --cpu-env DG_IGR_IGR_CHEB_ORDER=3
              --cpu-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
              --cuda-env DG_IGR_IGR_PARTIAL_ASSEMBLY=1
              --cuda-env DG_IGR_IGR_DEVICE_QF=1
              --cuda-env DG_IGR_DEBUG_QF=1
              --cuda-env DG_IGR_IGR_CHEB_ORDER=3
              --cuda-env DG_IGR_IGR_CHEB_BASE_OPERATOR=1
              --require-cpu-igr-converged
              --require-cuda-igr-converged
              --require-cpu-log "event=exact-dg-face-operator"
              --require-cpu-log "last_solver=minres"
              --require-cpu-log "last_precond=operatorchebyshev-base"
              --require-cpu-log "exact_face_effective=1"
              --require-cuda-log "event=exact-dg-face-operator"
              --require-cuda-log "mode=local-and-shared-2d"
              --require-cuda-log "last_solver=minres"
              --require-cuda-log "last_precond=operatorchebyshev-base"
              --require-cuda-log "exact_face_effective=1"
    )
    set_tests_properties(dg_igr_cylinder_l2_mpi_cpu_cuda_pvd_drift_exact_face_production
      PROPERTIES
        LABELS "cuda;mpi;parity;igr;vtk;exact-face;production"
        TIMEOUT 1200
        RUN_SERIAL TRUE
        PROCESSORS 2
    )
  endif()
endif()

# MPI parity checks: generate a 1-rank reference and compare against 2 ranks.
# Keep this list curated to suites with stable, decomposition-invariant metrics.
if (MPIEXEC_EXECUTABLE AND MPIEXEC_NUMPROC_FLAG)
  set(DG_IGR_PARITY_REF_FILE "${CMAKE_BINARY_DIR}/tests/dg_igr_parity_ref.txt")
  set(DG_IGR_PARITY_SUITES
    --dg-operator
    --dgsem-flux
    --dgsem-operator
    --dgsem-advection
    --dgsem-igr
    --diagnostics
    --time-defaults
  )
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

  add_test(NAME dg_igr_parity_ref_1rank
    COMMAND ${CMAKE_COMMAND} -E env
            DG_IGR_PARITY_REF=${DG_IGR_PARITY_REF_FILE}
            DG_IGR_PARITY_WRITE=1
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --parity ${DG_IGR_PARITY_SUITES}
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_parity_ref_1rank PROPERTIES
    FIXTURES_SETUP dg_igr_parity_ref
    LABELS "parity"
    TIMEOUT 120
  )

  add_test(NAME dg_igr_parity_check_2rank
    COMMAND ${CMAKE_COMMAND} -E env
            DG_IGR_PARITY_REF=${DG_IGR_PARITY_REF_FILE}
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --parity ${DG_IGR_PARITY_SUITES}
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_parity_check_2rank PROPERTIES
    FIXTURES_REQUIRED dg_igr_parity_ref
    LABELS "parity"
    TIMEOUT 120
  )

  add_test(NAME dg_igr_dgsem_operator_mpi_2rank
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --dgsem-operator
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_dgsem_operator_mpi_2rank PROPERTIES
    LABELS "mpi"
    TIMEOUT 120
  )

  if(MFEM_USE_CUDA)
    add_test(NAME dg_igr_dgsem_operator_mpi_2rank_cuda
      COMMAND ${CMAKE_COMMAND} -E env
              DG_IGR_TEST_DEVICE=cuda
              DG_IGR_DGSEM_DEVICE=cuda
              ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:dg_igr_unit_tests> --dgsem-operator
              ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(dg_igr_dgsem_operator_mpi_2rank_cuda PROPERTIES
      LABELS "cuda;mpi;dgsem;production;llf"
      TIMEOUT 180
      RUN_SERIAL TRUE
      PROCESSORS 2
    )

    add_test(NAME dg_igr_dgsem_operator_split_mpi_4rank_cuda
      COMMAND ${CMAKE_COMMAND} -E env
              DG_IGR_TEST_DEVICE=cuda
              DG_IGR_DGSEM_DEVICE=cuda
              DG_IGR_DGSEM_SPLIT_COMM_PROBE=1
              ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:dg_igr_unit_tests> --dgsem-operator
              ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
      dg_igr_dgsem_operator_split_mpi_4rank_cuda PROPERTIES
      LABELS "cuda;mpi;subcommunicator;dgsem;production;llf"
      TIMEOUT 180
      RUN_SERIAL TRUE
      PROCESSORS 4
    )
  endif()

  add_test(NAME dg_igr_diagnostics_mpi_2rank
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --diagnostics
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_diagnostics_mpi_2rank PROPERTIES
    LABELS "mpi"
    TIMEOUT 120
  )

  add_test(NAME dg_igr_dgsem_flux_mpi_2rank
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --dgsem-flux
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_dgsem_flux_mpi_2rank PROPERTIES
    LABELS "mpi"
    TIMEOUT 120
  )

  add_test(NAME dg_igr_igr_mpi_2rank
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --igr
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_igr_mpi_2rank PROPERTIES
    LABELS "mpi"
    TIMEOUT 120
  )

  add_test(NAME dg_igr_limiters_mpi_2rank
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:dg_igr_unit_tests> --limiters
            ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(dg_igr_limiters_mpi_2rank PROPERTIES
    LABELS "mpi"
    TIMEOUT 120
  )

  if (TARGET dgsem_smooth_advection)
    set(reg_dir "${CMAKE_BINARY_DIR}/tests/regression/dg_igr_mpi_dgsem_smooth_advection")
    file(MAKE_DIRECTORY "${reg_dir}")
    add_test(NAME dg_igr_dgsem_advection_mpi_smoke
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:dgsem_smooth_advection>
              -m ${CMAKE_SOURCE_DIR}/examples/dgsem_smooth_advection/smooth_advection.msh
              -rs 0 -rp 0 -o 2 -tf 1.0e-4 -dt 1.0e-4 -c 0.0
              -no-glvis -vs 1000000 -nan-scan
              -no-vtk -dt_save 1.0
              --dg-backend dgsem
              ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(dg_igr_dgsem_advection_mpi_smoke PROPERTIES
      WORKING_DIRECTORY "${reg_dir}"
      LABELS "mpi"
      TIMEOUT 180
    )
  endif()
endif()

if (MFEM_USE_CUDA AND TARGET inoue_hattori_stationary_shock_1dy)
  set(cpu_cuda_parity_script
      "${CMAKE_SOURCE_DIR}/tests/scripts/check_cpu_cuda_parity.sh")
  foreach(parity_backend IN ITEMS dg dgsem)
    set(parity_test_name
        "dg_igr_inoue_${parity_backend}_cpu_cuda_parity")
    add_test(NAME ${parity_test_name}
      COMMAND ${CMAKE_COMMAND} -E env
              OMP_NUM_THREADS=1
              DG_IGR_PARITY_SOL_RTL=5e-3
              DG_IGR_PARITY_SIGMA_RTL=1e-3
              DG_IGR_PARITY_SIGMA_ATOL=1e-6
              /bin/bash
              ${cpu_cuda_parity_script}
              $<TARGET_FILE:inoue_hattori_stationary_shock_1dy>
              ${parity_backend}
              --igr-linear-solver cg
              --igr-preconditioner none
    )
    set_tests_properties(${parity_test_name} PROPERTIES
      LABELS "cuda;parity"
      TIMEOUT 240
      SKIP_REGULAR_EXPRESSION "^SKIP:"
    )
  endforeach()
endif()

if (TARGET inoue_hattori_stationary_shock_1dy)
  set(perf_script "${CMAKE_SOURCE_DIR}/tests/scripts/dgsem_perf_harness.py")
  set(perf_gate_script
      "${CMAKE_SOURCE_DIR}/tests/scripts/check_dgsem_perf_regression.py")
  set(perf_dir "${CMAKE_BINARY_DIR}/tests/perf")
  file(MAKE_DIRECTORY "${perf_dir}")

  set(perf_cpu_output "${perf_dir}/dgsem_operator_perf_cpu.json")
  add_test(NAME dg_igr_dgsem_perf_cpu_benchmark
    COMMAND ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            DG_IGR_DGSEM_TIMING=1
            /usr/bin/env python3
            ${perf_script}
            --exe $<TARGET_FILE:inoue_hattori_stationary_shock_1dy>
            --device cpu
            --output ${perf_cpu_output}
  )
  set_tests_properties(dg_igr_dgsem_perf_cpu_benchmark PROPERTIES
    FIXTURES_SETUP dg_igr_dgsem_perf_cpu
    LABELS "perf"
    TIMEOUT 600
    DISABLED TRUE
    SKIP_REGULAR_EXPRESSION "^SKIP:"
  )

  set(perf_cpu_baseline
      "${CMAKE_SOURCE_DIR}/tests/perf_refs/dgsem_operator_inoue_cpu.json")
  if (EXISTS "${perf_cpu_baseline}")
    add_test(NAME dg_igr_dgsem_perf_cpu_gate
      COMMAND /usr/bin/env python3
              ${perf_gate_script}
              --baseline ${perf_cpu_baseline}
              --current ${perf_cpu_output}
              --threshold 0.10
    )
    set_tests_properties(dg_igr_dgsem_perf_cpu_gate PROPERTIES
      FIXTURES_REQUIRED dg_igr_dgsem_perf_cpu
      LABELS "perf"
      TIMEOUT 120
      DISABLED TRUE
      SKIP_REGULAR_EXPRESSION "^SKIP:"
    )
  endif()

  if (MFEM_USE_CUDA)
    set(perf_cuda_output "${perf_dir}/dgsem_operator_perf_cuda.json")
    add_test(NAME dg_igr_dgsem_perf_cuda_benchmark
      COMMAND ${CMAKE_COMMAND} -E env
              OMP_NUM_THREADS=1
              DG_IGR_DGSEM_TIMING=1
              /usr/bin/env python3
              ${perf_script}
              --exe $<TARGET_FILE:inoue_hattori_stationary_shock_1dy>
              --device cuda
              --output ${perf_cuda_output}
    )
    set_tests_properties(dg_igr_dgsem_perf_cuda_benchmark PROPERTIES
      FIXTURES_SETUP dg_igr_dgsem_perf_cuda
      LABELS "perf;cuda"
      TIMEOUT 600
      DISABLED TRUE
      SKIP_REGULAR_EXPRESSION "^SKIP:"
    )

    set(perf_cuda_baseline
        "${CMAKE_SOURCE_DIR}/tests/perf_refs/dgsem_operator_inoue_cuda.json")
    if (EXISTS "${perf_cuda_baseline}")
      add_test(NAME dg_igr_dgsem_perf_cuda_gate
        COMMAND /usr/bin/env python3
                ${perf_gate_script}
                --baseline ${perf_cuda_baseline}
                --current ${perf_cuda_output}
                --threshold 0.10
      )
      set_tests_properties(dg_igr_dgsem_perf_cuda_gate PROPERTIES
        FIXTURES_REQUIRED dg_igr_dgsem_perf_cuda
        LABELS "perf;cuda"
        TIMEOUT 120
        DISABLED TRUE
        SKIP_REGULAR_EXPRESSION "^SKIP:"
      )
    endif()
  endif()
endif()

# Mini-regressions: run each example for a tiny time window with IO disabled.
function(add_example_regression test_name target mesh_path_rel)
  if (TARGET ${target})
    set(reg_dir "${CMAKE_BINARY_DIR}/tests/regression/${test_name}")
    file(MAKE_DIRECTORY "${reg_dir}")

    # Allow per-test overrides without duplicating CLI flags.
    set(reg_tf "1.0e-4")
    set(reg_dt "1.0e-4")
    set(extra_args "")
    set(expect_tf OFF)
    set(expect_dt OFF)
    foreach(arg IN LISTS ARGN)
      if (expect_tf)
        set(reg_tf "${arg}")
        set(expect_tf OFF)
      elseif (expect_dt)
        set(reg_dt "${arg}")
        set(expect_dt OFF)
      elseif (arg STREQUAL "REG_TF")
        set(expect_tf ON)
      elseif (arg STREQUAL "REG_DT")
        set(expect_dt ON)
      else()
        list(APPEND extra_args "${arg}")
      endif()
    endforeach()
    if (expect_tf OR expect_dt)
      message(FATAL_ERROR "REG_TF/REG_DT must be followed by a value")
    endif()

    add_test(NAME ${test_name}
      COMMAND ${target}
              -m ${CMAKE_SOURCE_DIR}/${mesh_path_rel}
              -rs 0 -rp 0 -o 1 -tf ${reg_tf} -dt ${reg_dt} -c 0.0
              -no-glvis -vs 1000000 -nan-scan
              ${extra_args}
    )
    set_tests_properties(${test_name} PROPERTIES
      WORKING_DIRECTORY "${reg_dir}"
      LABELS "regression"
      TIMEOUT 180
    )
  endif()
endfunction()

add_example_regression(dg_igr_regression_supersonic
  supersonic_cylinder
  examples/supersonic_example/supersonic_cylinder.msh)
add_example_regression(dg_igr_regression_supersonic_igr
  supersonic_cylinder_w_igr
  examples/supersonic_example_w_igr/supersonic_cylinder.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_forward_facing
  forward_facing_step
  examples/forward_facing_step/forward_facing.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_stationary_shock
  stationary_shock
  examples/stationary_shock/stationary_shock.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_dgsem_stationary_shock
  dgsem_stationary_shock
  examples/dgsem_stationary_shock/stationary_shock.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_dgsem_shock_vortex
  dgsem_shock_vortex_interaction
  examples/dgsem_shock_vortex_interaction/shock_vortex.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_dgsem_forward_facing
  dgsem_forward_facing_step
  examples/dgsem_forward_facing_step/forward_facing.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_dgsem_sedov
  dgsem_sedov_taylor_blast
  examples/dgsem_sedov_taylor_blast/sedov_taylor.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_shock_vortex
  shock_vortex_interaction
  examples/shock_vortex_interaction/shock_vortex.msh
  -no-vtk -dt_save 1.0)
add_example_regression(dg_igr_regression_sedov
  sedov_taylor_blast
  examples/sedov_taylor_blast/sedov_taylor.msh
  -no-vtk -dt_save 1.0)
# NOTE: Temporarily disabled per request (double mach regression segfaults).
# add_example_regression(dg_igr_regression_double_mach
#   double_mach_ref
#   tests/meshes/double_mach_ref_regression.mesh
#   REG_TF 1.0e-6 REG_DT 1.0e-6
#   -no-vtk -dt_save 1.0)

if(MFEM_USE_MPI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/context/CMakeLists.txt")
  add_subdirectory(context)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/time_backend/CMakeLists.txt")
  add_subdirectory(time_backend)
endif()
add_subdirectory(hdg)
if(IGRLAB_BUILD_RESEARCH_TESTS AND TARGET dg_igr_hdg_generic_cpu AND MFEM_USE_MPI)
  add_executable(dg_igr_hdg_wall_evolution_probe research/igr_hdg_wall_evolution_probe.cpp)
  target_link_libraries(dg_igr_hdg_wall_evolution_probe PRIVATE
    dg_igr_hdg_generic_cpu mfem MPI::MPI_CXX)
  target_include_directories(dg_igr_hdg_wall_evolution_probe PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(dg_igr_hdg_wall_evolution_probe PRIVATE cxx_std_17)
  add_test(NAME dg_igr_hdg_wall_evolution_cpu COMMAND dg_igr_hdg_wall_evolution_probe --self-test)
  set_tests_properties(dg_igr_hdg_wall_evolution_cpu PROPERTIES
    LABELS "cpu-reference;igr;research;hdg" TIMEOUT 120)
  add_executable(dg_igr_hdg_wall_step_probe research/igr_hdg_wall_step_probe.cpp)
  target_link_libraries(dg_igr_hdg_wall_step_probe PRIVATE
    dg_igr_hdg_generic_cpu mfem MPI::MPI_CXX)
  target_include_directories(dg_igr_hdg_wall_step_probe PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(dg_igr_hdg_wall_step_probe PRIVATE cxx_std_17)
  add_test(NAME dg_igr_hdg_wall_step_cpu COMMAND dg_igr_hdg_wall_step_probe --self-test)
  set_tests_properties(dg_igr_hdg_wall_step_cpu PROPERTIES
    LABELS "cpu-reference;igr;research;hdg" TIMEOUT 120)
endif()
add_subdirectory(regularization)

# Private shared IGR Krylov algebra fixture, independent of the HDG runtime.
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA AND MFEM_USE_MPI)
  add_executable(dg_igr_device_krylov_test test_igr_device_krylov_cuda.cu)
  target_link_libraries(dg_igr_device_krylov_test PRIVATE igr_mfem mfem MPI::MPI_CXX)
  target_compile_options(dg_igr_device_krylov_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_device_krylov_test PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME dg_igr_device_krylov_cuda COMMAND dg_igr_device_krylov_test)
  set_tests_properties(dg_igr_device_krylov_cuda PROPERTIES
    LABELS "cuda;igr;krylov" TIMEOUT 120 RUN_SERIAL TRUE)

  add_executable(dg_igr_shared_reference_test test_shared_reference_cuda.cu)
  target_link_libraries(dg_igr_shared_reference_test PRIVATE igr_mfem mfem MPI::MPI_CXX)
  target_compile_options(dg_igr_shared_reference_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_shared_reference_test PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME dg_igr_shared_reference_cuda COMMAND dg_igr_shared_reference_test)
  set_tests_properties(dg_igr_shared_reference_cuda PROPERTIES
    LABELS "cuda;dg;dgsem;shared-reference" TIMEOUT 180 RUN_SERIAL TRUE)
  add_test(NAME dg_igr_dg_volume_memory_cuda
    COMMAND dg_igr_shared_reference_test --dg-volume-memory-only)
  set_tests_properties(dg_igr_dg_volume_memory_cuda PROPERTIES
    LABELS "cuda;dg;shared-reference;volume-memory" TIMEOUT 120 RUN_SERIAL TRUE)
  add_test(NAME dg_igr_dg_tensor_face_cuda
    COMMAND dg_igr_shared_reference_test --dg-tensor-face-only)
  set_tests_properties(dg_igr_dg_tensor_face_cuda PROPERTIES
    LABELS "cuda;dg;tensor-face" TIMEOUT 120 RUN_SERIAL TRUE)
endif()

# Shared-reference patch GEMM: actual patch columns and current-A acceptance.
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA AND MFEM_USE_MPI)
  add_executable(dg_igr_shared_patch_test test_igr_shared_patch_cuda.cu)
  target_link_libraries(dg_igr_shared_patch_test PRIVATE igr_mfem mfem MPI::MPI_CXX)
  target_compile_options(dg_igr_shared_patch_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
  set_target_properties(dg_igr_shared_patch_test PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME dg_igr_shared_patch_cuda COMMAND dg_igr_shared_patch_test)
  set_tests_properties(dg_igr_shared_patch_cuda PROPERTIES
    LABELS "cuda;igr;krylov" TIMEOUT 180 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
endif()

# Experimental tensor IGR pipeline: independent operator and solve oracles.
if(DG_IGR_ACCELERATOR_LANGUAGE STREQUAL "CUDA" AND MFEM_USE_CUDA AND MFEM_USE_MPI)
  foreach(component IN ITEMS volume multigrid center_map)
    add_executable(dg_igr_tensor_${component}_test test_igr_tensor_${component}_cuda.cu)
    target_link_libraries(dg_igr_tensor_${component}_test PRIVATE igr_mfem mfem MPI::MPI_CXX)
    target_compile_options(dg_igr_tensor_${component}_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr>)
    set_target_properties(dg_igr_tensor_${component}_test PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    add_test(NAME dg_igr_tensor_${component}_cuda COMMAND dg_igr_tensor_${component}_test)
    set_tests_properties(dg_igr_tensor_${component}_cuda PROPERTIES
      LABELS "cuda;igr;tensor-pipeline" TIMEOUT 300 RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
  endforeach()
endif()
