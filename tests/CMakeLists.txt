# Unit tests for dg_igr miniapp

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

target_link_libraries(dg_igr_unit_tests PRIVATE igr_mfem mfem)

target_include_directories(dg_igr_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
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
endif()

add_test(NAME dg_igr_euler_tests COMMAND dg_igr_unit_tests --euler)
add_test(NAME dg_igr_flux_tests COMMAND dg_igr_unit_tests --fluxes)
add_test(NAME dg_igr_limiter_tests COMMAND dg_igr_unit_tests --limiters)
add_test(NAME dg_igr_dg_operator_tests COMMAND dg_igr_unit_tests --dg-operator)
add_test(NAME dg_igr_dgsem_basis_tests COMMAND dg_igr_unit_tests --dgsem-basis)
add_test(NAME dg_igr_dgsem_flux_tests COMMAND dg_igr_unit_tests --dgsem-flux)
add_test(NAME dg_igr_dgsem_operator_tests COMMAND dg_igr_unit_tests --dgsem-operator)
add_test(NAME dg_igr_dgsem_advection_tests COMMAND dg_igr_unit_tests --dgsem-advection)
add_test(NAME dg_igr_dgsem_igr_tests COMMAND dg_igr_unit_tests --dgsem-igr)
add_test(NAME dg_igr_dgsem_portability_refactor_tests
         COMMAND dg_igr_unit_tests --dgsem-operator --dgsem-igr)
add_test(NAME dg_igr_bc_tests COMMAND dg_igr_unit_tests --bc)
add_test(NAME dg_igr_igr_tests COMMAND dg_igr_unit_tests --igr)
add_test(NAME dg_igr_time_defaults_tests COMMAND dg_igr_unit_tests --time-defaults)
add_test(NAME dg_igr_diagnostics_tests COMMAND dg_igr_unit_tests --diagnostics)

if (TARGET supersonic_cylinder_w_igr)
  add_test(NAME dg_igr_sigma_vtk_lifecycle_cpu
    COMMAND /usr/bin/env python3
            ${CMAKE_SOURCE_DIR}/tests/scripts/check_sigma_vtk_lifecycle.py
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

  foreach(restart_order IN ITEMS 2 3)
    add_test(NAME dg_igr_vtk_restart_order${restart_order}_cpu
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_vtk_restart.py
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
  endif()

  if (MFEM_USE_CUDA)
    add_test(NAME dg_igr_sigma_vtk_lifecycle_cuda
      COMMAND /usr/bin/env python3
              ${CMAKE_SOURCE_DIR}/tests/scripts/check_sigma_vtk_lifecycle.py
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
              --exe $<TARGET_FILE:supersonic_cylinder_w_igr>
              --device cuda
              --out-dir ${CMAKE_BINARY_DIR}/tests/vtk-restart-cuda
    )
    set_tests_properties(dg_igr_vtk_restart_cuda PROPERTIES
      LABELS "cuda;igr;vtk;restart;exact-face"
      TIMEOUT 300
      RUN_SERIAL TRUE
    )

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

  set(dg_nsys_script
      "${CMAKE_SOURCE_DIR}/tests/scripts/profile_dg_igr_nsys.sh")
  add_test(NAME dg_igr_inoue_dg_nsys_profile
    COMMAND ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            /bin/bash
            ${dg_nsys_script}
            $<TARGET_FILE:inoue_hattori_stationary_shock_1dy>
  )
  set_tests_properties(dg_igr_inoue_dg_nsys_profile PROPERTIES
    LABELS "perf;cuda;profiling"
    TIMEOUT 1200
    DISABLED TRUE
    SKIP_REGULAR_EXPRESSION "^SKIP:"
  )

  set(dg_ncu_script
      "${CMAKE_SOURCE_DIR}/tests/scripts/profile_dg_igr_ncu.sh")
  add_test(NAME dg_igr_inoue_dg_ncu_profile
    COMMAND ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            /bin/bash
            ${dg_ncu_script}
            $<TARGET_FILE:inoue_hattori_stationary_shock_1dy>
  )
  set_tests_properties(dg_igr_inoue_dg_ncu_profile PROPERTIES
    LABELS "perf;cuda;profiling"
    TIMEOUT 1200
    DISABLED TRUE
    SKIP_REGULAR_EXPRESSION "^SKIP:"
  )
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
      COMMAND $<TARGET_FILE:${target}>
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

add_subdirectory(hdg)
add_subdirectory(regularization)
