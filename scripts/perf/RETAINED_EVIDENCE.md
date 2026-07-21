# Retained Nsight Evidence Manifest

This manifest selects four canonical historical/protected evidence families. They are reproducibility pointers for later review, not portable certification or a performance gate. The protected interpretation source is WIP commit `1fdb07a8683e33ac0e0ed70dec037c8a0564d0d6`, blob `e83006894b740b57664de3b59d7e18cef3ba7df9` (especially lines 2429–2505).

## Artifact identities

All paths are repository-relative to the historical evidence root. Sizes and SHA-256 values were read only for these ten explicitly selected files.

| Family | Path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Nsys serial/current | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_nsys.nsys-rep` | 10945516 | `be8135ef97bb07687e22dcc3a2e0ea25d1b291cc297ba77880fd54a7548ce71e` |
|  | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_nsys.sqlite` | 28028928 | `f3712591b173fcfe88bf68c94362059a35e5aae54c317983c6698046dfe35c5f` |
|  | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_nsys_stats_nvtx_sum.csv` | 2438 | `2ae7b4fcae3b4a8e016c2430456319b57f108c9d30faee8a57234e93b87613df` |
| Nsys MPI2/current | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_mpi2_minres_basecheb_vtkoff_s120_20260705_current_nsys.nsys-rep` | 29064880 | `ea00b456315b2bb9305a7c897bd0b490606fafab5f69f4888eb5a3344a2ee7cc` |
|  | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_mpi2_minres_basecheb_vtkoff_s120_20260705_current_nsys.sqlite` | 74014720 | `d0f83141460e680426a44bd7eda46e7d06ab261085c8e1fd118eb9d3ee677dd9` |
|  | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_mpi2_minres_basecheb_vtkoff_s120_20260705_current_nsys_stats_nvtx_sum.csv` | 2784 | `73182d2d5d3aa5ab54910c2b4873917a40b25694865f77cb6232e84614215609` |
| NCU setup/early | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu.ncu-rep` | 21804805 | `139b67c3f34bfc8ce6c81dfed88ef58c4cf4bd19b4efcd86409786568fe53b9e` |
|  | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu.csv` | 1591139 | `f7d7c426f939a23015a6a3d7d1e77e0bd4fcd64f9994261d92f79854131b645f` |
| NCU Krylov window | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu_skip120.ncu-rep` | 9316384 | `9bb375538365e54fbedf5575bf46a08bf3c59279161c417758dfd15071533e6e` |
|  | `tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu_skip120.csv` | 1449406 | `0753134ad41fb300b14bc2e2084ebc7bed07f76d3cc2e3a31e8051e5e14c0cbd` |

## Retained provenance

The two Nsys SQLite sidecars identify **NVIDIA Nsight Systems 2026.1.3.243** through `META_DATA_EXPORT`. `META_DATA_CAPTURE` retains the application commands below and the event set CUDA/NVTX/OSRT with `RATE_HZ=0` (sampling disabled). The original profiler CLI shell command is **unavailable**: it was not retained, and must not be inferred from the application command.

Serial application command:

```text
./build-gpu-sm120/examples/supersonic_example_w_igr/supersonic_cylinder_w_igr --device cuda --dg-backend dg -o 2 -rs 0 -rp 1 -is L2 -ig BrokenL2 -ca 1 -tf 1.0e-3 -dt 1.0e-4 -c 0.5 --igr-sigma-cadence every-stage --igr-stage-rtol-scale 1 --igr-lin-rtol 1.0e-5 --igr-lin-atol 0.0 --igr-linear-solver minres --igr-preconditioner chebyshev --igr-exact-dg-face -no-vtk -no-glvis -vs 1000000 -iprint 0 -nan-scan -adm-scan
```

MPI2 application command:

```text
mpiexec -n 2 ./build-gpu-sm120/examples/supersonic_example_w_igr/supersonic_cylinder_w_igr --device cuda --dg-backend dg -o 2 -rs 0 -rp 1 -is L2 -ig BrokenL2 -ca 1 -tf 1.0e-3 -dt 1.0e-4 -c 0.5 --igr-sigma-cadence every-stage --igr-stage-rtol-scale 1 --igr-lin-rtol 1.0e-5 --igr-lin-atol 0.0 --igr-linear-solver minres --igr-preconditioner chebyshev --igr-exact-dg-face -no-vtk -no-glvis -vs 1000000 -iprint 0 -nan-scan -adm-scan
```

The NCU reports identify **Nsight Compute Target 2026.2.0.0 build 37790515** and retain their full profiler CLI commands. Setup/early uses `--launch-count 80`; the Krylov window uses `--launch-skip 120 --launch-count 80`.

```text
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cuda-13.3.0-a3bneobuiznprbss2rgxubwpdg7yg53v/nsight-compute-2026.2.0/target/linux-desktop-glibc_2_11_3-x64/ncu --target-processes application-only --set basic --kernel-name-base demangled --launch-count 80 --kill yes --force-overwrite --export tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu --log-file tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu.log ./build-gpu-sm120/examples/supersonic_example_w_igr/supersonic_cylinder_w_igr --device cuda --dg-backend dg -o 2 -rs 0 -rp 1 -is L2 -ig BrokenL2 -ca 1 -tf 1.0e-3 -dt 1.0e-4 -c 0.5 --igr-sigma-cadence every-stage --igr-stage-rtol-scale 1 --igr-lin-rtol 1.0e-5 --igr-lin-atol 0.0 --igr-linear-solver minres --igr-preconditioner chebyshev --igr-exact-dg-face -no-vtk -no-glvis -vs 1000000 -iprint 0 -nan-scan -adm-scan
```

```text
/home/jarias9/spack/opt/spack/linux-x86_64_v4/cuda-13.3.0-a3bneobuiznprbss2rgxubwpdg7yg53v/nsight-compute-2026.2.0/target/linux-desktop-glibc_2_11_3-x64/ncu --target-processes application-only --set basic --kernel-name-base demangled --launch-skip 120 --launch-count 80 --kill yes --force-overwrite --export tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu_skip120 --log-file tests/perf/nsight/miniapp_cylinder_l2_brokenl2_exact_face_serial_minres_basecheb_vtkoff_s120_20260705_current_ncu_skip120.log ./build-gpu-sm120/examples/supersonic_example_w_igr/supersonic_cylinder_w_igr --device cuda --dg-backend dg -o 2 -rs 0 -rp 1 -is L2 -ig BrokenL2 -ca 1 -tf 1.0e-3 -dt 1.0e-4 -c 0.5 --igr-sigma-cadence every-stage --igr-stage-rtol-scale 1 --igr-lin-rtol 1.0e-5 --igr-lin-atol 0.0 --igr-linear-solver minres --igr-preconditioner chebyshev --igr-exact-dg-face -no-vtk -no-glvis -vs 1000000 -iprint 0 -nan-scan -adm-scan
```

The protected WIP records that the matching Nsys CUDA-kernel and CUDA-memory CSVs were empty. Consequently, only the matching NVTX summary CSVs above are canonical Nsys summaries; no kernel or memory CSV is selected as evidence.

## Identity limits and observed context

The evidence root observed at manifest time was protected checkout `wip/gpu-post-merge-2026-07-18`, HEAD `1fdb07a8683e33ac0e0ed70dec037c8a0564d0d6`, tree `cb41e33cfa81e1f9c87697fb57371b425549832b`, with root status only `? tpl/mfem`. Its WIP MFEM gitlink was `d9d6526` (v4.9); the current submodule had only untracked `install-cpu`. This is observed checkout context, not asserted capture-time identity: the reports do not bind capture-time executable, MFEM, Git revision, or dirty identity, so those facts are unavailable.

The safe selected embedded environment facts are `OMP_NUM_THREADS=1`, `DG_IGR_EXPECT_CUDA_ARCH=120`, `DG_IGR_IGR_CHEB_ORDER=3`, `DG_IGR_IGR_EXACT_FACE_TIMING=1`, `DG_IGR_IGR_PARTIAL_ASSEMBLY=1`, `DG_IGR_IGR_DEVICE_QF=1`, and `DG_IGR_DEBUG_QF=1`; the wholesale embedded environment is intentionally not copied. Nsys `TARGET_INFO_GPU` records RTX 5080, compute capability 12.0, 84 SMs, and 17094475776 bytes.
