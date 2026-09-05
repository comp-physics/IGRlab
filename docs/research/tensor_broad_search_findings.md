# Broader DG + instantaneous IGR search: measured findings

The strongest complete-application candidate is **p4 DGSEM, H1 closure, SSPRK(10,4), fused ordinary FP64 CUDA, and a retained FP64 inverse used to precondition the current closure equation**. On the RTX 5080 it reaches physical time 0.4 in **0.835 s median**, versus **0.985 s** for calibrated ordinary CG on the same discretization: **1.18×**, including startup, setup and output. Native tensor instructions, lower precision and stationary correction add no established material gain to this candidate. These are smooth-wave, single-GPU results; they do not establish a shock solver or a scalable global inverse.

This extends [the initial precision investigation](tensor_precision_findings.md). The companion [results JSON](tensor_broad_search_results.json) retains commands, environments, source/binary identities, complete run summaries, comparisons, failures, profiling and test evidence. Screenings are single observations unless repeats are stated. No continuum model was adopted or changed as a production default.

The experiment uses the existing full-energy Euler active IGR wave on the periodic unit square. Closure is solved from the current state at every RHS unless explicitly testing prediction. Across physical-solution accuracy comparisons, **alpha = C_alpha h² = 1/36** is held fixed: N6, N12, N16, N18 and N24 use C_alpha = 1, 4, 64/9, 9 and 16, respectively. H1 versus SIPG L2 changes the spatial closure discretization. An L2 Gauss-to-Lobatto basis change, exact nodal interpolation and residual fusion are algebraic transformations. Time integrators and closure extrapolation change the temporal method. A retained first-operator inverse changes the preconditioner, while the equation and acceptance certificate use the current operator and RHS.

We chose a physical accuracy tier of **max conservative-component absolute L2 error ≤ 4e-6 and sigma relative L2 error ≤ 2e-4**, at t=0.4. This is a research comparison tier, not a universal accuracy requirement. The FP64 reference is p4/H1/N24, DOPRI5 dt=0.001, CG tolerance 1e-10. Comparison reconstructs the saved tensor polynomials and integrates on the union of mesh boundaries with five Gauss points per direction; it does not compare different meshes point by point. Reference N12→N24 differences are 1.46e-7 in conservative L2 and 9.72e-6 in relative sigma L2. Halving the N12 reference timestep changes conservative L2 by 9.48e-15 and relative sigma L2 by 3.54e-13; the comparison also changes optional diagnostics, previously checked for exact field parity. These are finite-reference errors, not continuum-exact certification.

| Accurate t=0.4 candidate | Complete seconds, median [range], 3 runs | Max U L2 | Sigma relative L2 |
|---|---:|---:|---:|
| p4/H1/N6, CG/Chebyshev | 0.985 [0.983, 1.968] | 3.362e-6 | 1.681e-4 |
| Same, FP64 inverse + FGMRES | **0.835 [0.830, 0.886]** | 3.361e-6 | 1.680e-4 |
| Same inverse, four stationary sweeps + fallback | 0.826 [0.814, 0.836] | 3.361e-6 | 1.680e-4 |
| Same FP64 inverse, native FP64 WMMA residual | 0.836 [0.832, 0.841] | 3.361e-6 | 1.680e-4 |
| Same, FP32 inverse + FGMRES | 0.899 [0.889, 0.907] | 3.361e-6 | 1.680e-4 |
| Same, TF32 inverse + FGMRES | 0.911 [0.901, 0.919] | 3.361e-6 | 1.680e-4 |
| p3/H1/N18, FP64 inverse + FGMRES | 1.723 [1.677, 1.745] | 6.513e-7 | 1.485e-4 |

All rows reach the horizon without rejected steps. The p4 rows use dt=0.04: ten steps, 100 RHS evaluations and 101 closure solves including final output. CG was calibrated over Jacobi and Chebyshev tolerances, degrees and eigenvalue-estimation work; the selected settings are relative tolerance 1e-4, Chebyshev degree 4, three power iterations. Inverse rows use FGMRES 1e-6. The first CG timing outlier is retained. Stationary correction accepts 56 of 101 trials and falls back 45 times; its 588 reported iterations include 404 stationary sweeps and 184 FGMRES iterations. Its roughly 1% median advantage is unresolved at the observed variability. FP64 and WMMA produce identical final VTU bytes in this cohort.

The N18 control uses dt=0.02 and 200 RHS evaluations; tuned CG there costs 2.008 s in one screening. N16 misses the sigma target (2.207e-4); halving its timestep and tightening the linear tolerance leaves 2.254e-4, identifying spatial error. N18 dt=0.025 fails before the horizon. Thus p4/N6 is about twice as fast as the strongest measured qualifying p3 control, although their error components differ within the common tier. N24/p3 also qualifies but is overresolved and is not the speedup denominator.

The broader search resolved several distinct hypotheses:

| Hypothesis | Cheapest useful evidence and decision |
|---|---|
| Spend more work per timestep to reduce closure calls | At p3/N6 and the earlier strict same-mesh temporal gate, DOPRI5 takes 42 RHS calls versus SSPRK(10,4)'s 100 through t=0.08; RK4 and a fifth-order finite-sweep Radau construction also pass at tested steps. At p4 and t=0.4, SSPRK(10,4)'s stability permits dt=0.04 and wins. DOPRI5 dt=0.016 looks successful at t=0.08 but fails at t=0.4; dt=0.008/0.01 succeeds. SSP dt=0.4/9 reaches t=0.4 but misses the error tier; dt=0.05 fails. Short-horizon speed alone gives the wrong ranking. |
| Predict expensive closure between accepted endpoints | Degree 1/2/3 extrapolation was tested against a matching every-stage reference, including the cadence-dependent face wave speed. Degree 3 at dt=0.001 passes the earlier strict gate but costs 5.237 s through t=0.08. It loses to current-state higher-order integration. No production lagging was adopted. |
| More local inverse work plus a conforming coarse solve | Enriching the coarse space Q1→Q2→Q3 reduces p3/N6 FGMRES iterations 5101→3548→1620 and wall time 7.281→4.514→2.195 s. The global retained inverse remains faster (~1.5 s in that screening). Both dense and sparse coarse solves were tested; stop developing this additive construction locally. |
| Move redundant global setup to the GPU | cuSolver pivoted FP64 LU and inverse construction extend the explicit research cap from 2048 to 4096 unknowns. At 576 unknowns, total setup is ~0.124 s, including ~0.054 s operator sampling and ~0.068 s inversion. For the p4/H1/N12 2304-unknown screening, setup is ~0.371 s. The sampling, validation, conversion and upload are charged. The inverse is retained only as a preconditioner. |
| Fuse physics, contraction, face lift and inverse mass | The integrated FP64 kernel computes pressure/flux on chip, reuses coefficients, applies the precombined central operator and unique-face lift, then diagonal mass. An exact nodal gather removes eligible sigma interpolation. The unique face-flux kernel and global reductions remain. CUDA parity checks cover central/LLF, orders 1–4, face orientation, floor behavior, SIMT/WMMA and gather paths. |
| Change to coupled implicit HDG | Existing full-application RODAS3 p3/N6 tensor and ordinary engines reach t=0.08 at dt=0.004 in 2.814 and 17.773 s, with maximum absolute saved conservative-component differences below 7e-15. This **6.32× engine-package ratio** includes different trace preconditioners; it is not an isolated MMA speedup. Both engines fail at dt=0.008. Halving the tensor timestep confirms a spatial-error plateau; sigma relative error ~3.43e-3 is much worse than p4 DGSEM. Previous tiny-step BE ablations were not full RODAS time-to-solution evidence. |

The Radau probe performs four simultaneous three-stage Picard sweeps after an Euler predictor: 13 RHS calls. Its independent stages suggest possible batching, but this implementation executes them serially. It is a fixed explicit iteration with tested fifth-order convergence, **not a converged implicit Radau method with its stability guarantees**. DOPRI5 here is its six-call principal fixed-step update, without an adaptive/embedded or FSAL claim. Research integrators require fixed positive dt, disabled CFL control, every-stage IGR and no limiter pipeline.

Native FP64 matrix execution was verified in the application, not inferred from source syntax. SM120 SASS contains `DMMA.8x8x4`; the specific counter `sm__inst_executed_pipe_fp64_op_dmma.sum` measures **9216 warp instructions per launch**, versus zero for SIMT, at p3/N48. The generic tensor-pipe counter is zero for both in these measurements. Two profiled residual launches give median **39.872 us WMMA versus 35.936 us SIMT**. Useful central work is 2,359,296 FLOPs; issued matrix work is 4,718,592 FLOPs. These kernels also include physics/lift/mass, so these counts are not whole-kernel FLOPs or a dense hardware peak measurement.

To remove the low-order padding objection, the test-only volume benchmark was extended through p7 and p15 (4096 elements, 100 timed calls, prescribed sigma, long-double same-operator reference). Fully aligned FP64 WMMA is still slower than a strong FP64 matrix-form SIMT control:

| Nodes per direction | FP64 SIMT / WMMA, us | FP32 SIMT, us; relative L2 error | Compensated TF32 / FP16, us |
|---|---:|---:|---:|
| 8 (p7) | **45.912 / 49.693** | 19.543; 1.29e-7 | 56.033 / 72.483 |
| 16 (p15) | **306.494 / 341.212** | 97.307; 2.00e-7 | 148.141 / 167.448 |

FP64 errors are approximately 1.53e-16 and 2.17e-16 for both implementations. WMMA has no matrix padding at these orders but uses three times the declared shared payload. Fused compensated tensor variants issue 12×/3× useful arithmetic at N8/N16 and lose to ordinary FP32 in both time and relative L2 error. Fully dense 2D and packed factorized variants also lose. Setup, warmup, scratch, conversions and pack/multiply/unpack timing are recorded; reconstructed setup-plus-three-call-warmup-plus-100-timed-call sums preserve the ranking. They exclude separate process initialization and host harness overhead and are not standalone process timings. This is a volume-only mechanism probe, not a higher-order application speedup claim.

Whole-process timings include initial/final VTK, current coefficient construction, all closure/face/transport/time work and fallback. Earlier screenings enabled optional host diagnostic scans. Four otherwise identical p4 runs give bitwise-identical U and sigma with scans disabled; one DOPRI case drops from 2.566 to 1.308 s. The final comparison disables them consistently, matching the application's defaults. Disabling `nan-scan` also disables its accepted-step early nonfinite stop/counters, so failure-status timing can change; final finite-state/closure certification and q_sigma admission remain. This configuration cost reduction is not attributed to tensors or a numerical method. Large first-process timings were also observed (10.606 s for the initial build's first application and 9.686 s for the first cuSolver application). Final repeats are fresh processes with warm system/JIT caches; no compilation overlapped them. Clocks were not locked.

Deployment inference requires different hardware measurements. [NVIDIA's RTX Blackwell table](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) lists RTX 5080 dense TF32 at 56.3 TF/s and dense FP16/BF16 with FP32 accumulation at 112.6 TF/s, versus ordinary FP32 56.3 TF/s. [H100](https://www.nvidia.com/en-us/data-center/h100/) and [H200](https://www.nvidia.com/en-us/data-center/h200/) SXM list FP64 34 and FP64 tensor 67 TF/s; removing the advertised sparsity factor gives dense TF32 494.5 and FP16/BF16 989.5 TF/s. Their low-precision opportunity is much larger; p3's 2× FP64 padding can consume Hopper's roughly 2× FP64 peak advantage. [NVIDIA's B200 table](https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf) gives dense TF32 1.1 PF/s and FP16/BF16 2.2 PF/s. None was measured here; these are arithmetic ceilings, not application speedup estimates.

The next unresolved question is whether a **scalable batched element/coarse closure with current-state correction** can exploit those deployment GPUs enough to beat calibrated CUDA. The present global inverse has O(n²) storage, O(n³) factorization, n host-synchronized operator probes, a single-RHS application and a single-rank 4096-unknown cap. Its small-domain success does not answer that question. The new evidence argues against further native FP64 transport tuning on this RTX 5080, while preserving mixed-precision and stage-batching questions for hardware with substantially different dense arithmetic ratios.

Useful implementations remain opt-in in the existing solver: GPU inverse setup, fused residual/WMMA, stationary correction, enriched two-level preconditioning, LGL closure basis, research time integrators and closure prediction. Exact nodal gathering is selected automatically only for an exact signed node permutation, with dense fallback. CUDA now links cuSolver; non-RK3 output names include the integrator. No remote resources, publishing or production continuum changes were used.

**Reproduce.** Run from this checkout, with Spack initialized and an exclusive local GPU lease. The authoritative built application for all final timing cohorts is commit `06c3f0d1cd7393c5fa3bfdcb5b518842cbb22731`, tree `01ed9f0b67aa4567a563b3735b3292501d89b885`, SHA256 `14de6d311a025f65cd6754ffed63bf259fd9e7b89a6f9803bb59bc870b17fb45`. Probe/runner source is `377cddc170c978c1829fa0c72f987abca41ffc9e`, tree `cd208317e84b1aca0f51d770043bae525028a2ec`. The final inactive-option parse fix is numerically neutral and separately checked below. Live checkout identities in run records may include concurrent test/runner edits; the recorded binary-to-built-source map resolves them. MFEM pin: `d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`; Release, CUDA 13.3, app SM120/MFEM SM80 virtual, HDG runtime on, LAPACK off, MPI rank 1, OMP threads 1, RTX 5080 16 GiB, driver 610.43.02.

```bash
source /home/jarias9/spack/share/spack/setup-env.sh
./build.sh --gpu --cuda-arch 120 --hdg-runtime --no-clean --no-deps --no-tests -j 16
# Set B to the app build directory printed by build.sh.
APP="$B/examples/dgsem_smooth_advection_periodic/dgsem_smooth_advection_periodic"
flock -n /tmp/dg-igr-gpu0.lock python3 scripts/perf/dgsem_precision_horizon.py \
  --executable "$APP" --output-dir /tmp/broad-fp64 \
  --order 4 --sigma-space H1 --integrator ssprk104 --dt .04 --final-time .4 \
  --fused-residual --diagnostic-scans off --dense-inverse fp64 \
  --dense-setup cusolver --linear-solver fgmres --linear-rtol 1e-6
# CG control: omit the dense/FGMRES options; use --linear-rtol 1e-4
# --cheb-order 4 --cheb-power-iterations 3, with a fresh output directory.
# Fine reference: order 4, H1, --refine 2 --c-alpha 16, dopri5,
# dt .001, final-time .4, CG tolerance 1e-10, scans off, timeout 120.
python3 scripts/perf/compare_dgsem_physical_fields.py REF.pvd CAND.pvd \
  --reference-order 4 --candidate-order 4 \
  --reference-sigma-space H1 --candidate-sigma-space H1 \
  --reference-c-alpha 16 --candidate-c-alpha 1 --output /tmp/physical-error.json
python3 scripts/perf/generate_dgsem_periodic_mesh.py --n 18 \
  --output /tmp/periodic-n18.mesh \
  --verify-n6 examples/dgsem_smooth_advection_periodic/smooth_advection_periodic_n6.mesh
flock -n /tmp/dg-igr-gpu0.lock \
  "$B/tests/hdg/runtime/cuda/dg_igr_dgsem_tensor_volume_benchmark" \
  --elements 4096 --iterations 100
flock -n /tmp/dg-igr-gpu0.lock ./build.sh --gpu --cuda-arch 120 --hdg-runtime \
  --no-clean --no-deps -j 16 --test-regex \
  '^dg_igr_(explicit_ssprk_(cpu|cuda)|research_rk_cpu|igr_extrapolation_history|dgsem_operator_(tests|cuda)|dgsem_igr_tests|igr_(stationary_correction|two_level_inverse|dense_inverse)_cuda)$'
```

Final source `67bbd82f47d3a0225ac041fc120fc6be9f025b14` (tree `0ea08736ef19a08d62279469eda35f8f1adf70c4`) passes **10/10** focused CPU/CUDA tests. Its application SHA256 is `270a7764a70298f53740b6a8f955858b3cdc23a6ff245542725777db500520a8`. Three final-source winner runs reproduce the primary cohort’s final VTU bytes exactly; their separate times are 1.759, 0.876, 0.857 s. They are retained separately from the rotated timing cohort. An earlier RK observed-order test failed because its coarse-step error-cancellation window was unsuitable; the test ODE was replaced with a nonautonomous case and tighter asymptotic-order checks, then passed. Integrator coefficients were unchanged by that test correction. Independent reviews covered numerical guards, fallback, source identities, physical comparisons and performance attribution; all reported issues were resolved or documented.
