**Mapped DG spectral-block hybrid — measured mapped Sedov results**

On this 512² Q2-mapped p5 DG Sedov case to t=.1, the fixed 50/2 hybrid completed in **760.120 s** versus **2,224.272 s** for the converged FP64 reference (**2.926×**). Maximum final conserved relative L2 difference is **8.2067e-7**, within the 1e-3 engineering comparison budget; sigma relative L2 differs by **1.529%**. The reference uses a different preconditioner, so this is an end-to-end method comparison. The earlier execution-only pairs are retained separately below.

**Row execution — matched short pair**

On the 512² Q2-mapped p5 DG case, whole-application wall time fell from 284.624 to 177.802 s (1.601×). Pre-final-output Analysis elapsed fell from 125.340 to 22.709 s (5.519×). Complete saved conserved fields differ by at most 3.2426e-14 in relative L2 at t=.0025. This qualifies the short execution change; the separate complete t=.1 comparison is reported below.

Both fresh runs retain SSPRK3/CFL=.3, C_alpha=1/6/max-h, exact face-coupled FP64 residuals, actual GL7 mass/GL6 diffusion, and the same warm-started element-block iteration: 50 initial corrections, then two at omega=.7. Euler pointwise-flux/dense-row kernels use FP64 SIMT. The unchanged per-element spectral local inverse uses FP64 WMMA tensor cores. Scans, limiters and modal damping remain off.

The fix packs each original physical weak-divergence matrix by columns and assigns a thread to each output coefficient. It preserves directional subtotals, node-addition order, actual dense inverse mass and current sigma interpolation. Original element execution remains selectable. Default rows require ordinary CUDA, disjoint L2 output ownership and the packed uniform layout.

Baseline profile 3439836 identified weak divergence (145.783 ms/call) and inverse mass (72.677 ms/call): their recorded spans cover 87.102% of the 15.0048 s capture after clipping its endpoint. Eighteen interior RK steps have no launch holes. The primary process reports a final CUPTI 49/50-buffer warning; driver 13.2 uses Nsight 2025.3.1/CUPTI 12.9 fallback. This supports bottleneck ordering without certifying complete tracing or hardware utilization. Much of the old host assembly span waits for previously queued Euler kernels.

Matched job 3439969 ran baseline then candidate on gh074, GH200 UUID `GPU-b7de2193-6349-6d24-39b1-464c7bef6ca1`, with one GPU, eight CPUs, 96G host memory and OMP1. Both accepted 162 steps without rejection, made 487 sigma calls (486 production plus one output-only), and executed 1,022 sweeps. Both remain uncertified fixed-sweep approximations.

| Timing scope, seconds | Baseline | Rows | Baseline / rows |
| --- | ---: | ---: | ---: |
| Pre-final-output Analysis elapsed | 125.339973 | 22.709105 | 5.519× |
| Native `done` stopwatch | 168.817911 | 65.360092 | 2.583× |
| Whole application process wall | 284.624003 | 177.801615 | 1.601× |

The native stopwatch starts after initial VTK output and includes final output/reporting. Process wall also includes setup, initial output and teardown. This is one sequential short observation with unlocked clocks. One-Hz total-device sampled peaks rose from 29,099 to 29,677 MiB (+578 MiB). Source-derived flux scratch is 576 MiB; samples can miss true maxima and do not isolate that allocation.

Baseline source `edba7971aacfa33b8886c94d67f38c6cbd9d6c23` uses app SHA256 `38e112c45baf6bfe1715b3710a3ecbc6109a11f4997a5f4c0a903be978f50a28`. Candidate source `6ae6930f0d3c5b1c46fedb68b9a4bb626864631f` uses `fdd7b9e152598f7a9e561fe88510b97644e2cfbafd2b096d70c0d51a53bb6021`. Canonical SM90/NVTX native job 3439953 passed all six focused CPU/CUDA suites. All 777 source hashes independently match frozen 6ae6930. Later Q2-face changes require fresh qualification.

Both complete final VTUs were hashed after timing and their transferred hashes independently rechecked. All 262,144 saved Q5 cells were compared using GL8×8 on matching verified Q2 geometry. Maximum conserved relative L2 is 3.24258540295972e-14; sigma relative L2 is 1.0824401928625688e-12. Conserved engineering budgets pass; sampled density, pressure and q_sigma are positive. These are differences between executions of the same approximate closure, not sigma accuracy against a converged solution. Polynomial squared norms are exact on the fitted Q2 map up to roundoff; Linf and positivity remain sampled.

The [compact measurements](benchmarks/igr-dg-mapped-pipeline-20261022.json) bind native, run, field and profile receipts/hashes. The earlier short coarse control used a different two-level element/Q1 cooperative-CG preconditioner; it was not the same algorithm as this execution baseline.

**Q2 tensor faces — separate matched short pair**

Canonical native job 3440081 passed all six focused CPU/CUDA suites on source `b53b40a56facbd4a47b198239158764da82b6295`, app SHA256 `007acaade17c5ec6e9adfc7e3ebe796da8406a28ad606041da77888e4f0986c2`. All 777 frozen source hashes, native commands/logs and the manifest were independently checked. The physical candidate logs Q2 geometry with sampled Jacobian validation, FP64 state/nonlinear flux, an element-owned lift and zero residual atomics. It uses six reference maps, three with sparse tangential maps, and no dense face-shape entries. Both arms retain FP64 SIMT row Euler and FP64 WMMA local P.

Job 3440099 ran face-off then FP64-face-on using this same binary on gh093, UUID `GPU-2a5f1b42-1214-cc5c-6e6e-132df93018af`. Only the face selector aliases and output prefixes differ. Both fresh runs reached t=.0025 with 162 accepted steps, 487 sigma calls and 1,022 sweeps, without rejection or fallback.

| Timing scope, seconds | Faces off | FP64 Q2 faces | Off / on |
| --- | ---: | ---: | ---: |
| Pre-final-output Analysis elapsed | 22.632057 | 16.976771 | 1.333× |
| Native `done` stopwatch | 65.435308 | 59.442664 | 1.101× |
| Whole application process wall | 177.584944 | 169.768591 | 1.046× |

One-Hz total-device sampled peaks fell from 29,677 to 27,649 MiB, a 2,028 MiB reduction; this is not an exact peak allocation measurement. The same VTK policy and the timing definitions above apply. This is one paired observation with unlocked clocks. The earlier row pair used a different GPU: its 5.519× and this pair's 1.333× ratios must not be composed into a directly measured cumulative speedup.

Both complete final VTU hashes were independently rechecked against the run/transfer/comparison receipts. The GL8×8 comparison on all matching Q2/Q5 cells gives maximum conserved relative L2 **4.063260876695138e-14** and sigma relative L2 **1.1116402531354572e-12**; sampled density, pressure and q_sigma are positive. This passes the 1e-10 conserved execution-equivalence gate. This short face test reports execution equivalence; the separate complete comparison below measures final flow agreement with the converged reference.

**Complete mapped Sedov comparison to t=.1**

Job **3440138** completed candidate first, then reference, on gh010, GPU UUID `GPU-68c936db-7ee6-cb65-ab25-aecc4c8fde9d`. Both use the same tested b53b40a source/app SHA above, mapped mesh, FP64 row Euler/Q2 faces and output schedule at 0,.02,.04,.06,.08,.1. Both accepted 6,663 steps without rejection and made 19,990 sigma calls.

The hybrid uses warm-started projected element blocks with FP64 WMMA, 50 initial/two subsequent corrections at omega=.7: **40,028 corrections**, with no convergence certificate. The reference uses **FP64 FGMRES**, RHS-relative tolerance 1e-6, and retained initial element blocks plus a Q1 cooperative FP64-CG two-level preconditioner. All **19,990** current-A certificates pass; maximum relative residual is 9.9999306e-7 and the final residual is 9.4663387611e-7. Its 330,526 iterations and different preconditioner make the measured speedup an end-to-end comparison, not an isolated stopping-rule result.

| Timing scope, seconds | Hybrid | Converged reference | Reference / hybrid |
| --- | ---: | ---: | ---: |
| Pre-final-output Analysis elapsed | 603.587655 | 2,056.821112 | 3.408× |
| Native `done` stopwatch | 646.740361 | 2,101.535439 | 3.249× |
| Whole application process wall | 760.120074 | 2,224.272248 | 2.926× |

Pre-final elapsed **includes intermediate VTK saves** and analysis work; it is not pure evolution. The native stopwatch additionally includes final output/reporting, while process wall also includes setup and initial output. This is one sequential observation with unlocked clocks. One-Hz sampled total-device peaks were 27,649 MiB for the hybrid and 34,839 MiB for the reference; these samples can miss true peaks.

Complete final Q5 polynomials on all 262,144 matching Q2 cells were compared with GL8×8. Relative L2 differences are 1.7330307356e-7 for density, 8.2066961524e-7 maximum over momentum components, and 2.6156257072e-7 for energy; derived pressure is 2.3239014153e-7. Maximum conserved sampled relative Linf is 1.9605446561e-6. Every conserved component passes the **1e-3 relative L2 / 1e-2 sampled relative Linf engineering budgets**. These thresholds are not discretization-error bounds.

Sigma differs by **0.0152884680954 relative L2 (1.52885%)** and **0.0253437547007 sampled relative Linf (2.53438%)**. It is reported separately from the conserved-flow gate; the hybrid remains an uncertified approximate closure. At final GL8 samples the hybrid minima are density 0.4494201726, pressure 1.0003187640 and q_sigma 1.4004462696, all positive. This establishes neither positivity between samples/times nor accuracy on untested geometries. Scope is this recorded mapped Sedov configuration at the final saved t=.1 frame.

The benchmark JSON binds the unchanged run receipt (`5c45e27a30c714bb67abd242e8f7c3221643903d7283f6182555fe056e5b97a2`), full-field comparison (`b2108492a98d0ad8751b9a721dfac096544d45579d375acc4fcc2f1e8116ecb7`), native/source identities, commands/logs and both raw final VTU hashes. The complete VTU hashes were independently rechecked after transfer. Full results are complete for this comparison.
