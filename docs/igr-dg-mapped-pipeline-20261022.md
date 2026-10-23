**Mapped DG dense Euler row execution — measured short result**

On the 512² Q2-mapped p5 DG case, whole-application wall time fell from 284.624 to 177.802 s (1.601×). Pre-final-output Analysis elapsed fell from 125.340 to 22.709 s (5.519×). Complete saved conserved fields differ by at most 3.2426e-14 in relative L2 at t=.0025. This qualifies the short execution change; full-time physical accuracy remains to be compared with a converged reference.

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

Recommended next run: after same-binary Q2-face off/on qualification, use a new helper to run the fixed 50/2 candidate fresh to t=.1 first, then a converged FP64 two-level element/Q1 cooperative-CG reference on the same new binary, Euler policy, mesh and GPU. Its different preconditioner must be explicit: this compares end-to-end physical accuracy, not isolated stopping-rule performance. Earlier t=.0025 coarse/local-P reference flow agreement was 1.2254e-13; late-time results remain unqualified. Keep scans off, no restart/retry, outputs every .02, all three timing scopes and current-A certificates. Proposed caps are 3,600 s candidate and 12,000 s reference plus 600 s postprocessing (4h30 allocation); evolving costs remain unmeasured. Generalize output/sweep accounting and row/face audits, compare complete fields at common saved times, and report sigma separately from the conserved-flow budget. No full runner was implemented or submitted here.
