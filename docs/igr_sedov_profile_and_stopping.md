# Large Sedov profiling and IGR stopping

RHS-relative CUDA FGMRES stopping halved Krylov work on the fixed large Sedov
case and reduced application time by 14.8% against a matched scan-off baseline.
The maximum final conservative-field relative L2 difference was 1.97e-9.
Both methods passed every original-current-FP64 residual certificate at the
same requested relative tolerance, 1e-6. This is a successful bounded comparison
on one case, not a repeated benchmark distribution or a continuum-error study.

## Fixed case and source

The case is ordinary DG with Gauss–Legendre nodes, p5, 512 by 512 affine
Cartesian squares, L2/BrokenL2 spaces, and natural scalar IGR boundaries.
It has 262,144 elements, 9,437,184 scalar IGR DOFs, and 37,748,736 conserved
coefficients. Both runs advance from t=0 to 0.1 using RK3 SSP, CFL=0.3,
every-stage IGR, and binary VTK output every 0.02. Positivity limiting, modal
damping, nonfinite scans, and optional admissibility scans are off.

Physical alpha is h_max²/(p+1) = 6.357828776041666e-7, supplied through the
existing CLI as C_alpha=1/6 and max-length policy. The initial Gaussian width
remains 0.1875. Background density and pressure are one, deposited energy is
one, and gamma=1.4. This is the existing finite-width, finite-pressure Sedov
example, not a singular point explosion in a cold background.

The reusable spectral inverse uses four explicit TF32 cuBLAS GEMMs per
application. Euler uses FP32 tensor-line contractions. Conserved state,
nonlinear fluxes, current IGR equation, outer FGMRES, and residual acceptance
remain FP64. Tensor volume is off; exact SIPG faces remain enabled.

Implementation commits are `e65b912` (optional RHS-relative stopping) and
`3e8a6db` (diagnostic scan policy). The frozen executable SHA256 is
`334fabb78ea6ccbc194062a461ff681dc7471f770a95c8ae33819846c472fa3c`.
The retained build receipt identifies all 770 tested source files; all seven
changed source/test files match the committed changes. DeltaAI build 3406084
used canonical `build.sh`; `dg_igr_time_defaults_tests` and
`dg_igr_device_krylov_cuda` passed in 2.91 and 1.63 seconds respectively.

## Diagnostic scans and stopping semantics

The application defaults already disabled both optional scan types. Earlier
experiment launchers explicitly enabled `--nan-scan`; that was an experiment
configuration mistake. The launchers now explicitly use `--no-nan-scan` and
`--no-admissibility-scan`. Final-state and failure-path diagnostic scans that
bypassed the option now honor it. Disabled diagnostics do not invent state
finiteness evidence or trigger a failure snapshot. Scalar failure detection,
intrinsic limiter/input validation, and algebraic residual acceptance remain.

The previous recurrence criterion was
`max(atol, rtol * ||r0||)`, while the true residual gate used
`max(atol, rtol * ||b||)`. Near t=0.073, warm starts had
||r0||/||b|| about 2.34e-4–4.75e-4. The recurrence consequently requested
roughly 2,100–4,300 times tighter accuracy than the RHS gate.

`IGRLAB_IGR_KRYLOV_RESIDUAL_SCALE=rhs` uses the RHS norm for the recurrence too.
The default remains `initial`. FP64 true-residual recomputation, independent
original-operator certification, restart/breakdown handling, iteration limits,
and residual-replacement handling are preserved. The observed improvement
comes from fewer iterations, not newly accepting zero-iteration warm starts.

## Matched unprofiled result

The argument lists are identical. Recorded application environment differences
are only the stopping selector and output paths. Both use the same executable,
mesh, runner, build receipt, physical alpha, and scan-off policy.

| Metric | Initial-relative | RHS-relative |
|---|---:|---:|
| Application wall time, including startup/output | 1,539.797 s | 1,311.811 s |
| Accepted / rejected steps | 4,939 / 0 | 4,939 / 0 |
| Converged IGR solves | 14,818 / 14,818 | 14,818 / 14,818 |
| Mean / maximum iterations | 7.064 / 11 | 3.491 / 5 |
| Total iterations / inverse applications | 104,676 | 51,736 |
| TF32 GEMM calls | 418,704 | 206,944 |
| Linear-solve time | 611.416 s | 361.046 s |
| Sampled process GPU memory peak | 28,894 MiB | 28,006 MiB |
| Final FP64 RHS-relative residual | 1.379e-10 | 3.646e-7 |
| Full-process ns / conserved DOF / timestep | 8.259 | 7.036 |

RHS stopping reduced iterations and GEMMs by 50.6%, solve time by 40.9%, and
application time by 14.8% (1.174 times faster). Every certificate passed;
there were no residual-replacement retries or rejected steps. Each method
performed 14,817 production solves and one final output solve.

The reported loop averages before final output are 0.2877 and 0.2414 seconds
per step; these exclude startup and final output and must not be confused
with the full-process normalization in the table. All per-DOF figures use
37,748,736 conserved coefficients as denominator while including IGR work.

Jobs were 3406088_0 (raw Slurm ID 3406211, node gh104) and 3406088_1
(raw ID 3406088, node gh035). Each reserved one GPU, eight CPUs and 96G host
memory on account `bdiy-dtai-gh`, partition `ghx4`, constraint `projects`.
They are single observations on separate scheduled nodes. Component host
timers can move waits between phases when synchronization changes, so their
differences are not independent GPU work measurements.

The historical scan-on r1 run took 2,138.239 seconds. The new initial-relative
scan-off run took 28.0% less time; combining the scan and stopping changes
reduced time by 38.65%. That historical comparison includes the scan-policy
revision and is distinct from the matched 14.8% stopping-only improvement.

## Final physical fields

The two final native VTU files were downloaded and verified against remote
SHA256 hashes. The comparison matches all Cartesian cells and evaluates each
saved Q5 polynomial at 6 by 6 Gauss points: 9,437,184 points per field. GL6
integrates squared Q5 differences exactly up to FP64 roundoff on this mesh.
Linf below is the maximum at those Gauss points, not a continuous supremum.

| Conserved component | Relative L2 difference | Sampled relative Linf |
|---|---:|---:|
| Density | 6.851e-10 | 6.574e-9 |
| x momentum | 1.963e-9 | 9.486e-9 |
| y momentum | 1.964e-9 | 8.190e-9 |
| Energy | 5.383e-10 | 3.872e-9 |

All pass the earlier engineering budgets of L2<=1e-3 and sampled Linf<=1e-2.
Pressure and speed, derived from interpolated conserved fields, differ by
5.286e-10 and 1.436e-9 in relative quadrature L2 norm. Conservative integral
differences are at rounding scale; these do not imply accuracy below FP64
accumulation limits. This checks the final saved state, not unsaved times,
analytic Sedov accuracy, gradient convergence, or general shock robustness.

## Nsight findings

The original scan-on Systems capture contains 15.004 seconds and 37 complete
RK steps near t=0.073. The accepted-state CPU nonfinite scan occupies 4.444
seconds, 29.62% of the window. It contains no CUDA operations. The raw host
access lacks an explicit HostRead; the capture alone does not establish
whether that host buffer was current. RK finite kernels themselves total
only 0.0504 seconds; most of their enclosing host range is waiting for earlier
queued work and must not be counted again as scan computation.

CUDA kernels/copies/memsets cover a union of 10.335 seconds, 68.88% of the
window. This is active timeline coverage, not hardware throughput. The TF32
GEMMs total 0.855 seconds, 5.70% of elapsed time. Euler Rusanov faces, current
IGR diffusion faces, and Euler flux preparation dominate the kernel totals.

The subsequent scan-off Compute capture sampled one evolved RK step using
kernel replay, cache flushing, and unlocked GPU clocks. Representative medians:

| Stored kernel family | Samples | Kernel duration | SM throughput | DRAM throughput | Occupancy |
|---|---:|---:|---:|---:|---:|
| Euler face flux | 2 | 17.60 ms | 15.67% | 31.28% | 47.20% |
| Euler volume flux preparation | 2 | 10.76 ms | 1.57% | 44.27% | 47.39% |
| Current IGR diffusion face | 2 | 1.419 ms | 53.68% | 39.58% | 71.92% |
| Spectral packing | 2 | 0.559 ms | 37.53% | 6.51% | 95.43% |
| TF32 GEMM NN | 4 | 0.212 ms | 68.70% | 28.56% | 17.04% |
| TF32 GEMM NT | 2 | 0.285 ms | 48.08% | 15.79% | 18.01% |

Sampled NN/NT GEMMs show 48–69% tensor-pipe activity. Euler face and flux
kernels have only about 0.04 and 0.02 eligible warps per scheduler cycle,
with long-scoreboard stalls near 189 and 389 cycles per issued instruction.
These support investigating memory dependencies and data access. Occupancy
alone does not establish useful throughput; the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
defines these metrics and replay limitations.

Coverage is explicitly partial: the report and raw CSV contain six of seven
expected visible variants, with no TN GEMM sample. Console progress names
disagree with stored records; an independent audit confirms the CSV parser
and native record identities agree. The cause remains unresolved. Counters
were not relabeled, no algorithm-change claim is made, and no GPU rerun was
needed to analyze the valid samples. Both profiler jobs' failed Slurm statuses
reflect postprocessing checks; their usable captures and native application
outcomes are preserved separately.

The next implementation candidates are coalesced Euler trace/flux access and
removing redundant dense sigma interpolation where the nodal maps coincide.
The current volume-flux kernel assigns one whole element to each thread and
loops over an element-specific sigma interpolation matrix. These are proposed
optimizations informed by source and counters; no new speedup is claimed.

## Artifacts and use

Enable the tested stopping option with
`IGRLAB_IGR_KRYLOV_RESIDUAL_SCALE=rhs`; use
`--no-nan-scan --no-admissibility-scan` for normal performance runs.
The remaining full case arguments and backend selectors are frozen in
`run-sedov-scanoff-rhs2.py` and each run's `run.json`.

Remote artifacts are under
`/projects/bdiy/jarias9/igrlab-deltaai-wmma-20261016/` in
`sedov-profile-20261018/` and `sedov-oversolve-20261018/`.
The local mirror is `/home/jarias9/.cache/igrlab-experiments/` with the same
directory names. It includes original reports/raw counters, analysis scripts,
source/build receipts, native logs, final VTUs and one-frame PVDs,
`final-metrics-review.json`, and `field-comparison-initial-vs-rhs.json`.
No additional GPU jobs are needed to complete this investigation.
