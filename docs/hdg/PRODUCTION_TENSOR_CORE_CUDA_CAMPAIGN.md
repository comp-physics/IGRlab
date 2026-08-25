# Production Tensor-Core CUDA HDG Campaign

## Purpose and immutable evidence handoff

This tracked document fixes the architecture, inherited capability, implemented
numerical policy, qualification map, performance method, and maturity boundaries
for the production tensor-core CUDA engine in the general Euler HDG application.
It is intentionally not a mutable qualification report: the exact candidate
commit cannot name itself, and final qualification, performance, profiling, and
independent review occur only after that commit is frozen.

Final dispositions therefore live in one external, content-addressed promotion
ledger. The ledger names the exact commit and tree containing this document,
hashes every retained build, test, benchmark, profiler, SASS, and review
artifact, and has its own detached SHA-256 identity. It is supplied with the
promotion request. No result is back-filled into this file, and any source
change requires a new candidate and a new ledger.

The campaign extends the production generic HDG path. It does not promote a
frozen oracle, an SC25-specific solver, a benchmark-only implementation, or a
separate IGR application.

## Frozen controls

The historical numerical and performance control is application commit
`30afa08f8ad19a3ad37e627463ef63c40da1ae91`, whose Git tree is
`03313cf6b5b17d05fc3d768a6c77770295ced343`.

The independently reviewed candidate
`55d9a33e689be5a4087d6d3afde84102ea7e4870` has that exact same tree. The
merge commit and reviewed candidate therefore differ in history, not source
content. The frozen control must not be rebased, amended, or replaced by a
later ordinary-CUDA result.

The historical benchmark executable uses one harness-only overlay on that
control: `tests/hdg/runtime/cuda/benchmark_generic_euler_accepted_step.cu`
accepts `--order 1|2|3` so the same frozen numerical application can supply all
six p/mode controls. The overlay changes only benchmark option parsing and
order selection; it does not change the application lifecycle, CUDA numerical
engine, mathematical policy, tolerances, or timed region. The promotion ledger
must retain the exact overlay diff and SHA-256, the frozen build manifest, and
the frozen executable SHA-256. The declared historical application identity
remains the commit and tree above; the executable provenance must also disclose
the harness overlay rather than representing it as a pristine checkout.
The historical schema-1 stream predates typed trace-policy and tensor-work
accounting. Frozen summaries retain their schema-1 shape and omit those fields;
they do not synthesize null policies, false fallback flags, tensor subsections,
or other schema-2 metadata.

Performance has two controls:

1. tensor CUDA versus ordinary CUDA built from the same final candidate tree;
2. final-tree ordinary CUDA versus the frozen commit above.

The first measures the complete final-tree engine package, including the
tensor condensation leaf. In IGR mode both engines use explicitly recorded
BJ-ILU0+RCM with one preconditioner application per callback, but ordinary
CUDA uses direct factor triangular solves and refreshes the factor for every
trace solve, while tensor CUDA uses fixed five-iteration lower and upper
Jacobi solves and reuses the first factor within an attempt. Conventional
tensor CUDA retains its separately recorded diagonal trace policy. Neither
paired engine comparison represents the contribution of MMA instructions
alone; final profiler attribution must quantify that contribution separately,
and both the triangular-solve and factor-lifetime differences remain explicit
in every IGR comparison. The second control detects shared refactoring that
slows ordinary CUDA and would otherwise manufacture a tensor speedup.
Per-sample observed outer iterations, exact factor setup/reuse counts, maximum
aggregate factor age, and derived ILU/Jacobi work bounds expose recurring
solver work without removing it from the accepted-step timing.

### Provisional same-policy attribution

The existing accepted-step campaign also has a bounded
`--comparison-mode same-policy-attribution` ablation. It does not replace
either promotion control. For every conventional/IGR x p1/p2/p3 cell it runs
this exact triplet from one final tree and build:

- `O_base`: ordinary CUDA with `--trace-policy engine-default`;
- `O_common`: ordinary CUDA with `--trace-policy common-fast`;
- `T_common`: tensor CUDA with `--trace-policy common-fast`.

`--trace-policy` is a benchmark-only attribution control. It is not an
application configuration option, environment selector, hidden production
fallback, or new engine. `engine-default` leaves the typed production policies
unchanged. `common-fast` selects diagonal scaling for both conventional
engines. For IGR it selects the same ILU0+RCM, five-lower/five-upper iterative-
Jacobi application, and first-factor-per-attempt lifetime for both engines.
The benchmark emits the selected policy at top level and in every bounded
sample. The driver validates every preconditioner, ILU, triangular-solve,
accounting, and factor-lifetime dimension exactly, then requires `O_common`
and `T_common` metadata to match before computing a ratio.
`engine-default` records `variant=production` and
`production_selectable=true`; `common-fast` records `variant=ablation` and
`production_selectable=false`. The campaign validator rejects either label if
it disagrees with the explicit trace-policy selection, so attribution records
cannot be mistaken for production-engine evidence.

Triplet order rotates `O_base/O_common/T_common`,
`O_common/T_common/O_base`, and `T_common/O_base/O_common`; cell order also
alternates. The triplet count is at least three and divisible by three so each
member occupies each launch position equally. Raw five-step samples and
artifact hashes are retained for every invocation. Paired-bootstrap intervals
are reported for policy gain (`O_base/O_common`), tensor-local gain
(`O_common/T_common`), and the disclosed package check (`O_base/T_common`).
The latter equals the product of the first two within each triplet.

This ablation is deliberately smoke-mode, provisional, and ineligible for
promotion even when an interval excludes one. It requires a clean exact
commit/tree, canonical build identity, live SM120 identity, and the exclusive
GPU selection, but runs neither the frozen control nor the export control.
Its retained data are bounded benchmark samples and aggregate solver metadata,
not callback history or a live event stream. It adds no flight recorder,
replay path, every-event capture, or public diagnostic API. Separate NCU/SASS
evidence is still required to attribute any `T_common` gain to MMA execution.

## Capability baseline at the frozen tree

At the frozen tree, CPU and ordinary CUDA were peers behind the generic Euler
HDG request and the shared application lifecycle. `--device cpu|cuda` selected
the profile; there was no tensor-engine selector. The table below records the
supported surface that the tensor engine inherits.

| Surface | Generic CPU at the frozen tree | Ordinary CUDA at the frozen tree |
| --- | --- | --- |
| Application | General Euler HDG main adapter using shared case resolution, mesh construction, projection, boundary lowering, time planning, transactional output, and diagnostics | The same general Euler HDG application contract and lifecycle, with a CUDA numerical session |
| Mathematical modes | Conventional Euler HDG and optional IGR augmentation | Conventional Euler HDG and optional IGR augmentation |
| Unknown contracts | Conventional `U4/Uhat4`; IGR `U7/Uhat5` | Conventional `U4/Uhat4`; IGR `U7/Uhat5` |
| External state | Borrowed component-major binary64 initial state; application projection from a supplied four-component coefficient | Same; copied into persistent device-owned state during initialization |
| Initial trace | Optional supplied independent true trace at the generic seam; application independently projects the trace from the case initial condition | Same generic and application semantics |
| Static source | Optional four-component static Euler source; empty means exact zero | Same |
| Boundary topology | Periodic topology and physical boundaries lowered from the shared semantic boundary plan | Same; physical policy data are copied to a device table at initialization |
| Physical Euler boundaries | Supersonic inlet, subsonic inlet, supersonic outlet, subsonic pressure outlet, far field, nonreflecting outlet, and slip wall | Same seven physical policies; at most 32 retained static policy entries |
| IGR physical closure | Euler physical rows plus homogeneous-Neumann auxiliary trace closure in the same nonlinear system | Same |
| Mesh | Externally supplied, conforming, two-dimensional quadrilateral `ParMesh` | Same |
| Refinement | Nonnegative serial and parallel uniform refinement through the shared mesh path | Same; frozen application gates covered both serial and parallel refinement |
| Polynomial degree | p1-p3 | p1-p3 |
| Time integration | Backward Euler only; arbitrary positive fixed step or committed-state CFL control, including final/output clipping | Same |
| Nonlinear admission | Finite-state checks, positive density and pressure, nonlinear stopping, Armijo line search, and true-residual acceptance | Same |
| HDG algebra | Residual/Jacobian assembly, local factorization, condensation, true-trace assembly/solve, and recovery | Same, executed by the persistent ordinary-CUDA engine |
| Acceptance gates | Recovery, conservation/source balance, boundary-flux balance, finite/admissible committed state, and true residual | Same |
| Transactions | Prepare/Commit/Abort, stale or divergent-token rejection, rollback, and retry from the unchanged committed state | Same, including failure-injection and retry qualification at the session boundary |
| Application failure policy | An unrecoverable failed Prepare ends the run; the session itself remains failure-atomic | Same |
| Diagnostics | Committed time/step/generation and maximum-wave/stable-step data; detailed attempt health is available to tests | Same, with CUDA work and transfer evidence available to tests/benchmarks |
| Output | Fixed-interval ParaView scheduling with initial/final policy, publishing density, momentum, energy, velocity, and pressure; IGR also publishes sigma; no-output operation is supported | Same, with explicit prepared-state acknowledgement before commit |
| Restart/checkpoint | Generic borrowed restart is explicitly unsupported and atomic. The generic main path has no checkpoint reader/writer; the legacy `save_state_files` option is not a supported HDG capability | Restart and state-file requests are explicitly rejected |
| GLVis | Explicitly unsupported | Explicitly unsupported |
| MPI communicator | Request/mesh communicator is authoritative. Generic runtime gates covered ranks 1-3, subcommunicators, passive world ranks, and zero-owned ranks; application process gates covered ranks 1-2 | Active communicator must contain exactly one or two ranks. Subcommunicators with passive world ranks are supported for active sizes one and two |
| Device topology | CPU execution; no device ordinal | Device ordinal zero, one physical GPU. Rank two is host-staged and rejects zero-owned ranks |
| Precision | Binary64 state and algebra | Ordinary FP64 CUDA state and algebra |

The rank-two CUDA path is a correctness and application-functionality surface,
not a multi-GPU or scaling claim. It requires both ranks to use the same
physical GPU and uses host-staged MPI communication. CUDA-aware MPI, multiple
GPUs, rank counts above two, and zero-owned CUDA ranks were outside the frozen
surface.

### Intentionally unsupported application surfaces

The campaign preserves these boundaries unless separately authorized:

- dimensions other than two, nonconforming meshes, triangles, tetrahedra,
  hexahedra, and other mesh families;
- polynomial orders outside p1-p3;
- Euler gamma other than the application contract value 1.4;
- explicit Runge-Kutta, IMEX, multistep, or other new HDG integrators;
- generic HDG restart/checkpoint import or a new checkpoint format;
- GLVis and unrelated output systems;
- multi-GPU execution, CUDA-aware MPI scaling, and active CUDA communicators
  other than rank one or rank two;
- a new IGR solver family, IGR parameter claims, or changes to the Euler/IGR
  mathematical contract.

The repository also retains frozen scalar, SC25, oracle, and research
fixtures. They remain test/reference assets and are not eligible tensor-engine
application paths.

## Production architecture

### Typed selection

CUDA engine selection is an explicit typed part of resolved configuration and
the backend-neutral request:

- application type: ordinary CUDA or tensor-core CUDA;
- generic type: `CudaEngineKind::ordinary` or
  `CudaEngineKind::tensor_core`;
- CLI spelling: `--hdg-cuda-engine ordinary|tensor`.

Ordinary is the default. Tensor is valid only with
`--discretization hdg --device cuda`. Invalid combinations fail during
collective request admission,
before runtime construction or device allocation. Selection participates in
rank consensus and is recorded in diagnostics, benchmark metadata, and output
provenance. An environment variable is not a selector.

### One lifecycle, two CUDA leaves

Both CUDA engines use the same `GenericEulerHdgCudaSession` and the same
application route:

```text
case/CLI resolution and MPI consensus
  -> mesh, initial/source/trace projection, boundary lowering
  -> backend-neutral EulerHdgRequest
  -> GenericEulerHdgCudaSession
  -> ordinary or tensor-core CUDA numerical leaf
  -> shared Prepare / Commit / Abort and output scheduling
```

The tensor leaf may change internal layouts, packing, batched local algebra,
factorization, condensation, Schur work, recovery, kernel fusion, and
workspace ownership. It must not copy the main backend, create a second
application, or split conventional Euler and IGR into different lifecycle
families.

Conventional and IGR dimensions are mathematical/external contracts, not
mandatory internal storage layouts. Padding or transformed storage is allowed
provided it is private to the numerical leaf.

### Implemented tensor numerical leaf

The production tensor selection retains the existing FP64 residual/Jacobian,
global true-trace assembly, recovery, validation, and transactional code.  Its
local condensation leaf keeps the ordinary engine's pivoted FP64 LU and
triangular solves, then evaluates the material
`C * A^{-1} * [B f]` product with native FP64 WMMA `m8n8k4` operations.  A
four-warp element block processes 8-by-8 output tiles; row, column, and inner
tails are explicitly zero-filled, so the same kernel admits every current
conventional `U4/Uhat4` and IGR `U7/Uhat5` p1-p3 shape.  Schur and condensed
right-hand-side products remain binary64.

The recurring true-trace GMRES preconditioner is selected by both CUDA engine
and mathematical mode:

| CUDA engine | Conventional | IGR |
| --- | --- | --- |
| Ordinary | HYPRE BJ-ILU0, natural, one application, direct factor solves, refresh each solve | HYPRE BJ-ILU0, RCM, one application, direct factor solves, refresh each solve |
| Tensor | HYPRE diagonal scaling | HYPRE BJ-ILU0, RCM, one application, five lower and five upper Jacobi iterations, first factor per attempt |

This is an explicit numerical policy recorded for every accepted attempt.
Conventional attribution therefore separates the tensor condensation kernel
from the independent diagonal-scaling effect. Tensor and ordinary IGR both use
external policy name `hypre_ilu0`, with exact metadata type 0, fill level 0,
local RCM reordering 1, and one application iteration. Their factor triangular
solve is nevertheless an explicit typed difference: ordinary IGR records
`direct` with zero Jacobi iterations, while tensor IGR records
`iterative_jacobi` with five lower and five upper iterations. Tensor IGR also
uses typed factor policy `first_trace_solve_per_attempt`; the ordinary policies
use `refresh_each_trace_solve`, and diagonal scaling records both factor and
triangular-solve policies as `not_applicable`.
Exact factor setups and reuses, maximum aggregate factor age, observed outer
Krylov iterations, derived callback/inner-work and Jacobi factor-SpMV bounds,
nonlinear work, line
searches, transfers, tensor tiles, and fallback work are retained in each
aggregate benchmark sample. MMA contribution is established separately from
profiler and SASS evidence; an end-to-end engine speedup is never represented
as the speedup of MMA alone or as a factor-reuse-only result.

HYPRE exposes the completed GMRES outer-iteration count but neither a
cumulative preconditioner-callback count nor a restart-cycle count. Its ILU
iteration getter reports only the most recent application. Intercepting the
GMRES callbacks to manufacture cumulative counters was rejected because it
would replace HYPRE's function/data association and add recurring bookkeeping
inside the timed solver. Production accounting therefore leaves the MFEM/
HYPRE call path untouched and reports the explicit
`ilu_accounting_kind=derived_bounds_v1` contract.

For each successful ILU-preconditioned trace solve with `I` observed outer
iterations, fixed GMRES restart dimension `K`, and configured ILU application
count `M`, the pinned implementation derives:

- zero callback and inner-work bounds when `I=0`;
- callback lower bound `I + ceil(I/K)` and upper bound `2*I` when `I>0`;
- inner-work lower and upper bounds equal to the corresponding callback bound
  times `M`.

For tensor IGR, each ILU application performs five fixed lower-Jacobi and five
fixed upper-Jacobi iterations. HYPRE starts each side from zero, so the first
iterate elides the factor SpMV; the bounded recurring factor work is therefore
`(5-1) + (5-1) = 8` factor-SpMV updates per application. The fields
`ilu_jacobi_factor_spmv_updates_lower_bound` and
`ilu_jacobi_factor_spmv_updates_upper_bound` are exactly the corresponding
callback bound times `M` times 8. Direct ordinary ILU and diagonal tensor
samples report both update bounds as zero.

Every addition and product is checked for `uint64_t` overflow. Bounds are
derived and accumulated separately for every trace solve; they are never
recomputed from an accepted step's aggregate outer-iteration count because
the ceiling term is not additive. Each accepted step records the fixed restart
dimension `K` and its number of trace solves. The independent aggregate
validator requires the reported restart-cycle count (callback lower bound
minus aggregate outer work) to be feasible for a partition of that work over
at most the reported number of positive solves. It retains no per-solve
partition or event history. The callback upper bound remains exactly `2*I`;
a zero aggregate requires all four bounds to be exactly zero. Inner bounds
must be the exact callback bounds times `M`, with checked overflow. Factor
operations are exact
aggregate counts under the typed lifetime policy: refresh-per-solve requires
one setup per trace solve, zero reuses, and zero maximum age; first-solve-per-
attempt requires one setup and `trace_solves-1` reuses, with maximum age equal
to `trace_solves-1` because each completed solve advances one generation. The age is
one aggregate maximum, never an event history. These counts do not require
replacing a solve callback. The sample reports
`ilu_accounting_complete=true` only when both factor and derived-bound
contracts hold, including the triangular-solve policy and Jacobi factor-SpMV
bounds.
Diagonal-preconditioned samples report `ilu_accounting_kind=unavailable`, zero
ILU bounds/setups/reuses/age, factor policy `not_applicable`, and complete
accounting because no ILU work is applicable.
This is bounded aggregate scalar provenance only. It does not retain callback
events or per-solve history and introduces no flight recorder, trace/event
stream, replay facility, every-event capture, or public production evidence
API. Final call-structure and instruction evidence comes from external,
one-off profiler captures.

A provisional rank-one p2 tuning diagnostic also evaluated three Jacobi
iterations per factor side. The reduced local work increased outer Krylov work
and regressed accepted-step time relative to the selected five-per-side policy,
so it was rejected rather than promoted. The unbound provisional samples are
not final-candidate qualification evidence and no exact diagnostic timing or
iteration result is promoted by this document.

An attempted tensor-IGR `hypre_bj_iluk1_rcm` policy was also rejected. Source
inspection of the pinned HYPRE CUDA path and direct setup reproduction showed
that device-memory ILU setup does not support fill level greater than zero;
continuing after that rejection left null factor data and crashed. Production
therefore retains the supported ILU0 configuration rather than hiding an
ordinary-CUDA or other
preconditioner fallback. No fill-level-1 result is admissible campaign evidence.

### Precision and acceptance policy

The implemented engine is an all-FP64 policy. All borrowed inputs, persistent
state, pivoted LU and triangular solves, WMMA operands and accumulation, Schur
and condensed right-hand sides, true-trace algebra, recovered state, host
exports, and published fields are binary64. Native FP64 WMMA lowers to the
SM120 DMMA instruction family; this candidate does not use TF32, BF16, FP16, or
FP8.

Acceptance-critical work remains FP64:

- finite-state, density, pressure, and sound-speed admission are evaluated in
  FP64;
- Armijo and nonlinear stopping decisions use an FP64 residual evaluation;
- the true-trace residual, recovery consistency, conservation/source balance,
  and boundary-flux balance are recomputed in FP64 before acceptance;
- committed CFL diagnostics and the final committed/output state are FP64.

No mixed-precision scaling, residual replacement, iterative refinement,
compensation, or error-correction route is active in this implementation, so
that work is zero rather than excluded from timing. No numerical tolerance was
changed or weakened for tensor selection. Introducing any such policy is a new
numerical candidate and requires renewed three-way qualification and
performance evidence.

Bitwise identity and identical iteration counts are not required. Nonlinear,
Krylov, line-search, retry, transfer, tensor-tile, and fallback counters make
the active algorithmic differences observable.

### SM120 admission and no fallback

The production tensor engine is an SM120 path. The canonical architecture
contract is dependency CUDA architecture 120, MFEM `CUDA_ARCHS=80-virtual`,
and application `CUDA_ARCHITECTURES=120`.

Tensor selection is fail-closed. Initialization must collectively reject an
unsupported GPU, CUDA capability, tensor instruction form, shape, layout, or
precision policy with a clear tensor-engine diagnostic. It must not silently
execute the ordinary-CUDA kernel, and ordinary-CUDA work does not count as
tensor acceleration. The promotion candidate requires zero fallback
executions in every qualifying tensor run.

The rank-two tensor surface remains the existing one-GPU host-staged
correctness path. It does not broaden the hardware or MPI claim.

## Numerical qualification contract

Existing CPU/CUDA tests are parameterized for the engine selector rather than
copied into a tensor-only suite. Exact-candidate evidence compares all three
pairs:

- CPU versus ordinary CUDA;
- CPU versus tensor CUDA;
- ordinary CUDA versus tensor CUDA.

Conventional and IGR must each be covered at p1, p2, and p3. Qualification
uses global fields, norms, invariants, and solver-health quantities rather
than hashes or rank-local storage order. Required coverage is:

- uniform preservation;
- smooth periodic evolution and every supported physical Euler boundary;
- zero and nonzero static sources;
- supplied and independently projected initial traces;
- spatial behavior and Backward-Euler temporal behavior;
- residual/Jacobian, local condensation, true-trace action/solve, and recovery;
- conservation, source balance, and boundary-flux balance;
- positive density and pressure at every solver admission point;
- one-step and multi-step trajectory agreement;
- Prepare/Commit/Abort, rejected attempts, retry, rollback, and failure
  atomicity;
- fixed-step and CFL-controlled application operation, output/no-output
  schedules, and sigma publication;
- rank-one and rank-two application and runtime gates, including the
  one-GPU host-staged CUDA path and communicator-scoped tests;
- representative discontinuous IGR-enabled evolution with admissible state
  and localized regularization.

Any tolerance change requires a diagnosed numerical cause, comparison against
both controls, and explicit review before a gate changes.

### Implemented qualification map

The following stable test families map the contract above to canonical
qualification. Braces denote the selections registered by CMake. The external
promotion ledger records the exact `build.sh` command, complete log, test list,
and disposition for the candidate; this table identifies coverage and is not a
substitute for those results.

| Qualification surface | Canonical test family |
| --- | --- |
| Generic CPU lifecycle, transactions, algebra, and MPI | `hdg_generic_euler_cpu_rank{1,2}` plus the generic CPU runtime suite |
| CPU spatial and Backward-Euler behavior | `hdg_generic_euler_cpu_convergence_{conventional,igr}_p{1,2,3}` |
| Three-way global-field, norm, invariant, trajectory, solver-health, and localized-IGR alignment | `dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_p{1,2,3}_rank1` |
| Shared CPU/ordinary/tensor spatial and Backward-Euler behavior | `dg_igr_hdg_runtime_cuda_generic_euler_convergence_{conventional,igr}_p{1,2,3}_rank1` |
| Residual/Jacobian, condensation, deterministic true-trace Schur action, recovery, sources, boundaries, supplied/projected traces, rejection, and rollback | `dg_igr_hdg_runtime_cuda_generic_euler_extension[_tensor]_p{1,2,3}_rank{1,2}` |
| Prepare/Commit/Abort tokens, retry, failure atomicity, and ordinary/tensor rank-one/rank-two sessions | `dg_igr_hdg_runtime_cuda_generic_euler_session*` |
| Active subcommunicators and passive world ranks | `dg_igr_hdg_runtime_cuda_generic_euler_subcommunicator[_tensor]_rank{1,2}` |
| Full application routing, conventional/IGR p1-p3, physical boundaries, external meshes, refinement, fixed/CFL steps, and rank-one/rank-two host staging | `dg_igr_hdg_main_backend_*cpu*`, `dg_igr_hdg_main_backend_*cuda*`, including the `physical_extension`, `igr`, `cfl`, `refine`, and `representative_cases` families |
| Output/no-output scheduling, VTK fields, sigma publication, and intentionally unsupported routes | `dg_igr_hdg_main_backend_*vtk*`, `*_no_vtk`, `*_rejects_*`, and `*_unsupported_state_routes` |
| Native production-linked tensor instruction | `dg_igr_hdg_runtime_cuda_tensor_sass_evidence` |

The three-way tests compare global values and invariants, not hashes or
rank-local layout. Hashes remain useful only for transactional identity inside
session tests.

## Performance and profiling contract

The existing `dg_igr_hdg_cuda_accepted_step` benchmark and its paired campaign
driver are the measurement authority. Together they carry the typed engine
selector and report engine, mode, degree, workload geometry, numerical policy,
solver work, transfer and fallback counts, initialization, memory observations,
build identity, and hardware identity.

Final-tree records use the schema-2 contract. Each measured sample must carry
complete aggregate nonlinear, observed outer-Krylov, derived ILU callback,
inner-iteration, and Jacobi factor-SpMV bounds,
line-search, retry, recovery, reduction, transfer, synchronization,
acceptance-gate, condensation-launch, and MMA-tile accounting.
Precision-correction accounting must be complete and
all-FP64 with every mixed-precision/refinement/replacement/correction count
zero. Fallback accounting must be complete with count zero. Tensor records
must report supported hardware and positive production condensation/MMA work;
ordinary records must report no tensor-condensation work. The selected trace
preconditioner, all-FP64 policy, exact tolerance/stopping policy, rank-one SM120
GPU identity, and embedded source/build/cache identity are validated rather
than accepted as free-form metadata.

### Frozen performance workloads

The primary saturated matrix is conventional/IGR crossed with p1/p2/p3 using
one backend-neutral periodic workload:

- `mesh_n=52`, `domain_periods=1`, and `dt=1e-5`;
- 2704 quadrilateral elements and 85 principal element blocks for 84 SMs;
- no host export in the primary accepted-step timing;
- six alternating pair rounds for final-tree ordinary versus tensor and six
  alternating pair rounds for final-tree ordinary versus the frozen control;
- one benchmark warmup step and five measured accepted steps per invocation;
- an additional export-enabled IGR p3 invocation for each final-tree engine.

The 85-block/84-SM geometry is the predeclared saturation basis; it is not
selected after inspecting results. Mesh, timestep, mathematical mode,
tolerances, stopping rules, output settings, executable build, GPU lease, and
hardware conditions are identical within each paired comparison. The recorded
trace policy follows the mode-aware table above. In IGR, type 0, fill level 0,
RCM ordering, and the one-application count are identical between the engines;
the explicitly reported direct-versus-five-per-side-Jacobi triangular policy
and factor-lifetime policy differ. All
recurring trace work remains inside the accepted-step timing; aggregate
solver-work records disclose exact factor work and derived solve bounds, while
profiler evidence separately attributes the tensor-condensation MMA
contribution.

A second, separate break-even campaign uses `mesh_n=4`,
`domain_periods=1`, and `dt=1e-5`, with the same six-cell, six-pair, no-export
method. It characterizes launch- and solver-overhead-dominated behavior and
does not replace, pool with, or redefine the saturated primary matrix. Its
IGR results are disclosed even when tensor execution is at break-even or
regresses.

The campaign driver exposes `--evidence-mode smoke|qualification` and admits
promotion evidence only in qualification mode. That mode fixes the exact
workload labels `sm120-mesh52-saturated` and `sm120-mesh4-break-even`, the
geometries and export control above, exactly six pairs per cell, and at least
10,000 paired-bootstrap resamples. It also requires one clean expected
commit/tree, a canonical Release SM120 performance build with HDG runtime on
and observability/NVTX off, its MFEM identity, the exact
frozen-control source/overlay/build/executable relationship, live
`nvidia-smi` metadata, and `CUDA_VISIBLE_DEVICES` equal to the exclusive GPU
index. Source, executables, build inputs, raw benchmark streams, driver, and
hardware identity are rechecked before a completed manifest is admitted.
The MFEM identity must expose a concrete HYPRE MPI/CUDA SM120 spec, its exact
32-character Spack DAG hash, a hash-bound installation prefix, and the same
spec/hash pair in the canonical reuse-ABI record. The admitted campaign retains
that identity explicitly. Any HYPRE source revision or configuration fields
exposed by the canonical manifest are also retained and must be nonempty. This
binding is part of the derived GMRES/ILU accounting contract: any exposed
revision must be an exact commit identity and any exposed SHA-256 must be
complete. A version label such as `develop` without the exact DAG identity is
insufficient.

`status=completed` means that the declared run finished; it is not the
performance disposition. The saturated promotion gate is positive only when
all six conventional/IGR x p1/p2/p3 tensor-versus-final-ordinary paired 95%
confidence intervals have lower bound greater than one and no final-tree
ordinary-versus-frozen cell has a regression interval wholly below one. Every
nominal median regression and every interval classification is still
disclosed. The mesh-4 result is always characterization-only and cannot pass
or fail the saturated promotion gate.

Primary timing is the no-export accepted step from Prepare through Commit.
Recurring packing, conversion, validation, synchronization, transfers, and any
fallback work remain inside that region. Initialization and the IGR p3
export-enabled control are reported separately.

Device memory has three deliberately distinct observations. The primary timing
records `cudaMemGetInfo` snapshots before problem construction, after
initialization, and after samples; their maximum is only a maximum of those
observations, not an in-step peak. MFEM and HYPRE Umpire allocator
high-watermarks are allocator-scoped and may omit direct CUDA, library, or
driver allocations.

After all primary timing is complete, `hdg_tensor_memory_campaign.py` runs a
dedicated, untimed control for each saturated final-tree
ordinary/tensor x conventional/IGR x p1/p2/p3 cell three times. It launches
the same schema-2 accepted-step benchmark command while polling the selected
GPU's `timestamp,uuid,memory.total,memory.used,memory.free` row with
`nvidia-smi --loop-ms=50`. Ten stable pre-run and ten post-run samples bind the
baseline; active sample gaps may be no greater than five requested intervals;
baseline range/drift may be no greater than 64 MiB; and the three incremental
peaks must have spread no greater than the larger of 64 MiB or two percent of
their median. Each externally observed absolute whole-device peak must also be
at least the benchmark's internal `cudaMemGetInfo` used-memory floor, allowing
1 MiB for reporting granularity. Commands, benchmark stdout/stderr, every poll
row,
timestamps, process status, aggregation, and file identities are retained.

The memory driver accepts qualification evidence only. It fixes the saturated
label/geometry/timestep, exactly three repeats and a 50-ms poll request; binds
one clean expected commit/tree, the same observability/NVTX-off canonical
performance build/MFEM identity, canonical
benchmark path and embedded schema identity; requires the live SM120 UUID and
`CUDA_VISIBLE_DEVICES` index to agree; and rehashes the source, executable,
build inputs, driver, validator, sampler, raw artifacts, and hardware identity
before setting `qualification_pass=true`.

That sampler is a whole-device high-water observation, not a per-PID allocator
measurement. It includes driver and unrelated allocations on the selected GPU,
and the incremental result depends on the measured pre-run baseline. The
50-ms polling cadence can miss shorter-lived peaks, while a scheduling gap up
to 250 ms can still pass the declared gate. Process launch and the first
post-run sample bound the active window; the sampler does not attribute bytes
to individual CUDA contexts, kernels, or allocations. These limitations, and
the primary snapshot/allocator observations, remain separate in the ledger;
none is relabeled as an exact instantaneous application peak.

All timing and post-timing memory controls use one exclusive authorized SM120
lease. Retained metadata includes
commit/tree, clean state, build and executable identities, GPU UUID/name,
driver, CUDA runtime/toolkit, compilers, architecture tuple, clocks, power,
temperature and throttling observations, commands, and raw samples. Every
matrix cell reports medians, MAD/IQR or equivalent spread, the paired bootstrap
95% confidence interval, and speedup. Classification is fixed before results:

- interval lower bound greater than 1: win beyond observed paired noise;
- interval upper bound less than 1: regression beyond observed paired noise;
- otherwise: indeterminate/break-even.

All six cells and both controls remain reportable under that classification;
selected wins cannot suppress regressions or indeterminate cells.

Accepted-step attribution must separate at least:

- residual/Jacobian assembly;
- the fused local factorization/algebra/condensation kernel category;
- true-trace assembly and solve;
- recovery;
- reductions and validation;
- layout conversion or packing;
- synchronization and transfers.

Final-candidate profiling must show tensor instructions executing in the
intended production hotspots and quantify their time/work contribution. A
tensor-capable library call alone is not evidence. Retained profiler output
and SASS must identify the exact candidate binary and selected tensor engine.

`profile_harness.py --evidence-mode qualification` binds profiling to the
clean exact candidate, a canonical Release SM120 build with observability/NVTX
enabled, its MFEM identity, the profiled executable, the live profiler
executable and exact version, and the selected compute-capability-12.0 GPU.
Its NCU qualification route additionally fixes the production accepted-step
benchmark, tensor engine, no-export measured step, and one explicit proof kind.
`tensor-instructions` fixes the condensation NVTX range, demangled
`TensorBorrowedCondenseKernel` filter, launch count, and exact FP64
DMMA/duration metrics; it rejects zero DMMA, zero duration, missing launches,
or numeric rows from another kernel. `accepted-step-attribution` instead fixes
the outer profiled-accepted-step NVTX range, duration-only metric, no kernel
filter, and a positive launch limit. It imports the native raw table, requires
the material HDG categories and positive tensor-condensation work, and retains
category/kernel totals plus the tensor kernel's fraction of captured GPU
kernel duration. The local pivoted LU, triangular solves, and Schur/RHS product
execute in one production kernel and therefore remain one fused attribution
category; the native kernel table and DMMA proof quantify that category without
pretending an artificial kernel boundary exists. The captured fraction excludes
host/MPI, CUDA-API, and memcpy time and is not DMMA proof, so both NCU kinds are
required. The Nsys qualification route validates the same exact tensor
accepted-step application arguments, identities, and native report, then
generates and hash-binds separate `nvtx_sum`, `cuda_api_sum`,
`cuda_gpu_mem_time_sum`, and `cuda_gpu_mem_size_sum` CSV tables. It requires
nonempty native-tool output but does not turn those descriptive tables into an
unjustified numerical pass gate. Smoke-mode completion has none of these
qualification meanings.

The content-addressed profiler bundle retains these exact derivatives:

- the clean candidate commit/tree, canonical `--nvtx-profile` build manifest,
  profiled executable size/SHA-256, profiler executable size/SHA-256 and
  version, complete command, environment, GPU identity, and raw report;
- NCU condensation captures for representative conventional and IGR accepted
  steps, with the exact native kernel symbol, NVTX selection, launch count,
  `gpu__time_duration` result, and FP64 DMMA metric/count;
- qualified NCU `accepted-step-attribution` captures and derived tables that
  quantify local assembly, local algebra/condensation, trace assembly/solve,
  recovery, reductions/validation, packing/conversion, and nonlinear update,
  retain optional synchronization/transfer categories when present, and report
  unclassified outer-range work and the tensor kernel fraction;
- Nsys reports and derived NVTX/CUDA-API tables for representative conventional
  and IGR accepted steps, including every extraction command and artifact hash;
- an exact production-linked SASS dump or canonical SASS-test log containing
  the selected native sm_120 function, cuobjdump path/version and command,
  profiled binary hash, `DMMA.8x8x4` count, and retained artifact hash.

Every derived CSV or text summary is hashed beside the raw report and records
the tool/import command that produced it. Diagnostic MMA tile counters establish
selected work but are not substituted for NCU and SASS instruction evidence.

Performance success requires an end-to-end improvement beyond paired-run
noise on representative conventional and IGR workloads. All six p/mode cells
remain reportable even when some are below break-even. Ordinary CUDA remains
the default regardless of selected tensor wins.

### Interpretation limits

- Tensor instructions accelerate the local condensation product only. The
  existing residual/Jacobian assembly, global trace operations, recovery, and
  acceptance gates remain ordinary FP64 CUDA kernels.
- HYPRE diagonal scaling in tensor-conventional mode is an independent
  algorithmic effect and its work is reported separately from MMA. Tensor IGR
  matches ordinary IGR's type-0, fill-0, RCM, one-application `hypre_ilu0`
  base configuration but uses five-per-side iterative Jacobi instead of direct
  factor solves and a different, explicitly counted factor lifetime. Any IGR
  end-to-end effect from iterative triangular solves or within-attempt factor
  reuse is therefore disclosed separately from tensor-condensation MMA
  attribution. The rejected three-per-side and unsupported fill-level-1 tuning paths are not
  part of the production engine, and neither may silently fall back.
- Preliminary external profiling identified sparse direct triangular solve as
  the dominant tensor-IGR accepted-step hotspot. That observation motivated the
  fixed iterative policy but is not final-candidate performance evidence.
- p1-p3 all use zero-filled 8-by-8/inner-dimension tails. Tile utilization and
  fixed launch costs are therefore degree- and size-dependent; the mesh-4
  campaign defines the disclosed break-even region, including small IGR
  break-even or regressions.
- On the authorized WSL profiler path, Nsys retains host, NVTX, CUDA API, and
  transfer structure but does not provide a usable GPU-kernel timeline. NCU is
  the quantitative kernel-time and tensor-instruction authority; this
  limitation is carried into the ledger rather than inferred away.
- Rank two remains a one-GPU host-staged correctness path. No performance,
  multi-GPU, CUDA-aware MPI, or scaling claim is made from it.
- Unsupported tensor hardware or policy fails closed. There is no ordinary-CUDA
  fallback whose work could count toward tensor acceleration.

## Candidate freeze and external promotion ledger

One exact clean candidate commit and tree is frozen before final qualification.
Any source change after the freeze invalidates that candidate's canonical tests,
performance matrix, profiler/SASS evidence, and independent review. Exact
results are appended only to the external ledger, never to the candidate tree.

The ledger is a sorted, immutable manifest whose referenced artifacts carry
size and SHA-256 identities. A detached SHA-256 identifies the completed ledger
itself. It contains at least:

| Promotion record | Required immutable payload |
| --- | --- |
| Candidate identity | Repository origin, exact clean commit/tree/status, candidate branch, and this document's blob/SHA-256 |
| Canonical builds | Absolute tracked `build.sh`, full commands/logs, CPU and SM120 manifests, compiler/toolkit/dependency/MFEM identities, architecture tuple, NVTX-profile state, and executable hashes |
| Frozen ordinary control | Commit `30afa08f8ad19a3ad37e627463ef63c40da1ae91`, tree `03313cf6b5b17d05fc3d768a6c77770295ced343`, harness-only `--order` overlay diff/hash, canonical build manifest, and executable hash |
| Canonical qualification | Complete CPU, ordinary-CUDA, and tensor-CUDA commands/logs; exact test inventory; per-test outcome; rank/resource conditions; and any diagnosed tolerance disposition |
| Three-way numerical matrix | Conventional/IGR p1-p3 CPU/ordinary/tensor comparisons, global errors/norms/invariants, convergence and trajectory gates, solver health, localized IGR measures, and pass/fail criteria |
| Primary performance matrix | Mesh-52 six-cell/six-pair campaign manifest, raw JSON/stderr identities, paired statistics and classifications, solver-work comparison, hardware snapshots, zero-fallback census, initialization, and memory observations |
| Historical regression control | Mesh-52 final-tree ordinary versus frozen ordinary six-cell/six-pair manifest and every disclosed regression/indeterminate cell |
| Break-even and export controls | Separate mesh-4 six-cell/six-pair manifest and the mesh-52 export-enabled IGR p3 ordinary/tensor records |
| Post-timing memory controls | Three 50-ms whole-device polling repeats per declared final-tree engine/workload cell, raw poll rows and benchmark streams, baseline/gap/spread/internal-floor gates, incremental and absolute peaks, and the sampling/PID-attribution limitations above |
| Profiler and SASS proof | The raw and derived NCU/Nsys/SASS identities, commands, tools, kernel symbols, category attribution, DMMA evidence, and WSL limitation specified above |
| Fallback census | Count and identity of every qualifying tensor test, benchmark, and profiler run, with aggregate fallback count exactly zero |
| Independent review | Distinct author, exact reviewed commit/tree, evidence-ledger hash, engineering/numerical-solving/performance/scientific/selection findings, and final `dg-igr-review` disposition |
| Promotion decision | Approval or rejection, approving authority, timestamp, limitations, and the five maturity dispositions below |

The ledger may be stored with the promotion request or another durable artifact
store; its location is not hard-coded into the application. Missing,
unavailable, dirty, differently bound, or unhashed evidence is reported as such
and cannot be interpreted as a pass.

## Maturity dispositions

The external ledger reports these axes separately; success on one does not
imply success on another.

- **Engineering maturity:** qualified for opt-in SM120 production use only when
  the shared lifecycle, typed selection, fail-closed admission, full supported
  CPU/CUDA application surface, canonical rank-one/rank-two gates, exact build
  provenance, zero-fallback census, and distinct-author review are positive.
- **Numerical-solving maturity:** qualified for the supported conventional and
  IGR p1-p3 selection only when the three-way field/norm/invariant,
  convergence, transaction, conservation, admissibility, and localized-IGR
  matrix is positive without weakened tolerances. It does not require bitwise
  identity or identical iteration counts.
- **Performance maturity:** qualified on the declared representative saturated
  SM120 workloads only when conventional and IGR show end-to-end wins beyond
  paired noise, all six cells and the mesh-4 break-even region are disclosed,
  actual DMMA work is proven and attributed, and ordinary CUDA has no
  unexplained regression from the frozen control.
- **Scientific maturity:** not established by this campaign. Optimal IGR
  parameters, shock accuracy, entropy convergence, invariant-domain
  properties, long-time robustness, scientific superiority, and multi-GPU
  scalability remain outside its claims regardless of engineering outcome.
- **Selection maturity:** ordinary CUDA remains the production default. After
  the other promotion gates and approval are positive, tensor CUDA is ready
  only as an explicit SM120 opt-in with its recorded break-even and profiler
  limitations; unsupported selection fails rather than silently falling back.
