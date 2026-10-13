# Repeatable GPU optimization studies

Use `perf.sh` to compare two source versions on PACE A100 hardware. It freezes
inputs, invokes the canonical `build.sh`, schedules correctness validation,
collects independent paired timings, and then captures bounded kernel profiles.
The site policy is the free `gts-sbryngelson3` account with QoS `embers`.
Compilation runs on CPU compute nodes. Device validation, timing, profiling,
and native trace import run on A100 compute nodes.

Initialize the pinned MFEM submodule and the tested build Spack as described in
[the PACE build guide](pace.md). From the PACE login node, run one command:

```bash
./perf.sh submit --site pace \
  --baseline BASELINE_REF --candidate HEAD \
  --run-dir /absolute/scratch/studies/my-change-01 \
  --site-root /absolute/scratch/igrlab-dependencies
```

New suites default to 3600 seconds for `timings_budget_seconds`,
`profiles_budget_seconds`, `command_timeout_seconds`, and each case's
`profile.timeout`. Each accepts any positive finite number, so a focused run
can request a shorter budget or a long application can request more than one
hour. These values do not extend an allocation: the worker also enforces the
actual Slurm end time and its 120-second cleanup reserve. A command stops at
the earliest applicable deadline.

Explicit limits in existing frozen suites and executors remain unchanged.
Changing an execution budget requires a new immutable run; it does not change
the mesh, steps, precision, tolerances, numerical gates, or paired protocol.
Retain successful partial processes through `continue-timings` when supported,
rather than repeating them. The free one-hour allocation policy still applies.

`BASELINE_REF` is an existing Git commit, branch, or tag. Both versions must
contain the supported portable build driver and use the same PACE dependency
policy. For the DG/HDG optimizations first tested in October 2026, `bef3361`
is a suitable pre-change baseline. Old revisions lacking the portable driver
are rejected before queue submission. There is no implicit driver overlay.
Use `--candidate working-tree` to include staged, unstaged, deleted and
nonignored untracked regular files. Their actual bytes are fingerprinted; a
dirty flag alone is insufficient. MFEM must be the clean pinned upstream
submodule with its annotated `v4.9` tag.

The run directory must be new. `--site-root` retains concrete dependencies
and separate CPU/A100 Spack environments. Spack's `location -r` supplies the
tool root; `--spack-root /absolute/path/to/spack/core` is available when shell
support is not initialized. Package installation paths come from
`spack -e ENV location -i PACKAGE` on compute nodes.

Source snapshots and incremental builds live in retained
`SITE_ROOT/study-cache` by default; `--cache-root DIR` selects another absolute
cache root. Application keys bind exact source content, build flags, and the
dependency identity. Changing only the correctness selection reuses the
application build directory. MFEM has a separate key covering its clean source
and release tag, build policy, site configuration and Spack tool identity.
Application-only changes therefore share MFEM, while the canonical driver
checks the concrete dependency/compiler ABI and immutable artifact stamp under
an independent cache lock. Interrupted matching cache fills can resume; foreign
partials or changed identities fail without replacing their files.
Identical baseline/candidate builds are submitted once. Leave cache entries in
place while their studies are retained; the coordinator never silently cleans
or replaces them.

These capabilities come from each frozen source's public driver. New drivers
run dependent validation with `--tests-only`, checking a successful build's
receipt before executing tests. Historical drivers keep their original cache
and validation behavior, which can include build steps in the validation job;
their capability is recorded in the plan and report. The coordinator applies no
driver overlay. Already queued studies retain their frozen implementation.

Add `--prepare-only` to freeze inputs and inspect saved build plans without
submitting jobs. Submit that prepared study with `resume --stage builds`.
Preparation performs Git/file operations and `build.sh --print-plan`; it does
not compile on the login node.

Existing canonical builds can be imported without rebuilding or moving their
source/output directories. First run the original source's canonical
`--tests-only` command through `scripts/build/test_result_receipt.py` in the
owned compute allocation, using the prebuilt receipt at
`APP_BUILD/igrlab-prebuilt-validation-v1.json`. The wrapper retains the actual
command, selected CTest graph, individual Passed outcomes, exact logs,
pre/post artifact identity and live allocation identity. A prebuilt receipt or
a successful scheduler exit alone is insufficient.

Create an absolute-path artifact manifest:

```json
{
  "schema_version": 1,
  "builds": {
    "baseline": {
      "build_request": "/absolute/site/pace/requests/build.BASELINE.json",
      "build_job_id": "REAL_BUILD_JOB_ID",
      "test_result": "/absolute/validation-results/BASELINE_ATTEMPT/result.json"
    },
    "candidate": {
      "build_request": "/absolute/site/pace/requests/build.CANDIDATE.json",
      "build_job_id": "REAL_BUILD_JOB_ID",
      "test_result": "/absolute/validation-results/CANDIDATE_ATTEMPT/result.json"
    }
  }
}
```

```bash
./perf.sh import --artifacts /absolute/artifacts.json \
  --suite scripts/perf/suites/pace-full-gpu-solvers.json \
  --run-dir /absolute/scratch/studies/imported-pair-01 --prepare-only
```

Import derives the exact roots/features from the real public requests and each
frozen source's `--print-plan`. Sources must already be standalone Git checkouts
with the clean pinned MFEM submodule; shared clones with alternates are rejected.
It pins selected executable bytes and the original test records. The measured
worker repeats canonical prebuilt and dependency-binding admission before
using those artifacts. Passed tests must cover the suite's correctness regex;
filtered graph backtrace numbering does not affect test command identity.
No validation job ID is invented and no build/test command is rerun by import.

`--prepare-only` is allowed inside the interactive allocation. Run the new
frozen executor's `worker --stage timings --run-dir NEW_RUN`, then
`worker --stage profiles --run-dir NEW_RUN` in that same owned free A100 shell.
Otherwise import schedules only timing/profiling stages from the login node;
prepared imports use `resume --stage timings`, never `resume --stage builds`.
A live test allocation must be the same owned interactive job at measurement.
A later terminal allocation state, including TIMEOUT after the sealed tests,
is retained honestly and is separate from the earlier Passed test outcomes.
Missing, interrupted or modified final receipts never qualify. Retain all
referenced source, artifact and result paths for the life of the study.

For CPU HDG, add a `cpu_builds` object with the same baseline/candidate record
shape, pointing to genuine CPU-only builds and their actual passed test
results. Each CPU member must use the exact same source snapshot as its CUDA
member. The existing CPU admission selects its concrete MPI/runtime prefixes;
the CUDA HDG facade is never substituted for a CPU build.

For a focused two-case study, shorter allocations can fit smaller scheduler
windows:

```bash
./perf.sh submit --site pace \
  --baseline BASELINE_REF --candidate HEAD \
  --run-dir /absolute/scratch/studies/focused-change-01 \
  --site-root /absolute/scratch/igrlab-dependencies \
  --cases dg-p1,hdg-igr-ordinary-p3 \
  --validation-time 00:15:00 --timings-time 00:15:00 --profiles-time 00:30:00
```

These limits are explicit choices for a small study; the defaults remain two
hours for correctness and one hour each for timings and profiles. Study limits
use whole minutes, with a one-minute minimum for correctness, a five-minute
minimum for timing/profiling stages, and those defaults as maxima.
The frozen plan records the limits. They do not change the compiled-artifact
cache key. For a baseline whose canonical driver predates `--validation-time`,
the coordinator verifies the recorded validation job and shortens its pending
allocation through Slurm; failures retain job IDs for a safe retry.
Timing and profiling workers subtract setup time and reserve two minutes for
writing results, also respecting Slurm's reported allocation deadline.
Choose larger limits when adding cases or expensive profiler sections.

## The optimization loop

1. Choose a baseline and an unchanged numerical workload. An identical-source
   control can quantify repeatability.
   Run the selected [numerical acceptance checks](numerical-validation.md) for
   changes to discretization, solver tolerances, precision, or kernel arithmetic;
   use the convergence selection when the change could affect accuracy order.
2. Use Nsight Systems attribution to choose expensive kernels or launch/sync
   overhead. Inspect those kernels with Nsight Compute.
3. Make one focused application/library change. Compare with the same suite,
   precision policy, dependency graph, and hardware target.
4. Admit a speed claim after correctness, useful-work counts and separate
   physical-output checks pass. Inspect paired variation; repeat on a new
   allocation when an improvement is close to noise.
5. Save compact records with the source change. Use the accepted source as the
   next baseline.

The maintained suite is [pace-a100.json](../scripts/perf/suites/pace-a100.json):
six default DG/HDG cases, three independent process pairs, separate process
warmup, a focused correctness selection, and bounded kernel filters.
The study's name-based correctness selection remains a quick regression gate;
the broader numerical acceptance labels are explicit canonical-driver selections.
Use `--cases dg-p1,dg-p3` for a targeted study, or `--suite FILE` for a saved,
versioned workload variant. An explicit case selection enables optional cases.
Do not change suite parameters between comparison members.

The [pace-full-backends.json](../scripts/perf/suites/pace-full-backends.json)
matrix enables eight cases on the saved CUDA build: CPU and CUDA DG/DGSEM,
fused SIMT and FP64 tensor DGSEM, and ordinary/tensor CUDA HDG. They use 36
periodic cells for ten steps. CPU field equivalence covers DG and DGSEM;
HDG retains comparisons between source versions. The optional `hdg-cpu-p3`
entry requires a CPU-profile executable; the CUDA HDG entry point rejects
`--device cpu`. A new submission enabling CPU HDG automatically creates separate
canonical CPU build and correctness jobs for both source versions. Measurement
waits for CPU and CUDA validation, verifies their prebuilt receipts and actual
dependency bindings, and launches CPU children with that profile's MPI executable
and library paths. CPU DG/DGSEM share the CPU artifact when it is present.
A retry of a CUDA-only study cannot add CPU HDG without a prior admitted CPU
build; use a new submission. This is an eight-case default, not a complete CPU/CUDA/tensor matrix for
every discretization. Its small grid validates execution and equivalence; it
does not establish representative GPU efficiency. The separate
[pace-full-gpu-solvers.json](../scripts/perf/suites/pace-full-gpu-solvers.json)
companion runs six CUDA full-application cases with three measured process
pairs and one warmup:

| Cases | Periodic cells | Order | Fixed horizon / timestep |
|---|---:|---:|---|
| DG ordinary; DGSEM unfused, fused SIMT and fused FP64 tensor | 1024 | 3 | `0.002 / 1e-5` (200 steps) |
| HDG ordinary and tensor | 256 | 3 | `5e-5 / 1e-5` (five steps) |

These use tracked meshes already available in `d735f57`. HDG uses the 64-cell
`hdg_smooth_transition_periodic_n8.mesh` with `serial_refine: 1`; the application
adapter accepts integer refinement levels 0..2 and defaults to zero. The
companion captures whole-application Nsight Systems timelines and bounded
Nsight Compute kernel selections. Command and allocation deadlines still
apply; completion within an hour is not established by host fixtures.

CPU equivalence belongs to the small matrix's own mesh and horizon. The GPU
companion retains accepted-work, finite-state and separate physical-field
gates between source versions; it has no same-work CPU reference and therefore
makes no larger-grid CPU equivalence claim. Full applications frozen at
`d735f57` compile named DG/HDG NVTX ranges but leave their runtime gate disabled.
CUDA/API/OSRT timelines remain available; absent named phase ranges must not be
reported as measured phase attribution. New full-solver builds support
`IGRLAB_PROFILE_NVTX=1` (legacy `DG_IGR_PROFILE_NVTX`) to enable their existing
regions when compiled with `--nvtx-profile`. The study runner sets both flags
only for CUDA profiler calls and pins them to zero for unprofiled timings and
field exports, recording the setting in each command receipt. Verify actual
DG/IGR regions in the resulting trace before claiming phase attribution;
the switch cannot add instrumentation to older or compiled-out binaries.
The accepted-step benchmark enables
NVTX only around its selected step. The companion's NCU selections use the
first invocation of each selected kernel name, which diagnoses those launches
without establishing steady-state kernel cost or full-solver speedup.

The [pace-full-gpu-igr.json](../scripts/perf/suites/pace-full-gpu-igr.json)
suite exercises active IGR through the same six GPU application routes, using
36 cells and ten steps. DG/DGSEM pin the FP64 exact-face CG/Chebyshev closure,
require converged production solves and the native final closure certificate,
and compare finite, nonzero exported sigma with the existing field tolerances.
HDG checks its coupled production configuration, successful accepted horizon
and sigma fields; it does not emit the DG/DGSEM closure certificate. The active
wave has no analytic error reference. This is a small closure diagnostic, with
no CPU IGR or larger-grid scaling claim. Whole-application Nsight Systems
captures include closure work. Nsight Compute requires transport and the
observed DG/DGSEM closure families: diffusion-face record/gather, PA mass and
diffusion, Chebyshev application, and reductions. The selection also includes
common vector kernels; those names alone do not establish exclusive closure
attribution. HDG retains its coupled volume, face, condensation, recovery and
trace-solve selections. Each selection remains bounded to the first invocation
of each matching kernel name. Use a profile-only suite override to refine
these diagnostics from a new timeline without repeating admitted timings.

The [pace-igr-scaling.json](../scripts/perf/suites/pace-igr-scaling.json)
suite provides a larger active-IGR size ladder. DG and all three DGSEM CUDA
routes use the tracked 1024-cell mesh with zero, one or two refinements:
1024, 4096 and 16384 cells. Ordinary and tensor HDG use 256 and 1024 cells.
Every point retains order three, ten fixed steps (`dt=1e-5`), the same strict
IGR controls, three independent process pairs, one warmup and the full
closure-inclusive profiler selections. The default calibration selection is
DG at 1024 cells and both HDG engines at 256 cells; select other points with
`--cases`, for example `dg-igr-cuda-p3-n128` or
`dgsem-igr-fused-dmma-p3-n128`.

Calibrate runtime, convergence and memory before promoting a point to the
optimization workload. At 16384 cells the current one-thread-per-face DG
closure launches about 128 blocks, enough to reach each A100 SM, but this alone
does not establish efficient occupancy or saturation. HDG uses dense element
blocks and a different trace solve, so its size must be assessed separately.
Ten steps are an initial calibration horizon; check startup share and increase
the horizon in a versioned suite when necessary. This GPU size ladder does
not itself supply CPU IGR parity.
Native captures and their processing stay on PACE compute nodes. A larger
trace can exceed the compact GPU activity export bound; retain that explicit
unavailability rather than interpreting it as zero activity.

Use one focused immutable run per size/backend when working in a one-hour
interactive allocation. An ordinary `retry --prepare-only --suite ... --cases
...` can reuse completed build/correctness receipts while collecting fresh
paired timings. Preserve the baseline source identity: changing the executor
or suite does not rebuild the application. After a production change, build
and validate that source through the canonical driver before comparing it
with a freshly measured baseline. Commit each validated solver revision batch
locally on the optimization branch; saved timings alone cannot admit a speedup.

The [pace-igr-cpu-parity.json](../scripts/perf/suites/pace-igr-cpu-parity.json)
companion pairs CPU and CUDA active IGR for ordinary DG and unfused DGSEM.
Start with its four default 36-cell cases; the matching 1024-cell cases are
optional selections with the `-n32` suffix. Both sizes retain order three and
the same ten-step horizon, exact-face CG/Chebyshev policy, final closure
certificate and five physical fields plus nonzero sigma. CPU and CUDA may
need different Krylov iteration counts; accepted work, numerical policy and
physical outputs must still agree. CPU execution uses one thread on the A100
node's host. This is a paired backend comparison, not a CPU scaling study.
Three optional `hdg-igr-{cpu,cuda,tensor}-p3-n6` cases add matched 36-cell HDG
work. For example, submit this suite with
`--cases hdg-igr-cpu-p3-n6,hdg-igr-cuda-p3-n6,hdg-igr-tensor-p3-n6` to request
the independent CPU artifact automatically. Both frozen sources must support
canonical shared-MFEM caches and prebuilt validation. CPU HDG process wall time
includes its rank-one MPI launcher, startup and shutdown; it is not a kernel
time. CPU counters and CUDA occupancy are not collected for CPU cases.
CPU fused/tensor routes remain unsupported. Host fixtures cover these contracts;
successful native runs are still required to establish HDG timing and parity.

The [pace-igr-dg-orders.json](../scripts/perf/suites/pace-igr-dg-orders.json)
research sweep selects ordinary DG orders one through ten on 1024 cells.
The additional `(p, cells)` points `(1, 16384)`, `(3, 4096)`, `(7, 1024)` and
`(10, 576)` have 65536, 65536, 65536 and 69696 scalar polynomial DOFs,
respectively. These complement the fixed-mesh sweep with approximately
constant work size; they are not equal-accuracy discretizations. CPU endpoint
cases at orders one and ten support same-order CPU/CUDA field comparisons.
Only the familiar p3 point is enabled by default; select research points
explicitly and establish native convergence, backend identity and memory use
before treating their times as admitted performance results.

Ten is this sweep's endpoint, not a study-runner order ceiling. The runner
accepts positive orders representable by the native command-line integer.
The selected executable enforces its actual kernel and storage constraints;
unsupported orders remain failed attempts, never successful CPU fallbacks.
In particular, the current fused DGSEM dispatch covers p1 through p10 and
has a concrete shared-memory bound. Admitting a study above p10 does not
claim that this dispatch implements it.

All order points initially use the same ten fixed steps and `dt=1e-5`.
Record Krylov work and numerical acceptance at each order. If stability or
convergence requires a different timestep, preserve the failed attempt and
declare a new workload; never silently shorten the horizon or loosen the
closure tolerances. Compare physical arrays only between matching meshes and
orders. This ordinary-DG suite does not establish high-order support for the
separate DGSEM or HDG implementations; their actual kernel/storage paths must
be extended and validated rather than merely changing study admission.

The order-ten implementation extends DGSEM's fused path with WMMA row tiling
and bounded shared storage. Packed/unfused DGSEM and dynamic generic HDG
use checked dimensions and allocation sizes rather than a chosen order
ceiling. The previous DGSEM research switch is no longer required.
HDG's default IGR quadrature grows as `max(12, 2*p+2)` so its
high-order element matrices are not built from an underresolved rule.
Existing low-order quadrature and single-site reduction paths are retained.
The source changes and host checks do not establish native p10 readiness;
run the extended DGSEM parity and overintegration tests and these HDG checks
through the canonical build driver before promoting larger cases:

- `dg_igr_hdg_runtime_cuda_cpu_numerical_alignment_p10_rank1`
- `dg_igr_hdg_runtime_cuda_generic_euler_extension_p10_ordinary_rank1`
- `dg_igr_hdg_runtime_cuda_generic_euler_extension_p10_tensor_rank1`

The p10 HDG checks use one physical element and nine periodic IGR elements,
with a single committed step and the existing numerical gates. High-order
HDG dense storage grows rapidly; its checked arena index capacity and device
memory remain real constraints. Calibrate size by order rather than applying
the p3 cell count blindly. Preserve failures and distinguish source support,
native numerical acceptance and measured performance.

Reports list every enabled frozen-suite case with its discretization, device,
engine, DGSEM residual, timing scope, and individual timing/profile state.
A two-case selection cannot establish coverage of every backend. CPU cases use
one thread on the host CPUs of the A100 allocation; their hardware/affinity and
thread settings are retained. CUDA occupancy does not apply to CPU execution,
and CPU hardware counters are not collected. Generic HDG application cases
measure whole-process wall time including startup, setup and shutdown. Compare
their CPU/CUDA/tensor results within that scope; do not compare those values to
DG/DGSEM solver elapsed time, HDG accepted-step timing, or DGSEM kernel timing.

| Adapter | Primary timing | Numerical/work gate | Diagnostic scope |
|---|---|---|---|
| DG/DGSEM application | Fixed-horizon full-solver time | Accepted steps/RHS work, zero rejects, NaN scan, geometry and five physical fields | CPU, ordinary CUDA, and selected DGSEM fused arithmetic |
| Generic HDG application | Whole-process wall time | Production success, fixed accepted horizon, field comparison; same-work CPU reference when selected | CPU, ordinary CUDA, tensor CUDA; whole-run timeline |
| HDG | Accepted-step median per process | Schema-2 policies/gates, zero retries/fallback, same work, separate exported checksums | Measured step zero; volume, condensation, face, recovery and trace-solve families |
| DGSEM, optional | Selected volume method time | Complete method/error observations against the existing long-double reference | Prescribed-sigma central-volume probe only |

HDG's five steps within one process produce one observation; they are not five
independent repetitions. Baseline/candidate process order alternates. Reports
retain raw observations, medians, paired percentage changes/ranges and relative
standard deviations. There is no automatic speed threshold or significance
claim. Ordinary and tensor engines have different trace policies: each case
compares the same engine across source versions. DGSEM volume probes establish neither
full-solver accuracy nor full-solver speedup.

For an identical-source control, the report also lists each case's independent
pair count, median and maximum absolute paired offset, and each member's
coefficient of variation. These describe the observations in that allocation;
the largest observed offset is not a statistical noise floor or a threshold
that transfers to another workload or allocation. Pending timings produce no
repeatability estimate. If a candidate's apparent gain is comparable to the
observed variation, collect more independent pairs on another allocation and
remeasure both members. A saved baseline build can be reused; its old timings
cannot replace the paired baseline measurements.

Nsight Compute records achieved/theoretical occupancy, register/shared-memory
limits, grid waves, compute/memory throughput, scheduler issue/eligible warps,
stall aggregates, memory traffic/cache behavior, and available roofline inputs.
Unavailable optional counters are explicit. DMMA counts identify tensor
instructions when available. Local memory traffic alone does not prove register
spilling. Nsight Systems keeps overlapping kernel, API and copy sums separate.
Profiler replay durations never replace unprofiled timing. Tool identities,
help/default policies, GPU identity and selected kernel coverage are retained.
Before/after driver, clock, power, temperature and ECC snapshots are endpoint
samples; they do not establish average operating conditions within solver runs.
High occupancy alone is not an optimization objective. See NVIDIA's
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) and
[CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

## Inspect, resume, and retrieve

```bash
./perf.sh status --run-dir /absolute/scratch/studies/my-change-01
./perf.sh report --run-dir /absolute/scratch/studies/my-change-01
./perf.sh resume --stage profiles --run-dir /absolute/scratch/studies/my-change-01
./perf.sh collect --run-dir /absolute/scratch/studies/my-change-01 \
  --destination /absolute/scratch/exports/my-change-01
```

Stages have independent state: `timings_complete`, `profiles_partial`, or
`analysis_complete`. Each attempt gets a new directory and exact command/exit
receipts with bounded process timeouts. Failed attempts and scheduler outcomes
remain inspectable. A profile retry preserves timings and recaptures both
members on the same fresh GPU UUID. Its UUID may differ from the timing
allocation. Captures from separate allocations are never mixed into a matched
cohort.

Use `resume --stage builds` after failed/partial build submission; incremental
caches and prior receipts remain. Active jobs cannot be duplicated. A completed
timing stage cannot be rerun in place; start a new study for new measurements.
`resume --stage timings` retries incomplete timings after both correctness jobs
pass. Source, executor, suite, binary, MFEM/Spack identities and completed
measurement hashes are rechecked before reuse. Changed inputs need a new study.

Dependent submissions use Slurm's `--kill-on-invalid-dep=yes`. For older pending
jobs, recovery releases only recorded jobs with `DependencyNeverSatisfied`, a
matching user/account/name/command, and a proven failed recorded predecessor.
The cancellation receipt is retained; running or unrelated jobs are protected.
Timing retries automatically enqueue a fresh dependent profiling job.

If a frozen measurement executor needs a fix, preserve that run and create a new
measurement run:

```bash
./perf.sh retry --from-run /absolute/scratch/studies/old-run \
  --run-dir /absolute/scratch/studies/retry-run \
  --timings-time 00:15:00 --profiles-time 00:30:00
```

Retry verifies the old immutable source/executor/suite,
matching completed build receipts and successful correctness jobs, and requires
every recorded job to be inactive. It retains the exact source/build paths and
suite by default, freezes the current measurement executor and build-artifact hashes, and
records the old plan as its origin. Fresh compute admission still checks live
dependency bindings. No old timing/profile samples or admission are copied;
both members are measured again. `--prepare-only` creates the new plan without
submitting; use `resume --stage timings` afterward. Limits inherit the old frozen
plan unless explicitly overridden, including when its jobs were later shortened
through the scheduler. The old run remains unchanged; partial submission receipts
belong to the new run and prevent an unnoticed duplicate retry.

When timings completed successfully and only the profiler executor needs a fix,
reuse those observations in a new profile-only run:

```bash
./perf.sh retry --from-run /absolute/scratch/studies/completed-timings \
  --run-dir /absolute/scratch/studies/profiles-retry \
  --profiles-only --prepare-only --profiles-time 00:30:00
```

Without a suite override, this mode preserves the exact suite bytes. It always
rejects `--cases`. To profile different kernels from the same measured workload,
add `--suite FILE`; only each existing case's `profile` subtree may change.
Case IDs, order, selection, parameters, suite name, repetitions, budgets and
correctness settings must remain identical after canonical suite validation
supplies omitted defaults. Explicit defaults in the frozen timing suite remain
authoritative, so changed tolerances or nondefault physics still reject.
The replacement suite is validated
and frozen byte-for-byte, and the plan records the original timing suite hash,
new profile suite hash and each changed profile setting.
It copies the completed timing summary unchanged, binds every original timing
artifact by path and hash, and verifies the retained source, binaries, build
receipts and successful correctness jobs. Keep the original run and its
artifacts available. Reports and status label these timings as reused; the new
run adds fresh profiles and cannot submit or execute a timing stage.

Without `--prepare-only`, retry submits only profiles. A prepared run can use
`resume --stage profiles`, or its frozen interactive wrapper with the `profiles`
stage. Capture admission requires the original physical A100 UUID and model;
a newly scheduled allocation on another GPU fails before capture. The original
timing allocation may remain active only when it is the owned free one-A100,
single-rank interactive shell, has no running attempt or recorded active batch
stage, and all build and correctness jobs completed successfully. The original
run's lock excludes concurrent workers during retry preparation. Ordinary retry
continues to require every recorded job to be inactive.

To keep successful tools after a partial profiling attempt, use a new immutable
continuation instead of recapturing them:

```bash
./perf.sh retry --from-run /absolute/scratch/studies/partial-profiles \
  --run-dir /absolute/scratch/studies/profiles-continuation \
  --profiles-only --continue-profiles --prepare-only --profiles-time 00:30:00
```

This mode preserves the exact suite and timing bytes and rejects `--suite` and
`--cases`. It seals each completed tool's checkpoint, application command/log
receipts and profiler identity; failed, pending or partial tools are collected
again. The worker checks retained native-report and export hashes on compute and
validates every retained application log against the original unprofiled work
before launching any missing tool. A completed Nsys checkpoint can therefore be
retained even when NCU failed before the old application gate. A member is complete
only when both Nsys and NCU are complete and its application logs pass.

The continuation may use another A100. Each tool retains its actual GPU UUID,
origin job and profiler version. Reports label these as independent diagnostic
captures, with no matched profiler timing or speedup claim; original unprofiled
paired timings remain the timing authority. Keep all referenced runs and files.
Without `--prepare-only`, only the missing-tool profiling stage is submitted;
otherwise use the new run's `resume --stage profiles` or frozen interactive
wrapper. Preparation requires inactive scheduler jobs and a closed attempt.
A hard scheduler kill that leaves the old attempt marked `running` is rejected;
this command does not repair old state. Ordinary profile-only retry retains its
original-GPU rule and recaptures both members.

The same explicit continuation can recover a run whose profiling admission
failed before any capture. It requires completed timings and no orphan profile
outputs. For this zero-capture case, optional `--profile-timeout-seconds 3600`
and `--profiles-budget-seconds 3600` set the new capture and stage budgets;
the original run and timing bytes remain unchanged. The new worker still
reserves 120 seconds before its actual allocation ends.

To inspect an admission mismatch, run
`./perf.sh diagnose-admission --run-dir /absolute/scratch/studies/old-run`
inside an owned compute allocation. It recomputes the frozen admission without
running applications or changing the study, and reports the first difference
separately for the legacy record and immutable artifact identity. Inherited
shell search-path tails are excluded from identity only when the selected
dependency prefixes and thread policy match; executable, library, source,
configuration and dependency changes still fail admission.

Use `retry --cases` to measure additional cases already present in the old
suite, including optional DGSEM probes, without submitting compilation or
correctness jobs:

```bash
./perf.sh retry --from-run /absolute/scratch/studies/old-run \
  --run-dir /absolute/scratch/studies/dgsem-volume-01 \
  --cases dgsem-n4,dgsem-n8 --prepare-only \
  --timings-time 00:15:00 --profiles-time 00:30:00
```

`retry --suite FILE` freezes a validated replacement suite; `--cases` then
selects from that suite. Its `correctness_regex` must exactly match the old
suite, because retry reuses those completed correctness results. Every selected
executable must already exist uniquely in each retained build and is newly
fingerprinted before any measurement submission. Missing or changed artifacts
fail without a build fallback. The new plan records both suite identities;
omitting both options preserves the old suite bytes exactly. Fresh compute
admission checks the selected binaries and dependency bindings again.

The DGSEM probe measures all ordinary GPU and tensor methods against an untimed
host long-double reference. Its selected GPU method and representative kernel
profiles do not supply CPU performance or full DGSEM time to solution. A study
claiming a CPU/GPU/tensor matrix must include explicit supported workload cases
and timing scopes for those backends; enabling these two probes alone is insufficient.

For interactive debugging, prepare the retry on the PACE login node without
submitting workers:

```bash
./perf.sh retry --from-run /absolute/scratch/studies/old-run \
  --run-dir /absolute/scratch/studies/interactive-retry \
  --prepare-only --timings-time 00:15:00 --profiles-time 00:30:00
```

This freezes the current executor and preserves the old run, source/build
identities and completed correctness results. In one free `gts-sbryngelson3` /
`embers` allocation on `gpu-a100`, with one rank and one allocated A100, invoke
the prepared run's frozen wrapper directly. Allow enough allocation time for
both stages; the workers still enforce their saved budgets and Slurm deadline.
Run profiles after timings succeeds:

```bash
run_dir=/absolute/scratch/studies/interactive-retry
bash "$run_dir/executor/pace_runtime.sh" "$run_dir/plan.json" timings "$run_dir/executor"
bash "$run_dir/executor/pace_runtime.sh" "$run_dir/plan.json" profiles "$run_dir/executor"
```

These commands submit no jobs. They use the same environment setup, allocation
and GPU checks, completed correctness checks, live dependency admission,
numerical/work gates and immutable attempt records as batch workers. Both stages
collect fresh observations; profiler replay does not replace timing samples.
Keep native captures and all trace processing on the compute node, retrieve only
the compact `collect` export, and release the interactive allocation when finished.

`plan.json` freezes the experiment, `admission.json` binds binaries, CMake/MFEM
artifacts, Spack locks and loader libraries, and `state.json` retains attempts.
`report.json` and `report.md` summarize accepted results. Logs, fields, native
`.nsys-rep`/`.ncu-rep`/`.ncu-repz`, full CSV exports and SQLite databases stay on
PACE. Native import and profiler summary generation run in the compute
allocation. `collect` copies a positive allowlist of compact records and
checksums. Transfer only that export directory to the local machine.

## Validation scope

Host contract tests are registered as
`igrlab_{gpu_study,study_build,study_workloads,study_profiles,study_retry}_python`. They
exercise real Git/MFEM snapshots, exact dirty bytes, cache relocation, canonical
submission, queue dependencies, literal argv, numerical/field gates, both Nsight
CSV layouts, compressed reports, unavailable metrics, timeouts, immutable
timings, duplicate-job rejection and compact collection.

Earlier PACE validation of the portable build driver and numerical changes is
documented in [pace.md](pace.md). That evidence alone does not validate a new
coordinator run. Prepared plans, host fixtures, submitted jobs, completed
timings and completed profiling are separate outcomes; check saved study state
before making a hardware performance claim.
