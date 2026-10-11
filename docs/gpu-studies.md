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
cache root. The key binds exact source/MFEM content, build flags, and site
configuration. Identical inputs reuse the same snapshot and canonical build
directory across studies. Differing configurations share dependencies.
Identical baseline/candidate builds are submitted once. Leave cache entries in
place while their studies are retained; the coordinator never silently cleans
or replaces them.

Add `--prepare-only` to freeze inputs and inspect saved build plans without
submitting jobs. Submit that prepared study with `resume --stage builds`.
Preparation performs Git/file operations and `build.sh --print-plan`; it does
not compile on the login node.

## The optimization loop

1. Choose a baseline and an unchanged numerical workload. An identical-source
   control can quantify repeatability.
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
Use `--cases dg-p1,dg-p3` for a targeted study, or `--suite FILE` for a saved,
versioned workload variant. An explicit case selection enables optional cases.
Do not change suite parameters between comparison members.

| Adapter | Primary timing | Numerical/work gate | Diagnostic scope |
|---|---|---|---|
| DG | Fixed-horizon full-solver time | Accepted steps/RHS work, zero rejects, NaN scan, geometry and five physical fields | Changed mass kernels and whole-run timeline |
| HDG | Accepted-step median per process | Schema-2 policies/gates, zero retries/fallback, same work, separate exported checksums | Measured step zero; volume, condensation, face, recovery and trace-solve families |
| DGSEM, optional | Selected volume method time | Complete method/error observations against the existing long-double reference | Prescribed-sigma central-volume probe only |

HDG's five steps within one process produce one observation; they are not five
independent repetitions. Baseline/candidate process order alternates. Reports
retain raw observations, medians, paired percentage changes/ranges and relative
standard deviations. There is no automatic speed threshold or significance
claim. Ordinary and tensor engines have different trace policies: each case
compares the same engine across source versions. DGSEM establishes neither
full-solver accuracy nor full-solver speedup.

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

`plan.json` freezes the experiment, `admission.json` binds binaries, CMake/MFEM
artifacts, Spack locks and loader libraries, and `state.json` retains attempts.
`report.json` and `report.md` summarize accepted results. Logs, fields, native
`.nsys-rep`/`.ncu-rep`/`.ncu-repz`, full CSV exports and SQLite databases stay on
PACE. Native import and profiler summary generation run in the compute
allocation. `collect` copies a positive allowlist of compact records and
checksums. Transfer only that export directory to the local machine.

## Validation scope

Host contract tests are registered as
`igrlab_{gpu_study,study_build,study_workloads,study_profiles}_python`. They
exercise real Git/MFEM snapshots, exact dirty bytes, cache relocation, canonical
submission, queue dependencies, literal argv, numerical/field gates, both Nsight
CSV layouts, compressed reports, unavailable metrics, timeouts, immutable
timings, duplicate-job rejection and compact collection.

Earlier PACE validation of the portable build driver and numerical changes is
documented in [pace.md](pace.md). That evidence alone does not validate a new
coordinator run. Prepared plans, host fixtures, submitted jobs, completed
timings and completed profiling are separate outcomes; check saved study state
before making a hardware performance claim.
