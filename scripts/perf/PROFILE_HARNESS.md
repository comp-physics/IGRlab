# Scenario-driven profiling harness

`profile_harness.py` is a codebase-wide runner, not a benchmark policy. The
caller owns the scenario label, executable, application arguments, output
directory, and every profiling selection. It supplies no DG, DGSEM, HDG, IGR,
CUDA, MPI, mesh, order, time-step, or physics default.

## Levels and evidence modes

- `--level l0` runs exactly one unprofiled command. `--evidence-mode` is not
  accepted.
- `--level l1 --warmups 1 --repetitions N` performs one warm-up followed by
  five to seven measured repetitions, retaining the raw durations, median, and
  population-standard-deviation dispersion. `--evidence-mode` is not
  accepted.
- `--level l2-nsys` requires an explicit profiler executable, trace set,
  sampling mode, evidence mode, and at least one
  `--require-data NAME=RELATIVE_PATH`.
- `--level l2-ncu` requires an explicit profiler executable, target-process
  policy, one or more sets/sections/metrics, evidence mode, and at least one
  `--require-data NAME=RELATIVE_PATH`.

Every L2 invocation must choose exactly one of these contracts:

- `--evidence-mode smoke` preserves the caller's profiler and application
  choices literally. It may use a fake tool and may be a dry run. Success means
  that the command completed and every named nonempty artifact exists; it does
  not validate a native profiler report, an exact candidate, tensor
  instructions, or profiler-table semantics.
- `--evidence-mode qualification` is fail-closed promotion evidence. It
  implies exact-candidate binding, cannot be a dry run, and imposes the common
  identity and native-report checks below. NCU additionally requires one exact
  `--ncu-qualification-kind` described in the next section. Nsys qualification
  validates identity and a native report, but deliberately does not claim
  quantitative kernel or NVTX table semantics.

L2 command construction preserves explicit trace/sample or metric/filter
choices. In smoke mode, `--ncu-kernel-name`,
`--ncu-kernel-name-base`, launch limits, and NVTX include/exclude options are
ordinary opt-in filters. In qualification mode, NCU accepts only its exact
condensation-instruction, volume-instruction, or accepted-step-attribution
selection.

## Common qualification binding

Qualification requires all of the following, and revalidates them after the
profiled process exits:

- `--source-root`, `--expected-commit`, `--expected-tree`, and
  `--require-clean-source` identify one unchanged clean Git candidate;
- `--build-manifest` is that candidate's `CMakeCache.txt`, the profiled
  executable is below its build directory, and the cache records Release,
  application CUDA architecture 120, HDG runtime enabled, observability
  enabled, and NVTX enabled;
- `--mfem-identity` is the canonical GPU/`cuda-sm120` identity, with dependency
  CUDA architecture 120, MFEM `80-virtual`, HYPRE `cuda_arch=120`, and an
  `MFEM_DIR` consistent with the application cache;
- `--tool-executable` resolves to `ncu` or `nsys` as appropriate, its regular
  file identity is retained, its live `--version` banner is an NVIDIA banner,
  and that exact output equals `--tool-version`;
- `--gpu-index`, `--gpu-identity`, and `--nvidia-smi-executable` resolve one
  live compute-capability-12.0 GPU whose UUID matches the supplied identity;
  `CUDA_VISIBLE_DEVICES` must select only that index or UUID and must be named
  with `--environment CUDA_VISIBLE_DEVICES`;
- the requested report is exactly
  `--require-data report=capture.ncu-rep` or
  `--require-data report=capture.nsys-rep`, is a confined nonempty regular
  file, and has the corresponding native report signature.

The executable, source, cache, MFEM identity, profiler executable/version, GPU
identity, native report, and derived NCU evidence artifacts are pinned and
checked again before a success manifest is written. Any changed or unavailable
binding is `validation_failure`, not partial qualification.

## NCU qualification modes

The three NCU qualification kinds are intentionally narrower than general
profiling. Each requires the production
`dg_igr_hdg_runtime_cuda_accepted_step_benchmark`, with exactly these
application semantics:

- `--engine tensor`, `--trace-policy engine-default`,
  `--mode conventional|igr`, `--order 1|2|3`, a valid explicit periodic
  `--mesh-n`/`--domain-periods`, positive `--dt`, one
  `--profile-measured-step 0|1|2|3|4`, and exactly one `--no-export`;
- application-side `--expected-source-revision`,
  `--expected-source-tree`, `--expected-source-status clean`,
  `--expected-build-directory`, and `--expected-cmake-cache` exactly match the
  harness-side candidate and cache;
- target processes are exactly `application-only`, kernel-name base is
  `demangled`, no set or section is accepted, and `--ncu-launch-count` is
  positive. Qualification requires `--ncu-launch-skip` to remain at its zero
  default so the declared launch count always covers the complete selected
  range.

Profiler qualification remains bound to the tensor engine's production
`engine-default` trace policy. The accepted-step benchmark's
`--trace-policy common-fast` option exists only for the separate bounded
`hdg_tensor_campaign.py --comparison-mode same-policy-attribution` ablation.
That O_base/O_common/T_common result is provisional and cannot become profiler
or promotion qualification evidence. A diagnostic smoke capture may profile
it when the caller supplies the option explicitly, but it does not change the
NCU/SASS proof required for the production engine package. The ablation keeps
only bounded benchmark samples and aggregate policy/work metadata; neither the
campaign nor this profiler harness is a flight recorder or replay facility.

`--ncu-qualification-kind tensor-instructions` is the condensation native-
instruction proof. Its only metrics are
`sm__inst_executed_pipe_fp64_op_dmma.sum` and
`gpu__time_duration.avg`; its kernel filter is exactly
`regex:.*TensorBorrowedCondenseKernel.*`; and its only NVTX include is
`HDG\/condensation/`. After capture, the harness imports the native report
itself with
`ncu --import REPORT --csv --page raw --print-units base`. Explicit base units
prevent Nsight Compute release defaults from rescaling the duration metric. The
imported table must
contain exactly the requested number of `TensorBorrowedCondenseKernel`
launches, no numeric rows from another kernel, and positive FP64 DMMA and
GPU-duration values for every launch. The harness also parses the benchmark's
one finite schema-2 summary from the existing execution log and requires the
selected measured sample to be the production tensor engine on supported
hardware with zero fallback. The caller's launch count must equal that sample's
condensation-launch count. An independent test-side shape calculation splits
the planned work into four positive scalar totals—blocked-LU trailing updates,
blocked forward-RHS updates, blocked backward-RHS updates, and Schur
contraction—and their sum must equal both the sample's MMA-tile total and the
NCU DMMA total, with the complete per-launch total observed on every selected
launch. The condensed RHS shares a padded Schur edge tile only when it fits;
otherwise its final contraction is scalar. This closes prefix-truncation and
partial-work ambiguities while preserving the fused production kernel boundary.

`--ncu-qualification-kind tensor-volume-instructions` is the corresponding
volume-contraction proof. It uses the same two metrics, the exact
`regex:.*BorrowedEulerTensorVolumeContractionKernel.*` filter, and only the
`HDG\/local-assembly/` NVTX range. The benchmark summary supplies the bounded
aggregate `volume_assembly_launches` and `volume_mma_tiles`; the former counts
successful production assembly invocations, not contraction-kernel launches.
The harness independently reconstructs the production coefficient-workspace
capacity, deterministic quadrature chunks (including a zero-padded final
tail), contraction-launch count, and FP64 `m8n8k4` work from mode, order, mesh,
and element shape. It then requires the imported NCU rows to match the exact
total and complete chunk-shape distribution. The manifest retains only a
constant-size chunk-shape distribution and DMMA histogram, not ordered or
per-launch history.

`--ncu-qualification-kind accepted-step-attribution` is the all-kernel
accepted-step view. Its only metric is `gpu__time_duration.avg`; it forbids a
kernel filter; and its only NVTX include is
`HDG\/profiled-accepted-step/`. Its report import also requests
`--print-units base`. The imported raw table must expose one
unambiguous NVTX push/pop column, nanosecond durations, and at least one row but
strictly fewer rows than the declared launch limit, all inside that outer
range. Reaching the limit is treated as possible truncation, not a complete
capture. It requires positive
material rows for local assembly, condensation, trace assembly, trace solve,
interior recovery, reductions, validation, packing/conversion, and nonlinear
update, plus positive `TensorBorrowedCondenseKernel` and
`BorrowedEulerTensorVolumeContractionKernel` rows. The manifest retains
per-category and per-kernel launch/duration totals, fractions of captured GPU
kernel duration, separate tensor-condensation and tensor-volume fractions, and
any unclassified outer-range work. Both tensor-kernel launch counts must equal
their independently derived selected-sample counts. Local factorization, device
transfer, and synchronization
categories are retained when present but are not mandatory positive categories
for every capture.

For every qualification kind, the raw report, imported CSV, import stderr,
commands, sizes, SHA-256 identities, and the fixed selected-step tensor-work
summary are
retained. This is bounded post-run validation of one finite benchmark log; it
does not add callbacks, event capture, replay, or production instrumentation.
The accepted-step attribution contains
GPU kernel duration only; CUDA API and memcpy time require Nsys, and host/MPI
time is not included. Its positive tensor-kernel duration is not instruction
proof, so promotion requires separate condensation and volume instruction
captures. A positive application MMA-tile counter or a tensor-capable library
call is not substituted for either proof.

An exact tensor-instruction qualification invocation has this shape (shell
variables denote already-resolved exact paths or identities):

```bash
python3 scripts/perf/profile_harness.py \
  --level l2-ncu --evidence-mode qualification \
  --scenario igr-p3-condensation-tensor-proof \
  --exe "$BUILD/tests/hdg/runtime/cuda/dg_igr_hdg_runtime_cuda_accepted_step_benchmark" \
  --out-dir "$OUT" --source-root "$SOURCE" \
  --expected-commit "$COMMIT" --expected-tree "$TREE" \
  --require-clean-source --build-manifest "$BUILD/CMakeCache.txt" \
  --mfem-identity "$MFEM_IDENTITY" \
  --gpu-index 0 --gpu-identity "$GPU_UUID" \
  --nvidia-smi-executable /usr/lib/wsl/lib/nvidia-smi \
  --environment CUDA_VISIBLE_DEVICES \
  --tool-executable "$NCU" --tool-version "$NCU_VERSION" \
  --ncu-target-processes application-only \
  --ncu-qualification-kind tensor-instructions \
  --ncu-metric sm__inst_executed_pipe_fp64_op_dmma.sum \
  --ncu-metric gpu__time_duration.avg \
  --ncu-kernel-name-base demangled \
  --ncu-kernel-name 'regex:.*TensorBorrowedCondenseKernel.*' \
  --ncu-launch-count "$CONDENSATION_LAUNCHES" \
  --ncu-nvtx-include 'HDG\/condensation/' \
  --require-data report=capture.ncu-rep \
  --arg=--mode --arg igr --arg=--engine --arg tensor \
  --arg=--trace-policy --arg engine-default \
  --arg=--order --arg 3 --arg=--mesh-n --arg 52 \
  --arg=--domain-periods --arg 1 --arg=--dt --arg 1e-5 \
  --arg=--profile-measured-step --arg 2 --arg=--no-export \
  --arg=--expected-source-revision --arg "$COMMIT" \
  --arg=--expected-source-tree --arg "$TREE" \
  --arg=--expected-source-status --arg clean \
  --arg=--expected-build-directory --arg "$BUILD" \
  --arg=--expected-cmake-cache --arg "$BUILD/CMakeCache.txt"
```

`CONDENSATION_LAUNCHES` is taken from an unprofiled exact-candidate run of the
same accepted-step selection. Qualification then checks it again against the
profiled sample and fails if nonlinear work changed or the capture was only a
prefix.

The volume instruction capture uses the same exact candidate and application,
replacing only the NCU selection with:

```bash
--ncu-qualification-kind tensor-volume-instructions \
--ncu-target-processes application-only \
--ncu-metric sm__inst_executed_pipe_fp64_op_dmma.sum \
--ncu-metric gpu__time_duration.avg \
--ncu-kernel-name-base demangled \
--ncu-kernel-name 'regex:.*BorrowedEulerTensorVolumeContractionKernel.*' \
--ncu-launch-count "$VOLUME_CONTRACTION_LAUNCHES" \
--ncu-nvtx-include 'HDG\/local-assembly/' \
--require-data report=capture.ncu-rep
```

`VOLUME_CONTRACTION_LAUNCHES` is derived from the unprofiled sample's successful
assembly-invocation count and the exact quadrature chunk plan, then checked
again against both the profiled sample and the native report.

The accepted-step attribution uses the same exact candidate, benchmark, GPU,
application, report, and tool bindings, replacing only the NCU selection with:

```bash
--ncu-qualification-kind accepted-step-attribution \
--ncu-target-processes application-only \
--ncu-metric gpu__time_duration.avg \
--ncu-kernel-name-base demangled \
--ncu-launch-count "$CAPTURE_LIMIT" \
--ncu-nvtx-include 'HDG\/profiled-accepted-step/' \
--require-data report=capture.ncu-rep
```

## Nsys qualification and WSL limitation

Nsys qualification uses the same exact source/build/MFEM/tool/GPU binding and
requires a native `capture.nsys-rep`. It also requires the same production
accepted-step benchmark and exact tensor/no-export mode, order, mesh,
timestep, profiled step, source, and build application arguments as NCU
qualification. Qualification fixes exactly `--nsys-trace cuda,nvtx`,
`--nsys-sample none`, `--nsys-capture-range cudaProfilerApi`, and
`--nsys-capture-range-end stop`, so the native report is scoped to the selected
accepted step. From that native report the harness force-exports and hash-binds
separate `nvtx_sum`, `cuda_api_sum`, `cuda_gpu_mem_time_sum`, and
`cuda_gpu_mem_size_sum` CSV tables, their stderr files, and every extraction
command. Nonempty native-tool output and unchanged identities are required;
the descriptive tables are not reinterpreted as a tensor-instruction or
numerical-acceptance proof.

On the authorized WSL path, Nsys is useful for host, NVTX, CUDA API, and
transfer structure but does not provide a usable GPU-kernel timeline. NCU is
the quantitative kernel-duration and tensor-instruction authority. A
successful native Nsys qualification report must not be interpreted as tensor
instruction proof or complete kernel-time attribution.

A smoke dry run, which intentionally has none of the qualification meaning
above, can be constructed as follows:

```bash
python3 scripts/perf/profile_harness.py \
  --level l2-nsys --evidence-mode smoke \
  --scenario operator-window --exe /path/to/app \
  --out-dir /tmp/operator-window --tool-executable /path/to/nsys \
  --nsys-trace cuda,nvtx,osrt --nsys-sample none \
  --nsys-capture-range nvtx --nsys-capture-range-end repeat::defer \
  --nsys-nvtx-capture accepted-step@dg-igr \
  --require-data nvtx-summary=nvtx-summary.csv --dry-run
```

## Outcomes, confinement, and manifest

A report or table is admitted only when the caller names its expected relative
path with `--require-data`. Absent or empty data yields `data_unavailable`
(exit 4), not a pass or skip. A dry run is the distinct `dry_run` outcome; a
missing profiler is `tool_unavailable` (exit 3), a nonzero process or launch
`OSError` is `execution_failure` (exit 5), and invalid identity or evidence is
`validation_failure` (exit 6). Invalid inputs use argparse exit 2.

Outputs are confined to a newly created `--out-dir`; path traversal and
absolute requested-data paths are rejected. Harness-owned
`profile-provenance.json`, `execution.log`, `profile-manifest.json`, and the
derived NCU evidence files are reserved. Required data must be a nonempty regular
file with no symlink path component and must resolve under the output
directory. The provenance sidecar is validated and integrity-pinned before
execution, then revalidated unchanged afterward.

`profile-manifest.json` is sorted JSON containing deterministic artifact paths,
byte counts, SHA-256 identities, generating commands, executable/build/MFEM/
tool/GPU bindings, environment availability facts, source dirty-state facts,
the qualification disposition, the fixed profiled-step tensor-work summary,
and any L0/L1 timings. It does not hash itself.
Explicit executable, output, source-root, and slash-containing profiler paths
are resolved before the runner changes its working directory.

Use `--environment OMP_NUM_THREADS` (or another non-sensitive explicit key) to
record an environment input. The provenance denylist rejects secret-bearing
environment component names. The harness does not acquire a GPU lease, invoke
MPI, choose hardware or profiler policy, or certify performance/scientific
results.
