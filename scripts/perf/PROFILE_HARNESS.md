# Scenario-driven profiling harness

`profile_harness.py` is a codebase-wide runner, not a benchmark policy. The
caller owns the scenario label, executable, application arguments, output
directory, and every profiling selection. It supplies no DG, DGSEM, HDG, IGR,
CUDA, MPI, mesh, order, time-step, or physics default.

## Levels and outcomes

- `--level l0` runs exactly one unprofiled command.
- `--level l1 --warmups 1 --repetitions N` performs one warm-up then five to
  seven measured repetitions, recording raw durations, median, and population
  standard-deviation dispersion.
- `--level l2-nsys` requires `--tool-executable`, `--nsys-trace`,
  `--nsys-sample`, and at least one `--require-data NAME=RELATIVE_PATH`.
- `--level l2-ncu` requires `--tool-executable`, explicit
  `--ncu-target-processes application-only|all`, an explicit `--ncu-set` or
  `--ncu-section`, and at least one `--require-data NAME=RELATIVE_PATH`.

L2 command construction preserves requested trace/sample or metric/filter
choices literally. `--ncu-kernel-name`, `--ncu-kernel-name-base`, launch
limits, and NVTX include/exclude options are opt-in filters only. A report or
table is admitted only when the caller names its expected relative path with
`--require-data`; absent or empty data yields `data_unavailable` (exit 4), not
a pass or skip. A dry run is the distinct `dry_run` outcome; a missing profiler
is `tool_unavailable` (exit 3), a non-zero process or launch `OSError` is
`execution_failure` (exit 5), and invalid provenance is `validation_failure` (exit 6). Invalid inputs
use argparse exit 2.

Outputs are confined to a newly-created `--out-dir`; path traversal and
absolute requested-data paths are rejected. Harness-owned
`profile-provenance.json`, `execution.log`, and `profile-manifest.json` are
reserved from requested data and use exclusive no-follow creation. Required
data must be a non-empty regular file with no symlink path component and must
resolve under the output directory. The provenance sidecar is validated and
integrity-pinned before execution, then revalidated unchanged afterward; any
deletion, replacement, symlink, or byte/schema change is a validation failure.
Final manifest provenance fields come solely from that initial pin.
`profile-provenance.json` reuses
the accepted v1 provenance writer/validator. `profile-manifest.json` is sorted
JSON containing deterministic artifact paths, byte counts, SHA-256, generating
command, supplied tool/executable/MFEM/GPU identities, executable byte count
and SHA-256, provenance-derived environment availability facts, and the
provenance dirty-state fact. It does not hash itself. Explicit executable,
output, source-root, and slash-containing profiler paths are resolved before
the runner changes its working directory.

The runner requires available reduced-provenance facts for `source.root`,
`source.revision`, `source.tree`, `source.status`, and `capture.output_stem`.
For a present L2 tool it also requires `tool.executable`. Manifest environment
facts are read only from successfully validated provenance; a provenance-
validation failure records no environment values.

Example dry-run (no profiler is invoked):

```bash
python3 scripts/perf/profile_harness.py \
  --level l2-nsys --scenario operator-window --exe /path/to/app \
  --out-dir /tmp/operator-window --tool-executable /path/to/nsys \
  --nsys-trace cuda,nvtx,osrt --nsys-sample none \
  --require-data nvtx-summary=nvtx-summary.csv --dry-run
```

Use `--environment OMP_NUM_THREADS` (or another non-sensitive explicit key)
to record an environment input. The provenance denylist rejects secret-bearing
environment component names. The harness does not invoke MPI, query hardware,
choose a report/table, or certify performance/scientific results.
