# Parametric Runner

Run multiple case executables with per-case parameter sweeps, sequentially.

## Quick Start

```bash
python3 scripting/runner/parametric_runner.py \
  --config scripting/runner/example_cases.json \
  --dry-run
```

Run for real:

```bash
python3 scripting/runner/parametric_runner.py \
  --config scripting/runner/example_cases.json
```

The runner writes logs + summary JSON under `scripting/runner/out/<timestamp>/`.

Run IDs (and log filenames) now include sweep metadata, not just a sequence
number. Example shape:

- `sedov_taylor_blast__o_3_rs_0_rp_1_tf_1.0_dt_save_0.01_c_0.25__0002.log`

This makes `summary.json` and downstream render grouping easier to inspect.

## Merge Multiple Summary Files

If you ran sweeps in separate batches and want one post-process pass, merge
their `summary.json` files first:

```bash
python3 scripting/runner/merge_summaries.py \
  --summary scripting/runner/out/<runA>/summary.json \
  --summary scripting/runner/out/<runB>/summary.json \
  --out scripting/runner/out/merged_summary.json
```

Defaults:

- duplicate handling is `--dedupe run-id` (last input wins on duplicate run_id)
- merged runs include `summary_source` so provenance is preserved

Alternative dedupe modes:

- `--dedupe none`: keep all runs
- `--dedupe run-id-log`: consider both run_id and log_path

## Config Schema (JSON)

- `schema_version`: currently `1`
- `defaults` (optional):
  - `launcher`: command prefix, e.g. `["mpirun","-np","1"]`
  - `base_args`: args appended to every case
  - `workdir`: working directory for runs
  - `env`: environment variable map
  - `keep_going`: keep running after failures
- `cases`: list of case entries:
  - `name`: unique case name
  - `executable`: path to executable
  - `launcher` (optional): additional launcher prefix
  - `base_args` (optional): additional case-level args
  - `workdir` (optional): case working directory
  - `env` (optional): case environment overrides
  - `sweep` (optional): option -> list of values (cartesian product)

### Sweep Value Rules

- `true`: include flag only (no value), e.g. `"-foo": [true]`
- `false` or `null`: omit that option for that run
- scalar (`string`, `number`): emit `option value`
- list: emit `option` followed by all items in that list

## Useful CLI Options

- `--only-case <name>`: run only selected case(s), repeatable
- `--env KEY=VALUE`: inject/override env vars, repeatable
- `--keep-going`: continue after run failures
- `--out-dir <path>`: custom output directory
- `--dry-run`: print and record commands without executing
