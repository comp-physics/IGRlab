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

- `sedov_taylor_blast__o_3__rs_0__rp_1__tf_1p0__dt_save_0p01__c_0p25__0002.log`

This makes `summary.json` and downstream render grouping easier to inspect.

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
