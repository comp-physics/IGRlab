# DeltaAI job-local output

DeltaAI compute nodes provide a shared 3.9 TB SSD at `/tmp`, purged after each
job ([NCSA storage guide](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html)).
Use `scripts/perf/deltaai_job_local.py` inside an authorized one-node Slurm job.
It creates a private directory under `/tmp` or a `SLURM_TMPDIR` that resolves
under `/tmp`, requires 32 GiB free by default, and never falls back to `/projects`.
The required new `--results-dir` must resolve outside `/tmp`, so retained
results survive the job's purge. For a results parent under `/projects`, the
wrapper requires its numeric project ID and inheritance flag `P` from
`lfs project -d`, then checks
`lfs quota -p ID /projects` before starting the command. Current project usage
plus the entire retention cap must fit the smaller positive soft/hard block
limit; retained files and directories must also fit positive inode limits.
Missing, failed, or ambiguous quota queries stop the run. Filesystem-wide
free space is not a substitute for this project quota check, and the check
does not reserve quota against concurrent writers.

For a future run, after loading its normal runtime environment:

```bash
python3 /absolute/checkout/scripts/perf/deltaai_job_local.py \
  --results-dir /projects/ACCOUNT/USER/results/new-job-id \
  --keep vtk/case/Cycle000005/data.pvtu \
  --keep vtk/case/Cycle000005/proc000000.vtu \
  -- /absolute/frozen/application <case-arguments>
```

Replace `<case-arguments>` with the application arguments, and select the
actual completed final frame and its metadata explicitly. The VTK prefix
is set by the wrapper; the application does not need an output-path argument. A command may use `{run}` and `{scratch}` in its
arguments. Its working directory, temporary-file variables, canonical and
legacy VTK prefixes, build output-root environment variables, and
`IGRLAB_JOB_OUTPUT_DIR` point into scratch. Conventional `--output`,
`--output-root`, `--run-dir`, and `--out-dir` arguments outside scratch are
rejected. This is a storage wrapper, not a filesystem sandbox: commands and
scripts must honor their output arguments and must not contain persistent
output paths. It does not submit jobs or grant GPU/build authorization.

Only `stdout.log`, `stderr.log`, and a compact `run.json` are retained by
default. Repeat `--keep` for exact files relative to `{run}`; directories,
globs, missing files, escaping paths, and incomplete VTK/XML footers are
rejected. Retention is capped at 4 GiB including the receipt; use explicit
`--max-retained-gib` and `--min-free-gib` limits when appropriate. Copies are
checked by size and SHA256 before a new results directory is published by
atomic rename. Every retention level includes the serialized receipt in its
budget. Invalid or oversized selections fall back to logs only, then a
receipt only if needed; storage failures make the wrapper fail. XML footer checks
detect truncated writes and do not replace full VTK/field validation.

Normal exit, failure, TERM, INT, and USR1 stop the owned child process group
before retention and cleanup of only the private scratch directory. Reserve
time for copying selected results, for example with a Slurm early USR1 signal.
SIGKILL and node loss bypass this cleanup; site purging is the fallback.
Transient output still needs capacity: twelve 1.16 GB frames occupy about
13.9 GB even when only two finals are retained.

Historical campaign launchers hard-code `/projects` output paths and absolute
paths in qualification gates. They are retired for new jobs and must not be
reused unchanged with this wrapper. Existing code, golden finals, and pinned
inputs are not automatically moved or rewritten. Preserve original receipts;
any archive relocation needs an explicit path/hash mapping and compatible
readers. A fresh canonical `build.sh --output-root '{scratch}/builds'` requires
initial configuration before `--iterate` can be used there.

This wrapper has host-only filesystem validation. Its node capacity,
performance, and cluster runtime have not been tested by a GPU job.
