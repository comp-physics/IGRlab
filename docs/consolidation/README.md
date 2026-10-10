# IGRlab consolidation

The consolidation preserves the recovered numerical work and qualified A100
fixes in an isolated application branch, with research data in a verified
recovery archive. DG, DGSEM and HDG retain their CPU, CUDA and specialized
manual tensor implementations. Research executables are available explicitly
through `--research-tests`; ordinary correctness tests remain enabled.

## Source and compatibility

The recovered source starts from investigation HEAD
`1c75579a94bca0d067aedd44c26aff13c5831b26`, including its dirty files and
untracked production/test dependencies. The candidate also contains the live
`dev` integration history through
`db4d2ae1af3f55fe87f3b0360804d4834169a691` without replacing the newer recovered
source with older code. The MFEM submodule remains pinned to
`d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec`.

Code/build branding is IGRlab. Both global and MFEM-scoped `dg_igr` namespace
aliases remain usable. `IGRlab::core` aliases the existing library target;
legacy executable, CTest, cache and environment spellings remain supported.
Runtime `IGRLAB_*` values take precedence even when empty. Build-driver
aliases use nonempty canonical values and then explicit CLI options. CMake
cache aliases accept either spelling and reject contradictory edits.

Checkpoint strings, durable schemas and MFEM identity stamps retain their
existing formats. Repository and checkout names are a separate final
migration, so the GitHub origin is still `comp-physics/dg_igr_miniapp`.

M16 conservative modal damping is retained in the shared DG/DGSEM operator at
each Runge–Kutta stage. It remains opt-in, with the affine tensor-cell and
every-stage IGR admission rules described in the [main README](../../README.md).
The original one-dimensional study showed a useful mitigation; it does not
establish elimination of ringing in every application or geometry.

## A100 preservation

The candidate includes native SM80 policy for dependencies, MFEM and the app;
dependency artifact/ABI checks; site-compatible static HYPRE and library
layout handling; CUDA host-compiler/template workarounds; the opaque CUDA
engine host facade; grid-constant trace parameters; face scheduling fixes;
and the TF32 rounding correction. See [A100 support](../a100.md).

The September A100 records distinguish qualified source from an unfinished
p4 experiment. The qualified jobs recorded 9/9 selected tests and a clean p1
trace memcheck, plus 80/80 admissible horizons to `t=0.1`. Those are historical
results tied to their original source and software tuple. The dirty PACE p4
experiment failed 12/15 tests and remains archived research.

## Recovery archive

The local recovery root is
`/home/jarias9/archives/igrlab/consolidation-20261010`. It contains the full
common Git object database, refs, stashes, reflogs and submodule metadata;
all tracked, untracked and ignored primary files; and all three original
detached worktrees, including dirty PACE and Sedov changes.

`recovery_state.json`, per-volume manifests and `SHA256SUMS` record verification
of every archived regular file and symlink. The raw Git archive also preserves
unreachable objects and stash untracked-file parents. The archive's README
and `restore_proof.json` describe the isolated restoration drill. Audit
inventories and validation logs are stored under its `audit/` directory.

This archive is presently on the same host. A separate backup is advisable
before making it the sole retained copy of the primary research outputs.

The compact active tree retires 55 historical campaign/evidence files (14.82
MiB) from the candidate, retaining their exact bytes in the recovery volume
and Git history. [archived-paths.json](archived-paths.json) lists original paths
and checksums. All 69 research source files and every recovered dependency
remain available, including opt-in probes.

## Reviewable cleanup

[The exact cleanup proposal](cleanup-proposal.md) lists guarded ref SHAs and
worktree states for 39 local branches, 61 remote branches, three old
worktrees, and superseded PRs #37 and #38. Both PR patches already exist in
the recovered source; current communicator and collective-admission tests
must pass before closure. Close the PRs before retiring their head/base refs.

The first batch retains main/dev, the original investigation branch, this
candidate, both stashes, six historical archive refs, twelve unmatched
experiment refs and the historical remote DGSEM ref. The distinct unmatched
patches were reviewed and preserved with provenance; none supplies a missing
required backend. Their numerical-policy and benchmark ablations are not
promoted automatically.

Deletion, PR closure, publication, merge and the repository/checkout migration
are final approval steps. Before deletion, refresh the expected SHA and exact
dirty/untracked state for every affected object. Any changed object is
recaptured first. The original primary directory remains preserved until its
research retirement is separately approved.

## Validation

Current-source validation results, configurations and remaining limitations
are recorded in [validation.md](validation.md). A print plan verifies build
policy; compilation verifies source compatibility; device execution verifies
only the cases actually run. New A100 device execution requires an authorized
A100 allocation and is not inferred from local SM120 tests.
