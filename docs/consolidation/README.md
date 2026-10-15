# IGRlab consolidation

The consolidation preserves the recovered numerical work and qualified A100
fixes in the active application, with research data in a verified
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

Validation also exposed two pre-existing correctness defects. The candidate
constructs the IGR solver at its final heap address so host coefficients retain
valid parameter references, and explicitly marks rebuilt CUDA operator maps
as host writes after mesh/space changes. Numerical settings and formulas are
unchanged. The tensor assembly check selects the exact `UseTensor=true`
specialization beside the retained SIMT specialization.

Checkpoint strings, durable schemas and MFEM identity stamps retain their
existing formats. The repository migration destination is
[`comp-physics/IGRlab`](https://github.com/comp-physics/IGRlab), with a clean
primary checkout named `IGRlab`. The consolidation integrates through `dev`
and then promotes that tree to default `main`, preserving merge history.
The archive's `MIGRATION.md` records the completed remote and physical checkout
steps, exact published revisions and compatibility paths.

The dirty original research checkout remains intact in a separate preserved
location. Its old pathname is a compatibility symlink for historical chats,
dependency views and existing build caches. New builds from the clean primary
use the canonical driver's fresh physical-path scope. Existing caches and
reviewed MFEM identity records retain their recorded paths and hashes.

M16 conservative modal damping is retained in the shared DG/DGSEM operator at
each Runge–Kutta stage. It remains opt-in, with the affine tensor-cell and
every-stage IGR admission rules described in the [method guide](../methods.md#conservative-modal-damping).
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
experiment passed 12/15 tests (three checks failed) and remains archived research.

## Recovery archive

The local recovery root is
`/home/jarias9/archives/igrlab/consolidation-20261010`. It contains the full
common Git object database, refs, stashes, reflogs and submodule metadata;
all inventoried tracked, untracked and ignored primary files; and all three original
detached worktrees, including dirty PACE and Sedov changes.

`recovery_state.json`, per-volume manifests and `SHA256SUMS` record verification
of every archived regular file and symlink. The raw Git archive also preserves
unreachable objects and stash untracked-file parents. The archive's README
and `restore_proof.json` describe the isolated restoration drill. The raw Git
snapshot contained 210 pre-existing empty loose blob files. Their exact bodies
were reconstructed from archived research files and verified against their
original Git IDs in a separate supplement. Restoration verifies all named
refs, stashes and historical reflogs with zero missing reachable objects.
The original database was repaired using those verified bytes while retaining
every displaced empty file; independent checks confirm that all inventoried
original working files, dirty patches and refs remain intact. MFEM working
files are reconstructed from the pinned archived submodule objects; external
dependency installations and bodies behind external symlinks are excluded
from the physical worktree volumes. Audit inventories,
repair receipts and validation logs are stored under `audit/`.

This archive is presently on the same host. A separate backup is advisable
before making it the sole retained copy of the primary research outputs.

The compact active tree retires 56 historical campaign/evidence/helper files (14.82
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
The committed proposal is a review snapshot pinned to the tested numerical
source. The final candidate HEAD and refreshed approval guards are recorded
separately under the recovery archive’s `audit/planning-git/` directory,
avoiding a self-referential commit hash in the committed proposal.

The first batch retains main/dev, the original investigation branch, this
candidate, both stashes, six historical archive refs, twelve unmatched
experiment refs and the historical remote DGSEM ref. The distinct unmatched
patches were reviewed and preserved with provenance; none supplies a missing
required backend. Their numerical-policy and benchmark ablations are not
promoted automatically.

The approved first cleanup batch completed on October 10, 2026: 39 local
branches and 61 remote branches were retired, three old worktrees were removed
through Git, and PRs #37 and #38 were closed without merging. The recovery
archive's `CLEANUP.md`, execution receipts and post-cleanup checksum seal record
the completed actions and verification. The committed proposal above remains
the original review snapshot. The original primary research files remain
preserved until their retirement is separately approved.

## Validation

Current-source validation results, configurations and remaining limitations
are recorded in [validation.md](validation.md). A print plan verifies build
policy; compilation verifies source compatibility; device execution verifies
only the cases actually run. New A100 device execution requires an authorized
A100 allocation and is not inferred from local SM120 tests.
