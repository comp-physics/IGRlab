# CPU binding to the frozen shared two-phase runtime ABI

This leaf deliberately defines only CPU-qualified phase, configuration,
evidence, injection, and affine facade-handle types. It does not define the
canonical `ConventionalEulerSmoothPersistentV1Tag`, a common `FailurePhase`,
or cross-backend run/evidence DTOs.

The facade is source-bound to the accepted `PersistentRunContract` bytes at
commit `c87984feab42a519b7bb4e186e0c224c58a2f71f`. Its narrow bind points are:

1. numerical completion and ordered product issuance precede
   `FreezeCandidate`;
2. fallible checkpoint/output staging precedes `PrepareCommit`;
3. the facade retains the move-only `StagingToken`, `PreparedCommitToken`, and
   `ProvisionalCommitToken` internally and exposes only move-only numeric CPU
   handles with owner-incarnation and candidate-generation identity;
4. `CommitPrepared` performs the reversible whole-slot exchange while the old
   logical publication remains visible;
5. `FinalizeCommit` is the sole logical success transition;
6. `AbortPrepared` rolls back without exchange, and `RevertCommit` invokes the
   exact self-inverse exchange before ordinary rollback.

No additional shared commit primitive is requested. The only future common
ABI request remains a coordinator-owned phase/disposition vocabulary so CPU,
CUDA, CLI, and I/O can share spellings without duplicating unqualified DTOs.
The CPU facade's allocation ledger intentionally covers selected W8-owned
persistent object and vector-backing allocations only; opaque MFEM/HYPRE
internals and standard-library shared-ownership control blocks are explicitly
unobserved. Their census must be supplied by a future instrumented shared
build policy if the campaign requires it.
Active-run baseline/reuse events and independently owned checkpoint snapshot
events are separate observed domains; checkpoint allocation is never reported
as a zero-event numerical attempt.
