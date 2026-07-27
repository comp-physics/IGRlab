# Minimal shared runtime ABI request from the CPU persistent adapter

This leaf deliberately defines only `CpuFailurePhase`, CPU configuration,
CPU evidence, and CPU injection types. It does not define the canonical
`ConventionalEulerSmoothPersistentV1Tag`, a shared `FailurePhase`, or any
cross-backend run/evidence DTO.

The accepted `PersistentRunContract::FinishAttempt` validates and publishes
the active state internally through a trusted `noexcept` handle swap. That is
sufficient for the CPU facade's explicit sequence:

1. `PrepareAdvance` or `PrepareRetry` runs assembly through diagnostics and
   exposes an attempt-qualified, read-only numerical candidate with entry and
   proposed target coordinates plus coefficient hashes.
2. An outer owner may stage checkpoint/output/history bytes from that view.
3. A checkpoint-staging, output-staging, or proposed-commit failure calls
   `AbortPrepared`, which enters `FinishAttempt` with a failure and rolls the
   candidate back without publication.
4. Successful staging calls `FinalizePrepared`, which enters
   `FinishAttempt` and publishes.

It is not sufficient for a fallible operation after state publication. The
minimal future shared extension is therefore one common two-phase commit
capability, not a second CPU publication protocol:

- `ValidateAttempt(...) -> PreparedCommitCapability` performs all current
  `FinishAttempt` validation and product consumption without calling
  `StoragePolicy::Publish`;
- `AbortPrepared(capability, reason)` invalidates the prepared capability and
  rolls back the candidate; and
- `PublishPrepared(capability) noexcept` performs the existing handle swap,
  coordinate/telemetry publication, and no later recoverable work.

The outer lifecycle must complete fallible history append and checkpoint/VTK
staging between validation and `PublishPrepared`. A true failure after
`PublishPrepared` cannot satisfy the card's natural `Commit` rollback rule and
must not be simulated as postcommit staging. A coordinator-owned shared phase
enum can accompany that extension so CPU/CUDA/CLI/I/O use one spelling; this
leaf requests no broader DTO hierarchy or virtual interface.
