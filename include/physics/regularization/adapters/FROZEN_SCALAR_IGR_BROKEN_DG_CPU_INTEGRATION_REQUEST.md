# Coordinator integration request: frozen-scalar IGR BrokenDG CPU adapter

## Accepted child boundary

The adapter is opt-in and host-selected once through
`ConfigureFrozenScalarIGRBrokenDGCPUAdapter`. It supports only
`ElementL2 + GlobalLinearSolve + DG + MFEMCpu +
ConservativeStateBrokenDG + FrozenRKStage + EntropyNeutral`. Its five static
operations consume explicitly evaluated `mfem::Vector` point arrays. The
requirements are operation-local: coefficient uses state; source uses state
and conservative gradient; volume uses state and scalar sigma; normal uses
state, scalar sigma, and normal; bounds uses those three plus a per-point scalar
base characteristic speed. Wall, diagnostics, device execution, H1 gradients,
trace state, restart persistence, and production activation remain absent.
Default construction is explicitly `Disabled`.

The source is already discovered by the existing recursive source inventory in
`src/CMakeLists.txt`; no root or source-list edit is required.

## Minimum shared edits, kept separable

1. **Explicit configuration opt-in only**
   (`include/utils/config/igr_config.hpp`, the relevant parsing in
   `include/sim/config/case_config.hpp`, and its focused config tests): add one
   disabled-by-default choice for this exact adapter and carry `alpha`, solver
   density floor, flux density floor, and sigma-aware-bound enablement. Do not
   add a configuration base-speed constant; the accepted contract supplies base
   characteristic speed per evaluated point. Reject every other adapter tuple.
   This is separable because it changes selection data only; existing scalar
   IGR defaults and behavior remain byte-for-byte selected when the option is
   absent.

2. **One-time adapter ownership**
   (`include/sim/solver/solver_helpers.hpp` and `include/sim/solver/solver.hpp`):
   when and only when the new option is selected, construct one
   `FrozenScalarIGRBrokenDGCPUAdapter` after the existing IGR parameters are
   finalized, verify `IsConfigured()` and `HostSelectionCount() == 1`, and keep
   it with the DG operator for the run. Do not add the physical tag to the
   shared fixture `ContinuumModelKind` switch; the accepted runtime selection
   is adapter-local. This is separable from point production because it binds
   no evaluated storage and launches no operation.

3. **Explicit DG evaluated-point production**
   (a new narrow producer owned beside the DG quadrature implementation under
   `src/numerics/dg/**`, called from the DG operator wiring): evaluate only
   each operation's required fields into dedicated `mfem::Vector` arrays:
   state for coefficient; state/gradient for source; state/sigma for volume;
   state/sigma/normal for normal; and state/sigma/normal/per-point base speed
   for bounds. Call the five operation-specific
   `MakeFrozenScalarIGR*Batch` functions with the real point count, a nonzero
   identity unique to that point set, and independent producer generations.
   Coefficient, source, volume, normal, and bounds point sets must not be
   inferred from `GridFunction::Size()` and must be allowed to differ. This is separable from RK provenance because it owns FE evaluation
   and storage layout only.

4. **Authoritative stage/state/producer lifecycle**
   (`include/physics/igr/igr_coupling.hpp`, using the existing
   `RKStageAwareOperator::SetRKStageState` calls from
   `src/numerics/timestepping/RK3SSPWithLimiter.cpp`): pass the zero-based RK
   stage and nonzero state generation already owned by the stepper into the DG
   point producer. Advance separate nonzero counters whenever state, gradient,
   sigma, normal, or base-speed evaluated arrays are regenerated, and put
   nonzero expected producer authority in `RegularizationValidationContext`
   only for fields required by the selected operation. Stage 0
   must remain valid; `-1` remains the only unbound stage. This is separable
   from quadrature evaluation because it supplies provenance, not values or
   layouts.

5. **Cold restart behavior only**
   (the existing restart-to-solver handoff in `include/sim/solver/solver.hpp`):
   do not restore an adapter sigma/trace payload. After restart, recompute sigma
   from the restored state, allocate a fresh nonzero sigma producer generation,
   and expose the descriptor as `ColdRecomputed` with restart identity zero.
   Existing output and serialized IGR behavior must not be relabeled as adapter
   ownership. This is separable because it changes only post-restart producer
   invalidation/regeneration.

## Required coordinator integration gates

- The option absent reproduces all current scalar-equation, default, output,
  restart, boundary, DGSEM, HDG, and time-default tests unchanged.
- The option present runs the focused adapter test plus an end-to-end DG CPU
  stage-0 and later-stage case with distinct volume/face point sets and unequal
  valid producer epochs.
- Wrong tuple, stale/zero authority, shape, site/count/identity, alias, and
  overflow failures occur before MFEM access or writes.
- Wall and diagnostics requests remain `UnsupportedCapability`; MFEMDevice,
  DGSEM, HDG, trace/hybridized, pointwise/local, H1/None, lagged/fully coupled,
  and entropy-conservative/stable remain `UnsupportedCombination`.

No shared default, wall facade, device registration, output identity, persisted
sigma/trace claim, two-state face producer, scientific-validity claim, or
performance claim belongs in this integration.
