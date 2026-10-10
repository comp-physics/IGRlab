# Frozen scalar IGR adapter contract handoff

CONTRACT-ADAPTER-02 extends the static regularization contract so a later
frozen-scalar BrokenDG CPU adapter can describe its inputs and capabilities
truthfully. It registers only a neutral `BrokenDGContractProbeModelTag`.
It does not define a physical frozen-scalar model, call scalar IGR algebra, or
wire configuration, DG operators, time integration, boundaries, restart, or
output.

## Static selection and support boundary

`GradientTreatmentKind` is an independent axis with exactly `None` and
`ConservativeStateBrokenDG`. Existing fixture combinations retain `None`.
The neutral probe accepts only:

- `ElementL2` auxiliary discretization;
- `GlobalLinearSolve` auxiliary solution;
- `DG` spatial backend and `MFEMCpu` execution;
- `ConservativeStateBrokenDG` gradient treatment;
- `FrozenRKStage` time treatment; and
- `EntropyNeutral` entropy treatment.

`ModelCombinationSupport<ModelTag, ...>` is the model-specific customization
point. The generic `StaticCombinationSupport` delegates to it instead of
hard-coding future model matrices in one runtime Boolean. Runtime validation
and static dispatch must agree with that same model-specific support result.

`ConfigureRegularization` performs the host selection once and binds a
`RegularizationStaticFunctionTable`. Each enabled operation has its own
statically specialized function. An MFEM point kernel contains no model or
operation switch. `ExecuteRegularization` is only the one-shot wrapper around
that configured path.

## Operation table

Every launch selects one `RegularizationOperationKind`, one evaluated point
batch, and one writable output. Point counts and point-set identities are
operation-specific; no shared coefficient/source/volume/face/wall quadrature is
implied.

| Operation | Site | Probe-required evaluated fields | Selected output | Probe capability |
| --- | --- | --- | --- | --- |
| `AuxiliaryCoefficient` | Volume | state, auxiliary | `auxiliary_coefficient` | enabled |
| `AuxiliarySource` | Volume | state, conservative-state-gradient, auxiliary | `auxiliary_source` | enabled |
| `VolumeCorrection` | Volume | state, auxiliary | `volume_correction` | enabled |
| `NormalCorrection` | Face | state, auxiliary, normal | `normal_correction` | enabled |
| `WallLaw` | Wall | state, trace when a model enables it | `wall_flux` | disabled and hook absent |
| `CharacteristicBounds` | Face | state, normal, base-characteristic-speed | `characteristic_bounds` | enabled |
| `Diagnostics` | Volume | state, auxiliary, trace when a model enables it | `diagnostics` | disabled and hook absent |

Coefficient and source are independent capabilities and hooks. The same is true
for volume correction, normal correction, wall law, characteristic bounds, and
diagnostics. Trait discovery requires a hook only when the corresponding
constexpr capability is true. Selecting a disabled operation returns
`UnsupportedCapability` before batch or output validation, so its hook,
inputs, output, and expected producer generations may all be absent.

## Explicit evaluated-point batches

`EvaluatedPointBatch` names six possible producer-owned inputs:

- evaluated state;
- evaluated conservative-state gradient;
- evaluated auxiliary;
- evaluated trace;
- evaluated normal; and
- evaluated base characteristic speed.

Each `EvaluatedPointFieldBinding` binds exactly an `mfem::Vector` containing
an explicitly allocated and filled evaluated array. Its metadata records point
count, component extent, dimension, `EvaluationSiteKind`, nonzero point-set
identity, RK stage, state generation, and its own producer generation. Raw
`GridFunction` or `ParGridFunction` DOFs and
`ParGridFunction::Size()` are not evaluated-point storage or point counts.

The `base_characteristic_speed` field has exactly one scalar component at each
evaluated point. It uses the same site/count/dimension/point-set/stage/state
metadata contract and its own nonzero authoritative producer generation. A
configuration scalar is not an evaluated value, and this contract does not
infer the field from state or normal storage.

Validation considers only fields required by the statically selected operation.
Required fields must agree with their batch on site, point count, point-set
identity, dimension, RK stage, and state generation. Unrequired fields may be
absent and need no storage or producer expectation. The probe therefore needs
no trace or wall buffer.

## Authoritative provenance

RK stages are zero-based. Stage 0 is valid; `-1` is the explicit
unbound/invalid default in evaluated-field metadata, batch metadata, evaluated
batches, auxiliary provenance, and `RegularizationValidationContext`.
A valid provenance stamp has a nonnegative stage, a nonzero state generation,
and a nonzero producer generation.

`RegularizationValidationContext` supplies the authoritative expected RK
stage, state generation, and
`ExpectedEvaluatedPointProducerGenerations` for state, conservative-state
gradient, auxiliary, trace, normal, and base characteristic speed. For each
operation-required field:

1. its own expected producer generation must be nonzero;
2. its binding producer generation must be nonzero; and
3. the binding generation must equal that field's expected generation.

A zero producer authority is `InvalidProvenanceStamp`; a mismatched
nonnegative stage or mismatched nonzero state/producer generation is
`StaleProvenance`. Producer counters are independent: state, gradient,
auxiliary, trace, normal, and base-characteristic-speed generations are never
compared with one another, and valid unequal values are expected.

## Model preflight and access ordering

Each trait provides a host-only `ModelPreflight(operation, batch.Metadata())`.
It runs after static configuration selection and before validation can acquire
any MFEM `Read`, `Write`, or `ReadWrite` pointer. The neutral BrokenDG
probe requires:

- dimension in `[1,3]`;
- Euler state extent `dimension + 2`;
- conservative-gradient extent
  `state_components * dimension` for source evaluation;
- scalar auxiliary input where that operation requires auxiliary data;
- Volume site for coefficient, source, and volume correction; and
- Face site with a dimension-vector normal for normal correction and bounds.

The probe opts `CharacteristicBounds` into the required
`base_characteristic_speed` field. Configured validation rejects every
non-scalar extent and all ordinary required-field provenance, size, storage, and
alias failures before output pointer acquisition.
A failed preflight returns `InvalidModelShape` and preserves every writable

output, including when the supplied storage is deliberately missing.

## Writable output and alias boundary

Only the output selected by the current operation is shape-validated, acquired,
and written. Other operation outputs may be absent. Before acquisition, the
selected writable `mfem::Vector` object is rejected if it is also any present
evaluated input object. Input and selected-output extent multiplication is
checked for overflow.

MFEM exposes no portable pre-acquisition overlap query for distinct
`mfem::Vector` objects. Distinct vector objects with overlapping backing
storage remain unsupported caller error. Likewise, output bindings for
operations not being launched do not create a cross-operation alias contract.

## Auxiliary restart truthfulness

`AuxiliaryRestartKind` has exactly:

- `None`: no serialized auxiliary state and restart identity zero;
- `Serialized`: persisted auxiliary state with a nonzero restart identity;
  and
- `ColdRecomputed`: restart identity zero, followed by regeneration with a
  fresh producer epoch after restart.

`ColdRecomputed` never claims persisted sigma or trace data. Evaluated point
arrays carry neither restart identity nor output identity.

## Later adapter responsibility

A later adapter requires a separately reviewed model tag, traits, physical
algebra boundary, producer implementation, and opt-in production wiring. It
must produce the physical Euler base characteristic speed at every evaluated
face point from the same point-local state and normal used by that evaluation,
bind it as the scalar `base_characteristic_speed` field, and stamp its own
nonzero authoritative producer generation alongside the shared batch
provenance. The configuration
`RegularizationDevicePolicy::characteristic_speed` scalar is not a truthful
substitute for that varying field. This contract documentation neither
implements nor revives the rejected physical frozen-scalar adapter. It does not
admit wall behavior, device execution, restart persistence, H1, trace
production, two-state face production, scientific validity, or a performance
claim for that future adapter.
