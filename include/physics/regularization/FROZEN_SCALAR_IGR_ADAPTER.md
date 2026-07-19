# Frozen scalar IGR adapter inventory

CONTRACT-01 intentionally registers only `ZeroFixtureModelTag` and
`ConstantFixtureModelTag`. It does not change, reinterpret, or wire the current
scalar IGR implementation.

## Existing frozen scalar IGR surfaces

- `include/physics/igr/IGR.hpp` owns the auxiliary solve and the scalar sigma
  field lifecycle.
- `src/physics/igr/IGR_device_kernels.hpp` contains the current coefficient and
  source point algebra used by the device coefficient path.
- `include/physics/igr/igr_flux.hpp` and `src/physics/igr/igr_flux.cpp` contain
  the existing tensor and normal-flux correction behavior.
- `include/physics/igr/igr_coupling.hpp` records the current state/sigma
  coupling boundary.
- `include/utils/config/igr_config.hpp`,
  `include/utils/config/dg_igr_defaults.hpp`, and
  `include/sim/time/time_defaults.hpp` own the current configuration and
  default-selection behavior.
- `tests/test_igr.cpp`, `tests/test_dgsem_igr.cpp`, and
  `tests/test_time_defaults.cpp` are the existing scalar behavior/default
  fixtures that must remain unchanged.

## Future adapter boundary

A later adapter may be added without modifying the files above:

1. Define a new, explicitly named model tag and a
   `RegularizationTraits<ModelTag>` specialization under
   `include/physics/regularization/`.
2. Copy only trivially-copyable scalar parameters into
   `RegularizationDevicePolicy`; keep MFEM grid functions, solvers, and
   ownership on the host side.
3. Bind the existing sigma storage through a typed
   `ScalarAuxiliaryFieldDescriptor`, with RK-stage/state-generation provenance
   supplied by the existing stage owner.
4. Reuse or factor the already-tested scalar coefficient/source and flux
   algebra only after its owners approve that refactor. Do not duplicate or
   invent equations in the adapter.
5. Add the model to host validation and host dispatch only after its
   discretization, solve, backend, time, and entropy support matrix is agreed.
6. Preserve all current IGR option defaults and numerical fixtures; adapter
   registration must be opt-in until a separate wiring task changes that
   policy.

No backend registration, configuration parsing, time integration, boundary
condition, output/restart, or scalar IGR source is changed by CONTRACT-01.
