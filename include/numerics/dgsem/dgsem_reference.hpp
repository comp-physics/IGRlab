#pragma once

/**
 * @file numerics/dgsem/dgsem_reference.hpp
 * @brief Frozen CPU-reference contracts for the DGSEM scientific oracle.
 * @ingroup numerics_dgsem
 *
 * This header intentionally contains data-only policies.  It is C++17
 * compatible, has no run-time polymorphism, and is not a production tuning
 * interface.  The constants below are the pre-declared acceptance contract
 * used by the registered DGSEM tests and their machine-readable evidence.
 */

#include <array>
#include <cstddef>

namespace mfem
{
namespace dgsem
{

struct DGSEMReferenceConsumer
{
  const char* consumer;
  const char* path;
  const char* state_contract;
  const char* auxiliary_contract;
};

/**
 * @brief Complete inventory of production and reference consumers reached by
 * the current DGSEM state/auxiliary path.
 *
 * Generic lifecycle consumers are listed because DGSEM state passes through
 * them even though their implementation is coordinator-owned.  The inventory
 * is descriptive and does not authorize edits outside the DGSEM-owned paths.
 */
inline constexpr std::array<DGSEMReferenceConsumer, 19>
  kDGSEMReferenceConsumers{{
    {"DGHyperbolicConservationLaws::Mult",
     "src/numerics/dg/DGHyperbolicConservationLaws.cpp",
     "true/local conservative state -> DGSEM weak residual",
     "delegates frozen auxiliary reads to DGSEM volume/surface fluxes"},
    {"DGSEMOperator::AssembleVolume",
     "src/numerics/dgsem/dgsem_operator.cpp",
     "element-local LGL conservative states and contravariant metrics",
     "FluxWithIGR sigma sampled at the current element point when present"},
    {"DGSEMOperator::AssembleSurface",
     "src/numerics/dgsem/dgsem_operator.cpp",
     "local, local-neighbor, shared-neighbor, and boundary-ghost states",
     "side-explicit frozen sigma sampled by the selected numerical flux"},
    {"DGSEMOperator::ApplyInverseMass",
     "src/numerics/dgsem/dgsem_operator.cpp",
     "element weak residual -> element strong time derivative",
     "none"},
    {"DGSEMSurfaceFlux",
     "src/numerics/riemann/dgsem_surface_flux.cpp",
     "left/right conservative state and scaled outward normal",
     "left/right frozen sigma for entropy and dissipation semantics"},
    {"RusanovFluxWithSides",
     "src/numerics/riemann/rusanov.cpp",
     "left/right conservative state and scaled outward normal",
     "side-explicit sigma and separately configured CFL/dissipation speeds"},
    {"DiffusiveRusanovFlux",
     "src/numerics/riemann/diffusive_rusanov.cpp",
     "left/right conservative state and scaled outward normal",
     "wrapped FluxWithIGR semantics only when that flux is supplied"},
    {"FluxWithIGR",
     "src/physics/igr/igr_flux.cpp",
     "Euler conservative state",
     "non-owning, stage-frozen scalar sigma field; zero when absent"},
    {"EulerIGRFluxBundle sigma ownership handoff",
     "include/sim/wiring/flux_bundle.hpp:29",
     "constructs the Euler/IGR numerical-flux chain; does not own stage state",
     "hands IGRSolver-owned sigma to FluxWithIGR as a non-owning pointer"},
    {"BdrHyperbolicFluxIntegrator/BoundaryState",
     "src/numerics/dg/BdrHyperbolicFluxIntegrator.cpp; src/physics/bc/boundary_state.cpp",
     "interior state and normal -> boundary ghost state",
     "no auxiliary mutation; numerical flux reads the frozen interior side"},
    {"RK3SSPWithLimiter",
     "src/numerics/timestepping/RK3SSPWithLimiter.cpp",
     "stage conservative states and DGSEM time derivative",
     "stage provenance is owned upstream; DGSEM does not update sigma"},
    {"limiter pipeline",
     "src/numerics/limiters/limiter_pipeline.cpp",
     "post-stage DGSEM conservative state",
     "no direct DGSEM auxiliary read"},
    {"solver/backend wiring",
     "include/sim/solver/solver.hpp; include/sim/wiring/dg_operator.hpp",
     "selects tensor LGL state space and DGSEM residual mode",
     "passes existing scalar-IGR field ownership without reinterpretation"},
    {"DGSEM flux wiring",
     "include/sim/wiring/dgsem_flux_config.hpp; include/sim/wiring/igr_physics.hpp",
     "selects declared volume/surface numerical flux policies",
     "configures only existing scalar-IGR semantics"},
    {"diagnostics/output/restart",
     "include/sim/io; src/io",
     "observes accepted stage state after the DGSEM operator",
     "observes existing sigma lifecycle; does not define DGSEM semantics"},
    {"smooth-advection examples",
     "examples/dgsem_smooth_advection; examples/dgsem_smooth_advection_periodic",
     "Euler conservative solution state",
     "none"},
    {"wall/shock examples",
     "examples/dgsem_stationary_shock; examples/dgsem_forward_facing_step; examples/dgsem_supersonic_cylinder",
     "Euler conservative solution state with declared boundary policies",
     "existing scalar-IGR semantics only when wired; no wall-stability claim"},
    {"blast/vortex examples",
     "examples/dgsem_sedov_taylor_blast; examples/dgsem_shock_vortex_interaction",
     "Euler conservative solution state",
     "existing scalar-IGR semantics only when wired"},
    {"registered DGSEM reference suites",
     "tests/test_dgsem_basis.cpp; tests/test_dgsem_flux.cpp; tests/test_dgsem_operator.cpp; tests/test_dgsem_advection.cpp; tests/test_dgsem_igr.cpp",
     "deterministic CPU fixtures and bounded trajectories",
     "zero/constant/frozen prescribed sigma only"},
  }};

struct DGSEMConvergenceContract
{
  inline static constexpr std::array<int, 4> orders{{1, 2, 3, 4}};
  inline static constexpr std::array<int, 3> elements_per_direction{{3, 6, 12}};
  inline static constexpr double final_time = 1.0e-2;
  inline static constexpr double timestep_coefficient = 5.0e-2;
  inline static constexpr double curved_map_amplitude = 2.5e-2;
  inline static constexpr double monotone_error_factor = 1.0;

  static constexpr double ExpectedRateFloor(int order)
  {
    // The asymptotic interpolation target is p+1.  A fixed 0.75-order
    // pre-asymptotic allowance is frozen before executing either ladder.
    return static_cast<double>(order) + 0.25;
  }
};

struct DGSEMReferenceTolerances
{
  inline static constexpr double lgl = 2.0e-12;
  inline static constexpr double sbp = 2.0e-11;
  inline static constexpr double pointwise_fixture = 2.0e-12;
  inline static constexpr double constant_affine = 2.0e-10;
  inline static constexpr double constant_curved = 2.0e-9;
  inline static constexpr double conservation = 5.0e-10;
  inline static constexpr double face_balance = 2.0e-11;
  inline static constexpr double inverse_mass = 2.0e-12;
  inline static constexpr double mpi_parity = 1.0e-9;
};

inline constexpr std::size_t kDGSEMAttributionSamples = 9;

inline constexpr const char* kDGSEMScalarIGRScope =
  "Euler plus zero/constant/prescribed stage-frozen scalar sigma only; this "
  "oracle neither repairs B-SCI-001 nor claims production wall stability, "
  "and it has no dependency on CONTRACT-01.";

} // namespace dgsem
} // namespace mfem
