#pragma once

/**
 * @file sim/wiring/igr_physics.hpp
 * @brief Euler + IGR physics bundle helper.
 */

#include "sim/context/example_context.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/wiring/dg_operator.hpp"
#include "sim/wiring/dgsem_flux_config.hpp"
#include "sim/wiring/flux_bundle.hpp"

namespace dg_igr_examples
{

/// @brief Euler+IGR physics bundle with BC wiring applied.
template <typename NumericalFlux>
struct EulerIGRPhysics
{
  EulerIGRFluxBundle<NumericalFlux> flux;
  mfem::DGHyperbolicConservationLaws op;

  template <typename BcSetup>
  EulerIGRPhysics(ExampleContext& ctx,
                  double gamma,
                  dg_igr::IGRSolver& igr,
                  const BoundaryMarkerMap& markers,
                  BcSetup&& bc_setup)
    : flux(ctx.dim, gamma, igr),
      op(MakeDGOperator(ctx, flux.numerical_flux))
  {
    ConfigureDGSEMSurfaceFlux(flux.numerical_flux, ctx.run_cfg, gamma);
    EulerBoundaryWiring<NumericalFlux> bc(
        op, flux.numerical_flux, ctx.run_cfg.int_order_offset, ctx.dim);
    bc_setup(bc, markers);
  }
};

/// @brief Build Euler+IGR physics with per-case BC wiring.
template <typename NumericalFlux, typename BcSetup>
inline EulerIGRPhysics<NumericalFlux> MakeEulerIGRPhysics(
    ExampleContext& ctx,
    double gamma,
    dg_igr::IGRSolver& igr,
    const BoundaryMarkerMap& markers,
    BcSetup&& bc_setup)
{
  return EulerIGRPhysics<NumericalFlux>(
      ctx, gamma, igr, markers, std::forward<BcSetup>(bc_setup));
}

} // namespace dg_igr_examples
