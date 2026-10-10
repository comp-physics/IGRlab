#pragma once

#include "igrlab/compat.hpp"

/**
 * @file sim/wiring/igr_physics.hpp
 * @brief Euler + IGR physics bundle helper.
 */

#include "sim/context/example_context.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/wiring/dg_operator.hpp"
#include "sim/wiring/dgsem_flux_config.hpp"
#include "sim/wiring/flux_bundle.hpp"

namespace dg_sim
{

/// @brief Euler+IGR physics bundle with BC wiring applied.
template <typename NumericalFlux> struct EulerIGRPhysics
{
  EulerIGRFluxBundle<NumericalFlux> flux;
  mfem::DGHyperbolicConservationLaws op;

  template <typename BcSetup>
  EulerIGRPhysics(ExampleContext& ctx,
                  double gamma,
                  ::igrlab::IGRSolver& igr,
                  const BoundaryMarkerMap& markers,
                  BcSetup&& bc_setup)
    : flux(ctx.dim,
           gamma,
           igr,
           ctx.run_cfg.igr_opts.sigma_aware_cfl,
           ctx.run_cfg.igr_opts.sigma_aware_rusanov,
           ctx.run_cfg.igr_opts.igr_invariant_domain_flux,
           UsesFrozenSigmaSpectralSpeed(ctx.run_cfg),
           ctx.run_cfg.case_config.rk3_config.rho_min,
           ctx.run_cfg.case_config.rk3_config.p_min),
      op(MakeDGOperator(ctx, flux.numerical_flux))
  {
    ConfigureDGSEMSurfaceFlux(flux.numerical_flux, ctx.run_cfg, gamma);
    EulerBoundaryWiring<NumericalFlux> bc(
      op,
      flux.numerical_flux,
      ctx.run_cfg.case_config.int_order_offset,
      ctx.dim,
      gamma,
      (ctx.run_cfg.igr_opts.igr_invariant_domain_flux ||
       UsesFrozenSigmaSpectralSpeed(ctx.run_cfg))
        ? ctx.run_cfg.case_config.rk3_config.rho_min
        : mfem::kEulerBoundaryDensityFloor,
      (ctx.run_cfg.igr_opts.igr_invariant_domain_flux ||
       UsesFrozenSigmaSpectralSpeed(ctx.run_cfg))
        ? ctx.run_cfg.case_config.rk3_config.p_min
        : mfem::kEulerBoundaryPressureFloor);
    bc_setup(bc, markers);
    igr.SetNodalSourceSlipBoundaryMarker(bc.SlipWallMarker());
  }
};

/// @brief Build Euler+IGR physics with per-case BC wiring.
template <typename NumericalFlux, typename BcSetup>
inline EulerIGRPhysics<NumericalFlux>
MakeEulerIGRPhysics(ExampleContext& ctx,
                    double gamma,
                    ::igrlab::IGRSolver& igr,
                    const BoundaryMarkerMap& markers,
                    BcSetup&& bc_setup)
{
  return EulerIGRPhysics<NumericalFlux>(ctx,
                                        gamma,
                                        igr,
                                        markers,
                                        std::forward<BcSetup>(bc_setup));
}

} // namespace dg_sim
