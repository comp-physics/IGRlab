#pragma once

/**
 * @file sim/wiring/flux_bundle.hpp
 * @brief Flux bundle helpers for Euler + IGR.
 */

#include "physics/igr/igr_flux.hpp"

namespace dg_sim
{

/// @brief Flux bundle for Euler + IGR wiring with a chosen numerical flux.
template <typename NumericalFlux> struct EulerIGRFluxBundle
{
  mfem::EulerFlux base_flux;
  mfem::FluxWithIGR igr_flux;
  NumericalFlux numerical_flux;

  EulerIGRFluxBundle(int dim,
                     double gamma,
                     dg_igr::IGRSolver& igr,
                     bool sigma_aware_cfl,
                     bool sigma_aware_rusanov,
                     bool invariant_domain_flux,
                     double limiter_density_floor,
                     double limiter_pressure_floor)
    : base_flux(dim, gamma),
      igr_flux(base_flux, gamma),
      numerical_flux(igr_flux)
  {
    igr_flux.SetSigmaField(&igr.GetSigma());
    igr_flux.SetSigmaAwareCFL(sigma_aware_cfl);
    igr_flux.SetSigmaAwareRusanov(sigma_aware_rusanov);
    if (invariant_domain_flux)
    {
      igr_flux.SetInvariantDomainFloors(limiter_density_floor,
                                        limiter_pressure_floor);
    }
    igr_flux.SetInvariantDomainFlux(invariant_domain_flux);
  }
};

} // namespace dg_sim
