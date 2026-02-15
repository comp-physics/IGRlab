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

  EulerIGRFluxBundle(int dim, double gamma, dg_igr::IGRSolver& igr)
    : base_flux(dim, gamma),
      igr_flux(base_flux),
      numerical_flux(igr_flux)
  {
    igr_flux.SetSigmaField(&igr.GetSigma());
  }
};

} // namespace dg_sim
