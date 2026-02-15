#pragma once

/**
 * @file sim/wiring/dgsem_flux_config.hpp
 * @brief Helper for configuring DGSEM surface flux selection.
 */

#include <type_traits>

#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "sim/config/case_config.hpp"

namespace dg_sim
{

template <typename NumericalFlux>
inline void ConfigureDGSEMSurfaceFlux(NumericalFlux& flux,
                                      const RunConfig& run_cfg,
                                      double gamma)
{
  (void)flux;
  (void)run_cfg;
  (void)gamma;
}

template <>
inline void
ConfigureDGSEMSurfaceFlux<mfem::DGSEMSurfaceFlux>(mfem::DGSEMSurfaceFlux& flux,
                                                  const RunConfig& run_cfg,
                                                  double gamma)
{
  auto to_mfem_surface = [](DGSEMSurfaceFlux sf)
  {
    switch (sf)
    {
    case DGSEMSurfaceFlux::Rusanov:
      return mfem::DGSEMSurfaceFluxType::Rusanov;
    case DGSEMSurfaceFlux::DiffusiveRusanov:
      return mfem::DGSEMSurfaceFluxType::DiffusiveRusanov;
    case DGSEMSurfaceFlux::EntropyStableWinters:
      return mfem::DGSEMSurfaceFluxType::EntropyStableWinters;
    case DGSEMSurfaceFlux::EntropyStableLFEntropyVars:
      return mfem::DGSEMSurfaceFluxType::EntropyStableLFEntropyVars;
    }
    return mfem::DGSEMSurfaceFluxType::Rusanov;
  };
  auto to_entropy_type = [](DGSEMVolumeFlux vf)
  {
    switch (vf)
    {
    case DGSEMVolumeFlux::Central:
      return mfem::dgsem::EntropyFluxType::Ranocha;
    case DGSEMVolumeFlux::Chandrashekar:
      return mfem::dgsem::EntropyFluxType::Chandrashekar;
    case DGSEMVolumeFlux::Ranocha:
      return mfem::dgsem::EntropyFluxType::Ranocha;
    }
    return mfem::dgsem::EntropyFluxType::Ranocha;
  };
  auto to_entropy_jacobian = [](DGSEMEntropyJacobian mode)
  {
    switch (mode)
    {
    case DGSEMEntropyJacobian::FiniteDifference:
      return mfem::dgsem::EntropyJacobianMode::FiniteDifference;
    case DGSEMEntropyJacobian::Analytic:
      return mfem::dgsem::EntropyJacobianMode::Analytic;
    }
    return mfem::dgsem::EntropyJacobianMode::FiniteDifference;
  };

  flux.SetSurfaceFluxType(to_mfem_surface(run_cfg.dgsem_surface_flux));
  flux.SetEntropyFluxType(to_entropy_type(run_cfg.dgsem_volume_flux));
  flux.SetEntropyJacobianMode(
    to_entropy_jacobian(run_cfg.dgsem_entropy_jacobian));
  flux.SetGamma(gamma);
}

} // namespace dg_sim
