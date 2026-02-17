#pragma once

/**
 * @file sim/wiring/dg_operator.hpp
 * @brief DG operator helper for Euler examples.
 */

#include "mfem.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"

#include "sim/context/example_context.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace dg_sim
{

/// @brief Build a DG operator with a numerical flux and shared settings.
template <typename NumericalFlux>
inline mfem::DGHyperbolicConservationLaws
MakeDGOperator(ExampleContext& ctx, const NumericalFlux& numerical_flux)
{
  auto to_mfem_stab = [](DGSEMStabilization stab)
  {
    switch (stab)
    {
    case DGSEMStabilization::None:
      return mfem::DGSEMStabilization::None;
    case DGSEMStabilization::SplitForm:
      return mfem::DGSEMStabilization::SplitForm;
    case DGSEMStabilization::Filter:
      return mfem::DGSEMStabilization::Filter;
    }
    return mfem::DGSEMStabilization::None;
  };
  auto to_mfem_vol = [](DGSEMVolumeFlux flux)
  {
    switch (flux)
    {
    case DGSEMVolumeFlux::Central:
      return mfem::DGSEMVolumeFlux::Central;
    case DGSEMVolumeFlux::Chandrashekar:
      return mfem::DGSEMVolumeFlux::Chandrashekar;
    case DGSEMVolumeFlux::Ranocha:
      return mfem::DGSEMVolumeFlux::Ranocha;
    }
    return mfem::DGSEMVolumeFlux::Central;
  };
  auto to_mfem_vol_diss = [](DGSEMVolumeDissipation diss)
  {
    switch (diss)
    {
    case DGSEMVolumeDissipation::None:
      return mfem::DGSEMVolumeDissipation::None;
    case DGSEMVolumeDissipation::LLF:
      return mfem::DGSEMVolumeDissipation::LLF;
    case DGSEMVolumeDissipation::EntropyVarsLF:
      return mfem::DGSEMVolumeDissipation::EntropyVarsLF;
    }
    return mfem::DGSEMVolumeDissipation::None;
  };
  auto env_flag = [](const char* name, bool default_value)
  {
    const char* raw = std::getenv(name);
    if (!raw)
    {
      return default_value;
    }
    const std::string value(raw);
    return !(value.empty() || value == "0" || value == "false" ||
             value == "FALSE" || value == "off" || value == "OFF");
  };
  bool preassemble_weak_div = ctx.run_cfg.opts.preassemble_weak_div;
  const bool force_device_matrix_free =
    env_flag("DG_IGR_DG_FORCE_MATRIX_FREE", false);
  if (ctx.run_cfg.dg_backend == DGBackend::DG && preassemble_weak_div &&
      mfem::Device::Allows(mfem::Backend::DEVICE_MASK) &&
      force_device_matrix_free)
  {
    if (mfem::Mpi::Root())
    {
      std::cout << "[dg-device] event=weak-divergence"
                << " requested=element-assembly"
                << " effective=matrix-free"
                << " reason=env-force-matrix-free\n";
    }
    preassemble_weak_div = false;
  }
  else if (ctx.run_cfg.dg_backend == DGBackend::DG && preassemble_weak_div &&
           mfem::Device::Allows(mfem::Backend::DEVICE_MASK) &&
           mfem::Mpi::Root())
  {
    std::cout << "[dg-device] event=weak-divergence"
              << " requested=element-assembly"
              << " effective=element-assembly"
              << " reason=device-default\n";
  }
  mfem::DGHyperbolicConservationLaws op(
    ctx.spaces.vfes,
    std::unique_ptr<mfem::HyperbolicFormIntegrator>(
      new mfem::HyperbolicFormIntegrator(
        numerical_flux,
        ctx.run_cfg.case_config.int_order_offset)),
    preassemble_weak_div);
  op.SetDGSEMGamma(ctx.run_cfg.case_config.gamma);
  op.SetFaceQuadratureOrderOffset(ctx.run_cfg.case_config.int_order_offset);
  op.SetNumericalFlux(&numerical_flux);
  if (ctx.run_cfg.dg_backend == DGBackend::DGSEM)
  {
    op.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    op.SetDGSEMVolumeFlux(to_mfem_vol(ctx.run_cfg.dgsem_volume_flux));
    op.SetDGSEMVolumeDissipation(
      to_mfem_vol_diss(ctx.run_cfg.dgsem_volume_dissipation));
    op.SetDGSEMEntropyJacobianAnalytic(ctx.run_cfg.dgsem_entropy_jacobian ==
                                       DGSEMEntropyJacobian::Analytic);
    op.SetDGSEMGamma(ctx.run_cfg.case_config.gamma);
    if (ctx.run_cfg.dgsem_stab != DGSEMStabilization::None)
    {
      op.SetDGSEMStabilization(to_mfem_stab(ctx.run_cfg.dgsem_stab));
    }
    if (ctx.dim == 2)
    {
      op.SetFaceIntegrationRule(
        mfem::dgsem::GetLGLFaceRule(ctx.run_cfg.opts.order));
    }
  }
  return op;
}

} // namespace dg_sim
