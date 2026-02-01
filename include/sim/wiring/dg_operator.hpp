#pragma once

/**
 * @file sim/wiring/dg_operator.hpp
 * @brief DG operator helper for Euler examples.
 */

#include "mfem.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"

#include "sim/context/example_context.hpp"

namespace dg_igr_examples
{

/// @brief Build a DG operator with a numerical flux and shared settings.
template <typename NumericalFlux>
inline mfem::DGHyperbolicConservationLaws MakeDGOperator(
    ExampleContext& ctx,
    const NumericalFlux& numerical_flux)
{
  mfem::DGHyperbolicConservationLaws op(
      ctx.spaces.vfes,
      std::unique_ptr<mfem::HyperbolicFormIntegrator>(
          new mfem::HyperbolicFormIntegrator(numerical_flux,
                                             ctx.run_cfg.int_order_offset)),
      ctx.run_cfg.opts.preassemble_weak_div);
  op.SetNumericalFlux(&numerical_flux);
  if (ctx.run_cfg.dg_backend == DGBackend::DGSEM)
  {
    op.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    if (ctx.dim == 2)
    {
      op.SetFaceIntegrationRule(
          mfem::dgsem::GetLGLFaceRule(ctx.run_cfg.order));
    }
  }
  return op;
}

} // namespace dg_igr_examples
