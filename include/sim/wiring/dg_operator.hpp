#pragma once

/**
 * @file sim/wiring/dg_operator.hpp
 * @brief DG operator helper for Euler examples.
 */

#include "mfem.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"

#include "sim/context/example_context.hpp"

namespace dg_igr_examples
{

/// @brief Build a DG operator with a numerical flux and shared settings.
template <typename NumericalFlux>
inline mfem::DGHyperbolicConservationLaws MakeDGOperator(
    ExampleContext& ctx,
    const NumericalFlux& numerical_flux)
{
  return mfem::DGHyperbolicConservationLaws(
      ctx.spaces.vfes,
      std::unique_ptr<mfem::HyperbolicFormIntegrator>(
          new mfem::HyperbolicFormIntegrator(numerical_flux,
                                             ctx.run_cfg.int_order_offset)),
      ctx.run_cfg.opts.preassemble_weak_div);
}

} // namespace dg_igr_examples
