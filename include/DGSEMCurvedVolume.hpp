#pragma once
// DGSEMCurvedVolume.hpp
// Curved-element DGSEM-style volume contribution using a volume Rusanov flux.
// This header declares:
//   - struct DGSEMOptions (placeholder for runtime knobs)
//   - class  VolumeRusanovFlux (uses FluxFunction only; no NumericalFlux)
//   - void   AddDGSEMVolumeContribution(...)
// Keep everything in namespace mfem to match your call site/linker.

#include "mfem.hpp"

namespace mfem
{

// Runtime switches; extend as needed.
struct DGSEMOptions
{
   bool assume_affine = false;   // if true, use element-center metrics (for testing)
   double small_detj  = 1e-14;   // guard for degenerate Jacobians
};

// Volume Rusanov flux built from a PDE FluxFunction.
// Uses FluxFunction::ComputeFluxDotN with a *scaled* normal (contravariant column).
class VolumeRusanovFlux
{
public:
   explicit VolumeRusanovFlux(const FluxFunction &F) : fluxFunction_(F)
   {
#ifndef MFEM_THREAD_SAFE
      fluxN1_.SetSize(F.num_equations);
      fluxN2_.SetSize(F.num_equations);
#endif
   }

   // Compute flux-difference flux at a *node* with scaled normal 'nor'.
   // Tr is the element transformation at that node.
   real_t Eval(const Vector &U1, const Vector &U2,
               const Vector &nor, ElementTransformation &Tr,
               Vector &flux_out) const;

private:
   const FluxFunction &fluxFunction_;
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1_, fluxN2_;
#endif
};

// Main entry: assemble DGSEM-style volume term into z (same layout as MFEM NF).
// Signature must MATCH the call site exactly.
void AddDGSEMVolumeContribution(const ParFiniteElementSpace &vfes,
                                const ParGridFunction       &U,
                                const FluxFunction          &fluxF,
                                const DGSEMOptions          &opt,
                                Vector                      &z);

} // namespace mfem
