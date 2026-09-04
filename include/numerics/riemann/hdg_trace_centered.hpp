#pragma once

/**
 * @file numerics/riemann/hdg_trace_centered.hpp
 * @brief Face-local flux obtained from equal-penalty HDG trace elimination.
 * @ingroup numerics_riemann
 */

#include "mfem.hpp"

#include "numerics/riemann/face_side.hpp"

namespace mfem
{

class FluxWithIGR;

/**
 * Face-local Euler flux induced by eliminating an equal-scalar-penalty HDG
 * trace at one quadrature point.
 *
 * On an interior face, conservation of
 *
 *   F(Uhat,sigmahat).n + tau (U - Uhat)
 *
 * with the same scalar tau on both sides gives Uhat=(UL+UR)/2.  For the
 * primal broken-L2 IGR field we use sigmahat=(sigmaL+sigmaR)/2, hence
 *
 *   F* = F(Uhat,sigmahat).n + 0.5 tau ||n|| (UL-UR).
 *
 * tau is the maximum configured dissipation-speed bound at UL, UR, and
 * Uhat.  The returned CFL speed is likewise the maximum of the three
 * configured CFL bounds.  This differs from Rusanov for nonlinear Euler
 * fluxes because it evaluates the physical flux at the arithmetic trace
 * instead of averaging the two physical fluxes.
 *
 * Physical boundaries use the application's existing exterior/ghost state
 * in the same formula.  That extension preserves the established boundary
 * policies, but is not claimed to be exact elimination of a polynomial HDG
 * boundary trace problem.
 */
class HdgTraceCenteredFlux : public NumericalFlux
{
public:
  explicit HdgTraceCenteredFlux(const FluxFunction& flux_function);

  real_t Eval(const Vector& state1,
              const Vector& state2,
              const Vector& nor,
              FaceElementTransformations& Tr,
              Vector& flux) const override;

  void Grad(int side,
            const Vector& state1,
            const Vector& state2,
            const Vector& nor,
            FaceElementTransformations& Tr,
            DenseMatrix& grad) const override;

  real_t Average(const Vector& state1,
                 const Vector& state2,
                 const Vector& nor,
                 FaceElementTransformations& Tr,
                 Vector& flux) const override;

  void AverageGrad(int side,
                   const Vector& state1,
                   const Vector& state2,
                   const Vector& nor,
                   FaceElementTransformations& Tr,
                   DenseMatrix& grad) const override;

private:
  const FluxFunction& flux_fn_;
  const FaceSideExplicitFlux* side_explicit_flux_ = nullptr;
  const FluxWithIGR* igr_flux_ = nullptr;

#ifndef MFEM_THREAD_SAFE
  mutable Vector state_hat_;
  mutable Vector flux_n_left_;
  mutable Vector flux_n_right_;
  mutable Vector flux_n_hat_;
#endif

  real_t TraceFluxAndSpeeds_(const Vector& state_hat,
                             real_t sigma_hat,
                             const Vector& nor,
                             FaceElementTransformations& Tr,
                             Vector& flux_n_hat,
                             real_t& dissipation_speed) const;
};

} // namespace mfem
