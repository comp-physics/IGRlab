#pragma once

#include "mfem.hpp"
#include "numerics/riemann/face_side.hpp"

/**
 * @file numerics/riemann/rusanov.hpp
 * @brief Rusanov (Lax-Friedrichs) numerical flux with face-side awareness.
 * @ingroup numerics_riemann
 */

namespace mfem
{

class FluxWithIGR;

/**
 * @brief Rusanov flux (local Lax-Friedrichs) that mirrors MFEM's reference
 * implementation, with one addition: if the wrapped FluxFunction implements
 * FaceSideExplicitFlux, left/right side evaluation is passed explicitly for each
 * call so side-dependent fields (e.g., Sigma for IGR) are selected consistently.
 *
 * Math, scaling, and Jacobians are otherwise identical to MFEM's RusanovFlux.
 */
class RusanovFluxWithSides : public NumericalFlux
{
public:
  /// @param fluxFunction physical flux F(u, x)
  explicit RusanovFluxWithSides(const FluxFunction& fluxFunction);

  /// Fhat n = 1/2(F(u^-)·n + F(u^+)·n) - 1/2 lambda ||n|| (u^+ - u^-)
  real_t Eval(const Vector& state1,
              const Vector& state2,
              const Vector& nor,
              FaceElementTransformations& Tr,
              Vector& flux) const override;

  /// Grad w.r.t. (side == 1 ? u^- : u^+); matches MFEM's RusanovFlux exactly.
  void Grad(int side,
            const Vector& state1,
            const Vector& state2,
            const Vector& nor,
            FaceElementTransformations& Tr,
            DenseMatrix& grad) const override;

  /// Average variant, exactly as in MFEM (only sets side if supported).
  real_t Average(const Vector& state1,
                 const Vector& state2,
                 const Vector& nor,
                 FaceElementTransformations& Tr,
                 Vector& flux) const override;

  /// AverageGrad, identical structure to MFEM (only sets side if supported).
  void AverageGrad(int side,
                   const Vector& state1,
                   const Vector& state2,
                   const Vector& nor,
                   FaceElementTransformations& Tr,
                   DenseMatrix& grad) const override;

protected:
  const FaceSideExplicitFlux* side_explicit_flux_ = nullptr;
  const FluxWithIGR* igr_flux_ = nullptr;
#ifndef MFEM_THREAD_SAFE
  mutable Vector fluxN1, fluxN2;
  mutable DenseMatrix JDotN;
#endif
};

} // namespace mfem
