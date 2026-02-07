#pragma once

#include <algorithm>
#include "mfem.hpp"
#include "utils/config/dg_igr_defaults.hpp"
#include "numerics/riemann/face_side.hpp"

/**
 * @file physics/igr/igr_flux.hpp
 * @brief IGR-aware flux decorator for Euler systems.
 * @ingroup physics_igr
 */

namespace mfem
{

/**
 * @brief Decorator that adds Information Geometric Regularization (IGR) to a base FluxFunction.
 *
 * Applies p -> p + Sigma consistently:
 *  - Tensor flux F(U):           momentum rows +Sigma I, energy row +Sigma u.
 *  - Normal flux (F(U)·n):       momentum +Sigma n,   energy +Sigma (u·n).
 *
 * Notes
 *  - The returned max characteristic speed comes from the base flux (unchanged).
 *  - On faces, Sigma is taken from the explicit side argument in
 *    ComputeFluxDotNAtSide/ComputeAvgFluxDotNAtSide.
 *    On boundary faces (Elem2 == nullptr) the interior (Left) value is used.
 */
class FluxWithIGR : public FluxFunction, public FaceSideExplicitFlux
{
public:
  /// Construct an IGR-decorated flux around an existing base FluxFunction.
  explicit FluxWithIGR(const FluxFunction &base)
    : FluxFunction(base.num_equations, base.dim), base_(base) {}

  /// Provide the scalar Sigma field (same mesh as the solution). Pass nullptr to disable IGR.
  void SetSigmaField(const mfem::ParGridFunction *sigma) { sigma_ = sigma; }

  /// Density floor used only for u = m / rho in the IGR terms (default FluxDefaults::kRhoVelFloor).
  void SetVelocityRhoFloor(double f) { rho_vel_floor_ = f; }

  /// Side-explicit normal flux path used by side-aware numerical fluxes.
  real_t ComputeFluxDotNAtSide(const Vector &state, const Vector &normal,
                               FaceElementTransformations &Tr,
                               FaceSide side,
                               Vector &fluxDotN) const override;

  /// Side-explicit average normal flux path for non-breaking migration.
  real_t ComputeAvgFluxDotNAtSide(const Vector &state1, const Vector &state2,
                                  const Vector &normal,
                                  FaceElementTransformations &Tr,
                                  FaceSide side,
                                  Vector &fluxDotN) const override;

  /// Compute tensor flux F(U). Adds IGR tensor terms if Sigma is set; returns base max speed.
  real_t ComputeFlux(const Vector &state, ElementTransformation &Tr,
                     DenseMatrix &flux) const override;

  /// Compute normal flux F(U)·n. Adds IGR normal terms if Sigma is set; returns base max speed.
  real_t ComputeFluxDotN(const Vector &state, const Vector &normal,
                         FaceElementTransformations &Tr,
                         Vector &fluxN) const override;

  /// Jacobians: forwarded to the base (unchanged).
  void ComputeFluxJacobian(const Vector &state, ElementTransformation &Tr,
                           DenseTensor &J_) const override
  { base_.ComputeFluxJacobian(state, Tr, J_); }

  void ComputeFluxJacobianDotN(const Vector &state, const Vector &normal,
                               ElementTransformation &Tr,
                               DenseMatrix &JDotN) const override
  { base_.ComputeFluxJacobianDotN(state, normal, Tr, JDotN); }

  /// Get Sigma at the current element integration point (0 if IGR is off).
  double GetSigmaAtPoint(ElementTransformation &Tr) const;

  /// Get Sigma for a face side at the current face integration point (0 if off).
  double GetSigmaAtSide(FaceElementTransformations &Tr, FaceSide side) const;

private:
  const FluxFunction *GetBase() const { return &base_; }

  const FluxFunction &base_;
  const mfem::ParGridFunction *sigma_ = nullptr;
  double rho_vel_floor_ = dg_igr::FluxDefaults::kRhoVelFloor;

  // helpers

  inline void AddTensorSigmaTerms(const Vector &U, double sigma, DenseMatrix &F) const
  {
    // momentum rows: +sigma I
    for (int d = 0; d < dim; ++d) { F(1 + d, d) += sigma; }
    // energy row: +sigma u
    const double rho = std::max(U(0), rho_vel_floor_);
    for (int d = 0; d < dim; ++d) { F(1 + dim, d) += sigma * (U(1 + d) / rho); }
  }

  inline void AddDotNSigmaTerms(const Vector &U, const Vector &n,
                                double sigma_face, Vector &Fn) const
  {
    // momentum rows: +sigma n_d  (MFEM uses a scaled normal; stay consistent)
    for (int d = 0; d < dim; ++d) { Fn(1 + d) += sigma_face * n(d); }
    // energy row: +sigma (u·n)
    const double rho = std::max(U(0), rho_vel_floor_);
    double un = 0.0; for (int d = 0; d < dim; ++d) un += U(1 + d) * n(d);
    Fn(1 + dim) += sigma_face * (un / rho);
  }
};

} // namespace mfem
