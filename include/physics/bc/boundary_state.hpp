#pragma once

#include "mfem.hpp"

/**
 * @file physics/bc/boundary_state.hpp
 * @brief Boundary-state policies for DG hyperbolic problems.
 * @ingroup physics_bc
 */

namespace mfem
{

/**
 * @brief Abstract boundary “ghost state” policy for DG hyperbolic problems.
 *
 * Given the interior (minus) conservative state at a boundary face, a
 * (scaled) outward normal vector `nor`, and the face element transformations
 * at the current quadrature point, produce the exterior (plus) conservative
 * state that will be fed into the numerical flux \f$\hat F(U^-,U^+)\cdot n\f$.
 *
 * Implementations should be conservative-state based:
 *   - 2D Euler: U = (rho, rho*u_x, rho*u_y, E)
 *   - 3D Euler: U = (rho, rho*u_x, rho*u_y, rho*u_z, E)
 *
 * @note `nor` is the *scaled* face normal as used by MFEM (Jacobian scaling
 * is already included). If you need a unit normal for your math, normalize it.
 */
class BoundaryState
{
public:
  virtual ~BoundaryState() = default;

  /**
   * @brief Compute the boundary ghost state Uplus.
   *
   * @param[in] Uminus  Interior conservative state at the face point.
   * @param[in] nor     Scaled normal vector (MFEM convention).
   * @param[in] Tr      Face element transformations (at the current quad pt).
   * @param[out] Uplus  Exterior/ghost conservative state to use in the Riemann flux.
   */
  virtual void Eval(const mfem::Vector& Uminus,
                    const mfem::Vector& nor,
                    mfem::FaceElementTransformations& Tr,
                    mfem::Vector& Uplus) const = 0;
};

/**
 * @brief Supersonic inlet boundary: enforce a constant conservative state.
 *
 * Sets U^+ = U_in (constant), regardless of the interior state.
 */
class SupersonicInletState : public BoundaryState
{
public:
  explicit SupersonicInletState(const mfem::Vector& Uin);

  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;

private:
  mfem::Vector Uin_;
};

/**
 * @brief Supersonic outlet boundary: transmissive/do-nothing.
 *
 * Sets U^+ = U^-.
 */
class SupersonicOutletState : public BoundaryState
{
public:
  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;
};

/**
 * @brief Adiabatic slip wall: reflect normal velocity, keep rho and E.
 *
 * Implements:
 *   u' = u - 2 (u · n_hat) n_hat ,  with n_hat the unit outward normal.
 * Then sets (rho*u)' = rho u' and leaves rho and E unchanged.
 */
class SlipWallState : public BoundaryState
{
public:
  explicit SlipWallState(int dim);

  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;

private:
  int dim_;
};

} // namespace mfem
