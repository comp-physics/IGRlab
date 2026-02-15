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
 * @brief Subsonic outlet with prescribed static pressure.
 *
 * Uses a characteristic-compatible closure:
 *   - preserve outgoing acoustic invariant from the interior,
 *   - impose static pressure at the boundary,
 *   - keep interior tangential velocity.
 */
class SubsonicOutletPressureState : public BoundaryState
{
public:
  SubsonicOutletPressureState(int dim, double gamma, double pressure_out);

  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;

private:
  int dim_ = 0;
  double gamma_ = 1.4;
  double p_out_ = 1.0;
};

/**
 * @brief Approximate non-reflecting outlet anchored to a reference state.
 *
 * Uses a one-way characteristic closure:
 *   - outgoing acoustic invariant from interior,
 *   - incoming acoustic invariant from reference state,
 *   - tangential velocity from interior.
 */
class NonReflectingOutletState : public BoundaryState
{
public:
  NonReflectingOutletState(const mfem::Vector& Uref, int dim, double gamma);

  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;

private:
  mfem::Vector Uref_;
  int dim_ = 0;
  double gamma_ = 1.4;
};

/**
 * @brief Subsonic inlet anchored to a reference conservative state.
 *
 * Uses a characteristic-compatible closure:
 *   - outgoing acoustic invariant from interior,
 *   - incoming acoustic invariant/entropy/tangential velocity from reference.
 */
class SubsonicInletState : public BoundaryState
{
public:
  SubsonicInletState(const mfem::Vector& Uin, int dim, double gamma);

  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;

private:
  mfem::Vector Uin_;
  int dim_ = 0;
  double gamma_ = 1.4;
};

/**
 * @brief Far-field state: branch to supersonic/subsonic inflow/outflow modes.
 *
 * Supersonic branches reduce to Dirichlet inflow / transmissive outflow.
 * Subsonic branches use characteristic inlet/non-reflecting outlet closures.
 */
class FarFieldState : public BoundaryState
{
public:
  FarFieldState(const mfem::Vector& Ufar, int dim, double gamma);

  void Eval(const mfem::Vector& Uminus,
            const mfem::Vector& nor,
            mfem::FaceElementTransformations& Tr,
            mfem::Vector& Uplus) const override;

private:
  mfem::Vector Ufar_;
  int dim_ = 0;
  double gamma_ = 1.4;
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
