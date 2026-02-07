#pragma once

/**
 * @file examples/supersonic_example/supersonic_cylinder.hpp
 * @brief Supersonic cylinder example utilities (Euler ICs and mesh defaults).
 *
 * Implements example problems for the Euler equations and provides utilities
 * to build initial conditions for a supersonic cylinder case. The DG operator
 * solves:
 *
 *   (u_t, v)_T - (F(u), grad v)_T + <Fhat(u, n), [[v]]>_F = 0.
 *
 * DGHyperbolicConservationLaws::Mult applies:
 *
 *   u -> M^{-1}(-DF(u) + NF(u)),
 *
 * where M is the mass matrix, DF is the weak divergence of flux, and NF is the
 * interface flux. The inverse of the mass matrix is computed element-wise via
 * the block-diagonal structure, and flux terms use HyperbolicFormIntegrator.
 */

#include "mfem.hpp"

namespace mfem
{

///// Mesh Utilities

/// @brief Default mesh filename for the supersonic cylinder case.
inline const char* SupersonicCylinderDefaultMeshPath()
{
   return "supersonic_cylinder.msh";
}

/// @brief Euler initial condition for the supersonic cylinder case.
VectorFunctionCoefficient EulerInitialCondition()
{
   return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2, "");
      const real_t density = 1.4;
      const real_t velocity_x = 3.0;
      const real_t velocity_y = 0.0;
      const real_t energy = 8.8;

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = density * velocity_y;
      y(3) = energy;
   });
}

/// @brief Euler initial condition for a uniform inflow state.
inline VectorFunctionCoefficient EulerInitialCondition(const Vector &Uin)
{
   return VectorFunctionCoefficient(4, [Uin](const Vector &, Vector &y)
   {
      y = Uin;
   });
}



} // namespace mfem
