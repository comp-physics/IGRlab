#pragma once

/**
 * @file examples/sedov_taylor_blast/sedov_taylor_blast.hpp
 * @brief Sedov-Taylor blast example utilities.
 *
 * Implements example problems for the Euler equations and provides utilities
 * to build initial conditions for the Sedov-Taylor blast case. The DG operator
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

#include <algorithm>
#include <cmath>
#include <functional>

#include "mfem.hpp"

namespace mfem
{

///// Mesh Utilities

/// @brief Default mesh filename for the Sedov-Taylor blast case.
inline const char* SedovTaylorDefaultMeshPath()
{
   return "sedov_taylor.msh";
}

/**
 * @brief Create a uniform Euler initial condition from a given conservative state.
 *
 * @param Uin   Conservative freestream state [ρ, ρu_x, ρu_y, E].
 * @return VectorFunctionCoefficient defining u₀(x) = Uin everywhere.
 *
 * This is intended for initializing the domain consistently with
 * your inlet boundary conditions when you already have Uin in conservative form.
 */
/// @brief Build a uniform Euler initial condition from a conservative state.
VectorFunctionCoefficient EulerInitialCondition(const mfem::Vector &Uin)
{
   MFEM_VERIFY(Uin.Size() >= 4, "Uin must contain at least [rho, rho*u_x, rho*u_y, E]");
   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &, mfem::Vector &y)
      {
         y.SetSize(4);
         y(0) = Uin(0);
         y(1) = Uin(1);
         y(2) = Uin(2);
         y(3) = Uin(3);
      });
}

/**
 * @brief Sedov–Taylor blast initial condition on Ω = [0,1]×[0,1] with a 32×32 mesh.
 *
 * The blast is initialized as a smoothed, radially symmetric deposition of
 * internal energy centered at (1/2, 1/2). We use a Gaussian kernel of width
 * σ = 6 Δ_min and choose α so that the *total added internal energy* equals
 * E_blast (discrete normalization on a 32×32 uniform grid).
 *
 * Background: ρ_bg = 1, u_bg = 0, p_amb = 1.
 *
 * @param gamma     Specific heat ratio γ.
 * @param E_blast   Total blast energy (internal) to deposit (default 1.0).
 * @param mx        Number of cells in x (default 32).
 * @param my        Number of cells in y (default 32).
 *
 * @return A std::function (IC(x,y)) in conservative variables [ρ, ρu_x, ρu_y, E].
 */
/// @brief Sedov-Taylor blast initial condition as a vector function.
std::function<void(const Vector&, Vector&)> GetSedovTaylorBlastInit(
   const real_t gamma,
   const real_t E_blast = 1.0,
   const int mx = 32,
   const int my = 32)
{
   // Domain and mesh
   const real_t Lx = 1.0;
   const real_t Ly = 1.0;

   const real_t dx = Lx / mx;
   const real_t dy = Ly / my;
   const real_t dmin = std::min(dx, dy);
   const real_t sigma = 6.0 * dmin;

   // Blast center
   const real_t xc = 0.5;
   const real_t yc = 0.5;

   // Cell area and Gaussian normalization (discrete sum over cell centers)
   const real_t A_cell = dx * dy;

   real_t sum_eta = 0.0;
   for (int i = 0; i < mx; ++i)
   {
      const real_t x_i = (i + 0.5) * dx; // cell center in x
      const real_t dx_c = x_i - xc;

      for (int j = 0; j < my; ++j)
      {
         const real_t y_j = (j + 0.5) * dy; // cell center in y
         const real_t dy_c = y_j - yc;

         const real_t r = std::sqrt(dx_c*dx_c + dy_c*dy_c);
         const real_t rs = r / sigma;
         const real_t eta = std::exp(-rs * rs);

         sum_eta += eta * A_cell;
      }
   }

   MFEM_VERIFY(sum_eta > 0.0,
               "Sedov–Taylor blast normalization sum_eta must be positive.");

   // Discrete normalization:
   //   E_blast = ∑ ( p_bump / (γ-1) ) A_cell
   //           = α / (γ-1) ∑ η A_cell  =>  α = (γ-1) E_blast / ∑ η A_cell
   const real_t alpha = (gamma - 1.0) * E_blast / sum_eta;

   // Background state
   const real_t rho_bg = 1.0;
   const real_t p_amb  = 1.0;

   return [=](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2,
                  "Sedov–Taylor IC is implemented for 2D coordinates.");

      y.SetSize(4);

      const real_t dx_c = x(0) - xc;
      const real_t dy_c = x(1) - yc;
      const real_t r = std::sqrt(dx_c*dx_c + dy_c*dy_c);
      const real_t rs = r / sigma;

      const real_t eta = std::exp(-rs * rs);
      const real_t p_bump = alpha * eta;

      const real_t rho = rho_bg;
      const real_t ux  = 0.0;
      const real_t uy  = 0.0;
      const real_t p   = p_amb + p_bump;

      // E = p/(γ-1) + ½ ρ |u|²; here u = 0 → E = p/(γ-1)
      const real_t E = p / (gamma - 1.0);

      y(0) = rho;
      y(1) = rho * ux;
      y(2) = rho * uy;
      y(3) = E;
   };
}

/**
 * @brief Convenience wrapper: Sedov–Taylor blast IC as a VectorFunctionCoefficient.
 */
/// @brief Euler initial condition for the Sedov-Taylor blast case.
VectorFunctionCoefficient SedovTaylorBlastInitialCondition(
   const real_t gamma,
   const real_t E_blast = 1.0,
   const int mx = 32,
   const int my = 32)
{
   auto ic_fun = GetSedovTaylorBlastInit(gamma, E_blast, mx, my);
   return mfem::VectorFunctionCoefficient(4, ic_fun);
}

} // namespace mfem
