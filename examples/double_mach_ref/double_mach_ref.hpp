#pragma once

/**
 * @file examples/double_mach_ref/double_mach_ref.hpp
 * @brief Double Mach reflection example utilities.
 *
 * Implements example problems for the Euler equations and provides utilities
 * to build initial conditions for the double Mach reflection case. The DG
 * operator solves:
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

#include <functional>
#include "mfem.hpp"
#include "BdrHyperbolicFluxIntegrator.hpp"
#include "DGHyperbolicConservationLaws.hpp"
#include "eulerigr.hpp"
#include "fluxfunctions.hpp"
#include "IGR.hpp"

namespace mfem
{

///// Mesh Utilities

/// @brief Default mesh filename for the double Mach reflection case.
inline const char* DoubleMachRefDefaultMeshPath()
{
   return "double_mach_ref.msh";
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
 * @brief Double Mach reflection initial condition (Eyob & Schäfer Example 10).
 *
 * Domain (physical): x1 ∈ [0, Lx] ≈ [0,4], x2 ∈ [0, Ly] ≈ [0,1].
 * Uniform mesh with mx × my elements.
 *
 * The incident shock makes a 60° angle with the x1-axis. Its “foot” is at
 * (x1 = x0, x2 = 0). The signed distance to the shock line is
 *
 *   eps(x1,x2) = x2 - tan(pi/3) * (x1 - x0),
 *
 * and the smooth step is
 *
 *   S(x1,x2) = 0.5 * (1 + tanh(eps / varpi)),
 *
 * where varpi = 3 * min{ dx_phys, 4*dy_phys } (matching the paper’s
 * θ_min = min{1/mx, 4/my} after rescaling).
 *
 * Ahead-of-shock (pre-shock) state Ua:
 *   rho_a = 1.4, u_a = (0,0), p_a = 1.0, E_a = p_a/(γ-1).
 *
 * Behind-shock (post-shock) state Ub from M = 2 normal shock; velocity
 * direction aligned with 60° shock orientation.
 *
 * Conservative IC:
 *   U(x,0) = S * Ua + (1 - S) * Ub
 * in conservative variables [ρ, ρu_x, ρu_y, E].
 */
/// @brief Initial condition for the double Mach reflection problem.
inline mfem::VectorFunctionCoefficient DoubleMachReflectionInitialCondition(
   int mx,
   int my,
   double gamma = 1.4,
   double Lx = 4.0,
   double Ly = 1.0)
{
   // --- Pre-shock (ahead) state Ua ---
   const double rho_a = 1.4;
   const double p_a   = 1.0;
   const double ux_a  = 0.0;
   const double uy_a  = 0.0;
   const double E_a   = p_a / (gamma - 1.0) + 0.5 * rho_a * (ux_a*ux_a + uy_a*uy_a);

   const double mu_a_x = rho_a * ux_a;
   const double mu_a_y = rho_a * uy_a;

   // --- Post-shock (behind) state Ub from M = 2 normal shock ---
   const double M      = 2.0;
   const double M2     = M * M;

   // Density ratio: ρ_b / ρ_a = (γ+1)/(γ-1 + 2/M^2)
   const double rho_b =
      rho_a * (gamma + 1.0) / (gamma - 1.0 + 2.0 / M2);

   // Pressure ratio (standard normal-shock relation)
   const double p_b =
      p_a * (1.0 + 2.0 * gamma / (gamma + 1.0) * (M2 - 1.0));

   // Shock orientation: 60° wrt x1-axis
   const double angle = M_PI / 3.0; // 60 degrees

   // Following the paper: post-shock velocity magnitude proportional to
   // M * (1 - ρ_a/ρ_b); with c_a = sqrt(γ p_a / ρ_a) = 1 here.
   const double u_mag_b = M * (1.0 - rho_a / rho_b);
   const double ux_b    = u_mag_b * std::cos(angle);
   const double uy_b    = -u_mag_b * std::sin(angle);

   const double mu_b_x  = rho_b * ux_b;
   const double mu_b_y  = rho_b * uy_b;
   const double E_b     = p_b / (gamma - 1.0)
                        + 0.5 * rho_b * (ux_b*ux_b + uy_b*uy_b);

   // --- Smooth step parameters ---
   // Physical cell sizes for [0,Lx]×[0,Ly]
   const double dx = Lx / static_cast<double>(mx);
   const double dy = Ly / static_cast<double>(my);

   // Paper uses θ_min = min{1/mx, 4/my}; here we mirror that in physical units.
   const double theta_min = std::min(dx, 4.0 * dy);
   const double varpi     = 3.0 * theta_min;

   // Shock foot location on lower boundary:
   // paper uses x0 = 4/6.
   const double x0 = 4.0 / 6.0;

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &y)
      {
         y.SetSize(4);

         const double x1 = x(0);
         const double x2 = x(1);

         // Signed distance to shock line (eps = 0 is the shock)
         const double tan60 = std::tan(M_PI / 3.0);
         const double eps   = x2 - tan60 * (x1 - x0);

         // Smooth step S in [0,1]; S ≈ 1 ahead, ≈0 behind
         // const double S = 0.5 * (1.0 + std::tanh(eps / varpi));
         const double S = 0.5 * (1.0 - std::tanh(eps / varpi));

         // Conservative blend: U = S*Ua + (1-S)*Ub
         const double rho = S * rho_a   + (1.0 - S) * rho_b;
         const double mux = S * mu_a_x  + (1.0 - S) * mu_b_x;
         const double muy = S * mu_a_y  + (1.0 - S) * mu_b_y;
         const double E   = S * E_a     + (1.0 - S) * E_b;

         y(0) = rho;
         y(1) = mux;
         y(2) = muy;
         y(3) = E;
      });
}

/// @brief Pre-shock conservative state for the double Mach setup.
inline mfem::Vector DoubleMachPreShockState(double gamma = 1.4)
{
   // Ahead-of-shock state from the paper:
   //   ρ_a = 1.4, u_a = (0,0), p_a = 1.0
   const double rho_a = 1.4;
   const double p_a   = 1.0;
   const double ux_a  = 0.0;
   const double uy_a  = 0.0;

   const double E_a = p_a / (gamma - 1.0)
                    + 0.5 * rho_a * (ux_a*ux_a + uy_a*uy_a);

   mfem::Vector Ua(4);
   Ua(0) = rho_a;
   Ua(1) = rho_a * ux_a;
   Ua(2) = rho_a * uy_a;
   Ua(3) = E_a;
   return Ua;
}

/// @brief Post-shock conservative state for the double Mach setup.
inline mfem::Vector DoubleMachPostShockState(double gamma = 1.4,
                                             double M     = 2.0)
{
   // Start from the pre-shock state so everything is consistent with the IC.
   mfem::Vector Ua = DoubleMachPreShockState(gamma);

   const double rho_a = Ua(0);
   const double ux_a  = 0.0;                 // by construction
   const double uy_a  = 0.0;
   const double E_a   = Ua(3);
   const double p_a   = (gamma - 1.0) *
                        (E_a - 0.5 * rho_a * (ux_a*ux_a + uy_a*uy_a));

   const double M2 = M * M;

   // Density ratio from the paper:
   //   ρ_b / ρ_a = (γ+1) / (γ-1 + 2/M²)
   const double rho_b =
      rho_a * (gamma + 1.0) / (gamma - 1.0 + 2.0 / M2);

   // Standard normal-shock pressure ratio:
   //   p_b / p_a = 1 + (2γ/(γ+1)) (M² - 1)
   const double p_b =
      p_a * (1.0 + 2.0 * gamma / (gamma + 1.0) * (M2 - 1.0));

   // Shock orientation: 60° w.r.t. x-axis
   const double angle = M_PI / 3.0;

   // Paper’s formula (Eq. 5.42):
   //   u_b = [cos(π/3), sin(π/3)]ᵀ * M * (1 - ρ_a/ρ_b)
   const double u_mag_b = M * (1.0 - rho_a / rho_b);
   const double ux_b    = u_mag_b * std::cos(angle);
   const double uy_b    = u_mag_b * std::sin(angle);
   const double ub2     = ux_b*ux_b + uy_b*uy_b;

   const double E_b = p_b / (gamma - 1.0) + 0.5 * rho_b * ub2;

   mfem::Vector Ub(4);
   Ub(0) = rho_b;
   Ub(1) = rho_b * ux_b;
   Ub(2) = rho_b * uy_b;
   Ub(3) = E_b;
   return Ub;
}


} // namespace mfem
