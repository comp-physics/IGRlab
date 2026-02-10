#pragma once

/**
 * @file examples/stationary_shock/stationary_shock.hpp
 * @brief Stationary shock and shock-vortex example utilities.
 *
 * Implements example problems for the Euler equations and provides utilities
 * to build initial conditions for stationary shock and shock-vortex cases. The
 * DG operator solves:
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
#include "utils/math/constants.hpp"

namespace mfem
{

///// Mesh Utilities

/// @brief Default mesh filename for the stationary shock case.
inline const char* StationaryShockDefaultMeshPath()
{
   return "stationary_shock.msh";
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
 * @brief Shock–vortex interaction initial condition on Ω = [0,1]×[0,1].
 *
 * Mesh is assumed uniform with mx×my elements; mx,my only enter through
 * the shock smoothing width δ = 3 * smoothing_scale * min(1/mx, 1/my).
 *
 * Boundary conditions (for your driver, not enforced here):
 *   - x = 0: Dirichlet (inflow with upstream vortex state)
 *   - x = 1: homogeneous Neumann (do-nothing / outflow)
 *   - y = 0,1: periodic
 *
 * Configuration (dimensionless):
 *   - Upstream base: ρ₁ = 1, p₁ = 1
 *   - γ: specific heat ratio (e.g. 1.4)
 *   - M₁ = 1.5
 *   - Normal shock at x_s = 0.6 with smooth step S(x)
 *   - Isentropic vortex centered at (x_v, y_v) = (0.3, 0.5) with strength ε = 5
 *
 * Returns conservative variables [ρ, ρu_x, ρu_y, E].
 */
/// @brief Shock-vortex interaction initial condition.
std::function<void(const Vector&, Vector&)> GetShockVortexInteractionInit(
   const real_t gamma,
   const int mx = 32,
   const int my = 32,
   const real_t smoothing_scale = 1.0)
{
   // --- Upstream base state U1 ---
   const real_t M1   = 1.5;
   const real_t rho1 = 1.0;
   const real_t p1   = 1.0;
   const real_t T1   = p1 / rho1; // R = 1
   const real_t c1   = std::sqrt(gamma * p1 / rho1);

   const real_t u1x  = M1 * c1;
   const real_t u1y  = 0.0;
   const real_t v1sq = u1x * u1x + u1y * u1y;

   // We won't use E1 explicitly in the mixing, but it’s here for completeness:
   const real_t E1_density = p1 / (gamma - 1.0) + 0.5 * rho1 * v1sq;
   (void)E1_density;

   // --- Downstream state U2 from normal shock relations ---
   const real_t M1sq    = M1 * M1;
   const real_t invM1sq = 1.0 / M1sq;

   // Given in the statement:
   //   ρ₂ = ρ₁ * (γ+1) / (γ-1 + 2/M₁²)
   const real_t rho2 =
      rho1 * (gamma + 1.0) / (gamma - 1.0 + 2.0 * invM1sq);

   // u₂x = u₁x * ρ₁ / ρ₂
   const real_t u2x = u1x * (rho1 / rho2);
   const real_t u2y = 0.0;

   // Standard normal-shock pressure jump:
   //   p₂ / p₁ = 1 + (2γ/(γ+1)) (M₁² - 1)
   const real_t p2 =
      p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (M1sq - 1.0));

   const real_t v2sq = u2x * u2x + u2y * u2y;
   const real_t E2_density = p2 / (gamma - 1.0) + 0.5 * rho2 * v2sq;
   (void)E2_density;

   // Momentum forms of U1, U2
   const real_t mu1x = rho1 * u1x;
   const real_t mu1y = rho1 * u1y;

   const real_t mu2x = rho2 * u2x;
   const real_t mu2y = rho2 * u2y;
   (void)mu1x; (void)mu1y; // base upstream momentum not used directly in mixing

   // --- Shock smoothing S(x) ---
   const real_t xs = 0.6;
   const real_t inv_mx = 1.0 / static_cast<real_t>(mx);
   const real_t inv_my = 1.0 / static_cast<real_t>(my);
   const real_t smooth = std::max(smoothing_scale, static_cast<real_t>(1.0e-6));
   const real_t delta = 3.0 * smooth * std::min(inv_mx, inv_my);

   // --- Vortex parameters (upstream) ---
   const real_t xv = 0.3;
   const real_t yv = 0.5;
   const real_t eps = 5.0;

   const real_t pi = static_cast<real_t>(dg_sim::math::kPi);
   const real_t two_pi = static_cast<real_t>(dg_sim::math::kTwoPi);

   const real_t eps_over_2pi = eps / two_pi;
   const real_t coeff_Tp =
      - (gamma - 1.0) * eps * eps / (8.0 * gamma * pi * pi);

   return [=](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2,
                  "Shock–vortex IC is implemented for 2D coordinates.");
      y.SetSize(4);

      const real_t xx = x(0);
      const real_t yy = x(1);

      // Smooth shock indicator S(x)
      const real_t S = 0.5 * (1.0 + std::tanh((xx - xs) / delta));

      // --- Upstream vortex perturbations (isentropic vortex) ---
      const real_t dxv = xx - xv;
      const real_t dyv = yy - yv;
      const real_t r2  = dxv * dxv + dyv * dyv;

      const real_t exp_half = std::exp((1.0 - r2) / 2.0);
      const real_t exp_full = std::exp(1.0 - r2);

      const real_t uprime_x = -eps_over_2pi * dyv * exp_half;
      const real_t uprime_y =  eps_over_2pi * dxv * exp_half;

      const real_t Tprime   = coeff_Tp * exp_full;

      const real_t T  = T1 + Tprime;
      const real_t Tr = T / T1;

      // Isentropic relations (upstream vortex state)
      const real_t rho_up = rho1 * std::pow(Tr, 1.0 / (gamma - 1.0));
      const real_t p_up   = p1   * std::pow(Tr, gamma / (gamma - 1.0));

      const real_t u_up_x = u1x + uprime_x;
      const real_t u_up_y = u1y + uprime_y;

      const real_t mu_up_x = rho_up * u_up_x;
      const real_t mu_up_y = rho_up * u_up_y;

      // --- Mix upstream vortex state with downstream state via S(x) ---
      const real_t rho = S * rho2      + (1.0 - S) * rho_up;
      const real_t mux = S * mu2x      + (1.0 - S) * mu_up_x;
      const real_t muy = S * mu2y      + (1.0 - S) * mu_up_y;
      const real_t p   = S * p2        + (1.0 - S) * p_up;

      const real_t mom2 = mux * mux + muy * muy;
      const real_t E    = p / (gamma - 1.0) + 0.5 * mom2 / rho;

      y(0) = rho;
      y(1) = mux;
      y(2) = muy;
      y(3) = E;
   };
}

/**
 * @brief Convenience wrapper: shock–vortex IC as a VectorFunctionCoefficient.
 */
/// @brief Euler initial condition for the shock-vortex interaction case.
VectorFunctionCoefficient ShockVortexInteractionInitialCondition(
   const real_t gamma,
   const int mx = 32,
   const int my = 32,
   const real_t smoothing_scale = 1.0)
{
   auto ic_fun = GetShockVortexInteractionInit(gamma, mx, my, smoothing_scale);
   return mfem::VectorFunctionCoefficient(4, ic_fun);
}

/**
 * @brief Shock–vortex interaction IC (Inoue–Hattori canonical case).
 *
 * Domain (physical): X ∈ [xL, xR] = [-30, 10], Y ∈ [yB, yT] = [-20, 20].
 * Mesh is assumed uniform with mx × my elements.
 *
 * Parameters:
 *  - gamma : specific heat ratio (e.g., 1.4)
 *  - mx,my : number of cells in X and Y (for shock smoothing width δ_phys)
 *  - Ms    : incident shock Mach number (default 1.2)
 *  - Mv    : vortex Mach number / strength (default 0.25)
 *  - smoothing_scale : scale for shock thickness (1.0 = default)
 *
 * Returns u₀(X,Y) = [ρ, ρu_x, ρu_y, E].
 */
/// @brief Shock-vortex IC following Inoue/Hattori setup.
inline std::function<void(const Vector&, Vector&)> GetShockVortexInoueHattoriInit(
   const real_t gamma,
   const int mx,
   const int my,
   const real_t Ms = 1.2,
   const real_t Mv = 0.25,
   const real_t smoothing_scale = 1.0)
{
   // --- Physical domain bounds (X,Y) ---
   const real_t xL = -30.0;
   const real_t xR =  10.0;
   const real_t yB = -20.0;
   const real_t yT =  20.0;

   const real_t Lx = xR - xL; // 40
   const real_t Ly = yT - yB; // 40

   const real_t dX = Lx / static_cast<real_t>(mx);
   const real_t dY = Ly / static_cast<real_t>(my);
   const real_t smooth = std::max(smoothing_scale, static_cast<real_t>(1.0e-6));
   const real_t delta_phys = 2.0 * smooth * std::min(dX, dY); // smooth shock width

   // --- Upstream base state U1 (to the right, X > 0) ---
   const real_t rho1 = 1.0;
   const real_t p1   = 1.0;
   const real_t M1   = Ms;

   const real_t c1   = std::sqrt(gamma * p1 / rho1);
   // Flow from right -> left (negative u1x)
   const real_t u1x  = -M1 * c1;
   const real_t u1y  = 0.0;

   const real_t v1sq = u1x * u1x + u1y * u1y;
   const real_t E1_density = p1 / (gamma - 1.0) + 0.5 * rho1 * v1sq;
   (void)E1_density; // not used explicitly, but here for completeness

   // --- Downstream state U2 (to the left, X < 0) from normal-shock relations ---
   const real_t M1sq    = M1 * M1;
   const real_t invM1sq = 1.0 / M1sq;

   // ρ₂ / ρ₁ = (γ+1) / (γ - 1 + 2/M₁²)
   const real_t rho2 =
      rho1 * (gamma + 1.0) / (gamma - 1.0 + 2.0 * invM1sq);

   // u₂x = u₁x * ρ₁ / ρ₂   (mass conservation)
   const real_t u2x = u1x * (rho1 / rho2);
   const real_t u2y = 0.0;

   // p₂ / p₁ = 1 + (2γ/(γ+1)) (M₁² - 1)
   const real_t p2 =
      p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (M1sq - 1.0));

   const real_t v2sq = u2x * u2x + u2y * u2y;
   const real_t E2_density = p2 / (gamma - 1.0) + 0.5 * rho2 * v2sq;
   (void)E2_density;

   const real_t mu2x = rho2 * u2x;
   const real_t mu2y = rho2 * u2y;

   // --- Shock location ---
   const real_t Xs = -10.0; // stationary shock at X=0

   // --- Vortex parameters ---
   const real_t Xv = 50.0; // Remove vortex from space
   const real_t Yv = 0.0;

   // Mv is the canonical vortex Mach number; choose eps = 2π Mv so that
   // u_θ = Mv r exp((1 - r²)/2) matches the standard isentropic vortex.
   const real_t pi = static_cast<real_t>(dg_sim::math::kPi);
   const real_t two_pi = static_cast<real_t>(dg_sim::math::kTwoPi);

   const real_t eps    = two_pi * Mv;
   const real_t eps_over_2pi = eps / two_pi; // = Mv

   // T' coefficient for isentropic vortex
   const real_t coeff_Tp =
      - (gamma - 1.0) * eps * eps / (8.0 * gamma * pi * pi);

   // --- IC lambda in physical coordinates (X,Y) ---
   return [=](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2,
                  "Shock–vortex IC is implemented for 2D coordinates.");
      y.SetSize(4);

      // x(0), x(1) are already physical coordinates (X,Y) from the mesh
      const real_t X = x(0);
      const real_t Y = x(1);

      // Smooth shock indicator S(X):
      //  X << 0  → S ≈ 0 → U ≈ U2 (downstream)
      //  X >> 0  → S ≈ 1 → U ≈ upstream vortex state
      const real_t S = 0.5 * (1.0 + std::tanh((X - Xs) / delta_phys));

      // --- Upstream vortex perturbations (isentropic vortex) ---
      const real_t dXv = X - Xv;
      const real_t dYv = Y - Yv;
      const real_t r2  = dXv * dXv + dYv * dYv;

      const real_t exp_half = std::exp((1.0 - r2) / 2.0);
      const real_t exp_full = std::exp(1.0 - r2);

      const real_t uprime_x = -eps_over_2pi * dYv * exp_half;
      const real_t uprime_y =  eps_over_2pi * dXv * exp_half;

      const real_t Tprime   = coeff_Tp * exp_full;

      const real_t T  = p1 / rho1 + Tprime; // T1 = p1 / rho1 (R=1)
      const real_t T1 = p1 / rho1;
      const real_t Tr = T / T1;

      // Isentropic relations for upstream vortex state
      const real_t rho_up = rho1 * std::pow(Tr, 1.0 / (gamma - 1.0));
      const real_t p_up   = p1   * std::pow(Tr, gamma / (gamma - 1.0));

      const real_t u_up_x = u1x + uprime_x;
      const real_t u_up_y = u1y + uprime_y;

      const real_t mu_up_x = rho_up * u_up_x;
      const real_t mu_up_y = rho_up * u_up_y;

      // --- Mix upstream vortex state with downstream state via S(X) ---
      // For X << 0: S ≈ 0 → (ρ,μ,p) ≈ downstream (ρ₂,μ₂,p₂)
      // For X >> 0: S ≈ 1 → (ρ,μ,p) ≈ upstream vortex state
      const real_t rho = S * rho_up + (1.0 - S) * rho2;
      const real_t mux = S * mu_up_x + (1.0 - S) * mu2x;
      const real_t muy = S * mu_up_y + (1.0 - S) * mu2y;
      const real_t p   = S * p_up   + (1.0 - S) * p2;

      const real_t mom2 = mux * mux + muy * muy;
      const real_t E    = p / (gamma - 1.0) + 0.5 * mom2 / rho;

      y(0) = rho;
      y(1) = mux;
      y(2) = muy;
      y(3) = E;
   };
}

/**
 * @brief Convenience wrapper: Inoue–Hattori shock–vortex IC as a VectorFunctionCoefficient.
 */
/// @brief Euler initial condition for the Inoue/Hattori shock-vortex setup.
inline VectorFunctionCoefficient ShockVortexInoueHattoriInitialCondition(
   const real_t gamma,
   const int mx,
   const int my,
   const real_t Ms = 1.2,
   const real_t Mv = 0.25)
{
   auto ic_fun = GetShockVortexInoueHattoriInit(gamma, mx, my, Ms, Mv);
   return mfem::VectorFunctionCoefficient(4, ic_fun);
}

} // namespace mfem
