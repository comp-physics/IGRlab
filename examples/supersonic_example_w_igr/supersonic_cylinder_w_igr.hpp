#pragma once

/**
 * @file examples/supersonic_example_w_igr/supersonic_cylinder_w_igr.hpp
 * @brief Supersonic cylinder example utilities with IGR support.
 *
 * Implements example problems for the Euler equations and provides utilities
 * to build initial conditions for the supersonic cylinder + IGR case. The DG
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

/// @brief Default mesh filename for the supersonic cylinder + IGR case.
inline const char* SupersonicCylinderIGRDefaultMeshPath()
{
   return "supersonic_cylinder.msh";
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
 * @brief Smoothed IC for the supersonic cylinder: velocity goes from 0 at
 *        the cylinder to freestream far away, radially smoothly.
 *
 * We keep density and energy from Uin *uniform everywhere* and only modify
 * the velocity (i.e. momentum components).
 *
 * Model:
 *   - Let (xc, yc) be the cylinder center, with radius r_cyl.
 *   - Define r = distance from (xc, yc).
 *   - For r <= r_cyl       : S(r) = 0      (stagnant right at the cylinder)
 *   - For r >  r_cyl       : S(r) = 1 - exp(-((r - r_cyl)/delta)^2)
 *                             (smoothly transitions to 1 as r increases)
 *
 * Then:
 *   u(x,y)  = S(r) * u_inf
 *   v(x,y)  = S(r) * v_inf
 *   rho(x,y) = rho_inf (from Uin)
 *   E(x,y)   = E_inf   (from Uin)
 *
 * Parameters:
 *   - Uin      : conservative freestream state [ρ, ρu_x, ρu_y, E]
 *   - xc, yc   : cylinder center in physical coordinates
 *   - r_cyl    : cylinder radius
 *   - delta    : radial smoothing length scale (how quickly you ramp to free-stream)
 */
inline mfem::VectorFunctionCoefficient SupersonicCylinderSmoothedVelocityIC(
   const mfem::Vector &Uin,
   double xc, double yc,
   double r_cyl,
   double delta)
{
   MFEM_VERIFY(Uin.Size() >= 4,
               "Uin must contain at least [rho, rho*u_x, rho*u_y, E]");
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");
   MFEM_VERIFY(r_cyl >= 0.0, "r_cyl must be non-negative.");

   // Decode freestream conservative state
   const double rho_inf   = Uin(0);
   const double rhoux_inf = Uin(1);
   const double rhouy_inf = Uin(2);
   const double E_inf     = Uin(3);

   const double ux_inf = rhoux_inf / rho_inf;
   const double uy_inf = rhouy_inf / rho_inf;

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &y)
      {
         y.SetSize(4);

         const double X = x(0);
         const double Y = x(1);

         const double dx = X - xc;
         const double dy = Y - yc;
         const double r  = std::sqrt(dx*dx + dy*dy);

         // Radial blending factor S(r): 0 at cylinder, →1 far away
         double S = 1.0;
         if (r <= r_cyl)
         {
            S = 0.0;
         }
         else
         {
            const double s = (r - r_cyl) / delta;
            // Smooth transition: S ~ 0 just outside r_cyl, →1 as r grows
            double val = 1.0 - std::exp(-s*s);
            // Just in case of any numerical weirdness, clamp to [0,1]
            if (val < 0.0) val = 0.0;
            if (val > 1.0) val = 1.0;
            S = val;
         }

         const double ux = S * ux_inf;
         const double uy = S * uy_inf;

         // Leave ρ and E as in Uin; only change the momentum components
         y(0) = rho_inf;
         y(1) = rho_inf * ux;
         y(2) = rho_inf * uy;
         y(3) = E_inf;
      });
}

/////// Additional initial conditions to try and stabilize solution

/**
 * @brief Smooth shock-tube-like IC: left state = Uin, right state = quiescent.
 *
 * Left side (x << x0): U = Uin
 * Right side (x >> x0): same rho and p as Uin, but u=0 (quiescent), and E reset accordingly.
 *
 * Transition uses theta(x) = 0.5*(1 - tanh((x - x0)/delta)).
 *
 * @param Uin   Conservative left state [rho, rho*u..., E]
 * @param gamma Ratio of specific heats
 * @param x0    Shock location (in x)
 * @param delta Smoothing thickness (in x)
 */
inline mfem::VectorFunctionCoefficient SmoothShockIC_FromUinToQuiescent(
   const mfem::Vector &Uin,
   const double gamma,
   const double x0,
   const double delta)
{
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");
   const int num_eq = Uin.Size();
   const int dim = num_eq - 2;
   MFEM_VERIFY(dim == 2 || dim == 3, "Uin must be size 4 (2D) or 5 (3D).");

   const double rhoL = Uin(0);

   // Left velocity
   mfem::Vector uL(dim);
   for (int d = 0; d < dim; ++d) { uL(d) = Uin(1 + d) / rhoL; }

   // Left pressure from conservative state
   double mom2 = 0.0;
   for (int d = 0; d < dim; ++d) { mom2 += Uin(1 + d) * Uin(1 + d); }
   const double kin = 0.5 * mom2 / rhoL;
   const double pL  = (gamma - 1.0) * (Uin(1 + dim) - kin);

   MFEM_VERIFY(pL > 0.0, "Uin implies non-positive pressure. Check Uin(3/4) vs velocity.");

   // Right state: quiescent, same rho and p
   const double rhoR = rhoL;
   const double pR   = pL;

   return mfem::VectorFunctionCoefficient(
      num_eq,
      [=](const mfem::Vector &x, mfem::Vector &U)
      {
         U.SetSize(num_eq);

         // Smooth step: theta~1 on left, ~0 on right
         const double s = (x(0) - x0) / delta;
         const double theta = 0.5 * (1.0 - std::tanh(s));

         // Blend primitives (rho, u, p)
         const double rho = theta * rhoL + (1.0 - theta) * rhoR;
         const double p   = theta * pL   + (1.0 - theta) * pR;

         double u2 = 0.0;
         U(0) = rho;
         for (int d = 0; d < dim; ++d)
         {
            const double ud = theta * uL(d) + (1.0 - theta) * 0.0; // quiescent on right
            U(1 + d) = rho * ud;
            u2 += ud * ud;
         }

         // Total energy density
         U(1 + dim) = p / (gamma - 1.0) + 0.5 * rho * u2;
      });
}

inline mfem::Vector MakeEulerCons2D(double rho, double u, double v,
                                   double p, double gamma)
{
   mfem::Vector U(4);
   U(0) = rho;
   U(1) = rho * u;
   U(2) = rho * v;
   U(3) = p/(gamma - 1.0) + 0.5 * rho * (u*u + v*v);
   return U;
}

inline mfem::VectorFunctionCoefficient SmoothBlastIC_2D(
   const mfem::Vector &Uin,   // left state (post-shock / driven state)
   const mfem::Vector &Uamb,  // right state (still chamber / ambient)
   const double gamma,
   const double x0,
   const double delta)
{
   MFEM_VERIFY(Uin.Size() == 4 && Uamb.Size() == 4, "2D Euler expects size 4.");
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");

   auto pressure = [&](const mfem::Vector &U) -> double
   {
      const double rho = U(0);
      const double rhou = U(1), rhov = U(2), E = U(3);
      const double kin = 0.5 * (rhou*rhou + rhov*rhov) / rho;
      return (gamma - 1.0) * (E - kin);
   };

   // Left primitives
   const double rhoL = Uin(0);
   const double uL   = Uin(1) / rhoL;
   const double vL   = Uin(2) / rhoL;
   const double pL   = pressure(Uin);
   MFEM_VERIFY(pL > 0.0, "Uin implies non-positive pressure.");

   // Right primitives
   const double rhoR = Uamb(0);
   const double uR   = Uamb(1) / rhoR;
   const double vR   = Uamb(2) / rhoR;
   const double pR   = pressure(Uamb);
   MFEM_VERIFY(pR > 0.0, "Uamb implies non-positive pressure.");

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &U)
      {
         U.SetSize(4);

         // theta~1 on left, ~0 on right
         const double s = (x(0) - x0) / delta;
         const double theta = 0.5 * (1.0 - std::tanh(s));

         const double rho = theta * rhoL + (1.0 - theta) * rhoR;
         const double u   = theta * uL   + (1.0 - theta) * uR;
         const double v   = theta * vL   + (1.0 - theta) * vR;
         const double p   = theta * pL   + (1.0 - theta) * pR;

         U(0) = rho;
         U(1) = rho * u;
         U(2) = rho * v;
         U(3) = p/(gamma - 1.0) + 0.5 * rho * (u*u + v*v);
      });
}


inline double PressureFromCons2D(const mfem::Vector &U, double gamma)
{
   const double rho  = U(0);
   const double rhou = U(1);
   const double rhov = U(2);
   const double E    = U(3);
   const double kin  = 0.5 * (rhou*rhou + rhov*rhov) / rho;
   return (gamma - 1.0) * (E - kin);
}

/**
 * @brief Inflow-buffer IC: U = Uin near the inlet, then smoothly transitions
 *        to a quiescent ambient state inside the domain.
 *
 * Left side (x << x_s): U = Uin
 * Right side (x >> x_s): U = Uamb_quiescent = [rho_amb, 0, 0, p_amb/(gamma-1)]
 *
 * Blending is done in primitives and E is recomputed.
 *
 * @param Uin     Inlet conservative state [rho, rho*u, rho*v, E]
 * @param gamma   Specific heat ratio
 * @param x_s     Center of the transition (move this inward from the inlet)
 * @param delta   Smoothing thickness (pick ~2–4 elements)
 * @param rho_amb Ambient density (<=0 => use rho from Uin)
 * @param p_amb   Ambient pressure (<=0 => use p implied by Uin)
 */
inline mfem::VectorFunctionCoefficient InflowBufferIC_2D(
   const mfem::Vector &Uin,
   double gamma,
   double x_s,
   double delta,
   double rho_amb = -1.0,
   double p_amb   = -1.0)
{
   MFEM_VERIFY(Uin.Size() == 4, "2D Euler expects Uin size 4.");
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");

   // Left primitives from Uin
   const double rhoL = Uin(0);
   const double uL   = Uin(1) / rhoL;
   const double vL   = Uin(2) / rhoL;
   const double pL   = PressureFromCons2D(Uin, gamma);
   MFEM_VERIFY(pL > 0.0, "Uin implies non-positive pressure.");

   // Ambient (still) primitives
   const double rhoR = (rho_amb > 0.0) ? rho_amb : rhoL;
   const double pR   = (p_amb   > 0.0) ? p_amb   : pL;
   const double uR   = 0.0;
   const double vR   = 0.0;

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &U)
      {
         U.SetSize(4);

         // theta ~ 1 on left, ~0 on right
         const double theta = 0.5 * (1.0 - std::tanh((x(0) - x_s) / delta));

         const double rho = theta * rhoL + (1.0 - theta) * rhoR;
         const double u   = theta * uL   + (1.0 - theta) * uR;
         const double v   = theta * vL   + (1.0 - theta) * vR;
         const double p   = theta * pL   + (1.0 - theta) * pR;

         U(0) = rho;
         U(1) = rho * u;
         U(2) = rho * v;
         U(3) = p/(gamma - 1.0) + 0.5 * rho * (u*u + v*v);
      });
}

// Compute the quiescent ambient (rhoR,pR) that connects to Uin by a right-going
// isentropic simple wave in x, ending at uR=0.
inline void AmbientFromUin_IsentropicSimpleWaveToRest_2D(
   const mfem::Vector &Uin, double gamma,
   double &rhoR, double &pR)
{
   MFEM_VERIFY(Uin.Size() == 4, "Uin must be size 4 for 2D Euler.");

   const double rhoL = Uin(0);
   const double uL   = Uin(1) / rhoL;  // x-velocity component
   const double pL   = PressureFromCons2D(Uin, gamma);
   MFEM_VERIFY(pL > 0.0, "Uin implies non-positive pressure.");

   const double aL = std::sqrt(gamma * pL / rhoL);

   // Right-going simple wave: J- constant, enforce uR = 0
   const double aR = aL - 0.5 * (gamma - 1.0) * uL;
   MFEM_VERIFY(aR > 0.0,
               "Requested uR=0 is too far from Uin for an isentropic simple wave (aR<=0).");

   // Constant entropy: K = p / rho^gamma
   const double K = pL / std::pow(rhoL, gamma);

   rhoR = std::pow((aR*aR) / (gamma * K), 1.0/(gamma - 1.0));
   pR   = K * std::pow(rhoR, gamma);
}

// Your smooth inflow-buffer IC (left Uin -> right quiescent ambient)
inline mfem::VectorFunctionCoefficient SmoothInflowBufferIC_2D(
   const mfem::Vector &Uin,
   const double gamma,
   const double rho_amb,
   const double p_amb,
   const double x_s,
   const double delta)
{
   MFEM_VERIFY(Uin.Size() == 4, "2D Euler expects Uin size 4.");
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");

   const double rhoL = Uin(0);
   const double uL   = Uin(1) / rhoL;
   const double vL   = Uin(2) / rhoL;
   const double pL   = PressureFromCons2D(Uin, gamma);
   MFEM_VERIFY(pL > 0.0, "Uin implies non-positive pressure.");

   const double rhoR = rho_amb;
   const double pR   = p_amb;
   const double uR   = 0.0;
   const double vR   = 0.0;

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &U)
      {
         U.SetSize(4);
         const double theta = 0.5 * (1.0 - std::tanh((x(0) - x_s) / delta)); // 1 left, 0 right

         const double rho = theta*rhoL + (1.0-theta)*rhoR;
         const double u   = theta*uL   + (1.0-theta)*uR;
         const double v   = theta*vL   + (1.0-theta)*vR;
         const double p   = theta*pL   + (1.0-theta)*pR;

         U(0) = rho;
         U(1) = rho * u;
         U(2) = rho * v;
         U(3) = p/(gamma - 1.0) + 0.5 * rho * (u*u + v*v);
      });
}

// Solve for upstream (quiescent) state U1 and shock speed Vs,
// given downstream/post-shock state U2 = Uin, assuming upstream u1=v1=0,
// and a *normal shock in x*.
// This enforces full Rankine–Hugoniot consistency => only one shock.
inline void UpstreamQuiescentFromPostShock_2D(
   const Vector &U2, double gamma,
   Vector &U1, double &Vs)
{
   MFEM_VERIFY(U2.Size() == 4, "U2 must be [rho, rhou, rhov, E] (2D).");

   const double rho2 = U2(0);
   const double u2x  = U2(1) / rho2;
   const double u2y  = U2(2) / rho2;
   const double p2   = PressureFromCons2D(U2, gamma);

   MFEM_VERIFY(p2 > 0.0, "Post-shock state implies non-positive pressure.");

   // For a single normal shock into a fully resting upstream state,
   // tangential velocity should be ~0, otherwise you'll generate extra shear/contact content.
   MFEM_VERIFY(std::abs(u2y) < 1e-12,
               "u2y != 0: a single normal shock into quiescent upstream is impossible without extra waves.");

   const double H2 = (gamma/(gamma-1.0)) * (p2/rho2); // static enthalpy

   // Residual function for Vs using RH:
   // 1) mass:   rho1*Vs = rho2*(Vs - u2x) => rho1 = rho2*(Vs-u2x)/Vs
   // 2) mom:    p1 + rho1*Vs^2 = p2 + rho2*(Vs-u2x)^2 => p1 = ...
   // 3) energy: h1 + 0.5*Vs^2 = h2 + 0.5*(Vs-u2x)^2  with h = gamma/(gamma-1)*p/rho
   auto residual = [&](double V) -> double
   {
      const double u1s = V;          // upstream speed in shock frame (since u1=0)
      const double u2s = V - u2x;    // downstream speed in shock frame

      if (u2s <= 0.0) { return 1e300; } // invalid

      const double rho1 = rho2 * (u2s / u1s);
      if (rho1 <= 0.0) { return 1e300; }

      const double p1 = p2 + rho2*u2s*u2s - rho1*u1s*u1s;
      if (p1 <= 0.0) { return 1e300; }

      const double H1 = (gamma/(gamma-1.0)) * (p1/rho1);

      return (H1 + 0.5*u1s*u1s) - (H2 + 0.5*u2s*u2s);
   };

   // Bracket Vs: must be > u2x (shock outruns downstream)
   double lo = std::max(u2x + 1e-6, 1e-3);
   double hi = lo * 2.0;

   double rlo = residual(lo);
   double rhi = residual(hi);

   // Expand hi until sign change
   for (int k = 0; k < 60 && (rlo*rhi > 0.0 || !std::isfinite(rhi)); ++k)
   {
      hi *= 2.0;
      rhi = residual(hi);
   }
   MFEM_VERIFY(std::isfinite(rlo) && std::isfinite(rhi) && rlo*rhi <= 0.0,
               "Failed to bracket shock speed Vs from given U2. Check U2 is a plausible post-shock state.");

   // Bisection
   for (int it = 0; it < 80; ++it)
   {
      const double mid = 0.5*(lo + hi);
      const double rm  = residual(mid);
      if (!std::isfinite(rm)) { lo = mid; continue; }

      if (rlo * rm <= 0.0) { hi = mid; rhi = rm; }
      else                 { lo = mid; rlo = rm; }
   }
   Vs = 0.5*(lo + hi);

   // Build U1
   const double u1s = Vs;
   const double u2s = Vs - u2x;

   const double rho1 = rho2 * (u2s / u1s);
   const double p1   = p2 + rho2*u2s*u2s - rho1*u1s*u1s;

   U1.SetSize(4);
   U1 = 0.0;
   U1(0) = rho1;
   U1(1) = 0.0;
   U1(2) = 0.0;
   U1(3) = p1/(gamma - 1.0); // u=v=0
}

/**
 * @brief Smooth single-shock IC: left=U2 (your Uin), right=upstream quiescent computed from RH.
 *
 * Only one admissible shock (mollified), controllable thickness delta, centered at x_s.
 * Blend in primitives + recompute E for clean EOS consistency.
 */
inline VectorFunctionCoefficient SmoothSingleShockIC_FromUin_2D(
   const Vector &Uin, double gamma,
   double x_s, double delta)
{
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");

   Vector Uup;
   double Vs = 0.0;
   UpstreamQuiescentFromPostShock_2D(Uin, gamma, Uup, Vs);

   // Left primitives
   const double rhoL = Uin(0);
   const double uL   = Uin(1) / rhoL;
   const double vL   = Uin(2) / rhoL;
   const double pL   = PressureFromCons2D(Uin, gamma);

   // Right primitives (quiescent upstream)
   const double rhoR = Uup(0);
   const double uR   = 0.5;
   const double vR   = 0.0;
   const double pR   = PressureFromCons2D(Uup, gamma); // = (gamma-1)*E

   return VectorFunctionCoefficient(
      4,
      [=](const Vector &x, Vector &U)
      {
         U.SetSize(4);
         const double theta = 0.5 * (1.0 - std::tanh((x(0) - x_s)/delta)); // 1 left, 0 right

         const double rho = theta*rhoL + (1.0-theta)*rhoR;
         const double u   = theta*uL   + (1.0-theta)*uR;
         const double v   = theta*vL   + (1.0-theta)*vR;
         const double p   = theta*pL   + (1.0-theta)*pR;

         U(0) = rho;
         U(1) = rho*u;
         U(2) = rho*v;
         U(3) = p/(gamma-1.0) + 0.5*rho*(u*u + v*v);
      });
}


/**
 * @brief Build inlet and ambient (still) conservative states from inlet primitives.
 *
 * Inlet (freestream) primitives are given by (rho_in, u_in, v_in, p_in).
 *
 * Ambient state is chosen to mimic the paper’s setup:
 *   same rho and p as inlet, but u=v=0.
 *
 * Outputs:
 *   - Uin  = [rho, rho*u, rho*v, E]
 *   - Uamb = [rho, 0, 0, p/(gamma-1)]
 */
inline void BuildInletAndAmbientStates_FromPrimitives(
   double rho_in, double u_in, double v_in, double p_in,
   double gamma,
   mfem::Vector &Uin,
   mfem::Vector &Uamb)
{
   Uin  = MakeEulerCons2D(rho_in, u_in, v_in, p_in, gamma);
   Uamb = MakeEulerCons2D(rho_in, 0.0, 0.0, p_in, gamma);
}

/**
 * @brief Smoothly transitions from inlet state Uin (left) to ambient state Uamb (right)
 *        using a tanh profile centered at x_s with thickness delta.
 *
 * Blends in primitives and recomputes E for EOS consistency.
 */
inline mfem::VectorFunctionCoefficient SmoothInflowBufferIC_2D(
   const mfem::Vector &Uin,
   const mfem::Vector &Uamb,
   double gamma,
   double x_s,
   double delta)
{
   MFEM_VERIFY(Uin.Size() == 4 && Uamb.Size() == 4, "Expected size 4 states.");
   MFEM_VERIFY(delta > 0.0, "delta must be positive.");

   // Left primitives
   const double rhoL = Uin(0);
   const double uL   = Uin(1) / rhoL;
   const double vL   = Uin(2) / rhoL;
   const double pL   = PressureFromCons2D(Uin, gamma);
   MFEM_VERIFY(pL > 0.0, "Uin implies non-positive pressure.");

   // Right primitives (ambient)
   const double rhoR = Uamb(0);
   const double uR   = Uamb(1) / rhoR; // should be 0
   const double vR   = Uamb(2) / rhoR; // should be 0
   const double pR   = PressureFromCons2D(Uamb, gamma);
   MFEM_VERIFY(pR > 0.0, "Uamb implies non-positive pressure.");

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &U)
      {
         U.SetSize(4);

         // theta ~ 1 on left, ~0 on right
         const double theta = 0.5 * (1.0 - std::tanh((x(0) - x_s) / delta));

         const double rho = theta*rhoL + (1.0-theta)*rhoR;
         const double u   = theta*uL   + (1.0-theta)*uR;
         const double v   = theta*vL   + (1.0-theta)*vR;
         const double p   = theta*pL   + (1.0-theta)*pR;

         U(0) = rho;
         U(1) = rho*u;
         U(2) = rho*v;
         U(3) = p/(gamma - 1.0) + 0.5*rho*(u*u + v*v);
      });
}


} // namespace mfem
