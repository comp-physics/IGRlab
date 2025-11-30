//                  MFEM Example 18 - Serial/Parallel Shared Code
//                      (Implementation of Time-dependent DG Operator)
//
// This code provide example problems for the Euler equations and implements
// the time-dependent DG operator given by the equation:
//
//            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
//
// This operator is designed for explicit time stepping methods. Specifically,
// the function DGHyperbolicConservationLaws::Mult implements the following
// transformation:
//
//                             u ↦ M⁻¹(-DF(u) + NF(u))
//
// where M is the mass matrix, DF is the weak divergence of flux, and NF is the
// interface flux. The inverse of the mass matrix is computed element-wise by
// leveraging the block-diagonal structure of the DG mass matrix. Additionally,
// the flux-related terms are computed using the HyperbolicFormIntegrator.
//
// The maximum characteristic speed is determined for each time step. For more
// details, refer to the documentation of DGHyperbolicConservationLaws::Mult.
//

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

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
         return Mesh("supersonic_cylinder.msh");
      case 2:
         return Mesh("supersonic_cylinder.msh");
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition
std::function<void(const Vector&, Vector&)> GetMovingVortexInit(
   const real_t radius, const real_t Minf, const real_t beta,
   const real_t gas_constant, const real_t specific_heat_ratio)
{
   return [specific_heat_ratio,
           gas_constant, Minf, radius, beta](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2, "");

      const real_t xc = 2.0, yc = 0.0;

      // Nice units
      const real_t vel_inf = 1.;
      const real_t den_inf = 1.;

      // Derive remainder of background state from this and Minf
      const real_t pres_inf = (den_inf / specific_heat_ratio) *
                              (vel_inf / Minf) * (vel_inf / Minf);
      const real_t temp_inf = pres_inf / (den_inf * gas_constant);

      real_t r2rad = 0.0;
      r2rad += (x(0) - xc) * (x(0) - xc);
      r2rad += (x(1) - yc) * (x(1) - yc);
      r2rad /= (radius * radius);

      const real_t shrinv1 = 1.0 / (specific_heat_ratio - 1.);

      const real_t velX =
         vel_inf * (1 - beta * (x(1) - yc) / radius * std::exp(-0.5 * r2rad));
      const real_t velY =
         vel_inf * beta * (x(0) - xc) / radius * std::exp(-0.5 * r2rad);
      const real_t vel2 = velX * velX + velY * velY;

      const real_t specific_heat =
         gas_constant * specific_heat_ratio * shrinv1;
      const real_t temp = temp_inf - 0.5 * (vel_inf * beta) *
                          (vel_inf * beta) / specific_heat *
                          std::exp(-r2rad);

      const real_t den = den_inf * std::pow(temp / temp_inf, shrinv1);
      const real_t pres = den * gas_constant * temp;
      const real_t energy = shrinv1 * pres / den + 0.5 * vel2;

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = den * energy;
   };
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


} // namespace mfem
