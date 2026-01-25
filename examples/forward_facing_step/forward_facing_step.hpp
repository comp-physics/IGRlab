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
         return Mesh("forward_facing.msh");
      case 2:
         return Mesh("forward_facing.msh");
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
 * @brief Shock-tube-like IC for forward-facing step:
 *        upstream freestream -> downstream quiescent shocked state,
 *        blended in *conservative* variables to avoid overshoot in ρu_x.
 *
 * Left (upstream) state U1 = Uin:
 *   [rho1, rhou1x, rhou1y, E1]
 *
 * Right (downstream) quiescent shocked state Uq:
 *   - u_q = 0
 *   - rho_q, p_q from normal-shock formulas based on upstream Mach
 *   - E_q = p_q / (gamma - 1)
 *
 * Transition:
 *   X <= x_s - delta_x : U = U1 (pure upstream)
 *   X >= x_s + delta_x : U = Uq (pure quiescent shocked)
 *   in between         : U = (1-S)*U1 + S*Uq  with C^1 smoothstep S(X)
 */
inline mfem::VectorFunctionCoefficient ForwardFacingStepQuiescentShockIC(
   const mfem::Vector &Uin,
   double gamma,
   double step_x,
   double delta_x,
   double x_s = -1.0)
{
   MFEM_VERIFY(Uin.Size() >= 4,
               "Uin must contain at least [rho, rho*u_x, rho*u_y, E]");

   // Default: center the transition halfway to the step
   if (x_s < 0.0)
   {
      x_s = 0.5 * step_x;
   }

   // ---- Upstream conservative state U1 ----
   const double rho1   = Uin(0);
   const double rhou1x = Uin(1);
   const double rhou1y = Uin(2);
   const double E1     = Uin(3);

   const double u1x = rhou1x / rho1;
   const double u1y = rhou1y / rho1;
   const double v1sq = u1x*u1x + u1y*u1y;

   const double p1 = (gamma - 1.0) * (E1 - 0.5 * rho1 * v1sq);
   MFEM_VERIFY(p1 > 0.0,
               "Upstream pressure must be positive in ForwardFacingStepQuiescentShockIC.");

   const double c1 = std::sqrt(gamma * p1 / rho1);
   const double M1 = std::sqrt(v1sq) / c1;

   // ---- Shock-like jump for rho and p (normal-shock formulas) ----
   double rho_q = rho1;
   double p_q   = p1;

   if (M1 > 1.0)
   {
      const double M1sq = M1 * M1;

      // rho2 / rho1 = ( (γ+1) M1^2 ) / ( (γ-1) M1^2 + 2 )
      rho_q = rho1 * ( (gamma + 1.0) * M1sq ) /
                      ( (gamma - 1.0) * M1sq + 2.0 );

      // p2 / p1 = 1 + 2γ/(γ+1) (M1^2 - 1)
      p_q = p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (M1sq - 1.0));
   }
   // else: subsonic, keep rho_q, p_q = upstream

   // ---- Downstream quiescent shocked conservative state Uq ----
   const double u_qx = 0.0;
   const double u_qy = 0.0;
   const double E_q  = p_q / (gamma - 1.0); // since u = 0

   const double rhoq_uqx = rho_q * u_qx; // = 0
   const double rhoq_uqy = rho_q * u_qy; // = 0

   // Transition band
   const double x0 = x_s - delta_x;
   const double x1 = x_s + delta_x;

   return mfem::VectorFunctionCoefficient(
      4,
      [=](const mfem::Vector &x, mfem::Vector &y)
      {
         y.SetSize(4);
         const double X = x(0);

         // Mixing factor S: 0 -> U1, 1 -> Uq
         double S = 0.0;

         if (X <= x0)
         {
            S = 0.0; // pure upstream
         }
         else if (X >= x1)
         {
            S = 1.0; // pure quiescent shocked
         }
         else
         {
            const double s_raw = (X - x0) / (x1 - x0); // in (0,1)
            const double s     = std::max(0.0, std::min(1.0, s_raw));
            // C^1 smoothstep
            S = s*s*(3.0 - 2.0*s);
         }

         const double one_minus_S = 1.0 - S;

         // Linear blend in *conservative* variables:
         const double rho   = one_minus_S*rho1    + S*rho_q;
         const double rhoux = one_minus_S*rhou1x  + S*rhoq_uqx; // 0 on right
         const double rhouy = one_minus_S*rhou1y  + S*rhoq_uqy; // 0 on right
         const double E     = one_minus_S*E1      + S*E_q;

         y(0) = rho;
         y(1) = rhoux;
         y(2) = rhouy;
         y(3) = E;
      });
}





} // namespace mfem
