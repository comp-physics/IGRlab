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

inline const char* DoubleMachRefDefaultMeshPath()
{
   return "double_mach_ref.msh";
}

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
         return Mesh(DoubleMachRefDefaultMeshPath());
      case 2:
         return Mesh(DoubleMachRefDefaultMeshPath());
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
