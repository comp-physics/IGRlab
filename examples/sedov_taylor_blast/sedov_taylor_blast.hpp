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
#include <cmath>

#include "mfem.hpp"
#include "BdrHyperbolicFluxIntegrator.hpp"
#include "DGHyperbolicConservationLaws.hpp"
#include "eulerigr.hpp"
#include "fluxfunctions.hpp"
#include "IGR.hpp"

namespace mfem
{

///// Mesh Utilities

inline const char* SedovTaylorDefaultMeshPath()
{
   return "sedov_taylor.msh";
}

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
         return Mesh(SedovTaylorDefaultMeshPath());
      case 2:
         return Mesh(SedovTaylorDefaultMeshPath());
      // You can add a new case here for the Sedov mesh if desired, e.g.:
      // case 3:
      //    return Mesh("sedov_taylor_blast.msh");
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition: moving vortex (existing)
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

      y.SetSize(4);
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
