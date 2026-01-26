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

inline const char* SupersonicCylinderDefaultMeshPath()
{
   return "supersonic_cylinder.msh";
}

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
         return Mesh(SupersonicCylinderDefaultMeshPath());
      case 2:
         return Mesh(SupersonicCylinderDefaultMeshPath());
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

VectorFunctionCoefficient EulerInitialCondition(const int problem,
                                                const real_t specific_heat_ratio,
                                                const real_t gas_constant)
{
   switch (problem)
   {
      case 1: // Supersonic Farfield
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
      case 2: // slow moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.05, 1. / 50., gas_constant,
                                          specific_heat_ratio));
      default:
         MFEM_ABORT("Problem Undefined");
   }
}



} // namespace mfem
