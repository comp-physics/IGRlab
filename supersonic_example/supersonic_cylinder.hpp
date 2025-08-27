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

namespace mfem
{

/// @brief Time dependent DG operator for hyperbolic conservation laws
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
private:
   const int num_equations; // the number of equations
   const int dim;
   FiniteElementSpace &vfes; // vector finite element space
   // Element integration form. Should contain ComputeFlux
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
   // Base Nonlinear Form
   std::unique_ptr<NonlinearForm> nonlinearForm;
   // element-wise inverse mass matrix
   std::vector<DenseMatrix> invmass; // local scalar inverse mass
   std::vector<DenseMatrix> weakdiv; // local weak divergence (trial space ByDim)
   // global maximum characteristic speed. Updated by form integrators
   mutable real_t max_char_speed;
   // auxiliary variable used in Mult
   mutable Vector z;

   // Compute element-wise inverse mass matrix
   void ComputeInvMass();
   // Compute element-wise weak-divergence matrix
   void ComputeWeakDivergence();

public:
   /**
    * @brief Construct a new DGHyperbolicConservationLaws object
    *
    * @param vfes_ vector finite element space. Only tested for DG [Pₚ]ⁿ
    * @param formIntegrator_ integrator (F(u,x), grad v)
    * @param preassembleWeakDivergence preassemble weak divergence for faster
    *                                  assembly
    */
   DGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
      bool preassembleWeakDivergence=true);
   /**
    * @brief Apply nonlinear form to obtain M⁻¹(DIVF + JUMP HAT(F))
    *
    * @param x current solution vector
    * @param y resulting dual vector to be used in an EXPLICIT solver
    */
   void Mult(const Vector &x, Vector &y) const override;
   // get global maximum characteristic speed to be used in CFL condition
   // where max_char_speed is updated during Mult.
   real_t GetMaxCharSpeed() { return max_char_speed; }
   void Update();

   // Boundary condition methods
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *integrator, 
                            Array<int> &bdr_marker)
   {
      nonlinearForm->AddBdrFaceIntegrator(integrator, bdr_marker);
   }

   void AddBdrFaceIntegrator(NonlinearFormIntegrator *integrator)
   {
      nonlinearForm->AddBdrFaceIntegrator(integrator);
   }

};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
   FiniteElementSpace &vfes_,
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
   bool preassembleWeakDivergence)
   : TimeDependentOperator(vfes_.GetTrueVSize()),
     num_equations(formIntegrator_->num_equations),
     dim(vfes_.GetMesh()->SpaceDimension()),
     vfes(vfes_),
     formIntegrator(std::move(formIntegrator_)),
     z(vfes_.GetTrueVSize())
{
   // Standard local assembly and inversion for energy mass matrices.
   ComputeInvMass();
#ifndef MFEM_USE_MPI
   nonlinearForm.reset(new NonlinearForm(&vfes));
#else
   ParFiniteElementSpace *pvfes = dynamic_cast<ParFiniteElementSpace *>(&vfes);
   if (pvfes)
   {
      nonlinearForm.reset(new ParNonlinearForm(pvfes));
   }
   else
   {
      nonlinearForm.reset(new NonlinearForm(&vfes));
   }
#endif
   if (preassembleWeakDivergence)
   {
      ComputeWeakDivergence();
   }
   else
   {
      nonlinearForm->AddDomainIntegrator(formIntegrator.get());
   }
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get());
   nonlinearForm->UseExternalIntegrators();

}

void DGHyperbolicConservationLaws::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());

   invmass.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i),
                                     invmass[i]);
   }
}

void DGHyperbolicConservationLaws::ComputeWeakDivergence()
{
   TransposeIntegrator weak_div(new GradientIntegrator());
   DenseMatrix weakdiv_bynodes;

   weakdiv.resize(vfes.GetNE());
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      weakdiv_bynodes.SetSize(dof, dof*dim);
      weak_div.AssembleElementMatrix2(*vfes.GetFE(i), *vfes.GetFE(i),
                                      *vfes.GetElementTransformation(i),
                                      weakdiv_bynodes);
      weakdiv[i].SetSize(dof, dof*dim);
      // Reorder so that trial space is ByDim.
      // This makes applying weak divergence to flux value simpler.
      for (int j=0; j<dof; j++)
      {
         for (int d=0; d<dim; d++)
         {
            weakdiv[i].SetCol(j*dim + d, weakdiv_bynodes.GetColumn(d*dof + j));
         }
      }

   }
}


void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   formIntegrator->ResetMaxCharSpeed();
   // 1. Apply Nonlinear form to obtain an auxiliary result
   //         z = - <F̂(u_h,n), [[v]]>_e
   //    If weak-divergence is not preassembled, we also have weak-divergence
   //         z = - <F̂(u_h,n), [[v]]>_e + (F(u_h), ∇v)
   nonlinearForm->Mult(x, z);
   if (!weakdiv.empty()) // if weak divergence is pre-assembled
   {
      // Apply weak divergence to F(u_h), and inverse mass to z_loc + weakdiv_loc
      Vector current_state; // view of current state at a node
      DenseMatrix current_flux; // flux of current state
      DenseMatrix flux; // element flux value. Whose column is ordered by dim.
      DenseMatrix current_xmat; // view of current states in an element, dof x num_eq
      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
      Array<int> vdofs;
      Vector xval, zval;
      for (int i=0; i<vfes.GetNE(); i++)
      {
         ElementTransformation* Tr = vfes.GetElementTransformation(i);
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         x.GetSubVector(vdofs, xval);
         current_xmat.UseExternalData(xval.GetData(), dof, num_equations);
         flux.SetSize(num_equations, dim*dof);
         for (int j=0; j<dof; j++) // compute flux for all nodes in the element
         {
            current_xmat.GetRow(j, current_state);
            current_flux.UseExternalData(flux.GetData() + num_equations*dim*j,
                                         num_equations, dof);
            fluxFunction.ComputeFlux(current_state, *Tr, current_flux);
         }
         // Compute weak-divergence and add it to auxiliary result, z
         // Recalling that weakdiv is reordered by dim, we can apply
         // weak-divergence to the transpose of flux.
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
         mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, current_zmat);
         // Apply inverse mass to auxiliary result to obtain the final result
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   else
   {
      // Apply block inverse mass
      Vector zval; // z_loc, dof*num_eq

      DenseMatrix current_zmat; // view of element auxiliary result, dof x num_eq
      DenseMatrix current_ymat; // view of element result, dof x num_eq
      Array<int> vdofs;
      for (int i=0; i<vfes.GetNE(); i++)
      {
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   max_char_speed = formIntegrator->GetMaxCharSpeed();
}

void DGHyperbolicConservationLaws::Update()
{
   nonlinearForm->Update();
   height = nonlinearForm->Height();
   width = height;
   z.SetSize(height);

   ComputeInvMass();
   if (!weakdiv.empty()) {ComputeWeakDivergence();}
}

// Boundary condition flux functions for supersonic cylinder
// Inlet boundary - prescribe inlet state (same as initial condition)
class InletBoundaryFlux : public NumericalFlux
{
public:
   InletBoundaryFlux(const FluxFunction &fluxFunc) : NumericalFlux(fluxFunc) {}

   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override
   {
      // Prescribe inlet state directly (same as case 6 initial condition)
      Vector inlet_vals(4);
      inlet_vals(0) = 1.4;        // density
      inlet_vals(1) = 1.4 * 3.0;  // rho * u
      inlet_vals(2) = 1.4 * 0.0;  // rho * v  
      inlet_vals(3) = 8.8;        // energy
      
      return fluxFunction.ComputeFluxDotN(inlet_vals, nor, Tr, flux);
   }
};

// Slip wall boundary - reflect normal velocity component  
class SlipWallFlux : public NumericalFlux
{
public:
   SlipWallFlux(const FluxFunction &fluxFunc) : NumericalFlux(fluxFunc) {}

   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override
   {
      Vector wall_state = state1; // Start with interior state
      
      // Get normalized normal vector
      Vector n = nor;
      real_t nor_mag = n.Norml2();
      n /= nor_mag;
      
      // Reflect normal momentum component: u_wall = u_interior - 2(u·n)n
      real_t rho = wall_state(0);
      real_t u_normal = (wall_state(1)*n(0) + wall_state(2)*n(1)) / rho;
      
      wall_state(1) -= 2.0 * rho * u_normal * n(0); // rho * u_x
      wall_state(2) -= 2.0 * rho * u_normal * n(1); // rho * u_y
      
      return fluxFunction.ComputeFluxDotN(wall_state, nor, Tr, flux);
   }
};

// Outlet boundary - use interior state (free boundary)
class OutletBoundaryFlux : public NumericalFlux  
{
public:
   OutletBoundaryFlux(const FluxFunction &fluxFunc) : NumericalFlux(fluxFunc) {}

   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override
   {
      // Use interior state as-is (zero gradient boundary condition)
      return fluxFunction.ComputeFluxDotN(state1, nor, Tr, flux);
   }
};

std::function<void(const Vector&, Vector&)> GetMovingVortexInit(
   const real_t radius, const real_t Minf, const real_t beta,
   const real_t gas_constant, const real_t specific_heat_ratio)
{
   return [specific_heat_ratio,
           gas_constant, Minf, radius, beta](const Vector &x, Vector &y)
   {
      MFEM_ASSERT(x.Size() == 2, "");

      const real_t xc = 0.0, yc = 0.0;

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

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
      case 2:
      case 3:
      case 5:
         return Mesh("../data/periodic-square.mesh");
         break;
      case 4:
         return Mesh("../data/periodic-segment.mesh");
         break;
      case 6:
         return Mesh("../supersonic_cylinder.msh");
         break;
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Initial condition
VectorFunctionCoefficient EulerInitialCondition(const int problem,
                                                const real_t specific_heat_ratio,
                                                const real_t gas_constant)
{
   switch (problem)
   {
      case 1: // fast moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.5, 1. / 5., gas_constant,
                                          specific_heat_ratio));
      case 2: // slow moving vortex
         return VectorFunctionCoefficient(
                   4, GetMovingVortexInit(0.2, 0.05, 1. / 50., gas_constant,
                                          specific_heat_ratio));
      case 3: // moving sine wave
         return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 2, "");
            const real_t density = 1.0 + 0.2 * std::sin(M_PI*(x(0) + x(1)));
            const real_t velocity_x = 0.7;
            const real_t velocity_y = 0.3;
            const real_t pressure = 1.0;
            const real_t energy =
               pressure / (1.4 - 1.0) +
               density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = density * velocity_y;
            y(3) = energy;
         });
      case 4:
         return VectorFunctionCoefficient(3, [](const Vector &x, Vector &y)
         {
            MFEM_ASSERT(x.Size() == 1, "");
            const real_t density = 1.0 + 0.2 * std::sin(M_PI * 2 * x(0));
            const real_t velocity_x = 1.0;
            const real_t pressure = 1.0;
            const real_t energy =
               pressure / (1.4 - 1.0) + density * 0.5 * (velocity_x * velocity_x);

            y(0) = density;
            y(1) = density * velocity_x;
            y(2) = energy;
         });
      case 5: // 2D Gaussian pulse in velocity - corrected version
         return VectorFunctionCoefficient(4, [](const Vector &x, Vector &y)  // 4 components for 2D Euler
         {
            MFEM_ASSERT(x.Size() == 2, "Expected 2D coordinates");  // Fix: 2D mesh
            
            // Background state - based on other examples in ex18
            const real_t rho_bg = 1.0;           // standard reference density
            const real_t p_bg = 1.0;             // standard reference pressure  
            const real_t u_bg = 0.0;             // start from rest (cleaner than arbitrary 0.1)
            const real_t v_bg = 0.0;             // start from rest
            
            // Gaussian pulse parameters - based on domain and physics
            const real_t x_center = 0.0;         // center of [0,1] domain (periodic-square.mesh)
            const real_t y_center = 0.0;         // center of [0,1] domain
            const real_t sigma = 0.2;            // 10% of domain width - reasonable resolution
            const real_t u_amplitude = 0.5;      // Start conservative: 50% of sound speed (c ≈ 1.18 for γ=1.4, p=1, ρ=1)
            
            // Compute 2D Gaussian pulse centered at (0.5, 0.5)
            const real_t dx = x(0) - x_center;
            const real_t dy = x(1) - y_center;
            const real_t r_squared = dx*dx + dy*dy;
            const real_t gaussian_factor = std::exp(-r_squared/(2.0*sigma*sigma));
            
            // Apply pulse to x-velocity only (creates primarily x-direction shock)
            const real_t density = rho_bg;
            const real_t velocity_x = u_bg + u_amplitude * gaussian_factor;
            const real_t velocity_y = v_bg;  // no y-component initially
            const real_t pressure = p_bg;
            
            // Compute total energy correctly: E = p/(γ-1) + ρ(u²+v²)/2
            const real_t gamma = 1.4;  // standard air
            const real_t kinetic_energy = 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);
            const real_t internal_energy = pressure / (gamma - 1.0);
            const real_t total_energy = internal_energy + density * kinetic_energy;

            // Conservative variables for 2D Euler: [ρ, ρu, ρv, E]
            y(0) = density;                    // density
            y(1) = density * velocity_x;       // x-momentum  
            y(2) = density * velocity_y;       // y-momentum
            y(3) = total_energy;               // total energy
         });
      case 6: // Supersonic Farfield
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
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

} // namespace mfem
