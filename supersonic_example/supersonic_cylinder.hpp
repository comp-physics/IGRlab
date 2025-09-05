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
   // keep external boundary integrators alive, and keep a copy of markers
   std::vector<std::unique_ptr<NonlinearFormIntegrator>> ext_bdr_int_;
   std::vector<std::unique_ptr<Array<int>>>              ext_bdr_markers_;

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

   // Add a boundary-face integrator with a marker, MFEM style.
   // Ownership: we take the raw pointer, the class owns it.
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                             const Array<int> &bdr_marker)
   {
      // 1) Own the integrator so NonlinearForm can UseExternalIntegrators()
      ext_bdr_int_.emplace_back(nfi);
      // 2) Make a heap copy of the marker so its address is stable
      ext_bdr_markers_.emplace_back(std::make_unique<Array<int>>(bdr_marker));
      // 3) Register with NonlinearForm using the *referenced* stable Array<int>
      nonlinearForm->AddBdrFaceIntegrator(nfi, *ext_bdr_markers_.back());
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
            // current_flux.UseExternalData(flux.GetData() + num_equations*dim*j,
            //                  num_equations, dim);
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

///// Mesh Utilities

Mesh EulerMesh(const int problem)
{
   switch (problem)
   {
      case 1:
         return Mesh("../supersonic_cylinder.msh");
         break;
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

// Helper: make a boundary marker from a set of attribute ids {a1, a2, ...}
// Example: MakeMarker(pmesh, {4}) creates a marker with only attribute 4 active.
static void MakeMarker(mfem::Array<int> &marker,
                       mfem::ParMesh &pmesh,
                       std::initializer_list<int> attrs)
{
   const int nb = pmesh.bdr_attributes.Max();
   MFEM_VERIFY(nb > 0, "Mesh has no boundary attributes.");
   marker.SetSize(nb); marker = 0;
   for (int a : attrs)
   {
      MFEM_VERIFY(a >= 1 && a <= nb,
            "Requested boundary attribute " << a << " but mesh has [1.." << nb << "]");
      marker[a-1] = 1;
   }
}


// Initial condition
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
      default:
         MFEM_ABORT("Problem Undefined");
   }
}

} // namespace mfem
