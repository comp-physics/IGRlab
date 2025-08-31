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

// Boundary condition flux functions for supersonic cylinder
class BoundaryState {
public:
  virtual ~BoundaryState() = default;
  // Uplus at a face point: given interior Uminus, scaled normal 'nor',
  // unit normal 'nhat', and the physical face point x_f.
  virtual void Eval(const mfem::Vector& Uminus,
                    const mfem::Vector& nor,
                    const mfem::Vector& nhat,
                    const mfem::Vector& xf,
                    mfem::Vector&       Uplus) const = 0;
};

// Supersonic inlet: U+ = conservative U_in(x)
class SupersonicInletState : public BoundaryState {
public:
  explicit SupersonicInletState(const mfem::Vector &Uin) : Uin_(Uin) {}
  void Eval(const mfem::Vector&, const mfem::Vector&, const mfem::Vector&,
            const mfem::Vector&, mfem::Vector& Uplus) const override {
    Uplus = Uin_;
  }
private:
  mfem::Vector Uin_;
};

// Supersonic outlet: do-nothing/transmissive → U+ = U-
class SupersonicOutletState : public BoundaryState {
public:
  void Eval(const mfem::Vector& Uminus, const mfem::Vector&, const mfem::Vector&,
            const mfem::Vector&, mfem::Vector& Uplus) const override {
    Uplus = Uminus;
  }
};

// Slip wall (adiabatic): reflect normal velocity, keep rho & E
class SlipWallState : public BoundaryState {
public:
  explicit SlipWallState(int dim) : dim_(dim) {}
  void Eval(const mfem::Vector& Uminus, const mfem::Vector&,
            const mfem::Vector& nhat, const mfem::Vector&,
            mfem::Vector& Uplus) const override {
    Uplus = Uminus;
    const double rho = Uminus(0);
    mfem::Vector u(dim_);
    for (int d = 0; d < dim_; ++d) u(d) = Uminus(1+d) / rho;
    const double un = mfem::InnerProduct(u, nhat);
    for (int d = 0; d < dim_; ++d) {
      const double uprime_d = u(d) - 2.0 * un * nhat(d);
      Uplus(1+d) = rho * uprime_d;
    }
    // Uplus(0) and Uplus(1+dim_) (E) already copied
  }
private:
  int dim_;
};

///// Boundary Integrator
class BdrHyperbolicFluxIntegrator : public mfem::NonlinearFormIntegrator
{
public:
  BdrHyperbolicFluxIntegrator(const mfem::NumericalFlux& numFlux,
                              BoundaryState* bc,
                              int intOrderOffset = 0)
   : numFlux_(numFlux),
     fluxFunction_(numFlux.GetFluxFunction()),
     bc_(bc),
     intOrderOffset_(intOrderOffset),
     neq_(fluxFunction_.num_equations),
     dim_(fluxFunction_.dim) {}

  // NOTE: use AssembleFaceVector (this is virtual in MFEM); we'll attach as BDR.
  void AssembleFaceVector(const mfem::FiniteElement& el1,
                          const mfem::FiniteElement& /*el2*/,  // unused
                          mfem::FaceElementTransformations& Tr,
                          const mfem::Vector& elfun,
                          mfem::Vector& elvect) override
  {
    using namespace mfem;
    const int nd1 = el1.GetDof();
    elvect.SetSize(nd1 * neq_);
    elvect = 0.0;

    const int p = el1.GetOrder();
    const IntegrationRule *ir =
      IntRule ? IntRule : &IntRules.Get(Tr.FaceGeom, 2*p + intOrderOffset_);

    Vector shape1(nd1), Uminus(neq_), Uplus(neq_), fluxN(neq_);
    Vector nor(dim_), nhat(dim_), xf(dim_);

    for (int iq = 0; iq < ir->GetNPoints(); ++iq)
    {
      const IntegrationPoint &ip = ir->IntPoint(iq);

      // Map face ip to element #1 and eval basis
      IntegrationPoint eip1; Tr.Loc1.Transform(ip, eip1);
      el1.CalcShape(eip1, shape1);

      // Face geometry (boundary path still provides Tr.Face)
      Tr.Face->SetIntPoint(&ip);
      CalcOrtho(Tr.Face->Jacobian(), nor);         // scaled normal
      nhat = nor; const double nrm = nhat.Norml2(); if (nrm > 0.0) nhat /= nrm;
      Tr.Face->Transform(ip, xf);                  // physical point (optional)

      // Reconstruct U-
      for (int k = 0; k < neq_; ++k)
      {
        const int off = k*nd1;
        double uk = 0.0; for (int i = 0; i < nd1; ++i) uk += shape1(i) * elfun(off+i);
        Uminus(k) = uk;
      }

      // Ghost state U+
      bc_->Eval(Uminus, nor, nhat, xf, Uplus);

      // Rusanov from your hyperbolic.hpp
      numFlux_.Eval(Uminus, Uplus, nor, Tr, fluxN);

      const double w = Tr.Face->Weight() * ip.weight;  // -<Fhat n, v>
      for (int k = 0; k < neq_; ++k)
      {
        const int off = k*nd1; const double fn = fluxN(k);
        for (int i = 0; i < nd1; ++i) elvect(off+i) -= fn * shape1(i) * w;
      }
    }
  }

private:
  const mfem::NumericalFlux& numFlux_;
  const mfem::FluxFunction&  fluxFunction_;
  std::unique_ptr<BoundaryState> bc_;
  int intOrderOffset_, neq_, dim_;
};


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
