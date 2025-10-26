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
#include "fluxfunctions.hpp"
#include "IGR.hpp"

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
   // Define an integrator for (∇v, w) bilinear form, but transposed.
   // GradientIntegrator normally gives:
   //    (∇ϕ_i, ψ_j)_T   (row = test v, col = trial u).
   // TransposeIntegrator flips roles of test/trial, giving:
   //    (ϕ_i, ∇ψ_j)_T
   // which is exactly the weak divergence operator we want.
   TransposeIntegrator weak_div(new GradientIntegrator());

   DenseMatrix weakdiv_bynodes; // temporary storage of assembled element matrix

   // Resize container to hold one weak-divergence matrix per element.
   // Each weakdiv[i] is a DenseMatrix of size (dof × (dof*dim)).
   // - Rows = test functions (dof)
   // - Columns = trial function per dimension (dof*dim)
   weakdiv.resize(vfes.GetNE());

   // Loop over all elements in the mesh
   for (int i=0; i<vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();

      // Temporary matrix: (dof × (dof*dim))
      weakdiv_bynodes.SetSize(dof, dof*dim);

      // Assemble the weak divergence operator for element i:
      // weakdiv_bynodes(j, d*dof + k) ≈ ∫_T ϕ_j · ∂_d ϕ_k dx
      // i.e. applying derivative in each spatial dimension d.
      weak_div.AssembleElementMatrix2(*vfes.GetFE(i), *vfes.GetFE(i),
                                      *vfes.GetElementTransformation(i),
                                      weakdiv_bynodes);

      // Allocate storage for reordered version
      weakdiv[i].SetSize(dof, dof*dim);

      // Reorder columns: trial space grouped by node, then dimension.
      // Original layout = [all DOFs in x-dir][all DOFs in y-dir][all DOFs in z-dir]
      // New layout      = [d0_x, d0_y, (d0_z)], [d1_x, d1_y, (d1_z)], ...
      // i.e. “ByDim per node” → makes multiplying by nodal fluxes F(u) easier.
      for (int j=0; j<dof; j++)         // loop over trial DOFs
      {
         for (int d=0; d<dim; d++)      // loop over dimensions
         {
            // Copy column from assembled to reordered
            weakdiv[i].SetCol(j*dim + d,
                              weakdiv_bynodes.GetColumn(d*dof + j));
         }
      }
   }
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // (0) Reset the running max wave speed (used for CFL after Mult()).
   formIntegrator->ResetMaxCharSpeed();

   // (1) Build the face contribution (and possibly the volume term) into z:
   //     z = - < F̂(u_h, n), [[v]] >_faces
   //     If weak-divergence is NOT preassembled, HyperbolicFormIntegrator also adds
   //     the volume term (F(u_h), ∇v) so that:
   //     z = - < F̂(u_h, n), [[v]] >_faces + (F(u_h), ∇v)
   nonlinearForm->Mult(x, z);

   if (!weakdiv.empty()) // weak divergence WAS preassembled element-wise
   {
      // We'll now compute (I_h F(u_h), ∇v) element-by-element and add it into z,
      // then finish with the inverse mass to get y = M^{-1} z.

      Vector current_state;        // view of the state at one node (size = num_eq)
      DenseMatrix current_flux;    // view of flux at that node (num_eq × dim)
      DenseMatrix flux;            // element flux: stacked by nodes (num_eq × (dim * dof))
      DenseMatrix current_xmat;    // element's states as (dof × num_eq) view into x
      DenseMatrix current_zmat;    // element's accumulated RHS as (dof × num_eq) view into z
      DenseMatrix current_ymat;    // element's output du/dt as (dof × num_eq) before scatter
      const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
      Array<int> vdofs;            // vector DOF indices for this element
      Vector xval, zval;

      for (int i = 0; i < vfes.GetNE(); i++)   // loop over elements T
      {
         ElementTransformation* Tr = vfes.GetElementTransformation(i);
         const FiniteElement &fe   = *vfes.GetFE(i);
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);

         // Pull element's DOFs from x and view them as (dof × num_equations)
         x.GetSubVector(vdofs, xval);
         current_xmat.UseExternalData(xval.GetData(), dof, num_equations);

         // Prepare buffer for physical flux at all dof points
         flux.SetSize(num_equations, dim*dof);

         const IntegrationRule &nodes = fe.GetNodes();
         MFEM_ASSERT(nodes.Size() == dof,
                     "FE is not nodal or GetNodes() != dof; "
                     "need to evaluate at an appropriate rule.");

         // (2) For each local node j, evaluate the conservative state and compute F(u_j)
         for (int j = 0; j < dof; j++)
         {
            // NEW: set the element integration point to node j
            const IntegrationPoint &ip = nodes.IntPoint(j);
            Tr->SetIntPoint(&ip);

            // current_state := U_h at node j  (size num_eq)
            current_xmat.GetRow(j, current_state);

            // current_flux is a (num_eq × dim) view into the j-th block inside 'flux'
            current_flux.UseExternalData(flux.GetData() + num_equations*dim*j,
                                         num_equations, /*ncols=*/dim);

            // Compute physical flux tensor F(U, x) at this node (fills current_flux)
            // Mathematically this is: F(U_j) ∈ ℝ^{num_eq×dim}
            fluxFunction.ComputeFlux(current_state, *Tr, current_flux);
         }

         // (3) Add the preassembled weak-divergence term to z:
         //     current_zmat += weakdiv[i] * flux^T
         // weakdiv[i] encodes ∫_T (·, ∇v) with a “dim-reordered” layout so
         // multiplying by flux^T yields (I_h F(u_h), ∇v)_T contribution.
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);

         // AddMult_a_ABt: current_zmat += 1.0 * weakdiv[i] * (flux)^T
         // (Note ABᵗ: weakdiv[i] multiplies by flux transposed as expected by the layout)
         mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, current_zmat);

         // (4) Apply inverse mass to obtain time derivative on this element:
         //     current_ymat = invmass[i] * current_zmat
         // i.e. y_T = M_T^{-1} * z_T
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);

         // Scatter back to the global vector y at the element DOFs.
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }
   else
   {
      // weak divergence NOT preassembled: z already has both face and volume terms.
      // Just apply block inverse mass element-by-element to get y = M^{-1} z.

      Vector zval;                 // element slice of z  (dof*num_eq)
      DenseMatrix current_zmat;    // (dof × num_eq) view into z
      DenseMatrix current_ymat;    // (dof × num_eq) element output
      Array<int> vdofs;

      for (int i = 0; i < vfes.GetNE(); i++)   // loop over elements T
      {
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);

         // View element slice of z as (dof × num_eq)
         z.GetSubVector(vdofs, zval);
         current_zmat.UseExternalData(zval.GetData(), dof, num_equations);

         // Apply inverse mass on the element block
         current_ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], current_zmat, current_ymat);

         // Scatter to y
         y.SetSubVector(vdofs, current_ymat.GetData());
      }
   }

   // (5) Expose the maximum characteristic speed accumulated inside the integrators
   max_char_speed = formIntegrator->GetMaxCharSpeed();
}


void DGHyperbolicConservationLaws::Update()
{
   // Tell the nonlinear form to refresh its internal data.
   // This could recompute quadrature data, face integrators, etc.,
   // especially if the mesh or finite element space has changed.
   nonlinearForm->Update();

   // Height = global vector length (size of the DOF vector).
   // Width = operator width = also the same, since this is square (u -> u).
   height = nonlinearForm->Height();
   width = height;

   // Resize the auxiliary work vector z used in Mult().
   // z stores the assembled right-hand side BEFORE multiplying by invmass.
   z.SetSize(height);

   // Compute per-element inverse mass matrices:
   // invmass[i] ≈ M_T^{-1}, where
   //    M_T = (ϕ_m, ϕ_n)_T   (local mass matrix on element T).
   // These are needed to apply y = M^{-1} z in Mult().
   ComputeInvMass();

   // If we are using the “weak divergence preassembly” mode:
   //    weakdiv[i] encodes the (∇v, ·) operator on element i,
   //    arranged so that we can later do flux × weakdiv efficiently.
   // Only compute this once when mesh/FE changes.
   if (!weakdiv.empty()) {
      ComputeWeakDivergence();
   }
}


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


/// --------------------------------------------------------------
/// Wrapper operator: recompute Σ each Mult() and forward to DG.
/// --------------------------------------------------------------
class EulerWithIGR : public TimeDependentOperator
{
public:
  EulerWithIGR(DGHyperbolicConservationLaws& base,
               IGRSolver& igr,
               ParGridFunction& state)
    : TimeDependentOperator(base.Height()),
      base_(base), igr_(igr), state_(state) {}

  void SetTime(double t) override { base_.SetTime(t); }

  void Mult(const Vector& U, Vector& dUdt) const override
  {
    // Recompute Σ with a few warm-started sweeps;
    // FluxWithIGR reads Σ directly (as a ParGridFunction*) so
    // no explicit plumbing into DG is needed here.
    igr_.UpdateFromState(state_, /*warm_start=*/true);

    // IMPORTANT: make Σ available on face neighbors across MPI subdomains
    igr_.GetSigma().ExchangeFaceNbrData();

    // Forward to the DG operator
    base_.Mult(U, dUdt);
  }

  real_t GetMaxCharSpeed() { return base_.GetMaxCharSpeed(); }

private:
  DGHyperbolicConservationLaws& base_;
  IGRSolver&                    igr_;
  ParGridFunction&              state_;  // view of current state (vdim=num_eq)
};

class VelocityFromConsCoefficient : public mfem::VectorCoefficient
{
public:
  VelocityFromConsCoefficient(int dim, const mfem::ParGridFunction* U,
                              double rho_floor)
    : mfem::VectorCoefficient(dim), dim_(dim), U_(U), rho_floor_(rho_floor) {}

  void SetState(const mfem::ParGridFunction* U) { U_ = U; }

  virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                    const mfem::IntegrationPoint &ip) override
  {
    MFEM_VERIFY(U_, "VelocityFromConsCoefficient: U is null");
    T.SetIntPoint(&ip);
    mfem::Vector state;
    U_->GetVectorValue(T, ip, state); // [rho, m(0..dim-1), E]
    const double rho = std::max(state(0), rho_floor_);
    V.SetSize(dim_);
    for (int d = 0; d < dim_; ++d) { V(d) = state(1 + d) / rho; }
  }
private:
  int dim_;
  const mfem::ParGridFunction* U_;
  double rho_floor_;
};

// --- Diagnostics from an H1 vector velocity field (uH1) ---
struct GradFrobNormCoeff : public mfem::Coefficient {
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  GradFrobNormCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G); // G(i,j) = ∂_j u_i
    double s = 0.0; for (int i=0;i<dim;i++) for (int j=0;j<dim;j++) s += G(i,j)*G(i,j);
    return std::sqrt(s);
  }
};

struct DivCoeff : public mfem::Coefficient {
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  DivCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::DenseMatrix G(dim, dim);
    v->GetVectorGradient(T, G);
    double tr = 0.0; for (int i=0;i<dim;i++) tr += G(i,i);
    return tr;
  }
};

struct VelMagCoeff : public mfem::Coefficient {
  const mfem::ParGridFunction* v = nullptr; int dim = 0;
  VelMagCoeff(const mfem::ParGridFunction* vel, int d) : v(vel), dim(d) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (!v) return 0.0; T.SetIntPoint(&ip);
    mfem::Vector u; v->GetVectorValue(T, ip, u);
    return u.Norml2();
  }
};


} // namespace mfem
