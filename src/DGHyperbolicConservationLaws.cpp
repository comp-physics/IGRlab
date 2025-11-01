// Implementation of DGHyperbolicConservationLaws
// --------------------------------------------------------------
//
// Mult() assembles the face contribution (via HyperbolicFormIntegrator)
// and then adds a volume contribution depending on the chosen VolumeMode:
//
//  • WeakDivergence (default):
//      If preassembled, add (I_h F(u), ∇v)_T per element with
//         Z_T += WeakDiv_T * Flux(U_T)^T
//      where Flux(U_T) is the physical flux evaluated at the nodal points.
//      Finally, y_T = M_T^{-1} Z_T.
//
//      If not preassembled (matrix-free path), HyperbolicFormIntegrator
//      already contributes (F(u), ∇v) into z; we only apply M_T^{-1}.
//
//  • DGSEMFluxDiff:
//      Replace the volume term with a DGSEM-style flux-difference contribution
//      built from pairwise numerical fluxes at the element nodes.
//      This is added into z, and then M_T^{-1} is applied.
//
// Notes:
//  - max_char_speed is reset before nonlinearForm->Mult(x, z) and
//    is updated by the face integrators; we expose it for CFL updates.
//  - The class stores per-element inverse mass matrices and (optionally)
//    weak-divergence matrices, recomputed in Update() if needed.
//

#include "DGHyperbolicConservationLaws.hpp"

namespace mfem
{

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
   // (A) Compute inverse mass matrices (one per element)
   ComputeInvMass();

   // (B) Build the nonlinear form for face/domain contributions.
#ifndef MFEM_USE_MPI
   nonlinearForm = std::make_unique<NonlinearForm>(&vfes);
#else
   if (auto *pvfes = dynamic_cast<ParFiniteElementSpace*>(&vfes))
      nonlinearForm = std::make_unique<ParNonlinearForm>(pvfes);
   else
      nonlinearForm = std::make_unique<NonlinearForm>(&vfes);
#endif

   // (C) Either preassemble weak divergence, or let the integrator contribute volume terms.
   if (preassembleWeakDivergence) { ComputeWeakDivergence(); }
   else                           { nonlinearForm->AddDomainIntegrator(formIntegrator.get()); }

   // (D) Faces are always handled by the form integrator
   nonlinearForm->AddInteriorFaceIntegrator(formIntegrator.get());
   nonlinearForm->UseExternalIntegrators();
}

// -----------------------------------------------------------------------------
// Compute inverse mass matrices
// -----------------------------------------------------------------------------
//
// For each element T, assemble M_T^{-1} by applying InverseIntegrator(MassIntegrator).
// We store the dense inverse per element, so we can do:
//     y_T = M_T^{-1} z_T
// cheaply in Mult().
//
void DGHyperbolicConservationLaws::ComputeInvMass()
{
   InverseIntegrator inv_mass(new MassIntegrator());
   invmass.resize(vfes.GetNE());

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      int dof = vfes.GetFE(i)->GetDof();
      invmass[i].SetSize(dof);
      inv_mass.AssembleElementMatrix(*vfes.GetFE(i),
                                     *vfes.GetElementTransformation(i),
                                     invmass[i]);
   }
}

// -----------------------------------------------------------------------------
// Preassemble weak divergence matrices
// -----------------------------------------------------------------------------
//
// We need a matrix that maps nodal flux values to test-function contributions:
//
//   weakdiv[i] (dof × (dof*dim)) with columns ordered as
//      [node0_x, node0_y, (node0_z)], [node1_x, node1_y, (node1_z)], ...
//
// We get this by assembling the transpose of GradientIntegrator:
//   TransposeIntegrator(GradientIntegrator()) gives (ϕ_i, ∇ψ_j)_T.
//
// Then we reorder the columns to ByDim-per-node layout to match how we pack
// the nodal flux tensor F(U_j) ∈ ℝ^{num_eq×dim} for each node j.
//
void DGHyperbolicConservationLaws::ComputeWeakDivergence()
{
   TransposeIntegrator weak_div(new GradientIntegrator());
   DenseMatrix tmp;

   weakdiv.resize(vfes.GetNE());

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      const FiniteElement &fe = *vfes.GetFE(i);
      int dof = fe.GetDof();

      tmp.SetSize(dof, dof * dim);
      weak_div.AssembleElementMatrix2(fe, fe, *vfes.GetElementTransformation(i), tmp);

      weakdiv[i].SetSize(dof, dof * dim);
      for (int j = 0; j < dof; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            // Copy column from the assembled block [all dofs for dimension d]
            // into the “ByDim per node” layout at (node j, dimension d).
            weakdiv[i].SetCol(j * dim + d, tmp.GetColumn(d * dof + j));
         }
      }
   }
}

// -----------------------------------------------------------------------------
// Apply DG operator
// -----------------------------------------------------------------------------
//
// Steps:
// (0) Reset max wave speed.
// (1) nonlinearForm->Mult(x, z) assembles faces (and domain, if matrix-free).
// (2) Add volume term depending on vol_mode_.
// (3) Apply inverse mass element-wise and scatter to y.
// (4) Update max_char_speed from integrator.
//
void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const
{
   // (0) Reset the running max wave speed (used for CFL after Mult()).
   formIntegrator->ResetMaxCharSpeed();

   // (1) Build the face contribution (and possibly the volume term) into z:
   //     z = - < F̂(u_h, n), [[v]] >_faces
   //     If weak-divergence is NOT preassembled, HyperbolicFormIntegrator also adds
   //     the volume term (F(u_h), ∇v):
   //     z += (F(u_h), ∇v)
   nonlinearForm->Mult(x, z);

   if (vol_mode_ == VolumeMode::WeakDivergence){
   if (!weakdiv.empty()) // weak divergence WAS preassembled element-wise
   {
      // We'll now compute (I_h F(u_h), ∇v) element-by-element and add it into z,
      // then finish with the inverse mass to get y = M^{-1} z.

      Vector state;               // local conservative state at one node (size = num_eq)
      DenseMatrix flux;           // packed nodal fluxes for the element (num_eq × (dim*dof))
      DenseMatrix Xmat;           // element's states as (dof × num_eq) view into x
      DenseMatrix Zmat;           // element's assembled RHS as (dof × num_eq) view into z
      DenseMatrix Ymat;           // element's output du/dt as (dof × num_eq) before scatter
      const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
      Array<int> vdofs;           // vector DOF indices for this element
      Vector xval, zval;

      for (int i = 0; i < vfes.GetNE(); i++)   // loop over elements T
      {
         ElementTransformation *Tr = vfes.GetElementTransformation(i);
         const FiniteElement &fe   = *vfes.GetFE(i);
         int dof = fe.GetDof();
         vfes.GetElementVDofs(i, vdofs);

         // Pull element's DOFs from x and view them as (dof × num_equations)
         x.GetSubVector(vdofs, xval);
         Xmat.UseExternalData(xval.GetData(), dof, num_equations);

         // Pack physical flux tensor F(U_j) at all nodal points j into 'flux'
         flux.SetSize(num_equations, dim * dof);
         const IntegrationRule &nodes = fe.GetNodes();
         MFEM_ASSERT(nodes.Size() == dof,
                     "FE is not nodal or GetNodes() != dof; "
                     "need to evaluate at an appropriate rule.");

         for (int j = 0; j < dof; j++)
         {
            // Set the element integration point to node j
            const IntegrationPoint &ip = nodes.IntPoint(j);
            Tr->SetIntPoint(&ip);

            // state := U_h at node j
            Xmat.GetRow(j, state);

            // flux_j is a (num_eq × dim) view into the j-th block inside 'flux'
            DenseMatrix flux_j(flux.GetData() + j * num_equations * dim,
                                 num_equations, /*ncols=*/dim);

            // Compute physical flux tensor F(U, x) at this node (fills flux_j)
            fluxFunction.ComputeFlux(state, *Tr, flux_j);
         }

         // Add the preassembled weak-divergence term to z:
         // Z_T += WeakDiv_T * Flux(U_T)^T  (weakdiv has the ByDim-per-node layout)
         z.GetSubVector(vdofs, zval);
         Zmat.UseExternalData(zval.GetData(), dof, num_equations);
         mfem::AddMult_a_ABt(1.0, weakdiv[i], flux, Zmat);

         // Apply inverse mass: y_T = M_T^{-1} * Z_T
         Ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], Zmat, Ymat);

         // Scatter back to the global vector y
         y.SetSubVector(vdofs, Ymat.GetData());
      }
   }
   else
   {
      // Weak divergence NOT preassembled:
      // z already has both face and volume terms.
      // Just apply block inverse mass element-by-element to get y = M^{-1} z.

      Array<int> vdofs;
      Vector zval;
      DenseMatrix Zmat, Ymat;

      for (int i = 0; i < vfes.GetNE(); i++)
      {
         int dof = vfes.GetFE(i)->GetDof();
         vfes.GetElementVDofs(i, vdofs);

         z.GetSubVector(vdofs, zval);
         Zmat.UseExternalData(zval.GetData(), dof, num_equations);

         Ymat.SetSize(dof, num_equations);
         mfem::Mult(invmass[i], Zmat, Ymat);

         y.SetSubVector(vdofs, Ymat.GetData());
      }
   }
}
else // VolumeMode::DGSEMFluxDiff
{
    auto is_finite_vec = [](const Vector& v, const char* tag) {
        for (int i = 0; i < v.Size(); ++i) {
            double a = v[i];
            if (!(a == a) || !std::isfinite(a)) {
                mfem::out << "[NaN@" << tag << "] idx=" << i << " val=" << a << std::endl;
                MFEM_ABORT("Non-finite detected");
            }
        }
    };

    // ---------- (0) sanity: x must be finite ----------
    is_finite_vec(x, "x_true_in");

    // We require Par space for the true<->local maps
    auto *pvfes = dynamic_cast<ParFiniteElementSpace*>(&vfes);
    MFEM_VERIFY(pvfes, "DGSEM mode requires ParFiniteElementSpace");

    // ---------- (1) faces-only in TRUE space ----------
    formIntegrator->ResetMaxCharSpeed();
    Vector z_true(height); z_true = 0.0;
    nonlinearForm->Mult(x, z_true);            // faces only (make sure NO domain integrator was added)
    is_finite_vec(z_true, "z_true_after_faces");

    // ---------- (2) TRUE -> LOCAL maps ----------
    Vector x_loc(vfes.GetVSize()); x_loc = 0.0;
   Vector z_face_loc(vfes.GetVSize()); z_face_loc = 0.0;

   if (const Operator *P = vfes.GetProlongationMatrix())
   {
      P->Mult(x,      x_loc);
      P->Mult(z_true, z_face_loc);
   }
   else
   {
      // DG often: identity (no constraints)
      x_loc      = x;
      z_face_loc = z_true;
   }
   is_finite_vec(x_loc,      "x_loc_after_Px");
   is_finite_vec(z_face_loc, "z_face_loc_after_Pz");

    // ---------- (2a) OPTIONAL: identity check for P * Pt ----------
    // This verifies your maps are correct. Uncomment temporarily:
    // {
    //     Vector back(height); back = 0.0;
    //     if (const Operator *P = vfes.GetProlongationMatrix()) {
    //         const auto *Ph = dynamic_cast<const HypreParMatrix*>(P);
    //         HypreParVector ytmp(Ph->GetComm(), vfes.GlobalTrueVSize(), (HYPRE_Int*)nullptr, Ph->GetRowStarts());
    //         Ph->MultTranspose(x_loc, back);
    //         // Compare 'back' vs 'x' (they won't be bitwise equal in parallel with ghosts, but no NaNs!)
    //         is_finite_vec(back, "Pt_x_loc_back_to_true");
    //     }
    // }

    // ---------- (3) build a LOCAL GridFunction view on x_loc ----------
    ParGridFunction Uview(pvfes, x_loc.GetData()); // non-owning LOCAL view
    // quick finite peek at a few entries:
    // mfem::out << "Uview(0)=" << Uview[0] << "\n";

    // ---------- (4) DGSEM volume in LOCAL space ----------
    Vector z_vol_loc(vfes.GetVSize()); z_vol_loc = 0.0;

    // ===== HARD KILL SWITCHES FOR ISOLATION =====
    // A) comment the next two lines to check faces-only path through maps:
    const FluxFunction &fluxFunction = formIntegrator->GetFluxFunction();
    AddDGSEMVolumeContribution(*pvfes, Uview, fluxFunction, DGSEMOptions{}, z_vol_loc);
    // ============================================

    is_finite_vec(z_vol_loc, "z_vol_loc_after_dgsem");

    // ---------- (5) combine face+volume in LOCAL ----------
    Vector z_loc(vfes.GetVSize()); z_loc = z_face_loc; z_loc += z_vol_loc;
    is_finite_vec(z_loc, "z_loc_combined");

    // ---------- (6) apply M^{-1} in LOCAL space ----------
    Vector y_loc(vfes.GetVSize()); y_loc = 0.0;
    Array<int> vdofs;
    Vector zslice;
    DenseMatrix Zmat, Ymat;

    for (int e = 0; e < vfes.GetNE(); ++e) {
        const int dof = vfes.GetFE(e)->GetDof();
        vfes.GetElementVDofs(e, vdofs);

        z_loc.GetSubVector(vdofs, zslice);
        Zmat.UseExternalData(zslice.GetData(), dof, num_equations);

        // Guard: inverse mass must be finite
        for (int r = 0; r < dof; ++r)
            for (int c = 0; c < dof; ++c) {
                double a = invmass[e](r,c);
                if (!(a == a) || !std::isfinite(a)) {
                    mfem::out << "[NaN@invMass e="<<e<<" ("<<r<<","<<c<<")]="<<a<<std::endl;
                    MFEM_ABORT("Non-finite in invmass");
                }
            }

        Ymat.SetSize(dof, num_equations);
        mfem::Mult(invmass[e], Zmat, Ymat);

        y_loc.SetSubVector(vdofs, Ymat.GetData());
    }
    is_finite_vec(y_loc, "y_loc_after_invMass");

    // ---------- (7) LOCAL -> TRUE map ----------
    if (const Operator *P = vfes.GetProlongationMatrix())
   {
      P->MultTranspose(y_loc, y);
   }
   else
   {
      y = y_loc;
   }
   is_finite_vec(y, "y_true_out");

    // Only faces update MFEM’s internal speed:
    max_char_speed = formIntegrator->GetMaxCharSpeed();
}


   // (4) Expose the maximum characteristic speed accumulated inside the integrators
   max_char_speed = formIntegrator->GetMaxCharSpeed();
}

// -----------------------------------------------------------------------------
// Update internal data (mass matrices, etc.) when mesh/space changes
// -----------------------------------------------------------------------------
void DGHyperbolicConservationLaws::Update()
{
   // Refresh the nonlinear form's internal data (quadrature, face pairs, etc.)
   nonlinearForm->Update();

   // The operator is square: height = width = global dof size
   height = nonlinearForm->Height();
   width  = height;

   // Scratch workspace for assembled RHS before inverse mass
   z.SetSize(height);

   // Recompute per-element inverse mass matrices
   ComputeInvMass();

   // If we already had weakdiv matrices, refresh them too
   if (!weakdiv.empty()) { ComputeWeakDivergence(); }
}

// -----------------------------------------------------------------------------
// Add boundary face integrator with attribute marker (keeps objects alive)
// -----------------------------------------------------------------------------
void DGHyperbolicConservationLaws::AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                                                        const Array<int> &bdr_marker)
{
   // 1) Own the integrator so NonlinearForm can UseExternalIntegrators()
   ext_bdr_int_.emplace_back(nfi);

   // 2) Make a heap copy of the marker so its address is stable
   ext_bdr_markers_.emplace_back(std::make_unique<Array<int>>(bdr_marker));

   // 3) Register with NonlinearForm using the *referenced* stable Array<int>
   nonlinearForm->AddBdrFaceIntegrator(nfi, *ext_bdr_markers_.back());
}

} // namespace mfem
