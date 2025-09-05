// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of hyperbolic conservation laws

#include "mfem.hpp"
#include "BdrHyperbolicFluxIntegrator.hpp"
// #include "hyperbolic.hpp"
// #include "nonlinearform.hpp"
// #include "pnonlinearform.hpp"

namespace mfem
{

// ---------- SupersonicInletState ----------

SupersonicInletState::SupersonicInletState(const Vector& Uin)
  : Uin_(Uin) {}

void SupersonicInletState::Eval(const Vector&,
                                const Vector&,
                                FaceElementTransformations&,
                                Vector& Uplus) const
{
  Uplus = Uin_;
}

// ---------- SupersonicOutletState ----------

void SupersonicOutletState::Eval(const Vector& Uminus,
                                 const Vector&,
                                 FaceElementTransformations&,
                                 Vector& Uplus) const
{
  Uplus = Uminus;
}

// ---------- SlipWallState ----------

SlipWallState::SlipWallState(int dim)
  : dim_(dim) {}

void SlipWallState::Eval(const Vector& Uminus,
                         const Vector& nor,
                         FaceElementTransformations&,
                         Vector& Uplus) const
{
  // Copy conservative state
  Uplus = Uminus;

  const double rho = Uminus(0);

  // Unit normal from scaled normal
  Vector nhat(nor);
  const double nrm = nhat.Norml2();
  if (nrm > 0.0) { nhat /= nrm; }

  // Reconstruct velocity u^- from momentum
  Vector u(dim_);
  for (int d = 0; d < dim_; ++d) { u(d) = Uminus(1 + d) / rho; }

  // Reflect normal component
  const double un = InnerProduct(u, nhat);
  for (int d = 0; d < dim_; ++d)
  {
    const double uprime_d = u(d) - 2.0 * un * nhat(d);
    Uplus(1 + d) = rho * uprime_d; // new momentum
  }
  // ρ (index 0) and E (index 1+dim_) remain unchanged (adiabatic slip wall)
}

// --------- Boundary-face integrator ---------

BdrHyperbolicFluxIntegrator::BdrHyperbolicFluxIntegrator(
   const NumericalFlux &numFlux,
   std::unique_ptr<BoundaryState> bc_,
   const int IntOrderOffset,
   real_t sign)
   : NonlinearFormIntegrator(),
     numFlux(numFlux),
     fluxFunction(numFlux.GetFluxFunction()),
     bc(std::move(bc_)),
     IntOrderOffset(IntOrderOffset),
     sign(sign),
     num_equations(fluxFunction.num_equations)
{
#ifndef MFEM_THREAD_SAFE
   state.SetSize(num_equations);
   flux.SetSize(num_equations, fluxFunction.dim);
   state1.SetSize(num_equations);
   state2.SetSize(num_equations);
   fluxN.SetSize(num_equations);
   JDotN.SetSize(num_equations);
   nor.SetSize(fluxFunction.dim);
#endif
   ResetMaxCharSpeed();
}

void BdrHyperbolicFluxIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   // current elements' the number of degrees of freedom
   // does not consider the number of equations
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   // Local storage for element integration

   // shape function value at an integration point - first elem
   Vector shape1(dof1);
   // shape function value at an integration point - second elem
   // Vector shape2(dof2); // Unused for BC
   // normal vector (usually not a unit vector)
   Vector nor(Tr.GetSpaceDim());
   // state value at an integration point - first elem
   Vector state1(num_equations);
   // state value at an integration point - second elem
   Vector state2(num_equations);
   // hat(F)(u,x)
   Vector fluxN(num_equations);
#else
   shape1.SetSize(dof1);
   // shape2.SetSize(dof2); // Unused for BC
#endif

   elvect.SetSize((dof1 + dof2) * num_equations);
   elvect = 0.0;

   const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
//    const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
//                                 num_equations);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
//    DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
//                            num_equations);

   // Obtain integration rule. If integration is rule is given, then use it.
   // Otherwise, get (2*p + IntOrderOffset) order integration rule
   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      const int order = 2*el1.GetOrder() + IntOrderOffset;
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }
   // loop over integration points
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    //   el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, state1);
    //   elfun2_mat.MultTranspose(shape2, state2);

      // Get the normal vector and the flux on the face
      if (nor.Size() == 1)  // if 1D, use 1 or -1.
      {
         // This assume the 1D integration point is in (0,1). This may not work
         // if this changes.
         nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      // Compute State 2 s.t. it imposed the BC
      // Ghost state U+
      bc->Eval(state1, nor, Tr, state2);

      // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
      // Compute hat(F) using evaluated quantities
      const real_t speed = numFlux.Eval(state1, state2, nor, Tr, fluxN);

      // Update the global max char speed
      max_char_speed = std::max(speed, max_char_speed);

      // pre-multiply integration weight to flux
      AddMult_a_VWt(-ip.weight*sign, shape1, fluxN, elvect1_mat);
    //   AddMult_a_VWt(+ip.weight*sign, shape2, fluxN, elvect2_mat);
   }
}


} // namespace mfem
