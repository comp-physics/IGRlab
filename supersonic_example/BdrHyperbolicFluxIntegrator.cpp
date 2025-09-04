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
#include "hyperbolic.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"

namespace mfem
{

BdrHyperbolicFluxIntegrator::BdrHyperbolicFluxIntegrator(
   const NumericalFlux &numFlux,
   const int IntOrderOffset,
   real_t sign)
   : NonlinearFormIntegrator(),
     numFlux(numFlux),
     fluxFunction(numFlux.GetFluxFunction()),
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

// void BdrHyperbolicFluxIntegrator::AssembleElementVector(const FiniteElement &el,
//                                                      ElementTransformation &Tr,
//                                                      const Vector &elfun,
//                                                      Vector &elvect)
// {
//    // current element's the number of degrees of freedom
//    // does not consider the number of equations
//    const int dof = el.GetDof();

// #ifdef MFEM_THREAD_SAFE
//    // Local storage for element integration

//    // shape function value at an integration point
//    Vector shape(dof);
//    // derivative of shape function at an integration point
//    DenseMatrix dshape(dof, Tr.GetSpaceDim());
//    // state value at an integration point
//    Vector state(num_equations);
//    // flux value at an integration point
//    DenseMatrix flux(num_equations, el.GetDim());
// #else
//    // resize shape and gradient shape storage
//    shape.SetSize(dof);
//    dshape.SetSize(dof, Tr.GetSpaceDim());
// #endif

//    // setDegree-up output vector
//    elvect.SetSize(dof * num_equations);
//    elvect = 0.0;

//    // make state variable and output dual vector matrix form.
//    const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
//    DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

//    // obtain integration rule. If integration is rule is given, then use it.
//    // Otherwise, get (2*p + IntOrderOffset) order integration rule
//    const IntegrationRule *ir = IntRule;
//    if (!ir)
//    {
//       const int order = el.GetOrder()*2 + IntOrderOffset;
//       ir = &IntRules.Get(Tr.GetGeometryType(), order);
//    }

//    // loop over integration points
//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       Tr.SetIntPoint(&ip);

//       el.CalcShape(ip, shape);
//       el.CalcPhysDShape(Tr, dshape);
//       // compute current state value with given shape function values
//       elfun_mat.MultTranspose(shape, state);
//       // compute F(u,x) and point maximum characteristic speed

//       const real_t mcs = fluxFunction.ComputeFlux(state, Tr, flux);
//       // update maximum characteristic speed
//       max_char_speed = std::max(mcs, max_char_speed);
//       // integrate (F(u,x), grad v)
//       AddMult_a_ABt(ip.weight * Tr.Weight() * sign, dshape, flux, elvect_mat);
//    }
// }

// void HyperbolicFormIntegrator::AssembleElementGrad(
//    const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
//    DenseMatrix &grad)
// {
//    // current element's the number of degrees of freedom
//    // does not consider the number of equations
//    const int dof = el.GetDof();

// #ifdef MFEM_THREAD_SAFE
//    // Local storage for element integration

//    // shape function value at an integration point
//    Vector shape(dof);
//    // derivative of shape function at an integration point
//    DenseMatrix dshape(dof, Tr.GetSpaceDim());
//    // state value at an integration point
//    Vector state(num_equations);
//    // Jacobian value at an integration point
//    DenseTensor J(num_equations, num_equations, fluxFunction.dim);
// #else
//    // resize shape, gradient shape and Jacobian storage
//    shape.SetSize(dof);
//    dshape.SetSize(dof, Tr.GetSpaceDim());
//    J.SetSize(num_equations, num_equations, fluxFunction.dim);
// #endif

//    // setup output gradient matrix
//    grad.SetSize(dof * num_equations);
//    grad = 0.0;

//    // make state variable and output dual vector matrix form.
//    const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
//    //DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

//    // obtain integration rule. If integration is rule is given, then use it.
//    // Otherwise, get (2*p + IntOrderOffset) order integration rule
//    const IntegrationRule *ir = IntRule;
//    if (!ir)
//    {
//       const int order = el.GetOrder()*2 + IntOrderOffset;
//       ir = &IntRules.Get(Tr.GetGeometryType(), order);
//    }

//    // loop over integration points
//    for (int q = 0; q < ir->GetNPoints(); q++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(q);
//       Tr.SetIntPoint(&ip);

//       el.CalcShape(ip, shape);
//       el.CalcPhysDShape(Tr, dshape);
//       // compute current state value with given shape function values
//       elfun_mat.MultTranspose(shape, state);

//       // compute J(u,x)
//       fluxFunction.ComputeFluxJacobian(state, Tr, J);

//       // integrate (J(u,x), grad v)
//       const real_t w = ip.weight * Tr.Weight() * sign;
//       for (int di = 0; di < num_equations; di++)
//          for (int dj = 0; dj < num_equations; dj++)
//             for (int i = 0; i < dof; i++)
//                for (int j = 0; j < dof; j++)
//                   for (int d = 0; d < fluxFunction.dim; d++)
//                   {
//                      grad(di*dof+i, dj*dof+j) += w * dshape(i,d) * shape(j) * J(di,dj,d);
//                   }
//    }
// }

void HyperbolicFormIntegrator::AssembleFaceVector(
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

// void HyperbolicFormIntegrator::AssembleFaceGrad(
//    const FiniteElement &el1, const FiniteElement &el2,
//    FaceElementTransformations &Tr, const Vector &elfun, DenseMatrix &elmat)
// {
//    // current elements' the number of degrees of freedom
//    // does not consider the number of equations
//    const int dof1 = el1.GetDof();
//    const int dof2 = el2.GetDof();

// #ifdef MFEM_THREAD_SAFE
//    // Local storage for element integration

//    // shape function value at an integration point - first elem
//    Vector shape1(dof1);
//    // shape function value at an integration point - second elem
//    Vector shape2(dof2);
//    // normal vector (usually not a unit vector)
//    Vector nor(Tr.GetSpaceDim());
//    // state value at an integration point - first elem
//    Vector state1(num_equations);
//    // state value at an integration point - second elem
//    Vector state2(num_equations);
//    // hat(J)(u,x)
//    DenseMatrix JDotN(num_equations);
// #else
//    shape1.SetSize(dof1);
//    shape2.SetSize(dof2);
// #endif

//    elmat.SetSize((dof1 + dof2) * num_equations);
//    elmat = 0.0;

//    const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
//    const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
//                                 num_equations);

//    // Obtain integration rule. If integration is rule is given, then use it.
//    // Otherwise, get (2*p + IntOrderOffset) order integration rule
//    const IntegrationRule *ir = IntRule;
//    if (!ir)
//    {
//       const int order = 2*std::max(el1.GetOrder(), el2.GetOrder()) + IntOrderOffset;
//       ir = &IntRules.Get(Tr.GetGeometryType(), order);
//    }
//    // loop over integration points
//    for (int q = 0; q < ir->GetNPoints(); q++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(q);

//       Tr.SetAllIntPoints(&ip); // set face and element int. points

//       // Calculate basis functions on both elements at the face
//       el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
//       el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

//       // Interpolate elfun at the point
//       elfun1_mat.MultTranspose(shape1, state1);
//       elfun2_mat.MultTranspose(shape2, state2);

//       // Get the normal vector and the flux on the face
//       if (nor.Size() == 1)  // if 1D, use 1 or -1.
//       {
//          // This assume the 1D integration point is in (0,1). This may not work
//          // if this changes.
//          nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
//       }
//       else
//       {
//          CalcOrtho(Tr.Jacobian(), nor);
//       }

//       // Trial side 1

//       // Compute hat(J) using evaluated quantities
//       numFlux.Grad(1, state1, state2, nor, Tr, JDotN);

//       const int ioff = fluxFunction.num_equations * dof1;

//       for (int di = 0; di < fluxFunction.num_equations; di++)
//          for (int dj = 0; dj < fluxFunction.num_equations; dj++)
//          {
//             // pre-multiply integration weight to Jacobian
//             const real_t w = -ip.weight * sign * JDotN(di,dj);
//             for (int j = 0; j < dof1; j++)
//             {
//                // Test side 1
//                for (int i = 0; i < dof1; i++)
//                {
//                   elmat(i+dof1*di, j+dof1*dj) += w * shape1(i) * shape1(j);
//                }

//                // Test side 2
//                for (int i = 0; i < dof2; i++)
//                {
//                   elmat(ioff+i+dof2*di, j+dof1*dj) -= w * shape2(i) * shape1(j);
//                }
//             }
//          }

//       // Trial side 2

//       // Compute hat(J) using evaluated quantities
//       numFlux.Grad(2, state1, state2, nor, Tr, JDotN);

//       const int joff = ioff;

//       for (int di = 0; di < fluxFunction.num_equations; di++)
//          for (int dj = 0; dj < fluxFunction.num_equations; dj++)
//          {
//             // pre-multiply integration weight to Jacobian
//             const real_t w = +ip.weight * sign * JDotN(di,dj);
//             for (int j = 0; j < dof2; j++)
//             {
//                // Test side 1
//                for (int i = 0; i < dof1; i++)
//                {
//                   elmat(i+dof1*di, joff+j+dof2*dj) += w * shape1(i) * shape2(j);
//                }

//                // Test side 2
//                for (int i = 0; i < dof2; i++)
//                {
//                   elmat(ioff+i+dof2*di, joff+j+dof2*dj) -= w * shape2(i) * shape2(j);
//                }
//             }
//          }
//    }
// }



} // namespace mfem
