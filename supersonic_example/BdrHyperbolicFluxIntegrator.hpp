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


#ifndef MFEM_HYPERBOLIC_BDR
#define MFEM_HYPERBOLIC_BDR

#include "mfem.hpp"
#include "hyperbolic.hpp"

namespace mfem
{

// This file contains general hyperbolic conservation element/face form
// integrators. HyperbolicFormIntegrator and NumericalFlux are defined.
//
// HyperbolicFormIntegrator is a NonlinearFormIntegrator that implements
// element weak divergence and interface flux
//
//     ∫_K F(u):∇v,   -∫_f F̂(u)⋅n[v]
//
// Here, K is an element, f is a face, n normal and [⋅] is jump. This form
// integrator is coupled with NumericalFlux that implements the numerical flux
// F̂. For NumericalFlux, the Rusanov flux, also known as local Lax-Friedrichs
// flux, or component-wise upwinded flux are provided.
//
// To implement a specific hyperbolic conservation laws, users can create
// derived classes from FluxFunction with overloaded ComputeFlux. One can
// optionally overload ComputeFluxDotN to avoid creating dense matrix when
// computing normal flux. Several example equations are also defined including:
// advection, Burgers', shallow water, and Euler equations. Users can control
// the quadrature rule by either providing the integration rule, or integration
// order offset. Integration will use 2*p + IntOrderOffset order quadrature
// rule.
//
// At each call of HyperbolicFormIntegrator::AssembleElementVector
// HyperbolicFormIntegrator::AssembleFaceVector, the maximum characteristic
// speed will be updated. This will not be reinitialized automatically. To
// reinitialize, use HyperbolicFormIntegrator::ResetMaxCharSpeed. See, ex18.hpp.
//
// Note: To avoid communication overhead, we update the maximum characteristic
// speed within each MPI process only. Use the appropriate MPI routine to gather
// the information.

/**
 * @brief Abstract hyperbolic form integrator, assembling (F(u, x), ∇v) and
 * <F̂(u⁻,u⁺,x) n, [v]> terms for scalar finite elements.
 *
 * This form integrator is coupled with a NumericalFlux that implements the
 * numerical flux F̂ at the faces. The flux F is obtained from the FluxFunction
 * assigned to the aforementioned NumericalFlux.
 */
class BdrHyperbolicFluxIntegrator : public NonlinearFormIntegrator
{
private:
   // The maximum characteristic speed, updated during element/face vector assembly
   real_t max_char_speed;
   const NumericalFlux &numFlux;   // Numerical flux that maps F(u±,x) to F̂
   const FluxFunction &fluxFunction;
   const int IntOrderOffset; // integration order offset, 2*p + IntOrderOffset.
   const real_t sign;
#ifndef MFEM_THREAD_SAFE
   // Local storage for element integration
   Vector shape;              // shape function value at an integration point
   Vector state;              // state value at an integration point
   DenseMatrix flux;          // flux value at an integration point
   DenseTensor J;             // Jacobian matrix at an integration point
   DenseMatrix dshape;  // derivative of shape function at an integration point

   Vector shape1;  // shape function value at an integration point - first elem
   // Vector shape2;  // shape function value at an integration point - second elem // not used
   Vector state1;  // state value at an integration point - first elem
   Vector state2;  // state value at an integration point - second elem
   Vector nor;     // normal vector, see mfem::CalcOrtho()
   Vector fluxN;   // F̂(u±,x) n
   DenseMatrix JDotN;   // Ĵ(u±,x) n
#endif

public:
   const int num_equations;  // the number of equations
   /**
    * @brief Construct a new Hyperbolic Form Integrator object
    *
    * @param[in] numFlux numerical flux
    * @param[in] IntOrderOffset integration order offset
    * @param[in] sign sign of the convection term
    */
   BdrHyperbolicFluxIntegrator(
      const NumericalFlux &numFlux,
      const int IntOrderOffset = 0,
      const real_t sign = 1.);

   /**
    * @brief Reset the Max Char Speed 0
    *
    */
   void ResetMaxCharSpeed()
   {
      max_char_speed = 0.0;
   }

   real_t GetMaxCharSpeed()
   {
      return max_char_speed;
   }

   const FluxFunction &GetFluxFunction() { return fluxFunction; }

   // /**
   //  * @brief Implements (F(u), ∇v) with abstract F computed by
   //  * FluxFunction::ComputeFlux()
   //  *
   //  * @param[in] el local finite element
   //  * @param[in] Tr element transformation
   //  * @param[in] elfun local coefficient of basis
   //  * @param[out] elvect evaluated dual vector
   //  */
   // void AssembleElementVector(const FiniteElement &el,
   //                            ElementTransformation &Tr,
   //                            const Vector &elfun, Vector &elvect) override;

   // /**
   //  * @brief Implements (J(u), ∇v) with abstract J computed by
   //  * FluxFunction::ComputeFluxJacobian()
   //  *
   //  * @param[in] el local finite element
   //  * @param[in] Tr element transformation
   //  * @param[in] elfun local coefficient of basis
   //  * @param[out] grad evaluated Jacobian
   //  */
   // void AssembleElementGrad(const FiniteElement &el,
   //                          ElementTransformation &Tr,
   //                          const Vector &elfun, DenseMatrix &grad) override;

   /**
    * @brief Implements <-F̂(u⁻,u⁺,x) n, [v]> with abstract F̂ computed by
    * NumericalFlux::Eval() of the numerical flux object
    *
    * @param[in] el1 finite element of the first element
    * @param[in] el2 finite element of the second element
    * @param[in] Tr face element transformations
    * @param[in] elfun local coefficient of basis from both elements
    * @param[out] elvect evaluated dual vector <-F̂(u⁻,u⁺,x) n, [v]>
    */
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

   // /**
   //  * @brief Implements <-Ĵ(u⁻,u⁺,x) n, [v]> with abstract Ĵ computed by
   //  * NumericalFlux::Grad() of the numerical flux object
   //  *
   //  * @param[in] el1 finite element of the first element
   //  * @param[in] el2 finite element of the second element
   //  * @param[in] Tr face element transformations
   //  * @param[in] elfun local coefficient of basis from both elements
   //  * @param[out] elmat evaluated Jacobian matrix <-Ĵ(u⁻,u⁺,x) n, [v]>
   //  */
   // void AssembleFaceGrad(const FiniteElement &el1,
   //                       const FiniteElement &el2,
   //                       FaceElementTransformations &Tr,
   //                       const Vector &elfun, DenseMatrix &elmat) override;
};




} // namespace mfem

#endif // MFEM_HYPERBOLIC_BDR
