// Implementation of hyperbolic conservation laws based on MFEM

#include "mfem.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "physics/igr/igr_flux.hpp"

#include <cmath>
// #include "hyperbolic.hpp"
// #include "nonlinearform.hpp"
// #include "pnonlinearform.hpp"

namespace mfem
{

namespace
{

bool EulerStateMeetsInvariantDomainBoundaryContract(
  const Vector& state,
  const int equations,
  const int dim,
  const real_t gamma,
  const real_t density_floor,
  const real_t pressure_floor)
{
  bool finite = state.Size() == equations;
  for (int equation = 0; finite && equation < state.Size(); ++equation)
  {
    finite = std::isfinite(state(equation));
  }
  if (!finite || !(state(0) >= density_floor))
  {
    return false;
  }

  real_t momentum_squared = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    momentum_squared += state(1 + d) * state(1 + d);
  }
  const real_t pressure =
    (gamma - 1.0) *
    (state(1 + dim) - 0.5 * momentum_squared / state(0));
  return std::isfinite(pressure) && pressure >= pressure_floor;
}

void VerifyInvariantDomainBoundaryPolicy(const BoundaryState& boundary,
                                         const int equations,
                                         const int dim,
                                         const real_t gamma,
                                         const real_t density_floor,
                                         const real_t pressure_floor)
{
  const auto verify_state = [&](const Vector& state, const char* name)
  {
    MFEM_VERIFY(EulerStateMeetsInvariantDomainBoundaryContract(state,
                                                                equations,
                                                                dim,
                                                                gamma,
                                                                density_floor,
                                                                pressure_floor),
                "DG invariant-domain " << name
                                        << " violates the configured finite "
                                           "density/pressure contract.");
  };
  const auto verify_dimension = [&](const int boundary_dim, const char* name)
  {
    MFEM_VERIFY(boundary_dim == dim,
                "DG invariant-domain " << name
                                        << " dimension differs from the "
                                           "Euler flux dimension.");
  };
  const auto verify_gamma = [&](const real_t boundary_gamma, const char* name)
  {
    MFEM_VERIFY(boundary_gamma == gamma,
                "DG invariant-domain " << name
                                        << " gamma differs from the Euler+IGR "
                                           "flux gamma.");
  };
  const auto verify_floors = [&](const real_t boundary_density_floor,
                                 const real_t boundary_pressure_floor,
                                 const char* name)
  {
    MFEM_VERIFY(boundary_density_floor == density_floor &&
                  boundary_pressure_floor == pressure_floor,
                "DG invariant-domain " << name
                                        << " floors differ from the admitted-"
                                           "state flux contract.");
  };

  if (const auto* inlet =
        dynamic_cast<const SupersonicInletState*>(&boundary))
  {
    verify_state(inlet->GetInletState(), "supersonic-inlet reference state");
    return;
  }
  if (dynamic_cast<const SupersonicOutletState*>(&boundary))
  {
    return;
  }
  if (const auto* outlet =
        dynamic_cast<const SubsonicOutletPressureState*>(&boundary))
  {
    verify_dimension(outlet->GetDimension(), "subsonic-outlet-pressure");
    verify_gamma(outlet->GetGamma(), "subsonic-outlet-pressure");
    verify_floors(outlet->GetDensityFloor(),
                  outlet->GetPressureFloor(),
                  "subsonic-outlet-pressure");
    MFEM_VERIFY(std::isfinite(outlet->GetOutletPressure()) &&
                  outlet->GetOutletPressure() >= pressure_floor,
                "DG invariant-domain outlet pressure violates the configured "
                "pressure floor.");
    return;
  }
  if (const auto* outlet =
        dynamic_cast<const NonReflectingOutletState*>(&boundary))
  {
    verify_dimension(outlet->GetDimension(), "nonreflecting-outlet");
    verify_gamma(outlet->GetGamma(), "nonreflecting-outlet");
    verify_floors(outlet->GetDensityFloor(),
                  outlet->GetPressureFloor(),
                  "nonreflecting-outlet");
    verify_state(outlet->GetReferenceState(),
                 "nonreflecting-outlet reference state");
    return;
  }
  if (const auto* inlet = dynamic_cast<const SubsonicInletState*>(&boundary))
  {
    verify_dimension(inlet->GetDimension(), "subsonic-inlet");
    verify_gamma(inlet->GetGamma(), "subsonic-inlet");
    verify_floors(inlet->GetDensityFloor(),
                  inlet->GetPressureFloor(),
                  "subsonic-inlet");
    verify_state(inlet->GetInletState(), "subsonic-inlet reference state");
    return;
  }
  if (const auto* farfield = dynamic_cast<const FarFieldState*>(&boundary))
  {
    verify_dimension(farfield->GetDimension(), "far-field");
    verify_gamma(farfield->GetGamma(), "far-field");
    verify_floors(farfield->GetDensityFloor(),
                  farfield->GetPressureFloor(),
                  "far-field");
    verify_state(farfield->GetFarFieldState(), "far-field reference state");
    return;
  }
  if (const auto* wall = dynamic_cast<const SlipWallState*>(&boundary))
  {
    verify_dimension(wall->GetDimension(), "slip-wall");
    verify_floors(wall->GetDensityFloor(),
                  wall->GetPressureFloor(),
                  "slip-wall");
    return;
  }

  MFEM_ABORT("DG invariant-domain mode does not support this physical "
             "boundary-state policy.");
}

} // namespace

// --------- Boundary-face integrator ---------

BdrHyperbolicFluxIntegrator::BdrHyperbolicFluxIntegrator(
  const NumericalFlux& numFlux,
  std::unique_ptr<BoundaryState> bc_,
  const int IntOrderOffset,
  real_t sign)
  : NonlinearFormIntegrator(),
    //   max_char_speed(0.0),
    numFlux(numFlux),
    fluxFunction(numFlux.GetFluxFunction()),
    bc(std::move(bc_)),
    IntOrderOffset(IntOrderOffset),
    sign(sign),
    num_equations(fluxFunction.num_equations)
{
  MFEM_VERIFY(bc != nullptr,
              "BdrHyperbolicFluxIntegrator requires a boundary-state policy.");
  if (const auto* igr_flux = dynamic_cast<const FluxWithIGR*>(&fluxFunction);
      igr_flux && igr_flux->GetInvariantDomainFlux())
  {
    invariant_domain_boundary_contract_ = true;
    invariant_domain_gamma_ = igr_flux->GetGamma();
    invariant_domain_density_floor_ = igr_flux->GetVelocityRhoFloor();
    invariant_domain_pressure_floor_ =
      igr_flux->GetInvariantDomainPressureFloor();
    MFEM_VERIFY(std::isfinite(invariant_domain_gamma_) &&
                  invariant_domain_gamma_ > 1.0 &&
                  std::isfinite(invariant_domain_density_floor_) &&
                  invariant_domain_density_floor_ > 0.0 &&
                  std::isfinite(invariant_domain_pressure_floor_) &&
                  invariant_domain_pressure_floor_ > 0.0,
                "DG invariant-domain physical boundaries require finite "
                "Euler gamma and positive density/pressure floors.");
    VerifyInvariantDomainBoundaryPolicy(*bc,
                                        num_equations,
                                        fluxFunction.dim,
                                        invariant_domain_gamma_,
                                        invariant_domain_density_floor_,
                                        invariant_domain_pressure_floor_);
  }
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

void BdrHyperbolicFluxIntegrator::BeginBoundaryFluxIntegralCapture()
{
  boundary_flux_integral_.SetSize(num_equations);
  boundary_flux_integral_ = 0.0;
  boundary_flux_integral_capture_active_ = true;
}

void BdrHyperbolicFluxIntegrator::AssembleFaceVector(
  const FiniteElement& el1,
  const FiniteElement&,
  FaceElementTransformations& Tr,
  const Vector& elfun,
  Vector& elvect)
{
  // current elements' the number of degrees of freedom
  // does not consider the number of equations
  const int dof1 = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
  // Local storage for element integration

  // shape function value at an integration point - first elem
  Vector shape1(dof1);
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
#endif

  elvect.SetSize(dof1 * num_equations);
  elvect = 0.0;

  const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);

  // Obtain integration rule. If integration is rule is given, then use it.
  // Otherwise, get (2*p + IntOrderOffset) order integration rule
  const IntegrationRule* ir = IntRule;
  if (!ir)
  {
    const int order = 2 * el1.GetOrder() + IntOrderOffset;
    ir = &IntRules.Get(Tr.GetGeometryType(), order);
  }
  // loop over integration points
  for (int i = 0; i < ir->GetNPoints(); i++)
  {
    const IntegrationPoint& ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip); // set face and element int. points

    // Calculate basis functions on the interior element at the face.
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);

    // Interpolate elfun at the point
    elfun1_mat.MultTranspose(shape1, state1);

    if (invariant_domain_boundary_contract_)
    {
      MFEM_VERIFY(EulerStateMeetsInvariantDomainBoundaryContract(
                    state1,
                    num_equations,
                    fluxFunction.dim,
                    invariant_domain_gamma_,
                    invariant_domain_density_floor_,
                    invariant_domain_pressure_floor_),
                  "DG invariant-domain physical-boundary interior state is "
                  "non-finite or outside the configured admitted set.");
    }

    // Get the normal vector and the flux on the face
    if (nor.Size() == 1) // if 1D, use 1 or -1.
    {
      nor(0) = 2.0 * Tr.GetElement1IntPoint().x - 1.0;
    }
    else
    {
      CalcOrtho(Tr.Jacobian(), nor);
    }
    // Compute State 2 s.t. it imposed the BC
    // Ghost state U+
    bc->Eval(state1, nor, Tr, state2);

    if (invariant_domain_boundary_contract_)
    {
      MFEM_VERIFY(EulerStateMeetsInvariantDomainBoundaryContract(
                    state2,
                    num_equations,
                    fluxFunction.dim,
                    invariant_domain_gamma_,
                    invariant_domain_density_floor_,
                    invariant_domain_pressure_floor_),
                  "DG invariant-domain physical-boundary ghost state is "
                  "non-finite or outside the configured admitted set.");
    }

    // Compute F(u+, x) and F(u-, x) with maximum characteristic speed
    // Compute hat(F) using evaluated quantities
    const real_t speed = numFlux.Eval(state1, state2, nor, Tr, fluxN);

    if (boundary_flux_integral_capture_active_)
    {
      boundary_flux_integral_.Add(ip.weight, fluxN);
    }

    // Update the global max char speed
    max_char_speed = std::max(speed, max_char_speed);

    // pre-multiply integration weight to flux contribution
    AddMult_a_VWt(-ip.weight * sign, shape1, fluxN, elvect1_mat);
    //   AddMult_a_VWt(+ip.weight*sign, shape2, fluxN, elvect2_mat);
  }
}

} // namespace mfem
