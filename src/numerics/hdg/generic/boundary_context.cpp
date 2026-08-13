#include "numerics/hdg/generic/boundary_context.hpp"

#include "physics/bc/boundary_state.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"

#include <cmath>

namespace dg::hdg::generic
{

namespace
{

mfem::EulerExteriorStateKind ToExteriorStateKind(
    const EulerBoundaryPolicyKind kind)
{
  switch (kind)
  {
  case EulerBoundaryPolicyKind::supersonic_inlet:
    return mfem::EulerExteriorStateKind::SupersonicInlet;
  case EulerBoundaryPolicyKind::subsonic_inlet:
    return mfem::EulerExteriorStateKind::SubsonicInlet;
  case EulerBoundaryPolicyKind::supersonic_outlet:
    return mfem::EulerExteriorStateKind::SupersonicOutlet;
  case EulerBoundaryPolicyKind::subsonic_outlet_pressure:
    return mfem::EulerExteriorStateKind::SubsonicOutletPressure;
  case EulerBoundaryPolicyKind::far_field:
    return mfem::EulerExteriorStateKind::FarField;
  case EulerBoundaryPolicyKind::nonreflecting_outlet:
    return mfem::EulerExteriorStateKind::NonReflectingOutlet;
  case EulerBoundaryPolicyKind::slip_wall:
    return mfem::EulerExteriorStateKind::SlipWall;
  case EulerBoundaryPolicyKind::periodic_topology:
    break;
  }
  return mfem::EulerExteriorStateKind::SupersonicOutlet;
}

bool NeedsReferenceState(const EulerBoundaryPolicyKind kind) noexcept
{
  return kind == EulerBoundaryPolicyKind::supersonic_inlet ||
      kind == EulerBoundaryPolicyKind::subsonic_inlet ||
      kind == EulerBoundaryPolicyKind::far_field ||
      kind == EulerBoundaryPolicyKind::nonreflecting_outlet;
}

} // namespace

BoundaryOperatorView EulerBoundaryContext::View() const noexcept
{
  return {this, Evaluate};
}

bool EulerBoundaryContext::IsPeriodicTopologyAttribute(const int attribute) const noexcept
{
  return PolicyForAttribute(attribute) ==
      EulerBoundaryPolicyKind::periodic_topology;
}

EulerBoundaryPolicyKind EulerBoundaryContext::PolicyForAttribute(
    const int attribute) const noexcept
{
  if (attribute <= 0 || static_cast<std::size_t>(attribute) > policies_.size())
  {
    return EulerBoundaryPolicyKind::periodic_topology;
  }
  return policies_[static_cast<std::size_t>(attribute - 1)].kind;
}

bool EulerBoundaryContext::Evaluate(const void* const context,
                                    const BoundaryPointView& point,
                                    const double* const interior_state,
                                    double* const exterior_state) noexcept
{
  const auto* const boundary = static_cast<const EulerBoundaryContext*>(context);
  if (boundary == nullptr || interior_state == nullptr || exterior_state == nullptr ||
      point.outward_normal == nullptr ||
      point.space_dimension != boundary->space_dimension_ ||
      point.boundary_attribute <= 0 ||
      static_cast<std::size_t>(point.boundary_attribute) > boundary->policies_.size())
  {
    return false;
  }

  const EulerBoundaryPolicy& policy =
      boundary->policies_[static_cast<std::size_t>(point.boundary_attribute - 1)];
  if (policy.kind == EulerBoundaryPolicyKind::periodic_topology)
  {
    return false;
  }

  try
  {
    mfem::Vector interior(const_cast<double*>(interior_state),
                          euler_2d_conserved_components);
    mfem::Vector normal(const_cast<double*>(point.outward_normal),
                        boundary->space_dimension_);
    mfem::Vector exterior(exterior_state, euler_2d_conserved_components);
    mfem::Vector reference(const_cast<double*>(policy.reference_state.data()),
                           euler_2d_conserved_components);
    mfem::EvaluateEulerExteriorState(
        ToExteriorStateKind(policy.kind),
        interior,
        normal,
        boundary->space_dimension_,
        boundary->gamma_,
        NeedsReferenceState(policy.kind) ? &reference : nullptr,
        policy.pressure_out,
        exterior);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

} // namespace dg::hdg::generic
