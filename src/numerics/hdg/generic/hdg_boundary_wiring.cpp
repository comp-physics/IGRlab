#include "sim/wiring/hdg_boundary_wiring.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dg::hdg::generic
{

namespace
{

[[noreturn]] void Reject(const char* const reason)
{
  throw std::invalid_argument(reason);
}

void ValidateState(const mfem::Vector& state)
{
  if (state.Size() != euler_2d_conserved_components)
  {
    Reject("HDG boundary lowering requires a four-component Euler state.");
  }
  for (int component = 0; component < state.Size(); ++component)
  {
    if (!std::isfinite(state(component)))
    {
      Reject("HDG boundary lowering requires finite Euler state data.");
    }
  }
}

EulerBoundaryPolicy PolicyFor(const dg_sim::EulerBoundaryPlanEntry& entry)
{
  EulerBoundaryPolicy policy;
  switch (entry.kind)
  {
  case dg_sim::EulerBoundaryKind::Periodic:
    policy.kind = EulerBoundaryPolicyKind::periodic_topology;
    return policy;
  case dg_sim::EulerBoundaryKind::Inlet:
    if (!entry.state)
    {
      Reject("HDG boundary lowering requires an inlet state callback.");
    }
    policy.kind = entry.inlet_model == dg_sim::EulerInletModel::Supersonic
        ? EulerBoundaryPolicyKind::supersonic_inlet
        : EulerBoundaryPolicyKind::subsonic_inlet;
    break;
  case dg_sim::EulerBoundaryKind::Outlet:
    switch (entry.outlet_model)
    {
    case dg_sim::EulerOutletModel::Supersonic:
      policy.kind = EulerBoundaryPolicyKind::supersonic_outlet;
      return policy;
    case dg_sim::EulerOutletModel::SubsonicPressure:
      if (!entry.pressure)
      {
        Reject("HDG boundary lowering requires an outlet pressure callback.");
      }
      policy.kind = EulerBoundaryPolicyKind::subsonic_outlet_pressure;
      policy.pressure_out = entry.pressure();
      if (!std::isfinite(policy.pressure_out))
      {
        Reject("HDG boundary lowering requires a finite outlet pressure.");
      }
      return policy;
    case dg_sim::EulerOutletModel::FarField:
      if (!entry.state)
      {
        Reject("HDG boundary lowering requires a far-field state callback.");
      }
      policy.kind = EulerBoundaryPolicyKind::far_field;
      break;
    }
    break;
  case dg_sim::EulerBoundaryKind::Slip:
    policy.kind = EulerBoundaryPolicyKind::slip_wall;
    return policy;
  case dg_sim::EulerBoundaryKind::NonReflecting:
    if (!entry.state)
    {
      Reject("HDG boundary lowering requires a non-reflecting state callback.");
    }
    policy.kind = EulerBoundaryPolicyKind::nonreflecting_outlet;
    break;
  }

  const mfem::Vector state = entry.state();
  ValidateState(state);
  for (int component = 0; component < euler_2d_conserved_components; ++component)
  {
    policy.reference_state[static_cast<std::size_t>(component)] = state(component);
  }
  return policy;
}

} // namespace

EulerBoundaryContext LowerHdgEulerBoundaryPlan(
    const dg_sim::EulerBoundaryPlan& plan,
    const int space_dimension,
    const double gamma,
    const int boundary_attribute_count)
{
  if (space_dimension != 2)
  {
    Reject("HDG boundary lowering supports only two-dimensional Euler states.");
  }
  if (!std::isfinite(gamma) || gamma <= 1.0)
  {
    Reject("HDG boundary lowering requires finite gamma greater than one.");
  }
  if (boundary_attribute_count < 0)
  {
    Reject("HDG boundary lowering requires a non-negative attribute count.");
  }

  std::vector<EulerBoundaryPolicy> policies(
      static_cast<std::size_t>(boundary_attribute_count));
  std::vector<bool> assigned(static_cast<std::size_t>(boundary_attribute_count), false);
  for (const dg_sim::EulerBoundaryPlanEntry& entry : plan.Entries())
  {
    if (entry.marker.Size() != boundary_attribute_count)
    {
      Reject("HDG boundary lowering marker size must match boundary attributes.");
    }
    const EulerBoundaryPolicy policy = PolicyFor(entry);
    bool selects_attribute = false;
    for (int index = 0; index < boundary_attribute_count; ++index)
    {
      const int marked = entry.marker[index];
      if (marked != 0 && marked != 1)
      {
        Reject("HDG boundary lowering markers must contain only zero or one.");
      }
      if (marked == 0)
      {
        continue;
      }
      selects_attribute = true;
      if (assigned[static_cast<std::size_t>(index)])
      {
        Reject("HDG boundary lowering does not permit overlapping attributes.");
      }
      assigned[static_cast<std::size_t>(index)] = true;
      policies[static_cast<std::size_t>(index)] = policy;
    }
    if (!selects_attribute)
    {
      Reject("HDG boundary lowering entries must select a boundary attribute.");
    }
  }
  for (const bool is_assigned : assigned)
  {
    if (!is_assigned)
    {
      Reject("HDG boundary lowering requires every boundary attribute to be covered.");
    }
  }
  return EulerBoundaryContext(space_dimension, gamma, std::move(policies));
}

} // namespace dg::hdg::generic
