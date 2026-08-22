#include "physics/bc/boundary_state.hpp"
#include "sim/wiring/hdg_boundary_wiring.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
constexpr int kDimension = 2;
constexpr int kComponents = 4;
constexpr int kAttributes = 8;
constexpr double kGamma = 1.4;

mfem::Vector State(double rho, double u, double v, double p)
{
  mfem::Vector state(kComponents);
  state(0) = rho;
  state(1) = rho * u;
  state(2) = rho * v;
  state(3) = p / (kGamma - 1.0) + 0.5 * rho * (u * u + v * v);
  return state;
}

mfem::Array<int> Marker(int attribute)
{
  mfem::Array<int> marker(kAttributes);
  marker = 0;
  marker[attribute - 1] = 1;
  return marker;
}

bool Equal(const std::array<double, kComponents>& actual,
           const mfem::Vector& expected)
{
  for (int component = 0; component < kComponents; ++component)
  {
    if (std::abs(actual[static_cast<std::size_t>(component)] -
                 expected(component)) > 1.0e-12)
    {
      return false;
    }
  }
  return true;
}

bool Call(const dg::hdg::generic::BoundaryOperatorView view,
          const mfem::Vector& normal,
          int attribute,
          const mfem::Vector& interior,
          std::array<double, kComponents>& exterior)
{
  const dg::hdg::generic::BoundaryPointView point{
      nullptr, normal.GetData(), kDimension, attribute};
  return view.exterior_state(view.context, point, interior.GetData(), exterior.data());
}

bool CheckSemanticParity()
{
  const mfem::Vector reference = State(1.1, -0.35, 0.2, 1.3);
  dg_sim::EulerBoundaryPlan plan;
  plan.AddSupersonicInlet(Marker(1), [reference] { return reference; });
  plan.AddSubsonicInlet(Marker(2), [reference] { return reference; });
  plan.AddSupersonicOutlet(Marker(3));
  plan.AddSubsonicOutletPressure(Marker(4), [] { return 0.85; });
  plan.AddFarField(Marker(5), [reference] { return reference; });
  plan.AddNonReflectingOutlet(Marker(6), [reference] { return reference; });
  plan.AddSlipWall(Marker(7));
  plan.AddPeriodic(Marker(8));
  const auto context = dg::hdg::generic::LowerHdgEulerBoundaryPlan(
      plan, kDimension, kGamma, kAttributes);
  const auto view = context.View();
  if (view.context != &context || view.exterior_state == nullptr ||
      view.policies.policies == nullptr ||
      view.policies.count != static_cast<std::size_t>(kAttributes) ||
      view.policies.space_dimension != kDimension ||
      view.policies.gamma != kGamma ||
      !context.IsPeriodicTopologyAttribute(8) || context.IsPeriodicTopologyAttribute(7))
  {
    return false;
  }
  using dg::hdg::generic::EulerBoundaryPolicyKind;
  const auto* const policies = view.policies.policies;
  if (policies[0].kind != EulerBoundaryPolicyKind::supersonic_inlet ||
      policies[1].kind != EulerBoundaryPolicyKind::subsonic_inlet ||
      policies[2].kind != EulerBoundaryPolicyKind::supersonic_outlet ||
      policies[3].kind != EulerBoundaryPolicyKind::subsonic_outlet_pressure ||
      policies[4].kind != EulerBoundaryPolicyKind::far_field ||
      policies[5].kind != EulerBoundaryPolicyKind::nonreflecting_outlet ||
      policies[6].kind != EulerBoundaryPolicyKind::slip_wall ||
      policies[7].kind != EulerBoundaryPolicyKind::periodic_topology ||
      !Equal(policies[0].reference_state, reference) ||
      !Equal(policies[1].reference_state, reference) ||
      !Equal(policies[4].reference_state, reference) ||
      !Equal(policies[5].reference_state, reference) ||
      policies[3].pressure_out != 0.85)
  {
    return false;
  }

  const mfem::Vector interior = State(1.2, 0.25, -0.1, 1.1);
  mfem::Vector normal(kDimension);
  normal = 0.0;
  normal(0) = 1.0;
  mfem::FaceElementTransformations transforms;
  mfem::Vector expected(kComponents);
  std::array<double, kComponents> actual{};

  mfem::SupersonicInletState supersonic_inlet(reference);
  supersonic_inlet.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 1, interior, actual) || !Equal(actual, expected)) return false;
  mfem::SubsonicInletState subsonic_inlet(reference, kDimension, kGamma);
  subsonic_inlet.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 2, interior, actual) || !Equal(actual, expected)) return false;
  mfem::SupersonicOutletState supersonic_outlet;
  supersonic_outlet.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 3, interior, actual) || !Equal(actual, expected)) return false;
  mfem::SubsonicOutletPressureState pressure(kDimension, kGamma, 0.85);
  pressure.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 4, interior, actual) || !Equal(actual, expected)) return false;
  mfem::FarFieldState far_field(reference, kDimension, kGamma);
  far_field.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 5, interior, actual) || !Equal(actual, expected)) return false;
  mfem::NonReflectingOutletState nonreflecting(reference, kDimension, kGamma);
  nonreflecting.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 6, interior, actual) || !Equal(actual, expected)) return false;
  mfem::SlipWallState slip(kDimension);
  slip.Eval(interior, normal, transforms, expected);
  if (!Call(view, normal, 7, interior, actual) || !Equal(actual, expected)) return false;

  const dg::hdg::generic::BoundaryPointView periodic{
      nullptr, normal.GetData(), kDimension, 8};
  return !view.exterior_state(
      view.context, periodic, interior.GetData(), actual.data());
}

bool Rejects(const dg_sim::EulerBoundaryPlan& plan, const char* expected)
{
  try
  {
    static_cast<void>(dg::hdg::generic::LowerHdgEulerBoundaryPlan(
        plan, kDimension, kGamma, kAttributes));
  }
  catch (const std::invalid_argument& error)
  {
    return std::string(error.what()) == expected;
  }
  return false;
}

bool CheckCharacteristicSlipAndMarkers()
{
  const mfem::Vector reference = State(1.0, -1.0, 0.0, 1.0);
  dg_sim::EulerBoundaryPlan plan;
  plan.AddFarField(Marker(1), [reference] { return reference; });
  plan.AddSupersonicOutlet(Marker(2));
  plan.AddSupersonicOutlet(Marker(3));
  plan.AddSupersonicOutlet(Marker(4));
  plan.AddSupersonicOutlet(Marker(5));
  plan.AddSupersonicOutlet(Marker(6));
  plan.AddSlipWall(Marker(7));
  plan.AddPeriodic(Marker(8));
  const auto context = dg::hdg::generic::LowerHdgEulerBoundaryPlan(
      plan, kDimension, kGamma, kAttributes);
  const auto view = context.View();
  mfem::Vector normal(kDimension);
  normal = 0.0;
  normal(0) = 1.0;
  std::array<double, kComponents> exterior{};
  const mfem::Vector inflow = State(0.9, -2.0, 0.1, 1.0);
  if (!Call(view, normal, 1, inflow, exterior) || !Equal(exterior, reference))
  {
    return false;
  }
  const mfem::Vector wall = State(1.25, 3.0, -1.5, 1.0);
  if (!Call(view, normal, 7, wall, exterior) ||
      std::abs(exterior[0] - wall(0)) > 1.0e-12 ||
      std::abs(exterior[1] + wall(1)) > 1.0e-12 ||
      std::abs(exterior[2] - wall(2)) > 1.0e-12 ||
      std::abs(exterior[3] - wall(3)) > 1.0e-12)
  {
    return false;
  }

  dg_sim::EulerBoundaryPlan overlap;
  overlap.AddSlipWall(Marker(1));
  overlap.AddSupersonicOutlet(Marker(1));
  if (!Rejects(overlap,
               "HDG boundary lowering does not permit overlapping attributes."))
  {
    return false;
  }
  dg_sim::EulerBoundaryPlan uncovered;
  uncovered.AddSlipWall(Marker(1));
  if (!Rejects(uncovered,
               "HDG boundary lowering requires every boundary attribute to be covered."))
  {
    return false;
  }
  dg_sim::EulerBoundaryPlan missing_callback;
  missing_callback.AddSupersonicInlet(Marker(1), {});
  if (!Rejects(missing_callback,
               "HDG boundary lowering requires an inlet state callback."))
  {
    return false;
  }
  dg_sim::EulerBoundaryPlan invalid;
  mfem::Array<int> marker(kAttributes);
  marker = 0;
  marker[0] = 2;
  invalid.AddSlipWall(std::move(marker));
  return Rejects(invalid,
                 "HDG boundary lowering markers must contain only zero or one.");
}

} // namespace

int main()
{
  return CheckSemanticParity() && CheckCharacteristicSlipAndMarkers() ? 0 : 1;
}
