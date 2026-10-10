#pragma once

/**
 * @file numerics/hdg/generic/boundary_context.hpp
 * @brief Owned Euler boundary-policy context for the generic HDG contract.
 *
 * The context owns immutable per-attribute policies.  Its `View()` returns a
 * borrowed `BoundaryOperatorView` suitable for one engine invocation; callers
 * must keep the context alive for at least as long as that view is used.
 */

#include "numerics/hdg/generic/contract.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace dg_sim
{
class EulerBoundaryPlan;
}

namespace dg::hdg::generic
{

class EulerBoundaryContext
{
public:
  EulerBoundaryContext() = default;

  BoundaryOperatorView View() const noexcept;
  int SpaceDimension() const noexcept { return space_dimension_; }
  double Gamma() const noexcept { return gamma_; }
  std::size_t AttributeCount() const noexcept { return policies_.size(); }
  bool IsPeriodicTopologyAttribute(int attribute) const noexcept;
  EulerBoundaryPolicyKind PolicyForAttribute(int attribute) const noexcept;

private:
  friend EulerBoundaryContext LowerHdgEulerBoundaryPlan(
      const dg_sim::EulerBoundaryPlan&, int, double, int);

  EulerBoundaryContext(int space_dimension,
                       double gamma,
                       std::vector<EulerBoundaryPolicy> policies)
    : space_dimension_(space_dimension),
      gamma_(gamma),
      policies_(std::move(policies))
  {
  }

  static bool Evaluate(const void* context,
                       const BoundaryPointView& point,
                       const double* interior_state,
                       double* exterior_state) noexcept;

  int space_dimension_ = 0;
  double gamma_ = 0.0;
  std::vector<EulerBoundaryPolicy> policies_;
};

} // namespace dg::hdg::generic
