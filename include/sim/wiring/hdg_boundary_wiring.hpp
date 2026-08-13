#pragma once

/**
 * @file sim/wiring/hdg_boundary_wiring.hpp
 * @brief Lower semantic Euler boundary plans to generic HDG boundary policies.
 */

#include "numerics/hdg/generic/boundary_context.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"

namespace dg::hdg::generic
{

/**
 * @brief Build an owned HDG boundary context from a backend-neutral plan.
 *
 * Every physical boundary attribute in `[1, boundary_attribute_count]` must
 * occur exactly once.  Periodic entries are recorded as topology only and do
 * not provide a physical exterior-state operator.
 *
 * @throws std::invalid_argument for invalid dimensions, markers, callbacks,
 *         overlapping attributes, or uncovered physical attributes.
 */
EulerBoundaryContext LowerHdgEulerBoundaryPlan(
    const dg_sim::EulerBoundaryPlan& plan,
    int space_dimension,
    double gamma,
    int boundary_attribute_count);

} // namespace dg::hdg::generic
