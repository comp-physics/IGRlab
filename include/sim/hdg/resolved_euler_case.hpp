#pragma once

/**
 * @file sim/hdg/resolved_euler_case.hpp
 * @brief Resolved sim-layer inputs for one generic Euler HDG case session.
 */

#include <string>
#include <type_traits>

#include "numerics/hdg/generic/contract.hpp"
#include "sim/time/backend/time_control.hpp"

namespace dg_sim::hdg
{

/** Typed time/control pairing retained by one resolved general Euler case. */
enum class HdgEulerTimeMode
{
  FixedBackwardEuler = 0,
  CFLBackwardEuler = 1,
  FixedPartitionedBackwardEuler = 2,
  CFLPartitionedBackwardEuler = 3,
  IMEXReserved = 4,
  FixedRodas3 = 5,
  AdaptiveRodas3 = 6
};

/** Intent only: publication implementation and fields remain caller-owned. */
struct HdgEulerOutputIntent
{
  bool enabled = false;
  std::string prefix_path;
  std::string collection_name;
};

/**
 * A resolved, caller-owned HDG case.  The generic request carries borrowed
 * mesh, initial/source field views, and backend admission data; the remaining
 * fields retain only time/output intent.  request.boundary is the sole
 * execution boundary view; lowering/provenance remain with the sim adapter
 * that constructs this resolved value.
 */
struct ResolvedHdgEulerCasePlan
{
  dg::hdg::generic::EulerHdgRequest request{};
  time::backend::TimeControl time_control{};
  time::backend::OutputSchedule output_schedule{};
  HdgEulerTimeMode time_mode = HdgEulerTimeMode::FixedBackwardEuler;
  HdgEulerOutputIntent output{};
};

static_assert(!std::is_polymorphic<ResolvedHdgEulerCasePlan>::value,
              "resolved HDG case plan must remain nonvirtual");

} // namespace dg_sim::hdg
