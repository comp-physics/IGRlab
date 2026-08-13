#pragma once

/**
 * @file numerics/hdg/runtime/cpu/generic_euler.hpp
 * @brief Borrowed-mesh conventional Euler HDG CPU assembly entry.
 *
 * This is deliberately a one-shot engine boundary, not another persistent
 * runtime.  It creates no mesh, does not repartition/refine/destroy the
 * caller's mesh, and retains neither the mesh nor any supplied field view.
 */

#include "numerics/hdg/generic/contract.hpp"

#include <cstdint>

namespace dg::hdg::runtime::cpu
{

/**
 * Execute one conventional Euler HDG CPU step over caller-borrowed inputs.
 * The external mesh and field storage remain caller-owned.  Rejected input,
 * callback failure, and unsupported restart leave all borrowed state intact.
 */
bool ExecuteGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept;

/** Generic borrowed-mesh restart is intentionally unsupported and atomic. */
bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept;

} // namespace dg::hdg::runtime::cpu
