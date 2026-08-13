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

enum class GenericEulerCpuStatus : std::uint8_t
{
  ready,
  accepted,
  rejected,
  boundary_failure,
  collective_failure,
  unsupported_restart
};

/**
 * Value-only evidence for one generic CPU assembly.  The result is written
 * only after the complete collective operation has succeeded.
 */
struct GenericEulerCpuResult
{
  GenericEulerCpuStatus status = GenericEulerCpuStatus::ready;
  generic::ContractStatus admission = generic::ContractStatus::accepted;
  int rank = -1;
  int ranks = 0;
  int local_elements = 0;
  int polynomial_order = 0;
  int neighbor_count = 0;
  std::uint64_t boundary_calls = 0;
  std::uint64_t local_residual_digest = 0;
  std::uint64_t ordered_exchange_digest = 0;
  bool assembled = false;
};

/**
 * Assemble one conventional Euler HDG CPU residual over caller-borrowed
 * inputs.  Initial and optional source views are copied only into temporary
 * local FE fields.  Shared-neighbor values are exchanged in rank order.
 *
 * On rejection, callback failure, MPI failure, or an internal exception this
 * returns false and leaves `result` byte-for-byte unchanged.
 */
bool AssembleGenericEulerHdgCpu(const generic::EulerHdgRequest& request,
                                GenericEulerCpuResult& result) noexcept;

/** Generic borrowed-mesh restart is intentionally unsupported and atomic. */
bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest& request,
                               GenericEulerCpuResult& result) noexcept;

} // namespace dg::hdg::runtime::cpu
