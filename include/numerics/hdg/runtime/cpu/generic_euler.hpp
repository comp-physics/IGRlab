#pragma once

/**
 * @file numerics/hdg/runtime/cpu/generic_euler.hpp
 * @brief Borrowed-mesh conventional Euler HDG CPU assembly entry.
 *
 * The compatibility entry below remains one-shot.  GenericEulerHdgCpuSession
 * is the compact persistent boundary: it owns one existing CPU numerics core,
 * while the mesh and field-view payloads remain caller-borrowed.
 */

#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/generic/session_contract.hpp"

#include <cstdint>
#include <cstddef>
#include <memory>

namespace dg::hdg::runtime::cpu::detail
{
class CpuNumericsCore;
}

namespace dg::hdg::runtime::cpu
{

enum class GenericEulerCpuSessionStatus : std::uint8_t
{
  uninitialized,
  initialized,
  prepared,
  committed,
  aborted,
  advanced,
  rejected
};

/** Compact committed-time coordinates; no history or runtime evidence leaks. */
struct GenericEulerCpuSessionCoordinates
{
  double time = 0.0;
  std::uint64_t step = 0;
  std::uint64_t generation = 0;
  double dt = 0.0;
};

/** Public session result is deliberately limited to status and coordinates. */
struct GenericEulerCpuSessionResult
{
  GenericEulerCpuSessionStatus status =
      GenericEulerCpuSessionStatus::uninitialized;
  GenericEulerCpuSessionCoordinates coordinates{};
};

/** Typed caller-owned host storage for committed physical interior values. */
struct GenericEulerCpuInteriorBuffer
{
  double* data = nullptr;
  std::size_t size = 0;
};

class GenericEulerHdgCpuSession;

/**
 * Backend-specific, move-only authority for one prepared CPU candidate.
 * It is minted only by GenericEulerHdgCpuSession::Prepare and cannot be used
 * with a different session or after Commit/Abort.  Every live authority must
 * be consumed collectively by Commit or Abort on its session communicator.
 * It has no destructor rollback: abandoning it deliberately leaves the
 * session fail-closed and unusable rather than inventing noncollective
 * recovery.
 */
class GenericEulerCpuPreparedStep
{
public:
  GenericEulerCpuPreparedStep() noexcept = default;
  GenericEulerCpuPreparedStep(const GenericEulerCpuPreparedStep&) = delete;
  GenericEulerCpuPreparedStep& operator=(const GenericEulerCpuPreparedStep&) =
      delete;
  GenericEulerCpuPreparedStep(GenericEulerCpuPreparedStep&& other) noexcept
  {
    MoveFrom(other);
  }
  GenericEulerCpuPreparedStep& operator=(GenericEulerCpuPreparedStep&&) =
      delete;

private:
  friend class GenericEulerHdgCpuSession;

  void MoveFrom(GenericEulerCpuPreparedStep& other) noexcept
  {
    owner_ = other.owner_;
    epoch_ = other.epoch_;
    generation_ = other.generation_;
    valid_ = other.valid_;
    other.valid_ = false;
  }

  const GenericEulerHdgCpuSession* owner_ = nullptr;
  std::uint64_t epoch_ = 0;
  std::uint64_t generation_ = 0;
  bool valid_ = false;
};

/**
 * Nonvirtual generic CPU session.  Initialize accepts the borrowed request
 * once; its mesh, field payloads, and boundary context must remain valid until
 * this session is destroyed.  Failed advances retain the last committed state
 * and coordinates.
 */
class GenericEulerHdgCpuSession
{
public:
  GenericEulerHdgCpuSession();
  ~GenericEulerHdgCpuSession();
  GenericEulerHdgCpuSession(const GenericEulerHdgCpuSession&) = delete;
  GenericEulerHdgCpuSession& operator=(const GenericEulerHdgCpuSession&) =
      delete;
  GenericEulerHdgCpuSession(GenericEulerHdgCpuSession&&) = delete;
  GenericEulerHdgCpuSession& operator=(GenericEulerHdgCpuSession&&) = delete;

  GenericEulerCpuSessionResult Initialize(
      const generic::EulerHdgRequest&) noexcept;
  GenericEulerCpuSessionResult Prepare(
      double dt, GenericEulerCpuPreparedStep&) noexcept;
  GenericEulerCpuSessionResult Commit(GenericEulerCpuPreparedStep&&) noexcept;
  GenericEulerCpuSessionResult Abort(GenericEulerCpuPreparedStep&&) noexcept;
  GenericEulerCpuSessionResult Advance(double dt) noexcept;
  GenericEulerCpuSessionCoordinates coordinates() const noexcept;
  std::uint32_t capabilities() const noexcept;
  /** Trace-inclusive diagnostics for the generation-bound committed state. */
  generic::HdgEulerCommittedDiagnostics diagnostics() const noexcept;

  /** Copy only the committed physical interior; trace values are never output. */
  bool ExportCommittedInterior(GenericEulerCpuInteriorBuffer) const noexcept;
  bool ExportCommittedInterior(mfem::ParGridFunction&) const noexcept;
  bool ExportPreparedInterior(const GenericEulerCpuPreparedStep&,
                              GenericEulerCpuInteriorBuffer) const noexcept;
  bool ExportPreparedInterior(const GenericEulerCpuPreparedStep&,
                              mfem::ParGridFunction&) const noexcept;

private:
  std::unique_ptr<detail::CpuNumericsCore> core_;
  generic::EulerHdgRequest request_{};
  GenericEulerCpuSessionCoordinates coordinates_{};
  generic::HdgEulerCommittedDiagnostics diagnostics_{};
  std::uint64_t session_epoch_ = 0;
  bool initialization_attempted_ = false;
  bool initialized_ = false;
};

/**
 * Execute one conventional Euler HDG CPU step over caller-borrowed inputs.
 * The external mesh and field storage remain caller-owned.  Rejected input,
 * callback failure, and unsupported restart leave all borrowed state intact.
 */
bool ExecuteGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept;

/** Generic borrowed-mesh restart is intentionally unsupported and atomic. */
bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept;

} // namespace dg::hdg::runtime::cpu
