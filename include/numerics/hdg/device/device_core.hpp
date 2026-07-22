#pragma once

/**
 * @file numerics/hdg/device/device_core.hpp
 * @brief MFEM-managed single-device FP64 HDG local condensation baseline.
 *
 * The engine intentionally owns only local, uniformly-shaped batches.  It
 * does not select a formulation, assemble a global trace system, or perform
 * MPI work.  When MFEM has not been configured for exactly one device, every
 * operation delegates to the accepted HostReferenceEngine.  That fallback is
 * part of the API contract rather than a numerically distinct code path.
 */

#include "numerics/hdg/core/host_reference.hpp"

#include <mfem.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dg::hdg::device
{
enum class ExecutionPath : unsigned char
{
  host_reference,
  mfem_device
};

/** Monotonic evidence for owned persistent storage and explicit transfers. */
struct TransferAccounting
{
  std::size_t persistent_bytes = 0;
  std::size_t allocation_events = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_bytes = 0;
  std::size_t host_reference_fallbacks = 0;
};

/**
 * Correctness-first, fixed-shape FP64 local HDG engine.
 *
 * Device kernels use one logical MFEM forall entry per element and retain
 * pivoted LU(A), B, C, D, and f for matrix-free Schur application and
 * recovery.  The persistent vectors use MFEM's selected memory manager, so
 * CUDA, RAJA/Umpire, and host-only builds retain their normal MFEM ownership
 * semantics.  There is no virtual dispatch in a local or device loop.
 */
class DeviceCoreEngine
{
public:
  core::Status Condense(const core::LocalSystemBatchView& input,
                        const core::CondensedBatchView& output) noexcept;
  core::Status ApplySchur(StridedBlockView<const double> trace,
                          StridedBlockView<double> result) noexcept;
  core::Status Recover(StridedBlockView<const double> trace,
                       StridedBlockView<double> interior) noexcept;

  bool ready() const noexcept { return ready_; }
  ExecutionPath execution_path() const noexcept { return path_; }
  bool using_device() const noexcept
  {
    return path_ == ExecutionPath::mfem_device;
  }
  LocalBlockShape shape() const noexcept { return shape_; }
  std::size_t batch_count() const noexcept { return batch_count_; }
  const TransferAccounting& accounting() const noexcept { return accounting_; }

private:
  bool ResizePersistent(std::size_t ni, std::size_t nt,
                        std::size_t batches) noexcept;
  bool DeviceEligible() const noexcept;
  void RecordHostToDevice(std::size_t bytes) noexcept;
  void RecordDeviceToHost(std::size_t bytes) noexcept;

  core::HostReferenceEngine reference_;
  LocalBlockShape shape_{};
  std::size_t batch_count_ = 0;
  bool ready_ = false;
  ExecutionPath path_ = ExecutionPath::host_reference;
  TransferAccounting accounting_{};

  // MFEM vectors own their memory through the active MFEM memory manager.
  mfem::Vector lu_, pivots_, b_, c_, d_, f_, solve_;
  mfem::Vector schur_, rhs_, trace_, action_, interior_;
};

static_assert(!std::is_polymorphic<DeviceCoreEngine>::value,
              "HDG device core must not use virtual dispatch");
}  // namespace dg::hdg::device
