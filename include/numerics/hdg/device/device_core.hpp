#pragma once

/**
 * @file numerics/hdg/device/device_core.hpp
 * @brief MFEM-managed single-device FP64 HDG local condensation baseline.
 *
 * The engine intentionally owns only local, uniformly-shaped batches.  It
 * does not select a formulation, assemble a global trace system, or perform
 * MPI work.  When MFEM has not selected a real CUDA or HIP backend, every
 * operation delegates to the accepted HostReferenceEngine.  That fallback is
 * part of the API contract rather than a numerically distinct code path.
 */

#include "numerics/hdg/core/host_reference.hpp"

#include <mfem.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dg::hdg::device
{
enum class ExecutionPath : unsigned char
{
  host_reference,
  mfem_device
};

/** MFEM backend facts used by the side-effect-free execution policy. */
struct BackendSelection
{
  bool configured = false;
  bool cuda_enabled = false;
  bool hip_enabled = false;
  bool debug_enabled = false;
  unsigned int visible_gpu_count = 0;
};

/**
 * A real CUDA or HIP backend is sufficient once MFEM has selected it.  The
 * visible-device count is deliberately not an eligibility constraint: MFEM
 * selects one device by id, and a machine may expose several.  DEBUG_DEVICE
 * is host-only instrumentation and must never be reported as GPU execution.
 */
constexpr bool IsGpuExecutionEligible(const BackendSelection selection) noexcept
{
  (void)selection.visible_gpu_count;
  return selection.configured && !selection.debug_enabled &&
         (selection.cuda_enabled || selection.hip_enabled);
}

/** Checked fixed-shape storage plan used before MFEM storage is established. */
struct PersistentStorageLayout
{
  std::size_t counts[12]{};
  std::size_t bytes = 0;
};

/** Pure overflow/capacity policy; does not allocate or query a device. */
bool MakePersistentStorageLayout(std::size_t interior_unknowns,
                                 std::size_t trace_unknowns,
                                 std::size_t batches,
                                 PersistentStorageLayout& layout) noexcept;

/** Monotonic evidence for owned persistent storage and explicit synchrony. */
struct TransferAccounting
{
  // Established MFEM vector storage.  The count is updated only after a
  // complete staged vector set has been committed.
  std::size_t persistent_bytes = 0;
  std::size_t allocation_events = 0;
  // Must remain unchanged for fixed-shape ApplySchur/Recover calls.
  std::size_t steady_state_allocation_events = 0;
  // Logical host/device synchronization requests, not a claim about physical
  // copies when MFEM uses managed or unified memory.
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
                        const core::CondensedBatchView& output);
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
                        std::size_t batches);
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
