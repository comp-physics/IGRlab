#pragma once

// Prepared authority and payload values shared by the CUDA facade and its
// out-of-line application adapter. No engine storage is exposed here.
#include <cstddef>
#include <cstdint>

namespace dg::hdg::runtime::cuda
{

class GenericEulerHdgCudaSession;

/**
 * Move-only authority for one staged native CUDA candidate.  The authority is
 * session-, epoch-, and export-object-specific.  It has no destructor
 * rollback: abandoning a live authority intentionally leaves its session
 * fail-closed instead of attempting a noncollective recovery.
 */
class GenericEulerCudaPreparedStep
{
public:
  GenericEulerCudaPreparedStep() noexcept = default;
  GenericEulerCudaPreparedStep(const GenericEulerCudaPreparedStep&) = delete;
  GenericEulerCudaPreparedStep& operator=(
      const GenericEulerCudaPreparedStep&) = delete;
  GenericEulerCudaPreparedStep(GenericEulerCudaPreparedStep&& other) noexcept
  {
    MoveFrom(other);
  }
  GenericEulerCudaPreparedStep& operator=(GenericEulerCudaPreparedStep&&) =
      delete;

private:
  friend class GenericEulerHdgCudaSession;

  void MoveFrom(GenericEulerCudaPreparedStep& other) noexcept
  {
    owner_ = other.owner_;
    epoch_ = other.epoch_;
    facade_incarnation_ = other.facade_incarnation_;
    native_export_ = other.native_export_;
    valid_ = other.valid_;
    other.valid_ = false;
  }

  void Clear() noexcept
  {
    owner_ = nullptr;
    epoch_ = 0;
    facade_incarnation_ = 0;
    native_export_ = nullptr;
    valid_ = false;
  }

  const GenericEulerHdgCudaSession* owner_ = nullptr;
  std::uint64_t epoch_ = 0;
  std::uint64_t facade_incarnation_ = 0;
  void* native_export_ = nullptr;
  bool valid_ = false;
};

/** Exact caller-owned destinations for one metadata-bound candidate payload. */
struct GenericEulerCudaMutableCheckpointPayloadV1
{
  double* interior = nullptr;
  std::size_t interior_count = 0;
  double* owned_true_trace = nullptr;
  std::size_t owned_true_trace_count = 0;
};

/** Immutable decoded payload view used only during a pristine restore. */
struct GenericEulerCudaCheckpointPayloadViewV1
{
  const double* interior = nullptr;
  std::size_t interior_count = 0;
  const double* owned_true_trace = nullptr;
  std::size_t owned_true_trace_count = 0;
};

/**
 * One bounded, collective classification for an immediately failed prepare.
 * This is neither retained attempt history nor a numerical-evidence export:
 * it exists only so the application can identify the last rejection if the
 * adaptive controller subsequently declares that rejection terminal.
 */
enum class GenericEulerCudaPrepareFailureClass : unsigned char
{
  not_failed,
  unavailable,
  zhang_shu_inadmissible_conservative_mean,
  finite_gate,
  admissibility_gate,
  trace_solve,
  recovery_gate,
  true_residual_gate,
  telemetry_gate,
  success_gate,
  collective_protocol,
  backend_or_contract,
  mixed_numerical_failure
};

} // namespace dg::hdg::runtime::cuda
