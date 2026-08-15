#pragma once

/**
 * @file numerics/hdg/runtime/cuda/generic_euler.cuh
 * @brief Narrow native-CUDA prepared-transaction facade for the W8A engine.
 *
 * This facade accepts an existing @ref CudaConfig with a caller-owned
 * periodic @ref mfem::ParMesh and an explicit borrowed byNODES L2 initial
 * field.  Optional static L2 source coefficients are copied at Initialize;
 * no caller field pointer is retained.  Physical interior exports use the
 * same explicit L2 ABI and never decode a checkpoint outside this engine.
 */

#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

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

/**
 * Header-only adapter over the existing persistent CUDA engine.  Its complete
 * lifecycle is PrepareStep -> staged output -> MarkOutputStaged ->
 * PrepareCommit -> CommitPrepared -> FinalizeCommit.  No second numerical
 * engine, allocator, kernel, or commit lifecycle is introduced here.  Every
 * member of MPI_COMM_WORLD must call the same public collective operation in
 * the same order; the facade converges mismatched operation tags and local
 * authority failures before entering an engine affine phase.
 */
class GenericEulerHdgCudaSession
{
public:
  using result_type = generic::HdgEulerSessionResult;
  using coordinates_type = generic::HdgEulerSessionCoordinates;
  // Retain the established fixed checkpoint DTO for false mode.  Borrowed
  // host sessions select the separate metadata-only DTO below so dynamic
  // storage is never represented as a fixed checkpoint payload.
  using native_export_type = cuda_w8a::CudaSerializedCheckpointV1;
  using prepared_output_type = cuda_w8a::CudaPreparedOutputMetadataV1;

  GenericEulerHdgCudaSession() = default;
  ~GenericEulerHdgCudaSession() = default;
  GenericEulerHdgCudaSession(const GenericEulerHdgCudaSession&) = delete;
  GenericEulerHdgCudaSession& operator=(const GenericEulerHdgCudaSession&) =
      delete;
  GenericEulerHdgCudaSession(GenericEulerHdgCudaSession&&) = delete;
  GenericEulerHdgCudaSession& operator=(GenericEulerHdgCudaSession&&) =
      delete;

  /** Borrowed CUDA sessions expose diagnostics for committed state only. */
  std::uint32_t capabilities() const noexcept
  {
    const std::uint32_t base = generic::HdgEulerSessionCapabilityBit(
        generic::HdgEulerSessionCapability::prepared_commit) |
        generic::HdgEulerSessionCapabilityBit(
            generic::HdgEulerSessionCapability::prepared_output) |
        generic::HdgEulerSessionCapabilityBit(
            generic::HdgEulerSessionCapability::generic_borrowed_inputs);
    return metadata_output_mode_ ? base |
        generic::HdgEulerSessionCapabilityBit(
            generic::HdgEulerSessionCapability::committed_cfl_diagnostics)
        : base;
  }

  /**
   * Admit only the already-supported borrowed periodic topology envelope.
   * Device setup remains in the existing engine; locally-invalid requests are
   * rejected before an engine is constructed.
   */
  result_type Initialize(const cuda_w8a::CudaConfig& config) noexcept
  {
    // This is deliberately the first collective for initialization: ranks
    // that call another facade operation must converge on the operation-tag
    // rejection before any config or topology collective is entered.
    const bool one_shot = !initialization_attempted_ &&
        session_collective_id_ == unassigned_session_collective_id;
    initialization_attempted_ = true;
    if (!AdmitOperation(operation_tag::initialize, one_shot))
    {
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    // Once every rank has agreed on Initialize, this admission is deliberately
    // evaluated everywhere, then gated again before allocation.
    const bool admitted = PreDeviceAdmission(config);
    if (!CollectiveAll(admitted))
    {
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    // The per-process ordinal is useful only after min/max proves every rank
    // selected the same next value.  It is a wrapper-owned admission key, not
    // caller export or token data, and must exist before any engine collective.
    if (!AssignSessionCollectiveId())
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    bool allocated = false;
    try
    {
      if (config.expected_rank_count == 1)
      {
        if (config.use_borrowed_host_inputs)
        {
          borrowed_rank1_ = std::make_unique<borrowed_rank1_engine>(config);
          allocated = borrowed_rank1_ != nullptr;
        }
        else
        {
          rank1_ = std::make_unique<rank1_engine>(config);
          allocated = rank1_ != nullptr;
        }
      }
      else
      {
        if (config.use_borrowed_host_inputs)
        {
          borrowed_rank2_ = std::make_unique<borrowed_rank2_engine>(config);
          allocated = borrowed_rank2_ != nullptr;
        }
        else
        {
          rank2_ = std::make_unique<rank2_engine>(config);
          allocated = rank2_ != nullptr;
        }
      }
    }
    catch (...)
    {
      allocated = false;
    }
    if (!CollectiveAll(allocated))
    {
      rank1_.reset();
      rank2_.reset();
      borrowed_rank1_.reset();
      borrowed_rank2_.reset();
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    // Every rank now owns the selected engine and enters its collective
    // initialization route together.
    initialized_ = Dispatch([&](auto& engine)
    { return engine.Initialize(setup_); });
    if (!CollectiveAll(initialized_))
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    coordinates_ = EngineCoordinates();
    metadata_output_mode_ = config.use_borrowed_host_inputs;
    if (metadata_output_mode_ && !RefreshCommittedDiagnostics())
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    return Result(generic::HdgEulerSessionStatus::initialized);
  }

  /** Stage the native candidate export; an explicit acknowledgement is next. */
  template <typename StagedOutput>
  result_type Prepare(const double dt, GenericEulerCudaPreparedStep& authority,
                      StagedOutput& native_export) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    // A destination authority is never an overwrite slot.  In particular, a
    // live authority from another session remains usable by its owner when
    // this session rejects the prepare collectively.
    const bool local_valid = ReadyForPrepare() && !authority.valid_ &&
        OutputMatchesMode<StagedOutput>() && std::isfinite(dt) && dt > 0.0 &&
        (!retry_ready_ || SameDouble(dt, retained_dt_));
    if (!AdmitOperation(operation_tag::prepare, local_valid))
      return Result(generic::HdgEulerSessionStatus::rejected);
    // Admission above deliberately carries only a validity bit; retain the
    // exact IEEE payload agreement as its own collective before any engine
    // affine phase so unequal finite proposals cannot silently proceed.
    if (!ExactDtConsensus(dt))
      return Result(generic::HdgEulerSessionStatus::rejected);

    cuda_w8a::CudaPrecommitSnapshot snapshot{};
    cuda_w8a::CudaStepEvidence evidence{};
    StagedOutput staged_export{};
    const bool prepared_locally = Dispatch([&](auto& engine)
    { return engine.PrepareStep(retry_ready_, dt, snapshot, evidence); });
    const bool prepared = CollectiveAll(prepared_locally);
    if (!prepared)
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    const bool staged_locally = Dispatch([&](auto& engine)
    {
      if constexpr (IsMetadataOutput<StagedOutput>())
        return engine.StagePreparedOutputMetadata(staged_export, evidence);
      else
        return engine.StagePreparedCheckpoint(staged_export, evidence);
    });
    const bool locally_staged = staged_locally &&
        snapshot.numerical_candidate_ready &&
        staged_export.candidate_frozen && staged_export.checkpoint_staged &&
        !staged_export.output_staged && !staged_export.successful_commit;
    if (!CollectiveAll(locally_staged))
    {
      // Output staging can leave a live affine staging token when a
      // collective staging operation fails.  Consume it through the engine's
      // existing abort route; never discard it or invent a second rollback.
      if (prepared)
      {
        (void)Dispatch([&](auto& engine)
        { return engine.AbortPrepared(staged_export, FailureReason::backend_failure,
                                      evidence); });
      }
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    native_export = staged_export;
    authority.Clear();
    ++epoch_;
    retained_dt_ = dt;
    retry_ready_ = false;
    prepared_active_ = true;
    output_acknowledged_ = false;
    active_native_export_ = &native_export;
    active_facade_incarnation_ = native_export.facade_incarnation;
    authority.owner_ = this;
    authority.epoch_ = epoch_;
    authority.facade_incarnation_ = active_facade_incarnation_;
    authority.native_export_ = &native_export;
    authority.valid_ = true;
    return Result(generic::HdgEulerSessionStatus::prepared);
  }

  template <typename StagedOutput>
  result_type Prepare(GenericEulerCudaPreparedStep& authority,
                      StagedOutput& native_export) noexcept
  { return Prepare(cuda_w8a::fixed_dt, authority, native_export); }

  /** Export the initial or later committed physical interior to byNODES L2. */
  bool ExportCommittedInterior(
      cuda_w8a::CudaMutableHostL2Output destination) noexcept
  {
    const bool local_valid = initialized_ && !failed_ && !prepared_active_ &&
        destination.coefficients != nullptr &&
        destination.components == cuda_w8a::conserved_components;
    if (!AdmitOperation(operation_tag::export_committed, local_valid))
      return false;
    return Dispatch([&](auto& engine)
    { return engine.ExportCommittedInterior(destination); });
  }

  /**
   * Export a physical candidate only while its affine authority is live.
   * This does not acknowledge or publish it: Commit still requires the
   * existing AcknowledgePreparedOutput call after the caller accepts bytes.
   */
  template <typename StagedOutput>
  bool ExportPreparedInterior(
      const GenericEulerCudaPreparedStep& authority,
      const StagedOutput& native_export,
      cuda_w8a::CudaMutableHostL2Output destination) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    const bool local_valid = OutputMatchesMode<StagedOutput>() &&
        Matches(authority, native_export) &&
        destination.coefficients != nullptr &&
        destination.components == cuda_w8a::conserved_components;
    if (!AdmitOperation(operation_tag::export_prepared, local_valid))
      return false;
    return Dispatch([&](auto& engine)
    { return engine.ExportPreparedInterior(destination); });
  }

  /** The caller attests that the staged native export is ready for publication. */
  template <typename StagedOutput>
  bool AcknowledgePreparedOutput(
      const GenericEulerCudaPreparedStep& authority,
      StagedOutput& native_export) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    const bool local_valid = OutputMatchesMode<StagedOutput>() &&
        Matches(authority, native_export) &&
        !output_acknowledged_;
    if (!AdmitOperation(operation_tag::acknowledge_output, local_valid))
      return false;
    const bool marked_locally = Dispatch([&](auto& engine)
    { return engine.MarkOutputStaged(native_export); });
    output_acknowledged_ = CollectiveAll(marked_locally);
    return output_acknowledged_;
  }

  /** Consume the authority and publish exactly once through the existing engine. */
  template <typename StagedOutput>
  result_type Commit(GenericEulerCudaPreparedStep&& authority,
                     StagedOutput& native_export) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    const bool local_valid = OutputMatchesMode<StagedOutput>() &&
        Matches(authority, native_export) &&
        output_acknowledged_;
    if (!AdmitOperation(operation_tag::commit, local_valid))
      return Result(generic::HdgEulerSessionStatus::rejected);
    cuda_w8a::CudaStepEvidence evidence{};
    const bool prepared = Dispatch([&](auto& engine)
    { return engine.PrepareCommit(native_export, evidence); });
    const bool prepared_live = prepared && EngineHasStatus(RunStatus::attempt_active);
    if (!CollectiveAll(prepared_live))
    {
      ConsumeAuthority(authority);
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    const bool provisional = Dispatch([&](auto& engine)
    { return engine.CommitPrepared(native_export, evidence); });
    const bool provisional_live = provisional &&
        EngineHasStatus(RunStatus::attempt_active);
    if (!CollectiveAll(provisional_live))
    {
      // CommitPrepared consumes or aborts its matching prepared token on every
      // documented failure path; no prepared authority remains to abort.
      ConsumeAuthority(authority);
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    const bool committed = Dispatch([&](auto& engine)
    { return engine.FinalizeCommit(native_export, evidence); });
    const bool finalized = committed && native_export.successful_commit &&
        EngineHasStatus(RunStatus::committed);
    ConsumeAuthority(authority);
    if (!CollectiveAll(finalized))
    {
      // FinalizeCommit either commits or invokes its existing RevertCommit
      // route for a reversible provisional publication; it leaves no facade
      // authority to replay.
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    coordinates_ = EngineCoordinates();
    // The engine reduced and cached candidate diagnostics before consuming its
    // final publication authority.  This is POD copying only.
    if (metadata_output_mode_) InstallCommittedDiagnostics();
    return Result(generic::HdgEulerSessionStatus::committed);
  }

  /** Consume the authority through the existing prepared-abort path. */
  template <typename StagedOutput>
  result_type Abort(GenericEulerCudaPreparedStep&& authority,
                    StagedOutput& native_export) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    if (!AdmitOperation(operation_tag::abort,
                        OutputMatchesMode<StagedOutput>() &&
                        Matches(authority, native_export)))
      return Result(generic::HdgEulerSessionStatus::rejected);
    cuda_w8a::CudaStepEvidence evidence{};
    const bool aborted = Dispatch([&](auto& engine)
    { return engine.AbortPrepared(native_export, FailureReason::backend_failure,
                                  evidence); });
    const bool rollback_retry = aborted && evidence.retry_available &&
        evidence.coordinates_preserved && evidence.committed_bytes_preserved &&
        !native_export.candidate_frozen && !native_export.checkpoint_staged &&
        !native_export.output_staged && !native_export.successful_commit;
    ConsumeAuthority(authority);
    if (!CollectiveAll(rollback_retry))
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    retry_ready_ = true;
    coordinates_ = EngineCoordinates();
    return Result(generic::HdgEulerSessionStatus::aborted);
  }

  coordinates_type coordinates() const noexcept { return coordinates_; }

  generic::HdgEulerCommittedDiagnostics diagnostics() const noexcept
  { return diagnostics_; }

private:
  enum class operation_tag : int
  {
    initialize = 1,
    prepare = 2,
    acknowledge_output = 3,
    commit = 4,
    abort = 5,
    export_committed = 6,
    export_prepared = 7
  };

  using rank1_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      CudaRank1OneGpuRuntimeSelection>;
  using rank2_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      CudaRank2OneGpuHostStagedRuntimeSelection>;
  using borrowed_rank1_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      BorrowedCudaRank1OneGpuRuntimeSelection>;
  using borrowed_rank2_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>;

  static coordinates_type ToCoordinates(
      const StateCoordinates& value) noexcept
  {
    return {value.time, value.dt, value.step, value.generation};
  }

  static bool SameDouble(const double left, const double right) noexcept
  { return std::memcmp(&left, &right, sizeof(double)) == 0; }

  static bool ExactDtConsensus(const double dt) noexcept
  {
    std::uint64_t bits = 0, minimum = 0, maximum = 0;
    std::memcpy(&bits, &dt, sizeof(bits));
    const int minimum_status = MPI_Allreduce(&bits, &minimum, 1,
        MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    const int maximum_status = MPI_Allreduce(&bits, &maximum, 1,
        MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    return minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS &&
        minimum == maximum;
  }

  bool RefreshCommittedDiagnostics() noexcept
  {
    double maximum_wave_speed = 0.0;
    double backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
    const bool measured = Dispatch([&](auto& engine)
    { return engine.CachedBorrowedEulerDiagnostics(maximum_wave_speed,
                                                    backend_stable_dt); });
    if (!measured) return false;
    diagnostics_ = {true, coordinates_, coordinates_.generation,
                    maximum_wave_speed, backend_stable_dt};
    return generic::IsValidHdgEulerCommittedDiagnostics(diagnostics_,
                                                         capabilities());
  }

  void InstallCommittedDiagnostics() noexcept
  {
    // FinalizeCommit can reach this point only after its candidate scan and
    // successful slot swap.  Do not introduce any collective or failure path.
    double maximum_wave_speed = 0.0;
    double backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
    (void)Dispatch([&](auto& engine)
    { return engine.CachedBorrowedEulerDiagnostics(maximum_wave_speed,
                                                    backend_stable_dt); });
    diagnostics_ = {true, coordinates_, coordinates_.generation,
                    maximum_wave_speed, backend_stable_dt};
  }

  result_type Result(const generic::HdgEulerSessionStatus status) const noexcept
  { return {status, coordinates_}; }

  static constexpr std::uint64_t unassigned_session_collective_id = 0;

  template <typename StagedOutput>
  static constexpr bool IsMetadataOutput() noexcept
  {
    return std::is_same<StagedOutput, prepared_output_type>::value;
  }

  template <typename StagedOutput>
  static constexpr bool IsSupportedOutput() noexcept
  {
    return IsMetadataOutput<StagedOutput>() ||
        std::is_same<StagedOutput, native_export_type>::value;
  }

  template <typename StagedOutput>
  bool OutputMatchesMode() const noexcept
  {
    return metadata_output_mode_ == IsMetadataOutput<StagedOutput>();
  }

  bool ReadyForPrepare() const noexcept
  {
    return initialized_ && !failed_ && !prepared_active_ &&
        active_native_export_ == nullptr && active_facade_incarnation_ == 0;
  }

  template <typename StagedOutput>
  bool Matches(const GenericEulerCudaPreparedStep& authority,
               const StagedOutput& native_export) const noexcept
  {
    return initialized_ && !failed_ && prepared_active_ && authority.valid_ &&
        authority.owner_ == this && authority.epoch_ == epoch_ &&
        authority.native_export_ == active_native_export_ &&
        active_native_export_ == static_cast<const void*>(&native_export) &&
        authority.facade_incarnation_ != 0 &&
        active_facade_incarnation_ != 0 &&
        authority.facade_incarnation_ == active_facade_incarnation_ &&
        native_export.facade_incarnation == active_facade_incarnation_;
  }

  bool AssignSessionCollectiveId() noexcept
  {
    const bool local_available = session_collective_id_ ==
            unassigned_session_collective_id &&
        next_session_collective_id_ != std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t local_candidate = local_available
        ? next_session_collective_id_ + 1 : unassigned_session_collective_id;
    std::uint64_t minimum = local_candidate;
    std::uint64_t maximum = local_candidate;
    const bool minimum_ready = MPI_Allreduce(
        &local_candidate, &minimum, 1, MPI_UINT64_T, MPI_MIN,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        &local_candidate, &maximum, 1, MPI_UINT64_T, MPI_MAX,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    if (!minimum_ready || !maximum_ready || local_candidate ==
            unassigned_session_collective_id ||
        minimum != maximum || minimum != local_candidate)
      return false;
    next_session_collective_id_ = local_candidate;
    session_collective_id_ = local_candidate;
    return true;
  }

  bool AdmitOperation(const operation_tag tag, const bool local_valid) const
      noexcept
  {
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS || !mpi_initialized ||
        MPI_Finalized(&mpi_finalized) != MPI_SUCCESS || mpi_finalized)
      return false;
    const std::uint64_t local[3] = {
        session_collective_id_, static_cast<std::uint64_t>(tag),
        local_valid ? 1U : 0U};
    std::uint64_t minimum[3] = {local[0], local[1], local[2]};
    std::uint64_t maximum[3] = {local[0], local[1], local[2]};
    const bool minimum_ready = MPI_Allreduce(
        local, minimum, 3, MPI_UINT64_T, MPI_MIN,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local, maximum, 3, MPI_UINT64_T, MPI_MAX,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool unassigned_initialize =
        tag == operation_tag::initialize &&
        minimum[0] == unassigned_session_collective_id &&
        maximum[0] == unassigned_session_collective_id;
    const bool assigned_operation = tag != operation_tag::initialize &&
        minimum[0] == maximum[0] &&
        minimum[0] != unassigned_session_collective_id;
    return minimum_ready && maximum_ready &&
        (unassigned_initialize || assigned_operation) &&
        minimum[1] == maximum[1] &&
        minimum[1] == static_cast<std::uint64_t>(tag) &&
        minimum[2] == maximum[2] && minimum[2] == 1;
  }

  static bool CollectiveAll(const bool local) noexcept
  {
    const int input = local ? 1 : 0;
    int output = 0;
    return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
  }

  bool EngineHasStatus(const RunStatus expected) const noexcept
  {
    return (rank1_ && rank1_->contract() &&
            rank1_->contract()->status() == expected) ||
        (rank2_ && rank2_->contract() &&
         rank2_->contract()->status() == expected) ||
        (borrowed_rank1_ && borrowed_rank1_->contract() &&
         borrowed_rank1_->contract()->status() == expected) ||
        (borrowed_rank2_ && borrowed_rank2_->contract() &&
         borrowed_rank2_->contract()->status() == expected);
  }

  void ConsumeAuthority(GenericEulerCudaPreparedStep& authority) noexcept
  {
    authority.Clear();
    prepared_active_ = false;
    output_acknowledged_ = false;
    active_native_export_ = nullptr;
    active_facade_incarnation_ = 0;
  }

  using config_descriptor = std::array<unsigned char, 256>;

  template <typename Value>
  static void AppendConfigValue(config_descriptor& descriptor,
                                std::size_t& cursor, bool& complete,
                                const Value& value) noexcept
  {
    static_assert(std::is_trivially_copyable<Value>::value,
                  "CUDA admission descriptor values must be stable");
    if (!complete || sizeof(Value) > descriptor.size() - cursor)
    {
      complete = false;
      return;
    }
    std::memcpy(descriptor.data() + cursor, &value, sizeof(Value));
    cursor += sizeof(Value);
  }

  static bool ConfigConsensus(const cuda_w8a::CudaConfig& config) noexcept
  {
    config_descriptor local{};
    std::size_t cursor = 0;
    bool complete = true;
    const std::uint64_t experiment_size = config.experiment.size();
    const std::uint8_t experiment_valid = config.experiment.valid() ? 1U : 0U;
    const auto append_bool = [&](const bool value)
    {
      const std::uint8_t bit = value ? 1U : 0U;
      AppendConfigValue(local, cursor, complete, bit);
    };
    AppendConfigValue(local, cursor, complete, experiment_size);
    AppendConfigValue(local, cursor, complete, experiment_valid);
    if (complete && experiment_size <= local.size() - cursor)
    {
      std::memcpy(local.data() + cursor, config.experiment.data(),
                  static_cast<std::size_t>(experiment_size));
      cursor += static_cast<std::size_t>(experiment_size);
    }
    else
    {
      complete = false;
    }
    AppendConfigValue(local, cursor, complete, config.order);
    AppendConfigValue(local, cursor, complete, config.mesh_n);
    AppendConfigValue(local, cursor, complete, config.dt);
    AppendConfigValue(local, cursor, complete, config.requested_device);
    AppendConfigValue(local, cursor, complete, config.expected_rank_count);
    append_bool(config.require_host_staged_rank2);
    append_bool(config.use_borrowed_topology);
    // Preserve the frozen legacy descriptor byte-for-byte.  The data-mode
    // branch carries only rank-common ABI facts: local L2 counts are expected
    // to vary with the borrowed partition and are validated locally later.
    if (config.use_borrowed_host_inputs)
    {
      append_bool(true);
      AppendConfigValue(local, cursor, complete,
                        config.borrowed_initial.components);
      AppendConfigValue(local, cursor, complete,
                        config.borrowed_source.components);
      append_bool(config.borrowed_source.coefficients != nullptr);
    }
    append_bool(config.capture_oracle);
    append_bool(config.force_private_scientific_gate_failure);
    AppendConfigValue(local, cursor, complete, config.injection.phase);
    AppendConfigValue(local, cursor, complete, config.injection.target_step);
    AppendConfigValue(local, cursor, complete, config.injection.source_rank);
    append_bool(config.injection.enabled);
    AppendConfigValue(local, cursor, complete, config.injection.token_fault);

    config_descriptor minimum = local;
    config_descriptor maximum = local;
    const int complete_bit = complete ? 1 : 0;
    int all_complete = 0;
    const bool complete_census = MPI_Allreduce(
        &complete_bit, &all_complete, 1, MPI_INT, MPI_MIN,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool minimum_census = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool maximum_census = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS;
    return complete_census && minimum_census && maximum_census &&
        all_complete == 1 && minimum == maximum;
  }

  static bool PhysicallyAdmissibleBorrowedInitial(
      const cuda_w8a::CudaBorrowedHostL2Input& field) noexcept
  {
    if (field.coefficients == nullptr || field.components !=
        cuda_w8a::conserved_components) return false;
    constexpr double gamma = 1.4;
    for (std::size_t scalar = 0; scalar < field.scalar_count; ++scalar)
    {
      const double rho = field.coefficients[scalar];
      const double mx = field.coefficients[field.scalar_count + scalar];
      const double my = field.coefficients[2 * field.scalar_count + scalar];
      const double energy = field.coefficients[3 * field.scalar_count + scalar];
      if (!std::isfinite(rho) || !std::isfinite(mx) || !std::isfinite(my) ||
          !std::isfinite(energy) || !(rho > 0.0)) return false;
      const double pressure = (gamma - 1.0) *
          (energy - 0.5 * (mx * mx + my * my) / rho);
      const double sound_squared = gamma * pressure / rho;
      if (!(pressure > 0.0) || !(sound_squared > 0.0) ||
          !std::isfinite(sound_squared) ||
          !std::isfinite(std::sqrt(sound_squared))) return false;
    }
    return true;
  }

  bool PreDeviceAdmission(const cuda_w8a::CudaConfig& config) const noexcept
  {
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS || !mpi_initialized ||
        MPI_Finalized(&mpi_finalized) != MPI_SUCCESS || mpi_finalized)
      return false;
    const bool config_consensus = ConfigConsensus(config);
    const bool empty_borrowed_fields =
        config.borrowed_initial.coefficients == nullptr &&
        config.borrowed_initial.scalar_count == 0 &&
        config.borrowed_initial.components == cuda_w8a::conserved_components &&
        config.borrowed_source.coefficients == nullptr &&
        config.borrowed_source.scalar_count == 0 &&
        config.borrowed_source.components == cuda_w8a::conserved_components;
    const bool borrowed_data_abi = config.use_borrowed_host_inputs
        ? config.borrowed_initial.coefficients != nullptr &&
            config.borrowed_initial.scalar_count != 0 &&
            config.borrowed_initial.components == cuda_w8a::conserved_components &&
            ((config.borrowed_source.coefficients == nullptr &&
              config.borrowed_source.scalar_count == 0 &&
              config.borrowed_source.components == cuda_w8a::conserved_components) ||
             (config.borrowed_source.coefficients != nullptr &&
              config.borrowed_source.scalar_count != 0 &&
              config.borrowed_source.components == cuda_w8a::conserved_components))
        : empty_borrowed_fields;
    int ranks = 0;
    int borrowed_world = MPI_UNEQUAL;
    int borrowed_mesh = MPI_UNEQUAL;
    bool local = config.use_borrowed_topology && config.borrowed_mesh != nullptr &&
        config.borrowed_communicator != MPI_COMM_NULL &&
        borrowed_data_abi &&
        MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS &&
        (ranks == 1 || ranks == 2) && ranks == config.expected_rank_count &&
        config.requested_device == 0 &&
        (ranks == 1 || config.require_host_staged_rank2) &&
        config.experiment == StableIdentity::Copy(cuda_w8a::experiment_id) &&
        !config.capture_oracle && !config.force_private_scientific_gate_failure &&
        !config.injection.enabled &&
        config.injection.phase ==
            cuda_w8a::CudaFailurePhase::after_recovery_before_commit &&
        config.injection.target_step == 2 && config.injection.source_rank == 0 &&
        config.injection.token_fault == cuda_w8a::CudaTokenFault::none &&
        std::isfinite(config.dt) && config.dt > 0.0 &&
        (config.use_borrowed_host_inputs ||
         std::memcmp(&config.dt, &cuda_w8a::fixed_dt, sizeof(double)) == 0) &&
        (config.order == cuda_w8a::smoke_polynomial_order ||
         config.order == cuda_w8a::functional_polynomial_order) &&
        config.mesh_n >= 0;
    long long local_elements = 0;
    long long local_face_measure = 0;
    long long global_elements = 0;
    long long global_face_measure = 0;
    if (local)
    {
      const MPI_Comm mesh_communicator = config.borrowed_mesh->GetComm();
      local = MPI_Comm_compare(config.borrowed_communicator, MPI_COMM_WORLD,
                               &borrowed_world) == MPI_SUCCESS &&
          (borrowed_world == MPI_IDENT || borrowed_world == MPI_CONGRUENT) &&
          MPI_Comm_compare(config.borrowed_communicator, mesh_communicator,
                           &borrowed_mesh) == MPI_SUCCESS &&
          (borrowed_mesh == MPI_IDENT || borrowed_mesh == MPI_CONGRUENT) &&
          !config.borrowed_mesh->Nonconforming() &&
          config.borrowed_mesh->Dimension() == cuda_w8a::dimension &&
          config.borrowed_mesh->SpaceDimension() == cuda_w8a::dimension &&
          config.borrowed_mesh->GetNE() > 0;
      local_elements = config.borrowed_mesh->GetNE();
      for (int element = 0; local && element < config.borrowed_mesh->GetNE();
           ++element)
        local = config.borrowed_mesh->GetElementBaseGeometry(element) ==
            mfem::Geometry::SQUARE;
      if (local)
      {
        const long long faces = config.borrowed_mesh->GetNumFaces();
        const long long shared = config.borrowed_mesh->GetNSharedFaces();
        local = faces >= 0 && shared >= 0 && shared <= faces &&
            faces <= std::numeric_limits<long long>::max() / 2;
        if (local) local_face_measure = 2 * faces - shared;
      }
      if (local && config.use_borrowed_host_inputs)
      {
        const std::size_t expected_scalar_count = static_cast<std::size_t>(
            config.borrowed_mesh->GetNE()) * static_cast<std::size_t>(
            (config.order + 1) * (config.order + 1));
        const bool source_empty = config.borrowed_source.coefficients == nullptr &&
            config.borrowed_source.scalar_count == 0;
        local = config.borrowed_initial.scalar_count == expected_scalar_count &&
            (source_empty || config.borrowed_source.scalar_count ==
                expected_scalar_count);
        // Count equality is established before dereferencing either caller
        // field, so wrong-count admission fails without speculative reads.
        for (std::size_t i = 0; local && i < expected_scalar_count *
                 static_cast<std::size_t>(cuda_w8a::conserved_components); ++i)
          local = std::isfinite(config.borrowed_initial.coefficients[i]) &&
              (source_empty ||
               std::isfinite(config.borrowed_source.coefficients[i]));
        local = local && PhysicallyAdmissibleBorrowedInitial(
            config.borrowed_initial);
      }
    }
    const int local_bit = local ? 1 : 0;
    int all_local = 0;
    const bool local_census = MPI_Allreduce(&local_bit, &all_local, 1,
                                             MPI_INT, MPI_MIN,
                                             MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool element_census = MPI_Allreduce(&local_elements, &global_elements,
        1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool face_census = MPI_Allreduce(&local_face_measure,
        &global_face_measure, 1, MPI_LONG_LONG_INT, MPI_SUM,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool periodic_measure_safe = global_elements >= 0 &&
        global_elements <= std::numeric_limits<long long>::max() / 4;
    return config_consensus && local_census && element_census && face_census &&
        all_local == 1 && periodic_measure_safe &&
        global_face_measure == 4 * global_elements;
  }

  template <typename Callback>
  bool Dispatch(Callback&& callback) noexcept
  {
    if (rank1_) return callback(*rank1_);
    if (rank2_) return callback(*rank2_);
    if (borrowed_rank1_) return callback(*borrowed_rank1_);
    if (borrowed_rank2_) return callback(*borrowed_rank2_);
    return false;
  }

  coordinates_type EngineCoordinates() const noexcept
  {
    if (rank1_ && rank1_->contract()) return ToCoordinates(rank1_->contract()->coordinates());
    if (rank2_ && rank2_->contract()) return ToCoordinates(rank2_->contract()->coordinates());
    if (borrowed_rank1_ && borrowed_rank1_->contract())
      return ToCoordinates(borrowed_rank1_->contract()->coordinates());
    if (borrowed_rank2_ && borrowed_rank2_->contract())
      return ToCoordinates(borrowed_rank2_->contract()->coordinates());
    return coordinates_;
  }

  std::unique_ptr<rank1_engine> rank1_;
  std::unique_ptr<rank2_engine> rank2_;
  std::unique_ptr<borrowed_rank1_engine> borrowed_rank1_;
  std::unique_ptr<borrowed_rank2_engine> borrowed_rank2_;
  cuda_w8a::CudaRunEvidence setup_{};
  coordinates_type coordinates_{};
  std::uint64_t epoch_ = 0;
  bool initialization_attempted_ = false;
  bool initialized_ = false;
  bool prepared_active_ = false;
  bool output_acknowledged_ = false;
  bool failed_ = false;
  bool retry_ready_ = false;
  double retained_dt_ = cuda_w8a::fixed_dt;
  generic::HdgEulerCommittedDiagnostics diagnostics_{};
  std::uint64_t session_collective_id_ = unassigned_session_collective_id;
  const void* active_native_export_ = nullptr;
  std::uint64_t active_facade_incarnation_ = 0;
  bool metadata_output_mode_ = false;
  inline static std::uint64_t next_session_collective_id_ =
      unassigned_session_collective_id;
};

}  // namespace dg::hdg::runtime::cuda
