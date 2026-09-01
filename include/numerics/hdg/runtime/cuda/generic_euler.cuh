#pragma once

/**
 * @file numerics/hdg/runtime/cuda/generic_euler.cuh
 * @brief Narrow native-CUDA prepared-transaction facade for the W8A engine.
 *
 * This facade accepts an existing @ref CudaConfig with a caller-owned
 * caller-owned conforming-quad @ref mfem::ParMesh and an explicit borrowed
 * byNODES L2 initial field. Optional static L2 source coefficients and the
 * admitted static physical-boundary table are copied at Initialize; no caller
 * field or policy pointer is retained. Physical interior exports use the same
 * explicit L2 ABI and never decode a checkpoint outside this engine.
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
#include <vector>

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

/**
 * Header-only adapter over the existing persistent CUDA engine.  Its complete
 * lifecycle is PrepareStep -> staged output -> MarkOutputStaged ->
 * PrepareCommit -> CommitPrepared -> FinalizeCommit.  No second numerical
 * engine, allocator, kernel, or commit lifecycle is introduced here.  Every
 * member of the admitted request/mesh communicator must call the same public
 * collective operation in the same order; the facade converges mismatched
 * operation tags and local authority failures before entering an engine
 * affine phase.
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
    const std::uint32_t embedded_assessment =
        RodasAssessmentRequired()
        ? generic::HdgEulerSessionCapabilityBit(
              generic::HdgEulerSessionCapability::
                  prepared_embedded_assessment)
        : 0U;
    return embedded_assessment | (metadata_output_mode_ ? base |
        generic::HdgEulerSessionCapabilityBit(
            generic::HdgEulerSessionCapability::committed_cfl_diagnostics)
        : base);
  }

  /**
   * Lower the shared backend-neutral request into the existing engine.  The
   * temporary value-only device policy table is copied synchronously by the
   * CudaConfig Initialize route; no request field or policy pointer is kept.
   */
  result_type Initialize(const generic::EulerHdgRequest& request) noexcept
  {
    return InitializeRequest(
        request, cuda_w8a::CudaTracePolicyKind::engine_default);
  }

private:
  result_type InitializeRequest(
      const generic::EulerHdgRequest& request,
      const cuda_w8a::CudaTracePolicyKind trace_policy) noexcept
  {
    communicator_ = request.communicator;
    const bool one_shot = !initialization_attempted_ &&
        session_collective_id_ == unassigned_session_collective_id;
    initialization_attempted_ = true;
    if (!AdmitOperation(operation_tag::initialize, one_shot))
      return Result(generic::HdgEulerSessionStatus::rejected);
    if (generic::ValidateEulerHdgRequest(request) !=
        generic::ContractStatus::accepted)
      return Result(generic::HdgEulerSessionStatus::rejected);

    cuda_w8a::CudaConfig config{};
    std::vector<cuda_w8a::CudaStaticBoundaryPolicy> device_policies;
    bool lowered = request.mesh != nullptr &&
        request.backend.kind == generic::BackendKind::cuda &&
        request.backend.device_ordinal >= 0 &&
        generic::IsKnownCudaEngineKind(request.backend.cuda_engine) &&
        generic::HasCapabilities(
            request.capabilities,
            generic::RequiredCapabilities(request.backend)) &&
        generic::ValidEulerHdgNumericalPolicy(request.numerical_policy) &&
        !generic::IsAbsentEulerBoundaryPolicyTable(
            request.boundary.policies) &&
        request.boundary.exterior_state != nullptr &&
        request.initial_state.components == cuda_w8a::conserved_components &&
        request.initial_state.scalar_count %
            static_cast<std::size_t>(cuda_w8a::conserved_components) == 0 &&
        request.initial_true_trace.components ==
            cuda_w8a::conserved_components &&
        request.initial_true_trace.scalar_count %
            static_cast<std::size_t>(cuda_w8a::conserved_components) == 0 &&
        request.source.components == cuda_w8a::conserved_components &&
        request.source.scalar_count %
            static_cast<std::size_t>(cuda_w8a::conserved_components) == 0;
    int ranks = 0;
    lowered = lowered &&
        MPI_Comm_size(communicator_, &ranks) == MPI_SUCCESS &&
        (ranks == 1 || ranks == 2);
    int boundary_attributes = 0;
    int retained_boundary_attributes = 0;
    std::vector<unsigned char> physical_boundary_attribute_mask;
    const bool boundary_attributes_ready =
        generic::GlobalPhysicalBoundaryAttributeCount(
            *request.mesh, communicator_, boundary_attributes,
            retained_boundary_attributes,
            physical_boundary_attribute_mask);
    lowered = lowered && boundary_attributes_ready;
    lowered = lowered && generic::IsValidEulerBoundaryPolicyTable(
        request.boundary.policies, cuda_w8a::dimension,
        boundary_attributes, retained_boundary_attributes,
        physical_boundary_attribute_mask) &&
        generic::DoubleBits(request.boundary.policies.gamma) ==
            generic::DoubleBits(1.4) &&
        request.boundary.policies.count <=
            cuda_w8a::maximum_static_boundary_policies;
    if (lowered)
    {
      try
      {
        device_policies.reserve(request.boundary.policies.count);
        for (std::size_t index = 0;
             index < request.boundary.policies.count; ++index)
        {
          const generic::EulerBoundaryPolicy& source =
              request.boundary.policies.policies[index];
          if (source.kind ==
              generic::EulerBoundaryPolicyKind::periodic_topology)
            continue;
          cuda_w8a::CudaStaticBoundaryPolicy destination{};
          destination.attribute = static_cast<int>(index) + 1;
          destination.reference_state = source.reference_state;
          destination.pressure_out = source.pressure_out;
          switch (source.kind)
          {
          case generic::EulerBoundaryPolicyKind::supersonic_inlet:
            destination.kind = cuda_w8a::CudaStaticBoundaryPolicyKind::
                supersonic_inlet;
            break;
          case generic::EulerBoundaryPolicyKind::subsonic_inlet:
            destination.kind = cuda_w8a::CudaStaticBoundaryPolicyKind::
                subsonic_inlet;
            break;
          case generic::EulerBoundaryPolicyKind::supersonic_outlet:
            destination.kind = cuda_w8a::CudaStaticBoundaryPolicyKind::
                supersonic_outlet;
            break;
          case generic::EulerBoundaryPolicyKind::subsonic_outlet_pressure:
            destination.kind = cuda_w8a::CudaStaticBoundaryPolicyKind::
                subsonic_outlet_pressure;
            break;
          case generic::EulerBoundaryPolicyKind::far_field:
            destination.kind =
                cuda_w8a::CudaStaticBoundaryPolicyKind::far_field;
            break;
          case generic::EulerBoundaryPolicyKind::nonreflecting_outlet:
            destination.kind = cuda_w8a::CudaStaticBoundaryPolicyKind::
                nonreflecting_outlet;
            break;
          case generic::EulerBoundaryPolicyKind::slip_wall:
            destination.kind =
                cuda_w8a::CudaStaticBoundaryPolicyKind::slip_wall;
            break;
          case generic::EulerBoundaryPolicyKind::periodic_topology:
            break;
          }
          device_policies.push_back(destination);
        }
      }
      catch (...) { lowered = false; }
    }

    config.order = request.polynomial_order;
    config.mesh_n = 0;
    config.dt = cuda_w8a::fixed_dt;
    config.requested_device = lowered ? request.backend.device_ordinal : -1;
    config.cuda_engine = request.backend.cuda_engine;
    config.expected_rank_count = ranks;
    config.require_host_staged_rank2 = ranks == 2;
    config.use_borrowed_topology = true;
    config.borrowed_mesh = const_cast<mfem::ParMesh*>(request.mesh);
    config.borrowed_communicator = request.communicator;
    config.use_borrowed_host_inputs = true;
    config.borrowed_initial = {
        request.initial_state.data,
        request.initial_state.scalar_count /
            static_cast<std::size_t>(cuda_w8a::conserved_components),
        request.initial_state.components};
    config.borrowed_initial_true_trace = {
        request.initial_true_trace.data,
        request.initial_true_trace.scalar_count /
            static_cast<std::size_t>(cuda_w8a::conserved_components),
        request.initial_true_trace.components};
    config.borrowed_source = {
        request.source.data,
        request.source.scalar_count /
            static_cast<std::size_t>(cuda_w8a::conserved_components),
        request.source.components};
    if (!device_policies.empty())
      config.borrowed_global_topology = StableIdentity::Copy(
          "generic.euler.request.topology.v1");
    config.borrowed_boundary_policies = {
        device_policies.empty() ? nullptr : device_policies.data(),
        device_policies.size(), request.boundary.policies.gamma};
    config.borrowed_boundary = request.boundary;
    config.use_generic_numerical_policy = true;
    config.numerical_policy = request.numerical_policy;
    config.time_treatment = request.time_treatment;
    const cuda_w8a::CudaTracePolicyKind admitted_trace_policy =
        trace_policy == cuda_w8a::CudaTracePolicyKind::engine_default
        ? cuda_w8a::DefaultCudaTracePolicyRequest(
              request.backend.cuda_engine,
              request.numerical_policy.stabilization.kind ==
                  generic::RegularizationKind::igr,
              ranks, request.time_treatment)
        : trace_policy;
    return InitializeAdmitted(config, admitted_trace_policy);
  }

public:

  /**
   * Admit the borrowed conforming-quad topology plus its complete static
   * physical-boundary policy coverage.
   * Device setup remains in the existing engine; locally-invalid requests are
   * rejected before an engine is constructed.
   */
  result_type Initialize(const cuda_w8a::CudaConfig& config) noexcept
  {
    communicator_ = cuda_w8a::CudaCollectiveCommunicator(config);
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
    return InitializeAdmitted(
        config, cuda_w8a::CudaTracePolicyKind::engine_default);
  }

private:
  result_type InitializeAdmitted(
      const cuda_w8a::CudaConfig& config,
      const cuda_w8a::CudaTracePolicyKind requested_trace_policy) noexcept
  {
    const bool igr_active = config.use_generic_numerical_policy &&
        config.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr;
    const cuda_w8a::CudaTracePolicyDescriptor trace_policy =
        cuda_w8a::ResolveCudaTracePolicy(
            config.cuda_engine, igr_active, config.expected_rank_count,
            config.order, requested_trace_policy);
    // Once every rank has agreed on Initialize, this admission is deliberately
    // evaluated everywhere, then gated again before allocation.
    const bool admitted = PreDeviceAdmission(config, trace_policy);
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
          if (config.time_treatment == generic::EulerHdgTimeTreatment::
                       partitioned_backward_euler)
          {
            partitioned_backward_euler_borrowed_rank1_.reset(
                new partitioned_backward_euler_borrowed_rank1_engine(
                    config, trace_policy));
            allocated =
                partitioned_backward_euler_borrowed_rank1_ != nullptr;
          }
          else if (config.time_treatment ==
                   generic::EulerHdgTimeTreatment::rodas3)
          {
            rodas3_borrowed_rank1_.reset(
                new rodas3_borrowed_rank1_engine(config, trace_policy));
            allocated = rodas3_borrowed_rank1_ != nullptr;
          }
          else
          {
            borrowed_rank1_.reset(
                new borrowed_rank1_engine(config, trace_policy));
            allocated = borrowed_rank1_ != nullptr;
          }
        }
        else
        {
          rank1_.reset(new rank1_engine(config, trace_policy));
          allocated = rank1_ != nullptr;
        }
      }
      else
      {
        if (config.use_borrowed_host_inputs)
        {
          if (config.time_treatment == generic::EulerHdgTimeTreatment::
                  partitioned_backward_euler)
          {
            partitioned_backward_euler_borrowed_rank2_.reset(
                new partitioned_backward_euler_borrowed_rank2_engine(
                    config, trace_policy));
            allocated =
                partitioned_backward_euler_borrowed_rank2_ != nullptr;
          }
          else if (config.time_treatment ==
                   generic::EulerHdgTimeTreatment::rodas3)
          {
            rodas3_borrowed_rank2_.reset(
                new rodas3_borrowed_rank2_engine(config, trace_policy));
            allocated = rodas3_borrowed_rank2_ != nullptr;
          }
          else
          {
            borrowed_rank2_.reset(
                new borrowed_rank2_engine(config, trace_policy));
            allocated = borrowed_rank2_ != nullptr;
          }
        }
        else
        {
          rank2_.reset(new rank2_engine(config, trace_policy));
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
      partitioned_backward_euler_borrowed_rank1_.reset();
      partitioned_backward_euler_borrowed_rank2_.reset();
      rodas3_borrowed_rank1_.reset();
      rodas3_borrowed_rank2_.reset();
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

public:

  /** Stage the native candidate export; an explicit acknowledgement is next. */
  template <typename StagedOutput>
  result_type Prepare(const double dt, GenericEulerCudaPreparedStep& authority,
                      StagedOutput& native_export,
                      GenericEulerCudaPrepareFailureClass*
                          prepare_failure_class = nullptr) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    if (prepare_failure_class != nullptr)
      *prepare_failure_class =
          GenericEulerCudaPrepareFailureClass::not_failed;
    // A destination authority is never an overwrite slot.  In particular, a
    // live authority from another session remains usable by its owner when
    // this session rejects the prepare collectively.
    const bool local_valid = ReadyForPrepare() && !authority.valid_ &&
        OutputMatchesMode<StagedOutput>() && std::isfinite(dt) && dt > 0.0 &&
        (metadata_output_mode_ || SameDouble(dt, cuda_w8a::fixed_dt)) &&
        (!retry_ready_ || SameDouble(dt, retained_dt_));
    if (!AdmitOperation(operation_tag::prepare, local_valid))
      return Result(generic::HdgEulerSessionStatus::rejected);
    ClearAssessmentAuthority();
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
      const bool retryable_failure = CollectiveAll(
          !prepared_locally && evidence.retry_available &&
          evidence.coordinates_preserved &&
          evidence.committed_bytes_preserved &&
          EngineHasStatus(RunStatus::retry_available));
      if (retryable_failure)
      {
        const auto collective_failure_class =
            CollectivePrepareFailureClass(evidence);
        if (prepare_failure_class != nullptr)
          *prepare_failure_class = collective_failure_class;
        retained_dt_ = dt;
        retry_ready_ = true;
        coordinates_ = EngineCoordinates();
        return Result(generic::HdgEulerSessionStatus::rejected);
      }
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

  /** Export conservative values and optional IGR sigma in one host bridge. */
  bool ExportCommittedFields(
      generic::HdgEulerHostInteriorFields destination) noexcept
  {
    const bool local_valid = initialized_ && !failed_ && !prepared_active_ &&
        (destination.conservative_coefficients != nullptr ||
         destination.scalar_count == 0);
    if (!AdmitOperation(operation_tag::export_committed_fields, local_valid))
      return false;
    return Dispatch([&](auto& engine)
    { return engine.ExportCommittedFields(destination); });
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

  template <typename StagedOutput>
  bool ExportPreparedFields(
      const GenericEulerCudaPreparedStep& authority,
      const StagedOutput& native_export,
      generic::HdgEulerHostInteriorFields destination) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    const bool local_valid = OutputMatchesMode<StagedOutput>() &&
        Matches(authority, native_export) &&
        (destination.conservative_coefficients != nullptr ||
         destination.scalar_count == 0);
    if (!AdmitOperation(operation_tag::export_prepared_fields, local_valid))
      return false;
    return Dispatch([&](auto& engine)
    { return engine.ExportPreparedFields(destination); });
  }

  /**
   * Materialize both parts of the exact metadata-staged candidate into
   * caller-owned storage.  The pointers are transient copy destinations and
   * are never checkpoint identity or serialized artifact fields.
   */
  bool ExportPreparedCheckpointPayload(
      const GenericEulerCudaPreparedStep& authority,
      const prepared_output_type& metadata,
      const GenericEulerCudaMutableCheckpointPayloadV1 destination) noexcept
  {
    const bool local_valid = metadata_output_mode_ &&
        Matches(authority, metadata) && destination.interior != nullptr &&
        destination.owned_true_trace != nullptr &&
        destination.interior_count == metadata.interior_count &&
        destination.owned_true_trace_count ==
            metadata.owned_true_trace_count;
    if (!AdmitOperation(
            operation_tag::export_prepared_checkpoint_payload, local_valid))
      return false;
    return Dispatch([&](auto& engine)
    {
      return engine.ExportPreparedCheckpointPayload(
          metadata, destination.interior, destination.interior_count,
          destination.owned_true_trace,
          destination.owned_true_trace_count);
    });
  }

  /** Collectively assess one live prepared RODAS3 candidate. */
  template <typename StagedOutput>
  bool AssessPrepared(
      const GenericEulerCudaPreparedStep& authority,
      const StagedOutput& native_export,
      const generic::HdgEulerPreparedAssessmentPolicy policy,
      generic::HdgEulerPreparedAssessment& assessment) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    assessment = {};
    ClearAssessmentAuthority();
    const bool local_valid = OutputMatchesMode<StagedOutput>() &&
        Matches(authority, native_export) && RodasAssessmentRequired() &&
        generic::IsValidHdgEulerPreparedAssessmentPolicy(policy);
    if (!AdmitOperation(operation_tag::assess_prepared, local_valid))
      return false;
    const bool assessed = Dispatch([&](auto& engine)
    { return engine.AssessPrepared(native_export, policy, assessment); });
    const bool authorized = assessed &&
        generic::IsValidHdgEulerPreparedAssessment(assessment) &&
        assessment.finite_gate && assessment.solver_gate &&
        assessment.residual_gate && assessment.admissibility_gate &&
        assessment.algebraic_closure_gate &&
        assessment.algebraic_correction_gate;
    if (authorized)
    {
      assessment_authority_ = true;
      assessed_epoch_ = authority.epoch_;
      assessed_facade_incarnation_ = authority.facade_incarnation_;
      assessed_native_export_ = &native_export;
      assessed_policy_ = policy;
      assessed_policy_bits_ = {{
          generic::DoubleBits(policy.absolute_tolerance),
          generic::DoubleBits(policy.relative_tolerance),
          generic::DoubleBits(policy.algebraic_correction_fraction)}};
    }
    return assessed;
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
        output_acknowledged_ &&
        (!RodasAssessmentRequired() ||
         AssessmentMatches(authority, native_export));
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

  /**
   * Decline a prepared adaptive candidate and the contract's exact same-dt
   * retry, leaving the committed state ready for a fresh timestep.
   */
  template <typename StagedOutput>
  result_type RejectPrepared(GenericEulerCudaPreparedStep&& authority,
                             StagedOutput& native_export) noexcept
  {
    static_assert(IsSupportedOutput<StagedOutput>(),
                  "CUDA prepared output must be checkpoint or metadata");
    if (!AdmitOperation(operation_tag::reject_prepared,
                        OutputMatchesMode<StagedOutput>() &&
                        Matches(authority, native_export)))
      return Result(generic::HdgEulerSessionStatus::rejected);
    cuda_w8a::CudaStepEvidence evidence{};
    const bool rejected = Dispatch([&](auto& engine)
    { return engine.RejectPrepared(
        native_export, FailureReason::backend_failure, evidence); });
    const bool expected_contract_state = coordinates_.step == 0 &&
            coordinates_.generation == 0 && coordinates_.time == 0.0
        ? EngineHasStatus(RunStatus::ready)
        : EngineHasStatus(RunStatus::committed);
    const bool rollback_fresh = rejected && !evidence.retry_available &&
        evidence.coordinates_preserved &&
        evidence.committed_bytes_preserved && expected_contract_state &&
        !native_export.candidate_frozen &&
        !native_export.checkpoint_staged &&
        !native_export.output_staged &&
        !native_export.successful_commit;
    ConsumeAuthority(authority);
    if (!CollectiveAll(rollback_fresh))
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    retry_ready_ = false;
    coordinates_ = EngineCoordinates();
    return Result(generic::HdgEulerSessionStatus::aborted);
  }

  /** Abandon a retryable adaptive failure that produced no prepared token. */
  result_type RejectFailedPrepare() noexcept
  {
    const bool local_valid = initialized_ && !failed_ && !prepared_active_ &&
        active_native_export_ == nullptr && active_facade_incarnation_ == 0 &&
        retry_ready_ && EngineHasStatus(RunStatus::retry_available);
    if (!AdmitOperation(operation_tag::reject_failed_prepare, local_valid))
      return Result(generic::HdgEulerSessionStatus::rejected);
    cuda_w8a::CudaStepEvidence evidence{};
    const bool rejected = Dispatch([&](auto& engine)
    { return engine.RejectFailedAttempt(evidence); });
    const bool expected_contract_state = coordinates_.step == 0 &&
            coordinates_.generation == 0 && coordinates_.time == 0.0
        ? EngineHasStatus(RunStatus::ready)
        : EngineHasStatus(RunStatus::committed);
    const bool rollback_fresh = rejected && !evidence.retry_available &&
        evidence.coordinates_preserved &&
        evidence.committed_bytes_preserved && expected_contract_state;
    if (!CollectiveAll(rollback_fresh))
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    retry_ready_ = false;
    coordinates_ = EngineCoordinates();
    return Result(generic::HdgEulerSessionStatus::aborted);
  }

  /**
   * Restore one fieldwise-decoded native checkpoint into a pristine session.
   * The decoder owns artifact schema/hash validation; the existing CUDA
   * engine independently revalidates the exact tuple, mesh, partition,
   * topology, rank, order, layout, payload hashes, and committed frontier.
   */
  result_type Restore(const native_export_type& checkpoint) noexcept
  {
    const bool local_valid = initialized_ && !failed_ && !prepared_active_ &&
        active_native_export_ == nullptr && active_facade_incarnation_ == 0 &&
        coordinates_.step == 0 && coordinates_.generation == 0 &&
        coordinates_.time == 0.0 && !metadata_output_mode_ &&
        checkpoint.successful_commit;
    if (!AdmitOperation(operation_tag::restore, local_valid))
      return Result(generic::HdgEulerSessionStatus::rejected);
    cuda_w8a::CudaStepEvidence evidence{};
    const bool restored_locally = Dispatch([&](auto& engine)
    { return engine.Restore(checkpoint, evidence); });
    const bool restored = restored_locally &&
        EngineHasStatus(RunStatus::committed);
    if (!CollectiveAll(restored))
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    coordinates_ = EngineCoordinates();
    ++epoch_;
    retained_dt_ = 0.0;
    retry_ready_ = false;
    ClearAssessmentAuthority();
    if (metadata_output_mode_ && !RefreshCommittedDiagnostics())
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    return Result(generic::HdgEulerSessionStatus::committed);
  }

  /** Restore one fieldwise-decoded generic checkpoint into a pristine run. */
  result_type Restore(
      const prepared_output_type& metadata,
      const GenericEulerCudaCheckpointPayloadViewV1 payload) noexcept
  {
    const bool local_valid = initialized_ && !failed_ && !prepared_active_ &&
        active_native_export_ == nullptr && active_facade_incarnation_ == 0 &&
        coordinates_.step == 0 && coordinates_.generation == 0 &&
        coordinates_.time == 0.0 && metadata_output_mode_ &&
        metadata.successful_commit && payload.interior != nullptr &&
        payload.owned_true_trace != nullptr &&
        payload.interior_count == metadata.interior_count &&
        payload.owned_true_trace_count ==
            metadata.owned_true_trace_count;
    if (!AdmitOperation(operation_tag::restore, local_valid))
      return Result(generic::HdgEulerSessionStatus::rejected);
    cuda_w8a::CudaStepEvidence evidence{};
    const bool restored_locally = Dispatch([&](auto& engine)
    {
      return engine.Restore(
          metadata, payload.interior, payload.interior_count,
          payload.owned_true_trace, payload.owned_true_trace_count,
          evidence);
    });
    const bool restored = restored_locally &&
        EngineHasStatus(RunStatus::committed);
    if (!CollectiveAll(restored))
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    coordinates_ = EngineCoordinates();
    ++epoch_;
    retained_dt_ = 0.0;
    retry_ready_ = false;
    ClearAssessmentAuthority();
    if (!RefreshCommittedDiagnostics())
    {
      failed_ = true;
      return Result(generic::HdgEulerSessionStatus::rejected);
    }
    return Result(generic::HdgEulerSessionStatus::committed);
  }

  coordinates_type coordinates() const noexcept { return coordinates_; }

  generic::HdgEulerCommittedDiagnostics diagnostics() const noexcept
  { return diagnostics_; }

private:
  friend struct cuda_w8a::detail::CudaQualificationTestAccess;

  enum class operation_tag : int
  {
    initialize = 1,
    prepare = 2,
    acknowledge_output = 3,
    commit = 4,
    abort = 5,
    export_committed = 6,
    export_prepared = 7,
    export_committed_fields = 8,
    export_prepared_fields = 9,
    assess_prepared = 10,
    reject_prepared = 11,
    reject_failed_prepare = 12,
    restore = 13,
    export_prepared_checkpoint_payload = 14
  };

  using rank1_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      CudaRank1OneGpuRuntimeSelection>;
  using rank2_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      CudaRank2OneGpuHostStagedRuntimeSelection>;
  using borrowed_rank1_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      BorrowedCudaRank1OneGpuRuntimeSelection>;
  using borrowed_rank2_engine = cuda_w8a::PersistentCudaSmoothEulerEngine<
      BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>;
  using partitioned_backward_euler_borrowed_rank1_engine =
      cuda_w8a::PersistentCudaSmoothEulerEngine<
          PartitionedBackwardEulerCudaRank1RuntimeSelection>;
  using partitioned_backward_euler_borrowed_rank2_engine =
      cuda_w8a::PersistentCudaSmoothEulerEngine<
          PartitionedBackwardEulerCudaRank2RuntimeSelection>;
  using rodas3_borrowed_rank1_engine =
      cuda_w8a::PersistentCudaSmoothEulerEngine<
          Rodas3CudaRank1RuntimeSelection>;
  using rodas3_borrowed_rank2_engine =
      cuda_w8a::PersistentCudaSmoothEulerEngine<
          Rodas3CudaRank2RuntimeSelection>;

  static coordinates_type ToCoordinates(
      const StateCoordinates& value) noexcept
  {
    return {value.time, value.dt, value.step, value.generation};
  }

  static bool SameDouble(const double left, const double right) noexcept
  { return std::memcmp(&left, &right, sizeof(double)) == 0; }

  bool ExactDtConsensus(const double dt) const noexcept
  {
    std::uint64_t bits = 0, minimum = 0, maximum = 0;
    std::memcpy(&bits, &dt, sizeof(bits));
    const int minimum_status = MPI_Allreduce(&bits, &minimum, 1,
        MPI_UINT64_T, MPI_MIN, communicator_);
    const int maximum_status = MPI_Allreduce(&bits, &maximum, 1,
        MPI_UINT64_T, MPI_MAX, communicator_);
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
    return IsMetadataOutput<StagedOutput>() == metadata_output_mode_;
  }

  bool ReadyForPrepare() const noexcept
  {
    return initialized_ && !failed_ && !prepared_active_ &&
        active_native_export_ == nullptr && active_facade_incarnation_ == 0;
  }

  bool RodasAssessmentRequired() const noexcept
  {
    return rodas3_borrowed_rank1_ != nullptr ||
        rodas3_borrowed_rank2_ != nullptr;
  }

  void ClearAssessmentAuthority() noexcept
  {
    assessment_authority_ = false;
    assessed_epoch_ = 0;
    assessed_facade_incarnation_ = 0;
    assessed_native_export_ = nullptr;
    assessed_policy_ = {};
    assessed_policy_bits_ = {};
  }

  template <typename StagedOutput>
  bool AssessmentMatches(
      const GenericEulerCudaPreparedStep& authority,
      const StagedOutput& native_export) const noexcept
  {
    const std::array<std::uint64_t, 3> policy_bits{{
        generic::DoubleBits(assessed_policy_.absolute_tolerance),
        generic::DoubleBits(assessed_policy_.relative_tolerance),
        generic::DoubleBits(
            assessed_policy_.algebraic_correction_fraction)}};
    return assessment_authority_ && authority.valid_ &&
        assessed_epoch_ == authority.epoch_ &&
        assessed_facade_incarnation_ == authority.facade_incarnation_ &&
        assessed_native_export_ == static_cast<const void*>(&native_export) &&
        generic::IsValidHdgEulerPreparedAssessmentPolicy(assessed_policy_) &&
        assessed_policy_bits_ == policy_bits;
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
    const int local_available_bit = local_available ? 1 : 0;
    int every_available = 0;
    std::uint64_t maximum_prior = next_session_collective_id_;
    const bool availability_ready = MPI_Allreduce(
        &local_available_bit, &every_available, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        &next_session_collective_id_, &maximum_prior, 1, MPI_UINT64_T,
        MPI_MAX, communicator_) == MPI_SUCCESS;
    if (!availability_ready || !maximum_ready || every_available != 1 ||
        maximum_prior == std::numeric_limits<std::uint64_t>::max())
      return false;
    next_session_collective_id_ = maximum_prior + 1;
    session_collective_id_ = next_session_collective_id_;
    return true;
  }

  bool AdmitOperation(const operation_tag tag, const bool local_valid) const
      noexcept
  {
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS || !mpi_initialized ||
        MPI_Finalized(&mpi_finalized) != MPI_SUCCESS || mpi_finalized ||
        communicator_ == MPI_COMM_NULL)
      return false;
    const std::uint64_t local[3] = {
        session_collective_id_, static_cast<std::uint64_t>(tag),
        local_valid ? 1U : 0U};
    std::uint64_t minimum[3] = {local[0], local[1], local[2]};
    std::uint64_t maximum[3] = {local[0], local[1], local[2]};
    const bool minimum_ready = MPI_Allreduce(
        local, minimum, 3, MPI_UINT64_T, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local, maximum, 3, MPI_UINT64_T, MPI_MAX,
        communicator_) == MPI_SUCCESS;
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

  bool CollectiveAll(const bool local) const noexcept
  {
    const int input = local ? 1 : 0;
    int output = 0;
    return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                         communicator_) == MPI_SUCCESS && output == 1;
  }

  static GenericEulerCudaPrepareFailureClass LocalPrepareFailureClass(
      const cuda_w8a::CudaStepEvidence& evidence) noexcept
  {
    if (evidence.positivity_limiter_inadmissible_mean_rejections != 0)
    {
      return GenericEulerCudaPrepareFailureClass::
          zhang_shu_inadmissible_conservative_mean;
    }
    switch (evidence.contract_reason)
    {
    case FailureReason::finite_gate:
      return GenericEulerCudaPrepareFailureClass::finite_gate;
    case FailureReason::admissibility_gate:
      return GenericEulerCudaPrepareFailureClass::admissibility_gate;
    case FailureReason::recovery_gate:
      return GenericEulerCudaPrepareFailureClass::recovery_gate;
    case FailureReason::true_residual_gate:
      return GenericEulerCudaPrepareFailureClass::true_residual_gate;
    case FailureReason::telemetry_gate:
      return GenericEulerCudaPrepareFailureClass::telemetry_gate;
    case FailureReason::success_gate:
      return GenericEulerCudaPrepareFailureClass::success_gate;
    case FailureReason::collective_protocol:
      return GenericEulerCudaPrepareFailureClass::collective_protocol;
    default:
      break;
    }
    if (evidence.phase == cuda_w8a::CudaFailurePhase::trace_solve)
      return GenericEulerCudaPrepareFailureClass::trace_solve;
    if (evidence.phase == cuda_w8a::CudaFailurePhase::recovery)
      return GenericEulerCudaPrepareFailureClass::recovery_gate;
    return GenericEulerCudaPrepareFailureClass::backend_or_contract;
  }

  GenericEulerCudaPrepareFailureClass CollectivePrepareFailureClass(
      const cuda_w8a::CudaStepEvidence& evidence) const noexcept
  {
    const auto local_class = LocalPrepareFailureClass(evidence);
    const unsigned local[2] = {
        evidence.positivity_limiter_inadmissible_mean_rejections != 0 ? 1U
                                                                       : 0U,
        static_cast<unsigned>(local_class)};
    unsigned minimum[2] = {local[0], local[1]};
    unsigned maximum[2] = {local[0], local[1]};
    const bool minimum_ready = MPI_Allreduce(
        local, minimum, 2, MPI_UNSIGNED, MPI_MIN, communicator_) ==
        MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local, maximum, 2, MPI_UNSIGNED, MPI_MAX, communicator_) ==
        MPI_SUCCESS;
    if (!minimum_ready || !maximum_ready)
      return GenericEulerCudaPrepareFailureClass::unavailable;
    // One inadmissible conservative cell mean is sufficient to invalidate the
    // collective candidate; unlike the remaining categories it need not be
    // observed independently on every rank.
    if (maximum[0] != 0)
    {
      return GenericEulerCudaPrepareFailureClass::
          zhang_shu_inadmissible_conservative_mean;
    }
    if (minimum[1] != maximum[1])
      return GenericEulerCudaPrepareFailureClass::mixed_numerical_failure;
    return local_class;
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
         borrowed_rank2_->contract()->status() == expected) ||
        (partitioned_backward_euler_borrowed_rank1_ &&
         partitioned_backward_euler_borrowed_rank1_->contract() &&
         partitioned_backward_euler_borrowed_rank1_->contract()->status() ==
             expected) ||
        (partitioned_backward_euler_borrowed_rank2_ &&
         partitioned_backward_euler_borrowed_rank2_->contract() &&
         partitioned_backward_euler_borrowed_rank2_->contract()->status() ==
             expected) ||
        (rodas3_borrowed_rank1_ && rodas3_borrowed_rank1_->contract() &&
         rodas3_borrowed_rank1_->contract()->status() == expected) ||
        (rodas3_borrowed_rank2_ && rodas3_borrowed_rank2_->contract() &&
         rodas3_borrowed_rank2_->contract()->status() == expected);
  }

  void ConsumeAuthority(GenericEulerCudaPreparedStep& authority) noexcept
  {
    authority.Clear();
    prepared_active_ = false;
    output_acknowledged_ = false;
    active_native_export_ = nullptr;
    active_facade_incarnation_ = 0;
    ClearAssessmentAuthority();
  }

  using config_descriptor = std::array<unsigned char, 1024>;
  using policy_descriptor = std::array<unsigned char, 2048>;
  using topology_descriptor = std::array<unsigned char, 128>;

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

  bool ConfigConsensus(
      const cuda_w8a::CudaConfig& config,
      const cuda_w8a::CudaTracePolicyDescriptor& trace_policy) const noexcept
  {
    bool borrowed_host_mode = false;
    // This min/max pair is unconditional and precedes every descriptor whose
    // shape depends on borrowed-host mode.  A mismatch therefore cannot send
    // ranks into different collective sequences.
    if (!cuda_w8a::CudaBorrowedHostModeConsensus(
            communicator_,
            config.use_borrowed_host_inputs, borrowed_host_mode))
      return false;
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
    AppendConfigValue(local, cursor, complete, config.cuda_engine);
    AppendConfigValue(local, cursor, complete, config.expected_rank_count);
    append_bool(config.require_host_staged_rank2);
    append_bool(config.use_borrowed_topology);
    // Preserve the frozen legacy descriptor byte-for-byte.  The data-mode
    // branch carries only rank-common ABI facts: local L2 counts are expected
    // to vary with the borrowed partition and are validated locally later.
    if (borrowed_host_mode)
    {
      append_bool(true);
      AppendConfigValue(local, cursor, complete,
                        config.borrowed_initial.components);
      AppendConfigValue(local, cursor, complete,
                        config.borrowed_initial_true_trace.components);
      append_bool(config.borrowed_initial_true_trace.coefficients != nullptr);
      AppendConfigValue(local, cursor, complete,
                        config.borrowed_source.components);
      append_bool(config.borrowed_source.coefficients != nullptr);
      append_bool(config.use_generic_numerical_policy);
      if (config.use_generic_numerical_policy)
      {
        const auto& policy = config.numerical_policy;
        AppendConfigValue(local, cursor, complete,
                          policy.volume_projection_quadrature_order);
        AppendConfigValue(local, cursor, complete,
                          policy.true_trace_projection_quadrature_order);
        AppendConfigValue(local, cursor, complete,
                          policy.conventional_assembly_quadrature_order);
        AppendConfigValue(local, cursor, complete,
                          policy.igr_assembly_quadrature_orders.volume_order);
        AppendConfigValue(local, cursor, complete,
                          policy.igr_assembly_quadrature_orders.face_order);
        AppendConfigValue(local, cursor, complete,
                          policy.maximum_nonlinear_updates);
        AppendConfigValue(local, cursor, complete,
                          policy.maximum_linear_iterations);
        AppendConfigValue(local, cursor, complete,
                          policy.maximum_line_search_trials);
        AppendConfigValue(local, cursor, complete,
                          policy.nonlinear_relative_tolerance);
        AppendConfigValue(local, cursor, complete,
                          policy.newton_correction_linear_relative_tolerance);
        AppendConfigValue(local, cursor, complete,
                          policy.final_coherent_linear_relative_tolerance);
        AppendConfigValue(
            local, cursor, complete,
            policy.true_residual_tightening_linear_relative_tolerance);
        AppendConfigValue(local, cursor, complete,
                          policy.true_residual_relative_tolerance);
        AppendConfigValue(local, cursor, complete,
                          policy.conservation_relative_tolerance);
        AppendConfigValue(local, cursor, complete,
                          policy.armijo_constant);
        AppendConfigValue(local, cursor, complete,
                          policy.stabilization.kind);
        AppendConfigValue(local, cursor, complete,
                          policy.stabilization.alpha);
        AppendConfigValue(local, cursor, complete,
                          policy.stabilization.beta);
      }
    }
    append_bool(config.capture_oracle);
    append_bool(config.force_private_scientific_gate_failure);
    AppendConfigValue(local, cursor, complete, config.injection.phase);
    AppendConfigValue(local, cursor, complete, config.injection.target_step);
    AppendConfigValue(local, cursor, complete, config.injection.source_rank);
    append_bool(config.injection.enabled);
    AppendConfigValue(local, cursor, complete, config.injection.token_fault);
    AppendConfigValue(local, cursor, complete, config.time_treatment);
    // Append-only so the engine-default prefix stays byte-identical.
    AppendConfigValue(
        local, cursor, complete, trace_policy.requested_selection);
    AppendConfigValue(
        local, cursor, complete, trace_policy.resolved_selection);
    AppendConfigValue(local, cursor, complete, trace_policy.rank_count);
    AppendConfigValue(local, cursor, complete, trace_policy.polynomial_order);
    // Keep the pre-limiter descriptor prefix byte-identical.  A legacy or
    // value-initialized policy appends only zero bytes here.
    if (borrowed_host_mode && config.use_generic_numerical_policy)
    {
      const auto& limiter = config.numerical_policy.positivity_limiter;
      AppendConfigValue(local, cursor, complete, limiter.kind);
      AppendConfigValue(local, cursor, complete, limiter.point_set);
      AppendConfigValue(local, cursor, complete, limiter.density_floor);
      AppendConfigValue(local, cursor, complete, limiter.pressure_floor);
      AppendConfigValue(
          local, cursor, complete, limiter.pressure_bisection_iterations);
    }

    config_descriptor minimum = local;
    config_descriptor maximum = local;
    const int complete_bit = complete ? 1 : 0;
    int all_complete = 0;
    const bool complete_census = MPI_Allreduce(
        &complete_bit, &all_complete, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool minimum_census = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_census = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, communicator_) == MPI_SUCCESS;
    const bool base_consensus = complete_census && minimum_census && maximum_census &&
        all_complete == 1 && minimum == maximum;
    if (!base_consensus || !borrowed_host_mode) return base_consensus;
    // The exact fixed-size policy consensus is always run in true mode.  Only
    // a nonempty policy table activates the physical-boundary identity ABI;
    // this leaves existing borrowed periodic callers unchanged.
    if (!PolicyTableConsensus(config) ||
        !generic::ExactEulerBoundaryPolicyConsensus(
            communicator_, config.borrowed_boundary.policies))
      return false;
    return config.borrowed_boundary_policies.count == 0 ||
        TopologyIdentityConsensus(config);
  }

  bool TopologyIdentityConsensus(
      const cuda_w8a::CudaConfig& config) const noexcept
  {
    topology_descriptor local{}, minimum{}, maximum{};
    const StableIdentity& identity = config.borrowed_global_topology;
    const std::uint64_t length = identity.size();
    const std::uint8_t valid = identity.valid() ? 1U : 0U;
    bool complete = cuda_w8a::IsCudaBorrowedGlobalTopologyIdentity(identity) &&
        length <= local.size() - sizeof(length) - sizeof(valid);
    std::size_t cursor = 0;
    if (complete)
    {
      std::memcpy(local.data() + cursor, &length, sizeof(length));
      cursor += sizeof(length);
      std::memcpy(local.data() + cursor, &valid, sizeof(valid));
      cursor += sizeof(valid);
      std::memcpy(local.data() + cursor, identity.data(),
                  static_cast<std::size_t>(length));
    }
    minimum = local;
    maximum = local;
    const int complete_bit = complete ? 1 : 0;
    int all_complete = 0;
    const bool complete_ready = MPI_Allreduce(
        &complete_bit, &all_complete, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool minimum_ready = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, communicator_) == MPI_SUCCESS;
    return complete_ready && minimum_ready && maximum_ready &&
        all_complete == 1 && minimum == maximum;
  }

  bool PolicyTableConsensus(
      const cuda_w8a::CudaConfig& config) const noexcept
  {
    policy_descriptor local{}, minimum{}, maximum{};
    std::size_t cursor = 0;
    bool complete = cuda_w8a::ValidateCudaStaticBoundaryPolicyTable(
        config.borrowed_boundary_policies);
    const auto append = [&](const auto& value)
    {
      using value_type = std::decay_t<decltype(value)>;
      if (!complete || sizeof(value_type) > local.size() - cursor)
      { complete = false; return; }
      std::memcpy(local.data() + cursor, &value, sizeof(value_type));
      cursor += sizeof(value_type);
    };
    append(config.borrowed_boundary_policies.count);
    append(config.borrowed_boundary_policies.gamma);
    const bool readable = config.borrowed_boundary_policies.policies != nullptr &&
        config.borrowed_boundary_policies.count <=
            cuda_w8a::maximum_static_boundary_policies;
    for (std::size_t index = 0;
         index < cuda_w8a::maximum_static_boundary_policies; ++index)
    {
      const cuda_w8a::CudaStaticBoundaryPolicy empty{};
      const cuda_w8a::CudaStaticBoundaryPolicy& policy = readable && index <
          config.borrowed_boundary_policies.count
          ? config.borrowed_boundary_policies.policies[index] : empty;
      append(policy.attribute);
      const std::uint8_t kind = static_cast<std::uint8_t>(policy.kind);
      append(kind);
      for (const double reference : policy.reference_state) append(reference);
      append(policy.pressure_out);
    }
    minimum = local;
    maximum = local;
    const int complete_bit = complete ? 1 : 0;
    int all_complete = 0;
    const bool complete_ready = MPI_Allreduce(
        &complete_bit, &all_complete, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool minimum_ready = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, communicator_) == MPI_SUCCESS;
    return complete_ready && minimum_ready && maximum_ready &&
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

  bool PreDeviceAdmission(
      const cuda_w8a::CudaConfig& config,
      const cuda_w8a::CudaTracePolicyDescriptor& trace_policy) const noexcept
  {
    int mpi_initialized = 0;
    int mpi_finalized = 0;
    if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS || !mpi_initialized ||
        MPI_Finalized(&mpi_finalized) != MPI_SUCCESS || mpi_finalized)
      return false;
    const bool config_consensus = ConfigConsensus(config, trace_policy);
    const bool empty_borrowed_fields =
        config.borrowed_initial.coefficients == nullptr &&
        config.borrowed_initial.scalar_count == 0 &&
        config.borrowed_initial.components == cuda_w8a::conserved_components &&
        config.borrowed_initial_true_trace.coefficients == nullptr &&
        config.borrowed_initial_true_trace.scalar_count == 0 &&
        config.borrowed_initial_true_trace.components ==
            cuda_w8a::conserved_components &&
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
    const bool initial_trace_abi = config.use_borrowed_host_inputs
        ? (config.borrowed_initial_true_trace.coefficients == nullptr &&
           config.borrowed_initial_true_trace.scalar_count == 0 &&
           config.borrowed_initial_true_trace.components ==
               cuda_w8a::conserved_components) ||
            (config.borrowed_initial_true_trace.coefficients != nullptr &&
             config.borrowed_initial_true_trace.scalar_count != 0 &&
             config.borrowed_initial_true_trace.components ==
                 cuda_w8a::conserved_components)
        : true;
    const bool numerical_policy_abi = config.use_generic_numerical_policy
        ? config.use_borrowed_host_inputs &&
            generic::ValidEulerHdgNumericalPolicy(config.numerical_policy)
        : true;
    const bool boundary_policy_abi = config.use_borrowed_host_inputs
        ? cuda_w8a::ValidateCudaStaticBoundaryPolicyTable(
              config.borrowed_boundary_policies)
        : config.borrowed_boundary_policies.policies == nullptr &&
            config.borrowed_boundary_policies.count == 0;
    const bool borrowed_topology_identity = config.use_borrowed_host_inputs
        ? (config.borrowed_boundary_policies.count == 0
            ? cuda_w8a::IsCudaAbsentBorrowedGlobalTopologyIdentity(
                config.borrowed_global_topology)
            : cuda_w8a::IsCudaBorrowedGlobalTopologyIdentity(
                config.borrowed_global_topology))
        : cuda_w8a::IsCudaAbsentBorrowedGlobalTopologyIdentity(
            config.borrowed_global_topology);
    const bool partitioned_backward_euler = config.time_treatment ==
        generic::EulerHdgTimeTreatment::partitioned_backward_euler;
    const bool partitioned_backward_euler_configuration =
        partitioned_backward_euler &&
        (config.expected_rank_count == 1 ||
         config.expected_rank_count == 2) &&
        config.use_borrowed_host_inputs &&
        config.use_generic_numerical_policy &&
        (config.numerical_policy.stabilization.kind ==
             generic::RegularizationKind::none ||
         config.numerical_policy.stabilization.kind ==
             generic::RegularizationKind::igr);
    const bool time_treatment_abi =
        generic::IsKnownEulerHdgTimeTreatment(config.time_treatment) &&
        (!partitioned_backward_euler ||
         partitioned_backward_euler_configuration);
    int ranks = 0;
    int borrowed_collective = MPI_UNEQUAL;
    int borrowed_mesh = MPI_UNEQUAL;
    bool local = config.use_borrowed_topology && config.borrowed_mesh != nullptr &&
        config.borrowed_communicator != MPI_COMM_NULL &&
        generic::IsKnownCudaEngineKind(config.cuda_engine) &&
        cuda_w8a::CudaTracePolicyDescriptorIsExact(trace_policy) &&
        trace_policy.selected_engine == config.cuda_engine &&
        trace_policy.rank_count == config.expected_rank_count &&
        trace_policy.polynomial_order == config.order &&
        trace_policy.igr_active ==
            (config.use_generic_numerical_policy &&
             config.numerical_policy.stabilization.kind ==
                 generic::RegularizationKind::igr) &&
        (trace_policy.requested_selection ==
             cuda_w8a::CudaTracePolicyKind::engine_default ||
         (config.expected_rank_count == 1 &&
          config.use_borrowed_host_inputs &&
          config.use_generic_numerical_policy)) &&
        borrowed_data_abi &&
        initial_trace_abi && numerical_policy_abi &&
        boundary_policy_abi &&
        borrowed_topology_identity &&
        time_treatment_abi &&
        MPI_Comm_size(communicator_, &ranks) == MPI_SUCCESS &&
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
        config.order >= generic::minimum_polynomial_order &&
        config.order <= generic::maximum_polynomial_order &&
        config.mesh_n >= 0;
    long long local_face_measure = 0;
    long long local_physical_faces = 0;
    long long global_elements = 0;
    long long global_face_measure = 0;
    long long global_physical_faces = 0;
    std::array<long long, cuda_w8a::maximum_static_boundary_policies>
        local_policy_uses{};
    std::array<long long, cuda_w8a::maximum_static_boundary_policies>
        global_policy_uses{};
    if (local)
    {
      const MPI_Comm mesh_communicator = config.borrowed_mesh->GetComm();
      local = MPI_Comm_compare(config.borrowed_communicator, communicator_,
                               &borrowed_collective) == MPI_SUCCESS &&
          (borrowed_collective == MPI_IDENT ||
           borrowed_collective == MPI_CONGRUENT) &&
          MPI_Comm_compare(config.borrowed_communicator, mesh_communicator,
                           &borrowed_mesh) == MPI_SUCCESS &&
          (borrowed_mesh == MPI_IDENT || borrowed_mesh == MPI_CONGRUENT) &&
          !config.borrowed_mesh->Nonconforming() &&
          config.borrowed_mesh->Dimension() == cuda_w8a::dimension &&
          config.borrowed_mesh->SpaceDimension() == cuda_w8a::dimension &&
          // The current rank1/rank2 CUDA trace/HYPRE construction dereferences
          // a local element FE during topology setup.  Until an empty local
          // space is represented end to end, reject any zero-owned rank here,
          // collectively and before device selection/allocation.  The shared
          // generic contract remains broader; this backend returns `rejected`.
          config.borrowed_mesh->GetNE() > 0;
      for (int element = 0; local && element < config.borrowed_mesh->GetNE();
           ++element)
        local = config.borrowed_mesh->GetElementBaseGeometry(element) ==
            mfem::Geometry::SQUARE;
      if (local)
      {
        try
        {
          const int face_count = config.borrowed_mesh->GetNumFaces();
          if (face_count < 0) local = false;
          std::vector<unsigned char> shared(
              local ? static_cast<std::size_t>(face_count) : 0, 0);
          for (int index = 0; local &&
               index < config.borrowed_mesh->GetNSharedFaces(); ++index)
          {
            const int face = config.borrowed_mesh->GetSharedFace(index);
            if (face < 0 || face >= static_cast<int>(shared.size())) local = false;
            else shared[static_cast<std::size_t>(face)] = 1;
          }
          for (int element = 0; local && element < config.borrowed_mesh->GetNE();
               ++element)
          {
            mfem::Array<int> faces, orientations;
            config.borrowed_mesh->GetElementEdges(element, faces, orientations);
            if (faces.Size() != cuda_w8a::faces_per_element) { local = false; break; }
            for (int local_face = 0; local_face < faces.Size(); ++local_face)
            {
              const int face = faces[local_face];
              mfem::FaceElementTransformations* transformations =
                  config.borrowed_mesh->GetFaceElementTransformations(face);
              if (face < 0 || face >= static_cast<int>(shared.size()) ||
                  transformations == nullptr || transformations->Face == nullptr ||
                  (transformations->Elem1No != element &&
                   transformations->Elem2No != element))
              { local = false; break; }
              const int boundary_element = config.borrowed_mesh->GetFaceToBdrElMap()[face];
              const bool is_shared = shared[static_cast<std::size_t>(face)] != 0;
              const bool physical = !is_shared && transformations->Elem2No < 0 &&
                  boundary_element >= 0;
              if (!physical)
              {
                if (!is_shared && transformations->Elem2No < 0) local = false;
                if (!local) break;
                continue;
              }
              const int attribute = config.borrowed_mesh->GetBdrAttribute(boundary_element);
              const cuda_w8a::CudaStaticBoundaryPolicy* found = nullptr;
              std::size_t found_index =
                  cuda_w8a::maximum_static_boundary_policies;
              for (std::size_t policy = 0;
                   policy < config.borrowed_boundary_policies.count; ++policy)
                if (config.borrowed_boundary_policies.policies[policy].attribute == attribute)
                {
                  found = config.borrowed_boundary_policies.policies + policy;
                  found_index = policy;
                }
              if (attribute <= 0 || found == nullptr ||
                  found_index >= config.borrowed_boundary_policies.count ||
                  !cuda_w8a::IsCudaAdmittedStaticBoundaryPolicy(found->kind) ||
                  local_policy_uses[found_index] ==
                      std::numeric_limits<long long>::max())
              { local = false; break; }
              ++local_policy_uses[found_index];
              ++local_physical_faces;
            }
          }
        }
        catch (...) { local = false; }
        const long long faces = config.borrowed_mesh->GetNumFaces();
        const long long shared_count = config.borrowed_mesh->GetNSharedFaces();
        local = local && faces >= 0 && shared_count >= 0 &&
            shared_count <= faces &&
            faces <= std::numeric_limits<long long>::max() / 2;
        if (local) local_face_measure = 2 * faces - shared_count;
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
                                             communicator_) == MPI_SUCCESS;
    const long long local_elements = local ? config.borrowed_mesh->GetNE() : 0;
    const bool element_census = MPI_Allreduce(&local_elements, &global_elements,
        1, MPI_LONG_LONG_INT, MPI_SUM, communicator_) == MPI_SUCCESS;
    const bool face_census = MPI_Allreduce(&local_face_measure,
        &global_face_measure, 1, MPI_LONG_LONG_INT, MPI_SUM,
        communicator_) == MPI_SUCCESS;
    const bool boundary_census = MPI_Allreduce(&local_physical_faces,
        &global_physical_faces, 1, MPI_LONG_LONG_INT, MPI_SUM,
        communicator_) == MPI_SUCCESS;
    const bool policy_usage_census = MPI_Allreduce(local_policy_uses.data(),
        global_policy_uses.data(),
        static_cast<int>(global_policy_uses.size()), MPI_LONG_LONG_INT,
        MPI_SUM, communicator_) == MPI_SUCCESS;
    bool every_policy_used = config.borrowed_boundary_policies.count == 0
        ? global_physical_faces == 0
        : global_physical_faces > 0;
    for (std::size_t policy = 0; every_policy_used &&
         policy < config.borrowed_boundary_policies.count; ++policy)
      every_policy_used = global_policy_uses[policy] > 0;
    return config_consensus && local_census && element_census && face_census &&
        boundary_census && policy_usage_census && every_policy_used &&
        all_local == 1 && global_elements >= 0 &&
        global_physical_faces >= 0 && global_elements <=
            (std::numeric_limits<long long>::max() - global_physical_faces) /
                cuda_w8a::faces_per_element &&
        global_face_measure == cuda_w8a::faces_per_element * global_elements +
            global_physical_faces;
  }

  template <typename Callback>
  bool Dispatch(Callback&& callback) noexcept
  {
    if (rank1_) return callback(*rank1_);
    if (rank2_) return callback(*rank2_);
    if (borrowed_rank1_) return callback(*borrowed_rank1_);
    if (borrowed_rank2_) return callback(*borrowed_rank2_);
    if (partitioned_backward_euler_borrowed_rank1_)
      return callback(*partitioned_backward_euler_borrowed_rank1_);
    if (partitioned_backward_euler_borrowed_rank2_)
      return callback(*partitioned_backward_euler_borrowed_rank2_);
    if (rodas3_borrowed_rank1_)
      return callback(*rodas3_borrowed_rank1_);
    if (rodas3_borrowed_rank2_)
      return callback(*rodas3_borrowed_rank2_);
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
    if (partitioned_backward_euler_borrowed_rank1_ &&
        partitioned_backward_euler_borrowed_rank1_->contract())
      return ToCoordinates(
          partitioned_backward_euler_borrowed_rank1_->contract()->
              coordinates());
    if (partitioned_backward_euler_borrowed_rank2_ &&
        partitioned_backward_euler_borrowed_rank2_->contract())
      return ToCoordinates(
          partitioned_backward_euler_borrowed_rank2_->contract()->
              coordinates());
    if (rodas3_borrowed_rank1_ && rodas3_borrowed_rank1_->contract())
      return ToCoordinates(
          rodas3_borrowed_rank1_->contract()->coordinates());
    if (rodas3_borrowed_rank2_ && rodas3_borrowed_rank2_->contract())
      return ToCoordinates(
          rodas3_borrowed_rank2_->contract()->coordinates());
    return coordinates_;
  }

  std::unique_ptr<rank1_engine> rank1_;
  std::unique_ptr<rank2_engine> rank2_;
  std::unique_ptr<borrowed_rank1_engine> borrowed_rank1_;
  std::unique_ptr<borrowed_rank2_engine> borrowed_rank2_;
  std::unique_ptr<partitioned_backward_euler_borrowed_rank1_engine>
      partitioned_backward_euler_borrowed_rank1_;
  std::unique_ptr<partitioned_backward_euler_borrowed_rank2_engine>
      partitioned_backward_euler_borrowed_rank2_;
  std::unique_ptr<rodas3_borrowed_rank1_engine> rodas3_borrowed_rank1_;
  std::unique_ptr<rodas3_borrowed_rank2_engine> rodas3_borrowed_rank2_;
  cuda_w8a::CudaRunEvidence setup_{};
  coordinates_type coordinates_{};
  MPI_Comm communicator_ = MPI_COMM_NULL;
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
  bool assessment_authority_ = false;
  std::uint64_t assessed_epoch_ = 0;
  std::uint64_t assessed_facade_incarnation_ = 0;
  const void* assessed_native_export_ = nullptr;
  generic::HdgEulerPreparedAssessmentPolicy assessed_policy_{};
  std::array<std::uint64_t, 3> assessed_policy_bits_{};
  inline static std::uint64_t next_session_collective_id_ =
      unassigned_session_collective_id;
};

}  // namespace dg::hdg::runtime::cuda
