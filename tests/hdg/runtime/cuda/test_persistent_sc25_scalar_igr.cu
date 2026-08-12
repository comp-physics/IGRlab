#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr.cuh"
#include "sc25_scalar_igr_independent_oracle.cuh"
#include "sc25_scalar_igr_independent_oracle.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "HDG-EXP-SCALAR-DEVICE-01 requires a native SM120 image"
#endif

namespace
{
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25;
namespace cpu = ::dg::hdg::runtime::cpu::sc25;
namespace oracle = ::dg_igr_hdg_sc25_cuda_test;
namespace cpu_oracle = ::dg_igr_hdg_sc25_cpu_test;

constexpr double comparison_tolerance = 1.0e-10;

bool CpuAccepted(const cpu::TrajectoryStatus status) noexcept
{
  return status == cpu::TrajectoryStatus::HdgSc25Accepted ||
      status == cpu::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

bool CudaAccepted(const scalar::ScalarCudaStatus status) noexcept
{
  return status == scalar::ScalarCudaStatus::accepted ||
      status == scalar::ScalarCudaStatus::accepted_after_one_retry;
}

bool SameCoordinates(const ::dg::hdg::runtime::StateCoordinates& left,
                     const ::dg::hdg::runtime::StateCoordinates& right)
    noexcept
{
  return left.time == right.time && left.step == right.step &&
      left.generation == right.generation && left.cursor == right.cursor &&
      left.dt == right.dt;
}

bool SameContext(const scalar::cuda_w8a::CudaPolicyContext& left,
                 const scalar::cuda_w8a::CudaPolicyContext& right) noexcept
{
  return left.attempt_sequence == right.attempt_sequence &&
      left.target_step == right.target_step &&
      left.generation == right.generation && left.rank == right.rank &&
      left.phase == right.phase;
}

bool SameTransferEvent(const scalar::cuda_w8a::TransferEventV1& left,
                       const scalar::cuda_w8a::TransferEventV1& right)
    noexcept
{
  return left.ordinal == right.ordinal && left.direction == right.direction &&
      left.category == right.category && left.site == right.site &&
      SameContext(left.context, right.context) && left.bytes == right.bytes &&
      left.source_token == right.source_token &&
      left.destination_token == right.destination_token &&
      left.source_capacity == right.source_capacity &&
      left.destination_capacity == right.destination_capacity &&
      left.endpoint_token == right.endpoint_token &&
      left.success == right.success;
}

bool SameAllocationEvent(const scalar::cuda_w8a::AllocationEventV1& left,
                         const scalar::cuda_w8a::AllocationEventV1& right)
    noexcept
{
  return left.ordinal == right.ordinal && left.site == right.site &&
      SameContext(left.context, right.context) && left.bytes == right.bytes &&
      left.alignment == right.alignment && left.capacity == right.capacity &&
      left.storage_token == right.storage_token &&
      left.generation == right.generation &&
      left.allocation == right.allocation && left.success == right.success;
}

bool SameTransferLedger(
    const scalar::cuda_w8a::CudaTransferLedgerV1& left,
    const scalar::cuda_w8a::CudaTransferLedgerV1& right) noexcept
{
  if (left.schema != right.schema || left.capacity != right.capacity ||
      !oracle::SameTransferTotals(left.totals, right.totals) ||
      left.count != right.count ||
      left.observed_count != right.observed_count ||
      left.dropped_count != right.dropped_count ||
      left.ordered_digest != right.ordered_digest ||
      left.overflow != right.overflow)
  {
    return false;
  }
  for (std::size_t index = 0; index < left.count; ++index)
    if (!SameTransferEvent(left.events[index], right.events[index]))
      return false;
  return true;
}

bool TransferLedgerAdvanced(
    const scalar::cuda_w8a::CudaTransferLedgerV1& before,
    const scalar::cuda_w8a::CudaTransferLedgerV1& after) noexcept
{
  if (before.schema != after.schema || before.capacity != after.capacity ||
      after.count <= before.count ||
      after.observed_count <= before.observed_count ||
      after.dropped_count != before.dropped_count ||
      after.overflow != before.overflow ||
      after.totals.host_to_device_bytes <
          before.totals.host_to_device_bytes ||
      after.totals.device_to_host_bytes <
          before.totals.device_to_host_bytes ||
      after.totals.device_to_device_bytes <
          before.totals.device_to_device_bytes ||
      after.totals.mpi_staging_bytes < before.totals.mpi_staging_bytes ||
      after.totals.synchronization_events <
          before.totals.synchronization_events ||
      after.totals.event_count <= before.totals.event_count ||
      !before.totals.complete || !after.totals.complete)
  {
    return false;
  }
  for (std::size_t index = 0; index < before.count; ++index)
    if (!SameTransferEvent(before.events[index], after.events[index]))
      return false;
  return true;
}

bool SameAllocationLedger(
    const scalar::cuda_w8a::CudaAllocationLedgerV1& left,
    const scalar::cuda_w8a::CudaAllocationLedgerV1& right) noexcept
{
  if (left.schema != right.schema || left.capacity != right.capacity ||
      left.count != right.count ||
      left.observed_count != right.observed_count ||
      left.dropped_count != right.dropped_count ||
      left.ordered_digest != right.ordered_digest ||
      left.overflow != right.overflow)
  {
    return false;
  }
  for (std::size_t index = 0; index < left.count; ++index)
    if (!SameAllocationEvent(left.events[index], right.events[index]))
      return false;
  return true;
}

bool SameLaunches(const scalar::ScalarCudaLaunchLedger& left,
                  const scalar::ScalarCudaLaunchLedger& right) noexcept
{
  return left.assemble_launches == right.assemble_launches &&
      left.condense_launches == right.condense_launches &&
      left.recover_launches == right.recover_launches &&
      left.fallback_calls == right.fallback_calls &&
      left.last_compiled_architecture == right.last_compiled_architecture &&
      left.all_launches_completed == right.all_launches_completed;
}

bool LaunchesAdvanced(const scalar::ScalarCudaLaunchLedger& before,
                      const scalar::ScalarCudaLaunchLedger& after) noexcept
{
  return after.assemble_launches >= before.assemble_launches &&
      after.condense_launches >= before.condense_launches &&
      after.recover_launches >= before.recover_launches &&
      (after.assemble_launches > before.assemble_launches ||
       after.condense_launches > before.condense_launches ||
       after.recover_launches > before.recover_launches) &&
      after.fallback_calls == before.fallback_calls &&
      after.last_compiled_architecture ==
          before.last_compiled_architecture &&
      before.all_launches_completed && after.all_launches_completed;
}

bool SameAllocations(const scalar::ScalarCudaAllocationEvidence& left,
                     const scalar::ScalarCudaAllocationEvidence& right)
    noexcept
{
  return left.double_arena == right.double_arena &&
      left.integer_arena == right.integer_arena &&
      left.true_matrix == right.true_matrix && left.gmres == right.gmres &&
      left.ilu == right.ilu &&
      left.baseline_observed_events == right.baseline_observed_events &&
      left.current_observed_events == right.current_observed_events &&
      left.baseline_owned_calls == right.baseline_owned_calls &&
      left.current_owned_calls == right.current_owned_calls &&
      left.baseline_deallocations == right.baseline_deallocations &&
      left.current_deallocations == right.current_deallocations &&
      left.baseline_mfem_high_water == right.baseline_mfem_high_water &&
      left.current_mfem_high_water == right.current_mfem_high_water &&
      left.baseline_hypre_high_water == right.baseline_hypre_high_water &&
      left.current_hypre_high_water == right.current_hypre_high_water &&
      left.mfem_high_water == right.mfem_high_water &&
      left.hypre_high_water == right.hypre_high_water &&
      left.warm_plateau == right.warm_plateau &&
      left.zero_post_baseline_observed_growth ==
          right.zero_post_baseline_observed_growth &&
      left.zero_post_baseline_owned_allocation ==
          right.zero_post_baseline_owned_allocation &&
      left.zero_post_baseline_owned_deallocation ==
          right.zero_post_baseline_owned_deallocation;
}

bool SameReuse(const scalar::ScalarCudaReuseEvidence& left,
               const scalar::ScalarCudaReuseEvidence& right) noexcept
{
  return left.owner == right.owner && left.topology == right.topology &&
      left.workspace == right.workspace &&
      left.interior_slots == right.interior_slots &&
      left.trace_slots == right.trace_slots &&
      left.topology_generation == right.topology_generation &&
      left.storage_generation == right.storage_generation &&
      left.factor_generation == right.factor_generation &&
      left.solver_generation == right.solver_generation &&
      left.identity_hash == right.identity_hash &&
      left.host_topology_hash == right.host_topology_hash &&
      left.device_topology_hash == right.device_topology_hash &&
      left.host_quadrature_hash == right.host_quadrature_hash &&
      left.device_quadrature_hash == right.device_quadrature_hash &&
      left.device_topology_matches == right.device_topology_matches &&
      left.unchanged == right.unchanged;
}

bool RollbackReuseAdvanced(const scalar::ScalarCudaReuseEvidence& before,
                           const scalar::ScalarCudaReuseEvidence& after)
    noexcept
{
  return before.owner == after.owner &&
      before.topology == after.topology &&
      before.workspace == after.workspace &&
      before.interior_slots == after.interior_slots &&
      before.trace_slots == after.trace_slots &&
      before.topology_generation == after.topology_generation &&
      before.storage_generation == after.storage_generation &&
      after.factor_generation >= before.factor_generation &&
      after.solver_generation >= before.solver_generation &&
      before.identity_hash == after.identity_hash &&
      before.host_topology_hash == after.host_topology_hash &&
      before.device_topology_hash == after.device_topology_hash &&
      before.host_quadrature_hash == after.host_quadrature_hash &&
      before.device_quadrature_hash == after.device_quadrature_hash &&
      before.device_topology_matches == after.device_topology_matches &&
      before.unchanged == after.unchanged;
}

bool SameResidency(const scalar::ScalarCudaResidencyEvidence& left,
                   const scalar::ScalarCudaResidencyEvidence& right) noexcept
{
  return left.mfem_cuda_device == right.mfem_cuda_device &&
      left.hypre_cuda == right.hypre_cuda &&
      left.umpire_device_pool == right.umpire_device_pool &&
      left.state_device_resident == right.state_device_resident &&
      left.topology_device_resident == right.topology_device_resident &&
      left.workspace_device_resident == right.workspace_device_resident &&
      left.trace_solve_device_resident ==
          right.trace_solve_device_resident &&
      left.host_numerical_fallback == right.host_numerical_fallback;
}

bool RestorePhysicalEvidenceAdvanced(
    const scalar::ScalarCudaEvidence& before,
    const scalar::ScalarCudaEvidence& after) noexcept
{
  return before.trace_restart == after.trace_restart &&
      before.trace_maximum_iterations == after.trace_maximum_iterations &&
      before.trace_relative_tolerance == after.trace_relative_tolerance &&
      before.trace_absolute_tolerance == after.trace_absolute_tolerance &&
      before.ilu_type == after.ilu_type &&
      before.ilu_level_of_fill == after.ilu_level_of_fill &&
      before.ilu_local_reordering == after.ilu_local_reordering &&
      SameResidency(before.residency, after.residency) &&
      SameAllocations(before.allocations, after.allocations) &&
      RollbackReuseAdvanced(before.reuse, after.reuse) &&
      LaunchesAdvanced(before.launches, after.launches) &&
      TransferLedgerAdvanced(before.transfers, after.transfers) &&
      SameAllocationLedger(before.allocation_ledger,
                           after.allocation_ledger);
}

bool SameNewtonEvidence(const scalar::ScalarCudaNewtonEvidence& left,
                        const scalar::ScalarCudaNewtonEvidence& right)
    noexcept
{
  return left.iteration == right.iteration &&
      left.entry_residual_norm == right.entry_residual_norm &&
      left.accepted_residual_norm == right.accepted_residual_norm &&
      left.accepted_line_search_index == right.accepted_line_search_index &&
      left.trace_iterations == right.trace_iterations &&
      left.trace_true_residual_norm == right.trace_true_residual_norm &&
      left.trace_right_hand_side_norm == right.trace_right_hand_side_norm &&
      left.maximum_local_application_residual ==
          right.maximum_local_application_residual &&
      left.trial_residual_norms == right.trial_residual_norms &&
      left.trial_admissible == right.trial_admissible &&
      left.trial_armijo == right.trial_armijo;
}

bool SameAttemptEvidence(const scalar::ScalarCudaAttemptEvidence& left,
                         const scalar::ScalarCudaAttemptEvidence& right)
    noexcept
{
  if (left.status != right.status ||
      left.failure_cause != right.failure_cause ||
      left.injection != right.injection ||
      left.target_step != right.target_step ||
      left.retry_index != right.retry_index ||
      left.attempted_newton_iterations !=
          right.attempted_newton_iterations ||
      left.newton_iterations != right.newton_iterations ||
      left.dt != right.dt ||
      left.entry_residual_norm != right.entry_residual_norm ||
      left.final_residual_norm != right.final_residual_norm ||
      left.nonlinear_residual_bound != right.nonlinear_residual_bound ||
      left.maximum_conservation_error !=
          right.maximum_conservation_error ||
      left.minimum_density != right.minimum_density ||
      left.minimum_pressure != right.minimum_pressure ||
      left.minimum_speed_margin != right.minimum_speed_margin ||
      !SameCoordinates(left.committed_coordinates,
                       right.committed_coordinates) ||
      !SameCoordinates(left.candidate_coordinates,
                       right.candidate_coordinates) ||
      !oracle::SameHashes(left.entry_hashes, right.entry_hashes) ||
      !oracle::SameHashes(left.candidate_hashes, right.candidate_hashes))
  {
    return false;
  }
  for (std::size_t index = 0; index < left.newton.size(); ++index)
    if (!SameNewtonEvidence(left.newton[index], right.newton[index]))
      return false;
  return true;
}

bool SameStepEvidence(const scalar::ScalarCudaStepEvidence& left,
                      const scalar::ScalarCudaStepEvidence& right) noexcept
{
  return left.accepted == right.accepted && left.step == right.step &&
      left.time == right.time && left.generation == right.generation &&
      left.attempt_index == right.attempt_index &&
      left.committed_integrals == right.committed_integrals &&
      left.accepted_integrals == right.accepted_integrals &&
      oracle::SameHashes(left.accepted_hashes, right.accepted_hashes);
}

bool SameLogicalEvidence(const scalar::ScalarCudaEvidence& left,
                         const scalar::ScalarCudaEvidence& right) noexcept
{
  if (left.status != right.status ||
      left.first_failure != right.first_failure ||
      left.accepted_steps != right.accepted_steps ||
      left.attempted_steps != right.attempted_steps ||
      left.retry_attempts != right.retry_attempts ||
      left.attempt_count != right.attempt_count ||
      left.step_evidence_count != right.step_evidence_count ||
      left.trace_restart != right.trace_restart ||
      left.trace_maximum_iterations != right.trace_maximum_iterations ||
      left.trace_relative_tolerance != right.trace_relative_tolerance ||
      left.trace_absolute_tolerance != right.trace_absolute_tolerance ||
      left.ilu_type != right.ilu_type ||
      left.ilu_level_of_fill != right.ilu_level_of_fill ||
      left.ilu_local_reordering != right.ilu_local_reordering ||
      !SameCoordinates(left.coordinates, right.coordinates))
  {
    return false;
  }
  for (std::size_t index = 0; index < left.attempts.size(); ++index)
    if (!SameAttemptEvidence(left.attempts[index], right.attempts[index]))
      return false;
  for (std::size_t index = 0; index < left.steps.size(); ++index)
    if (!SameStepEvidence(left.steps[index], right.steps[index])) return false;
  return true;
}

bool DefaultLogicalArrays(const scalar::ScalarCudaEvidence& evidence,
                          const std::size_t first_attempt = 0,
                          const std::size_t first_step = 0) noexcept
{
  const scalar::ScalarCudaAttemptEvidence empty_attempt{};
  const scalar::ScalarCudaStepEvidence empty_step{};
  for (std::size_t index = first_attempt; index < evidence.attempts.size();
       ++index)
    if (!SameAttemptEvidence(evidence.attempts[index], empty_attempt))
      return false;
  for (std::size_t index = first_step; index < evidence.steps.size(); ++index)
    if (!SameStepEvidence(evidence.steps[index], empty_step)) return false;
  return true;
}

bool ValidateLogicalRebase(const scalar::ScalarCudaEvidence& evidence,
                           const int global_step) noexcept
{
  const ::dg::hdg::runtime::StateCoordinates expected{
      static_cast<double>(global_step) * scalar::fixed_dt,
      static_cast<std::uint64_t>(global_step),
      static_cast<std::uint64_t>(global_step),
      static_cast<std::uint64_t>(global_step), scalar::fixed_dt};
  return evidence.status == scalar::ScalarCudaStatus::ready &&
      evidence.first_failure == scalar::ScalarCudaStatus::uninitialized &&
      evidence.accepted_steps == global_step &&
      evidence.attempted_steps == 0 && evidence.retry_attempts == 0 &&
      evidence.attempt_count == 0 && evidence.step_evidence_count == 0 &&
      SameCoordinates(evidence.coordinates, expected) &&
      DefaultLogicalArrays(evidence);
}

bool ValidateTerminalRestartHistory(
    const scalar::ScalarCudaEvidence& evidence) noexcept
{
  constexpr int local_steps =
      scalar::persistent_required_steps - 10;
  if (evidence.first_failure != scalar::ScalarCudaStatus::uninitialized ||
      evidence.accepted_steps != scalar::persistent_required_steps ||
      evidence.attempted_steps != local_steps ||
      evidence.retry_attempts != 0 || evidence.attempt_count != local_steps ||
      evidence.step_evidence_count != local_steps ||
      evidence.coordinates.step != scalar::persistent_required_steps ||
      evidence.coordinates.generation != scalar::persistent_required_steps ||
      evidence.coordinates.cursor != scalar::persistent_required_steps ||
      evidence.coordinates.time !=
          scalar::persistent_required_steps * scalar::fixed_dt ||
      evidence.coordinates.dt != scalar::fixed_dt)
  {
    return false;
  }
  for (int local = 0; local < local_steps; ++local)
  {
    const int global = 11 + local;
    const auto& attempt =
        evidence.attempts[static_cast<std::size_t>(local)];
    const auto& step = evidence.steps[static_cast<std::size_t>(local)];
    if (attempt.status != scalar::ScalarCudaStatus::accepted ||
        attempt.failure_cause != scalar::ScalarCudaFailureCause::none ||
        attempt.injection != scalar::FailureInjection::none ||
        attempt.target_step != global || attempt.retry_index != 0 ||
        attempt.committed_coordinates.step !=
            static_cast<std::uint64_t>(global - 1) ||
        attempt.candidate_coordinates.step !=
            static_cast<std::uint64_t>(global) ||
        !step.accepted || step.step != global ||
        step.generation != static_cast<std::uint64_t>(global) ||
        step.attempt_index != local)
    {
      return false;
    }
  }
  return DefaultLogicalArrays(
      evidence, static_cast<std::size_t>(local_steps),
      static_cast<std::size_t>(local_steps));
}

bool SameCheckpoint(const scalar::ScalarCudaCheckpoint& left_checkpoint,
                    const scalar::ScalarCudaCheckpoint& right_checkpoint)
    noexcept
{
  const auto& left = left_checkpoint.descriptor;
  const auto& right = right_checkpoint.descriptor;
  return left.schema_version == right.schema_version &&
      left.successful_commit == right.successful_commit &&
      left.restart_mode == right.restart_mode && left.tuple == right.tuple &&
      left.mfem_pin == right.mfem_pin &&
      left.mesh_hash == right.mesh_hash &&
      left.partition_hash == right.partition_hash &&
      left.topology_hash == right.topology_hash &&
      left.rank_count == right.rank_count &&
      left.polynomial_order == right.polynomial_order &&
      left.layout == right.layout && left.precision == right.precision &&
      left.cursor == right.cursor && left.dt == right.dt &&
      SameCoordinates(left.coordinates, right.coordinates) &&
      left.coordinate_fingerprint == right.coordinate_fingerprint &&
      left.interior_hash == right.interior_hash &&
      left.trace_hash == right.trace_hash &&
      left_checkpoint.projection == right_checkpoint.projection &&
      left_checkpoint.tuple == right_checkpoint.tuple &&
      left_checkpoint.scenario == right_checkpoint.scenario &&
      left_checkpoint.layout == right_checkpoint.layout &&
      left_checkpoint.interior_component_slabs ==
          right_checkpoint.interior_component_slabs &&
      left_checkpoint.trace_component_slabs ==
          right_checkpoint.trace_component_slabs &&
      oracle::SameHashes(left_checkpoint.raw_field_hashes,
                         right_checkpoint.raw_field_hashes) &&
      left_checkpoint.canonical_manifest_hash ==
          right_checkpoint.canonical_manifest_hash;
}

bool DefaultCheckpoint(const scalar::ScalarCudaCheckpoint& checkpoint)
    noexcept
{
  const scalar::ScalarCudaCheckpoint empty{};
  return SameCheckpoint(checkpoint, empty);
}

std::unique_ptr<cpu::PersistentSc25ScalarIgrCpuFoundation> CreateCpu(
    const int elements,
    cpu::FoundationStatus* const observed_status = nullptr)
{
  cpu::FoundationConfig config{};
  config.polynomial_degree = 1;
  config.elements_per_direction = elements;
  std::unique_ptr<cpu::PersistentSc25ScalarIgrCpuFoundation> result;
  const cpu::FoundationStatus status =
      cpu::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, cpu::CpuSerialTag{}, config, result);
  if (observed_status != nullptr) { *observed_status = status; }
  if (status != cpu::FoundationStatus::ready)
  {
    return nullptr;
  }
  return result;
}

std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank1> CreateCuda(
    const int elements, const bool coupled_fixture = false,
    scalar::ScalarCudaStatus* const observed_status = nullptr)
{
  scalar::ScalarCudaConfig config{};
  config.polynomial_degree = 1;
  config.elements_per_direction = elements;
  config.requested_device = 0;
  config.coupled_operator_fixture = coupled_fixture;
  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank1> result;
  const scalar::ScalarCudaStatus status =
      scalar::PersistentSc25ScalarIgrCudaRank1::Create(
          MPI_COMM_WORLD, config, result);
  if (observed_status != nullptr) { *observed_status = status; }
  if (status != scalar::ScalarCudaStatus::ready)
  {
    return nullptr;
  }
  return result;
}

bool CopyPair(cpu::PersistentSc25ScalarIgrCpuFoundation& cpu_run,
              scalar::PersistentSc25ScalarIgrCudaRank1& cuda_run,
              scalar::PublishedState& cpu_state,
              scalar::PublishedState& cuda_state,
              oracle::ComparisonEvidence& evidence)
{
  return cpu_run.CopyPublishedTrajectoryState(cpu_state) &&
      cuda_run.CopyPublishedTrajectoryState(cuda_state) &&
      oracle::CompareState(cpu_state, cuda_state, comparison_tolerance,
                           evidence);
}

bool ValidateAttemptStateEvidence(
    const scalar::ScalarCudaAttemptEvidence& attempt,
    const scalar::ScalarCudaStepEvidence& accepted_step,
    const int target_step, const scalar::PublishedState& entry,
    const scalar::PublishedState& accepted)
{
  const scalar::StateHashes entry_hashes = oracle::HashState(entry);
  const scalar::StateHashes accepted_hashes = oracle::HashState(accepted);
  const auto exact_coordinates = [](
      const ::dg::hdg::runtime::StateCoordinates& value, const int step)
  {
    const double expected_dt = step == 0 ? 0.0 : scalar::fixed_dt;
    return value.step == static_cast<std::uint64_t>(step) &&
        value.generation == static_cast<std::uint64_t>(step) &&
        value.cursor == static_cast<std::uint64_t>(step) &&
        value.dt == expected_dt &&
        value.time == static_cast<double>(step) * scalar::fixed_dt;
  };
  bool armijo_exact = true;
  for (const auto& newton : attempt.newton)
  {
    if (newton.iteration < 0) { continue; }
    for (int line = 0; line < scalar::line_search_trials; ++line)
    {
      const std::size_t index = static_cast<std::size_t>(line);
      if (newton.trial_admissible[index] == 0)
      {
        armijo_exact = armijo_exact && newton.trial_armijo[index] == 0;
        continue;
      }
      const double theta = std::ldexp(1.0, -line);
      const bool recomputed = std::isfinite(newton.trial_residual_norms[index]) &&
          newton.trial_residual_norms[index] <=
              (1.0 - scalar::armijo_coefficient * theta) *
                  newton.entry_residual_norm;
      armijo_exact = armijo_exact &&
          newton.trial_armijo[index] == (recomputed ? 1 : 0);
    }
  }
  const scalar::ScalarCudaStatus expected_status = attempt.retry_index == 0
      ? scalar::ScalarCudaStatus::accepted
      : scalar::ScalarCudaStatus::accepted_after_one_retry;
  return attempt.status == expected_status &&
      attempt.failure_cause == scalar::ScalarCudaFailureCause::none &&
      attempt.injection == scalar::FailureInjection::none &&
      attempt.target_step == target_step && attempt.dt == scalar::fixed_dt &&
      exact_coordinates(attempt.committed_coordinates, target_step - 1) &&
      exact_coordinates(attempt.candidate_coordinates, target_step) &&
      oracle::NonzeroHashes(attempt.entry_hashes) &&
      oracle::NonzeroHashes(attempt.candidate_hashes) &&
      oracle::SameHashes(attempt.entry_hashes, entry_hashes) &&
      oracle::SameHashes(attempt.candidate_hashes, accepted_hashes) &&
      accepted_step.accepted && accepted_step.step == target_step &&
      accepted_step.time == static_cast<double>(target_step) *
          scalar::fixed_dt &&
      accepted_step.generation == static_cast<std::uint64_t>(target_step) &&
      oracle::SameHashes(accepted_step.accepted_hashes, accepted_hashes) &&
      armijo_exact;
}

std::pair<scalar::ScalarCudaStatus, scalar::ScalarCudaFailureCause>
ExpectedInjectedFailure(const scalar::FailureInjection injection) noexcept
{
  if (injection == scalar::FailureInjection::local_factor)
    return {scalar::ScalarCudaStatus::injected_local_factor_failure,
            scalar::ScalarCudaFailureCause::injected_local_factor};
  if (injection == scalar::FailureInjection::trace_linear)
    return {scalar::ScalarCudaStatus::injected_trace_linear_failure,
            scalar::ScalarCudaFailureCause::injected_trace_linear};
  if (injection == scalar::FailureInjection::admissibility)
    return {scalar::ScalarCudaStatus::injected_admissibility_failure,
            scalar::ScalarCudaFailureCause::injected_admissibility};
  return {scalar::ScalarCudaStatus::injected_newton_iteration_limit,
          scalar::ScalarCudaFailureCause::injected_newton_limit};
}

scalar::ScalarCudaStatus CudaStatus(const cpu::TrajectoryStatus status) noexcept
{
  if (status == cpu::TrajectoryStatus::HdgSc25Accepted)
    return scalar::ScalarCudaStatus::accepted;
  if (status == cpu::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry)
    return scalar::ScalarCudaStatus::accepted_after_one_retry;
  if (status == cpu::TrajectoryStatus::HdgSc25InjectedLocalFactorFailure)
    return scalar::ScalarCudaStatus::injected_local_factor_failure;
  if (status == cpu::TrajectoryStatus::HdgSc25InjectedTraceLinearFailure)
    return scalar::ScalarCudaStatus::injected_trace_linear_failure;
  if (status == cpu::TrajectoryStatus::HdgSc25InjectedAdmissibilityFailure)
    return scalar::ScalarCudaStatus::injected_admissibility_failure;
  if (status == cpu::TrajectoryStatus::HdgSc25InjectedNewtonIterationLimit)
    return scalar::ScalarCudaStatus::injected_newton_iteration_limit;
  return scalar::ScalarCudaStatus::uninitialized;
}

bool CloseMetric(const double left, const double right) noexcept
{
  if (std::isinf(left) || std::isinf(right)) { return left == right; }
  return std::isfinite(left) && std::isfinite(right) &&
      std::abs(left - right) <=
          comparison_tolerance * std::max({1.0, std::abs(left),
                                            std::abs(right)});
}

template <typename AttemptEvidence>
bool ValidAttemptAdmissibility(
    const AttemptEvidence& attempt) noexcept
{
  return std::isfinite(attempt.minimum_density) &&
      attempt.minimum_density >= 1.0e-8 &&
      std::isfinite(attempt.minimum_pressure) &&
      attempt.minimum_pressure >= 1.0e-8 &&
      std::isfinite(attempt.minimum_speed_margin) &&
      attempt.minimum_speed_margin > 0.0;
}

bool ValidateAttemptParity(
    const scalar::ScalarCudaAttemptEvidence& cuda_attempt,
    const cpu::TrajectoryAttemptEvidence& cpu_attempt)
{
  if (cuda_attempt.status != CudaStatus(cpu_attempt.status) ||
      cuda_attempt.injection != static_cast<scalar::FailureInjection>(
          cpu_attempt.injection) ||
      cuda_attempt.target_step != cpu_attempt.target_step ||
      cuda_attempt.retry_index != cpu_attempt.retry_index ||
      cuda_attempt.dt != cpu_attempt.dt ||
      cuda_attempt.attempted_newton_iterations !=
          cpu_attempt.attempted_newton_iterations ||
      cuda_attempt.newton_iterations != cpu_attempt.newton_iterations ||
      !CloseMetric(cuda_attempt.entry_residual_norm,
                   cpu_attempt.entry_residual_norm) ||
      !CloseMetric(cuda_attempt.final_residual_norm,
                   cpu_attempt.final_residual_norm) ||
      !CloseMetric(cuda_attempt.nonlinear_residual_bound,
                   cpu_attempt.nonlinear_residual_bound) ||
      !CloseMetric(cuda_attempt.maximum_conservation_error,
                   cpu_attempt.maximum_conservation_error))
    return false;
  return !CudaAccepted(cuda_attempt.status) ||
      (ValidAttemptAdmissibility(cuda_attempt) &&
       ValidAttemptAdmissibility(cpu_attempt) &&
       cuda_attempt.final_residual_norm <=
           1.0e-12 + 1.0e-10 * cuda_attempt.entry_residual_norm);
}

bool ValidateCudaStateScience(
    cpu::PersistentSc25ScalarIgrCpuFoundation& cpu_run,
    const scalar::PublishedState& state,
    const std::array<double, scalar::state_components>* preceding_q12,
    const std::array<double, scalar::state_components>* preceding_q14,
    const scalar::ScalarCudaStepEvidence* cuda_step,
    const scalar::ScalarCudaAttemptEvidence* cuda_attempt,
    oracle::StateQuadratureAudit& q12,
    oracle::StateQuadratureAudit& q14)
{
  cpu::AuxiliaryResidualMetrics auxiliary_q12{}, auxiliary_q14{};
  if (!oracle::AuditPublishedState(
          state, cpu::coupled_fixture_quadrature_order, q12) ||
      !oracle::AuditPublishedState(
          state, cpu::coupled_fixture_repeat_quadrature_order, q14) ||
      q12.volume_points <= 0 || q12.side_points <= 0 ||
      q12.trace_points <= 0 || q14.volume_points <= 0 ||
      q14.side_points <= 0 || q14.trace_points <= 0 ||
      !cpu_oracle::EvaluateIndependentAuxiliaryResidualOracle(
          cpu_run, state, cpu::coupled_fixture_quadrature_order,
          auxiliary_q12) ||
      !cpu_oracle::EvaluateIndependentAuxiliaryResidualOracle(
          cpu_run, state, cpu::coupled_fixture_repeat_quadrature_order,
          auxiliary_q14) ||
      auxiliary_q12.normalized_maximum > 1.0e-10 ||
      auxiliary_q14.normalized_maximum > 1.0e-10)
    return false;
  const std::array<double, 3> normalized_q12{{
      auxiliary_q12.r_z_norm / auxiliary_q12.r_z_scale,
      auxiliary_q12.r_sigma_norm / auxiliary_q12.r_sigma_scale,
      auxiliary_q12.g_sigma_norm / auxiliary_q12.g_sigma_scale}};
  const std::array<double, 3> normalized_q14{{
      auxiliary_q14.r_z_norm / auxiliary_q14.r_z_scale,
      auxiliary_q14.r_sigma_norm / auxiliary_q14.r_sigma_scale,
      auxiliary_q14.g_sigma_norm / auxiliary_q14.g_sigma_scale}};
  for (std::size_t term = 0; term < normalized_q12.size(); ++term)
    if (std::abs(normalized_q14[term] - normalized_q12[term]) > 1.0e-11)
      return false;
  double maximum_conservation = 0.0;
  for (int component = 0; component < scalar::state_components; ++component)
  {
    const std::size_t index = static_cast<std::size_t>(component);
    if (std::abs(q14.integrals[index] - q12.integrals[index]) /
            std::max(1.0, std::abs(q12.integrals[index])) > 1.0e-11)
      return false;
    if (cuda_step != nullptr)
    {
      if (!CloseMetric(q12.integrals[index],
                       cuda_step->accepted_integrals[index]))
        return false;
    }
    if (preceding_q12 != nullptr && preceding_q14 != nullptr)
    {
      const double q12_error = std::abs(
          q12.integrals[index] - (*preceding_q12)[index]) /
          std::max(1.0, std::abs((*preceding_q12)[index]));
      const double q14_error = std::abs(
          q14.integrals[index] - (*preceding_q14)[index]) /
          std::max(1.0, std::abs((*preceding_q14)[index]));
      if (q12_error > 2.0e-10 || q14_error > 2.0e-10)
        return false;
      maximum_conservation = std::max(maximum_conservation, q12_error);
    }
  }
  return cuda_step == nullptr ||
      (cuda_step->accepted && cuda_attempt != nullptr &&
       CloseMetric(maximum_conservation,
                   cuda_attempt->maximum_conservation_error));
}

bool ValidateCopiedEvidenceStructure(
    const scalar::ScalarCudaEvidence& evidence) noexcept
{
  if (evidence.accepted_steps != scalar::persistent_required_steps ||
      evidence.attempted_steps != scalar::persistent_required_steps ||
      evidence.attempt_count != scalar::persistent_required_steps ||
      evidence.step_evidence_count != scalar::persistent_required_steps)
    return false;
  for (int step_index = 0; step_index < scalar::persistent_required_steps;
       ++step_index)
  {
    const auto& step = evidence.steps[static_cast<std::size_t>(step_index)];
    if (!step.accepted || step.step != step_index + 1 ||
        step.attempt_index != step_index)
      return false;
    const auto& attempt =
        evidence.attempts[static_cast<std::size_t>(step.attempt_index)];
    if (attempt.status != scalar::ScalarCudaStatus::accepted ||
        attempt.failure_cause != scalar::ScalarCudaFailureCause::none ||
        attempt.injection != scalar::FailureInjection::none ||
        attempt.target_step != step.step ||
        !oracle::SameHashes(attempt.candidate_hashes,
                            step.accepted_hashes) ||
        !oracle::NonzeroHashes(attempt.entry_hashes) ||
        !oracle::NonzeroHashes(attempt.candidate_hashes) ||
        !(attempt.final_residual_norm <=
          1.0e-12 + 1.0e-10 * attempt.entry_residual_norm))
      return false;
    if (step_index > 0 &&
        !oracle::SameHashes(
            attempt.entry_hashes,
            evidence.steps[static_cast<std::size_t>(step_index - 1)]
                .accepted_hashes))
      return false;
    double conservation = 0.0;
    for (int component = 0; component < scalar::state_components; ++component)
    {
      const std::size_t index = static_cast<std::size_t>(component);
      if (!std::isfinite(step.committed_integrals[index]) ||
          !std::isfinite(step.accepted_integrals[index]))
        return false;
      conservation = std::max(
          conservation,
          std::abs(step.accepted_integrals[index] -
                   step.committed_integrals[index]) /
              std::max(1.0, std::abs(step.committed_integrals[index])));
      if (step_index > 0 &&
          step.committed_integrals[index] !=
              evidence.steps[static_cast<std::size_t>(step_index - 1)]
                  .accepted_integrals[index])
        return false;
    }
    if (!CloseMetric(conservation, attempt.maximum_conservation_error))
      return false;
    for (int iteration = 0; iteration < attempt.newton_iterations; ++iteration)
    {
      const auto& newton =
          attempt.newton[static_cast<std::size_t>(iteration)];
      if (newton.iteration != iteration ||
          newton.accepted_line_search_index < 0 ||
          newton.accepted_line_search_index >= scalar::line_search_trials)
        return false;
      int first_armijo = -1;
      for (int line = 0; line <= newton.accepted_line_search_index; ++line)
      {
        const std::size_t index = static_cast<std::size_t>(line);
        const double theta = std::ldexp(1.0, -line);
        const bool recomputed = newton.trial_admissible[index] != 0 &&
            std::isfinite(newton.trial_residual_norms[index]) &&
            newton.trial_residual_norms[index] <=
                (1.0 - scalar::armijo_coefficient * theta) *
                    newton.entry_residual_norm;
        if (newton.trial_armijo[index] != (recomputed ? 1 : 0))
          return false;
        if (first_armijo < 0 && recomputed) { first_armijo = line; }
      }
      if (first_armijo != newton.accepted_line_search_index ||
          !CloseMetric(
              newton.accepted_residual_norm,
              newton.trial_residual_norms[static_cast<std::size_t>(
                  newton.accepted_line_search_index)]))
        return false;
    }
  }
  return true;
}

bool RunCopiedEvidenceMutationNegatives(
    const scalar::ScalarCudaEvidence& evidence)
{
  if (!ValidateCopiedEvidenceStructure(evidence)) { return false; }
  int newton_attempt = -1;
  for (int index = 0; index < evidence.attempt_count; ++index)
  {
    if (evidence.attempts[static_cast<std::size_t>(index)]
            .newton_iterations > 0 &&
        evidence.attempts[static_cast<std::size_t>(index)]
                .newton[0].accepted_line_search_index >= 0)
    {
      newton_attempt = index;
      break;
    }
  }
  if (newton_attempt < 0) { return false; }
  auto mutated = std::make_unique<scalar::ScalarCudaEvidence>(evidence);
  mutated->steps[0].accepted_hashes.u.bytes[0] ^= 0x01U;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  mutated->steps[0].accepted_integrals[0] += 1.0;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  mutated->attempts[0].final_residual_norm += 1.0;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  auto& newton = mutated->attempts[
      static_cast<std::size_t>(newton_attempt)].newton[0];
  newton.trial_armijo[static_cast<std::size_t>(
      newton.accepted_line_search_index)] ^= 0x01U;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  ++mutated->attempts[static_cast<std::size_t>(newton_attempt)]
        .newton[0].accepted_line_search_index;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  mutated->attempts[0].failure_cause =
      scalar::ScalarCudaFailureCause::publication;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  mutated->attempts[0].status = scalar::ScalarCudaStatus::publication_failure;
  if (ValidateCopiedEvidenceStructure(*mutated)) { return false; }
  *mutated = evidence;
  mutated->steps[0].attempt_index = 1;
  return !ValidateCopiedEvidenceStructure(*mutated);
}

bool RunC1()
{
  cpu::FoundationStatus cpu_creation_status =
      cpu::FoundationStatus::uninitialized;
  scalar::ScalarCudaStatus cuda_creation_status =
      scalar::ScalarCudaStatus::uninitialized;
  auto cpu_run = CreateCpu(3, &cpu_creation_status);
  auto cuda_run = CreateCuda(3, true, &cuda_creation_status);
  if (!cpu_run || !cuda_run)
  {
    std::cerr << "C1 CPU FoundationStatus="
              << static_cast<int>(cpu_creation_status) << '\n';
    std::cerr << "C1 CUDA ScalarCudaStatus="
              << static_cast<int>(cuda_creation_status) << '\n';
    std::cerr << "C1 creation failed\n";
    return false;
  }
  cpu_oracle::IndependentC1Evidence independent{};
  if (!cpu_oracle::RunIndependentC1Oracle(*cpu_run, independent))
  {
    for (const std::string& failure : independent.failures)
    {
      std::cerr << "C1 independent oracle: " << failure << '\n';
    }
    return false;
  }
  scalar::CoupledOperatorSnapshot cpu_snapshot{};
  scalar::CoupledOperatorSnapshot cuda_snapshot{};
  oracle::ComparisonEvidence comparison{};
  const bool exported = cpu_run->RunCoupledOperatorFixture(cpu_snapshot) &&
      cuda_run->ExportIndependentOperator(cuda_snapshot);
  const bool compared = exported && oracle::CompareOperator(
      cpu_snapshot, cuda_snapshot, comparison_tolerance, comparison);
  const auto& evidence = cuda_run->operator_evidence();
  const bool quadrature =
      independent.quadrature_repeat_difference <= comparison_tolerance &&
      cpu_snapshot.residual_quadrature_repeat_relative_l2 <=
          comparison_tolerance &&
      cpu_snapshot.jacobian_quadrature_repeat_relative_l2 <=
          comparison_tolerance;
  const bool solver = cuda_snapshot.solver.restart_dimension == 50 &&
      cuda_snapshot.solver.maximum_iterations == 200 &&
      cuda_snapshot.solver.relative_tolerance == 1.0e-12 &&
      cuda_snapshot.solver.absolute_tolerance == 1.0e-14 &&
      cuda_snapshot.solver.ilu_type == 0 &&
      cuda_snapshot.solver.ilu_level_of_fill == 0 &&
      cuda_snapshot.solver.ilu_local_reordering == 1 &&
      cuda_snapshot.solver.converged == 1 &&
      cuda_snapshot.solver.unpreconditioned_residual_norm <=
          cuda_snapshot.solver.absolute_tolerance +
              cuda_snapshot.solver.relative_tolerance *
                  cuda_snapshot.solver.right_hand_side_norm;
  const bool device = evidence.elements == 9 &&
      evidence.full_unknowns == 432 &&
      evidence.true_trace_unknowns == 180 &&
      evidence.mfem_periodic_topology && evidence.device_assembly &&
      evidence.device_condensation && evidence.device_recovery &&
      evidence.hypre_device_trace_solve &&
      evidence.launches.assemble_launches > 0 &&
      evidence.launches.condense_launches > 0 &&
      evidence.launches.recover_launches > 0 &&
      evidence.launches.fallback_calls == 0 &&
      evidence.launches.last_compiled_architecture == 1200 &&
      evidence.launches.all_launches_completed;
  if (!compared)
  {
    for (const std::string& failure : comparison.failures)
    {
      std::cerr << "C1 CUDA comparison: " << failure << '\n';
    }
  }
  const bool pass = compared && quadrature && solver && device &&
      comparison.maximum_relative_l2 <= comparison_tolerance;
  if (pass)
  {
    std::cout << "PASS: HDG-EXP-SCALAR-DEVICE-01 C1 432x432 CPU/CUDA "
              << "max-relative-L2=" << comparison.maximum_relative_l2
              << '\n';
    std::cout << "SC25_CUDA_RANK1_C1_PASS\n";
  }
  return pass;
}

bool RunTransactionNegatives()
{
  const auto step0_evidence =
      std::make_unique<scalar::ScalarCudaEvidence>();
  const auto prospective =
      std::make_unique<scalar::ScalarCudaEvidence>();
  const auto finalized_prospective =
      std::make_unique<scalar::ScalarCudaEvidence>();
  const auto fail_phase = [](const std::string_view phase)
  {
    std::cerr << "C2 transaction-negative phase=" << phase
              << " pass=0\n";
    return false;
  };
  auto foreign = CreateCuda(8);
  auto owner = CreateCuda(8);
  if (!owner || !foreign) { return fail_phase("create"); }
  if (owner->InitializePersistentTrajectory() !=
      scalar::ScalarCudaStatus::ready)
  {
    return fail_phase("initialize");
  }
  scalar::PublishedState step0{};
  if (!owner->CopyPublishedTrajectoryState(step0))
  {
    return fail_phase("step0-copy");
  }
  *step0_evidence = owner->trajectory_evidence();
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction invalid;
  if (owner->InstallPersistentTrajectoryTransaction(invalid))
  {
    return fail_phase("invalid-token-install");
  }

  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction prepared;
  if (owner->PreparePersistentTrajectoryTransaction(prepared) !=
          scalar::ScalarCudaStatus::accepted ||
      !prepared.valid() || prepared.provisionally_installed())
  {
    return fail_phase("prepare");
  }
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction live(
      std::move(prepared));
  if (prepared.valid()) { return fail_phase("moved-from-token"); }
  if (owner->InstallPersistentTrajectoryTransaction(prepared))
  {
    return fail_phase("moved-from-install");
  }
  if (foreign->InstallPersistentTrajectoryTransaction(live))
  {
    return fail_phase("foreign-install");
  }
  if (!owner->InstallPersistentTrajectoryTransaction(live))
  {
    return fail_phase("owner-install");
  }
  if (owner->InstallPersistentTrajectoryTransaction(live))
  {
    return fail_phase("duplicate-install");
  }
  if (!live.provisionally_installed())
  {
    return fail_phase("installed-token-state");
  }
  scalar::PublishedState installed{};
  scalar::ScalarCudaCheckpoint forbidden{};
  scalar::ScalarCudaCheckpoint staged_checkpoint{};
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction concurrent;
  if (!owner->CopyPublishedTrajectoryState(installed))
  {
    return fail_phase("installed-state-copy");
  }
  if (!oracle::SameStateBytes(step0, installed) ||
      owner->trajectory_evidence().coordinates.step != 0 ||
      owner->trajectory_evidence().accepted_steps != 0)
  {
    return fail_phase("installed-state-view");
  }
  if (owner->PreparePersistentTrajectoryTransaction(concurrent) !=
          scalar::ScalarCudaStatus::publication_failure ||
      concurrent.valid())
  {
    return fail_phase("concurrent-prepare-rejection");
  }
  if (owner->CaptureTrajectoryCheckpoint(forbidden))
  {
    return fail_phase("installed-checkpoint-rejection");
  }
  if (!owner->StageInstalledTrajectoryTransaction(
          live, staged_checkpoint, *prospective))
  {
    return fail_phase("stage-installed");
  }
  if (staged_checkpoint.descriptor.coordinates.step != 1 ||
      prospective->accepted_steps != 1 ||
      prospective->attempted_steps != 1 ||
      prospective->coordinates.step != 1)
  {
    return fail_phase("staged-evidence");
  }
  if (!owner->RevertPersistentTrajectoryTransaction(std::move(live)))
  {
    return fail_phase("revert-installed");
  }
  if (live.valid()) { return fail_phase("reverted-token-state"); }
  if (owner->RevertPersistentTrajectoryTransaction(std::move(live)))
  {
    return fail_phase("duplicate-revert");
  }
  scalar::PublishedState reverted{};
  const scalar::ScalarCudaEvidence& reverted_evidence =
      owner->trajectory_evidence();
  const bool reverted_copied =
      owner->CopyPublishedTrajectoryState(reverted);
  const bool state_rollback =
      reverted_copied && oracle::SameStateBytes(step0, reverted);
  const bool logical_rollback =
      SameLogicalEvidence(*step0_evidence, reverted_evidence);
  const bool physical_rollback =
      RestorePhysicalEvidenceAdvanced(*step0_evidence, reverted_evidence);
  if (!state_rollback || !logical_rollback || !physical_rollback)
  {
    const auto& before = *step0_evidence;
    const auto& after = reverted_evidence;
    std::cerr << "C2 transaction-negative phase=rollback pass=0"
              << " state=" << state_rollback
              << " logical=" << logical_rollback
              << " physical=" << physical_rollback
              << " factor=" << before.reuse.factor_generation << "->"
              << after.reuse.factor_generation
              << " solver=" << before.reuse.solver_generation << "->"
              << after.reuse.solver_generation
              << " launches=" << before.launches.assemble_launches << ','
              << before.launches.condense_launches << ','
              << before.launches.recover_launches << ','
              << before.launches.fallback_calls << "->"
              << after.launches.assemble_launches << ','
              << after.launches.condense_launches << ','
              << after.launches.recover_launches << ','
              << after.launches.fallback_calls
              << " transfers=" << before.transfers.count << ','
              << before.transfers.observed_count << ','
              << before.transfers.dropped_count << ','
              << before.transfers.ordered_digest << ','
              << before.transfers.overflow << "->"
              << after.transfers.count << ','
              << after.transfers.observed_count << ','
              << after.transfers.dropped_count << ','
              << after.transfers.ordered_digest << ','
              << after.transfers.overflow
              << " allocations=" << before.allocation_ledger.count << ','
              << before.allocation_ledger.observed_count << ','
              << before.allocation_ledger.dropped_count << ','
              << before.allocation_ledger.ordered_digest << ','
              << before.allocation_ledger.overflow << "->"
              << after.allocation_ledger.count << ','
              << after.allocation_ledger.observed_count << ','
              << after.allocation_ledger.dropped_count << ','
              << after.allocation_ledger.ordered_digest << ','
              << after.allocation_ledger.overflow << '\n';
    return false;
  }

  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction aborted;
  if (owner->PreparePersistentTrajectoryTransaction(aborted) !=
          scalar::ScalarCudaStatus::accepted ||
      aborted.provisionally_installed() ||
      !owner->RevertPersistentTrajectoryTransaction(std::move(aborted)) ||
      aborted.valid() ||
      !SameLogicalEvidence(*step0_evidence, owner->trajectory_evidence()))
  {
    return fail_phase("prepared-token-abort");
  }

  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction finalized;
  scalar::ScalarCudaCheckpoint finalized_checkpoint{};
  if (owner->PreparePersistentTrajectoryTransaction(finalized) !=
          scalar::ScalarCudaStatus::accepted ||
      !owner->InstallPersistentTrajectoryTransaction(finalized) ||
      !owner->StageInstalledTrajectoryTransaction(
          finalized, finalized_checkpoint, *finalized_prospective) ||
      !owner->FinalizePersistentTrajectoryTransaction(std::move(finalized)) ||
      finalized.valid() ||
      owner->FinalizePersistentTrajectoryTransaction(std::move(finalized)) ||
      !SameLogicalEvidence(*finalized_prospective,
                           owner->trajectory_evidence()))
  {
    return fail_phase("finalize");
  }
  if (owner->trajectory_evidence().accepted_steps != 1)
  {
    return fail_phase("finalized-evidence");
  }
  return true;
}

bool RunMoveWithLiveTransaction()
{
  auto owner = CreateCuda(8);
  if (!owner || owner->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction live;
  if (owner->PreparePersistentTrajectoryTransaction(live) !=
          scalar::ScalarCudaStatus::accepted ||
      !live.valid() || live.provisionally_installed())
  {
    return false;
  }

  scalar::PersistentSc25ScalarIgrCudaRank1 moved(std::move(*owner));
  if (owner->InstallPersistentTrajectoryTransaction(live) || !live.valid() ||
      !moved.InstallPersistentTrajectoryTransaction(live) ||
      !live.provisionally_installed() ||
      owner->FinalizePersistentTrajectoryTransaction(std::move(live)) ||
      !live.valid() || !live.provisionally_installed() ||
      !moved.FinalizePersistentTrajectoryTransaction(std::move(live)) ||
      live.valid() || live.provisionally_installed())
  {
    return false;
  }
  const auto& evidence = moved.trajectory_evidence();
  return moved.status() == scalar::ScalarCudaStatus::ready &&
      evidence.accepted_steps == 1 && evidence.attempted_steps == 1 &&
      evidence.attempt_count == 1 && evidence.step_evidence_count == 1 &&
      evidence.coordinates.step == 1;
}

bool RunStaleTransactionAfterOwnerDestruction()
{
  auto expired_owner = CreateCuda(8);
  if (!expired_owner || expired_owner->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction stale;
  if (expired_owner->PreparePersistentTrajectoryTransaction(stale) !=
          scalar::ScalarCudaStatus::accepted ||
      !expired_owner->InstallPersistentTrajectoryTransaction(stale) ||
      !stale.valid() || !stale.provisionally_installed())
  {
    return false;
  }
  expired_owner.reset();

  auto replacement = CreateCuda(8);
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction current;
  if (!replacement || replacement->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready ||
      replacement->PreparePersistentTrajectoryTransaction(current) !=
          scalar::ScalarCudaStatus::accepted ||
      !replacement->InstallPersistentTrajectoryTransaction(current))
  {
    return false;
  }
  if (replacement->FinalizePersistentTrajectoryTransaction(
          std::move(stale)) ||
      !stale.valid() || !stale.provisionally_installed() ||
      replacement->RevertPersistentTrajectoryTransaction(
          std::move(stale)) ||
      !stale.valid() || !stale.provisionally_installed() ||
      !replacement->FinalizePersistentTrajectoryTransaction(
          std::move(current)) ||
      current.valid() || current.provisionally_installed())
  {
    return false;
  }
  const auto& evidence = replacement->trajectory_evidence();
  return replacement->status() == scalar::ScalarCudaStatus::ready &&
      evidence.accepted_steps == 1 && evidence.attempted_steps == 1 &&
      evidence.attempt_count == 1 && evidence.step_evidence_count == 1 &&
      evidence.coordinates.step == 1;
}

bool RunRevertPhysicalEvidence()
{
  auto owner = CreateCuda(8);
  if (!owner || owner->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  const auto committed = std::make_unique<scalar::ScalarCudaEvidence>(
      owner->trajectory_evidence());
  const auto transfers_before =
      std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(
          owner->transfer_ledger());
  const auto allocations_before = owner->allocation_ledger();
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction abandoned;
  if (owner->PreparePersistentTrajectoryTransaction(abandoned) !=
          scalar::ScalarCudaStatus::accepted ||
      !owner->InstallPersistentTrajectoryTransaction(abandoned))
  {
    return false;
  }
  const auto transfers_installed =
      std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(
          owner->transfer_ledger());
  const auto allocations_installed = owner->allocation_ledger();
  if (!owner->RevertPersistentTrajectoryTransaction(std::move(abandoned)) ||
      abandoned.valid())
  {
    return false;
  }

  const auto reverted = std::make_unique<scalar::ScalarCudaEvidence>(
      owner->trajectory_evidence());
  const auto transfers_reverted =
      std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(
          owner->transfer_ledger());
  const auto allocations_reverted = owner->allocation_ledger();
  const auto launch_count = [](const scalar::ScalarCudaLaunchLedger& value)
  {
    return value.assemble_launches + value.condense_launches +
        value.recover_launches;
  };
  const auto same_coordinates = [](
      const ::dg::hdg::runtime::StateCoordinates& left,
      const ::dg::hdg::runtime::StateCoordinates& right)
  {
    return left.time == right.time && left.step == right.step &&
        left.generation == right.generation && left.cursor == right.cursor &&
        left.dt == right.dt;
  };
  const bool logical_revert =
      owner->status() == scalar::ScalarCudaStatus::ready &&
      reverted->status == committed->status &&
      reverted->first_failure == committed->first_failure &&
      reverted->accepted_steps == committed->accepted_steps &&
      reverted->attempted_steps == committed->attempted_steps &&
      reverted->retry_attempts == committed->retry_attempts &&
      reverted->attempt_count == committed->attempt_count &&
      reverted->step_evidence_count == committed->step_evidence_count &&
      same_coordinates(reverted->coordinates, committed->coordinates);
  const bool physical_revert =
      launch_count(reverted->launches) > launch_count(committed->launches) &&
      reverted->launches.assemble_launches >=
          committed->launches.assemble_launches &&
      reverted->launches.condense_launches >=
          committed->launches.condense_launches &&
      reverted->launches.recover_launches >=
          committed->launches.recover_launches &&
      reverted->transfers.count == transfers_installed->count &&
      reverted->transfers.observed_count ==
          transfers_installed->observed_count &&
      reverted->transfers.ordered_digest ==
          transfers_installed->ordered_digest &&
      transfers_reverted->count == transfers_installed->count &&
      transfers_reverted->ordered_digest ==
          transfers_installed->ordered_digest &&
      transfers_installed->count > transfers_before->count &&
      reverted->allocation_ledger.count == allocations_installed.count &&
      reverted->allocation_ledger.ordered_digest ==
          allocations_installed.ordered_digest &&
      allocations_reverted.count == allocations_installed.count &&
      allocations_reverted.ordered_digest ==
          allocations_installed.ordered_digest &&
      allocations_installed.count >= allocations_before.count &&
      reverted->allocations.current_observed_events >=
          committed->allocations.current_observed_events &&
      reverted->allocations.current_owned_calls >=
          committed->allocations.current_owned_calls &&
      reverted->allocations.current_deallocations >=
          committed->allocations.current_deallocations &&
      reverted->reuse.owner == committed->reuse.owner &&
      reverted->reuse.identity_hash == committed->reuse.identity_hash &&
      reverted->reuse.factor_generation >=
          committed->reuse.factor_generation &&
      reverted->reuse.solver_generation >=
          committed->reuse.solver_generation &&
      reverted->reuse.unchanged &&
      reverted->residency.state_device_resident &&
      reverted->residency.topology_device_resident &&
      reverted->residency.workspace_device_resident &&
      reverted->residency.trace_solve_device_resident &&
      oracle::ValidateLaunchTransferCoherence(
          *transfers_reverted, reverted->launches) &&
      oracle::ValidateAllocationLedger(
          allocations_reverted, reverted->allocations);
  if (!logical_revert || !physical_revert) { return false; }

  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction reentry;
  if (owner->PreparePersistentTrajectoryTransaction(reentry) !=
          scalar::ScalarCudaStatus::accepted ||
      !owner->InstallPersistentTrajectoryTransaction(reentry) ||
      !owner->FinalizePersistentTrajectoryTransaction(std::move(reentry)))
  {
    return false;
  }
  const auto reentered = std::make_unique<scalar::ScalarCudaEvidence>(
      owner->trajectory_evidence());
  const auto transfers_reentered =
      std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(
          owner->transfer_ledger());
  const auto allocations_reentered = owner->allocation_ledger();
  return owner->status() == scalar::ScalarCudaStatus::ready &&
      reentered->status == scalar::ScalarCudaStatus::accepted &&
      reentered->accepted_steps == 1 && reentered->attempted_steps == 1 &&
      reentered->attempt_count == 1 && reentered->step_evidence_count == 1 &&
      reentered->coordinates.step == 1 &&
      launch_count(reentered->launches) > launch_count(reverted->launches) &&
      reentered->launches.assemble_launches >=
          reverted->launches.assemble_launches &&
      reentered->launches.condense_launches >=
          reverted->launches.condense_launches &&
      reentered->launches.recover_launches >=
          reverted->launches.recover_launches &&
      transfers_reentered->count > transfers_reverted->count &&
      reentered->transfers.count == transfers_reentered->count &&
      reentered->transfers.ordered_digest ==
          transfers_reentered->ordered_digest &&
      allocations_reentered.count >= allocations_reverted.count &&
      reentered->allocation_ledger.count == allocations_reentered.count &&
      reentered->allocation_ledger.ordered_digest ==
          allocations_reentered.ordered_digest &&
      reentered->allocations.current_observed_events >=
          reverted->allocations.current_observed_events &&
      reentered->allocations.current_owned_calls >=
          reverted->allocations.current_owned_calls &&
      reentered->allocations.current_deallocations >=
          reverted->allocations.current_deallocations &&
      reentered->reuse.owner == reverted->reuse.owner &&
      reentered->reuse.identity_hash == reverted->reuse.identity_hash &&
      reentered->reuse.factor_generation >=
          reverted->reuse.factor_generation &&
      reentered->reuse.solver_generation >=
          reverted->reuse.solver_generation &&
      reentered->reuse.unchanged &&
      oracle::ValidateLaunchTransferCoherence(
          *transfers_reentered, reentered->launches) &&
      oracle::ValidateAllocationLedger(
          allocations_reentered, reentered->allocations);
}

bool RunFaultCampaign(const scalar::FailureInjection injection)
{
  auto baseline_cpu = CreateCpu(8);
  auto baseline_cuda = CreateCuda(8);
  constexpr int status_not_observed = -1;
  const auto report_failure = [injection](
      const std::string_view phase, const int step,
      const cpu::PersistentSc25ScalarIgrCpuFoundation* const cpu_owner,
      const scalar::PersistentSc25ScalarIgrCudaRank1* const cuda_owner,
      const int cpu_call_status, const int cuda_call_status,
      const int attempt_index)
  {
    std::cerr << "C2 fault-campaign diagnostic injection="
              << static_cast<int>(injection) << " phase=" << phase
              << " step=" << step << " cpu_call_status=" << cpu_call_status
              << " cuda_call_status=" << cuda_call_status
              << " attempt_index=" << attempt_index << '\n';
    if (cpu_owner != nullptr)
    {
      const auto& evidence = cpu_owner->trajectory_evidence();
      std::cerr << "C2 fault-campaign CPU evidence status="
                << static_cast<int>(evidence.status) << " first_failure="
                << static_cast<int>(evidence.first_failure)
                << " accepted_steps=" << evidence.accepted_steps
                << " attempted_steps=" << evidence.attempted_steps
                << " retry_attempts=" << evidence.retry_attempts
                << " attempt_count=" << evidence.attempt_count
                << " step_evidence_count=" << evidence.step_evidence_count
                << " coordinate_step=" << evidence.coordinates.step
                << " coordinate_generation=" << evidence.coordinates.generation
                << '\n';
      if (attempt_index >= 0 &&
          static_cast<std::size_t>(attempt_index) < evidence.attempts.size())
      {
        const auto& attempt = evidence.attempts[
            static_cast<std::size_t>(attempt_index)];
        std::cerr << "C2 fault-campaign CPU attempt index=" << attempt_index
                  << " status=" << static_cast<int>(attempt.status)
                  << " injection=" << static_cast<int>(attempt.injection)
                  << " target_step=" << attempt.target_step
                  << " retry_index=" << attempt.retry_index
                  << " attempted_newton_iterations="
                  << attempt.attempted_newton_iterations
                  << " newton_iterations=" << attempt.newton_iterations
                  << " entry_residual=" << attempt.entry_residual_norm
                  << " final_residual=" << attempt.final_residual_norm
                  << " residual_bound=" << attempt.nonlinear_residual_bound
                  << '\n';
      }
    }
    else
    {
      std::cerr << "C2 fault-campaign CPU evidence unavailable\n";
    }
    if (cuda_owner != nullptr)
    {
      const auto& evidence = cuda_owner->trajectory_evidence();
      std::cerr << "C2 fault-campaign CUDA evidence status="
                << static_cast<int>(evidence.status) << " first_failure="
                << static_cast<int>(evidence.first_failure)
                << " accepted_steps=" << evidence.accepted_steps
                << " attempted_steps=" << evidence.attempted_steps
                << " retry_attempts=" << evidence.retry_attempts
                << " attempt_count=" << evidence.attempt_count
                << " step_evidence_count=" << evidence.step_evidence_count
                << " coordinate_step=" << evidence.coordinates.step
                << " coordinate_generation=" << evidence.coordinates.generation
                << " assemble_launches=" << evidence.launches.assemble_launches
                << " condense_launches=" << evidence.launches.condense_launches
                << " recover_launches=" << evidence.launches.recover_launches
                << " fallback_calls=" << evidence.launches.fallback_calls
                << '\n';
      if (attempt_index >= 0 &&
          static_cast<std::size_t>(attempt_index) < evidence.attempts.size())
      {
        const auto& attempt = evidence.attempts[
            static_cast<std::size_t>(attempt_index)];
        std::cerr << "C2 fault-campaign CUDA attempt index=" << attempt_index
                  << " status=" << static_cast<int>(attempt.status)
                  << " failure_cause="
                  << static_cast<int>(attempt.failure_cause)
                  << " injection=" << static_cast<int>(attempt.injection)
                  << " target_step=" << attempt.target_step
                  << " retry_index=" << attempt.retry_index
                  << " attempted_newton_iterations="
                  << attempt.attempted_newton_iterations
                  << " newton_iterations=" << attempt.newton_iterations
                  << " entry_residual=" << attempt.entry_residual_norm
                  << " final_residual=" << attempt.final_residual_norm
                  << " residual_bound=" << attempt.nonlinear_residual_bound
                  << '\n';
        if (attempt.failure_cause ==
            scalar::ScalarCudaFailureCause::publication)
        {
          const auto& allocation = evidence.allocations;
          const auto& residency = evidence.residency;
          std::cerr << "C2 CUDA publication allocation observed="
                    << allocation.current_observed_events << '/'
                    << allocation.baseline_observed_events
                    << " owned=" << allocation.current_owned_calls << '/'
                    << allocation.baseline_owned_calls
                    << " dealloc=" << allocation.current_deallocations << '/'
                    << allocation.baseline_deallocations
                    << " mfem-high-water="
                    << allocation.current_mfem_high_water << '/'
                    << allocation.baseline_mfem_high_water
                    << " hypre-high-water="
                    << allocation.current_hypre_high_water << '/'
                    << allocation.baseline_hypre_high_water
                    << " zero-growth="
                    << allocation.zero_post_baseline_observed_growth
                    << " zero-owned="
                    << allocation.zero_post_baseline_owned_allocation
                    << " zero-dealloc="
                    << allocation.zero_post_baseline_owned_deallocation
                    << '\n'
                    << "C2 CUDA publication topology="
                    << evidence.reuse.device_topology_matches
                    << " reuse=" << evidence.reuse.unchanged
                    << " residency=" << residency.mfem_cuda_device
                    << residency.hypre_cuda
                    << residency.umpire_device_pool
                    << residency.state_device_resident
                    << residency.topology_device_resident
                    << residency.workspace_device_resident
                    << residency.trace_solve_device_resident
                    << " host-fallback="
                    << residency.host_numerical_fallback << '\n';
        }
      }
    }
    else
    {
      std::cerr << "C2 fault-campaign CUDA evidence unavailable\n";
    }
    return false;
  };
  cpu::TrajectoryStatus baseline_cpu_initialization =
      cpu::TrajectoryStatus::uninitialized;
  scalar::ScalarCudaStatus baseline_cuda_initialization =
      scalar::ScalarCudaStatus::uninitialized;
  if (!baseline_cpu || !baseline_cuda ||
      (baseline_cpu_initialization =
           baseline_cpu->InitializePersistentTrajectory()) !=
          cpu::TrajectoryStatus::ready ||
      (baseline_cuda_initialization =
           baseline_cuda->InitializePersistentTrajectory()) !=
          scalar::ScalarCudaStatus::ready)
  {
    return report_failure(
        "baseline-initialization", 0, baseline_cpu.get(), baseline_cuda.get(),
        static_cast<int>(baseline_cpu_initialization),
        static_cast<int>(baseline_cuda_initialization), -1);
  }
  oracle::ComparisonEvidence baseline_comparison{};
  scalar::PublishedState baseline_cpu_state{}, baseline_cuda_state{};
  scalar::PublishedState baseline_step7{};
  if (!CopyPair(*baseline_cpu, *baseline_cuda, baseline_cpu_state,
                baseline_cuda_state, baseline_comparison))
  {
    return report_failure("baseline-initial-copy", 0, baseline_cpu.get(),
                          baseline_cuda.get(), status_not_observed,
                          status_not_observed, -1);
  }
  for (int step = 1; step <= scalar::persistent_required_steps; ++step)
  {
    const scalar::PublishedState entry = baseline_cuda_state;
    cpu::TrajectoryStatus baseline_cpu_status =
        cpu::TrajectoryStatus::uninitialized;
    scalar::ScalarCudaStatus baseline_cuda_status =
        scalar::ScalarCudaStatus::uninitialized;
    if (!CpuAccepted(baseline_cpu_status =
                         baseline_cpu->AdvancePersistentTrajectory()) ||
        !CudaAccepted(baseline_cuda_status =
                          baseline_cuda->AdvancePersistentTrajectory()) ||
        !CopyPair(*baseline_cpu, *baseline_cuda, baseline_cpu_state,
                  baseline_cuda_state, baseline_comparison))
    {
      const std::string_view phase = !CpuAccepted(baseline_cpu_status)
          ? "baseline-cpu-advance"
          : !CudaAccepted(baseline_cuda_status)
          ? "baseline-cuda-advance"
          : "baseline-step-copy";
      return report_failure(
          phase, step, baseline_cpu.get(), baseline_cuda.get(),
          static_cast<int>(baseline_cpu_status),
          static_cast<int>(baseline_cuda_status),
          baseline_cuda->trajectory_evidence().attempt_count - 1);
    }
    const auto& cuda_evidence = baseline_cuda->trajectory_evidence();
    const auto& cpu_evidence = baseline_cpu->trajectory_evidence();
    if (!ValidateAttemptStateEvidence(
            cuda_evidence.attempts[step - 1], cuda_evidence.steps[step - 1],
            step, entry, baseline_cuda_state))
    {
      return report_failure("baseline-step-state-evidence", step,
                            baseline_cpu.get(), baseline_cuda.get(),
                            status_not_observed, status_not_observed,
                            step - 1);
    }
    if (!ValidateAttemptParity(
            cuda_evidence.attempts[step - 1],
            cpu_evidence.last_attempt))
    {
      return report_failure("baseline-step-parity", step,
                            baseline_cpu.get(), baseline_cuda.get(),
                            status_not_observed, status_not_observed,
                            step - 1);
    }
    if (step == 7) { baseline_step7 = baseline_cuda_state; }
  }
  const scalar::PublishedState baseline_final = baseline_cuda_state;
  baseline_cpu.reset();
  baseline_cuda.reset();

  auto cpu_run = CreateCpu(8);
  auto cuda_run = CreateCuda(8);
  cpu::TrajectoryStatus cpu_initialization =
      cpu::TrajectoryStatus::uninitialized;
  scalar::ScalarCudaStatus cuda_initialization =
      scalar::ScalarCudaStatus::uninitialized;
  if (!cpu_run || !cuda_run ||
      (cpu_initialization = cpu_run->InitializePersistentTrajectory()) !=
          cpu::TrajectoryStatus::ready ||
      (cuda_initialization = cuda_run->InitializePersistentTrajectory()) !=
          scalar::ScalarCudaStatus::ready)
  {
    return report_failure("fault-initialization", 0, cpu_run.get(),
                          cuda_run.get(), static_cast<int>(cpu_initialization),
                          static_cast<int>(cuda_initialization), -1);
  }
  oracle::ComparisonEvidence comparison{};
  scalar::PublishedState cpu_state{}, cuda_state{};
  if (!CopyPair(*cpu_run, *cuda_run, cpu_state, cuda_state, comparison))
  {
    return report_failure("fault-initial-copy", 0, cpu_run.get(),
                          cuda_run.get(), status_not_observed,
                          status_not_observed, -1);
  }
  oracle::StateQuadratureAudit preceding_q12{}, preceding_q14{};
  if (!ValidateCudaStateScience(*cpu_run, cuda_state, nullptr, nullptr,
                                nullptr, nullptr,
                                preceding_q12, preceding_q14))
  {
    std::cerr << "C2 initialized CUDA state science audit failed\n";
    return report_failure("fault-initial-state-science", 0, cpu_run.get(),
                          cuda_run.get(), status_not_observed,
                          status_not_observed, -1);
  }
  scalar::PublishedState fault_step7{};
  for (int step = 1; step <= scalar::persistent_required_steps; ++step)
  {
    const scalar::PublishedState entry_cuda = cuda_state;
    const scalar::FailureInjection fault =
        step == 7 ? injection : scalar::FailureInjection::none;
    const cpu::TrajectoryStatus cpu_status =
        cpu_run->AdvancePersistentTrajectory(fault);
    const scalar::ScalarCudaStatus cuda_status =
        cuda_run->AdvancePersistentTrajectory(fault);
    if (!CpuAccepted(cpu_status) || !CudaAccepted(cuda_status) ||
        (step == 7 &&
         (cpu_status !=
              cpu::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry ||
          cuda_status !=
              scalar::ScalarCudaStatus::accepted_after_one_retry)) ||
        !CopyPair(*cpu_run, *cuda_run, cpu_state, cuda_state, comparison))
    {
      const std::string_view phase = !CpuAccepted(cpu_status)
          ? "fault-cpu-advance"
          : !CudaAccepted(cuda_status)
          ? "fault-cuda-advance"
          : step == 7 &&
                    cpu_status !=
                        cpu::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry
          ? "fault-cpu-retry-status"
          : step == 7 &&
                    cuda_status !=
                        scalar::ScalarCudaStatus::accepted_after_one_retry
          ? "fault-cuda-retry-status"
          : "fault-step-copy";
      return report_failure(
          phase, step, cpu_run.get(), cuda_run.get(),
          static_cast<int>(cpu_status), static_cast<int>(cuda_status),
          cuda_run->trajectory_evidence().attempt_count - 1);
    }
    const auto& current = cuda_run->trajectory_evidence();
    const auto& cpu_current = cpu_run->trajectory_evidence();
    const int accepted_attempt = current.steps[step - 1].attempt_index;
    if (accepted_attempt < 0 ||
        !ValidateAttemptStateEvidence(
            current.attempts[static_cast<std::size_t>(accepted_attempt)],
            current.steps[step - 1], step, entry_cuda, cuda_state))
    {
      return report_failure("fault-accepted-attempt-state", step,
                            cpu_run.get(), cuda_run.get(),
                            static_cast<int>(cpu_status),
                            static_cast<int>(cuda_status), accepted_attempt);
    }
    if (!ValidateAttemptParity(
            current.attempts[static_cast<std::size_t>(accepted_attempt)],
            cpu_current.last_attempt))
    {
      return report_failure("fault-accepted-attempt-parity", step,
                            cpu_run.get(), cuda_run.get(),
                            static_cast<int>(cpu_status),
                            static_cast<int>(cuda_status), accepted_attempt);
    }
    oracle::StateQuadratureAudit current_q12{}, current_q14{};
    if (!ValidateCudaStateScience(
            *cpu_run, cuda_state, &preceding_q12.integrals,
            &preceding_q14.integrals, &current.steps[step - 1],
            &current.attempts[static_cast<std::size_t>(accepted_attempt)],
            current_q12, current_q14))
    {
      return report_failure("fault-state-science", step, cpu_run.get(),
                            cuda_run.get(), static_cast<int>(cpu_status),
                            static_cast<int>(cuda_status), accepted_attempt);
    }
    preceding_q12 = current_q12;
    preceding_q14 = current_q14;
    if (step == 7)
    {
      const auto expected = ExpectedInjectedFailure(injection);
      const auto& failed = current.attempts[6];
      const scalar::StateHashes entry_hashes = oracle::HashState(entry_cuda);
      if (failed.status != expected.first ||
          failed.failure_cause != expected.second ||
          failed.injection != injection || failed.target_step != 7 ||
          !oracle::NonzeroHashes(failed.entry_hashes) ||
          !oracle::NonzeroHashes(failed.candidate_hashes) ||
          !oracle::SameHashes(failed.entry_hashes, entry_hashes) ||
          !oracle::SameHashes(failed.candidate_hashes, entry_hashes) ||
          failed.retry_index != 0 || failed.dt != scalar::fixed_dt ||
          !ValidateAttemptParity(failed, cpu_current.first_failed_attempt) ||
          current.steps[6].attempt_index != 7 ||
          current.accepted_steps != 7 || current.retry_attempts != 1)
      {
        std::cerr << "C2 fault-campaign injected-contract expected_status="
                  << static_cast<int>(expected.first)
                  << " actual_status=" << static_cast<int>(failed.status)
                  << " expected_cause=" << static_cast<int>(expected.second)
                  << " actual_cause="
                  << static_cast<int>(failed.failure_cause)
                  << " failed_attempt_index=6 accepted_attempt_index="
                  << current.steps[6].attempt_index << '\n';
        return report_failure("fault-injected-attempt-contract", step,
                              cpu_run.get(), cuda_run.get(),
                              static_cast<int>(cpu_status),
                              static_cast<int>(cuda_status), 6);
      }
      fault_step7 = cuda_state;
    }
  }
  const auto& evidence = cuda_run->trajectory_evidence();
  const bool final_contract = evidence.retry_attempts == 1 &&
      evidence.attempt_count == 21 &&
      evidence.accepted_steps == 20 &&
      evidence.steps[6].attempt_index == 7 &&
      oracle::SameStateBytes(fault_step7, baseline_step7) &&
      oracle::SameStateBytes(cuda_state, baseline_final) &&
      comparison.maximum_relative_l2 <= comparison_tolerance;
  if (!final_contract)
  {
    std::cerr << "C2 fault-campaign terminal comparison_relative_l2="
              << comparison.maximum_relative_l2
              << " comparison_tolerance=" << comparison_tolerance
              << " step7_attempt_index=" << evidence.steps[6].attempt_index
              << '\n';
    return report_failure("fault-terminal-contract", 20, cpu_run.get(),
                          cuda_run.get(), status_not_observed,
                          status_not_observed,
                          evidence.steps[6].attempt_index);
  }
  return true;
}

bool RunAbandonedPreparedTokenNegative()
{
  auto owner = CreateCuda(8);
  if (!owner || owner->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  scalar::PublishedState before{};
  if (!owner->CopyPublishedTrajectoryState(before)) { return false; }
  {
    scalar::PersistentSc25ScalarIgrCudaRank1::Transaction abandoned;
    if (owner->PreparePersistentTrajectoryTransaction(abandoned) !=
            scalar::ScalarCudaStatus::accepted ||
        !abandoned.valid() ||
        owner->status() != scalar::ScalarCudaStatus::transaction_prepared)
    {
      return false;
    }
  }
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction forbidden;
  scalar::PublishedState after{};
  if (owner->status() != scalar::ScalarCudaStatus::transaction_prepared ||
      owner->PreparePersistentTrajectoryTransaction(forbidden) !=
          scalar::ScalarCudaStatus::publication_failure || forbidden.valid() ||
      !owner->CopyPublishedTrajectoryState(after) ||
      !oracle::SameStateBytes(before, after))
  {
    return false;
  }
  const auto& evidence = owner->trajectory_evidence();
  return owner->status() == scalar::ScalarCudaStatus::publication_failure &&
      evidence.accepted_steps == 0 && evidence.attempted_steps == 1 &&
      evidence.attempt_count == 1 && evidence.coordinates.step == 0 &&
      evidence.attempts[0].status ==
          scalar::ScalarCudaStatus::publication_failure &&
      evidence.attempts[0].failure_cause ==
          scalar::ScalarCudaFailureCause::publication &&
      evidence.attempts[0].injection == scalar::FailureInjection::none &&
      oracle::NonzeroHashes(evidence.attempts[0].entry_hashes) &&
      oracle::NonzeroHashes(evidence.attempts[0].candidate_hashes);
}

bool RunDirectPrepareInjectedFailureNegative()
{
  const auto retained = std::make_unique<scalar::ScalarCudaEvidence>();
  auto owner = CreateCuda(8);
  if (!owner || owner->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  scalar::PublishedState before{}, after{};
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction failed;
  if (!owner->CopyPublishedTrajectoryState(before) ||
      owner->PreparePersistentTrajectoryTransaction(
          failed, scalar::FailureInjection::local_factor) !=
          scalar::ScalarCudaStatus::injected_local_factor_failure ||
      failed.valid() ||
      owner->status() != scalar::ScalarCudaStatus::ready ||
      !owner->CopyPublishedTrajectoryState(after) ||
      !oracle::SameStateBytes(before, after))
  {
    return false;
  }
  *retained = owner->trajectory_evidence();
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction overwrite;
  if (retained->accepted_steps != 0 || retained->attempted_steps != 1 ||
      retained->retry_attempts != 1 || retained->attempt_count != 1 ||
      retained->coordinates.step != 0 ||
      retained->first_failure !=
          scalar::ScalarCudaStatus::injected_local_factor_failure ||
      retained->attempts[0].status !=
          scalar::ScalarCudaStatus::injected_local_factor_failure ||
      retained->attempts[0].failure_cause !=
          scalar::ScalarCudaFailureCause::injected_local_factor ||
      retained->attempts[0].injection !=
          scalar::FailureInjection::local_factor ||
      !oracle::NonzeroHashes(retained->attempts[0].entry_hashes) ||
      !oracle::NonzeroHashes(retained->attempts[0].candidate_hashes) ||
      owner->PreparePersistentTrajectoryTransaction(overwrite) !=
          scalar::ScalarCudaStatus::accepted ||
      !overwrite.valid() || overwrite.provisionally_installed() ||
      !owner->RevertPersistentTrajectoryTransaction(std::move(overwrite)))
  {
    return false;
  }
  const auto& unchanged = owner->trajectory_evidence();
  return unchanged.attempt_count == retained->attempt_count &&
      unchanged.retry_attempts == retained->retry_attempts &&
      unchanged.first_failure == retained->first_failure &&
      unchanged.attempts[0].status == retained->attempts[0].status &&
      unchanged.coordinates.step == 0;
}

bool RunInstalledAbandonmentNegative()
{
  const auto committed = std::make_unique<scalar::ScalarCudaEvidence>();
  auto owner = CreateCuda(8);
  if (!owner || owner->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  scalar::ScalarCudaCheckpoint committed_checkpoint{};
  scalar::PublishedState before{}, installed{};
  if (owner->AdvancePersistentTrajectory() !=
          scalar::ScalarCudaStatus::accepted ||
      !owner->CopyPublishedTrajectoryState(before) ||
      !owner->CaptureTrajectoryCheckpoint(committed_checkpoint))
  {
    return false;
  }
  *committed = owner->trajectory_evidence();
  {
    scalar::PersistentSc25ScalarIgrCudaRank1::Transaction abandoned;
    if (owner->PreparePersistentTrajectoryTransaction(abandoned) !=
            scalar::ScalarCudaStatus::accepted ||
        !owner->InstallPersistentTrajectoryTransaction(abandoned) ||
        !abandoned.provisionally_installed())
    {
      return false;
    }
  }
  const auto& hidden = owner->trajectory_evidence();
  scalar::ScalarCudaCheckpoint checkpoint{};
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction forbidden;
  return owner->status() == scalar::ScalarCudaStatus::transaction_prepared &&
      owner->CopyPublishedTrajectoryState(installed) &&
      oracle::SameStateBytes(before, installed) &&
      hidden.status == committed->status &&
      hidden.accepted_steps == committed->accepted_steps &&
      hidden.attempted_steps == committed->attempted_steps &&
      hidden.attempt_count == committed->attempt_count &&
      hidden.step_evidence_count == committed->step_evidence_count &&
      hidden.coordinates.step == committed->coordinates.step &&
      hidden.launches.assemble_launches >=
          committed->launches.assemble_launches &&
      hidden.launches.condense_launches >=
          committed->launches.condense_launches &&
      hidden.launches.recover_launches >=
          committed->launches.recover_launches &&
      hidden.transfers.count > committed->transfers.count &&
      hidden.reuse.identity_hash == committed->reuse.identity_hash &&
      !owner->CaptureTrajectoryCheckpoint(checkpoint) &&
      owner->RestoreTrajectoryCheckpoint(committed_checkpoint) ==
          scalar::ScalarCudaStatus::restart_identity_failure &&
      owner->PreparePersistentTrajectoryTransaction(forbidden) ==
          scalar::ScalarCudaStatus::publication_failure &&
      !forbidden.valid() &&
      owner->status() == scalar::ScalarCudaStatus::transaction_prepared;
}

bool RejectTransferCensusMutations(
    const scalar::cuda_w8a::CudaTransferLedgerV1& source,
    const scalar::ScalarCudaEvidence& evidence,
    const oracle::TransferCensusExpectation expectation)
{
  const auto reaccount = [](scalar::cuda_w8a::CudaTransferLedgerV1& ledger,
                            const std::uint64_t synchronizations)
  {
    ledger.totals = {};
    ledger.ordered_digest = oracle::fnv_offset;
    ledger.observed_count = ledger.count;
    ledger.dropped_count = 0;
    ledger.overflow = false;
    for (std::size_t index = 0; index < ledger.count; ++index)
    {
      auto& event = ledger.events[index];
      event.ordinal = index + 1;
      event.endpoint_token = oracle::IndependentEndpointToken(event);
      if (!oracle::AddTransfer(ledger.totals, event)) { return false; }
      ledger.ordered_digest ^= event.endpoint_token;
      ledger.ordered_digest *= oracle::fnv_prime;
    }
    ledger.totals.synchronization_events = synchronizations;
    return true;
  };
  if (source.count < 21 || source.count >= source.events.size())
  {
    return false;
  }
  const std::uint64_t synchronizations =
      source.totals.synchronization_events;

  auto mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(
      source);
  for (std::size_t index = 0; index + 1 < mutated->count; ++index)
  {
    mutated->events[index] = mutated->events[index + 1];
  }
  --mutated->count;
  if (!reaccount(*mutated, synchronizations) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  mutated->events[mutated->count] = mutated->events[20];
  ++mutated->count;
  if (!reaccount(*mutated, synchronizations) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  for (std::size_t index = mutated->count; index > 1; --index)
  {
    mutated->events[index] = mutated->events[index - 1];
  }
  mutated->events[1] = mutated->events[0];
  ++mutated->count;
  if (!reaccount(*mutated, synchronizations) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  mutated->events[0].bytes += sizeof(int);
  mutated->events[0].source_capacity = mutated->events[0].bytes;
  mutated->events[0].destination_capacity = mutated->events[0].bytes;
  if (!reaccount(*mutated, synchronizations) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  using Category = scalar::cuda_w8a::CudaTransferCategory;
  using Direction = scalar::cuda_w8a::CudaTransferDirection;
  using Site = scalar::cuda_w8a::CudaTransferSite;
  std::size_t reduction_index = source.count;
  for (std::size_t index = 20; index < source.count; ++index)
  {
    const auto& event = source.events[index];
    if (event.direction == Direction::device_to_host &&
        event.category == Category::scalar_diagnostic &&
        event.site == Site::scalar_status &&
        event.bytes == 16 * sizeof(double))
    {
      reduction_index = index;
      break;
    }
  }
  if (reduction_index == source.count || synchronizations == 0)
  {
    return false;
  }
  std::size_t extra_reduction_index = source.count;
  scalar::cuda_w8a::CudaPolicyContext extra_reduction_context{};
  for (std::size_t index = 20; index + 4 < source.count; ++index)
  {
    const auto& first = source.events[index];
    const auto& next = source.events[index + 4];
    if (first.direction == Direction::host_to_device &&
        first.category == Category::scalar_diagnostic &&
        first.site == Site::scalar_status &&
        next.direction == Direction::host_to_device &&
        next.category == Category::scalar_diagnostic &&
        next.site == Site::scalar_status)
    {
      extra_reduction_index = index + 4;
      extra_reduction_context = first.context;
      break;
    }
  }
  if (extra_reduction_index == source.count) { return false; }

  // Remove one otherwise-valid 128-byte reduction event and rebuild every
  // dependent ledger identity.  Exact operation evidence must still reject.
  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  for (std::size_t index = reduction_index; index + 1 < mutated->count;
       ++index)
  {
    mutated->events[index] = mutated->events[index + 1];
  }
  --mutated->count;
  if (!reaccount(*mutated, synchronizations - 1) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  // Insert a second route-valid reduction event, likewise with consistent
  // ordinals, endpoint tokens, totals, digest, and synchronization count.
  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  for (std::size_t index = mutated->count; index > extra_reduction_index;
       --index)
  {
    mutated->events[index] = mutated->events[index - 1];
  }
  mutated->events[extra_reduction_index] =
      mutated->events[reduction_index];
  mutated->events[extra_reduction_index].context =
      extra_reduction_context;
  ++mutated->count;
  if (!reaccount(*mutated, synchronizations + 1) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  std::size_t alternate_reduction_index = source.count;
  for (std::size_t index = reduction_index + 1; index < source.count;
       ++index)
  {
    const auto& event = source.events[index];
    if (event.direction == Direction::device_to_host &&
        event.category == Category::scalar_diagnostic &&
        event.site == Site::scalar_status &&
        event.bytes == 16 * sizeof(double) &&
        (event.context.attempt_sequence !=
             source.events[reduction_index].context.attempt_sequence ||
         event.context.target_step !=
             source.events[reduction_index].context.target_step ||
         event.context.generation !=
             source.events[reduction_index].context.generation ||
         event.context.phase !=
             source.events[reduction_index].context.phase))
    {
      alternate_reduction_index = index;
      break;
    }
  }
  if (alternate_reduction_index == source.count) { return false; }

  // Keep a route-valid reduction but give it another real attempt's complete
  // context.  Re-accounting must not hide the producer-context mismatch.
  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  mutated->events[reduction_index].context =
      mutated->events[alternate_reduction_index].context;
  if (!reaccount(*mutated, synchronizations) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  // Independently change the attempt/generation tuple and phase while
  // preserving the route invariant attempt == step == generation.
  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  auto& changed_context = mutated->events[reduction_index].context;
  changed_context.attempt_sequence = 1;
  changed_context.target_step = 1;
  changed_context.generation = 1;
  changed_context.phase =
      scalar::cuda_w8a::CudaFailurePhase::terminal_evidence;
  if (!reaccount(*mutated, synchronizations) ||
      oracle::ValidateTransferLedger(*mutated, evidence, expectation))
  {
    return false;
  }

  // Substitute the complete exact C1/9 upload byte census into a copied C2
  // ledger.  An explicit C2/64 expectation must not self-select that shape.
  constexpr std::uint64_t c1_elements = 9;
  constexpr std::uint64_t c1_true_trace = 20 * c1_elements;
  constexpr std::uint64_t c1_volume_points = 49 * c1_elements;
  constexpr std::uint64_t c1_face_points = 28 * c1_elements;
  const std::array<std::uint64_t, 20> c1_upload_bytes{{
      c1_elements * scalar::local_trace_unknowns * sizeof(int),
      c1_elements * scalar::local_trace_unknowns * sizeof(int),
      c1_elements * scalar::local_trace_unknowns *
          scalar::local_trace_unknowns * sizeof(int),
      (c1_true_trace + 1) * sizeof(int),
      c1_true_trace * 70 * sizeof(int),
      c1_volume_points * sizeof(double),
      c1_volume_points * scalar::volume_nodes * sizeof(double),
      c1_volume_points * scalar::volume_nodes * scalar::dimension *
          sizeof(double),
      c1_volume_points * sizeof(double),
      c1_volume_points * sizeof(double),
      c1_face_points * sizeof(double),
      c1_elements * 4 * sizeof(double),
      c1_face_points * scalar::volume_nodes * sizeof(double),
      c1_face_points * scalar::face_nodes * sizeof(double),
      c1_face_points * scalar::dimension * sizeof(double),
      c1_face_points * sizeof(double),
      c1_face_points * sizeof(double),
      c1_elements * scalar::volume_nodes * sizeof(double),
      c1_elements * scalar::volume_nodes * sizeof(double),
      c1_elements * 4 * scalar::face_nodes * sizeof(double)}};
  mutated = std::make_unique<scalar::cuda_w8a::CudaTransferLedgerV1>(source);
  for (std::size_t index = 0; index < c1_upload_bytes.size(); ++index)
  {
    mutated->events[index].bytes = c1_upload_bytes[index];
    mutated->events[index].source_capacity = c1_upload_bytes[index];
    mutated->events[index].destination_capacity = c1_upload_bytes[index];
  }
  return reaccount(*mutated, synchronizations) &&
      !oracle::ValidateTransferLedger(*mutated, evidence, expectation) &&
      !oracle::ValidateTransferLedger(
          source, evidence, oracle::c1_transfer_expectation);
}

bool RunCheckpointNegatives(
    const scalar::ScalarCudaCheckpoint& checkpoint,
    cpu::PersistentSc25ScalarIgrCpuFoundation& cpu_audit)
{
  auto destination = CreateCuda(8);
  scalar::ScalarCudaCheckpoint independently_rehashed = checkpoint;
  if (!destination)
  {
    std::cerr << "C2 checkpoint-negative setup: cuda-create\n";
    return false;
  }
  const scalar::ScalarCudaStatus initialization =
      destination->InitializePersistentTrajectory();
  if (initialization != scalar::ScalarCudaStatus::ready)
  {
    std::cerr << "C2 checkpoint-negative setup: cuda-initialize actual="
              << static_cast<int>(initialization) << " expected="
              << static_cast<int>(scalar::ScalarCudaStatus::ready) << '\n';
    return false;
  }
  if (!oracle::RehashCheckpoint(independently_rehashed))
  {
    std::cerr << "C2 checkpoint-negative setup: pristine-rehash\n";
    return false;
  }
  if (independently_rehashed.canonical_manifest_hash !=
      checkpoint.canonical_manifest_hash)
  {
    std::cerr << "C2 checkpoint-negative setup: pristine-manifest-hash\n";
    return false;
  }
  scalar::PublishedState before{};
  if (!destination->CopyPublishedTrajectoryState(before))
  {
    std::cerr << "C2 checkpoint-negative setup: baseline-state-copy\n";
    return false;
  }
  const auto reject = [&destination, &before](
      const std::string_view label,
      const scalar::ScalarCudaCheckpoint& candidate,
      const scalar::ScalarCudaStatus expected)
  {
    scalar::PublishedState after{};
    const scalar::ScalarCudaStatus actual =
        destination->RestoreTrajectoryCheckpoint(candidate);
    bool copied = false;
    bool preserved = false;
    bool ready = false;
    if (actual == expected)
    {
      copied = destination->CopyPublishedTrajectoryState(after);
    }
    if (copied) { preserved = oracle::SameStateBytes(before, after); }
    if (preserved)
    {
      ready = destination->status() == scalar::ScalarCudaStatus::ready;
    }
    if (actual == expected && copied && preserved && ready) { return true; }
    const std::string_view first_origin = actual != expected
        ? "restore-status"
        : !copied
        ? "state-copy"
        : !preserved
        ? "state-bytes"
        : "destination-status";
    std::cerr << "C2 checkpoint-negative " << label
              << " first_origin=" << first_origin
              << " actual=" << static_cast<int>(actual)
              << " expected=" << static_cast<int>(expected)
              << " copied=" << copied
              << " bytes_preserved=" << preserved
              << " destination_ready=" << ready << '\n';
    return false;
  };
  const auto reject_rehashed = [&reject](
      const std::string_view label,
      scalar::ScalarCudaCheckpoint candidate,
      const scalar::ScalarCudaStatus expected)
  {
    if (!oracle::RehashCheckpoint(candidate))
    {
      std::cerr << "C2 checkpoint-negative " << label
                << " first_origin=rehash\n";
      return false;
    }
    return reject(label, candidate, expected);
  };

  scalar::ScalarCudaCheckpoint identity = checkpoint;
  identity.tuple = ::dg::hdg::runtime::StableIdentity::Copy("wrong-tuple");
  if (!reject("tuple", identity,
              scalar::ScalarCudaStatus::restart_identity_failure))
  {
    return false;
  }
  scalar::ScalarCudaCheckpoint hash = checkpoint;
  reinterpret_cast<unsigned char*>(hash.interior_component_slabs.data())[0]
      ^= 0x01U;
  if (!reject("hash", hash, scalar::ScalarCudaStatus::restart_hash_failure))
  {
    return false;
  }
  const auto reject_coordinates = [&reject_rehashed](
      const std::string_view label,
      scalar::ScalarCudaCheckpoint candidate)
  {
    candidate.descriptor.coordinate_fingerprint =
        ::dg::hdg::runtime::detail::CoordinateFingerprint(
            candidate.descriptor.coordinates);
    return reject_rehashed(
        label, candidate, scalar::ScalarCudaStatus::restart_identity_failure);
  };
  scalar::ScalarCudaCheckpoint time = checkpoint;
  time.descriptor.coordinates.time += scalar::fixed_dt;
  scalar::ScalarCudaCheckpoint generation = checkpoint;
  ++generation.descriptor.coordinates.generation;
  scalar::ScalarCudaCheckpoint cursor = checkpoint;
  ++cursor.descriptor.coordinates.cursor;
  cursor.descriptor.cursor = cursor.descriptor.coordinates.cursor;
  scalar::ScalarCudaCheckpoint dt = checkpoint;
  dt.descriptor.coordinates.dt *= 2.0;
  dt.descriptor.dt = dt.descriptor.coordinates.dt;
  if (!reject_coordinates("time", time) ||
      !reject_coordinates("generation", generation) ||
      !reject_coordinates("cursor", cursor) ||
      !reject_coordinates("dt", dt))
  {
    return false;
  }
  scalar::ScalarCudaCheckpoint inadmissible = checkpoint;
  std::fill_n(inadmissible.interior_component_slabs.begin(),
              scalar::persistent_scalar_trace_nodes, -1.0);
  if (!reject_rehashed(
          "admissibility", inadmissible,
          scalar::ScalarCudaStatus::restart_admissibility_failure))
  {
    return false;
  }
  scalar::ScalarCudaCheckpoint assembly = checkpoint;
  assembly.interior_component_slabs[
      scalar::state_components * scalar::persistent_scalar_trace_nodes] =
      DBL_MAX;
  if (!reject_rehashed(
          "assembly", assembly,
          scalar::ScalarCudaStatus::restart_assembly_failure))
  {
    return false;
  }
  scalar::ScalarCudaCheckpoint auxiliary = checkpoint;
  auxiliary.interior_component_slabs[
      scalar::state_components * scalar::persistent_scalar_trace_nodes] +=
      1.0e-3;
  if (!reject_rehashed(
          "auxiliary", auxiliary,
          scalar::ScalarCudaStatus::restart_auxiliary_residual_failure))
  {
    return false;
  }
  const scalar::ScalarCudaStatus pristine_status =
      destination->RestoreTrajectoryCheckpoint(checkpoint);
  if (pristine_status != scalar::ScalarCudaStatus::ready)
  {
    const auto& evidence = destination->trajectory_evidence();
    const auto& allocation = evidence.allocations;
    const auto& reuse = evidence.reuse;
    const auto& residency = evidence.residency;
    std::cerr << "C2 checkpoint-negative pristine-restore first_origin="
              << "restore-status actual=" << static_cast<int>(pristine_status)
              << " expected="
              << static_cast<int>(scalar::ScalarCudaStatus::ready)
              << " allocation_observed="
              << allocation.current_observed_events << '/'
              << allocation.baseline_observed_events
              << " owned=" << allocation.current_owned_calls << '/'
              << allocation.baseline_owned_calls
              << " dealloc=" << allocation.current_deallocations << '/'
              << allocation.baseline_deallocations
              << " mfem_high_water=" << allocation.current_mfem_high_water
              << '/' << allocation.baseline_mfem_high_water
              << " hypre_high_water=" << allocation.current_hypre_high_water
              << '/' << allocation.baseline_hypre_high_water
              << " warm_plateau=" << allocation.warm_plateau
              << " zero_growth="
              << allocation.zero_post_baseline_observed_growth
              << " zero_owned="
              << allocation.zero_post_baseline_owned_allocation
              << " zero_dealloc="
              << allocation.zero_post_baseline_owned_deallocation << '\n'
              << "C2 checkpoint-negative pristine-restore reuse="
              << reuse.unchanged
              << " topology=" << reuse.device_topology_matches
              << " generations=" << reuse.topology_generation << ','
              << reuse.storage_generation << ',' << reuse.factor_generation
              << ',' << reuse.solver_generation
              << " residency=" << residency.mfem_cuda_device
              << residency.hypre_cuda << residency.umpire_device_pool
              << residency.state_device_resident
              << residency.topology_device_resident
              << residency.workspace_device_resident
              << residency.trace_solve_device_resident
              << " host_fallback=" << residency.host_numerical_fallback
              << '\n';
    return false;
  }
  scalar::PublishedState restored{};
  if (!destination->CopyPublishedTrajectoryState(restored))
  {
    std::cerr << "C2 checkpoint-negative pristine-state-copy\n";
    return false;
  }
  if (!oracle::SameStateBytes(restored,
                              oracle::DecodeCheckpoint(checkpoint)))
  {
    std::cerr << "C2 checkpoint-negative pristine-state-bytes\n";
    return false;
  }
  if (!oracle::SameHashes(oracle::HashState(restored),
                          checkpoint.raw_field_hashes))
  {
    std::cerr << "C2 checkpoint-negative pristine-state-hash\n";
    return false;
  }
  destination.reset();
  oracle::StateQuadratureAudit q12{}, q14{};
  if (!ValidateCudaStateScience(cpu_audit, restored, nullptr, nullptr,
                                nullptr, nullptr, q12, q14))
  {
    std::cerr << "C2 checkpoint-negative pristine-state-science"
              << " q12_points=" << q12.volume_points << ','
              << q12.side_points << ',' << q12.trace_points
              << " q12_min=" << q12.minimum_density << ','
              << q12.minimum_pressure << ',' << q12.minimum_speed_margin
              << " q14_points=" << q14.volume_points << ','
              << q14.side_points << ',' << q14.trace_points
              << " q14_min=" << q14.minimum_density << ','
              << q14.minimum_pressure << ',' << q14.minimum_speed_margin
              << '\n';
    return false;
  }
  return true;
}

bool RunC2()
{
  bool transaction_negatives_pass = true;
  const auto run_transaction_subcase =
      [&transaction_negatives_pass](const std::string_view name,
                                    bool (*run)())
  {
    const bool pass = run();
    std::cout << "C2 transaction-negative subcase=" << name
              << " pass=" << pass << '\n';
    transaction_negatives_pass = pass && transaction_negatives_pass;
  };
  run_transaction_subcase("strong-transaction", RunTransactionNegatives);
  run_transaction_subcase("move-live-token", RunMoveWithLiveTransaction);
  run_transaction_subcase("stale-token-after-owner-destruction",
                          RunStaleTransactionAfterOwnerDestruction);
  run_transaction_subcase("revert-physical-evidence",
                          RunRevertPhysicalEvidence);
  run_transaction_subcase("abandoned-prepared-token",
                          RunAbandonedPreparedTokenNegative);
  run_transaction_subcase("direct-prepare-injected-failure",
                          RunDirectPrepareInjectedFailureNegative);
  run_transaction_subcase("installed-token-abandonment",
                          RunInstalledAbandonmentNegative);
  if (!transaction_negatives_pass)
  {
    std::cerr << "C2 transaction negative campaign failed\n";
    return false;
  }
  const std::array<scalar::FailureInjection, 4> faults{{
      scalar::FailureInjection::local_factor,
      scalar::FailureInjection::trace_linear,
      scalar::FailureInjection::admissibility,
      scalar::FailureInjection::newton_iteration_limit}};
  bool fault_campaigns_pass = true;
  for (const scalar::FailureInjection fault : faults)
  {
    const bool pass = RunFaultCampaign(fault);
    std::cout << "C2 fault-campaign injection=" << static_cast<int>(fault)
              << " pass=" << pass << '\n';
    fault_campaigns_pass = pass && fault_campaigns_pass;
  }
  if (!fault_campaigns_pass)
  {
    std::cerr << "C2 one or more fresh-owner fault campaigns failed\n";
    return false;
  }

  const auto forbidden_step0_input =
      std::make_unique<scalar::ScalarCudaCheckpoint>();
  forbidden_step0_input->descriptor.successful_commit = true;
  forbidden_step0_input->interior_component_slabs[0] = 1.0;
  auto cpu_run = CreateCpu(8);
  auto cuda_run = CreateCuda(8);
  if (!cpu_run || !cuda_run ||
      cpu_run->InitializePersistentTrajectory() != cpu::TrajectoryStatus::ready ||
      cuda_run->InitializePersistentTrajectory() !=
          scalar::ScalarCudaStatus::ready)
  {
    return false;
  }
  const auto step0_physical =
      std::make_unique<scalar::ScalarCudaEvidence>(
          cuda_run->trajectory_evidence());
  scalar::ScalarCudaCheckpoint forbidden_step0 = *forbidden_step0_input;
  const bool capture_rejected =
      !cuda_run->CaptureTrajectoryCheckpoint(forbidden_step0);
  const bool output_cleared = DefaultCheckpoint(forbidden_step0);
  const bool output_unchanged =
      SameCheckpoint(forbidden_step0, *forbidden_step0_input);
  const std::string_view output_disposition = output_cleared ? "cleared" :
      output_unchanged ? "unchanged" : "other";
  const auto& step0_after = cuda_run->trajectory_evidence();
  const bool solver_policy_same =
      step0_physical->trace_restart == step0_after.trace_restart &&
      step0_physical->trace_maximum_iterations ==
          step0_after.trace_maximum_iterations &&
      step0_physical->trace_relative_tolerance ==
          step0_after.trace_relative_tolerance &&
      step0_physical->trace_absolute_tolerance ==
          step0_after.trace_absolute_tolerance &&
      step0_physical->ilu_type == step0_after.ilu_type &&
      step0_physical->ilu_level_of_fill == step0_after.ilu_level_of_fill &&
      step0_physical->ilu_local_reordering ==
          step0_after.ilu_local_reordering;
  const bool residency_same =
      SameResidency(step0_physical->residency, step0_after.residency);
  const bool allocations_same =
      SameAllocations(step0_physical->allocations, step0_after.allocations);
  const bool reuse_same = SameReuse(step0_physical->reuse, step0_after.reuse);
  const bool launches_same =
      SameLaunches(step0_physical->launches, step0_after.launches);
  const bool transfers_same =
      SameTransferLedger(step0_physical->transfers, step0_after.transfers);
  const bool allocation_ledger_same = SameAllocationLedger(
      step0_physical->allocation_ledger, step0_after.allocation_ledger);
  const bool physical_evidence_same = solver_policy_same && residency_same &&
      allocations_same && reuse_same && launches_same && transfers_same &&
      allocation_ledger_same;
  std::cout << "C2 step0 checkpoint capture-rejected=" << capture_rejected
            << " output-disposition=" << output_disposition
            << " physical-evidence-same=" << physical_evidence_same << '\n';
  if (!capture_rejected || !physical_evidence_same)
  {
    std::cerr << "C2 step0 checkpoint rejection/physical contract failed"
              << " capture-rejected=" << capture_rejected
              << " output-disposition=" << output_disposition
              << " solver-policy=" << solver_policy_same
              << " residency=" << residency_same
              << " allocations=" << allocations_same
              << " reuse=" << reuse_same
              << " launches=" << launches_same
              << " transfers=" << transfers_same
              << " allocation-ledger=" << allocation_ledger_same << '\n';
    return false;
  }
  oracle::ComparisonEvidence comparison{};
  scalar::PublishedState cpu_state{}, cuda_state{};
  if (!CopyPair(*cpu_run, *cuda_run, cpu_state, cuda_state, comparison))
  {
    return false;
  }
  oracle::StateQuadratureAudit preceding_q12{}, preceding_q14{};
  if (!ValidateCudaStateScience(*cpu_run, cuda_state, nullptr, nullptr,
                                nullptr, nullptr,
                                preceding_q12, preceding_q14))
  {
    std::cerr << "C2 initialized CUDA state science audit failed\n";
    return false;
  }
  cpu::PersistentTrajectoryCheckpoint cpu_checkpoint10{};
  scalar::ScalarCudaCheckpoint cuda_checkpoint10{};
  for (int step = 1; step <= scalar::persistent_required_steps; ++step)
  {
    const scalar::PublishedState entry_cuda = cuda_state;
    if (!CpuAccepted(cpu_run->AdvancePersistentTrajectory()) ||
        !CudaAccepted(cuda_run->AdvancePersistentTrajectory()) ||
        !CopyPair(*cpu_run, *cuda_run, cpu_state, cuda_state, comparison))
    {
      std::cerr << "C2 parity failed at step " << step << '\n';
      return false;
    }
    const auto& current = cuda_run->trajectory_evidence();
    const auto& cpu_current = cpu_run->trajectory_evidence();
    const int attempt_index = current.steps[step - 1].attempt_index;
    if (attempt_index != step - 1 ||
        !ValidateAttemptStateEvidence(
            current.attempts[static_cast<std::size_t>(attempt_index)],
            current.steps[step - 1], step, entry_cuda, cuda_state) ||
        !ValidateAttemptParity(
            current.attempts[static_cast<std::size_t>(attempt_index)],
            cpu_current.last_attempt))
    {
      std::cerr << "C2 hash/coordinate evidence failed at step "
                << step << '\n';
      return false;
    }
    oracle::StateQuadratureAudit current_q12{}, current_q14{};
    if (!ValidateCudaStateScience(
            *cpu_run, cuda_state, &preceding_q12.integrals,
            &preceding_q14.integrals, &current.steps[step - 1],
            &current.attempts[static_cast<std::size_t>(attempt_index)],
            current_q12, current_q14))
    {
      std::cerr << "C2 quadrature/admissibility/auxiliary audit failed at step "
                << step << '\n';
      return false;
    }
    preceding_q12 = current_q12;
    preceding_q14 = current_q14;
    if (step == 10)
    {
      const bool cpu_checkpoint_captured =
          cpu_run->CaptureTrajectoryCheckpoint(cpu_checkpoint10);
      const bool cuda_checkpoint_captured =
          cuda_run->CaptureTrajectoryCheckpoint(cuda_checkpoint10);
      std::cout << "C2 step10 checkpoint cpu=" << cpu_checkpoint_captured
                << " cuda=" << cuda_checkpoint_captured << '\n';
      if (!cpu_checkpoint_captured || !cuda_checkpoint_captured)
      {
        std::cerr << "C2 step10 checkpoint capture failed"
                  << " cpu=" << cpu_checkpoint_captured
                  << " cuda=" << cuda_checkpoint_captured << '\n';
        return false;
      }
    }
  }
  scalar::PublishedState cpu_final{}, cuda_final{};
  if (!CopyPair(*cpu_run, *cuda_run, cpu_final, cuda_final, comparison))
  {
    return false;
  }

  cpu::AuxiliaryResidualMetrics auxiliary{};
  if (!cpu_oracle::EvaluateIndependentAuxiliaryResidualOracle(
          *cpu_run, cuda_final,
          cpu::coupled_fixture_repeat_quadrature_order, auxiliary) ||
      auxiliary.normalized_maximum > comparison_tolerance)
  {
    return false;
  }

  const auto continuous_evidence =
      std::make_unique<scalar::ScalarCudaEvidence>(
          cuda_run->trajectory_evidence());
  const auto reuse = cuda_run->reuse_evidence();
  auto mutated_reuse = reuse;
  mutated_reuse.device_topology_hash.bytes[0] ^= 0x01U;
  cpu_run.reset();
  cuda_run.reset();

  auto cpu_restart = CreateCpu(8);
  auto cuda_restart = CreateCuda(8);
  const bool cpu_restart_created = cpu_restart != nullptr;
  const bool cuda_restart_created = cuda_restart != nullptr;
  bool cpu_restart_initialized = false;
  bool cuda_restart_initialized = false;
  if (cpu_restart_created)
  {
    cpu_restart_initialized =
        cpu_restart->InitializePersistentTrajectory() ==
        cpu::TrajectoryStatus::ready;
  }
  if (cuda_restart_created)
  {
    cuda_restart_initialized =
        cuda_restart->InitializePersistentTrajectory() ==
        scalar::ScalarCudaStatus::ready;
  }
  bool cpu_checkpoint_restored = false;
  bool cuda_checkpoint_restored = false;
  if (cpu_restart_initialized)
  {
    cpu_checkpoint_restored = CpuAccepted(
        cpu_restart->RestoreTrajectoryCheckpoint(cpu_checkpoint10));
  }
  if (cuda_restart_initialized)
  {
    cuda_checkpoint_restored =
        cuda_restart->RestoreTrajectoryCheckpoint(cuda_checkpoint10) ==
        scalar::ScalarCudaStatus::ready;
  }
  std::cout << "C2 fresh-restart cpu-created=" << cpu_restart_created
            << " cuda-created=" << cuda_restart_created
            << " cpu-initialized=" << cpu_restart_initialized
            << " cuda-initialized=" << cuda_restart_initialized
            << " cpu-restored=" << cpu_checkpoint_restored
            << " cuda-restored=" << cuda_checkpoint_restored << '\n';
  if (!cpu_restart_created || !cuda_restart_created ||
      !cpu_restart_initialized || !cuda_restart_initialized ||
      !cpu_checkpoint_restored || !cuda_checkpoint_restored)
  {
    std::cerr << "C2 fresh restart initialization/restore failed\n";
    return false;
  }
  if (cuda_restart->status() != scalar::ScalarCudaStatus::ready ||
      !ValidateLogicalRebase(cuda_restart->trajectory_evidence(), 10))
  {
    std::cerr << "C2 fresh step10 restore did not rebase logical evidence\n";
    return false;
  }
  scalar::PublishedState restart_cpu{}, restart_cuda{};
  if (!CopyPair(*cpu_restart, *cuda_restart, restart_cpu, restart_cuda,
                comparison) ||
      !oracle::SameStateBytes(
          restart_cuda, oracle::DecodeCheckpoint(cuda_checkpoint10)))
  {
    return false;
  }
  oracle::StateQuadratureAudit restart_q12{}, restart_q14{};
  if (!ValidateCudaStateScience(*cpu_restart, restart_cuda, nullptr, nullptr,
                                nullptr, nullptr,
                                restart_q12, restart_q14))
  {
    return false;
  }
  for (int step = 11; step <= scalar::persistent_required_steps; ++step)
  {
    const scalar::PublishedState entry_cuda = restart_cuda;
    const cpu::TrajectoryStatus cpu_status =
        cpu_restart->AdvancePersistentTrajectory();
    scalar::ScalarCudaStatus cuda_status =
        scalar::ScalarCudaStatus::uninitialized;
    if (CpuAccepted(cpu_status))
    {
      cuda_status = cuda_restart->AdvancePersistentTrajectory();
    }
    const bool accepted = CpuAccepted(cpu_status) && CudaAccepted(cuda_status);
    bool cpu_copied = false;
    bool cuda_copied = false;
    bool states_compared = false;
    const std::size_t first_new_failure = comparison.failures.size();
    if (accepted)
    {
      cpu_copied = cpu_restart->CopyPublishedTrajectoryState(restart_cpu);
      if (cpu_copied)
      {
        cuda_copied = cuda_restart->CopyPublishedTrajectoryState(restart_cuda);
      }
      if (cuda_copied)
      {
        states_compared = oracle::CompareState(
            restart_cpu, restart_cuda, comparison_tolerance, comparison);
      }
    }
    if (!accepted || !cpu_copied || !cuda_copied || !states_compared)
    {
      const std::string_view first_origin = !CpuAccepted(cpu_status)
          ? "cpu-advance"
          : !CudaAccepted(cuda_status)
          ? "cuda-advance"
          : !cpu_copied
          ? "cpu-copy"
          : !cuda_copied
          ? "cuda-copy"
          : "state-compare";
      std::cerr << "C2 restart parity failed at step " << step
                << " first_origin=" << first_origin
                << " cpu_status=" << static_cast<int>(cpu_status)
                << " cuda_status=" << static_cast<int>(cuda_status)
                << " copy_attempted=" << accepted
                << " cpu_copied=" << cpu_copied
                << " cuda_copied=" << cuda_copied
                << " states_compared=" << states_compared
                << " max_relative_l2=" << comparison.maximum_relative_l2
                << '\n';
      for (std::size_t failure = first_new_failure;
           failure < comparison.failures.size(); ++failure)
      {
        std::cerr << "C2 restart failed state field: "
                  << comparison.failures[failure] << '\n';
      }
      if (first_new_failure == comparison.failures.size())
      {
        std::cerr << "C2 restart failed state fields: not-reached\n";
      }
      return false;
    }
    const int local_step = step - 11;
    const auto& cuda_current = cuda_restart->trajectory_evidence();
    const auto& cpu_current = cpu_restart->trajectory_evidence();
    const int attempt_index =
        cuda_current.steps[static_cast<std::size_t>(local_step)].attempt_index;
    if (attempt_index != local_step ||
        !ValidateAttemptStateEvidence(
            cuda_current.attempts[static_cast<std::size_t>(attempt_index)],
            cuda_current.steps[static_cast<std::size_t>(local_step)], step,
            entry_cuda, restart_cuda) ||
        !ValidateAttemptParity(
            cuda_current.attempts[static_cast<std::size_t>(attempt_index)],
            cpu_current.last_attempt))
    {
      return false;
    }
    oracle::StateQuadratureAudit current_q12{}, current_q14{};
    if (!ValidateCudaStateScience(
            *cpu_restart, restart_cuda, &restart_q12.integrals,
            &restart_q14.integrals,
            &cuda_current.steps[static_cast<std::size_t>(local_step)],
            &cuda_current.attempts[static_cast<std::size_t>(attempt_index)],
            current_q12, current_q14))
    {
      return false;
    }
    restart_q12 = current_q12;
    restart_q14 = current_q14;
  }
  if (cuda_restart->status() != scalar::ScalarCudaStatus::ready ||
      !ValidateTerminalRestartHistory(
          cuda_restart->trajectory_evidence()))
  {
    std::cerr << "C2 terminal restart history is not global-step rebased\n";
    return false;
  }
  scalar::PublishedState cpu_restarted_final{}, cuda_restarted_final{};
  if (!CopyPair(*cpu_restart, *cuda_restart, cpu_restarted_final,
                cuda_restarted_final, comparison) ||
      !oracle::SameStateBytes(cpu_final, cpu_restarted_final) ||
      !oracle::SameStateBytes(cuda_final, cuda_restarted_final))
  {
    return false;
  }
  const auto terminal_physical =
      std::make_unique<scalar::ScalarCudaEvidence>(
          cuda_restart->trajectory_evidence());
  if (cuda_restart->RestoreTrajectoryCheckpoint(cuda_checkpoint10) !=
          scalar::ScalarCudaStatus::ready ||
      !ValidateLogicalRebase(cuda_restart->trajectory_evidence(), 10) ||
      !RestorePhysicalEvidenceAdvanced(
          *terminal_physical, cuda_restart->trajectory_evidence()))
  {
    std::cerr << "C2 used-owner terminal restore did not rebase cleanly\n";
    return false;
  }
  const scalar::ScalarCudaStatus recoverable =
      cuda_restart->AdvancePersistentTrajectory(
          scalar::FailureInjection::local_factor);
  const auto& stale = cuda_restart->trajectory_evidence();
  if (recoverable != scalar::ScalarCudaStatus::accepted_after_one_retry ||
      stale.first_failure !=
          scalar::ScalarCudaStatus::injected_local_factor_failure ||
      stale.accepted_steps != 11 || stale.attempted_steps != 2 ||
      stale.retry_attempts != 1 || stale.attempt_count != 2 ||
      stale.step_evidence_count != 1 || stale.coordinates.step != 11 ||
      stale.attempts[0].status !=
          scalar::ScalarCudaStatus::injected_local_factor_failure ||
      stale.attempts[0].failure_cause !=
          scalar::ScalarCudaFailureCause::injected_local_factor ||
      stale.attempts[0].target_step != 11 ||
      stale.attempts[1].status !=
          scalar::ScalarCudaStatus::accepted_after_one_retry ||
      stale.attempts[1].target_step != 11 ||
      !stale.steps[0].accepted || stale.steps[0].step != 11 ||
      stale.steps[0].attempt_index != 1 ||
      !DefaultLogicalArrays(stale, 2, 1))
  {
    std::cerr << "C2 used-owner recoverable history was not retained\n";
    return false;
  }
  const auto stale_physical =
      std::make_unique<scalar::ScalarCudaEvidence>(stale);
  if (cuda_restart->RestoreTrajectoryCheckpoint(cuda_checkpoint10) !=
          scalar::ScalarCudaStatus::ready ||
      !ValidateLogicalRebase(cuda_restart->trajectory_evidence(), 10) ||
      !RestorePhysicalEvidenceAdvanced(
          *stale_physical, cuda_restart->trajectory_evidence()))
  {
    std::cerr << "C2 used-owner second restore retained stale history\n";
    return false;
  }
  cuda_restart.reset();

  const auto& evidence = *continuous_evidence;
  const auto& transfers = evidence.transfers;
  const auto& allocations = evidence.allocation_ledger;
  const bool trajectory_gate = evidence.accepted_steps == 20 &&
      evidence.step_evidence_count == 20 &&
      evidence.attempt_count == 20;
  const bool solver_policy_gate = evidence.trace_restart == 50 &&
      evidence.trace_maximum_iterations == 200 &&
      evidence.trace_relative_tolerance == 1.0e-12 &&
      evidence.trace_absolute_tolerance == 1.0e-14 &&
      evidence.ilu_type == 0 && evidence.ilu_level_of_fill == 0 &&
      evidence.ilu_local_reordering == 1;
  const bool residency_gate = evidence.residency.mfem_cuda_device &&
      evidence.residency.hypre_cuda && evidence.residency.umpire_device_pool &&
      evidence.residency.state_device_resident &&
      evidence.residency.topology_device_resident &&
      evidence.residency.workspace_device_resident &&
      evidence.residency.trace_solve_device_resident &&
      !evidence.residency.host_numerical_fallback;
  const bool allocation_plateau_gate = evidence.allocations.warm_plateau &&
      evidence.allocations.baseline_mfem_high_water != 0 &&
      evidence.allocations.baseline_hypre_high_water != 0 &&
      evidence.allocations.current_mfem_high_water ==
          evidence.allocations.baseline_mfem_high_water &&
      evidence.allocations.current_hypre_high_water ==
          evidence.allocations.baseline_hypre_high_water &&
      evidence.allocations.zero_post_baseline_observed_growth &&
      evidence.allocations.zero_post_baseline_owned_allocation &&
      evidence.allocations.zero_post_baseline_owned_deallocation;
  const bool reuse_identity_gate = oracle::ValidateTopologyIdentity(reuse);
  const bool mutated_reuse_rejected =
      !oracle::ValidateTopologyIdentity(mutated_reuse);
  const bool reuse_gate = reuse.unchanged && reuse_identity_gate &&
      mutated_reuse_rejected;
  const bool launch_gate = evidence.launches.assemble_launches > 0 &&
      evidence.launches.condense_launches > 0 &&
      evidence.launches.recover_launches > 0 &&
      evidence.launches.fallback_calls == 0 &&
      evidence.launches.last_compiled_architecture == 1200 &&
      evidence.launches.all_launches_completed;
  const bool persistent_gate = trajectory_gate && solver_policy_gate &&
      residency_gate && allocation_plateau_gate && reuse_gate && launch_gate;
  const bool transfer_gate = oracle::ValidateTransferLedger(
      transfers, evidence, oracle::c2_transfer_expectation);
  const bool transfer_mutation_gate = RejectTransferCensusMutations(
      transfers, evidence, oracle::c2_transfer_expectation);
  const bool allocation_gate =
      oracle::ValidateAllocationLedger(allocations, evidence.allocations);
  const bool copy_mutation_gate = RunCopiedEvidenceMutationNegatives(evidence);
  const bool checkpoint_negative_gate =
      RunCheckpointNegatives(cuda_checkpoint10, *cpu_restart);
  std::cout << "C2 terminal-gates persistent=" << persistent_gate
            << " trajectory=" << trajectory_gate
            << " solver-policy=" << solver_policy_gate
            << " residency=" << residency_gate
            << " allocation-plateau=" << allocation_plateau_gate
            << " reuse=" << reuse_gate
            << " launches=" << launch_gate
            << " transfer=" << transfer_gate
            << " transfer-mutation=" << transfer_mutation_gate
            << " allocation=" << allocation_gate
            << " copy-mutation=" << copy_mutation_gate
            << " checkpoint-negative=" << checkpoint_negative_gate << '\n';
  if (!persistent_gate || !transfer_gate || !transfer_mutation_gate ||
      !allocation_gate || !copy_mutation_gate || !checkpoint_negative_gate)
  {
    std::cerr << "C2 one or more independent terminal gates failed\n";
    return false;
  }
  std::cout << "PASS: HDG-EXP-SCALAR-DEVICE-01 C2 20-step CPU/CUDA "
            << "max-relative-L2=" << comparison.maximum_relative_l2
            << " auxiliary=" << auxiliary.normalized_maximum << '\n';
  std::cout << "SC25_CUDA_RANK1_C2_PASS\n";
  return comparison.maximum_relative_l2 <= comparison_tolerance;
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1)
  {
    std::cerr << "HDG-EXP-SCALAR-DEVICE-01 is rank-one only\n";
    return 1;
  }
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool pass = mode == "--c1" ? RunC1() :
      mode == "--c2" ? (RunC1() && RunC2()) : false;
  if (!pass)
  {
    std::cerr << "HDG-EXP-SCALAR-DEVICE-01 CUDA rank-one audit failed\n";
  }
  return pass ? 0 : 1;
}
