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
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>


namespace
{
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25;
namespace oracle = ::dg_igr_hdg_sc25_cuda_test;

bool All(const bool local) noexcept
{
  int value = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && result == 1;
}

std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank1> Create(
    const bool coupled = false)
{
  scalar::ScalarCudaConfig config{};
  config.polynomial_degree = 1;
  config.elements_per_direction = coupled ? 3 : 8;
  config.requested_device = 0;
  config.coupled_operator_fixture = coupled;
  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank1> owner;
  if (scalar::PersistentSc25ScalarIgrCudaRank1::Create(
          MPI_COMM_WORLD, config, owner) != scalar::ScalarCudaStatus::ready)
    return nullptr;
  return owner;
}

bool RunC1()
{
  auto owner = Create(true);
  if (!owner) return false;
  scalar::CoupledOperatorSnapshot snapshot{};
  const auto& evidence = owner->operator_evidence();
  const bool valid = owner->ExportIndependentOperator(snapshot) &&
      evidence.elements == 9 && evidence.full_unknowns == 432 &&
      evidence.true_trace_unknowns == 180 && evidence.mfem_periodic_topology &&
      evidence.device_assembly && evidence.device_condensation &&
      evidence.device_recovery && evidence.hypre_device_trace_solve &&
      evidence.launches.assemble_launches > 0 &&
      evidence.launches.condense_launches > 0 &&
      evidence.launches.recover_launches > 0 &&
      evidence.launches.fallback_calls == 0 &&
      evidence.launches.last_compiled_architecture ==
          scalar::required_native_architecture &&
      evidence.launches.all_launches_completed;
  const bool passed = All(valid);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (passed && rank == 0) { std::cout << "SC25_CUDA_RANK1_C1_PASS\n"; }
  return passed;
}

bool CompactRatioNormalization()
{
  double maximum = 0.0;
  scalar::detail::UpdateMaximumNormalizedResidual(
      maximum, 100.0, 100.0, 0.0, 1.0);
  scalar::detail::UpdateMaximumNormalizedResidual(
      maximum, 2.0, 1.0, 0.0, 1.0);
  const double zero_bound = scalar::detail::NormalizedResidualRatio(
      0.0, 0.0, 0.0, 0.0);
  const double nonzero_zero_bound = scalar::detail::NormalizedResidualRatio(
      1.0, 0.0, 0.0, 0.0);
  const double invalid = scalar::detail::NormalizedResidualRatio(
      std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0, 1.0);
  double invalid_local = 0.0;
  scalar::detail::UpdateMaximumLocalApplicationResidual(
      invalid_local, 1.0, -1.0);
  return maximum == 2.0 && zero_bound == 0.0 &&
      std::isinf(nonzero_zero_bound) && std::isinf(invalid) &&
      std::isinf(invalid_local);
}

bool RunC2()
{
  if (!CompactRatioNormalization()) return false;
  auto owner = Create();
  if (!owner || owner->InitializePersistentTrajectory() !=
      scalar::ScalarCudaStatus::ready)
    return false;
  scalar::PublishedState initial{};
  if (!owner->CopyPublishedTrajectoryState(initial)) return false;

  const scalar::ScalarCudaEvidence live_before_stage =
      owner->trajectory_evidence();
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction staged{};
  scalar::ScalarCudaCheckpoint staged_checkpoint{};
  scalar::ScalarCudaEvidence prospective{};
  if (owner->PreparePersistentTrajectoryTransaction(staged) !=
          scalar::ScalarCudaStatus::accepted ||
      !owner->InstallPersistentTrajectoryTransaction(staged) ||
      !owner->StageInstalledTrajectoryTransaction(
          staged, staged_checkpoint, prospective))
    return false;
  const auto& live_during_stage = owner->trajectory_evidence();
  const bool staged_metrics =
      staged_checkpoint.descriptor.coordinates.step == 1 &&
      prospective.status == scalar::ScalarCudaStatus::accepted &&
      prospective.accepted_steps == 1 && prospective.attempted_steps == 1 &&
      prospective.rejected_steps == 0 && prospective.retry_attempts == 0 &&
      prospective.last_accepted_attempt.target_step == 1 &&
      prospective.last_accepted_step.accepted &&
      prospective.last_accepted_step.step == 1 &&
      prospective.last_accepted_step.attempt_index == 0 &&
      std::isfinite(prospective.minimum_density) &&
      prospective.minimum_density > 0.0 &&
      std::isfinite(prospective.minimum_pressure) &&
      prospective.minimum_pressure > 0.0 &&
      std::isfinite(prospective.minimum_speed_margin) &&
      prospective.minimum_speed_margin > 0.0 &&
      std::isfinite(prospective.maximum_trace_residual_ratio) &&
      prospective.maximum_trace_residual_ratio >= 0.0 &&
      prospective.maximum_trace_residual_ratio <= 1.0 &&
      std::isfinite(prospective.maximum_local_application_residual_ratio) &&
      prospective.maximum_local_application_residual_ratio >= 0.0;
  const bool live_unchanged =
      live_during_stage.status == live_before_stage.status &&
      live_during_stage.first_failure == live_before_stage.first_failure &&
      live_during_stage.accepted_steps == live_before_stage.accepted_steps &&
      live_during_stage.attempted_steps == live_before_stage.attempted_steps &&
      live_during_stage.rejected_steps == live_before_stage.rejected_steps &&
      live_during_stage.retry_attempts == live_before_stage.retry_attempts &&
      live_during_stage.maximum_attempted_newton_iterations ==
          live_before_stage.maximum_attempted_newton_iterations &&
      live_during_stage.maximum_newton_iterations ==
          live_before_stage.maximum_newton_iterations &&
      live_during_stage.maximum_trace_iterations ==
          live_before_stage.maximum_trace_iterations &&
      std::isinf(live_during_stage.minimum_density) &&
      std::isinf(live_during_stage.minimum_pressure) &&
      std::isinf(live_during_stage.minimum_speed_margin) &&
      live_during_stage.maximum_trace_residual_ratio == 0.0 &&
      live_during_stage.maximum_local_application_residual_ratio == 0.0 &&
      live_during_stage.coordinates.step == live_before_stage.coordinates.step &&
      live_during_stage.last_accepted_step.attempt_index ==
          live_before_stage.last_accepted_step.attempt_index;
  if (!staged_metrics || !live_unchanged ||
      !owner->FinalizePersistentTrajectoryTransaction(std::move(staged)))
    return false;
  const auto& finalized_stage = owner->trajectory_evidence();
  const bool finalize_matches_staged =
      finalized_stage.status == prospective.status &&
      finalized_stage.accepted_steps == prospective.accepted_steps &&
      finalized_stage.attempted_steps == prospective.attempted_steps &&
      finalized_stage.rejected_steps == prospective.rejected_steps &&
      finalized_stage.retry_attempts == prospective.retry_attempts &&
      finalized_stage.maximum_attempted_newton_iterations ==
          prospective.maximum_attempted_newton_iterations &&
      finalized_stage.maximum_newton_iterations ==
          prospective.maximum_newton_iterations &&
      finalized_stage.maximum_trace_iterations ==
          prospective.maximum_trace_iterations &&
      finalized_stage.minimum_density == prospective.minimum_density &&
      finalized_stage.minimum_pressure == prospective.minimum_pressure &&
      finalized_stage.minimum_speed_margin == prospective.minimum_speed_margin &&
      finalized_stage.maximum_conservation_error ==
          prospective.maximum_conservation_error &&
      finalized_stage.maximum_nonlinear_residual_ratio ==
          prospective.maximum_nonlinear_residual_ratio &&
      finalized_stage.maximum_trace_residual_ratio ==
          prospective.maximum_trace_residual_ratio &&
      finalized_stage.maximum_local_application_residual_ratio ==
          prospective.maximum_local_application_residual_ratio &&
      finalized_stage.last_attempt.status == prospective.last_attempt.status &&
      finalized_stage.last_attempt.target_step ==
          prospective.last_attempt.target_step &&
      finalized_stage.last_accepted_attempt.status ==
          prospective.last_accepted_attempt.status &&
      finalized_stage.last_accepted_attempt.target_step ==
          prospective.last_accepted_attempt.target_step &&
      finalized_stage.last_accepted_step.accepted &&
      finalized_stage.last_accepted_step.step == 1 &&
      finalized_stage.last_accepted_step.attempt_index == 0 &&
      finalized_stage.coordinates.step == prospective.coordinates.step;
  if (!finalize_matches_staged) return false;

  scalar::ScalarCudaCheckpoint checkpoint{};
  scalar::PublishedState committed{};
  if (!owner->CaptureTrajectoryCheckpoint(checkpoint) ||
      !owner->CopyPublishedTrajectoryState(committed))
    return false;
  if (owner->AdvancePersistentTrajectory(
          scalar::FailureInjection::admissibility) !=
          scalar::ScalarCudaStatus::accepted_after_one_retry)
    return false;
  const auto& observed_rejection = owner->trajectory_evidence();
  const bool observed_ratios = observed_rejection.rejected_steps == 1 &&
      observed_rejection.retry_attempts == 1 &&
      observed_rejection.first_failed_attempt.status !=
          scalar::ScalarCudaStatus::uninitialized &&
      std::isfinite(
          observed_rejection.first_failed_attempt.maximum_trace_residual_ratio) &&
      observed_rejection.first_failed_attempt.maximum_trace_residual_ratio >= 0.0 &&
      std::isfinite(observed_rejection.first_failed_attempt
                        .maximum_local_application_residual_ratio) &&
      observed_rejection.first_failed_attempt
              .maximum_local_application_residual_ratio >= 0.0 &&
      observed_rejection.maximum_trace_residual_ratio >=
          observed_rejection.first_failed_attempt.maximum_trace_residual_ratio &&
      observed_rejection.maximum_local_application_residual_ratio >=
          observed_rejection.first_failed_attempt
              .maximum_local_application_residual_ratio;
  if (!observed_ratios ||
      owner->RestoreTrajectoryCheckpoint(checkpoint) !=
          scalar::ScalarCudaStatus::ready)
    return false;
  scalar::PublishedState restored_state{};
  scalar::ScalarCudaCheckpoint restored_checkpoint{};
  if (!owner->CopyPublishedTrajectoryState(restored_state) ||
      !owner->CaptureTrajectoryCheckpoint(restored_checkpoint) ||
      !oracle::SameStateBytes(committed, restored_state) ||
      checkpoint.interior_component_slabs !=
          restored_checkpoint.interior_component_slabs ||
      checkpoint.trace_component_slabs !=
          restored_checkpoint.trace_component_slabs ||
      checkpoint.raw_field_hashes.interior !=
          restored_checkpoint.raw_field_hashes.interior ||
      checkpoint.raw_field_hashes.trace != restored_checkpoint.raw_field_hashes.trace ||
      checkpoint.canonical_manifest_hash !=
          restored_checkpoint.canonical_manifest_hash)
    return false;
  const auto& reset = owner->trajectory_evidence();
  const bool reset_records =
      reset.first_failed_attempt.status ==
          scalar::ScalarCudaStatus::uninitialized &&
      reset.first_failed_attempt.target_step == 0 &&
      reset.last_attempt.status == scalar::ScalarCudaStatus::uninitialized &&
      reset.last_attempt.target_step == 0 &&
      reset.last_accepted_attempt.status ==
          scalar::ScalarCudaStatus::uninitialized &&
      reset.last_accepted_attempt.target_step == 0 &&
      !reset.last_accepted_step.accepted && reset.last_accepted_step.step == 0 &&
      reset.last_accepted_step.attempt_index == -1;
  const bool reset_compact =
      reset.status == scalar::ScalarCudaStatus::ready &&
      reset.first_failure == scalar::ScalarCudaStatus::uninitialized &&
      reset.accepted_steps ==
          static_cast<int>(checkpoint.descriptor.coordinates.step) &&
      reset.attempted_steps == 0 && reset.rejected_steps == 0 &&
      reset.retry_attempts == 0 &&
      reset.maximum_attempted_newton_iterations == 0 &&
      reset.maximum_newton_iterations == 0 &&
      reset.maximum_trace_iterations == 0 &&
      std::isinf(reset.minimum_density) && reset.minimum_density > 0.0 &&
      std::isinf(reset.minimum_pressure) && reset.minimum_pressure > 0.0 &&
      std::isinf(reset.minimum_speed_margin) &&
      reset.minimum_speed_margin > 0.0 &&
      reset.maximum_conservation_error == 0.0 &&
      reset.maximum_nonlinear_residual_ratio == 0.0 &&
      reset.maximum_trace_residual_ratio == 0.0 &&
      reset.maximum_local_application_residual_ratio == 0.0 &&
      reset.coordinates.step == checkpoint.descriptor.coordinates.step &&
      reset.coordinates.generation ==
          checkpoint.descriptor.coordinates.generation &&
      reset.coordinates.cursor == checkpoint.descriptor.coordinates.cursor &&
      reset.coordinates.time == checkpoint.descriptor.coordinates.time &&
      reset.coordinates.dt == checkpoint.descriptor.coordinates.dt;
  if (!reset_compact || !reset_records ||
      owner->PreparePersistentTrajectoryTransaction(
          staged, scalar::FailureInjection::local_factor) !=
          scalar::ScalarCudaStatus::injected_local_factor_failure)
    return false;
  const auto& early_failure = owner->trajectory_evidence();
  const bool failed_minima = early_failure.accepted_steps == 1 &&
      early_failure.attempted_steps == 1 && early_failure.rejected_steps == 1 &&
      early_failure.retry_attempts == 1 &&
      early_failure.first_failure ==
          scalar::ScalarCudaStatus::injected_local_factor_failure &&
      early_failure.first_failed_attempt.target_step == 2 &&
      early_failure.first_failed_attempt.retry_index == 0 &&
      std::isinf(early_failure.minimum_density) &&
      early_failure.minimum_density > 0.0 &&
      std::isinf(early_failure.minimum_pressure) &&
      early_failure.minimum_pressure > 0.0 &&
      std::isinf(early_failure.minimum_speed_margin) &&
      early_failure.minimum_speed_margin > 0.0 &&
      early_failure.maximum_trace_residual_ratio == 0.0 &&
      early_failure.maximum_local_application_residual_ratio == 0.0 &&
      early_failure.first_failed_attempt.maximum_trace_residual_ratio == 0.0 &&
      early_failure.first_failed_attempt
              .maximum_local_application_residual_ratio == 0.0;
  scalar::PersistentSc25ScalarIgrCudaRank1::Transaction retry{};
  if (!failed_minima ||
      owner->PreparePersistentTrajectoryTransaction(retry) !=
          scalar::ScalarCudaStatus::accepted ||
      !owner->InstallPersistentTrajectoryTransaction(retry) ||
      !owner->FinalizePersistentTrajectoryTransaction(std::move(retry)))
    return false;
  scalar::PublishedState final_state{};
  oracle::StateQuadratureAudit audit{};
  if (!owner->CopyPublishedTrajectoryState(final_state) ||
      !oracle::AuditPublishedState(final_state, 4, audit))
    return false;
  const auto& evidence = owner->trajectory_evidence();
  const bool compact = evidence.accepted_steps == 2 &&
      evidence.attempted_steps == 2 &&
      evidence.rejected_steps == 1 && evidence.retry_attempts == 1 &&
      evidence.first_failure == scalar::ScalarCudaStatus::injected_local_factor_failure &&
      evidence.first_failed_attempt.target_step == 2 &&
      evidence.first_failed_attempt.retry_index == 0 &&
      evidence.last_accepted_attempt.target_step == 2 &&
      evidence.last_accepted_attempt.retry_index == 1 &&
      evidence.last_accepted_step.step == 2 &&
      evidence.last_accepted_step.attempt_index == 1 &&
      evidence.coordinates.step == 2 && evidence.transfer_totals.complete &&
      evidence.transfer_totals.event_count != 0 &&
      evidence.residency.mfem_cuda_device && evidence.residency.hypre_cuda &&
      evidence.residency.umpire_device_pool &&
      evidence.residency.state_device_resident &&
      evidence.residency.trace_solve_device_resident &&
      !evidence.residency.host_numerical_fallback &&
      evidence.reuse.unchanged && evidence.launches.fallback_calls == 0 &&
      evidence.launches.last_compiled_architecture ==
          scalar::required_native_architecture &&
      std::isfinite(evidence.minimum_density) && evidence.minimum_density > 0.0 &&
      std::isfinite(evidence.minimum_pressure) && evidence.minimum_pressure > 0.0 &&
      std::isfinite(evidence.minimum_speed_margin) &&
      evidence.minimum_speed_margin > 0.0 &&
      std::isfinite(evidence.maximum_trace_residual_ratio) &&
      evidence.maximum_trace_residual_ratio >= 0.0 &&
      evidence.maximum_trace_residual_ratio <= 1.0 &&
      std::isfinite(evidence.maximum_local_application_residual_ratio) &&
      evidence.maximum_local_application_residual_ratio >= 0.0 &&
      std::isfinite(audit.minimum_density) && std::isfinite(audit.minimum_pressure) &&
      std::isfinite(audit.minimum_speed_margin) &&
      sizeof(scalar::ScalarCudaEvidence) < 4096;
  const bool passed = All(compact);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (passed && rank == 0) { std::cout << "SC25_CUDA_RANK1_C2_PASS\n"; }
  return passed;
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool pass = ranks == 1 && (mode == "--c1" ? RunC1() :
      mode == "--c2" ? (RunC1() && RunC2()) : false);
  if (!pass && ranks == 1)
    std::cerr << "compact SC25 CUDA rank-one audit failed\n";
  return pass ? 0 : 1;
}
