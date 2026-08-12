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
      evidence.launches.last_compiled_architecture == 1200 &&
      evidence.launches.all_launches_completed;
  return All(valid);
}

bool RunC2()
{
  auto owner = Create();
  if (!owner || owner->InitializePersistentTrajectory() !=
      scalar::ScalarCudaStatus::ready)
    return false;
  scalar::PublishedState initial{};
  if (!owner->CopyPublishedTrajectoryState(initial) ||
      owner->AdvancePersistentTrajectory() != scalar::ScalarCudaStatus::accepted)
    return false;
  scalar::ScalarCudaCheckpoint checkpoint{};
  scalar::PublishedState committed{};
  if (!owner->CaptureTrajectoryCheckpoint(checkpoint) ||
      !owner->CopyPublishedTrajectoryState(committed))
    return false;
  if (owner->AdvancePersistentTrajectory(
          scalar::FailureInjection::local_factor) !=
          scalar::ScalarCudaStatus::accepted_after_one_retry ||
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
      owner->AdvancePersistentTrajectory(
          scalar::FailureInjection::local_factor) !=
          scalar::ScalarCudaStatus::accepted_after_one_retry)
    return false;
  scalar::PublishedState final_state{};
  oracle::StateQuadratureAudit audit{};
  if (!owner->CopyPublishedTrajectoryState(final_state) ||
      !oracle::AuditPublishedState(final_state, 4, audit))
    return false;
  const auto& evidence = owner->trajectory_evidence();
  const bool compact = evidence.accepted_steps == 2 &&
      evidence.rejected_steps == 1 && evidence.retry_attempts == 1 &&
      evidence.first_failure == scalar::ScalarCudaStatus::injected_local_factor_failure &&
      evidence.first_failed_attempt.target_step == 2 &&
      evidence.last_accepted_attempt.target_step == 2 &&
      evidence.last_accepted_step.step == 2 &&
      evidence.coordinates.step == 2 && evidence.transfer_totals.complete &&
      evidence.transfer_totals.event_count != 0 &&
      evidence.residency.mfem_cuda_device && evidence.residency.hypre_cuda &&
      evidence.residency.umpire_device_pool &&
      evidence.residency.state_device_resident &&
      evidence.residency.trace_solve_device_resident &&
      !evidence.residency.host_numerical_fallback &&
      evidence.reuse.unchanged && evidence.launches.fallback_calls == 0 &&
      evidence.launches.last_compiled_architecture == 1200 &&
      std::isfinite(audit.minimum_density) && std::isfinite(audit.minimum_pressure) &&
      std::isfinite(audit.minimum_speed_margin) &&
      sizeof(scalar::ScalarCudaEvidence) < 4096;
  return All(compact);
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
