#include "sc25_scalar_igr_independent_oracle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace
{

namespace sc25 = dg::hdg::runtime::cpu::sc25;

std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
CreatePersistentFoundation()
{
  sc25::FoundationConfig config;
  config.polynomial_degree = 1;
  config.elements_per_direction = sc25::persistent_elements_per_direction;
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> foundation;
  if (sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuSerialTag{}, config, foundation) !=
          sc25::FoundationStatus::ready)
  {
    return {};
  }
  return foundation;
}

bool Accepted(const sc25::TrajectoryStatus status) noexcept
{
  return status == sc25::TrajectoryStatus::HdgSc25Accepted ||
      status == sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

template <std::size_t N>
bool EqualFp64Bytes(const std::array<double, N>& left,
                    const std::array<double, N>& right) noexcept
{
  return std::memcmp(left.data(), right.data(), N * sizeof(double)) == 0;
}

bool EqualStateBytes(const sc25::PersistentTrajectoryState& left,
                     const sc25::PersistentTrajectoryState& right) noexcept
{
  return EqualFp64Bytes(left.u, right.u) &&
      EqualFp64Bytes(left.sigma, right.sigma) &&
      EqualFp64Bytes(left.z, right.z) &&
      EqualFp64Bytes(left.uhat, right.uhat) &&
      EqualFp64Bytes(left.sigmahat, right.sigmahat);
}

dg::hdg::runtime::StateHash InteriorHash(
    const sc25::PersistentTrajectoryState& state) noexcept
{
  dg::hdg::runtime::Sha256 hash;
  dg::hdg::runtime::Sha256Digest digest{};
  if (!hash.Update(state.u.data(), state.u.size() * sizeof(double)) ||
      !hash.Update(state.sigma.data(),
                   state.sigma.size() * sizeof(double)) ||
      !hash.Update(state.z.data(), state.z.size() * sizeof(double)) ||
      !hash.Finalize(digest))
  {
    return {};
  }
  return {digest};
}

dg::hdg::runtime::StateHash TraceHash(
    const sc25::PersistentTrajectoryState& state) noexcept
{
  dg::hdg::runtime::Sha256 hash;
  dg::hdg::runtime::Sha256Digest digest{};
  if (!hash.Update(state.uhat.data(), state.uhat.size() * sizeof(double)) ||
      !hash.Update(state.sigmahat.data(),
                   state.sigmahat.size() * sizeof(double)) ||
      !hash.Finalize(digest))
  {
    return {};
  }
  return {digest};
}

template <typename Array>
dg::hdg::runtime::StateHash FieldHash(const Array& values) noexcept
{
  return {dg::hdg::runtime::ComputeSha256(
      values.data(), values.size() * sizeof(double))};
}

class IndependentManifestBuilder final
{
public:
  bool Text(const std::string_view value) noexcept
  {
    return U64(value.size()) && hash_.Update(value.data(), value.size());
  }

  bool U64(const std::uint64_t value) noexcept
  {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
      bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
    return hash_.Update(bytes.data(), bytes.size());
  }

  bool Hash(const dg::hdg::runtime::StateHash& value) noexcept
  {
    return hash_.Update(value.bytes.data(), value.bytes.size());
  }

  dg::hdg::runtime::StateHash Finish() noexcept
  {
    dg::hdg::runtime::Sha256Digest digest{};
    return hash_.Finalize(digest) ? dg::hdg::runtime::StateHash{digest} :
                                    dg::hdg::runtime::StateHash{};
  }

private:
  dg::hdg::runtime::Sha256 hash_;
};

dg::hdg::runtime::StateHash IndependentManifestHash(
    const sc25::PersistentTrajectoryCheckpoint& checkpoint) noexcept
{
  IndependentManifestBuilder builder;
  const dg::hdg::runtime::CheckpointDescriptor& descriptor =
      checkpoint.descriptor;
  const dg::hdg::runtime::RuntimeTupleDescriptor& tuple = descriptor.tuple;
  bool valid = builder.Text(sc25::persistent_checkpoint_manifest_id) &&
      builder.Text(checkpoint.projection.view()) &&
      builder.Text(checkpoint.tuple.view()) &&
      builder.Text(checkpoint.scenario.view()) &&
      builder.U64(descriptor.schema_version) &&
      builder.U64(descriptor.successful_commit ? 1U : 0U) &&
      builder.U64(static_cast<std::uint64_t>(descriptor.restart_mode)) &&
      builder.Text(tuple.model.view()) &&
      builder.Text(tuple.formulation.view()) &&
      builder.Text(tuple.trace.view()) &&
      builder.Text(tuple.stabilization.view()) &&
      builder.Text(tuple.time_treatment.view()) &&
      builder.Text(tuple.backend.view()) &&
      builder.Text(tuple.precision.view()) &&
      builder.Text(descriptor.mfem_pin.view()) &&
      builder.Hash(descriptor.mesh_hash) &&
      builder.Hash(descriptor.partition_hash) &&
      builder.Hash(descriptor.topology_hash) &&
      builder.U64(descriptor.rank_count) &&
      builder.U64(static_cast<std::uint64_t>(descriptor.polynomial_order)) &&
      builder.Text(descriptor.layout.view()) &&
      builder.Text(descriptor.precision.view()) &&
      builder.U64(descriptor.cursor) &&
      builder.U64(dg::hdg::runtime::detail::DoubleBits(descriptor.dt)) &&
      builder.U64(dg::hdg::runtime::detail::DoubleBits(
          descriptor.coordinates.time)) &&
      builder.U64(descriptor.coordinates.step) &&
      builder.U64(descriptor.coordinates.generation) &&
      builder.U64(descriptor.coordinates.cursor) &&
      builder.U64(dg::hdg::runtime::detail::DoubleBits(
          descriptor.coordinates.dt)) &&
      builder.U64(descriptor.coordinate_fingerprint) &&
      builder.Hash(descriptor.interior_hash) &&
      builder.Hash(descriptor.trace_hash);
  for (const sc25::TrajectoryCheckpointFieldIdentity& field :
       checkpoint.fields)
  {
    valid = valid && builder.Text(field.name.view()) &&
        builder.U64(field.fp64_entries) && builder.U64(field.canonical_order);
  }
  const sc25::TrajectoryStateHashes& hashes = checkpoint.raw_field_hashes;
  valid = valid && builder.Hash(hashes.u) && builder.Hash(hashes.sigma) &&
      builder.Hash(hashes.z) && builder.Hash(hashes.uhat) &&
      builder.Hash(hashes.sigmahat);
  return valid ? builder.Finish() : dg::hdg::runtime::StateHash{};
}

void Rehash(sc25::PersistentTrajectoryCheckpoint& checkpoint) noexcept
{
  checkpoint.descriptor.interior_hash = InteriorHash(checkpoint.state);
  checkpoint.descriptor.trace_hash = TraceHash(checkpoint.state);
  checkpoint.raw_field_hashes = {
      FieldHash(checkpoint.state.u), FieldHash(checkpoint.state.sigma),
      FieldHash(checkpoint.state.z), FieldHash(checkpoint.state.uhat),
      FieldHash(checkpoint.state.sigmahat),
      checkpoint.descriptor.interior_hash,
      checkpoint.descriptor.trace_hash};
  checkpoint.canonical_manifest_hash = IndependentManifestHash(checkpoint);
}

bool EqualHashes(const sc25::TrajectoryStateHashes& left,
                 const sc25::TrajectoryStateHashes& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma &&
      left.z == right.z && left.uhat == right.uhat &&
      left.sigmahat == right.sigmahat &&
      left.interior == right.interior && left.trace == right.trace;
}

bool EqualCoordinates(const dg::hdg::runtime::StateCoordinates& left,
                      const dg::hdg::runtime::StateCoordinates& right)
    noexcept
{
  return dg::hdg::runtime::detail::DoubleBits(left.time) ==
          dg::hdg::runtime::detail::DoubleBits(right.time) &&
      left.step == right.step && left.generation == right.generation &&
      left.cursor == right.cursor &&
      dg::hdg::runtime::detail::DoubleBits(left.dt) ==
          dg::hdg::runtime::detail::DoubleBits(right.dt);
}

bool EqualReuse(const sc25::TrajectoryReuseSnapshot& left,
                const sc25::TrajectoryReuseSnapshot& right) noexcept
{
  return left.foundation_handle == right.foundation_handle &&
      left.topology_handle == right.topology_handle &&
      left.carrier_map_handle == right.carrier_map_handle &&
      left.contract_handle == right.contract_handle &&
      left.assembly_state_handle == right.assembly_state_handle &&
      left.assembly_handle == right.assembly_handle &&
      left.trial_assembly_handle == right.trial_assembly_handle &&
      left.factor_handle == right.factor_handle &&
      left.workspace_handle == right.workspace_handle &&
      left.quadrature_workspace_handle == right.quadrature_workspace_handle &&
      left.local_pattern_handle == right.local_pattern_handle &&
      left.true_matrix_handle == right.true_matrix_handle &&
      left.true_rhs_handle == right.true_rhs_handle &&
      left.true_solution_handle == right.true_solution_handle &&
      left.ilu_handle == right.ilu_handle &&
      left.gmres_handle == right.gmres_handle &&
      left.true_residual_handle == right.true_residual_handle &&
      left.element_to_true_csr_handle == right.element_to_true_csr_handle &&
      left.owned_boundary_identity_hash ==
          right.owned_boundary_identity_hash &&
      left.topology_generation == right.topology_generation &&
      left.storage_generation == right.storage_generation &&
      left.factor_generation == right.factor_generation &&
      left.workspace_generation == right.workspace_generation &&
      left.local_pattern_generation == right.local_pattern_generation &&
      left.solver_generation == right.solver_generation &&
      left.w8_owned_boundary_object_count ==
          right.w8_owned_boundary_object_count &&
      left.w8_owned_boundary_buffer_count ==
          right.w8_owned_boundary_buffer_count &&
      !left.opaque_mfem_hypre_observed &&
      !right.opaque_mfem_hypre_observed;
}

int RunTrajectoryHarness()
{
  int failures = 0;
  auto fail = [&failures](const std::string_view message)
  {
    std::cerr << "rank 0: " << message << '\n';
    ++failures;
  };
  sc25::PersistentTrajectoryState positive_zero_state{};
  sc25::PersistentTrajectoryState negative_zero_state{};
  negative_zero_state.u[0] = -0.0;
  if (EqualStateBytes(positive_zero_state, negative_zero_state))
  {
    fail("bytewise state comparison signed-zero negative control");
  }
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> baseline =
      CreatePersistentFoundation();
  if (!baseline)
  {
    fail("persistent trajectory foundation creation");
    return failures;
  }
  const sc25::TrajectoryStatus initialization =
      baseline->InitializePersistentTrajectory();
  if (initialization != sc25::TrajectoryStatus::ready)
  {
    std::cerr << "rank 0: persistent trajectory initialization status "
              << static_cast<int>(initialization) << '\n';
    return 1;
  }
  sc25::PersistentTrajectoryCheckpoint checkpoint10;
  for (int step = 1; step <= sc25::persistent_required_steps; ++step)
  {
    const sc25::TrajectoryStatus status =
        baseline->AdvancePersistentTrajectory();
    if (!Accepted(status))
    {
      std::cerr << "rank 0: baseline trajectory step " << step
                << " status " << static_cast<int>(status) << '\n';
      return 1;
    }
    if (step == 10 && !baseline->CaptureTrajectoryCheckpoint(checkpoint10))
    {
      fail("step-10 checkpoint capture");
      return failures;
    }
  }
  dg_igr_hdg_sc25_cpu_test::IndependentTrajectoryEvidence oracle;
  if (!dg_igr_hdg_sc25_cpu_test::RunIndependentTrajectoryOracle(
          *baseline, oracle))
  {
    for (const std::string& failure : oracle.failures) { fail(failure); }
  }
  sc25::PersistentTrajectoryState baseline_final;
  if (!baseline->CopyPublishedTrajectoryState(baseline_final))
  {
    fail("baseline final-state copy");
    return failures;
  }

  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> replay =
      CreatePersistentFoundation();
  if (!replay || replay->InitializePersistentTrajectory() !=
                     sc25::TrajectoryStatus::ready ||
      !Accepted(replay->RestoreTrajectoryCheckpoint(checkpoint10)))
  {
    fail("step-10 checkpoint restore");
  }
  else
  {
    sc25::PersistentTrajectoryState restored_start;
    const sc25::PersistentTrajectoryEvidence& restored_evidence =
        replay->trajectory_evidence();
    if (!replay->CopyPublishedTrajectoryState(restored_start) ||
        !EqualStateBytes(restored_start, checkpoint10.state) ||
        restored_evidence.checkpoint_step_offset != 10 ||
        restored_evidence.step_evidence_count != 0 ||
        restored_evidence.attempt_count != 0 ||
        !EqualCoordinates(restored_evidence.continuation_start_coordinates,
                          checkpoint10.descriptor.coordinates) ||
        !EqualStateBytes(restored_evidence.continuation_start_state,
                         checkpoint10.state) ||
        !EqualHashes(restored_evidence.continuation_start_hashes,
                     checkpoint10.raw_field_hashes) ||
        !restored_evidence.restored_loaded_state_match ||
        !restored_evidence.restored_loaded_coordinates_match ||
        !restored_evidence.restored_loaded_hashes_match)
    {
      fail("step-10 immediate loaded checkpoint equality/evidence reset");
    }
    for (const sc25::TrajectoryStepEvidence& stale :
         restored_evidence.steps)
    {
      if (stale.accepted)
      {
        fail("step-10 restore retained stale step evidence");
        break;
      }
    }
    for (int step = 11; step <= sc25::persistent_required_steps; ++step)
    {
      const sc25::TrajectoryStatus status =
          replay->AdvancePersistentTrajectory();
      if (!Accepted(status))
      {
        fail("restored trajectory advance");
        break;
      }
    }
    sc25::PersistentTrajectoryState replay_final;
    if (!replay->CopyPublishedTrajectoryState(replay_final) ||
        !EqualStateBytes(replay_final, baseline_final))
    {
      fail("step-10 restart bitwise replay");
    }
    const sc25::PersistentTrajectoryEvidence& continuation =
        replay->trajectory_evidence();
    if (continuation.checkpoint_step_offset != 10 ||
        continuation.step_evidence_count != 10 ||
        continuation.attempt_count != 10 ||
        continuation.steps[0].step != 11 ||
        continuation.steps[9].step != 20)
    {
      fail("step-10 honest continuation record window");
    }
    dg_igr_hdg_sc25_cpu_test::IndependentTrajectoryEvidence replay_oracle;
    if (!dg_igr_hdg_sc25_cpu_test::RunIndependentTrajectoryOracle(
            *replay, replay_oracle))
    {
      for (const std::string& failure : replay_oracle.failures)
      {
        fail(std::string("restored continuation ") + failure);
      }
    }
  }

  struct FailureCampaign final
  {
    sc25::TrajectoryFailureInjection injection;
    sc25::TrajectoryStatus expected_failure;
    std::string_view label;
  };
  constexpr std::array<FailureCampaign, 4> campaigns{{
      {sc25::TrajectoryFailureInjection::local_factor,
       sc25::TrajectoryStatus::HdgSc25InjectedLocalFactorFailure,
       "local-factor"},
      {sc25::TrajectoryFailureInjection::trace_linear,
       sc25::TrajectoryStatus::HdgSc25InjectedTraceLinearFailure,
       "trace-linear"},
      {sc25::TrajectoryFailureInjection::admissibility,
       sc25::TrajectoryStatus::HdgSc25InjectedAdmissibilityFailure,
       "all-trials-inadmissible"},
      {sc25::TrajectoryFailureInjection::newton_iteration_limit,
       sc25::TrajectoryStatus::HdgSc25InjectedNewtonIterationLimit,
       "Newton-iteration-limit"}}};
  for (const FailureCampaign& campaign : campaigns)
  {
    const std::string prefix = std::string(campaign.label) + " campaign ";
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> injected =
        CreatePersistentFoundation();
    if (!injected || injected->InitializePersistentTrajectory() !=
                         sc25::TrajectoryStatus::ready)
    {
      fail(prefix + "initialization");
      continue;
    }
    bool pre_failure_valid = true;
    for (int step = 1; step <= 6; ++step)
    {
      if (!Accepted(injected->AdvancePersistentTrajectory()))
      {
        fail(prefix + "pre-failure advance");
        pre_failure_valid = false;
        break;
      }
    }
    if (!pre_failure_valid) { continue; }

    sc25::PersistentTrajectoryState step6;
    if (!injected->CopyPublishedTrajectoryState(step6))
    {
      fail(prefix + "step-6 snapshot");
      continue;
    }
    const sc25::TrajectoryStatus retry_status =
        injected->AdvancePersistentTrajectory(campaign.injection);
    const sc25::PersistentTrajectoryEvidence& retry_evidence =
        injected->trajectory_evidence();
    const sc25::TrajectoryAttemptEvidence& failed =
        retry_evidence.attempts[6];
    const sc25::TrajectoryAttemptEvidence& retry =
        retry_evidence.attempts[7];
    const dg::hdg::runtime::StateCoordinates step6_coordinates{
        6.0 * sc25::persistent_dt, 6, 6, 6, sc25::persistent_dt};
    const dg::hdg::runtime::StateCoordinates step7_coordinates{
        7.0 * sc25::persistent_dt, 7, 7, 7, sc25::persistent_dt};
    const bool exact_failed_rollback =
        EqualHashes(failed.entry_hashes, failed.candidate_hashes) &&
        EqualHashes(failed.entry_hashes, failed.published_hashes) &&
        EqualCoordinates(failed.committed_coordinates, step6_coordinates) &&
        EqualCoordinates(failed.published_coordinates, step6_coordinates);
    const bool one_same_dt_retry = retry_status ==
            sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry &&
        retry_evidence.retry_attempts == 1 &&
        retry_evidence.attempt_count == 8 &&
        failed.status == campaign.expected_failure &&
        retry.status ==
            sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry &&
        failed.target_step == 7 && failed.retry_index == 0 &&
        retry.target_step == 7 && retry.retry_index == 1 &&
        dg::hdg::runtime::detail::DoubleBits(failed.dt) ==
            dg::hdg::runtime::detail::DoubleBits(retry.dt) &&
        dg::hdg::runtime::detail::DoubleBits(failed.dt) ==
            dg::hdg::runtime::detail::DoubleBits(sc25::persistent_dt) &&
        EqualCoordinates(retry.committed_coordinates, step6_coordinates) &&
        EqualCoordinates(retry.published_coordinates, step7_coordinates);
    const bool exact_newton_limit = campaign.injection !=
            sc25::TrajectoryFailureInjection::newton_iteration_limit ||
        (failed.attempted_newton_iterations ==
             sc25::persistent_newton_maximum_iterations &&
         failed.newton_iterations == 0);
    const sc25::TrajectoryReuseSnapshot& reuse =
        retry_evidence.reuse.baseline;
    const bool stable_storage = EqualReuse(failed.entry_reuse, reuse) &&
        EqualReuse(failed.exit_reuse, reuse) &&
        EqualReuse(retry.entry_reuse, reuse) &&
        EqualReuse(retry.exit_reuse, reuse) &&
        retry_evidence.reuse
                .detected_w8_owned_boundary_identity_changes == 0 &&
        retry_evidence.reuse.w8_device_transfer_calls == 0 &&
        !retry_evidence.reuse.opaque_mfem_hypre_observed;
    if (!exact_failed_rollback || !one_same_dt_retry || !stable_storage ||
        !exact_newton_limit)
    {
      fail(prefix + "step-7 atomic rollback and one same-dt retry");
    }

    sc25::PersistentTrajectoryState injected_step7;
    if (!injected->CopyPublishedTrajectoryState(injected_step7) ||
        !EqualStateBytes(
            injected_step7,
            baseline->trajectory_evidence().steps[6].accepted_state))
    {
      fail(prefix + "step-7 deterministic publication");
    }
    if (EqualStateBytes(step6, injected_step7))
    {
      fail(prefix + "step-7 accepted state made no progress");
    }
    bool post_retry_valid = true;
    for (int step = 8; step <= sc25::persistent_required_steps; ++step)
    {
      if (!Accepted(injected->AdvancePersistentTrajectory()))
      {
        fail(prefix + "post-retry advance");
        post_retry_valid = false;
        break;
      }
    }
    sc25::PersistentTrajectoryState injected_final;
    if (!post_retry_valid ||
        !injected->CopyPublishedTrajectoryState(injected_final) ||
        !EqualStateBytes(injected_final, baseline_final))
    {
      fail(prefix + "bitwise final state");
    }
    dg_igr_hdg_sc25_cpu_test::IndependentTrajectoryEvidence
        campaign_oracle;
    if (post_retry_valid &&
        !dg_igr_hdg_sc25_cpu_test::RunIndependentTrajectoryOracle(
            *injected, campaign_oracle))
    {
      for (const std::string& failure : campaign_oracle.failures)
      {
        fail(prefix + failure);
      }
    }
  }

  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> negative =
      CreatePersistentFoundation();
  if (!negative || negative->InitializePersistentTrajectory() !=
                       sc25::TrajectoryStatus::ready)
  {
    fail("negative-restore trajectory initialization");
  }
  else
  {
    sc25::PersistentTrajectoryState before;
    negative->CopyPublishedTrajectoryState(before);
    const auto require_rejection =
        [&negative, &before, &fail](
            const sc25::PersistentTrajectoryCheckpoint& candidate,
            const sc25::TrajectoryStatus expected,
            const std::string_view label)
    {
      sc25::PersistentTrajectoryState after;
      const sc25::TrajectoryStatus actual =
          negative->RestoreTrajectoryCheckpoint(candidate);
      if (actual != expected ||
          !negative->CopyPublishedTrajectoryState(after) ||
          !EqualStateBytes(before, after))
      {
        fail(label);
      }
    };
    sc25::PersistentTrajectoryCheckpoint wrong_identity = checkpoint10;
    wrong_identity.tuple =
        dg::hdg::runtime::StableIdentity::Copy("wrong-tuple");
    require_rejection(
        wrong_identity,
        sc25::TrajectoryStatus::HdgSc25RestartValidationFailureIdentity,
        "wrong-identity restart rejection");
    sc25::PersistentTrajectoryCheckpoint wrong_descriptor = checkpoint10;
    wrong_descriptor.descriptor.rank_count = 2;
    Rehash(wrong_descriptor);
    require_rejection(
        wrong_descriptor,
        sc25::TrajectoryStatus::HdgSc25RestartValidationFailureIdentity,
        "full-descriptor restart rejection before numerical consumers");
    sc25::PersistentTrajectoryCheckpoint wrong_hash = checkpoint10;
    auto* wrong_hash_bytes = reinterpret_cast<unsigned char*>(
        wrong_hash.state.u.data());
    wrong_hash_bytes[0] ^= 0x01U;
    require_rejection(
        wrong_hash,
        sc25::TrajectoryStatus::HdgSc25RestartValidationFailureHash,
        "wrong-hash restart rejection");
    sc25::PersistentTrajectoryCheckpoint inadmissible = checkpoint10;
    inadmissible.state.u[0] = 0.0;
    Rehash(inadmissible);
    require_rejection(
        inadmissible,
        sc25::TrajectoryStatus::HdgSc25RestartValidationFailureAdmissibility,
        "inadmissible restart rejection");
    if (negative->trajectory_evidence().first_assembly_failure !=
        sc25::TrajectoryAssemblyFailureCause::state_admissibility)
    {
      fail("typed restore Assemble first cause");
    }
    sc25::PersistentTrajectoryCheckpoint auxiliary = checkpoint10;
    auxiliary.state.sigma[0] += 1.0;
    Rehash(auxiliary);
    sc25::AuxiliaryResidualMetrics independent_aux_q12{};
    sc25::AuxiliaryResidualMetrics independent_aux_q14{};
    const bool independent_aux_q12_valid =
        dg_igr_hdg_sc25_cpu_test::
            EvaluateIndependentAuxiliaryResidualOracle(
                *negative, auxiliary.state,
                sc25::coupled_fixture_quadrature_order,
                independent_aux_q12);
    const bool independent_aux_q14_valid =
        dg_igr_hdg_sc25_cpu_test::
            EvaluateIndependentAuxiliaryResidualOracle(
                *negative, auxiliary.state,
                sc25::coupled_fixture_repeat_quadrature_order,
                independent_aux_q14);
    if (!independent_aux_q12_valid || !independent_aux_q14_valid ||
        !(independent_aux_q12.normalized_maximum > 1.0e-10) ||
        !(independent_aux_q14.normalized_maximum > 1.0e-10))
    {
      fail("independent auxiliary-residual negative control");
    }
    require_rejection(
        auxiliary,
        sc25::TrajectoryStatus::HdgSc25RestartValidationFailureAuxResidual,
        "auxiliary-residual restart rejection");
  }

  if (failures == 0)
  {
    const sc25::TrajectoryReuseEvidence& reuse =
        baseline->trajectory_evidence().reuse;
    std::cout
        << "SC25_REUSE_SCOPE scope=w8_owned_boundary_census "
        << "identity_changes="
        << reuse.detected_w8_owned_boundary_identity_changes << ' '
        << "w8_device_transfer_calls=" << reuse.w8_device_transfer_calls
        << " opaque_mfem_hypre_observed="
        << (reuse.opaque_mfem_hypre_observed ? 1 : 0) << '\n';
    std::cout << "PASS: " << sc25::persistent_fixture_id << " ("
              << oracle.checked_steps << " accepted steps, "
              << oracle.checked_newton_iterations << " Newton iterations, "
              << oracle.checked_line_search_trials
              << " independently checked line-search trials)\n";
  }
  return failures;
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  mfem::Hypre::Init();
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  int failures = 0;
  if (ranks != 1)
  {
    if (rank == 0) { std::cerr << "SC25 harness requires exactly one rank\n"; }
    failures = 1;
  }
  else
  {
    const bool trajectory = argc == 2 &&
        std::string_view(argv[1]) == "--trajectory";
    if (trajectory)
    {
      failures = RunTrajectoryHarness();
    }
    else
    {
    sc25::FoundationConfig config;
    config.polynomial_degree = 1;
    config.elements_per_direction = 8;
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> foundation;
    const sc25::FoundationStatus status =
        sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
            MPI_COMM_WORLD, sc25::CpuSerialTag{}, config, foundation);
    if (status != sc25::FoundationStatus::ready || !foundation)
    {
      std::cerr << "rank 0: MFEM-native SC25 C0 creation failed\n";
      failures = 1;
    }
    else
    {
      dg_igr_hdg_sc25_cpu_test::IndependentC0Evidence evidence;
      if (!dg_igr_hdg_sc25_cpu_test::RunIndependentC0Oracle(
              *foundation, evidence))
      {
        failures = static_cast<int>(evidence.failures.size());
        for (const std::string& failure : evidence.failures)
        {
          std::cerr << "rank 0: " << failure << '\n';
        }
      }
      else
      {
        std::cout << "PASS: MFEM-native SC25 C0 foundation ("
                  << evidence.checked_elements << " elements, "
                  << evidence.checked_restriction_entries
                  << " restriction entries)\n";
      }
    }
    sc25::FoundationConfig rejected_periodic_config;
    rejected_periodic_config.polynomial_degree = 1;
    rejected_periodic_config.elements_per_direction = 2;
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
        rejected_periodic_foundation;
    const sc25::FoundationStatus rejected_periodic_status =
        sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
            MPI_COMM_WORLD, sc25::CpuSerialTag{}, rejected_periodic_config,
            rejected_periodic_foundation);
    if (rejected_periodic_status !=
            sc25::FoundationStatus::invalid_argument ||
        rejected_periodic_foundation)
    {
      std::cerr << "rank 0: N<3 periodic mesh was not rejected safely\n";
      ++failures;
    }

    sc25::FoundationConfig coupled_config;
    coupled_config.polynomial_degree = 1;
    coupled_config.elements_per_direction = 3;
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
        coupled_foundation;
    const sc25::FoundationStatus coupled_status =
        sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
            MPI_COMM_WORLD, sc25::CpuSerialTag{}, coupled_config,
            coupled_foundation);
    if (coupled_status != sc25::FoundationStatus::ready ||
        !coupled_foundation)
    {
      std::cerr << "rank 0: MFEM-native SC25 C1 creation failed\n";
      ++failures;
    }
    else
    {
      dg_igr_hdg_sc25_cpu_test::IndependentC1Evidence coupled_evidence;
      if (!dg_igr_hdg_sc25_cpu_test::RunIndependentC1Oracle(
              *coupled_foundation, coupled_evidence))
      {
        failures += std::max(
            1, static_cast<int>(coupled_evidence.failures.size()));
        for (const std::string& failure : coupled_evidence.failures)
        {
          std::cerr << "rank 0: " << failure << '\n';
        }
      }
      else
      {
        std::cout << "PASS: " << sc25::coupled_fixture_id << " ("
                  << coupled_evidence.full_dimension << " unknowns, "
                  << coupled_evidence.checked_coordinate_directions
                  << " coordinate directions, max FD discrepancy "
                  << coupled_evidence.maximum_directional_discrepancy
                  << ")\n";
      }
    }
    }
  }

  mfem::Hypre::Finalize();
  MPI_Finalize();
  return failures == 0 ? 0 : 1;
}
