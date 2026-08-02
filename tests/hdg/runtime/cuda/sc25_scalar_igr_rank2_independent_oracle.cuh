#pragma once

#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2.cuh"
#include "sc25_scalar_igr_rank2_manifest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace dg_igr_hdg_sc25_cuda_rank2_test
{
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25;
namespace cpu = ::dg::hdg::runtime::cpu::sc25;
namespace manifest = ::dg_igr_hdg_sc25_rank2_manifest;
using StateHash = ::dg::hdg::runtime::StateHash;

inline bool SameStateBytes(const cpu::PersistentRank2TrajectoryState& left,
                           const cpu::PersistentRank2TrajectoryState& right)
    noexcept
{
  return manifest::SameStateBytes(left, right);
}

inline bool Finite(const cpu::PersistentRank2TrajectoryState& state) noexcept
{
  const auto finite = [](const std::vector<double>& values)
  {
    return std::all_of(values.begin(), values.end(),
                       [](const double value) { return std::isfinite(value); });
  };
  return finite(state.u) && finite(state.sigma) && finite(state.z) &&
      finite(state.uhat) && finite(state.sigmahat);
}

inline double RelativeL2(const std::vector<double>& reference,
                         const std::vector<double>& observed) noexcept
{
  if (reference.size() != observed.size())
    return std::numeric_limits<double>::infinity();
  long double difference = 0.0L;
  long double scale = 0.0L;
  for (std::size_t i = 0; i < reference.size(); ++i)
  {
    const long double delta = static_cast<long double>(observed[i]) -
        reference[i];
    difference += delta * delta;
    scale += static_cast<long double>(reference[i]) * reference[i];
  }
  return std::sqrt(static_cast<double>(difference /
      std::max(1.0L, scale)));
}

inline double MaximumRelativeL2(
    const cpu::PersistentRank2TrajectoryState& reference,
    const cpu::PersistentRank2TrajectoryState& observed) noexcept
{
  return std::max({RelativeL2(reference.u, observed.u),
                   RelativeL2(reference.sigma, observed.sigma),
                   RelativeL2(reference.z, observed.z),
                   RelativeL2(reference.uhat, observed.uhat),
                   RelativeL2(reference.sigmahat, observed.sigmahat)});
}

// The durable receipt in the shared manifest codec identifies the already
// accepted CPU composition. This proves only parity against those bytes; it
// is not an independent mathematical oracle.
inline bool CrossBackendParityAgainstAcceptedCpuReceipt(
    const cpu::PersistentRank2TrajectoryState& accepted_cpu,
    const cpu::PersistentRank2TrajectoryState& observed_cuda,
    const double tolerance) noexcept
{
  return Finite(observed_cuda) &&
      MaximumRelativeL2(accepted_cpu, observed_cuda) <= tolerance;
}

inline bool ExactCensus(const cpu::Rank2OwnershipCensus& census,
                        const int rank) noexcept
{
  if (rank < 0 || rank > 1) return false;
  constexpr int n = 8;
  constexpr int faces = 2 * n * n;
  constexpr int dofs_per_face = scalar::trace_components * 2;
  std::array<int, faces> owner{};
  std::array<std::uint8_t, faces> local_faces{};
  std::array<std::uint8_t, faces * dofs_per_face> global_true_seen{};
  std::array<std::uint8_t, faces * dofs_per_face> local_true_seen{};
  const auto element_rank = [](const int ex, const int ey)
  {
    const int global = ey * n + ex;
    return global < 32 ? 0 : 1;
  };
  int global_seen = 0;
  int owned_faces = 0;
  for (int ey = 0; ey < n; ++ey)
    for (int ex = 0; ex < n; ++ex)
    {
      const int vertical = ey * n + ex;
      const int horizontal = n * n + ey * n + ex;
      owner[vertical] = std::min(element_rank((ex + n - 1) % n, ey),
                                 element_rank(ex, ey));
      owner[horizontal] = std::min(element_rank(ex, (ey + n - 1) % n),
                                   element_rank(ex, ey));
      for (const int face : {vertical, horizontal})
      {
        if (owner[face] == rank) ++owned_faces;
        for (int dof = 0; dof < dofs_per_face; ++dof)
        {
          const int global_dof = face * dofs_per_face + dof;
          if (global_true_seen[global_dof] != 0) return false;
          global_true_seen[global_dof] = 1;
          ++global_seen;
        }
      }
    }
  for (int ey = 0; ey < n; ++ey)
    for (int ex = 0; ex < n; ++ex)
    {
      if (element_rank(ex, ey) != rank) continue;
      const std::array<int, 4> touched{{
          ey * n + ex,
          ey * n + (ex + 1) % n,
          n * n + ey * n + ex,
          n * n + ((ey + 1) % n) * n + ex}};
      for (const int face : touched) local_faces[face] = 1;
    }
  int local_face_count = 0;
  int local_owned_face_count = 0;
  int ghost_dofs = 0;
  for (int face = 0; face < faces; ++face)
  {
    if (local_faces[face] == 0) continue;
    ++local_face_count;
    if (owner[face] == rank) ++local_owned_face_count;
    for (int dof = 0; dof < dofs_per_face; ++dof)
    {
      const int global_dof = face * dofs_per_face + dof;
      if (local_true_seen[global_dof] != 0) return false;
      local_true_seen[global_dof] = 1;
      if (owner[face] != rank) ++ghost_dofs;
    }
  }
  return global_seen == faces * dofs_per_face && owned_faces ==
          (rank == 0 ? 72 : 56) && local_face_count == 72 &&
      local_owned_face_count == owned_faces &&
      census.rank == rank && census.ranks == 2 &&
      census.local_elements == 32 &&
      census.local_interior_unknowns == 896 &&
      census.local_faces == local_face_count &&
      census.owned_unique_faces == owned_faces &&
      census.scalar_owned_true_dofs == 2 * owned_faces &&
      census.combined_owned_trace_dofs == dofs_per_face * owned_faces &&
      census.combined_ghost_trace_dofs == ghost_dofs &&
      census.global_unique_faces == faces &&
      census.global_combined_true_dofs == faces * dofs_per_face;
}

inline bool Near(const double reference, const double observed,
                 const double tolerance = 1.0e-10) noexcept
{
  return std::isfinite(reference) && std::isfinite(observed) &&
      std::abs(observed - reference) <= tolerance *
          std::max({1.0, std::abs(reference), std::abs(observed)});
}

template <std::size_t N>
inline double ArrayVectorRelativeL2(const std::array<double, N>& observed,
                                    const std::vector<double>& reference,
                                    const std::size_t offset = 0) noexcept
{
  if (offset + reference.size() > observed.size())
    return std::numeric_limits<double>::infinity();
  long double difference = 0.0L;
  long double scale = 0.0L;
  for (std::size_t item = 0; item < reference.size(); ++item)
  {
    const long double delta = observed[offset + item] - reference[item];
    difference += delta * delta;
    scale += static_cast<long double>(reference[item]) * reference[item];
  }
  return std::sqrt(static_cast<double>(difference /
      std::max(1.0L, scale)));
}

inline bool SnapshotStateParity(
    const std::array<double, scalar::persistent_interior_unknowns>& interior,
    const std::array<double, scalar::persistent_max_owned_trace_unknowns>& trace,
    const cpu::PersistentRank2TrajectoryState& reference,
    const double tolerance) noexcept
{
  return ArrayVectorRelativeL2(interior, reference.u, 0) <= tolerance &&
      ArrayVectorRelativeL2(interior, reference.sigma, 512) <= tolerance &&
      ArrayVectorRelativeL2(interior, reference.z, 640) <= tolerance &&
      ArrayVectorRelativeL2(trace, reference.uhat, 0) <= tolerance &&
      ArrayVectorRelativeL2(trace, reference.sigmahat,
                            reference.uhat.size()) <= tolerance;
}

inline bool SameCoordinates(
    const ::dg::hdg::runtime::StateCoordinates& left,
    const ::dg::hdg::runtime::StateCoordinates& right) noexcept;

inline bool OperatorSnapshotParity(
    const manifest::CpuManifestStep& reference,
    const scalar::ScalarCudaSnapshotEvidence& observed,
    const int step, const double tolerance = 1.0e-10) noexcept
{
  const std::size_t full = reference.nonlinear_residual.size();
  const std::size_t owned = reference.condensed_right_hand_side.size();
  const scalar::ScalarCudaStatus expected_status = step == 0
      ? scalar::ScalarCudaStatus::ready : scalar::ScalarCudaStatus::accepted;
  const bool metadata = reference.operator_valid && observed.valid &&
      observed.initialization == (step == 0) &&
      observed.status == expected_status &&
      observed.target_step == step && observed.retry_index == 0 &&
      observed.iteration == reference.operator_iteration &&
      observed.full_vector_size == static_cast<int>(full) &&
      observed.owned_trace_size == static_cast<int>(owned) &&
      SameCoordinates(observed.coordinates, reference.summary.coordinates) &&
      observed.trace_iterations == reference.operator_trace_iterations;
  if (!metadata ||
      !Near(reference.operator_entry_residual_norm,
            observed.entry_residual_norm, tolerance) ||
      !Near(reference.operator_trace_true_residual_norm,
            observed.trace_true_residual_norm, tolerance) ||
      !Near(reference.operator_trace_right_hand_side_norm,
            observed.trace_right_hand_side_norm, tolerance) ||
      !Near(reference.operator_maximum_local_application_residual,
            observed.maximum_local_application_residual, tolerance) ||
      ArrayVectorRelativeL2(observed.nonlinear_residual,
                            reference.nonlinear_residual) > tolerance ||
      ArrayVectorRelativeL2(observed.jacobian_action_input,
                            reference.jacobian_action_input) > tolerance ||
      ArrayVectorRelativeL2(observed.jacobian_action_result,
                            reference.jacobian_action_result) > tolerance ||
      ArrayVectorRelativeL2(observed.condensed_right_hand_side,
                            reference.condensed_right_hand_side) > tolerance ||
      ArrayVectorRelativeL2(observed.true_trace_solution,
                            reference.true_trace_solution) > tolerance ||
      !SnapshotStateParity(observed.entry_interior, observed.entry_trace,
                           reference.operator_entry, tolerance) ||
      !SnapshotStateParity(observed.recovered_interior,
                           observed.recovered_trace,
                           reference.recovered_increment, tolerance) ||
      !std::isfinite(observed.paired_local_application_residual) ||
      !std::isfinite(observed.paired_local_right_hand_side_norm) ||
      observed.paired_local_application_residual >
          1.0e-12 + 1.0e-11 *
              observed.paired_local_right_hand_side_norm)
    return false;
  if (step > 0)
  {
    for (int component = 0; component < scalar::state_components; ++component)
    {
      if (!Near(reference.operator_committed_integrals[component],
                observed.committed_integrals[component], tolerance) ||
          !Near(reference.operator_conserved_integrals[component],
                observed.conserved_integrals[component], tolerance))
        return false;
    }
  }
  return true;
}

inline bool RuntimeEvidence(const scalar::ScalarCudaEvidence& evidence,
                            const int rank) noexcept
{
  return ExactCensus(evidence.census, rank) &&
      evidence.trace_restart == 50 &&
      evidence.trace_maximum_iterations == 200 &&
      evidence.trace_relative_tolerance == 1.0e-12 &&
      evidence.trace_absolute_tolerance == 1.0e-14 &&
      evidence.ilu_type == 0 && evidence.ilu_level_of_fill == 0 &&
      evidence.ilu_local_reordering == 1 &&
      evidence.row_partial_pivot_contract &&
      evidence.nonzero_small_pivot_contract &&
      evidence.residency.mfem_cuda_device && evidence.residency.hypre_cuda &&
      evidence.residency.umpire_device_pool &&
      evidence.residency.state_device_resident &&
      evidence.residency.topology_device_resident &&
      evidence.residency.workspace_device_resident &&
      evidence.residency.trace_solve_device_resident &&
      evidence.residency.host_staged_rank2 &&
      evidence.residency.same_device_uuid &&
      !evidence.residency.host_numerical_fallback &&
      evidence.launches.assemble_launches != 0 &&
      evidence.launches.condense_launches != 0 &&
      evidence.launches.recover_launches != 0 &&
      evidence.launches.remote_system_exchange_launches != 0 &&
      evidence.launches.owned_trace_distribution_launches != 0 &&
      evidence.launches.fallback_calls == 0 &&
      evidence.launches.last_compiled_architecture == 1200 &&
      evidence.launches.all_launches_completed &&
      evidence.transfers.totals.mpi_staging_bytes != 0 &&
      evidence.transfers.totals.host_to_device_bytes != 0 &&
      evidence.transfers.totals.device_to_host_bytes != 0 &&
      evidence.transfers.totals.complete && !evidence.transfers.overflow &&
      evidence.transfers.dropped_count == 0 &&
      evidence.allocations.warm_plateau &&
      evidence.allocations.zero_post_baseline_observed_growth &&
      evidence.allocations.zero_post_baseline_owned_allocation &&
      evidence.allocations.zero_post_baseline_owned_deallocation &&
      evidence.reuse.device_topology_matches && evidence.reuse.unchanged;
}

inline bool Conservation(const scalar::ScalarCudaEvidence& evidence) noexcept
{
  for (int i = 0; i < evidence.step_evidence_count; ++i)
  {
    const auto& step = evidence.steps[static_cast<std::size_t>(i)];
    if (!step.accepted) return false;
    for (int component = 0; component < scalar::state_components; ++component)
    {
      const double before = step.committed_integrals[component];
      const double after = step.accepted_integrals[component];
      if (std::abs(after - before) /
              std::max(1.0, std::abs(before)) > 2.0e-10)
        return false;
    }
  }
  return true;
}

inline bool SameCoordinates(
    const ::dg::hdg::runtime::StateCoordinates& left,
    const ::dg::hdg::runtime::StateCoordinates& right) noexcept
{
  return left.time == right.time && left.step == right.step &&
      left.generation == right.generation && left.cursor == right.cursor &&
      left.dt == right.dt;
}

inline bool SameHashes(const scalar::StateHashes& left,
                       const scalar::StateHashes& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma &&
      left.z == right.z && left.uhat == right.uhat &&
      left.sigmahat == right.sigmahat &&
      left.interior == right.interior && left.trace == right.trace &&
      left.shard == right.shard;
}

inline StateHash HashByteRanges(
    const std::initializer_list<std::pair<const void*, std::size_t>>& ranges)
    noexcept
{
  ::dg::hdg::runtime::Sha256 hash;
  ::dg::hdg::runtime::Sha256Digest digest{};
  for (const auto& range : ranges)
  {
    if ((range.second != 0 && range.first == nullptr) ||
        !hash.Update(range.first, range.second))
      return {};
  }
  return hash.Finalize(digest) ? StateHash{digest} : StateHash{};
}

inline bool ResealCheckpoint(scalar::ScalarCudaCheckpoint& checkpoint)
    noexcept
{
  const int scalar_nodes = scalar::persistent_elements * scalar::volume_nodes;
  const int owned = checkpoint.owned_scalar_trace_nodes;
  if (checkpoint.interior_component_slabs.size() !=
          scalar::persistent_interior_unknowns || owned <= 0 ||
      checkpoint.owned_trace_component_slabs.size() !=
          static_cast<std::size_t>(scalar::trace_components * owned))
    return false;
  std::vector<double> u(512U, 0.0);
  std::vector<double> sigma(128U, 0.0);
  std::vector<double> z(256U, 0.0);
  std::vector<double> uhat(static_cast<std::size_t>(scalar::state_components) *
                               owned,
                           0.0);
  std::vector<double> sigmahat(static_cast<std::size_t>(owned), 0.0);
  for (int element = 0; element < scalar::persistent_elements; ++element)
    for (int node = 0; node < scalar::volume_nodes; ++node)
    {
      const int scalar_node = element * scalar::volume_nodes + node;
      for (int component = 0; component < scalar::state_components;
           ++component)
      {
        u[(element * scalar::state_components + component) *
              scalar::volume_nodes + node] =
            checkpoint.interior_component_slabs[
                component * scalar_nodes + scalar_node];
      }
      sigma[scalar_node] = checkpoint.interior_component_slabs[
          scalar::state_components * scalar_nodes + scalar_node];
      for (int component = 0; component < scalar::flux_components;
           ++component)
      {
        z[(element * scalar::flux_components + component) *
              scalar::volume_nodes + node] =
            checkpoint.interior_component_slabs[
                (scalar::state_components + 1 + component) * scalar_nodes +
                scalar_node];
      }
    }
  std::copy_n(checkpoint.owned_trace_component_slabs.begin(), uhat.size(),
              uhat.begin());
  std::copy_n(checkpoint.owned_trace_component_slabs.begin() + uhat.size(),
              sigmahat.size(), sigmahat.begin());
  const auto range = [](const auto& values)
  {
    return std::make_pair(static_cast<const void*>(values.data()),
                          values.size() * sizeof(*values.data()));
  };
  checkpoint.raw_field_hashes.u = HashByteRanges({range(u)});
  checkpoint.raw_field_hashes.sigma = HashByteRanges({range(sigma)});
  checkpoint.raw_field_hashes.z = HashByteRanges({range(z)});
  checkpoint.raw_field_hashes.uhat = HashByteRanges({range(uhat)});
  checkpoint.raw_field_hashes.sigmahat = HashByteRanges({range(sigmahat)});
  checkpoint.raw_field_hashes.interior =
      HashByteRanges({range(u), range(sigma), range(z)});
  checkpoint.raw_field_hashes.trace =
      HashByteRanges({range(uhat), range(sigmahat)});
  checkpoint.raw_field_hashes.shard =
      HashByteRanges({range(u), range(sigma), range(z), range(uhat),
                      range(sigmahat)});
  checkpoint.descriptor.interior_hash =
      checkpoint.raw_field_hashes.interior;
  checkpoint.descriptor.trace_hash = checkpoint.raw_field_hashes.trace;

  ::dg::hdg::runtime::Sha256 shard_hash;
  ::dg::hdg::runtime::Sha256Digest shard_digest{};
  const auto text = [&shard_hash](const std::string_view value)
  {
    const std::uint64_t size = value.size();
    return shard_hash.Update(&size, sizeof(size)) &&
        shard_hash.Update(value.data(), value.size());
  };
  const auto bytes = [&shard_hash](const auto& value)
  {
    return shard_hash.Update(&value, sizeof(value));
  };
  const auto digest = [&shard_hash](const StateHash& value)
  {
    return shard_hash.Update(value.bytes.data(), value.bytes.size());
  };
  const auto& descriptor = checkpoint.descriptor;
  const bool shard_valid = text(checkpoint.projection.view()) &&
      text(checkpoint.tuple.view()) && text(checkpoint.scenario.view()) &&
      text(checkpoint.layout.view()) && bytes(checkpoint.shard_rank) &&
      bytes(checkpoint.owned_scalar_trace_nodes) &&
      bytes(descriptor.schema_version) &&
      bytes(descriptor.successful_commit) && bytes(descriptor.restart_mode) &&
      bytes(descriptor.tuple) && bytes(descriptor.mfem_pin) &&
      digest(descriptor.mesh_hash) && digest(descriptor.partition_hash) &&
      digest(descriptor.topology_hash) && bytes(descriptor.rank_count) &&
      bytes(descriptor.polynomial_order) && bytes(descriptor.layout) &&
      bytes(descriptor.precision) && bytes(descriptor.cursor) &&
      bytes(descriptor.dt) && bytes(descriptor.coordinates.time) &&
      bytes(descriptor.coordinates.step) &&
      bytes(descriptor.coordinates.generation) &&
      bytes(descriptor.coordinates.cursor) &&
      bytes(descriptor.coordinates.dt) &&
      bytes(descriptor.coordinate_fingerprint) &&
      digest(descriptor.interior_hash) && digest(descriptor.trace_hash) &&
      digest(checkpoint.raw_field_hashes.u) &&
      digest(checkpoint.raw_field_hashes.sigma) &&
      digest(checkpoint.raw_field_hashes.z) &&
      digest(checkpoint.raw_field_hashes.uhat) &&
      digest(checkpoint.raw_field_hashes.sigmahat) &&
      digest(checkpoint.raw_field_hashes.shard) &&
      shard_hash.Update(checkpoint.interior_component_slabs.data(),
                        checkpoint.interior_component_slabs.size() *
                            sizeof(double)) &&
      shard_hash.Update(checkpoint.owned_trace_component_slabs.data(),
                        checkpoint.owned_trace_component_slabs.size() *
                            sizeof(double)) &&
      shard_hash.Finalize(shard_digest);
  if (!shard_valid) return false;
  checkpoint.shard_manifest_hash = StateHash{shard_digest};

  std::array<std::uint8_t, 64> rank_ordered{};
  if (MPI_Allgather(checkpoint.shard_manifest_hash.bytes.data(), 32,
                    MPI_UINT8_T, rank_ordered.data(), 32, MPI_UINT8_T,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    return false;
  ::dg::hdg::runtime::Sha256 combined_hash;
  ::dg::hdg::runtime::Sha256Digest combined_digest{};
  const std::string_view identity =
      "HDG-SC25-CUDA-R2-COMBINED-CHECKPOINT-v1";
  const std::uint64_t rank_count = 2;
  if (!combined_hash.Update(identity.data(), identity.size()) ||
      !combined_hash.Update(&rank_count, sizeof(rank_count)) ||
      !combined_hash.Update(rank_ordered.data(), rank_ordered.size()) ||
      !combined_hash.Finalize(combined_digest))
    return false;
  checkpoint.canonical_manifest_hash = StateHash{combined_digest};
  return true;
}

inline bool SameCensus(const cpu::Rank2OwnershipCensus& left,
                       const cpu::Rank2OwnershipCensus& right) noexcept
{
  return left.rank == right.rank && left.ranks == right.ranks &&
      left.local_elements == right.local_elements &&
      left.local_interior_unknowns == right.local_interior_unknowns &&
      left.local_faces == right.local_faces &&
      left.owned_unique_faces == right.owned_unique_faces &&
      left.scalar_owned_true_dofs == right.scalar_owned_true_dofs &&
      left.combined_owned_trace_dofs == right.combined_owned_trace_dofs &&
      left.combined_ghost_trace_dofs == right.combined_ghost_trace_dofs &&
      left.global_unique_faces == right.global_unique_faces &&
      left.global_combined_true_dofs == right.global_combined_true_dofs;
}

inline bool SameNewton(const scalar::ScalarCudaNewtonEvidence& left,
                       const scalar::ScalarCudaNewtonEvidence& right) noexcept
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

inline bool SameAttempt(const scalar::ScalarCudaAttemptEvidence& left,
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
      !SameHashes(left.entry_hashes, right.entry_hashes) ||
      !SameHashes(left.candidate_hashes, right.candidate_hashes))
    return false;
  for (std::size_t index = 0; index < left.newton.size(); ++index)
    if (!SameNewton(left.newton[index], right.newton[index])) return false;
  return true;
}

inline bool SameStep(const scalar::ScalarCudaStepEvidence& left,
                     const scalar::ScalarCudaStepEvidence& right) noexcept
{
  return left.accepted == right.accepted && left.step == right.step &&
      left.time == right.time && left.generation == right.generation &&
      left.attempt_index == right.attempt_index &&
      left.committed_integrals == right.committed_integrals &&
      left.accepted_integrals == right.accepted_integrals &&
      SameHashes(left.accepted_hashes, right.accepted_hashes);
}

inline bool SameSnapshot(const scalar::ScalarCudaSnapshotEvidence& left,
                         const scalar::ScalarCudaSnapshotEvidence& right)
    noexcept
{
  return left.valid == right.valid &&
      left.initialization == right.initialization &&
      left.status == right.status &&
      SameCoordinates(left.coordinates, right.coordinates) &&
      left.target_step == right.target_step &&
      left.retry_index == right.retry_index &&
      left.iteration == right.iteration &&
      left.attempted_newton_iterations ==
          right.attempted_newton_iterations &&
      left.newton_iterations == right.newton_iterations &&
      left.full_vector_size == right.full_vector_size &&
      left.owned_trace_size == right.owned_trace_size &&
      left.trace_iterations == right.trace_iterations &&
      left.entry_residual_norm == right.entry_residual_norm &&
      left.trace_true_residual_norm == right.trace_true_residual_norm &&
      left.trace_right_hand_side_norm ==
          right.trace_right_hand_side_norm &&
      left.maximum_local_application_residual ==
          right.maximum_local_application_residual &&
      left.paired_local_application_residual ==
          right.paired_local_application_residual &&
      left.paired_local_right_hand_side_norm ==
          right.paired_local_right_hand_side_norm &&
      left.nonlinear_residual == right.nonlinear_residual &&
      left.jacobian_action_input == right.jacobian_action_input &&
      left.jacobian_action_result == right.jacobian_action_result &&
      left.condensed_right_hand_side == right.condensed_right_hand_side &&
      left.true_trace_solution == right.true_trace_solution &&
      left.entry_interior == right.entry_interior &&
      left.entry_trace == right.entry_trace &&
      left.recovered_interior == right.recovered_interior &&
      left.recovered_trace == right.recovered_trace &&
      left.committed_integrals == right.committed_integrals &&
      left.conserved_integrals == right.conserved_integrals;
}

inline bool SameLogicalEvidence(const scalar::ScalarCudaEvidence& left,
                                const scalar::ScalarCudaEvidence& right)
    noexcept
{
  if (left.status != right.status ||
      left.first_failure != right.first_failure ||
      left.accepted_steps != right.accepted_steps ||
      left.attempted_steps != right.attempted_steps ||
      left.retry_attempts != right.retry_attempts ||
      left.attempt_count != right.attempt_count ||
      left.step_evidence_count != right.step_evidence_count ||
      left.snapshot_count != right.snapshot_count ||
      left.trace_restart != right.trace_restart ||
      left.trace_maximum_iterations != right.trace_maximum_iterations ||
      left.trace_relative_tolerance != right.trace_relative_tolerance ||
      left.trace_absolute_tolerance != right.trace_absolute_tolerance ||
      left.ilu_type != right.ilu_type ||
      left.ilu_level_of_fill != right.ilu_level_of_fill ||
      left.ilu_local_reordering != right.ilu_local_reordering ||
      left.row_partial_pivot_contract != right.row_partial_pivot_contract ||
      left.nonzero_small_pivot_contract !=
          right.nonzero_small_pivot_contract ||
      !SameCensus(left.census, right.census) ||
      !SameCoordinates(left.coordinates, right.coordinates))
    return false;
  for (std::size_t index = 0; index < left.attempts.size(); ++index)
    if (!SameAttempt(left.attempts[index], right.attempts[index])) return false;
  for (std::size_t index = 0; index < left.steps.size(); ++index)
    if (!SameStep(left.steps[index], right.steps[index])) return false;
  for (std::size_t index = 0; index < left.snapshots.size(); ++index)
    if (!SameSnapshot(left.snapshots[index], right.snapshots[index]))
      return false;
  return true;
}

inline bool AcceptedAttemptResidualBounds(
    const scalar::ScalarCudaAttemptEvidence& attempt) noexcept
{
  const bool accepted = attempt.status == scalar::ScalarCudaStatus::accepted ||
      attempt.status == scalar::ScalarCudaStatus::accepted_after_one_retry;
  if (!accepted) return false;
  if (!(attempt.minimum_density >= 1.0e-8) ||
      !(attempt.minimum_pressure >= 1.0e-8) ||
      !(attempt.minimum_speed_margin > 0.0) ||
      attempt.maximum_conservation_error > 2.0e-10 ||
      !std::isfinite(attempt.final_residual_norm) ||
      !std::isfinite(attempt.nonlinear_residual_bound) ||
      attempt.final_residual_norm > attempt.nonlinear_residual_bound)
    return false;
  for (int index = 0; index < attempt.newton_iterations; ++index)
  {
    const auto& newton = attempt.newton[static_cast<std::size_t>(index)];
    const double trace_bound = scalar::absolute_trace_tolerance +
        scalar::relative_trace_tolerance *
            newton.trace_right_hand_side_norm;
    if (newton.iteration != index || newton.trace_iterations < 0 ||
        newton.trace_iterations > scalar::trace_maximum_iterations ||
        !std::isfinite(newton.trace_true_residual_norm) ||
        !std::isfinite(newton.trace_right_hand_side_norm) ||
        newton.trace_true_residual_norm > trace_bound ||
        !std::isfinite(newton.maximum_local_application_residual) ||
        newton.maximum_local_application_residual > 1.0e-10)
      return false;
  }
  return true;
}

inline bool AcceptedTrajectoryResidualBounds(
    const scalar::ScalarCudaEvidence& evidence) noexcept
{
  for (int index = 0; index < evidence.step_evidence_count; ++index)
  {
    const int attempt_index =
        evidence.steps[static_cast<std::size_t>(index)].attempt_index;
    if (attempt_index < 0 || attempt_index >= evidence.attempt_count ||
        !AcceptedAttemptResidualBounds(
            evidence.attempts[static_cast<std::size_t>(attempt_index)]))
      return false;
  }
  return true;
}

} // namespace dg_igr_hdg_sc25_cuda_rank2_test
