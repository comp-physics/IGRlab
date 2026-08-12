#include "sc25_scalar_igr_independent_oracle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

extern "C" void dg_igr_sc25_rank2_test_inject_boundary_drift(
    int enabled) noexcept;
extern "C" void dg_igr_sc25_rank2_test_inject_final_install_mismatch(
    int enabled) noexcept;

namespace dg::hdg::runtime::cpu::sc25::w8c_internal
{
[[gnu::visibility("hidden")]] bool PrimePersistentTrajectoryTransaction(
    PersistentSc25ScalarIgrCpuFoundation& owner,
    PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction& token) noexcept;
}

namespace
{
  const auto equal_double = [](const double a, const double b) noexcept
  { return dg::hdg::runtime::detail::DoubleBits(a) ==
        dg::hdg::runtime::detail::DoubleBits(b); };

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

std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
CreateRank2Foundation()
{
  sc25::FoundationConfig config;
  config.polynomial_degree = 1;
  config.elements_per_direction = sc25::persistent_elements_per_direction;
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> foundation;
  if (sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, sc25::CpuMpiRank2Tag{}, config, foundation) !=
          sc25::FoundationStatus::ready)
  {
    return {};
  }
  return foundation;
}

std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>
CreateRank1Peer()
{
  sc25::FoundationConfig config;
  config.polynomial_degree = 1;
  config.elements_per_direction = sc25::persistent_elements_per_direction;
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> foundation;
  if (sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_SELF, sc25::CpuSerialTag{}, config, foundation) !=
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

template <typename Range>
bool EqualFp64Bytes(const Range& left, const Range& right) noexcept
{
  return left.size() == right.size() &&
      (left.empty() || std::memcmp(left.data(), right.data(),
                                   left.size() * sizeof(double)) == 0);
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

bool EqualRank2StateBytes(const sc25::PersistentRank2TrajectoryState& left,
                          const sc25::PersistentRank2TrajectoryState& right)
    noexcept
{
  const auto equal = [](const std::vector<double>& a,
                        const std::vector<double>& b)
  {
    return a.size() == b.size() &&
        (a.empty() ||
         std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
  };
  return equal(left.u, right.u) && equal(left.sigma, right.sigma) &&
      equal(left.z, right.z) && equal(left.uhat, right.uhat) &&
      equal(left.sigmahat, right.sigmahat);
}

bool EqualRank2Hashes(const sc25::Rank2TrajectoryStateHashes& left,
                      const sc25::Rank2TrajectoryStateHashes& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma &&
      left.z == right.z && left.uhat == right.uhat &&
      left.sigmahat == right.sigmahat &&
      left.interior == right.interior && left.trace == right.trace &&
      left.shard == right.shard;
}

bool EqualTrajectoryEvidenceBytes(
    const sc25::PersistentTrajectoryEvidence& left,
    const sc25::PersistentTrajectoryEvidence& right) noexcept
{
  const auto equal_hashes = [](const sc25::TrajectoryStateHashes& a,
                               const sc25::TrajectoryStateHashes& b) noexcept
  {
    return a.u == b.u && a.sigma == b.sigma && a.z == b.z &&
        a.uhat == b.uhat && a.sigmahat == b.sigmahat &&
        a.interior == b.interior && a.trace == b.trace;
  };
  const auto equal_reuse = [](const sc25::TrajectoryReuseSnapshot& a,
                              const sc25::TrajectoryReuseSnapshot& b) noexcept
  {
    return a.foundation_handle == b.foundation_handle &&
        a.topology_handle == b.topology_handle &&
        a.carrier_map_handle == b.carrier_map_handle &&
        a.contract_handle == b.contract_handle &&
        a.assembly_state_handle == b.assembly_state_handle &&
        a.assembly_handle == b.assembly_handle &&
        a.trial_assembly_handle == b.trial_assembly_handle &&
        a.factor_handle == b.factor_handle && a.workspace_handle == b.workspace_handle &&
        a.quadrature_workspace_handle == b.quadrature_workspace_handle &&
        a.local_pattern_handle == b.local_pattern_handle &&
        a.true_matrix_handle == b.true_matrix_handle && a.true_rhs_handle == b.true_rhs_handle &&
        a.true_solution_handle == b.true_solution_handle && a.ilu_handle == b.ilu_handle &&
        a.gmres_handle == b.gmres_handle && a.true_residual_handle == b.true_residual_handle &&
        a.element_to_true_csr_handle == b.element_to_true_csr_handle &&
        a.owned_boundary_identity_hash == b.owned_boundary_identity_hash &&
        a.topology_generation == b.topology_generation && a.storage_generation == b.storage_generation &&
        a.factor_generation == b.factor_generation && a.workspace_generation == b.workspace_generation &&
        a.local_pattern_generation == b.local_pattern_generation && a.solver_generation == b.solver_generation &&
        a.w8_owned_boundary_object_count == b.w8_owned_boundary_object_count &&
        a.w8_owned_boundary_buffer_count == b.w8_owned_boundary_buffer_count &&
        a.opaque_mfem_hypre_observed == b.opaque_mfem_hypre_observed;
  };
  const auto equal_initialization = [&](const sc25::AuxiliaryInitializationEvidence& a,
                                        const sc25::AuxiliaryInitializationEvidence& b) noexcept
  {
    const auto bits = equal_double;
    const auto metrics = [&](const sc25::AuxiliaryResidualMetrics& x,
                             const sc25::AuxiliaryResidualMetrics& y) noexcept
    { return bits(x.r_z_norm,y.r_z_norm) && bits(x.r_z_scale,y.r_z_scale) &&
        bits(x.r_sigma_norm,y.r_sigma_norm) && bits(x.r_sigma_scale,y.r_sigma_scale) &&
        bits(x.g_sigma_norm,y.g_sigma_norm) && bits(x.g_sigma_scale,y.g_sigma_scale) &&
        bits(x.normalized_maximum,y.normalized_maximum); };
    return a.quadrature_order == b.quadrature_order && a.trace_restart_dimension == b.trace_restart_dimension &&
        a.trace_maximum_iterations == b.trace_maximum_iterations &&
        bits(a.trace_relative_tolerance,b.trace_relative_tolerance) && bits(a.trace_absolute_tolerance,b.trace_absolute_tolerance) &&
        a.ilu_type == b.ilu_type && a.ilu_level_of_fill == b.ilu_level_of_fill &&
        a.ilu_local_reordering == b.ilu_local_reordering && a.trace_iterations == b.trace_iterations &&
        bits(a.trace_true_residual_norm,b.trace_true_residual_norm) &&
        bits(a.trace_right_hand_side_norm,b.trace_right_hand_side_norm) && metrics(a.residual,b.residual);
  };
  const auto equal_attempt = [&](const sc25::TrajectoryAttemptEvidence& a,
                                 const sc25::TrajectoryAttemptEvidence& b) noexcept
  {
    return a.status == b.status && a.injection == b.injection &&
        a.target_step == b.target_step && a.retry_index == b.retry_index &&
        equal_double(a.dt, b.dt) &&
        a.attempted_newton_iterations == b.attempted_newton_iterations &&
        a.newton_iterations == b.newton_iterations &&
        a.final_line_search_trials == b.final_line_search_trials &&
        equal_double(a.entry_residual_norm, b.entry_residual_norm) &&
        equal_double(a.final_residual_norm, b.final_residual_norm) &&
        equal_double(a.nonlinear_residual_bound, b.nonlinear_residual_bound) &&
        equal_double(a.maximum_conservation_error, b.maximum_conservation_error) &&
        equal_double(a.minimum_density, b.minimum_density) &&
        equal_double(a.minimum_pressure, b.minimum_pressure) &&
        equal_double(a.minimum_speed_margin, b.minimum_speed_margin) &&
        dg::hdg::runtime::detail::SameCoordinates(
            a.committed_coordinates, b.committed_coordinates) &&
        dg::hdg::runtime::detail::SameCoordinates(
            a.published_coordinates, b.published_coordinates) &&
        equal_hashes(a.entry_hashes, b.entry_hashes) &&
        equal_hashes(a.candidate_hashes, b.candidate_hashes) &&
        equal_hashes(a.published_hashes, b.published_hashes) &&
        equal_reuse(a.entry_reuse, b.entry_reuse) &&
        equal_reuse(a.exit_reuse, b.exit_reuse);
  };
  const auto equal_accepted = [&](const sc25::TrajectoryAcceptedSummary& a,
                                  const sc25::TrajectoryAcceptedSummary& b) noexcept
  {
    return a.accepted == b.accepted && a.step == b.step &&
        equal_double(a.time, b.time) && equal_double(a.dt, b.dt) &&
        a.generation == b.generation && a.attempt_index == b.attempt_index &&
        EqualFp64Bytes(a.committed_integrals, b.committed_integrals) &&
        EqualFp64Bytes(a.accepted_integrals, b.accepted_integrals) &&
        equal_hashes(a.accepted_hashes, b.accepted_hashes) &&
        equal_reuse(a.reuse, b.reuse);
  };
  if (left.status != right.status || left.first_failure != right.first_failure ||
      left.first_assembly_failure != right.first_assembly_failure ||
      left.contract_status != right.contract_status ||
      left.contract_last_failure != right.contract_last_failure ||
      left.contract_first_failure != right.contract_first_failure ||
      left.accepted_steps != right.accepted_steps ||
      left.attempted_steps != right.attempted_steps ||
      left.retry_attempts != right.retry_attempts ||
      left.attempt_count != right.attempt_count ||
      left.checkpoint_step_offset != right.checkpoint_step_offset ||
      !dg::hdg::runtime::detail::SameCoordinates(
          left.continuation_start_coordinates, right.continuation_start_coordinates) ||
      !equal_hashes(left.continuation_start_hashes,
                    right.continuation_start_hashes) ||
      left.restored_loaded_state_match != right.restored_loaded_state_match ||
      left.restored_loaded_coordinates_match != right.restored_loaded_coordinates_match ||
      left.restored_loaded_hashes_match != right.restored_loaded_hashes_match ||
      left.trace_restart_dimension != right.trace_restart_dimension ||
      left.trace_maximum_iterations != right.trace_maximum_iterations ||
      !equal_double(left.trace_relative_tolerance, right.trace_relative_tolerance) ||
      !equal_double(left.trace_absolute_tolerance, right.trace_absolute_tolerance) ||
      left.ilu_type != right.ilu_type || left.ilu_level_of_fill != right.ilu_level_of_fill ||
      left.ilu_local_reordering != right.ilu_local_reordering ||
      !dg::hdg::runtime::detail::SameCoordinates(left.coordinates, right.coordinates) ||
      !equal_initialization(left.initialization, right.initialization) ||
      !equal_reuse(left.reuse.baseline, right.reuse.baseline) ||
      !equal_reuse(left.reuse.current, right.reuse.current) ||
      left.reuse.true_matrix_nonzeros != right.reuse.true_matrix_nonzeros ||
      left.reuse.detected_w8_owned_boundary_identity_changes !=
          right.reuse.detected_w8_owned_boundary_identity_changes ||
      left.reuse.w8_device_transfer_calls != right.reuse.w8_device_transfer_calls ||
      left.reuse.opaque_mfem_hypre_observed != right.reuse.opaque_mfem_hypre_observed ||
      !equal_attempt(left.first_failed_attempt, right.first_failed_attempt) ||
      !equal_attempt(left.last_attempt, right.last_attempt) ||
      !equal_accepted(left.last_accepted, right.last_accepted))
    return false;
  return true;
}

bool EqualRank2EvidenceBoundary(
    const sc25::PersistentRank2TrajectoryEvidence& left,
    const sc25::PersistentRank2TrajectoryEvidence& right) noexcept
{
  return left.status == right.status &&
      left.first_failure == right.first_failure &&
      std::memcmp(&left.census, &right.census, sizeof(left.census)) == 0 &&
      dg::hdg::runtime::detail::SameCoordinates(
          left.coordinates, right.coordinates) &&
      left.accepted_steps == right.accepted_steps &&
      left.attempted_steps == right.attempted_steps &&
      left.retry_attempts == right.retry_attempts &&
      left.attempt_count == right.attempt_count &&
      left.step_evidence_count == right.step_evidence_count &&
      left.topology_generation == right.topology_generation &&
      left.storage_generation == right.storage_generation &&
      left.solver_generation == right.solver_generation &&
      left.detected_w8_owned_boundary_identity_changes ==
          right.detected_w8_owned_boundary_identity_changes &&
      left.w8_device_transfer_calls == right.w8_device_transfer_calls &&
      left.w8_owned_boundary_identity_hash ==
          right.w8_owned_boundary_identity_hash &&
      left.w8_owned_boundary_object_count ==
          right.w8_owned_boundary_object_count &&
      left.w8_owned_boundary_buffer_count ==
          right.w8_owned_boundary_buffer_count &&
      left.opaque_mfem_hypre_observed ==
          right.opaque_mfem_hypre_observed;
}

int GeometricPeerElement(
    sc25::PersistentSc25ScalarIgrCpuFoundation& local,
    const int local_element,
    sc25::PersistentSc25ScalarIgrCpuFoundation& peer)
{
  mfem::ParMesh* local_mesh = const_cast<mfem::ParMesh*>(local.mesh());
  mfem::ParMesh* peer_mesh = const_cast<mfem::ParMesh*>(peer.mesh());
  if (!local_mesh || !peer_mesh || local_element < 0 ||
      local_element >= local_mesh->GetNE())
  {
    return -1;
  }
  const mfem::IntegrationPoint& center =
      mfem::Geometries.GetCenter(mfem::Geometry::SQUARE);
  mfem::Vector local_point(sc25::dimension), peer_point(sc25::dimension);
  mfem::ElementTransformation* local_transformation =
      local_mesh->GetElementTransformation(local_element);
  if (!local_transformation) { return -1; }
  local_transformation->Transform(center, local_point);
  for (int element = 0; element < peer_mesh->GetNE(); ++element)
  {
    mfem::ElementTransformation* peer_transformation =
        peer_mesh->GetElementTransformation(element);
    if (!peer_transformation) { continue; }
    peer_transformation->Transform(center, peer_point);
    if (std::abs(local_point[0] - peer_point[0]) <= 1.0e-13 &&
        std::abs(local_point[1] - peer_point[1]) <= 1.0e-13)
    {
      return element;
    }
  }
  return -1;
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
  positive_zero_state.u.assign(1, 0.0);
  negative_zero_state.u.assign(1, 0.0);
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

  {
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> owner =
        CreatePersistentFoundation();
    if (!owner || owner->InitializePersistentTrajectory() !=
                      sc25::TrajectoryStatus::ready)
    {
      fail("rank1 transaction foundation initialization");
    }
    else
    {
      dg::hdg::runtime::PersistentResourceDescriptor initial_resources;
      sc25::PersistentTrajectoryState step0_state;
      const auto step0_evidence =
          std::make_unique<sc25::PersistentTrajectoryEvidence>(
              owner->trajectory_evidence());
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction invalid;
      if (!owner->CopyPersistentResourceDescriptor(initial_resources) ||
          initial_resources.reuse.persistent_bytes == 0 ||
          !owner->CopyPublishedTrajectoryState(step0_state) ||
          owner->InstallPersistentTrajectoryTransaction(invalid))
      {
        fail("rank1 invalid transaction token rejection");
      }
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction prepared;
      if (!sc25::w8c_internal::PrimePersistentTrajectoryTransaction(
              *owner, prepared))
      {
        fail("rank1 transaction prime");
      }
      const sc25::TrajectoryStatus prepared_status =
          owner->PreparePersistentTrajectoryTransaction(prepared);
      sc25::PersistentTrajectoryState after_prepare;
      if (!Accepted(prepared_status) || !prepared.valid() ||
          prepared.provisionally_installed() ||
          prepared.future_checkpoint().descriptor.coordinates.step != 1 ||
          EqualStateBytes(prepared.candidate_state(), step0_state) ||
          !owner->CopyPublishedTrajectoryState(after_prepare) ||
          !EqualStateBytes(after_prepare, step0_state) ||
          !EqualTrajectoryEvidenceBytes(owner->trajectory_evidence(),
                                        *step0_evidence))
      {
        fail("rank1 prepare preserves exact step0 publication/evidence");
      }
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction live(
          std::move(prepared));
      if (prepared.valid() ||
          owner->InstallPersistentTrajectoryTransaction(prepared) ||
          !owner->InstallPersistentTrajectoryTransaction(live) ||
          !live.provisionally_installed())
      {
        fail("rank1 move-only stale token and provisional install");
      }
      sc25::PersistentTrajectoryState installed;
      if (!owner->CopyPublishedTrajectoryState(installed) ||
          !EqualStateBytes(installed, live.candidate_state()) ||
          owner->trajectory_evidence().coordinates.step != 1)
      {
        fail("rank1 provisional candidate publication");
      }
      if (!owner->RevertPersistentTrajectoryTransaction(std::move(live)) ||
          live.valid() ||
          owner->RevertPersistentTrajectoryTransaction(std::move(live)))
      {
        fail("rank1 exact revert and stale reuse rejection");
      }
      sc25::PersistentTrajectoryState after_revert;
      dg::hdg::runtime::PersistentResourceDescriptor reverted_resources;
      if (!owner->CopyPersistentResourceDescriptor(reverted_resources) ||
          reverted_resources.reuse.persistent_bytes !=
              initial_resources.reuse.persistent_bytes ||
          !owner->CopyPublishedTrajectoryState(after_revert) ||
          !EqualStateBytes(after_revert, step0_state) ||
          !EqualTrajectoryEvidenceBytes(owner->trajectory_evidence(),
                                        *step0_evidence))
      {
        fail("rank1 revert restores exact step0 state/evidence bytes");
      }

      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction
          finalized_transaction;
      if (!sc25::w8c_internal::PrimePersistentTrajectoryTransaction(
              *owner, finalized_transaction) ||
          !Accepted(owner->PreparePersistentTrajectoryTransaction(
              finalized_transaction)) ||
          !owner->InstallPersistentTrajectoryTransaction(
              finalized_transaction))
      {
        fail("rank1 second transaction prepare/install");
      }
      const dg::hdg::runtime::StateHash future_manifest =
          finalized_transaction.future_checkpoint().canonical_manifest_hash;
      if (!owner->FinalizePersistentTrajectoryTransaction(
              std::move(finalized_transaction)) ||
          finalized_transaction.valid() ||
          owner->FinalizePersistentTrajectoryTransaction(
              std::move(finalized_transaction)))
      {
        fail("rank1 finalize and stale reuse rejection");
      }
      sc25::PersistentTrajectoryCheckpoint finalized;
      if (!owner->CaptureTrajectoryCheckpoint(finalized) ||
          finalized.canonical_manifest_hash != future_manifest ||
          owner->trajectory_evidence().coordinates.step != 1)
      {
        fail("rank1 finalized checkpoint equals frozen future checkpoint");
      }
      sc25::PersistentTrajectoryState before_pending_restore;
      dg::hdg::runtime::PersistentResourceDescriptor before_pending_resources;
      dg::hdg::runtime::PersistentResourceDescriptor after_pending_resources;
      const auto before_pending_evidence =
          std::make_unique<sc25::PersistentTrajectoryEvidence>(
              owner->trajectory_evidence());
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction pending;
      if (!sc25::w8c_internal::PrimePersistentTrajectoryTransaction(
              *owner, pending) ||
          !owner->CopyPublishedTrajectoryState(before_pending_restore) ||
          !owner->CopyPersistentResourceDescriptor(before_pending_resources) ||
          !Accepted(owner->PreparePersistentTrajectoryTransaction(pending)) ||
          owner->RestoreTrajectoryCheckpoint(finalized) !=
              sc25::TrajectoryStatus::invalid_configuration ||
          !owner->CopyPublishedTrajectoryState(after_revert) ||
          !owner->CopyPersistentResourceDescriptor(after_pending_resources) ||
          !EqualStateBytes(before_pending_restore, after_revert) ||
          before_pending_resources.reuse.persistent_bytes !=
              after_pending_resources.reuse.persistent_bytes ||
          !EqualTrajectoryEvidenceBytes(owner->trajectory_evidence(),
                                        *before_pending_evidence) ||
          !owner->InstallPersistentTrajectoryTransaction(pending) ||
          !owner->RevertPersistentTrajectoryTransaction(std::move(pending)))
      {
        fail("rank1 active transaction restore rejection is atomic");
      }
      sc25::PersistentTrajectoryCheckpoint imported;
      dg::hdg::runtime::PersistentResourceDescriptor resources;
      if (!owner->CopyPersistentResourceDescriptor(resources) ||
          !owner->ImportTrajectoryCheckpoint(
              finalized.descriptor.coordinates, finalized.state, imported) ||
          imported.canonical_manifest_hash !=
              finalized.canonical_manifest_hash ||
          imported.raw_field_hashes.interior !=
              finalized.raw_field_hashes.interior ||
          imported.raw_field_hashes.trace !=
              finalized.raw_field_hashes.trace)
      {
        fail("rank1 exact app-codec checkpoint import");
      }
      dg::hdg::runtime::StateCoordinates corrupt_coordinates =
          finalized.descriptor.coordinates;
      ++corrupt_coordinates.cursor;
      if (owner->ImportTrajectoryCheckpoint(
              corrupt_coordinates, finalized.state, imported))
      {
        fail("rank1 mismatched-coordinate import rejection");
      }
    }
  }

  const auto independently_check_accepted = [&fail](
      sc25::PersistentSc25ScalarIgrCpuFoundation& foundation,
      const std::string_view label)
  {
    sc25::PersistentTrajectoryState state;
    sc25::AuxiliaryResidualMetrics q12{};
    sc25::AuxiliaryResidualMetrics q14{};
    if (!foundation.CopyPublishedTrajectoryState(state) ||
        !dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
            foundation, state, sc25::coupled_fixture_quadrature_order, q12) ||
        !dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
            foundation, state,
            sc25::coupled_fixture_repeat_quadrature_order, q14) ||
        q12.normalized_maximum > 1.0e-10 ||
        q14.normalized_maximum > 1.0e-10)
      fail(std::string(label) + " independent accepted-state residual");
  };

  sc25::PersistentTrajectoryCheckpoint checkpoint10;
  sc25::PersistentTrajectoryState baseline_step7;
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
    independently_check_accepted(*baseline, "baseline step");
    if (step == 7 && !baseline->CopyPublishedTrajectoryState(baseline_step7))
    {
      fail("step-7 baseline state capture");
      return failures;
    }
    if (step == 10 && !baseline->CaptureTrajectoryCheckpoint(checkpoint10))
    {
      fail("step-10 checkpoint capture");
      return failures;
    }
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
        restored_evidence.attempt_count != 0 ||
        !EqualCoordinates(restored_evidence.continuation_start_coordinates,
                          checkpoint10.descriptor.coordinates) ||
        !EqualHashes(restored_evidence.continuation_start_hashes,
                     checkpoint10.raw_field_hashes) ||
        !restored_evidence.restored_loaded_state_match ||
        !restored_evidence.restored_loaded_coordinates_match ||
        !restored_evidence.restored_loaded_hashes_match)
    {
      fail("step-10 immediate loaded checkpoint equality/evidence reset");
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
      independently_check_accepted(*replay, "restored continuation");
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
        continuation.attempt_count != 10 ||
        continuation.last_accepted.step != 20 ||
        continuation.last_accepted.attempt_index != 9)
    {
      fail("step-10 honest continuation record window");
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
        retry_evidence.first_failed_attempt;
    const sc25::TrajectoryAttemptEvidence& retry =
        retry_evidence.last_attempt;
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
        !EqualStateBytes(injected_step7, baseline_step7))
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
    if (post_retry_valid) independently_check_accepted(*injected, prefix);
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
    std::cout << "PASS: " << sc25::persistent_fixture_id
              << " (step-owned accepted-state residual checks)\n";
  }
  return failures;
}

int RunRank2TrajectoryHarness()
{
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int local_failures = 0;
  const auto fail = [rank, &local_failures](const std::string_view message)
  {
    std::cerr << "rank " << rank << ": " << message << '\n';
    ++local_failures;
  };
  const auto collectively_require = [&fail](const bool local,
                                             const std::string_view message)
  {
    const int value = local ? 1 : 0;
    int global = 0;
    if (MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS ||
        global == 0)
    {
      fail(message);
      return false;
    }
    return true;
  };

  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> baseline =
      CreateRank2Foundation();
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> peer =
      CreateRank1Peer();
  if (!collectively_require(baseline && peer,
                            "rank2/rank1-peer foundation creation"))
  {
    return 1;
  }
  sc25::PersistentRank2TrajectoryState unpublished;
  unpublished.u = {1.0, 2.0};
  unpublished.sigma = {3.0};
  unpublished.z = {4.0, 5.0};
  unpublished.uhat = {6.0};
  unpublished.sigmahat = {7.0};
  const sc25::PersistentRank2TrajectoryState unpublished_before =
      unpublished;
  if (baseline->CopyPublishedRank2TrajectoryState(unpublished) ||
      !EqualRank2StateBytes(unpublished, unpublished_before))
  {
    fail("rank2 unpublished local copy preserves output atomically");
  }
  if (!collectively_require(
          baseline->InitializePersistentRank2Trajectory() ==
                  sc25::TrajectoryStatus::ready &&
              peer->InitializePersistentTrajectory() ==
                  sc25::TrajectoryStatus::ready,
          "rank2/rank1-peer initialization"))
  {
    return 1;
  }
  sc25::PersistentRank2TrajectoryState initialized;
  sc25::PersistentRank2TrajectoryState initialized_again;
  if (!baseline->CopyPublishedRank2TrajectoryState(initialized) ||
      baseline->InitializePersistentRank2Trajectory() !=
          sc25::TrajectoryStatus::ready ||
      !baseline->CopyPublishedRank2TrajectoryState(initialized_again) ||
      !EqualRank2StateBytes(initialized, initialized_again))
  {
    fail("rank2 initialization reentry byte stability");
  }

  {
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> owner =
        CreateRank2Foundation();
    if (!collectively_require(
            owner && owner->InitializePersistentRank2Trajectory() ==
                         sc25::TrajectoryStatus::ready,
            "rank2 transaction foundation initialization"))
      return 1;
    sc25::PersistentRank2TrajectoryState step0_state;
    const auto step0_evidence =
        std::make_unique<sc25::PersistentRank2TrajectoryEvidence>(
            owner->rank2_trajectory_evidence());
    sc25::PersistentSc25ScalarIgrCpuFoundation::Rank2Transaction invalid;
    if (!collectively_require(
            owner->CopyPublishedRank2TrajectoryState(step0_state) &&
                !owner->InstallPersistentRank2TrajectoryTransaction(invalid),
            "rank2 invalid transaction token rejection"))
      return 1;
    sc25::PersistentSc25ScalarIgrCpuFoundation::Rank2Transaction prepared;
    const sc25::TrajectoryStatus prepared_status =
        owner->PreparePersistentRank2TrajectoryTransaction(prepared);
    sc25::PersistentRank2TrajectoryState after_prepare;
    if (!collectively_require(
            Accepted(prepared_status) && prepared.valid() &&
                !prepared.provisionally_installed() &&
                prepared.future_checkpoint().descriptor.coordinates.step ==
                    1 &&
                !EqualRank2StateBytes(prepared.candidate_state(),
                                      step0_state) &&
                owner->CopyPublishedRank2TrajectoryState(after_prepare) &&
                EqualRank2StateBytes(after_prepare, step0_state) &&
                EqualRank2EvidenceBoundary(
                    owner->rank2_trajectory_evidence(), *step0_evidence),
            "rank2 prepare preserves exact step0 publication/evidence"))
      return 1;
    sc25::PersistentSc25ScalarIgrCpuFoundation::Rank2Transaction live(
        std::move(prepared));
    if (!collectively_require(
            !prepared.valid() &&
                !owner->InstallPersistentRank2TrajectoryTransaction(
                    prepared) &&
                owner->InstallPersistentRank2TrajectoryTransaction(live) &&
                live.provisionally_installed(),
            "rank2 move-only stale token and provisional install"))
      return 1;
    sc25::PersistentRank2TrajectoryState installed;
    if (!collectively_require(
            owner->CopyPublishedRank2TrajectoryState(installed) &&
                EqualRank2StateBytes(installed, live.candidate_state()) &&
                owner->rank2_trajectory_evidence().coordinates.step == 1,
            "rank2 provisional candidate publication"))
      return 1;
    if (!collectively_require(
            owner->RevertPersistentRank2TrajectoryTransaction(
                std::move(live)) &&
                !live.valid() &&
                !owner->RevertPersistentRank2TrajectoryTransaction(
                    std::move(live)),
            "rank2 exact revert and stale reuse rejection"))
      return 1;
    sc25::PersistentRank2TrajectoryState after_revert;
    if (!collectively_require(
            owner->CopyPublishedRank2TrajectoryState(after_revert) &&
                EqualRank2StateBytes(after_revert, step0_state) &&
                EqualRank2EvidenceBoundary(
                    owner->rank2_trajectory_evidence(), *step0_evidence),
            "rank2 revert restores exact step0 state/evidence"))
      return 1;

    sc25::PersistentSc25ScalarIgrCpuFoundation::Rank2Transaction
        finalized_transaction;
    if (!collectively_require(
            Accepted(owner->PreparePersistentRank2TrajectoryTransaction(
                finalized_transaction)) &&
                owner->InstallPersistentRank2TrajectoryTransaction(
                    finalized_transaction),
            "rank2 second transaction prepare/install"))
      return 1;
    const dg::hdg::runtime::StateHash future_manifest =
        finalized_transaction.future_checkpoint().canonical_manifest_hash;
    if (!collectively_require(
            owner->FinalizePersistentRank2TrajectoryTransaction(
                std::move(finalized_transaction)) &&
                !finalized_transaction.valid() &&
                !owner->FinalizePersistentRank2TrajectoryTransaction(
                    std::move(finalized_transaction)),
            "rank2 finalize and stale reuse rejection"))
      return 1;
    sc25::PersistentRank2TrajectoryCheckpoint finalized;
    if (!collectively_require(
            owner->CaptureRank2TrajectoryCheckpoint(finalized) &&
                finalized.canonical_manifest_hash == future_manifest &&
                owner->rank2_trajectory_evidence().coordinates.step == 1,
            "rank2 finalized checkpoint equals frozen future checkpoint"))
      return 1;
    sc25::PersistentRank2TrajectoryCheckpoint imported;
    dg::hdg::runtime::PersistentResourceDescriptor resources;
    if (!collectively_require(
            owner->CopyPersistentResourceDescriptor(resources) &&
                owner->ImportRank2TrajectoryCheckpoint(
                    finalized.descriptor.coordinates, finalized.state,
                    imported) &&
                imported.canonical_manifest_hash ==
                    finalized.canonical_manifest_hash &&
                imported.raw_field_hashes.interior ==
                    finalized.raw_field_hashes.interior &&
                imported.raw_field_hashes.trace ==
                    finalized.raw_field_hashes.trace,
            "rank2 exact app-codec checkpoint import"))
      return 1;
    dg::hdg::runtime::StateCoordinates corrupt_coordinates =
        finalized.descriptor.coordinates;
    if (rank == 1) ++corrupt_coordinates.cursor;
    if (!collectively_require(
            !owner->ImportRank2TrajectoryCheckpoint(
                corrupt_coordinates, finalized.state, imported),
            "rank2 collective mismatched-coordinate import rejection"))
      return 1;
  }

  sc25::PersistentRank2TrajectoryCheckpoint checkpoint10;
  sc25::PersistentTrajectoryCheckpoint peer_checkpoint10;
  for (int step = 1; step <= sc25::persistent_required_steps; ++step)
  {
    const sc25::TrajectoryStatus rank2_status =
        baseline->AdvancePersistentRank2Trajectory();
    const sc25::TrajectoryStatus peer_status =
        peer->AdvancePersistentTrajectory();
    if (!collectively_require(Accepted(rank2_status) && Accepted(peer_status),
                              "rank2/rank1-peer baseline advance"))
    {
      return 1;
    }
    if (step == 10 &&
        !collectively_require(
            baseline->CaptureRank2TrajectoryCheckpoint(checkpoint10) &&
                peer->CaptureTrajectoryCheckpoint(peer_checkpoint10),
            "rank2 step-10 checkpoint capture"))
    {
      return 1;
    }
  }
  sc25::PersistentRank2TrajectoryState baseline_final;
  if (!collectively_require(
          baseline->CopyPublishedRank2TrajectoryState(baseline_final),
          "rank2 baseline final-state copy"))
  {
    return 1;
  }
  const sc25::PersistentRank2TrajectoryEvidence& baseline_evidence =
      baseline->rank2_trajectory_evidence();
  int baseline_newton_iterations = 0;
  int baseline_gmres_solves = 0;
  int baseline_gmres_iterations = 0;
  int baseline_line_search_trials = 0;
  for (int attempt = 0; attempt < baseline_evidence.attempt_count; ++attempt)
  {
    const sc25::Rank2TrajectoryAttemptEvidence& item =
        baseline_evidence.attempts[static_cast<std::size_t>(attempt)];
    baseline_newton_iterations += item.newton_iterations;
    baseline_gmres_solves += item.gmres_solves;
    baseline_gmres_iterations += item.gmres_iterations;
    baseline_line_search_trials += item.line_search_trials;
  }
  if (baseline_evidence.accepted_steps != 20 ||
      baseline_evidence.attempted_steps != 20 ||
      baseline_evidence.retry_attempts != 0 ||
      baseline_evidence.attempt_count != 20 ||
      baseline_evidence.step_evidence_count != 20 ||
      baseline_evidence.trace_restart_dimension != 50 ||
      baseline_evidence.trace_maximum_iterations != 200 ||
      baseline_evidence.ilu_type != 0 ||
      baseline_evidence.ilu_level_of_fill != 0 ||
      baseline_evidence.ilu_local_reordering != 1 ||
      baseline_evidence.detected_w8_owned_boundary_identity_changes != 0 ||
      baseline_evidence.w8_device_transfer_calls != 0 ||
      baseline_evidence.opaque_mfem_hypre_observed ||
      baseline_newton_iterations <= 0 ||
      baseline_gmres_solves != baseline_newton_iterations ||
      baseline_gmres_iterations <= 0 ||
      baseline_line_search_trials < baseline_newton_iterations)
  {
    fail("rank2 frozen trajectory/reuse evidence");
  }

  std::array<std::uint8_t, 32> local_manifest =
      checkpoint10.canonical_manifest_hash.bytes;
  std::array<std::uint8_t, 64> shard_manifests{};
  std::array<std::uint8_t, 32> local_shard =
      checkpoint10.raw_field_hashes.shard.bytes;
  std::array<std::uint8_t, 64> shard_hashes{};
  if (MPI_Allgather(local_manifest.data(), 32, MPI_UINT8_T,
                    shard_manifests.data(), 32, MPI_UINT8_T,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allgather(local_shard.data(), 32, MPI_UINT8_T,
                    shard_hashes.data(), 32, MPI_UINT8_T,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      checkpoint10.shard_rank != rank ||
      std::all_of(local_manifest.begin(), local_manifest.end(),
                  [](const std::uint8_t value) { return value == 0; }) ||
      std::memcmp(shard_manifests.data(), shard_manifests.data() + 32, 32) !=
          0 ||
      std::memcmp(shard_hashes.data(), shard_hashes.data() + 32, 32) == 0)
  {
    fail("rank2 two-shard manifest identity");
  }

  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> replay =
      CreateRank2Foundation();
  if (!collectively_require(
          replay && replay->InitializePersistentRank2Trajectory() ==
                  sc25::TrajectoryStatus::ready &&
              Accepted(replay->RestoreRank2TrajectoryCheckpoint(checkpoint10)),
          "rank2 step-10 two-shard restore"))
  {
    return 1;
  }
  sc25::PersistentRank2TrajectoryState restored;
  if (!replay->CopyPublishedRank2TrajectoryState(restored) ||
      !EqualRank2StateBytes(restored, checkpoint10.state) ||
      replay->rank2_trajectory_evidence().coordinates.step != 10 ||
      replay->rank2_trajectory_evidence().step_evidence_count != 0 ||
      replay->rank2_trajectory_evidence().attempt_count != 0 ||
      replay->rank2_trajectory_evidence().first_failure !=
          sc25::TrajectoryStatus::uninitialized)
  {
    fail("rank2 immediate restored shard equality/evidence reset");
  }
  for (int step = 11; step <= sc25::persistent_required_steps; ++step)
  {
    if (!collectively_require(
            Accepted(replay->AdvancePersistentRank2Trajectory()),
            "rank2 restored trajectory advance"))
    {
      return 1;
    }
  }
  sc25::PersistentRank2TrajectoryState replay_final;
  if (!replay->CopyPublishedRank2TrajectoryState(replay_final) ||
      !EqualRank2StateBytes(replay_final, baseline_final))
  {
    fail("rank2 step-10 bitwise shard replay");
  }

  constexpr std::array<sc25::TrajectoryFailureInjection, 4> injections = {
      sc25::TrajectoryFailureInjection::local_factor,
      sc25::TrajectoryFailureInjection::trace_linear,
      sc25::TrajectoryFailureInjection::admissibility,
      sc25::TrajectoryFailureInjection::newton_iteration_limit};
  for (const sc25::TrajectoryFailureInjection injection : injections)
  {
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> campaign =
        CreateRank2Foundation();
    if (!collectively_require(
            campaign && campaign->InitializePersistentRank2Trajectory() ==
                    sc25::TrajectoryStatus::ready,
            "rank2 failure campaign initialization"))
    {
      return 1;
    }
    for (int step = 1; step <= 6; ++step)
    {
      if (!collectively_require(
              Accepted(campaign->AdvancePersistentRank2Trajectory()),
              "rank2 failure campaign prefix"))
      {
        return 1;
      }
    }
    sc25::PersistentRank2TrajectoryState before_injection;
    campaign->CopyPublishedRank2TrajectoryState(before_injection);
    const sc25::TrajectoryStatus injected_status =
        campaign->AdvancePersistentRank2Trajectory(
            rank == 1 ? injection :
                        sc25::TrajectoryFailureInjection::none);
    if (!collectively_require(
            injected_status ==
                sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry,
            "rank2 rank1-origin same-dt retry status"))
    {
      return 1;
    }
    const sc25::PersistentRank2TrajectoryEvidence& campaign_evidence =
        campaign->rank2_trajectory_evidence();
    const sc25::Rank2TrajectoryAttemptEvidence& failed_attempt =
        campaign_evidence.attempts[6];
    const sc25::Rank2TrajectoryAttemptEvidence& retry_attempt =
        campaign_evidence.attempts[7];
    if (failed_attempt.target_step != 7 || failed_attempt.retry_index != 0 ||
        retry_attempt.target_step != 7 || retry_attempt.retry_index != 1 ||
        failed_attempt.dt != sc25::persistent_dt ||
        retry_attempt.dt != sc25::persistent_dt ||
        !EqualRank2Hashes(failed_attempt.entry_hashes,
                          failed_attempt.published_hashes) ||
        !EqualRank2Hashes(failed_attempt.entry_hashes,
                          retry_attempt.entry_hashes) ||
        !EqualRank2Hashes(retry_attempt.candidate_hashes,
                          retry_attempt.published_hashes) ||
        !EqualRank2StateBytes(
            campaign_evidence.stages[6].accepted_state,
            baseline_evidence.stages[6].accepted_state))
    {
      fail("rank2 failed-attempt atomicity/retry evidence");
    }
    if (rank == 1 && failed_attempt.injection != injection)
    {
      fail("rank2 rank1 injection provenance");
    }
    if (rank == 0 && failed_attempt.injection !=
                         sc25::TrajectoryFailureInjection::none)
    {
      fail("rank2 rank0 non-injection provenance");
    }
    sc25::TrajectoryStatus expected_failure =
        sc25::TrajectoryStatus::uninitialized;
    int expected_attempted_newton = 1;
    int expected_completed_newton = 0;
    int expected_gmres_solves = 0;
    int expected_line_trials = 0;
    if (injection == sc25::TrajectoryFailureInjection::local_factor)
    {
      expected_failure =
          sc25::TrajectoryStatus::HdgSc25InjectedLocalFactorFailure;
    }
    else if (injection == sc25::TrajectoryFailureInjection::trace_linear)
    {
      expected_failure =
          sc25::TrajectoryStatus::HdgSc25InjectedTraceLinearFailure;
    }
    else if (injection == sc25::TrajectoryFailureInjection::admissibility)
    {
      expected_failure =
          sc25::TrajectoryStatus::HdgSc25InjectedAdmissibilityFailure;
      expected_completed_newton = 1;
      expected_gmres_solves = 1;
      expected_line_trials = sc25::persistent_line_search_trials;
    }
    else
    {
      expected_failure =
          sc25::TrajectoryStatus::HdgSc25InjectedNewtonIterationLimit;
      expected_attempted_newton =
          sc25::persistent_newton_maximum_iterations;
    }
    if (campaign_evidence.first_failure != expected_failure ||
        failed_attempt.status != expected_failure ||
        failed_attempt.attempted_newton_iterations !=
            expected_attempted_newton ||
        failed_attempt.newton_iterations != expected_completed_newton ||
        failed_attempt.gmres_solves != expected_gmres_solves ||
        (expected_gmres_solves == 0 ?
             failed_attempt.gmres_iterations != 0 :
             failed_attempt.gmres_iterations <= 0) ||
        failed_attempt.line_search_trials != expected_line_trials)
    {
      fail("rank2 exact injected-attempt status/counters");
    }
    int observed_attempted_newton = 0;
    int observed_completed_newton = 0;
    int observed_visited_trials = 0;
    bool exact_newton_trial_layout = true;
    for (int iteration = 0;
         iteration < sc25::persistent_newton_maximum_iterations; ++iteration)
    {
      const sc25::Rank2TrajectoryNewtonEvidence& newton =
          failed_attempt.newton[static_cast<std::size_t>(iteration)];
      observed_attempted_newton += newton.attempted ? 1 : 0;
      observed_completed_newton += newton.completed ? 1 : 0;
      exact_newton_trial_layout = exact_newton_trial_layout &&
          newton.attempted == (iteration < expected_attempted_newton) &&
          newton.completed == (iteration < expected_completed_newton) &&
          newton.iteration ==
              (iteration < expected_attempted_newton ? iteration : -1);
      for (int line = 0; line < sc25::persistent_line_search_trials; ++line)
      {
        const bool expected_visited =
            injection == sc25::TrajectoryFailureInjection::admissibility &&
            iteration == 0;
        const std::size_t line_index = static_cast<std::size_t>(line);
        observed_visited_trials += newton.trial_visited[line_index] != 0 ?
            1 : 0;
        exact_newton_trial_layout = exact_newton_trial_layout &&
            (newton.trial_visited[line_index] != 0) == expected_visited;
        if (!expected_visited)
        {
          exact_newton_trial_layout = exact_newton_trial_layout &&
              std::isinf(newton.trial_residual_norms[line_index]) &&
              newton.trial_admissible[line_index] == 0 &&
              newton.trial_armijo[line_index] == 0;
        }
      }
    }
    if (!exact_newton_trial_layout ||
        observed_attempted_newton != expected_attempted_newton ||
        observed_completed_newton != expected_completed_newton ||
        observed_visited_trials != expected_line_trials)
    {
      fail("rank2 exact injected 20-Newton/13-trial evidence layout");
    }
    if (injection == sc25::TrajectoryFailureInjection::admissibility)
    {
      for (int line = 0; line < sc25::persistent_line_search_trials; ++line)
      {
        if (!std::isinf(failed_attempt.trial_residual_norms[
                static_cast<std::size_t>(line)]) ||
            failed_attempt.newton[0].trial_visited[
                static_cast<std::size_t>(line)] == 0 ||
            !std::isinf(failed_attempt.newton[0].trial_residual_norms[
                static_cast<std::size_t>(line)]) ||
            failed_attempt.trial_admissible[static_cast<std::size_t>(line)] !=
                0 ||
            failed_attempt.newton[0].trial_admissible[
                static_cast<std::size_t>(line)] != 0 ||
            failed_attempt.trial_armijo[static_cast<std::size_t>(line)] != 0 ||
            failed_attempt.newton[0].trial_armijo[
                static_cast<std::size_t>(line)] != 0)
        {
          fail("rank2 exact injected-admissibility trial evidence");
          break;
        }
      }
    }
    if (EqualRank2StateBytes(before_injection,
                             campaign_evidence.stages[6].accepted_state))
    {
      fail("rank2 retry failed to publish accepted step");
    }
    for (int step = 8; step <= sc25::persistent_required_steps; ++step)
    {
      if (!collectively_require(
              Accepted(campaign->AdvancePersistentRank2Trajectory()),
              "rank2 failure campaign suffix"))
      {
        return 1;
      }
    }
    sc25::PersistentRank2TrajectoryState campaign_final;
    if (!campaign->CopyPublishedRank2TrajectoryState(campaign_final) ||
        !EqualRank2StateBytes(campaign_final, baseline_final) ||
        campaign->rank2_trajectory_evidence().retry_attempts != 1 ||
        campaign->rank2_trajectory_evidence().attempt_count != 21)
    {
      fail("rank2 failure campaign bitwise final parity");
    }
    if (!collectively_require(
            Accepted(campaign->RestoreRank2TrajectoryCheckpoint(checkpoint10)),
            "rank2 successful restore after injected failure") ||
        campaign->rank2_trajectory_evidence().first_failure !=
            sc25::TrajectoryStatus::uninitialized ||
        campaign->rank2_trajectory_evidence().attempt_count != 0 ||
        campaign->rank2_trajectory_evidence().step_evidence_count != 0)
    {
      fail("rank2 successful restore clears stale failure/attempt evidence");
    }
  }

  {
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> drift =
        CreateRank2Foundation();
    if (!collectively_require(
            drift && drift->InitializePersistentRank2Trajectory() ==
                    sc25::TrajectoryStatus::ready,
            "rank2 boundary-drift negative initialization"))
    {
      return 1;
    }
    for (int step = 1; step <= 6; ++step)
    {
      if (!collectively_require(
              Accepted(drift->AdvancePersistentRank2Trajectory()),
              "rank2 boundary-drift negative prefix"))
      {
        return 1;
      }
    }
    sc25::PersistentRank2TrajectoryState before;
    sc25::PersistentRank2TrajectoryState after;
    drift->CopyPublishedRank2TrajectoryState(before);
    const sc25::PersistentRank2TrajectoryEvidence before_evidence =
        drift->rank2_trajectory_evidence();
    dg_igr_sc25_rank2_test_inject_boundary_drift(rank == 1 ? 1 : 0);
    const sc25::TrajectoryStatus status =
        drift->AdvancePersistentRank2Trajectory();
    drift->CopyPublishedRank2TrajectoryState(after);
    const sc25::PersistentRank2TrajectoryEvidence& after_evidence =
        drift->rank2_trajectory_evidence();
    const std::uint64_t local_detected =
        after_evidence.detected_w8_owned_boundary_identity_changes -
        before_evidence.detected_w8_owned_boundary_identity_changes;
    std::uint64_t globally_detected = 0;
    MPI_Allreduce(&local_detected, &globally_detected, 1, MPI_UINT64_T,
                  MPI_SUM, MPI_COMM_WORLD);
    if (!collectively_require(
            status == sc25::TrajectoryStatus::HdgSc25PublicationFailure,
            "rank2 collective late-publication drift status") ||
        !EqualRank2StateBytes(before, after) ||
        after_evidence.coordinates.time != before_evidence.coordinates.time ||
        after_evidence.coordinates.step != before_evidence.coordinates.step ||
        after_evidence.coordinates.generation !=
            before_evidence.coordinates.generation ||
        after_evidence.coordinates.cursor !=
            before_evidence.coordinates.cursor ||
        after_evidence.accepted_steps != before_evidence.accepted_steps ||
        after_evidence.step_evidence_count !=
            before_evidence.step_evidence_count ||
        after_evidence.attempt_count != before_evidence.attempt_count + 1 ||
        local_detected != static_cast<std::uint64_t>(rank == 1) ||
        globally_detected != 1)
    {
      fail("rank2 late-publication atomicity/deep boundary-drift detection");
    }
  }

  {
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> install =
        CreateRank2Foundation();
    if (!collectively_require(
            install && install->InitializePersistentRank2Trajectory() ==
                    sc25::TrajectoryStatus::ready,
            "rank2 final-install negative initialization"))
    {
      return 1;
    }
    sc25::PersistentRank2TrajectoryState before;
    sc25::PersistentRank2TrajectoryState after;
    install->CopyPublishedRank2TrajectoryState(before);
    const sc25::PersistentRank2TrajectoryEvidence before_evidence =
        install->rank2_trajectory_evidence();
    dg_igr_sc25_rank2_test_inject_final_install_mismatch(rank == 1 ? 1 : 0);
    const sc25::TrajectoryStatus status =
        install->RestoreRank2TrajectoryCheckpoint(checkpoint10);
    install->CopyPublishedRank2TrajectoryState(after);
    const sc25::PersistentRank2TrajectoryEvidence& after_evidence =
        install->rank2_trajectory_evidence();
    if (!collectively_require(
            status ==
                sc25::TrajectoryStatus::HdgSc25RestartValidationFailureHash,
            "rank2 collective final-install mismatch status") ||
        !EqualRank2StateBytes(before, after) ||
        after_evidence.coordinates.time != before_evidence.coordinates.time ||
        after_evidence.coordinates.step != before_evidence.coordinates.step ||
        after_evidence.coordinates.generation !=
            before_evidence.coordinates.generation ||
        after_evidence.coordinates.cursor !=
            before_evidence.coordinates.cursor ||
        after_evidence.accepted_steps != before_evidence.accepted_steps ||
        after_evidence.attempt_count != before_evidence.attempt_count)
    {
      fail("rank2 final-install validation atomicity");
    }
  }

  bool local_auxiliary_negative_proof = true;
  double rank2_auxiliary_q12_q14_sensitivity = 0.0;
  if (rank == 1)
  {
    const int peer_element = GeometricPeerElement(*baseline, 0, *peer);
    sc25::PersistentTrajectoryState corrupted = peer_checkpoint10.state;
    sc25::AuxiliaryResidualMetrics q12{};
    sc25::AuxiliaryResidualMetrics q14{};
    if (peer_element < 0)
    {
      local_auxiliary_negative_proof = false;
    }
    else
    {
      corrupted.sigma[static_cast<std::size_t>(peer_element) *
                          sc25::persistent_scalar_volume_nodes] += 1.0;
      local_auxiliary_negative_proof =
          dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
              *peer, corrupted, sc25::coupled_fixture_quadrature_order,
              q12) &&
          dg_igr_hdg_sc25_cpu_test::EvaluateIndependentAuxiliaryResidualOracle(
              *peer, corrupted,
              sc25::coupled_fixture_repeat_quadrature_order, q14) &&
          q12.normalized_maximum > 1.0e-10 &&
          q14.normalized_maximum > 1.0e-10;
      rank2_auxiliary_q12_q14_sensitivity = std::abs(
          q12.normalized_maximum - q14.normalized_maximum) /
          std::max({1.0, std::abs(q12.normalized_maximum),
                    std::abs(q14.normalized_maximum)});
      local_auxiliary_negative_proof = local_auxiliary_negative_proof &&
          rank2_auxiliary_q12_q14_sensitivity <= 1.0e-10;
    }
  }
  if (!collectively_require(local_auxiliary_negative_proof,
                            "rank2 independent auxiliary q12/q14 negative"))
  {
    return 1;
  }

  const auto rejection_control = [rank, &fail, &collectively_require,
                                  &checkpoint10](
                                     const int control)
  {
    std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> negative =
        CreateRank2Foundation();
    if (!collectively_require(
            negative && negative->InitializePersistentRank2Trajectory() ==
                    sc25::TrajectoryStatus::ready,
            "rank2 restart negative initialization"))
    {
      return false;
    }
    sc25::PersistentRank2TrajectoryCheckpoint candidate = checkpoint10;
    sc25::TrajectoryStatus expected =
        sc25::TrajectoryStatus::HdgSc25RestartValidationFailureIdentity;
    if (rank == 1)
    {
      if (control == 0)
      {
        candidate.tuple =
            dg::hdg::runtime::StableIdentity::Copy(sc25::cpu_rank1_tuple_id);
      }
      else if (control == 1)
      {
        std::uint8_t* raw = reinterpret_cast<std::uint8_t*>(
            candidate.state.u.data());
        raw[0] ^= 1U;
        expected =
            sc25::TrajectoryStatus::HdgSc25RestartValidationFailureHash;
      }
      else if (control == 2)
      {
        candidate.state.u[0] = 0.0;
        expected = sc25::TrajectoryStatus::
            HdgSc25RestartValidationFailureAdmissibility;
      }
      else
      {
        candidate.state.sigma[0] += 1.0;
        expected = sc25::TrajectoryStatus::
            HdgSc25RestartValidationFailureAuxResidual;
      }
    }
    else if (control == 1)
    {
      expected =
          sc25::TrajectoryStatus::HdgSc25RestartValidationFailureHash;
    }
    else if (control == 2)
    {
      expected = sc25::TrajectoryStatus::
          HdgSc25RestartValidationFailureAdmissibility;
    }
    else if (control == 3)
    {
      expected =
          sc25::TrajectoryStatus::HdgSc25RestartValidationFailureAuxResidual;
    }
    if (control == 2 &&
        (dg_igr_hdg_sc25_cpu_test::
             EvaluateIndependentRank2StateAdmissibilityOracle(
                 *negative, candidate.state,
                 sc25::coupled_fixture_quadrature_order) ||
         dg_igr_hdg_sc25_cpu_test::
             EvaluateIndependentRank2StateAdmissibilityOracle(
                 *negative, candidate.state,
                 sc25::coupled_fixture_repeat_quadrature_order)))
    {
      fail("rank2 independent state-admissibility negative proof");
      return false;
    }
    if ((control == 2 || control == 3) &&
        !dg_igr_hdg_sc25_cpu_test::RehashIndependentRank2Checkpoint(candidate))
    {
      fail("rank2 independent combined-manifest rehash");
      return false;
    }
    sc25::PersistentRank2TrajectoryState before;
    sc25::PersistentRank2TrajectoryState after;
    negative->CopyPublishedRank2TrajectoryState(before);
    const sc25::TrajectoryStatus status =
        negative->RestoreRank2TrajectoryCheckpoint(candidate);
    negative->CopyPublishedRank2TrajectoryState(after);
    if (!collectively_require(status == expected,
                              "rank2 collective restart rejection status") ||
        !EqualRank2StateBytes(before, after))
    {
      fail("rank2 restart rejection atomicity");
      return false;
    }
    return true;
  };
  for (int control = 0; control < 4; ++control)
  {
    if (!rejection_control(control)) { break; }
  }

  int global_failures = 0;
  MPI_Allreduce(&local_failures, &global_failures, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (global_failures == 0 && rank == 0)
  {
    std::cout
        << "SC25_RANK2_REUSE_SCOPE scope=w8_owned_boundary_census "
        << "identity_changes="
        << baseline_evidence.detected_w8_owned_boundary_identity_changes
        << " w8_device_transfer_calls="
        << baseline_evidence.w8_device_transfer_calls
        << " opaque_mfem_hypre_observed="
        << (baseline_evidence.opaque_mfem_hypre_observed ? 1 : 0) << '\n';
    std::cout << "SC25_RANK2_RUNTIME mpi_ranks=2 rank2_accepted_steps="
              << baseline_evidence.accepted_steps
              << " serial_peer_accepted_steps="
              << peer->trajectory_evidence().accepted_steps
              << " gmres_solves=" << baseline_gmres_solves
              << " gmres_iterations=" << baseline_gmres_iterations
              << " newton_iterations=" << baseline_newton_iterations
              << " line_search_trials=" << baseline_line_search_trials
              << '\n';
    std::cout << "PASS: " << sc25::persistent_rank2_fixture_id
              << " (checkpoint/retry/admissibility checks, Q12/Q14="
              << rank2_auxiliary_q12_q14_sensitivity << ")\n";
  }
  return global_failures;
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
  const bool rank2_trajectory = argc == 2 &&
      std::string_view(argv[1]) == "--rank2-trajectory";
  if (rank2_trajectory)
  {
    if (ranks != 2)
    {
      if (rank == 0)
      {
        std::cerr << "SC25 rank2 harness requires exactly two ranks\n";
      }
      failures = 1;
    }
    else
    {
      failures = RunRank2TrajectoryHarness();
    }
  }
  else if (ranks != 1)
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
