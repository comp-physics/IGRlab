#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <new>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
using dg::hdg::generic::HdgEulerSessionCapability;
using dg::hdg::generic::HdgEulerSessionCapabilityBit;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using namespace dg::hdg::runtime::cuda_w8a;
using GenericPreparedOutput = CudaPreparedOutputMetadataV1;

bool All(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

mfem::Mesh PeriodicSquareMesh(const int n)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  return mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
}

mfem::Mesh NonperiodicSquareMesh(const int n)
{
  return mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
}

CudaConfig BorrowedConfig(mfem::ParMesh& mesh, const int ranks,
                          const int order, const int n)
{
  // Each returned config receives one unique stable slot.  The bounded pool
  // fails closed rather than reusing/overwriting a queued config's input.
  using storage_type = std::array<double, maximum_rank_elements *
      maximum_interior_dofs_per_element>;
  static std::array<storage_type, 128> fixture_storage{};
  static std::size_t fixture_count = 0;
  if (fixture_count == fixture_storage.size()) throw std::bad_alloc();
  storage_type& initial = fixture_storage[fixture_count++];
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, conserved_components, mfem::Ordering::byNODES);
  const std::size_t scalar_count = static_cast<std::size_t>(space.GetVSize() /
      conserved_components);
  if (scalar_count * conserved_components > initial.size())
    throw std::bad_alloc();
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
  {
    const double perturbation = 1.0e-3 * static_cast<double>(scalar + 1);
    initial[scalar] = 1.0 + perturbation;
    initial[scalar_count + scalar] = 0.20 + 0.25 * perturbation;
    initial[2 * scalar_count + scalar] = -0.10 + 0.15 * perturbation;
    initial[3 * scalar_count + scalar] = 2.60 + 0.10 * perturbation;
  }
  CudaConfig config{};
  config.order = order;
  config.mesh_n = n;
  config.requested_device = 0;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = &mesh;
  config.borrowed_communicator = MPI_COMM_WORLD;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {initial.data(), scalar_count,
                             conserved_components};
  config.borrowed_source = {nullptr, 0, conserved_components};
  return config;
}

struct HostFields
{
  std::vector<double> initial;
  std::vector<double> source;
  std::size_t scalar_count = 0;
};

HostFields NonuniformHostFields(mfem::ParMesh& mesh, const int order,
                                const bool nonzero_source)
{
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, conserved_components, mfem::Ordering::byNODES);
  HostFields fields;
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() /
      conserved_components);
  fields.initial.resize(fields.scalar_count * conserved_components);
  if (nonzero_source)
    fields.source.resize(fields.initial.size());
  for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
  {
    const double value = 1.0e-3 * static_cast<double>(scalar + 1);
    fields.initial[scalar] = 1.0 + value;
    fields.initial[fields.scalar_count + scalar] = 0.20 + 0.25 * value;
    fields.initial[2 * fields.scalar_count + scalar] = -0.10 + 0.15 * value;
    fields.initial[3 * fields.scalar_count + scalar] = 2.60 + 0.10 * value;
    if (nonzero_source)
    {
      fields.source[scalar] = 0.01 * value;
      fields.source[fields.scalar_count + scalar] = -0.02 * value;
      fields.source[2 * fields.scalar_count + scalar] = 0.03 * value;
      fields.source[3 * fields.scalar_count + scalar] = 0.04 * value;
    }
  }
  return fields;
}

HostFields HighWaveHostFields(mfem::ParMesh& mesh, const int order)
{
  HostFields fields = NonuniformHostFields(mesh, order, false);
  for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
  {
    const double rho = 1.0 + 1.0e-3 * static_cast<double>(scalar + 1);
    const double u = 2.0;
    const double v = 0.0;
    const double pressure = 1.0;
    fields.initial[scalar] = rho;
    fields.initial[fields.scalar_count + scalar] = rho * u;
    fields.initial[2 * fields.scalar_count + scalar] = rho * v;
    fields.initial[3 * fields.scalar_count + scalar] = pressure / 0.4 +
        0.5 * rho * (u * u + v * v);
  }
  return fields;
}

CudaConfig BorrowedDataConfig(mfem::ParMesh& mesh, const int ranks,
                              const int order, const int n,
                              const HostFields& fields)
{
  CudaConfig config = BorrowedConfig(mesh, ranks, order, n);
  config.borrowed_initial = {fields.initial.data(), fields.scalar_count,
                             conserved_components};
  config.borrowed_source = fields.source.empty()
      ? CudaBorrowedHostL2Input{nullptr, 0, conserved_components}
      : CudaBorrowedHostL2Input{fields.source.data(), fields.scalar_count,
                                conserved_components};
  return config;
}

CudaConfig LegacyBorrowedConfig(mfem::ParMesh& mesh, const int ranks,
                                const int order, const int n)
{
  CudaConfig config = BorrowedConfig(mesh, ranks, order, n);
  config.use_borrowed_host_inputs = false;
  config.borrowed_initial = {nullptr, 0, conserved_components};
  config.borrowed_source = {nullptr, 0, conserved_components};
  return config;
}

bool SameValues(const std::vector<double>& left,
                const std::vector<double>& right) noexcept
{
  return left == right;
}

bool FullyFiniteOverwritten(const std::vector<double>& values,
                            const double sentinel) noexcept
{
  for (const double value : values)
    if (!std::isfinite(value) || value == sentinel) return false;
  return true;
}

bool SameCoordinates(const dg::hdg::generic::HdgEulerSessionCoordinates& left,
                     const dg::hdg::runtime::StateCoordinates& right) noexcept
{
  return left.time == right.time && left.dt == right.dt &&
      left.step == right.step && left.generation == right.generation;
}

bool SameNativeExport(const CudaPreparedOutputMetadataV1& left,
                      const CudaPreparedOutputMetadataV1& right) noexcept
{
  const auto same_attempt = [](const dg::hdg::runtime::AttemptIdentity& first,
                               const dg::hdg::runtime::AttemptIdentity& second)
  {
    return first.sequence == second.sequence &&
        first.committed_step == second.committed_step &&
        first.committed_generation == second.committed_generation &&
        first.retry_index == second.retry_index &&
        dg::hdg::runtime::detail::DoubleBits(first.dt) ==
            dg::hdg::runtime::detail::DoubleBits(second.dt);
  };
  const auto same_range = [](const CudaTransferLedgerRangeV1& first,
                             const CudaTransferLedgerRangeV1& second)
  {
    return first.begin_ordinal == second.begin_ordinal &&
        first.end_ordinal == second.end_ordinal &&
        first.ordered_digest == second.ordered_digest &&
        first.complete == second.complete;
  };
  return left.schema == right.schema && left.source_rank == right.source_rank &&
      left.rank_count == right.rank_count && same_attempt(left.attempt, right.attempt) &&
      left.facade_incarnation == right.facade_incarnation &&
      dg::hdg::runtime::detail::SameCheckpoint(left.descriptor, right.descriptor) &&
      left.local_mesh_hash == right.local_mesh_hash &&
      left.local_partition_hash == right.local_partition_hash &&
      left.local_topology_hash == right.local_topology_hash &&
      left.local_interior_hash == right.local_interior_hash &&
      left.local_trace_hash == right.local_trace_hash &&
      left.interior_count == right.interior_count &&
      left.owned_true_trace_count == right.owned_true_trace_count &&
      left.interior_source_token == right.interior_source_token &&
      left.trace_source_token == right.trace_source_token &&
      left.staging_generation == right.staging_generation &&
      same_range(left.staging_range, right.staging_range) &&
      left.candidate_frozen == right.candidate_frozen &&
      left.checkpoint_staged == right.checkpoint_staged &&
      left.output_staged == right.output_staged &&
      left.successful_commit == right.successful_commit;
}

bool CheckTransaction(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  const std::uint32_t expected_capabilities =
      HdgEulerSessionCapabilityBit(HdgEulerSessionCapability::prepared_commit) |
      HdgEulerSessionCapabilityBit(HdgEulerSessionCapability::prepared_output) |
      HdgEulerSessionCapabilityBit(
          HdgEulerSessionCapability::generic_borrowed_inputs);
  if (!All(initialized.status == HdgEulerSessionStatus::initialized &&
           session.capabilities() == expected_capabilities &&
           (session.capabilities() & HdgEulerSessionCapabilityBit(
               HdgEulerSessionCapability::committed_cfl_diagnostics)) == 0 &&
           (session.capabilities() & HdgEulerSessionCapabilityBit(
               HdgEulerSessionCapability::generic_borrowed_inputs)) != 0))
    return false;

  GenericEulerCudaPreparedStep prepared{};
  GenericPreparedOutput native_export{};
  const auto staged = session.Prepare(prepared, native_export);
  const auto before_commit = session.coordinates();
  const bool native_staged = staged.status == HdgEulerSessionStatus::prepared &&
      native_export.candidate_frozen && native_export.checkpoint_staged &&
      !native_export.output_staged && !native_export.successful_commit;
  const GenericPreparedOutput before_duplicate = native_export;
  GenericEulerCudaPreparedStep duplicate{};
  const auto duplicate_prepare = session.Prepare(duplicate, native_export);
  const bool duplicate_preserved =
      duplicate_prepare.status == HdgEulerSessionStatus::rejected &&
      SameNativeExport(native_export, before_duplicate);
  const std::uint64_t facade_incarnation = native_export.facade_incarnation;
  ++native_export.facade_incarnation;
  const bool aba_rejected = !session.AcknowledgePreparedOutput(
      prepared, native_export);
  native_export.facade_incarnation = facade_incarnation;
  const auto no_ack_commit = session.Commit(std::move(prepared), native_export);
  const bool acknowledged = session.AcknowledgePreparedOutput(
      prepared, native_export);
  if (!All(native_staged &&
           duplicate_preserved && aba_rejected &&
           no_ack_commit.status == HdgEulerSessionStatus::rejected &&
           session.coordinates().step == before_commit.step &&
           session.coordinates().generation == before_commit.generation &&
           acknowledged))
    return false;

  const auto committed = session.Commit(std::move(prepared), native_export);
  const auto after_commit = session.coordinates();
  if (!All(committed.status == HdgEulerSessionStatus::committed &&
           native_export.successful_commit &&
           SameCoordinates(after_commit, native_export.descriptor.coordinates) &&
           after_commit.step == before_commit.step + 1 &&
           after_commit.generation == before_commit.generation + 1))
    return false;

  GenericEulerCudaPreparedStep abort_authority{};
  GenericPreparedOutput aborted_export{};
  const auto abort_prepared = session.Prepare(abort_authority, aborted_export);
  GenericEulerHdgCudaSession foreign;
  const bool foreign_rejected = !foreign.AcknowledgePreparedOutput(
      abort_authority, aborted_export);
  const auto aborted = session.Abort(std::move(abort_authority), aborted_export);
  const bool stale_rejected = !session.AcknowledgePreparedOutput(
      abort_authority, aborted_export);
  const auto double_abort = session.Abort(std::move(abort_authority),
                                          aborted_export);
  if (!All(abort_prepared.status == HdgEulerSessionStatus::prepared &&
             foreign_rejected && aborted.status == HdgEulerSessionStatus::aborted &&
             session.coordinates().step == after_commit.step &&
             session.coordinates().generation == after_commit.generation &&
             !aborted_export.candidate_frozen &&
             !aborted_export.checkpoint_staged && !aborted_export.output_staged &&
             !aborted_export.successful_commit && stale_rejected &&
             double_abort.status == HdgEulerSessionStatus::rejected))
    return false;

  GenericEulerCudaPreparedStep retry_authority{};
  GenericPreparedOutput retry_export{};
  const auto retry_prepared = session.Prepare(retry_authority, retry_export);
  const bool retry_acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_export);
  const auto retry_committed = session.Commit(std::move(retry_authority),
                                              retry_export);
  return All(retry_prepared.status == HdgEulerSessionStatus::prepared &&
             retry_acknowledged &&
             retry_committed.status == HdgEulerSessionStatus::committed &&
             retry_export.successful_commit &&
             session.coordinates().step == after_commit.step + 1 &&
             session.coordinates().generation == after_commit.generation + 1);
}

bool CheckRankTwoCollectiveAdmission(const int ranks)
{
  if (ranks != 2) return true;
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  if (initialized.status !=
      HdgEulerSessionStatus::initialized)
    return false;
  GenericEulerCudaPreparedStep authority{};
  GenericPreparedOutput native_export{};
  const auto staged = session.Prepare(authority, native_export);
  const GenericPreparedOutput before_tamper = native_export;
  // This passes facade authority matching but makes one rank's checkpoint
  // descriptor stale, exercising engine-side collective atomicity.
  if (rank == 0) ++native_export.descriptor.coordinates.generation;
  const bool tamper_rejected = !session.AcknowledgePreparedOutput(
      authority, native_export);
  native_export.descriptor.coordinates = before_tamper.descriptor.coordinates;
  const bool tamper_atomic = tamper_rejected &&
      SameNativeExport(native_export, before_tamper) &&
      !native_export.output_staged && !native_export.successful_commit;
  const bool acknowledged = session.AcknowledgePreparedOutput(authority,
                                                               native_export);
  if (staged.status != HdgEulerSessionStatus::prepared || !tamper_atomic ||
      !acknowledged)
    return false;
  const auto mismatched = rank == 0
      ? session.Commit(std::move(authority), native_export)
      : session.Abort(std::move(authority), native_export);
  if (!All(mismatched.status == HdgEulerSessionStatus::rejected &&
           !native_export.successful_commit && session.coordinates().step == 0))
    return false;
  const auto initial_abort = session.Abort(std::move(authority), native_export);
  if (initial_abort.status !=
      HdgEulerSessionStatus::aborted)
    return false;

  GenericEulerCudaPreparedStep retry_authority{};
  GenericPreparedOutput retry_export{};
  const auto retry_prepared = session.Prepare(retry_authority, retry_export);
  if (retry_prepared.status !=
      HdgEulerSessionStatus::prepared)
    return false;
  GenericEulerCudaPreparedStep stale{};
  const auto stale_result = rank == 0
      ? session.Abort(std::move(stale), retry_export)
      : session.Abort(std::move(retry_authority), retry_export);
  if (!All(stale_result.status == HdgEulerSessionStatus::rejected &&
           session.coordinates().step == 0 && !retry_export.successful_commit))
    return false;
  const bool retry_acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_export);
  if (!retry_acknowledged)
    return false;
  const auto committed = session.Commit(std::move(retry_authority), retry_export);
  return All(committed.status == HdgEulerSessionStatus::committed &&
             retry_export.successful_commit && session.coordinates().step == 1 &&
             session.coordinates().generation == 1);
}

bool CheckRankTwoSessionIdentityAdmission(const int ranks)
{
  if (ranks != 2) return true;
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // This is collective-identity discrimination only.  The global CUDA
  // allocation baseline permits the last initialized engine (B) to advance;
  // A remains intentionally untouched after B establishes its baseline.
  GenericEulerHdgCudaSession a;
  GenericEulerHdgCudaSession b;
  const auto a_initialized = a.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  const auto b_initialized = b.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));

  GenericEulerCudaPreparedStep a_authority{};
  GenericEulerCudaPreparedStep b_authority{};
  GenericPreparedOutput a_export{};
  GenericPreparedOutput b_export{};
  const auto crossed_prepare = rank == 0
      ? a.Prepare(a_authority, a_export)
      : b.Prepare(b_authority, b_export);
  const bool a_untouched = a.coordinates().step == 0 &&
      a.coordinates().generation == 0 && !a_export.candidate_frozen &&
      !a_export.checkpoint_staged && !a_export.output_staged &&
      !a_export.successful_commit;
  const bool b_untouched = b.coordinates().step == 0 &&
      b.coordinates().generation == 0 && !b_export.candidate_frozen &&
      !b_export.checkpoint_staged && !b_export.output_staged &&
      !b_export.successful_commit;

  const auto b_prepared = b.Prepare(b_authority, b_export);
  const auto b_aborted = b.Abort(std::move(b_authority), b_export);
  return All(a_initialized.status == HdgEulerSessionStatus::initialized &&
             b_initialized.status == HdgEulerSessionStatus::initialized &&
             crossed_prepare.status == HdgEulerSessionStatus::rejected &&
             a_untouched && b_untouched &&
             b_prepared.status == HdgEulerSessionStatus::prepared &&
             b_aborted.status == HdgEulerSessionStatus::aborted &&
             b.coordinates().step == 0 && b.coordinates().generation == 0);
}

bool CheckForeignDestinationAuthority(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession owner;
  GenericEulerHdgCudaSession recipient;
  const auto recipient_initialized = recipient.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  const auto owner_initialized = owner.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  if (!All(owner_initialized.status == HdgEulerSessionStatus::initialized &&
           recipient_initialized.status == HdgEulerSessionStatus::initialized))
    return false;

  GenericEulerCudaPreparedStep owner_authority{};
  GenericPreparedOutput owner_export{};
  const auto owner_prepared = owner.Prepare(owner_authority, owner_export);
  const GenericPreparedOutput before_rejection = owner_export;
  GenericPreparedOutput recipient_export{};
  const GenericPreparedOutput recipient_before_rejection = recipient_export;
  const auto rejected = recipient.Prepare(owner_authority, recipient_export);
  const bool preserved = SameNativeExport(owner_export, before_rejection) &&
      SameNativeExport(recipient_export, recipient_before_rejection);
  const auto aborted = owner.Abort(std::move(owner_authority), owner_export);

  GenericEulerCudaPreparedStep retry_authority{};
  GenericPreparedOutput retry_export{};
  const auto retry_prepared = owner.Prepare(retry_authority, retry_export);
  const GenericPreparedOutput retry_before_rejection = retry_export;
  const auto retry_rejected = recipient.Prepare(retry_authority, recipient_export);
  const bool retry_preserved =
      SameNativeExport(retry_export, retry_before_rejection) &&
      SameNativeExport(recipient_export, recipient_before_rejection);
  const bool retry_acknowledged = owner.AcknowledgePreparedOutput(
      retry_authority, retry_export);
  const auto committed = owner.Commit(std::move(retry_authority), retry_export);
  return All(owner_prepared.status == HdgEulerSessionStatus::prepared &&
             rejected.status == HdgEulerSessionStatus::rejected && preserved &&
             aborted.status == HdgEulerSessionStatus::aborted &&
             retry_prepared.status == HdgEulerSessionStatus::prepared &&
             retry_rejected.status == HdgEulerSessionStatus::rejected &&
             retry_preserved &&
             retry_acknowledged &&
             committed.status == HdgEulerSessionStatus::committed &&
             retry_export.successful_commit &&
             recipient.coordinates().step == 0 &&
             recipient.coordinates().generation == 0);
}

bool CheckAddressReuseAba(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  using storage_type = std::aligned_storage_t<sizeof(GenericEulerHdgCudaSession),
                                              alignof(GenericEulerHdgCudaSession)>;
  storage_type storage{};
  auto* retired = new (&storage) GenericEulerHdgCudaSession;
  const auto retired_initialized = retired->Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  if (!All(retired_initialized.status == HdgEulerSessionStatus::initialized))
  {
    retired->~GenericEulerHdgCudaSession();
    return false;
  }
  GenericEulerCudaPreparedStep retired_authority{};
  GenericPreparedOutput shared_export{};
  const auto retired_prepared = retired->Prepare(retired_authority, shared_export);
  if (!All(retired_prepared.status == HdgEulerSessionStatus::prepared))
  {
    retired->~GenericEulerHdgCudaSession();
    return false;
  }
  retired->~GenericEulerHdgCudaSession();

  auto* replacement = new (&storage) GenericEulerHdgCudaSession;
  const auto replacement_initialized = replacement->Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  GenericEulerCudaPreparedStep current_authority{};
  shared_export = {};
  const auto current_prepared = replacement->Prepare(current_authority,
                                                      shared_export);
  // Session and export addresses are both reused.  Rejection must therefore
  // come from the retired facade incarnation carried by the authority.
  const bool stale_acknowledged = replacement->AcknowledgePreparedOutput(
      retired_authority, shared_export);
  const auto current_aborted = replacement->Abort(std::move(current_authority),
                                                   shared_export);
  const bool passed = All(replacement_initialized.status ==
                              HdgEulerSessionStatus::initialized &&
                          current_prepared.status == HdgEulerSessionStatus::prepared &&
                          !stale_acknowledged &&
                          current_aborted.status == HdgEulerSessionStatus::aborted);
  replacement->~GenericEulerHdgCudaSession();
  return passed;
}

bool CheckP2Admission(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(functional_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  const auto result = session.Initialize(BorrowedConfig(
      mesh, ranks, functional_polynomial_order,
      functional_elements_per_direction));
  return All(result.status == HdgEulerSessionStatus::initialized &&
             session.coordinates().step == 0 && session.coordinates().dt == 0.0);
}

bool CheckLegacyFacadeAdmission(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(LegacyBorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  GenericEulerCudaPreparedStep authority{};
  GenericPreparedOutput native_export{};
  const auto prepared = initialized.status == HdgEulerSessionStatus::initialized
      ? session.Prepare(authority, native_export)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const auto aborted = prepared.status == HdgEulerSessionStatus::prepared
      ? session.Abort(std::move(authority), native_export)
      : dg::hdg::generic::HdgEulerSessionResult{};
  return All(initialized.status == HdgEulerSessionStatus::initialized &&
             prepared.status == HdgEulerSessionStatus::prepared &&
             aborted.status == HdgEulerSessionStatus::aborted &&
             session.coordinates().step == 0 &&
             session.coordinates().generation == 0);
}

bool CheckBorrowedDataBridge(const int ranks, const int order, const int n)
{
  std::vector<double> candidate;
  {
    mfem::Mesh serial = PeriodicSquareMesh(n);
    mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
    HostFields zero = NonuniformHostFields(mesh, order, false);
    const std::vector<double> initial_before = zero.initial;
    GenericEulerHdgCudaSession session;
    const auto initialized = session.Initialize(
        BorrowedDataConfig(mesh, ranks, order, n, zero));
    // Initialize has copied input/source bytes.  The caller may immediately
    // overwrite and release both buffers without changing the engine state.
    std::fill(zero.initial.begin(), zero.initial.end(), -3.0);
    zero.initial.clear();
    zero.initial.shrink_to_fit();
    std::vector<double> committed(initial_before.size(), -7.0);
    const bool committed_export = session.ExportCommittedInterior(
        {committed.data(), zero.scalar_count, conserved_components});
    std::vector<double> invalid_destination(initial_before.size(), -13.0);
    const bool invalid_export = session.ExportCommittedInterior(
        {invalid_destination.data(), zero.scalar_count - 1,
         conserved_components});
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> divergent_destination(initial_before.size(), -15.0);
    const bool divergent_export = ranks != 2 ||
        session.ExportCommittedInterior(
            {divergent_destination.data(), rank == 0 ? zero.scalar_count :
             zero.scalar_count - 1, conserved_components});
    if (!All(initialized.status == HdgEulerSessionStatus::initialized &&
             committed_export && SameValues(committed, initial_before) &&
             !invalid_export &&
             std::all_of(invalid_destination.begin(),
                         invalid_destination.end(),
                         [](const double value) { return value == -13.0; }) &&
             (ranks != 2 || (!divergent_export && std::all_of(
                 divergent_destination.begin(), divergent_destination.end(),
                 [](const double value) { return value == -15.0; })))))
      return false;

    GenericEulerCudaPreparedStep authority{};
    GenericPreparedOutput native_export{};
    const auto prepared = session.Prepare(authority, native_export);
    candidate.assign(initial_before.size(), -9.0);
    const bool prepared_export = session.ExportPreparedInterior(
        authority, native_export,
        {candidate.data(), zero.scalar_count, conserved_components});
    const bool acknowledged = session.AcknowledgePreparedOutput(authority,
                                                                 native_export);
    const auto committed_step = session.Commit(std::move(authority),
                                                native_export);
    std::vector<double> committed_after(initial_before.size(), -17.0);
    const bool committed_after_export = session.ExportCommittedInterior(
        {committed_after.data(), zero.scalar_count, conserved_components});
    if (!All(prepared.status == HdgEulerSessionStatus::prepared &&
             prepared_export && FullyFiniteOverwritten(candidate, -9.0) &&
             acknowledged &&
             committed_step.status == HdgEulerSessionStatus::committed &&
             committed_after_export && SameValues(committed_after, candidate)))
      return false;
  }

  // The persistent CUDA runtime owns process-global allocation and transport
  // ledgers.  Retain only exported bytes across the source comparison; do not
  // overlap two live numerical engines.

  // A static source has its own persistent source slice: with identical ICs,
  // the prepared physical candidate must not collapse to the zero-source one.
  mfem::Mesh source_serial = PeriodicSquareMesh(n);
  mfem::ParMesh source_mesh(MPI_COMM_WORLD, source_serial);
  HostFields forced = NonuniformHostFields(source_mesh, order, true);
  GenericEulerHdgCudaSession forced_session;
  const auto forced_initialized = forced_session.Initialize(
      BorrowedDataConfig(source_mesh, ranks, order, n, forced));
  std::fill(forced.initial.begin(), forced.initial.end(), -5.0);
  std::fill(forced.source.begin(), forced.source.end(), -6.0);
  forced.initial.clear();
  forced.source.clear();
  forced.initial.shrink_to_fit();
  forced.source.shrink_to_fit();
  GenericEulerCudaPreparedStep forced_authority{};
  GenericPreparedOutput forced_export{};
  const auto forced_prepared = forced_session.Prepare(forced_authority,
                                                       forced_export);
  std::vector<double> forced_candidate(candidate.size(), -9.0);
  const bool forced_output = forced_session.ExportPreparedInterior(
      forced_authority, forced_export,
      {forced_candidate.data(), forced.scalar_count, conserved_components});
  const auto forced_aborted = forced_session.Abort(std::move(forced_authority),
                                                    forced_export);
  return All(forced_initialized.status == HdgEulerSessionStatus::initialized &&
             forced_prepared.status == HdgEulerSessionStatus::prepared &&
             forced_output && FullyFiniteOverwritten(forced_candidate, -9.0) &&
             forced_aborted.status ==
                 HdgEulerSessionStatus::aborted &&
             !SameValues(candidate, forced_candidate));
}

bool CheckBorrowedLaggedLlfHighWave(const int ranks, const int order,
                                    const int n)
{
  mfem::Mesh serial = PeriodicSquareMesh(n);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  HostFields fields = HighWaveHostFields(mesh, order);
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(
      BorrowedDataConfig(mesh, ranks, order, n, fields));
  if (!All(initialized.status == HdgEulerSessionStatus::initialized))
    return false;

  std::vector<double> prepared_one(fields.initial.size(), -11.0);
  std::vector<double> prepared_two(fields.initial.size(), -12.0);
  GenericEulerCudaPreparedStep first{};
  GenericPreparedOutput first_export{};
  const auto first_prepared = session.Prepare(first, first_export);
  const bool first_bytes = session.ExportPreparedInterior(
      first, first_export, {prepared_one.data(), fields.scalar_count,
                            conserved_components});
  const bool first_ack = session.AcknowledgePreparedOutput(first, first_export);
  const auto first_committed = session.Commit(std::move(first), first_export);
  std::vector<double> committed_one(fields.initial.size(), -13.0);
  const bool first_committed_bytes = session.ExportCommittedInterior(
      {committed_one.data(), fields.scalar_count, conserved_components});

  GenericEulerCudaPreparedStep second{};
  GenericPreparedOutput second_export{};
  const auto second_prepared = session.Prepare(second, second_export);
  const bool second_bytes = session.ExportPreparedInterior(
      second, second_export, {prepared_two.data(), fields.scalar_count,
                               conserved_components});
  const bool second_ack = session.AcknowledgePreparedOutput(second, second_export);
  const auto second_committed = session.Commit(std::move(second), second_export);
  std::vector<double> committed_two(fields.initial.size(), -14.0);
  const bool second_committed_bytes = session.ExportCommittedInterior(
      {committed_two.data(), fields.scalar_count, conserved_components});

  return All(first_prepared.status == HdgEulerSessionStatus::prepared &&
             first_bytes && FullyFiniteOverwritten(prepared_one, -11.0) &&
             first_ack && first_committed.status == HdgEulerSessionStatus::committed &&
             first_committed_bytes && SameValues(committed_one, prepared_one) &&
             second_prepared.status == HdgEulerSessionStatus::prepared &&
             second_bytes && FullyFiniteOverwritten(prepared_two, -12.0) &&
             second_ack && second_committed.status == HdgEulerSessionStatus::committed &&
             second_committed_bytes && SameValues(committed_two, prepared_two) &&
             session.coordinates().step == 2 &&
             session.coordinates().generation == 2 &&
             session.coordinates().time == 2.0 * fixed_dt);
}

bool Rejected(GenericEulerHdgCudaSession& session,
              const CudaConfig& config) noexcept
{
  const auto result = session.Initialize(config);
  return result.status == HdgEulerSessionStatus::rejected &&
      result.coordinates.step == 0 && result.coordinates.generation == 0;
}

bool CheckNegatives(const int ranks)
{
  mfem::Mesh periodic_serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh periodic(MPI_COMM_WORLD, periodic_serial);
  mfem::Mesh boundary_serial = NonperiodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh boundary(MPI_COMM_WORLD, boundary_serial);

  CudaConfig nonperiodic = BorrowedConfig(
      boundary, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  CudaConfig wrong_shape = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  wrong_shape.mesh_n = 4;
  CudaConfig p3 = BorrowedConfig(
      periodic, ranks, 3, smoke_elements_per_direction);
  CudaConfig wrong_device = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  wrong_device.requested_device = 1;
  CudaConfig wrong_ranks = BorrowedConfig(
      periodic, 3, smoke_polynomial_order, smoke_elements_per_direction);
  CudaConfig wrong_dt = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  wrong_dt.dt = 0.01;
  CudaConfig capture_oracle = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  capture_oracle.capture_oracle = true;
  CudaConfig injection = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  injection.injection.enabled = true;

  HostFields fields = NonuniformHostFields(periodic, smoke_polynomial_order,
                                           true);
  const std::vector<double> fields_initial_before = fields.initial;
  const std::vector<double> fields_source_before = fields.source;
  CudaConfig null_initial = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  null_initial.borrowed_initial.coefficients = nullptr;
  CudaConfig wrong_initial_count = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  --wrong_initial_count.borrowed_initial.scalar_count;
  CudaConfig wrong_initial_components = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  wrong_initial_components.borrowed_initial.components = 3;
  CudaConfig null_source = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  null_source.borrowed_source.coefficients = nullptr;
  HostFields initial_nan = fields;
  initial_nan.initial[0] = std::numeric_limits<double>::quiet_NaN();
  CudaConfig nonfinite_initial = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      initial_nan);
  HostFields source_infinite = fields;
  source_infinite.source[0] = std::numeric_limits<double>::infinity();
  CudaConfig nonfinite_source = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      source_infinite);
  HostFields nonphysical = fields;
  // Finite coefficients with negative pressure must fail pre-device borrowed
  // input admission; this is distinct from the existing NaN/Inf checks.
  nonphysical.initial[3 * nonphysical.scalar_count] = 0.01;
  CudaConfig nonphysical_initial = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      nonphysical);

  GenericEulerHdgCudaSession one, two, three, four, five, six, seven, eight;
  const bool nonperiodic_rejected = Rejected(one, nonperiodic);
  const bool wrong_shape_rejected = Rejected(two, wrong_shape);
  const bool p3_rejected = Rejected(three, p3);
  const bool wrong_device_rejected = Rejected(four, wrong_device);
  const bool wrong_ranks_rejected = Rejected(five, wrong_ranks);
  const bool wrong_dt_rejected = Rejected(six, wrong_dt);
  const bool capture_oracle_rejected = Rejected(seven, capture_oracle);
  const bool injection_rejected = Rejected(eight, injection);
  GenericEulerHdgCudaSession nine, ten, eleven, twelve, thirteen, fourteen,
      fifteen;
  const bool null_initial_rejected = Rejected(nine, null_initial);
  const bool wrong_initial_count_rejected = Rejected(ten, wrong_initial_count);
  const bool wrong_initial_components_rejected = Rejected(
      eleven, wrong_initial_components);
  const bool null_source_rejected = Rejected(twelve, null_source);
  const bool nonfinite_initial_rejected = Rejected(thirteen, nonfinite_initial);
  const bool nonfinite_source_rejected = Rejected(fourteen, nonfinite_source);
  const bool nonphysical_initial_rejected = Rejected(fifteen,
                                                      nonphysical_initial);
  bool rejected = nonperiodic_rejected && wrong_shape_rejected && p3_rejected &&
      wrong_device_rejected && wrong_ranks_rejected && wrong_dt_rejected &&
      capture_oracle_rejected && injection_rejected && null_initial_rejected &&
      wrong_initial_count_rejected && wrong_initial_components_rejected &&
      null_source_rejected && nonfinite_initial_rejected &&
      nonfinite_source_rejected && nonphysical_initial_rejected &&
      SameValues(fields.initial, fields_initial_before) &&
      SameValues(fields.source, fields_source_before);

  if (ranks == 2)
  {
    mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
    std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()), 0);
    mfem::ParMesh empty_local(MPI_COMM_WORLD, serial, partition.data());
    CudaConfig empty = BorrowedConfig(
        empty_local, ranks, smoke_polynomial_order,
        smoke_elements_per_direction);
    GenericEulerHdgCudaSession sixteen;
    const bool empty_rejected = Rejected(sixteen, empty);

    CudaConfig divergent = BorrowedConfig(
        periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) divergent.dt = 0.01;
    GenericEulerHdgCudaSession seventeen;
    const bool divergent_rejected = Rejected(seventeen, divergent);
    rejected = rejected && empty_rejected && divergent_rejected;
  }
  return All(rejected);
}
}  // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool known_mode = mode.empty() || mode == "--p2" || mode == "--negatives";
  bool pass = false;
  if (mode == "--negatives")
  {
    const bool device_unconfigured_before = !mfem::Device::IsConfigured();
    const bool negatives = CheckNegatives(ranks);
    const bool device_unconfigured_after = !mfem::Device::IsConfigured();
    pass = known_mode && device_unconfigured_before && negatives &&
        device_unconfigured_after;
  }
  else if (known_mode)
  {
    mfem::Device device("cuda");
    (void)device;
    if (mode == "--p2")
    {
      pass = CheckP2Admission(ranks) && CheckBorrowedDataBridge(
          ranks, functional_polynomial_order, functional_elements_per_direction) &&
          CheckBorrowedLaggedLlfHighWave(
              ranks, functional_polynomial_order,
              functional_elements_per_direction);
    }
    else
    {
      const bool transaction = CheckTransaction(ranks);
      const bool foreign_authority = CheckForeignDestinationAuthority(ranks);
      const bool address_reuse = CheckAddressReuseAba(ranks);
      const bool session_identity = CheckRankTwoSessionIdentityAdmission(ranks);
      const bool rank_two = CheckRankTwoCollectiveAdmission(ranks);
      const bool legacy_facade = CheckLegacyFacadeAdmission(ranks);
      const bool host_data = CheckBorrowedDataBridge(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      const bool high_wave = CheckBorrowedLaggedLlfHighWave(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      pass = transaction && foreign_authority && address_reuse &&
          session_identity && rank_two && legacy_facade && host_data && high_wave;
    }
  }
  if (!pass && ranks > 0)
    std::cerr << "CUDA generic Euler session audit failed\n";
  return pass ? 0 : 1;
}
