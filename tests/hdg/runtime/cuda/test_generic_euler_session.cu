#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>

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
                          const int order, const int n) noexcept
{
  CudaConfig config{};
  config.order = order;
  config.mesh_n = n;
  config.requested_device = 0;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = &mesh;
  config.borrowed_communicator = MPI_COMM_WORLD;
  return config;
}

bool SameCoordinates(const dg::hdg::generic::HdgEulerSessionCoordinates& left,
                     const StateCoordinates& right) noexcept
{
  return left.time == right.time && left.dt == right.dt &&
      left.step == right.step && left.generation == right.generation;
}

bool SameNativeExport(const CudaSerializedCheckpointV1& left,
                      const CudaSerializedCheckpointV1& right) noexcept
{
  const auto same_attempt = [](const AttemptIdentity& first,
                               const AttemptIdentity& second)
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
      left.interior == right.interior &&
      left.owned_true_trace == right.owned_true_trace &&
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
      HdgEulerSessionCapabilityBit(HdgEulerSessionCapability::prepared_output);
  if (!All(initialized.status == HdgEulerSessionStatus::initialized &&
           session.capabilities() == expected_capabilities &&
           (session.capabilities() & HdgEulerSessionCapabilityBit(
               HdgEulerSessionCapability::committed_cfl_diagnostics)) == 0 &&
           (session.capabilities() & HdgEulerSessionCapabilityBit(
               HdgEulerSessionCapability::generic_borrowed_inputs)) == 0))
    return false;

  GenericEulerCudaPreparedStep prepared{};
  CudaSerializedCheckpointV1 native_export{};
  const auto staged = session.Prepare(prepared, native_export);
  const auto before_commit = session.coordinates();
  const bool native_staged = staged.status == HdgEulerSessionStatus::prepared &&
      native_export.candidate_frozen && native_export.checkpoint_staged &&
      !native_export.output_staged && !native_export.successful_commit;
  const CudaSerializedCheckpointV1 before_duplicate = native_export;
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
  CudaSerializedCheckpointV1 aborted_export{};
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
  CudaSerializedCheckpointV1 retry_export{};
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
  CudaSerializedCheckpointV1 native_export{};
  const auto staged = session.Prepare(authority, native_export);
  const bool acknowledged = session.AcknowledgePreparedOutput(authority,
                                                               native_export);
  if (staged.status != HdgEulerSessionStatus::prepared || !acknowledged)
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
  CudaSerializedCheckpointV1 retry_export{};
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
  CudaSerializedCheckpointV1 owner_export{};
  const auto owner_prepared = owner.Prepare(owner_authority, owner_export);
  const CudaSerializedCheckpointV1 before_rejection = owner_export;
  CudaSerializedCheckpointV1 recipient_export{};
  const CudaSerializedCheckpointV1 recipient_before_rejection = recipient_export;
  const auto rejected = recipient.Prepare(owner_authority, recipient_export);
  const bool preserved = SameNativeExport(owner_export, before_rejection) &&
      SameNativeExport(recipient_export, recipient_before_rejection);
  const auto aborted = owner.Abort(std::move(owner_authority), owner_export);

  GenericEulerCudaPreparedStep retry_authority{};
  CudaSerializedCheckpointV1 retry_export{};
  const auto retry_prepared = owner.Prepare(retry_authority, retry_export);
  const CudaSerializedCheckpointV1 retry_before_rejection = retry_export;
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
  CudaSerializedCheckpointV1 retired_export{};
  const auto retired_prepared = retired->Prepare(retired_authority, retired_export);
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
  CudaSerializedCheckpointV1 current_export{};
  const auto current_prepared = replacement->Prepare(current_authority,
                                                      current_export);
  const bool stale_acknowledged = replacement->AcknowledgePreparedOutput(
      retired_authority, retired_export);
  const auto current_aborted = replacement->Abort(std::move(current_authority),
                                                   current_export);
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

  GenericEulerHdgCudaSession one, two, three, four, five, six, seven, eight;
  const bool nonperiodic_rejected = Rejected(one, nonperiodic);
  const bool wrong_shape_rejected = Rejected(two, wrong_shape);
  const bool p3_rejected = Rejected(three, p3);
  const bool wrong_device_rejected = Rejected(four, wrong_device);
  const bool wrong_ranks_rejected = Rejected(five, wrong_ranks);
  const bool wrong_dt_rejected = Rejected(six, wrong_dt);
  const bool capture_oracle_rejected = Rejected(seven, capture_oracle);
  const bool injection_rejected = Rejected(eight, injection);
  bool rejected = nonperiodic_rejected && wrong_shape_rejected && p3_rejected &&
      wrong_device_rejected && wrong_ranks_rejected && wrong_dt_rejected &&
      capture_oracle_rejected && injection_rejected;

  if (ranks == 2)
  {
    mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
    std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()), 0);
    mfem::ParMesh empty_local(MPI_COMM_WORLD, serial, partition.data());
    CudaConfig empty = BorrowedConfig(
        empty_local, ranks, smoke_polynomial_order,
        smoke_elements_per_direction);
    GenericEulerHdgCudaSession nine;
    const bool empty_rejected = Rejected(nine, empty);

    CudaConfig divergent = BorrowedConfig(
        periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) divergent.dt = 0.01;
    GenericEulerHdgCudaSession ten;
    const bool divergent_rejected = Rejected(ten, divergent);
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
      pass = CheckP2Admission(ranks);
    }
    else
    {
      const bool transaction = CheckTransaction(ranks);
      const bool foreign_authority = CheckForeignDestinationAuthority(ranks);
      const bool address_reuse = CheckAddressReuseAba(ranks);
      const bool rank_two = CheckRankTwoCollectiveAdmission(ranks);
      pass = transaction && foreign_authority && address_reuse && rank_two;
    }
  }
  if (!pass && ranks > 0)
    std::cerr << "CUDA generic Euler session audit failed\n";
  return pass ? 0 : 1;
}
