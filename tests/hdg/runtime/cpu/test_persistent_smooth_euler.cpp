#include "independent_oracle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cpu;
namespace cpu_detail = dg::hdg::runtime::cpu::detail;

struct TestEulerModelTag
{
  static constexpr std::string_view id = "euler.compressible.2d";
};
struct TestConventionalFormulationTag
{
  static constexpr std::string_view id = "hdg.conventional";
};
struct TestDgInterfaceTraceTag
{
  static constexpr std::string_view id = "mfem.dg_interface.trace";
};
struct TestFixedTauTag
{
  static constexpr std::string_view id = "tau.fixed";
};
struct TestBackwardEulerTag
{
  static constexpr std::string_view id = "backward_euler";
};
struct TestSerialBackendTag
{
  static constexpr std::string_view id = "cpu.serial";
};
struct TestRank2BackendTag
{
  static constexpr std::string_view id = "cpu.rank2";
};
struct TestFp64Tag
{
  static constexpr std::string_view id = "fp64";
};

template <typename Backend>
struct TestSelection : StaticRunSelection<TestEulerModelTag,
                                          TestConventionalFormulationTag,
                                          TestDgInterfaceTraceTag,
                                          TestFixedTauTag,
                                          TestBackwardEulerTag,
                                          Backend,
                                          TestFp64Tag>
{
  static constexpr std::string_view experiment_id =
      "exp.hdg.euler.conventional.smooth_persistent.w8a.v1";
};

using SerialSelection = TestSelection<TestSerialBackendTag>;
using Rank2Selection = TestSelection<TestRank2BackendTag>;

static_assert(!std::is_polymorphic<SerialSelection>::value, "static selection");
static_assert(!std::is_polymorphic<Rank2Selection>::value, "static selection");
static_assert(cpu_polynomial_order == 1 &&
                  cpu_elements_per_direction == 3 &&
                  cpu_assembly_quadrature_order == 11 &&
                  cpu_backward_euler_dt == 0.05,
              "accepted smoke aliases remain exact");
static_assert(cpu_functional_polynomial_order == 2 &&
                  cpu_functional_minimum_elements_per_direction == 3 &&
                  cpu_functional_principal_elements_per_direction == 6 &&
                  cpu_functional_maximum_elements_per_direction == 12 &&
                  cpu_functional_assembly_quadrature_order == 14 &&
                  cpu_functional_accepted_steps == 20 &&
                  cpu_functional_checkpoint_step == 10 &&
                  cpu_functional_retry_target_step == 14,
              "functional CPU tuple constants remain card exact");
static_assert(!std::is_copy_constructible<CpuPreparedStep>::value &&
                  std::is_move_constructible<CpuPreparedStep>::value &&
                  !std::is_copy_constructible<CpuPreparedCommit>::value &&
                  !std::is_copy_constructible<CpuProvisionalCommit>::value &&
                  !std::is_copy_constructible<
                      CpuCandidateConsumeCapability>::value,
              "CPU authorities are affine values, never pointer aliases");
static_assert(CpuStateStoragePolicy::publish_is_complete_slot_exchange &&
                  CpuStateStoragePolicy::
                      candidate_retains_exact_old_state_after_publish &&
                  CpuStateStoragePolicy::revert_publish_is_exact_byte_inverse &&
                  CpuStateStoragePolicy::
                      publish_and_revert_preserve_disjoint_backing,
              "two-phase reversible storage attestations are exact");

struct Checks
{
  int rank = -1;
  int failures = 0;

  void Require(const bool condition, const char* message)
  {
    if (condition) { return; }
    ++failures;
    std::cerr << "rank " << rank << ": " << message << '\n';
  }
};

constexpr double independent_true_residual_limit = 1.0e-10;

std::uint64_t IndependentRankOrderedStateIdentity(
    const CpuStateSnapshot& snapshot, const bool perturb_rank_one,
    const MPI_Comm communicator)
{
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &ranks);
  auto mix = [](std::uint64_t& hash, const void* data,
                const std::size_t bytes)
  {
    const auto* source = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < bytes; ++index)
    {
      hash ^= source[index];
      hash *= 1099511628211ULL;
    }
  };
  std::uint64_t local = 1469598103934665603ULL;
  mix(local, &snapshot.interior.size, sizeof(snapshot.interior.size));
  mix(local, snapshot.interior.data,
      snapshot.interior.size * sizeof(double));
  mix(local, &snapshot.trace.size, sizeof(snapshot.trace.size));
  mix(local, snapshot.trace.data, snapshot.trace.size * sizeof(double));
  if (perturb_rank_one && rank == 1) local ^= 0x9e3779b97f4a7c15ULL;
  std::array<std::uint64_t, 2> ordered{};
  MPI_Allgather(&local, 1, MPI_UINT64_T, ordered.data(), 1, MPI_UINT64_T,
                communicator);
  std::uint64_t global = 1469598103934665603ULL;
  mix(global, ordered.data(), static_cast<std::size_t>(ranks) *
                                  sizeof(std::uint64_t));
  return global;
}

std::vector<double> CopyView(const CpuCoefficientView view)
{
  return view.data ? std::vector<double>(view.data, view.data + view.size)
                   : std::vector<double>{};
}

bool ByteEqual(const CpuCoefficientView view,
               const std::vector<double>& values)
{
  return view.size == values.size() &&
      (values.empty() ||
       std::memcmp(view.data, values.data(),
                   values.size() * sizeof(double)) == 0);
}

bool CollectivelyTrue(bool local, MPI_Comm communicator);

template <typename Selection>
void CheckPhysicalInteriorIds(PersistentSmoothEulerCpuAdapter<Selection>& adapter,
                             Checks& checks, MPI_Comm communicator)
{
  const CpuStateSnapshot snapshot = adapter.Snapshot();
  const bool local_valid = snapshot.physical_interior_ids != nullptr &&
      snapshot.physical_interior_id_count == snapshot.interior.size &&
      snapshot.scalar_interior_dofs > 0;
  const bool globally_valid = CollectivelyTrue(local_valid, communicator);
  checks.Require(globally_valid,
                 "physical parent-cell map is complete on every rank");
  if (!globally_valid) return;
  std::vector<std::uint64_t> local;
  bool allocated = true;
  try
  {
    local.assign(snapshot.physical_interior_ids,
                 snapshot.physical_interior_ids +
                     snapshot.physical_interior_id_count);
  }
  catch (...) { allocated = false; }
  if (!CollectivelyTrue(allocated, communicator))
  {
    checks.Require(false, "physical parent-cell map allocation succeeds");
    return;
  }
  std::array<int, 2> counts{};
  std::array<int, 2> displacements{};
  const int local_count = static_cast<int>(local.size());
  const bool counts_ready = MPI_Gather(
      &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
      communicator) == MPI_SUCCESS;
  if (!CollectivelyTrue(counts_ready, communicator))
  {
    checks.Require(false, "physical parent-cell map counts gather");
    return;
  }
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &ranks);
  int total = 0;
  if (rank == 0)
    for (int peer = 0; peer < ranks; ++peer)
    {
      displacements[peer] = total;
      total += counts[peer];
    }
  std::vector<std::uint64_t> all;
  allocated = true;
  try { all.resize(static_cast<std::size_t>(total)); }
  catch (...) { allocated = false; }
  if (!CollectivelyTrue(allocated, communicator))
  {
    checks.Require(false, "physical parent-cell map aggregate allocation");
    return;
  }
  const bool gathered = MPI_Gatherv(
      local.data(), local_count, MPI_UINT64_T, all.data(), counts.data(),
      displacements.data(), MPI_UINT64_T, 0, communicator) == MPI_SUCCESS;
  if (!CollectivelyTrue(gathered, communicator))
  {
    checks.Require(false, "physical parent-cell map gather");
    return;
  }
  int complete_unique = 1;
  if (rank == 0)
  {
    std::sort(all.begin(), all.end());
    complete_unique = all.size() == 144 ? 1 : 0;
    for (std::uint64_t id = 0; id < all.size() && complete_unique == 1; ++id)
      complete_unique = all[static_cast<std::size_t>(id)] == id ? 1 : 0;
  }
  MPI_Bcast(&complete_unique, 1, MPI_INT, 0, communicator);
  checks.Require(complete_unique == 1,
                 "physical parent-cell/DOF VTK map has unique full coverage");
}

struct CpuCheckpointObservation
{
  std::array<unsigned char, sizeof(CheckpointDescriptor)> descriptor_bytes{};
  std::vector<double> interior;
  std::vector<double> trace;
  const double* interior_data = nullptr;
  const double* trace_data = nullptr;
  std::size_t interior_capacity = 0;
  std::size_t trace_capacity = 0;
  bool valid = false;
};

CpuCheckpointObservation ObserveCheckpoint(const CpuCheckpoint& checkpoint)
{
  CpuCheckpointObservation observation;
  std::memcpy(observation.descriptor_bytes.data(), &checkpoint.descriptor,
              sizeof(checkpoint.descriptor));
  const CpuCoefficientView interior = checkpoint.interior();
  const CpuCoefficientView trace = checkpoint.trace();
  observation.interior = CopyView(interior);
  observation.trace = CopyView(trace);
  observation.interior_data = interior.data;
  observation.trace_data = trace.data;
  observation.interior_capacity = interior.capacity;
  observation.trace_capacity = trace.capacity;
  observation.valid = checkpoint.valid;
  return observation;
}

bool CheckpointUnchanged(const CpuCheckpoint& checkpoint,
                         const CpuCheckpointObservation& before)
{
  const CpuCoefficientView interior = checkpoint.interior();
  const CpuCoefficientView trace = checkpoint.trace();
  return checkpoint.valid == before.valid &&
      std::memcmp(&checkpoint.descriptor, before.descriptor_bytes.data(),
                  sizeof(checkpoint.descriptor)) == 0 &&
      interior.data == before.interior_data &&
      trace.data == before.trace_data &&
      interior.capacity == before.interior_capacity &&
      trace.capacity == before.trace_capacity &&
      ByteEqual(interior, before.interior) && ByteEqual(trace, before.trace);
}

bool CollectivelyFalse(const bool local, MPI_Comm communicator)
{
  int value = local ? 1 : 0;
  int minimum = 1;
  int maximum = 1;
  const int minimum_status = MPI_Allreduce(
      &value, &minimum, 1, MPI_INT, MPI_MIN, communicator);
  const int maximum_status = MPI_Allreduce(
      &value, &maximum, 1, MPI_INT, MPI_MAX, communicator);
  return minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS &&
      minimum == 0 && maximum == 0;
}

bool CollectivelyTrue(const bool local, MPI_Comm communicator)
{
  return CollectivelyFalse(!local, communicator);
}

void SeedInvalidCheckpoint(CpuCheckpoint& checkpoint,
                           const CheckpointDescriptor& descriptor)
{
  checkpoint.descriptor = descriptor;
  checkpoint.descriptor.cursor ^= std::uint64_t{0x5a5a};
  checkpoint.valid = false;
}

bool NearTime(const double left, const double right)
{
  return std::abs(left - right) <=
      16.0 * std::numeric_limits<double>::epsilon();
}

bool CloseVectors(const CpuCoefficientView production,
                  const std::vector<double>& oracle,
                  const double relative_tolerance,
                  const double absolute_tolerance)
{
  if (production.size != oracle.size() ||
      (!production.data && production.size != 0))
  {
    return false;
  }
  double difference_squared = 0.0;
  double production_squared = 0.0;
  double oracle_squared = 0.0;
  double difference_infinity = 0.0;
  double scale_infinity = 1.0;
  for (std::size_t index = 0; index < oracle.size(); ++index)
  {
    const double difference = production.data[index] - oracle[index];
    difference_squared += difference * difference;
    production_squared += production.data[index] * production.data[index];
    oracle_squared += oracle[index] * oracle[index];
    difference_infinity = std::max(difference_infinity, std::abs(difference));
    scale_infinity = std::max(
        scale_infinity,
        std::max(std::abs(production.data[index]), std::abs(oracle[index])));
  }
  const double relative = std::sqrt(difference_squared) /
      std::max({std::sqrt(production_squared), std::sqrt(oracle_squared),
                1.0e-300});
  return relative <= relative_tolerance &&
      difference_infinity <= absolute_tolerance * scale_infinity;
}

template <typename Selection>
void CheckCanonicalIds(PersistentSmoothEulerCpuAdapter<Selection>& adapter,
                       Checks& checks, MPI_Comm communicator)
{
  const CpuStateSnapshot snapshot = adapter.Snapshot();
  std::vector<CpuCanonicalEntry> local(
      snapshot.interior.size + snapshot.trace.size);
  std::size_t written = 0;
  checks.Require(adapter.ExportCanonicalCommitted(
                     local.data(), local.size(), written) &&
                     written == local.size(),
                 "canonical export must fill exact local capacity");
  const int local_bytes = static_cast<int>(
      written * sizeof(CpuCanonicalEntry));
  int ranks = 0;
  int rank = 0;
  MPI_Comm_size(communicator, &ranks);
  MPI_Comm_rank(communicator, &rank);
  std::vector<int> byte_counts(static_cast<std::size_t>(ranks));
  MPI_Gather(&local_bytes, 1, MPI_INT, byte_counts.data(), 1, MPI_INT, 0,
             communicator);
  std::vector<int> displacements(static_cast<std::size_t>(ranks), 0);
  int total_bytes = 0;
  if (rank == 0)
  {
    for (int peer = 0; peer < ranks; ++peer)
    {
      displacements[static_cast<std::size_t>(peer)] = total_bytes;
      total_bytes += byte_counts[static_cast<std::size_t>(peer)];
    }
  }
  std::vector<CpuCanonicalEntry> gathered(
      static_cast<std::size_t>(total_bytes) / sizeof(CpuCanonicalEntry));
  MPI_Gatherv(local.data(), local_bytes, MPI_BYTE, gathered.data(),
              byte_counts.data(), displacements.data(), MPI_BYTE, 0,
              communicator);
  int global_valid = 1;
  if (rank == 0)
  {
    std::sort(gathered.begin(), gathered.end(),
              [](const auto& left, const auto& right)
              { return left.global_id < right.global_id; });
    constexpr std::uint64_t trace_prefix = std::uint64_t{1} << 63;
    global_valid = gathered.size() == 288 ? 1 : 0;
    for (std::uint64_t index = 0; index < 144 && global_valid == 1; ++index)
    {
      global_valid = gathered[static_cast<std::size_t>(index)].global_id ==
                         index &&
              gathered[static_cast<std::size_t>(144 + index)].global_id ==
                  (trace_prefix | index)
          ? 1 : 0;
    }
  }
  MPI_Bcast(&global_valid, 1, MPI_INT, 0, communicator);
  checks.Require(global_valid == 1,
                 "serial/rank2 canonical global-ID set must be identical");
}

template <typename Selection>
void CheckProductNegatives(Checks& checks)
{
  for (const CpuProductFault fault :
       {CpuProductFault::missing_solution,
        CpuProductFault::duplicate_residual,
        CpuProductFault::out_of_order_matrix})
  {
    PersistentSmoothEulerCpuAdapter<Selection> adapter;
    checks.Require(adapter.Initialize(), "product-negative setup");
    const CpuStateSnapshot entry = adapter.Snapshot();
    const std::vector<double> entry_interior = CopyView(entry.interior);
    const std::vector<double> entry_trace = CopyView(entry.trace);
    CpuFailureInjection injection;
    injection.product_fault = fault;
    const AttemptOutcome outcome = adapter.Advance(injection);
    checks.Require(!outcome.committed && outcome.retry_available,
                   "product protocol negative must roll back");
    const CpuStateSnapshot exit = adapter.Snapshot();
    checks.Require(ByteEqual(exit.interior, entry_interior) &&
                       ByteEqual(exit.trace, entry_trace) &&
                       exit.interior_hash == entry.interior_hash &&
                       exit.trace_hash == entry.trace_hash &&
                       exit.coordinates.step == 0,
                   "product negative must preserve committed state");
  }
}

bool ExactRank1FaultConsumption(
    MPI_Comm communicator, const int rank,
    const cpu_detail::CpuCollectiveTestFault expected,
    const std::uint64_t expected_owner,
    const std::uint64_t expected_rejected_owner,
    const bool expect_initialization_reset,
    const cpu_detail::CpuCollectiveTestConsumptionEvidence& evidence)
{
  const unsigned long long local_count =
      static_cast<unsigned long long>(evidence.local_consumption_count);
  unsigned long long global_count = 0;
  const int count_status = MPI_Allreduce(
      &local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
      communicator);
  const std::uint64_t expected_rejections =
      expected_rejected_owner == 0 ? 0 : 1;
  const bool local_exact = evidence.arm_generation != 0 &&
      evidence.owner_incarnation == expected_owner && expected_owner != 0 &&
      evidence.target_rank == 1 && !evidence.armed &&
      evidence.thread_confined &&
      evidence.local_wrong_owner_rejections == expected_rejections &&
      evidence.last_rejected_incarnation == expected_rejected_owner &&
      evidence.local_reset_completions ==
          (expect_initialization_reset ? 1U : 0U) &&
      evidence.reset_collectively_attested == expect_initialization_reset &&
      (rank == 1
           ? evidence.local_consumption_count == 1 &&
                 evidence.consumed_incarnation == expected_owner &&
                 evidence.consumed_rank == 1 &&
                 evidence.consumed_fault == expected
           : evidence.local_consumption_count == 0 &&
                 evidence.consumed_incarnation == 0 &&
                 evidence.consumed_rank == -1 &&
                 evidence.consumed_fault ==
                     cpu_detail::CpuCollectiveTestFault::none);
  int local_valid = local_exact ? 1 : 0;
  int all_valid = 0;
  const int validity_status = MPI_Allreduce(
      &local_valid, &all_valid, 1, MPI_INT, MPI_MIN, communicator);
  return count_status == MPI_SUCCESS && validity_status == MPI_SUCCESS &&
      global_count == 1 && all_valid == 1;
}

bool SameFaultObservation(
    const cpu_detail::CpuCollectiveTestConsumptionEvidence& left,
    const cpu_detail::CpuCollectiveTestConsumptionEvidence& right) noexcept
{
  return left.arm_generation == right.arm_generation &&
      left.local_consumption_count == right.local_consumption_count &&
      left.owner_incarnation == right.owner_incarnation &&
      left.consumed_incarnation == right.consumed_incarnation &&
      left.last_rejected_incarnation == right.last_rejected_incarnation &&
      left.local_wrong_owner_rejections ==
          right.local_wrong_owner_rejections &&
      left.local_reset_completions == right.local_reset_completions &&
      left.consumed_fault == right.consumed_fault &&
      left.consumed_rank == right.consumed_rank &&
      left.target_rank == right.target_rank && left.armed == right.armed &&
      left.reset_collectively_attested ==
          right.reset_collectively_attested &&
      left.thread_confined == right.thread_confined;
}

bool ExactWrongOwnerNonConsumption(
    MPI_Comm communicator, const std::uint64_t intended_owner,
    const std::uint64_t rejected_owner,
    const cpu_detail::CpuCollectiveTestConsumptionEvidence& evidence)
{
  const bool local_exact = intended_owner != 0 && rejected_owner != 0 &&
      intended_owner != rejected_owner && evidence.arm_generation != 0 &&
      evidence.owner_incarnation == intended_owner && evidence.armed &&
      evidence.local_consumption_count == 0 &&
      evidence.consumed_incarnation == 0 &&
      evidence.last_rejected_incarnation == rejected_owner &&
      evidence.local_wrong_owner_rejections == 1 &&
      evidence.local_reset_completions == 0 &&
      evidence.consumed_fault ==
          cpu_detail::CpuCollectiveTestFault::none &&
      evidence.consumed_rank == -1 && evidence.target_rank == 1 &&
      !evidence.reset_collectively_attested && evidence.thread_confined;
  int local_valid = local_exact ? 1 : 0;
  int all_valid = 0;
  const int status = MPI_Allreduce(
      &local_valid, &all_valid, 1, MPI_INT, MPI_MIN, communicator);
  return status == MPI_SUCCESS && all_valid == 1;
}

template <typename Selection>
void CheckAsymmetricCollectiveInitializationFaults(
    Checks& checks, MPI_Comm communicator, const int rank, const int ranks)
{
  if (ranks != 2) { return; }
  PersistentSmoothEulerCpuAdapter<Selection> truth;
  checks.Require(truth.Initialize(),
                 "uninterrupted initialization truth setup");
  const CpuStateSnapshot expected = truth.Snapshot();
  const std::vector<double> expected_interior = CopyView(expected.interior);
  const std::vector<double> expected_trace = CopyView(expected.trace);

  const std::array<cpu_detail::CpuCollectiveTestFault, 3> faults{{
      cpu_detail::CpuCollectiveTestFault::
          initialization_before_sparse_count_exchange,
      cpu_detail::CpuCollectiveTestFault::
          initialization_between_sparse_exchanges,
      cpu_detail::CpuCollectiveTestFault::initialization_before_gmres_prime}};
  for (const cpu_detail::CpuCollectiveTestFault fault : faults)
  {
    cpu_detail::ResetCpuCollectiveTestFault();
    PersistentSmoothEulerCpuAdapter<Selection> adapter;
    const std::uint64_t owner =
        cpu_detail::CpuCollectiveTestAccess::CoreIncarnation(adapter);
    cpu_detail::ArmCpuCollectiveTestFault(fault, 1, adapter);
    const bool initialized = adapter.Initialize();
    checks.Require(CollectivelyFalse(initialized, communicator),
                   "rank1 initialization fault fails collectively");
    const auto consumed = cpu_detail::ObserveCpuCollectiveTestFault();
    checks.Require(ExactRank1FaultConsumption(
                       communicator, rank, fault, owner, 0, true, consumed),
                   "initialization fault has one exact rank1 consumption");

    const bool reentered = adapter.Initialize();
    const auto after_reentry = cpu_detail::ObserveCpuCollectiveTestFault();
    const CpuStateSnapshot actual = adapter.Snapshot();
    const bool reentry_consumption_exact = ExactRank1FaultConsumption(
        communicator, rank, fault, owner, 0, true, after_reentry);
    checks.Require(reentered && SameFaultObservation(consumed, after_reentry) &&
                       reentry_consumption_exact,
                   "initialization seam is automatic one-shot before cleanup");
    checks.Require(ByteEqual(actual.interior, expected_interior) &&
                       ByteEqual(actual.trace, expected_trace) &&
                       actual.interior_hash == expected.interior_hash &&
                       actual.trace_hash == expected.trace_hash &&
                       actual.state_hash == expected.state_hash &&
                       actual.coordinates.step == 0 &&
                       NearTime(actual.coordinates.time, 0.0),
                   "reentered initialization matches uninterrupted truth");
    cpu_detail::ResetCpuCollectiveTestFault();
  }
}

template <typename Selection>
void CheckCollectiveFacadeInitialization(
    Checks& checks, MPI_Comm communicator, const int rank, const int ranks)
{
  if (ranks != 2) { return; }

  PersistentSmoothEulerCpuAdapter<Selection> truth;
  checks.Require(truth.Initialize(), "collective facade truth setup");
  const CpuStateSnapshot expected = truth.Snapshot();
  const std::vector<double> expected_interior = CopyView(expected.interior);
  const std::vector<double> expected_trace = CopyView(expected.trace);

  PersistentSmoothEulerCpuAdapter<Selection> preconditioned;
  CpuPersistentConfig asymmetric = MakeCpuSmokeConfig();
  if (rank == 1) { --asymmetric.maximum_newton_updates; }
  const bool admitted = preconditioned.Initialize(asymmetric);
  const CpuStateSnapshot rejected = preconditioned.Snapshot();
  checks.Require(CollectivelyFalse(admitted, communicator) &&
                     preconditioned.status() == RunStatus::invalid_request &&
                     rejected.interior.size == 0 && rejected.trace.size == 0 &&
                     rejected.state_hash == StateHash{},
                 "asymmetric facade precondition is rejected before core entry");
  const bool precondition_retry = preconditioned.Initialize();
  const CpuStateSnapshot precondition_state = preconditioned.Snapshot();
  checks.Require(CollectivelyTrue(precondition_retry, communicator) &&
                     ByteEqual(precondition_state.interior,
                               expected_interior) &&
                     ByteEqual(precondition_state.trace, expected_trace) &&
                     precondition_state.state_hash == expected.state_hash,
                 "precondition refusal preserves empty same-adapter retry");
  const bool existing_state_reentry = preconditioned.Initialize();
  const CpuStateSnapshot after_existing = preconditioned.Snapshot();
  checks.Require(CollectivelyFalse(existing_state_reentry, communicator) &&
                     ByteEqual(after_existing.interior, expected_interior) &&
                     ByteEqual(after_existing.trace, expected_trace) &&
                     after_existing.state_hash == expected.state_hash,
                 "existing facade state is collectively refused and retained");

  cpu_detail::ResetCpuCollectiveTestFault();
  PersistentSmoothEulerCpuAdapter<Selection> intended;
  PersistentSmoothEulerCpuAdapter<Selection> unrelated;
  const std::uint64_t intended_owner =
      cpu_detail::CpuCollectiveTestAccess::CoreIncarnation(intended);
  const std::uint64_t unrelated_owner =
      cpu_detail::CpuCollectiveTestAccess::CoreIncarnation(unrelated);
  cpu_detail::ArmCpuCollectiveTestFault(
      cpu_detail::CpuCollectiveTestFault::initialization_facade_readiness,
      1, intended);

  const bool unrelated_ready = unrelated.Initialize();
  const auto rejected_owner = cpu_detail::ObserveCpuCollectiveTestFault();
  const bool unrelated_common = CollectivelyTrue(
      unrelated_ready, communicator);
  const bool wrong_owner_exact = ExactWrongOwnerNonConsumption(
      communicator, intended_owner, unrelated_owner, rejected_owner);
  checks.Require(unrelated_common && wrong_owner_exact,
                 "different same-thread adapter cannot consume intended arm");

  const bool intended_ready = intended.Initialize();
  const auto consumed = cpu_detail::ObserveCpuCollectiveTestFault();
  const CpuStateSnapshot failed_snapshot = intended.Snapshot();
  const bool intended_consumption_exact = ExactRank1FaultConsumption(
      communicator, rank,
      cpu_detail::CpuCollectiveTestFault::initialization_facade_readiness,
      intended_owner, unrelated_owner, true, consumed);
  checks.Require(CollectivelyFalse(intended_ready, communicator) &&
                     intended.status() == RunStatus::invalid_request &&
                     intended.contract_telemetry() == nullptr &&
                     failed_snapshot.interior.size == 0 &&
                     failed_snapshot.trace.size == 0 &&
                     failed_snapshot.state_hash == StateHash{} &&
                     intended_consumption_exact,
                 "rank1 post-core facade fault resets without publication");

  const bool retried = intended.Initialize();
  const auto after_retry = cpu_detail::ObserveCpuCollectiveTestFault();
  const CpuStateSnapshot actual = intended.Snapshot();
  const bool retry_consumption_exact = ExactRank1FaultConsumption(
      communicator, rank,
      cpu_detail::CpuCollectiveTestFault::initialization_facade_readiness,
      intended_owner, unrelated_owner, true, after_retry);
  checks.Require(CollectivelyTrue(retried, communicator) &&
                     SameFaultObservation(consumed, after_retry) &&
                     retry_consumption_exact &&
                     ByteEqual(actual.interior, expected_interior) &&
                     ByteEqual(actual.trace, expected_trace) &&
                     actual.state_hash == expected.state_hash,
                 "collectively attested reset permits same-adapter retry");
  cpu_detail::ResetCpuCollectiveTestFault();

  PersistentSmoothEulerCpuAdapter<Selection> poisoned;
  const std::uint64_t poisoned_owner =
      cpu_detail::CpuCollectiveTestAccess::CoreIncarnation(poisoned);
  cpu_detail::ArmCpuCollectiveTestFault(
      cpu_detail::CpuCollectiveTestFault::
          initialization_facade_readiness_reset_refusal,
      1, poisoned);
  const bool poisoned_initialization = poisoned.Initialize();
  const auto refused_reset = cpu_detail::ObserveCpuCollectiveTestFault();
  const bool refused_reset_exact = ExactRank1FaultConsumption(
      communicator, rank,
      cpu_detail::CpuCollectiveTestFault::
          initialization_facade_readiness_reset_refusal,
      poisoned_owner, 0, false, refused_reset);
  const bool poisoned_retry = poisoned.Initialize();
  const auto after_poisoned_retry =
      cpu_detail::ObserveCpuCollectiveTestFault();
  const CpuStateSnapshot poisoned_snapshot = poisoned.Snapshot();
  const bool poisoned_initialization_common = CollectivelyFalse(
      poisoned_initialization, communicator);
  const bool poisoned_retry_common = CollectivelyFalse(
      poisoned_retry, communicator);
  checks.Require(poisoned_initialization_common && poisoned_retry_common &&
                     refused_reset_exact &&
                     SameFaultObservation(refused_reset,
                                          after_poisoned_retry) &&
                     poisoned.status() == RunStatus::invalid_request &&
                     poisoned.contract_telemetry() == nullptr &&
                     poisoned_snapshot.interior.size == 0 &&
                     poisoned_snapshot.trace.size == 0 &&
                     poisoned_snapshot.state_hash == StateHash{},
                 "unattested reset terminally poisons same-adapter reentry");
  cpu_detail::ResetCpuCollectiveTestFault();
}

template <typename Selection>
void CheckAsymmetricCollectiveNumericsFaults(
    Checks& checks, MPI_Comm communicator, const int rank, const int ranks)
{
  if (ranks != 2) { return; }
  PersistentSmoothEulerCpuAdapter<Selection> truth;
  const bool truth_initialized = truth.Initialize();
  const AttemptOutcome truth_advance = truth.Advance();
  checks.Require(truth_initialized && truth_advance.committed,
                 "uninterrupted numerical truth setup");
  const CpuStateSnapshot expected = truth.Snapshot();
  const std::vector<double> expected_interior = CopyView(expected.interior);
  const std::vector<double> expected_trace = CopyView(expected.trace);
  struct Case
  {
    cpu_detail::CpuCollectiveTestFault fault;
    CpuFailurePhase phase;
  };
  const std::array<Case, 4> cases{{
      {cpu_detail::CpuCollectiveTestFault::trace_load_preparation,
       CpuFailurePhase::setup},
      {cpu_detail::CpuCollectiveTestFault::assembly_completion,
       CpuFailurePhase::assembly},
      {cpu_detail::CpuCollectiveTestFault::residual_map_preparation,
       CpuFailurePhase::assembly},
      {cpu_detail::CpuCollectiveTestFault::condensation_outgoing_preparation,
       CpuFailurePhase::condensation}}};

  for (const Case& test_case : cases)
  {
    cpu_detail::ResetCpuCollectiveTestFault();
    PersistentSmoothEulerCpuAdapter<Selection> adapter;
    checks.Require(adapter.Initialize(),
                   "asymmetric collective numerical seam setup");
    const CpuStateSnapshot entry = adapter.Snapshot();
    const std::vector<double> interior = CopyView(entry.interior);
    const std::vector<double> trace = CopyView(entry.trace);

    const std::uint64_t owner =
        cpu_detail::CpuCollectiveTestAccess::CoreIncarnation(adapter);
    cpu_detail::ArmCpuCollectiveTestFault(test_case.fault, 1, adapter);
    CpuPreparedStep prepared;
    const bool prepared_call = adapter.PrepareAdvance(prepared);
    const bool commonly_not_ready = CollectivelyFalse(
        prepared.numerically_ready, communicator);
    checks.Require(prepared_call && commonly_not_ready &&
                       prepared.failure_phase == test_case.phase,
                   "one-rank numerical preparation fault fails commonly");
    const auto consumed = cpu_detail::ObserveCpuCollectiveTestFault();
    checks.Require(ExactRank1FaultConsumption(
                       communicator, rank, test_case.fault, owner, 0, false,
                       consumed),
                   "numerical fault has one exact rank1 consumption");
    const AttemptOutcome failure = adapter.FinalizePrepared(prepared);
    const CpuStateSnapshot rolled_back = adapter.Snapshot();
    checks.Require(CollectivelyFalse(failure.committed, communicator) &&
                       failure.retry_available &&
                       ByteEqual(rolled_back.interior, interior) &&
                       ByteEqual(rolled_back.trace, trace) &&
                       rolled_back.interior_hash == entry.interior_hash &&
                       rolled_back.trace_hash == entry.trace_hash &&
                       rolled_back.state_hash == entry.state_hash &&
                       rolled_back.coordinates.step == 0 &&
                       rolled_back.coordinates.generation == 0 &&
                       rolled_back.coordinates.cursor == 0 &&
                       NearTime(rolled_back.coordinates.time, 0.0),
                   "asymmetric numerical fault rolls back without publication");

    CpuPreparedStep retry;
    checks.Require(adapter.PrepareRetry(retry) && retry.numerically_ready,
                   "asymmetric numerical fault permits collective retry");
    const AttemptOutcome retried = adapter.FinalizePrepared(retry);
    const auto after_retry = cpu_detail::ObserveCpuCollectiveTestFault();
    const CpuStateSnapshot actual = adapter.Snapshot();
    const bool retry_consumption_exact = ExactRank1FaultConsumption(
        communicator, rank, test_case.fault, owner, 0, false, after_retry);
    checks.Require(retried.committed && adapter.coordinates().step == 1 &&
                       NearTime(adapter.coordinates().time, 0.05) &&
                       SameFaultObservation(consumed, after_retry) &&
                       retry_consumption_exact,
                   "automatic one-shot numerical seam retries before cleanup");
    checks.Require(ByteEqual(actual.interior, expected_interior) &&
                       ByteEqual(actual.trace, expected_trace) &&
                       actual.interior_hash == expected.interior_hash &&
                       actual.trace_hash == expected.trace_hash &&
                       actual.state_hash == expected.state_hash &&
                       actual.coordinates.step == 1 &&
                       NearTime(actual.coordinates.time, 0.05),
                   "numerical retry matches uninterrupted step-1 truth");
    cpu_detail::ResetCpuCollectiveTestFault();
  }
}

template <typename Selection>
void CheckPhaseRankInjection(Checks& checks, const int ranks)
{
  const std::array<CpuFailurePhase, 6> phases{{
      CpuFailurePhase::assembly,
      CpuFailurePhase::condensation,
      CpuFailurePhase::trace_solve,
      CpuFailurePhase::recovery,
      CpuFailurePhase::after_recovery_before_commit,
      CpuFailurePhase::diagnostics}};
  for (const CpuFailurePhase phase : phases)
    for (int source_rank = 0; source_rank < ranks; ++source_rank)
    {
      PersistentSmoothEulerCpuAdapter<Selection> adapter;
      checks.Require(adapter.Initialize(), "phase-injection setup");
      checks.Require(adapter.Advance().committed,
                     "phase-injection prefix step 1");
      const CpuStateSnapshot entry = adapter.Snapshot();
      const std::vector<double> interior = CopyView(entry.interior);
      const std::vector<double> trace = CopyView(entry.trace);
      CpuFailureInjection injection;
      injection.phase = phase;
      injection.target_step = 2;
      injection.source_rank = source_rank;
      CpuPreparedStep prepared;
      checks.Require(adapter.PrepareAdvance(prepared, injection),
                     "injected phase must enter prepared rollback seam");
      checks.Require(!prepared.numerically_ready &&
                         prepared.failure_phase == phase,
                     "injected phase/source must be retained");
      const AttemptOutcome failure = adapter.FinalizePrepared(prepared);
      checks.Require(!failure.committed && failure.retry_available,
                     "phase injection must collectively roll back");
      const CpuStateSnapshot exit = adapter.Snapshot();
      checks.Require(ByteEqual(exit.interior, interior) &&
                         ByteEqual(exit.trace, trace) &&
                         exit.interior_hash == entry.interior_hash &&
                         exit.trace_hash == entry.trace_hash &&
                         exit.coordinates.step == 1 &&
                     adapter.evidence().last_attempt.failure_source_rank ==
                             source_rank &&
                         adapter.evidence().last_attempt.failure_phase == phase &&
                         (adapter.evidence().last_attempt.
                                  candidate_generation == 0 ||
                          adapter.evidence().last_attempt.
                              candidate_generation_invalidated),
                     "phase/rank rollback must preserve committed bytes and IDs");
    }
}

template <typename Selection>
void CheckNaturalFailurePrecedesLaterInjection(Checks& checks,
                                               const int ranks)
{
  PersistentSmoothEulerCpuAdapter<Selection> adapter;
  checks.Require(adapter.Initialize(), "first-fired failure setup");
  cpu_detail::ArmCpuCollectiveTestFault(
      cpu_detail::CpuCollectiveTestFault::assembly_completion,
      ranks - 1, adapter);
  CpuFailureInjection later_injection;
  later_injection.phase = CpuFailurePhase::recovery;
  later_injection.target_step = 1;
  later_injection.source_rank = ranks - 1;
  CpuPreparedStep prepared;
  checks.Require(adapter.PrepareAdvance(prepared, later_injection) &&
                     !prepared.numerically_ready,
                 "natural failure precedes configured later phase");
  const AttemptOutcome outcome = adapter.FinalizePrepared(prepared);
  const CpuAttemptEvidence& evidence = adapter.evidence().last_attempt;
  checks.Require(!outcome.committed && outcome.retry_available &&
                     evidence.status == CpuAttemptStatus::runtime_failure &&
                     evidence.failure_phase == CpuFailurePhase::assembly &&
                     evidence.failure_source_rank == -1,
                 "first actual failure owns classification and provenance");
  CpuPreparedStep retry;
  checks.Require(adapter.PrepareRetry(retry) &&
                     adapter.FinalizePrepared(retry).committed,
                 "natural failure retains one retry");
  cpu_detail::ResetCpuCollectiveTestFault();
}

template <typename Selection>
void CheckOuterStageAbort(Checks& checks, const int ranks)
{
  const std::array<CpuFailurePhase, 3> phases{{
      CpuFailurePhase::checkpoint_staging,
      CpuFailurePhase::output_staging,
      CpuFailurePhase::commit}};
  for (const CpuFailurePhase phase : phases)
  {
    PersistentSmoothEulerCpuAdapter<Selection> adapter;
    checks.Require(adapter.Initialize(), "outer-stage abort setup");
    const CpuStateSnapshot entry = adapter.Snapshot();
    const std::vector<double> interior = CopyView(entry.interior);
    const std::vector<double> trace = CopyView(entry.trace);
    CpuPreparedStep prepared;
    checks.Require(adapter.PrepareAdvance(prepared) &&
                       prepared.numerically_ready &&
                       prepared.entry_coordinates.step == 0 &&
                       NearTime(prepared.target_coordinates.time, 0.05) &&
                       prepared.target_coordinates.step == 1,
                   "outer stage must receive a qualified numerical snapshot");
    const AttemptOutcome failure = adapter.AbortPrepared(
        prepared, phase, ranks == 1 ? 0 : 1);
    const CpuStateSnapshot exit = adapter.Snapshot();
    checks.Require(!failure.committed && failure.retry_available &&
                       ByteEqual(exit.interior, interior) &&
                       ByteEqual(exit.trace, trace) &&
                       exit.coordinates.step == 0 &&
                       adapter.evidence().last_attempt.failure_phase == phase &&
                       adapter.evidence().last_attempt.status ==
                           CpuAttemptStatus::injected_failure,
                   "prepublication outer-stage abort must roll back naturally");
    checks.Require(adapter.PrepareRetry(prepared),
                   "staging abort exposes exactly one retry");
    checks.Require(adapter.FinalizePrepared(prepared).committed,
                   "staging abort retry must commit the original step 1");
  }
}

template <typename Selection>
void CheckLiveOutputAuthority(Checks& checks)
{
  PersistentSmoothEulerCpuAdapter<Selection> output_owner;
  checks.Require(output_owner.Initialize(), "live-output rejection setup");
  CpuPreparedStep live_output;
  checks.Require(output_owner.PrepareAdvance(live_output) &&
                     !output_owner.PrepareAdvance(live_output) &&
                     live_output.valid &&
                     output_owner.AbortPrepared(
                         live_output, CpuFailurePhase::output_staging, 0)
                         .retry_available,
                 "PrepareAdvance rejects and preserves a live output handle");
}

template <typename Selection>
void CheckForeignLiveOutputAuthorities(Checks& checks, const int ranks)
{
  PersistentSmoothEulerCpuAdapter<Selection> advance_owner;
  PersistentSmoothEulerCpuAdapter<Selection> foreign_advance_owner;
  CpuPreparedStep foreign_advance;
  CpuPreparedStep advance;
  checks.Require(advance_owner.Initialize() &&
                     foreign_advance_owner.Initialize() &&
                     foreign_advance_owner.PrepareAdvance(foreign_advance) &&
                     !advance_owner.PrepareAdvance(foreign_advance) &&
                     foreign_advance.valid &&
                     advance_owner.PrepareAdvance(advance),
                 "PrepareAdvance rejects a foreign live output before begin");
  checks.Require(
      foreign_advance_owner.AbortPrepared(
          foreign_advance, CpuFailurePhase::output_staging, ranks - 1)
              .retry_available &&
          advance_owner.AbortPrepared(
              advance, CpuFailurePhase::output_staging, ranks - 1)
              .retry_available,
      "PrepareAdvance rejection preserves both original authorities");

  PersistentSmoothEulerCpuAdapter<Selection> retry_owner;
  PersistentSmoothEulerCpuAdapter<Selection> foreign_retry_owner;
  CpuPreparedStep failed;
  CpuPreparedStep foreign_retry;
  checks.Require(retry_owner.Initialize() &&
                     retry_owner.PrepareAdvance(failed) &&
                     retry_owner.AbortPrepared(
                         failed, CpuFailurePhase::output_staging, ranks - 1)
                         .retry_available &&
                     foreign_retry_owner.Initialize() &&
                     foreign_retry_owner.PrepareAdvance(foreign_retry) &&
                     !retry_owner.PrepareRetry(foreign_retry) &&
                     foreign_retry.valid,
                 "PrepareRetry rejects a foreign live output before retry");
  CpuPreparedStep retry;
  checks.Require(
      retry_owner.PrepareRetry(retry) &&
          retry_owner.FinalizePrepared(retry).committed &&
          foreign_retry_owner.AbortPrepared(
              foreign_retry, CpuFailurePhase::output_staging, ranks - 1)
              .retry_available,
      "PrepareRetry rejection preserves both original authorities");
}

template <typename Selection>
void CheckFinalizePreparedCommitRejection(Checks& checks, const int ranks)
{
  PersistentSmoothEulerCpuAdapter<Selection> adapter;
  checks.Require(adapter.Initialize(), "FinalizePrepared rejection setup");
  cpu_detail::ArmCpuCollectiveTestFault(
      cpu_detail::CpuCollectiveTestFault::commit_prepared_rejection,
      ranks - 1, adapter);
  CpuPreparedStep staging;
  checks.Require(adapter.PrepareAdvance(staging) && staging.numerically_ready,
                 "FinalizePrepared rejection freezes candidate");
  const AttemptOutcome rejected = adapter.FinalizePrepared(staging);
  checks.Require(!rejected.committed && rejected.retry_available &&
                     rejected.reason == FailureReason::backend_failure &&
                     adapter.last_transition_outcome().reason ==
                         FailureReason::backend_failure &&
                     adapter.evidence().last_attempt.failure_phase ==
                         CpuFailurePhase::commit,
                 "FinalizePrepared returns recorded CommitPrepared outcome");
  CpuPreparedStep retry;
  checks.Require(adapter.PrepareRetry(retry) &&
                     adapter.FinalizePrepared(retry).committed,
                 "CommitPrepared rejection retains one retry");
  cpu_detail::ResetCpuCollectiveTestFault();
}

template <typename Selection>
void CheckLiveMoveDestination(Checks& checks)
{
  PersistentSmoothEulerCpuAdapter<Selection> move_left_owner;
  PersistentSmoothEulerCpuAdapter<Selection> move_right_owner;
  CpuPreparedStep move_left;
  CpuPreparedStep move_right;
  checks.Require(move_left_owner.Initialize() && move_right_owner.Initialize() &&
                     move_left_owner.PrepareAdvance(move_left) &&
                     move_right_owner.PrepareAdvance(move_right),
                 "live move-destination setup");
  move_left = std::move(move_right);
  checks.Require(move_left.valid && move_right.valid &&
                     move_left_owner.AbortPrepared(
                         move_left, CpuFailurePhase::output_staging, 0)
                         .retry_available &&
                     move_right_owner.AbortPrepared(
                         move_right, CpuFailurePhase::output_staging, 0)
                         .retry_available,
                 "move assignment rejects a live destination without consumption");
}

template <typename Selection>
void CheckLiveCommitDestinations(Checks& checks)
{
  PersistentSmoothEulerCpuAdapter<Selection> source_owner;
  CpuPreparedStep source_staging;
  checks.Require(source_owner.Initialize() &&
                     source_owner.PrepareAdvance(source_staging),
                 "live commit source setup");
  {
    PersistentSmoothEulerCpuAdapter<Selection> destination_owner;
    CpuPreparedStep destination_staging;
    CpuPreparedCommit live_prepared_destination;
    checks.Require(destination_owner.Initialize() &&
                       destination_owner.PrepareAdvance(destination_staging) &&
                       destination_owner.PrepareCommit(
                           destination_staging, live_prepared_destination) &&
                       !source_owner.PrepareCommit(
                           source_staging, live_prepared_destination) &&
                       live_prepared_destination.valid,
                   "PrepareCommit rejects a live destination authority");
    checks.Require(destination_owner.AbortPrepared(
                       live_prepared_destination,
                       CpuFailurePhase::output_staging, 0).retry_available,
                   "rejected prepared destination remains consumable by owner");
  }
  CpuPreparedCommit source_prepared;
  checks.Require(source_owner.PrepareCommit(source_staging, source_prepared),
                 "rejected PrepareCommit preserves source staging");
  {
    PersistentSmoothEulerCpuAdapter<Selection> provisional_owner;
    CpuPreparedStep provisional_staging;
    CpuPreparedCommit provisional_prepared;
    CpuProvisionalCommit live_provisional_destination;
    checks.Require(provisional_owner.Initialize() &&
                       provisional_owner.PrepareAdvance(provisional_staging) &&
                       provisional_owner.PrepareCommit(
                           provisional_staging, provisional_prepared) &&
                       provisional_owner.CommitPrepared(
                           provisional_prepared,
                           live_provisional_destination) &&
                       !source_owner.CommitPrepared(
                           source_prepared, live_provisional_destination) &&
                       source_prepared.valid &&
                       live_provisional_destination.valid,
                   "CommitPrepared rejects a live provisional destination");
    checks.Require(provisional_owner.RevertCommit(
                       live_provisional_destination,
                       CpuFailurePhase::commit, 0).retry_available,
                   "rejected provisional destination remains owner-consumable");
  }
  CpuProvisionalCommit source_provisional;
  checks.Require(source_owner.CommitPrepared(source_prepared,
                                             source_provisional) &&
                     source_owner.RevertCommit(
                         source_provisional, CpuFailurePhase::commit, 0)
                         .retry_available,
                 "rejected CommitPrepared preserves prepared source");
}

template <typename Selection>
void CheckTwoPhaseAndAffineAuthorities(Checks& checks, const int ranks)
{
  PersistentSmoothEulerCpuAdapter<Selection> adapter;
  checks.Require(adapter.Initialize(), "two-phase adapter setup");
  const CpuStateSnapshot old = adapter.Snapshot();
  const std::vector<double> old_interior = CopyView(old.interior);
  const std::vector<double> old_trace = CopyView(old.trace);

  CpuPreparedStep staging;
  checks.Require(adapter.PrepareAdvance(staging) && staging.numerically_ready,
                 "freeze numerical candidate");
  auto consume = adapter.AcquirePreparedCandidate(staging);
  checks.Require(consume.has_value(), "acquire affine candidate capability");
  CpuCandidateConsumeCapability moved_consume;
  if (consume) { moved_consume = std::move(*consume); }
  CpuStateSnapshot ignored;
  checks.Require(consume &&
                     !adapter.ConsumePreparedCandidate(std::move(*consume),
                                                       ignored) &&
                     adapter.ConsumePreparedCandidate(
                         std::move(moved_consume), ignored),
                 "moved candidate authority rejects old value exactly once");

  auto stale_after_revert = adapter.AcquirePreparedCandidate(staging);
  CpuPreparedCommit prepared;
  checks.Require(adapter.PrepareCommit(staging, prepared) && prepared.valid,
                 "FreezeCandidate to PrepareCommit binding");
  CpuProvisionalCommit provisional;
  checks.Require(adapter.CommitPrepared(prepared, provisional) &&
                     provisional.valid,
                 "PrepareCommit to provisional slot exchange");
  const CpuStateSnapshot logical_old = adapter.LogicalPublishedSnapshot();
  CpuCheckpoint premature_checkpoint;
  checks.Require(ByteEqual(logical_old.interior, old_interior) &&
                     ByteEqual(logical_old.trace, old_trace) &&
                     logical_old.coordinates.step == 0 &&
                     !adapter.CaptureCheckpoint(premature_checkpoint),
                 "provisional exchange retains old logical publication");
  const AttemptOutcome reverted = adapter.RevertCommit(
      provisional, CpuFailurePhase::commit, 0);
  const RunTelemetry* reverted_telemetry = adapter.contract_telemetry();
  checks.Require(!reverted.committed && reverted.retry_available &&
                     ByteEqual(adapter.Snapshot().interior, old_interior) &&
                     ByteEqual(adapter.Snapshot().trace, old_trace) &&
                     adapter.RevertCommit(provisional,
                                          CpuFailurePhase::commit, 0).reason ==
                         FailureReason::stale_attempt &&
                     stale_after_revert &&
                     !adapter.ConsumePreparedCandidate(
                         std::move(*stale_after_revert), ignored) &&
                     reverted_telemetry && reverted_telemetry->rollbacks == 1 &&
                     reverted_telemetry->commits == 0 &&
                     reverted_telemetry->attempted_transfers.complete &&
                     reverted_telemetry->rolled_back_transfers.complete &&
                     reverted_telemetry->attempted_transfers.allocation_events ==
                         0 &&
                     reverted_telemetry->rolled_back_transfers
                             .allocation_events == 0,
                 "provisional failure exactly reverts and invalidates lineage");
  CpuPreparedStep retry;
  checks.Require(adapter.PrepareRetry(retry) &&
                     adapter.FinalizePrepared(retry).committed,
                 "reverted step retries and finalizes once");

  PersistentSmoothEulerCpuAdapter<Selection> finalized;
  checks.Require(finalized.Initialize(), "finalize visibility setup");
  const std::vector<double> finalized_old =
      CopyView(finalized.Snapshot().interior);
  CpuPreparedStep finalize_staging;
  CpuPreparedCommit finalize_prepared;
  CpuProvisionalCommit finalize_provisional;
  checks.Require(finalized.PrepareAdvance(finalize_staging) &&
                     finalized.PrepareCommit(finalize_staging,
                                             finalize_prepared) &&
                     finalized.CommitPrepared(finalize_prepared,
                                               finalize_provisional) &&
                     ByteEqual(finalized.LogicalPublishedSnapshot().interior,
                               finalized_old),
                 "old publication remains logical through provisional phase");
  CpuProvisionalCommit moved_provisional{std::move(finalize_provisional)};
  checks.Require(finalized.FinalizeCommit(finalize_provisional).reason ==
                         FailureReason::stale_attempt &&
                     finalized.FinalizeCommit(moved_provisional).committed &&
                     finalized.coordinates().step == 1 &&
                     !ByteEqual(finalized.LogicalPublishedSnapshot().interior,
                                finalized_old),
                 "FinalizeCommit alone advances logical publication and move authority");

  PersistentSmoothEulerCpuAdapter<Selection> first;
  PersistentSmoothEulerCpuAdapter<Selection> second;
  checks.Require(first.Initialize() && second.Initialize(),
                 "cross-owner adapter setup");
  CpuPreparedStep first_staging;
  CpuPreparedStep second_staging;
  checks.Require(first.PrepareAdvance(first_staging) &&
                     second.PrepareAdvance(second_staging),
                 "cross-owner candidates frozen");
  const CheckpointDescriptor moved_checkpoint =
      first_staging.future_checkpoint;
  const StateHash moved_state_hash = first_staging.state_hash;
  CpuPreparedStep moved_staging{std::move(first_staging)};
  CpuPreparedCommit rejected;
  checks.Require(!first_staging.valid && moved_staging.valid &&
                     moved_staging.future_checkpoint.coordinate_fingerprint ==
                         moved_checkpoint.coordinate_fingerprint &&
                     moved_staging.state_hash == moved_state_hash &&
                     !first.PrepareCommit(first_staging, rejected) &&
                     !second.PrepareCommit(moved_staging, rejected),
                 "moved-from and cross-owner staging authorities fail closed");
  CpuPreparedCommit first_prepared;
  checks.Require(first.PrepareCommit(moved_staging, first_prepared),
                 "foreign rejection does not consume live owner token");
  const AttemptOutcome first_abort = first.AbortPrepared(
      first_prepared, CpuFailurePhase::output_staging, 0);
  checks.Require(first_abort.retry_available &&
                     first.AbortPrepared(first_prepared,
                                         CpuFailurePhase::output_staging,
                                         0).reason ==
                         FailureReason::stale_attempt,
                 "prepared authority is single-use and stale after abort");
  const AttemptOutcome second_abort = second.AbortPrepared(
      second_staging, CpuFailurePhase::checkpoint_staging, 0);
  checks.Require(second_abort.retry_available,
                 "independent frozen owner remains abortable");
  CheckLiveOutputAuthority<Selection>(checks);
  CheckForeignLiveOutputAuthorities<Selection>(checks, ranks);
  CheckFinalizePreparedCommitRejection<Selection>(checks, ranks);
  CheckLiveMoveDestination<Selection>(checks);
  CheckLiveCommitDestinations<Selection>(checks);
}

template <typename Selection>
void CheckAsymmetricCheckpointImportFailures(
    const CpuCheckpoint& checkpoint,
    const std::vector<double>& interior,
    const std::vector<double>& trace,
    Checks& checks,
    MPI_Comm communicator,
    const int rank,
    const int ranks)
{
  if (ranks != 2) { return; }
  checks.Require(!interior.empty() && !trace.empty(),
                 "rank2 asymmetric import payloads are nonempty");

  auto require_collective_failure =
      [&](const CheckpointDescriptor& descriptor,
          const double* interior_data,
          const std::size_t interior_size,
          const double* trace_data,
          const std::size_t trace_size,
          CpuCheckpoint& output,
          const char* message)
      {
        const CpuCheckpointObservation before = ObserveCheckpoint(output);
        const bool imported = ImportCpuCheckpoint(
            descriptor, interior_data, interior_size, trace_data, trace_size,
            output);
        checks.Require(CollectivelyFalse(imported, communicator) &&
                           CheckpointUnchanged(output, before),
                       message);
      };

  CpuCheckpoint preexisting;
  checks.Require(ImportCpuCheckpoint(
                     checkpoint.descriptor, interior.data(), interior.size(),
                     trace.data(), trace.size(), preexisting),
                 "rank2 asymmetric import pre-valid seed");
  if (rank == 1) { preexisting = CpuCheckpoint{}; }
  require_collective_failure(
      checkpoint.descriptor, interior.data(), interior.size(), trace.data(),
      trace.size(), preexisting,
      "one-rank pre-valid output fails collectively without mutation");

  CpuCheckpoint output;
  SeedInvalidCheckpoint(output, checkpoint.descriptor);
  CheckpointDescriptor unsuccessful = checkpoint.descriptor;
  if (rank == 1) { unsuccessful.successful_commit = false; }
  require_collective_failure(
      unsuccessful, interior.data(), interior.size(), trace.data(),
      trace.size(), output,
      "one-rank unsuccessful descriptor fails admission collectively");

  output = CpuCheckpoint{};
  SeedInvalidCheckpoint(output, checkpoint.descriptor);
  CheckpointDescriptor mismatched = checkpoint.descriptor;
  if (rank == 1) { mismatched.coordinate_fingerprint ^= std::uint64_t{1}; }
  require_collective_failure(
      mismatched, interior.data(), interior.size(), trace.data(), trace.size(),
      output, "one-rank descriptor identity mismatch fails collectively");

  output = CpuCheckpoint{};
  SeedInvalidCheckpoint(output, checkpoint.descriptor);
  const double* asymmetric_interior = rank == 1 ? nullptr : interior.data();
  require_collective_failure(
      checkpoint.descriptor, asymmetric_interior, interior.size(), trace.data(),
      trace.size(), output,
      "one-rank null pointer with nonzero size fails admission collectively");

  output = CpuCheckpoint{};
  SeedInvalidCheckpoint(output, checkpoint.descriptor);
  const std::size_t asymmetric_size =
      rank == 1 && !interior.empty() ? interior.size() - 1 : interior.size();
  require_collective_failure(
      checkpoint.descriptor, interior.data(), asymmetric_size, trace.data(),
      trace.size(), output,
      "one-rank serialized size mismatch fails final hash consensus");

  output = CpuCheckpoint{};
  SeedInvalidCheckpoint(output, checkpoint.descriptor);
  std::vector<double> nonfinite_interior = interior;
  if (rank == 1 && !nonfinite_interior.empty())
  {
    nonfinite_interior[0] = std::numeric_limits<double>::quiet_NaN();
  }
  require_collective_failure(
      checkpoint.descriptor, nonfinite_interior.data(),
      nonfinite_interior.size(), trace.data(), trace.size(), output,
      "one-rank NaN fails finite consensus before collective hashes");
}

template <typename Selection>
void CheckCheckpointImportAndIndependence(
    PersistentSmoothEulerCpuAdapter<Selection>& source,
    const CpuCheckpoint& checkpoint,
    Checks& checks,
    MPI_Comm communicator,
    const int rank,
    const int ranks)
{
  const std::vector<double> interior = CopyView(checkpoint.interior());
  const std::vector<double> trace = CopyView(checkpoint.trace());
  unsigned long long local_size =
      static_cast<unsigned long long>(interior.size() + trace.size());
  unsigned long long minimum_size = 0;
  unsigned long long maximum_size = 0;
  MPI_Allreduce(&local_size, &minimum_size, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_size, &maximum_size, 1, MPI_UNSIGNED_LONG_LONG,
                MPI_MAX, MPI_COMM_WORLD);
  checks.Require(ranks == 1 || minimum_size != maximum_size,
                 "rank2 checkpoint exercise uses unequal local payload sizes");

  CheckAsymmetricCheckpointImportFailures<Selection>(
      checkpoint, interior, trace, checks, communicator, rank, ranks);

  CpuCheckpoint imported;
  checks.Require(ImportCpuCheckpoint(
                     checkpoint.descriptor, interior.data(), interior.size(),
                     trace.data(), trace.size(), imported) && imported.valid &&
                     imported.interior().data != checkpoint.interior().data &&
                     imported.trace().data != checkpoint.trace().data,
                 "serialized payload imports into an independent opaque slot");
  PersistentSmoothEulerCpuAdapter<Selection> fresh;
  checks.Require(fresh.Initialize(), "fresh-process import adapter setup");
  const RestoreOutcome restored = fresh.Restore(imported);
  checks.Require(restored.restored && fresh.coordinates().step == 1 &&
                     ByteEqual(fresh.Snapshot().interior, interior) &&
                     ByteEqual(fresh.Snapshot().trace, trace),
                 "unequal rank-local payloads restore under rank-common identity");
  checks.Require(fresh.Advance().committed &&
                     fresh.coordinates().step == 2,
                 "successful re-entry import restores and continues collectively");

  CpuCheckpoint later;
  checks.Require(source.CaptureCheckpoint(later) && later.valid &&
                     later.interior().data != checkpoint.interior().data &&
                     ByteEqual(checkpoint.interior(), interior) &&
                     ByteEqual(checkpoint.trace(), trace),
                 "each checkpoint owns an immutable independent snapshot");
}

struct StoredCpuState
{
  StateCoordinates coordinates{};
  StateHash interior_hash{};
  StateHash trace_hash{};
  std::vector<double> interior;
  std::vector<double> trace;
};

StoredCpuState StoreState(const CpuStateSnapshot& snapshot)
{
  StoredCpuState stored;
  stored.coordinates = snapshot.coordinates;
  stored.interior_hash = snapshot.interior_hash;
  stored.trace_hash = snapshot.trace_hash;
  stored.interior = CopyView(snapshot.interior);
  stored.trace = CopyView(snapshot.trace);
  return stored;
}

bool StorageUnchanged(const CpuStorageEvidence& initial,
                      const CpuStorageEvidence& current) noexcept
{
  return current.committed_storage_token == initial.committed_storage_token &&
      current.candidate_storage_token == initial.candidate_storage_token &&
      current.interior_capacity == initial.interior_capacity &&
      current.trace_capacity == initial.trace_capacity &&
      current.workspace_bytes == initial.workspace_bytes &&
      current.sparse_owned_term_capacity ==
          initial.sparse_owned_term_capacity &&
      current.sparse_outgoing_term_capacity ==
          initial.sparse_outgoing_term_capacity &&
      current.sparse_incoming_slot_capacity ==
          initial.sparse_incoming_slot_capacity &&
      current.sparse_owned_term_token == initial.sparse_owned_term_token &&
      current.sparse_outgoing_term_token ==
          initial.sparse_outgoing_term_token &&
      current.sparse_incoming_slot_token ==
          initial.sparse_incoming_slot_token &&
      current.topology_generation == initial.topology_generation &&
      current.storage_generation == initial.storage_generation &&
      current.allocation_events_after_baseline == 0;
}

bool AcceptedStepPasses(const CpuAcceptedEvidence& accepted,
                        const std::uint64_t expected_step) noexcept
{
  const CpuAttemptEvidence& step = accepted.attempt;
  return accepted.valid && accepted.coordinates.step == expected_step &&
      accepted.coordinates.generation == expected_step &&
      NearTime(accepted.coordinates.time,
               static_cast<double>(expected_step) * cpu_backward_euler_dt) &&
      step.identity.sequence != 0 && step.transfers.allocation_events == 0 &&
      step.nonlinear_updates > 0 && step.linear_iterations >= 0 &&
      step.work.complete_assemblies ==
          1 + step.work.trial_reassemblies &&
      step.work.complete_condensations ==
          step.work.accepted_newton_updates + 1 &&
      step.work.complete_factor_setups == step.work.complete_trace_solves &&
      step.work.complete_trace_solves ==
          step.work.accepted_newton_updates + 1 &&
      step.work.complete_recoveries ==
          step.work.accepted_newton_updates + 1 &&
      step.transfers.host_to_device_bytes == 0 &&
      step.transfers.device_to_host_bytes == 0 &&
      step.transfers.mpi_staging_bytes == 0 &&
      step.transfers.allocation_events == 0 && step.transfers.complete &&
      step.transfers.synchronization_events == 0 &&
      std::isfinite(step.initial_nonlinear_residual) &&
      std::isfinite(step.final_nonlinear_residual) &&
      step.final_nonlinear_residual <= cpu_nonlinear_tolerance *
          (1.0 + step.initial_nonlinear_residual) &&
      std::isfinite(step.maximum_true_relative_residual) &&
      step.maximum_true_relative_residual <= cpu_nonlinear_tolerance &&
      std::isfinite(step.minimum_density) && step.minimum_density >= 1.0e-8 &&
      std::isfinite(step.minimum_pressure) &&
      step.minimum_pressure >= 1.0e-8 &&
      std::isfinite(step.maximum_wave_speed) &&
      step.maximum_wave_speed < cpu_fixed_tau && step.finite_gate &&
      step.admissibility_gate && step.nonlinear_gate &&
      step.true_residual_gate;
}

struct CommittedScientificEvaluation
{
  double state_l2_error = std::numeric_limits<double>::infinity();
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double minimum_density = -std::numeric_limits<double>::infinity();
  double minimum_pressure = -std::numeric_limits<double>::infinity();
  bool discrete_oracle_match = false;
  bool coordinate_oracle_match = false;
  bool independent_true_residual = false;
  bool passed = false;
};

void ManufacturedTestState(const double x, const double y,
                           std::array<double, 4>& state) noexcept
{
  constexpr double pi_value = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi_value * x);
  const double sy = std::sin(2.0 * pi_value * y);
  const double cx = std::cos(2.0 * pi_value * x);
  const double cy = std::cos(2.0 * pi_value * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double pressure = 1.0 + 0.1 * cx * cy;
  state = {{rho, rho * u, rho * v,
            pressure / 0.4 + 0.5 * rho * (u * u + v * v)}};
}

CommittedScientificEvaluation EvaluateCommittedScientificState(
    const CpuStateSnapshot& snapshot,
    const int elements_per_direction,
    const dg_igr_hdg_cpu_test::IndependentOracleResult& oracle,
    const MPI_Comm communicator)
{
  CommittedScientificEvaluation evaluation;
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &ranks);
  bool local_oracle = CloseVectors(snapshot.interior, oracle.interior,
                                   1.0e-8, 2.0e-8) &&
      CloseVectors(snapshot.trace, oracle.true_trace, 1.0e-8, 2.0e-8);
  int oracle_match = local_oracle ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &oracle_match, 1, MPI_INT, MPI_MIN,
                communicator);
  evaluation.discrete_oracle_match = oracle_match == 1;
  evaluation.coordinate_oracle_match =
      oracle.accepted_step == snapshot.coordinates.step &&
      NearTime(oracle.accepted_time, snapshot.coordinates.time);
  evaluation.independent_true_residual = oracle.passed &&
      std::isfinite(oracle.maximum_true_residual) &&
      oracle.maximum_true_residual <= independent_true_residual_limit;

  static_assert(std::is_trivially_copyable<CpuCanonicalEntry>::value,
                "scientific state records require MPI-byte transport");
  const bool exported = snapshot.interior.size <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()) /
              sizeof(CpuCanonicalEntry) &&
      snapshot.interior.data != nullptr &&
      snapshot.physical_interior_ids != nullptr &&
      snapshot.physical_interior_id_count == snapshot.interior.size;
  int export_ok = exported ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &export_ok, 1, MPI_INT, MPI_MIN, communicator);
  if (export_ok != 1) return evaluation;
  std::vector<CpuCanonicalEntry> local;
  int allocation_ok = 1;
  try
  {
    local.resize(snapshot.interior.size);
    for (std::size_t index = 0; index < snapshot.interior.size; ++index)
      local[index] = {snapshot.physical_interior_ids[index],
                      snapshot.interior.data[index]};
  }
  catch (...) { allocation_ok = 0; }
  MPI_Allreduce(MPI_IN_PLACE, &allocation_ok, 1, MPI_INT, MPI_MIN,
                communicator);
  if (allocation_ok != 1) return evaluation;
  const int local_count = static_cast<int>(
      local.size() * sizeof(CpuCanonicalEntry));
  std::array<int, 2> counts{};
  MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
             communicator);
  std::array<int, 2> displacements{};
  int total_bytes = 0;
  int census_ok = 1;
  if (rank == 0)
  {
    for (int source = 0; source < ranks; ++source)
    {
      displacements[static_cast<std::size_t>(source)] = total_bytes;
      const int count = counts[static_cast<std::size_t>(source)];
      if (count < 0 || count % static_cast<int>(sizeof(CpuCanonicalEntry)) != 0 ||
          total_bytes > std::numeric_limits<int>::max() - count)
      {
        census_ok = 0;
        break;
      }
      total_bytes += count;
    }
  }
  MPI_Bcast(&census_ok, 1, MPI_INT, 0, communicator);
  if (census_ok != 1) return evaluation;
  std::vector<CpuCanonicalEntry> global;
  allocation_ok = 1;
  try
  {
    if (rank == 0)
      global.resize(static_cast<std::size_t>(total_bytes) /
                    sizeof(CpuCanonicalEntry));
  }
  catch (...) { allocation_ok = 0; }
  MPI_Allreduce(MPI_IN_PLACE, &allocation_ok, 1, MPI_INT, MPI_MIN,
                communicator);
  if (allocation_ok != 1) return evaluation;
  int gathered = MPI_Gatherv(
      local.data(), local_count, MPI_BYTE, global.data(), counts.data(),
      displacements.data(), MPI_BYTE, 0, communicator) == MPI_SUCCESS ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &gathered, 1, MPI_INT, MPI_MIN, communicator);
  if (gathered != 1) return evaluation;

  if (rank == 0)
  {
    try
    {
      mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
          elements_per_direction, elements_per_direction,
          mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
      std::vector<mfem::Vector> translations(2, mfem::Vector(2));
      translations[0][0] = 1.0;
      translations[1][1] = 1.0;
      mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
          original, original.CreatePeriodicVertexMapping(translations));
      mfem::L2_FECollection collection(
          cpu_functional_polynomial_order, 2,
          mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
      mfem::FiniteElementSpace space(
          &mesh, &collection, 4, mfem::Ordering::byNODES);
      mfem::GridFunction field(&space);
      field = 0.0;
      const int global_elements =
          elements_per_direction * elements_per_direction;
      const int scalar_dofs =
          (cpu_functional_polynomial_order + 1) *
          (cpu_functional_polynomial_order + 1);
      const std::size_t expected_records =
          static_cast<std::size_t>(global_elements) * scalar_dofs * 4;
      if (global.size() != expected_records)
        throw std::runtime_error("scientific state census is invalid");
      std::sort(global.begin(), global.end(),
                [](const CpuCanonicalEntry& left,
                   const CpuCanonicalEntry& right)
                { return left.global_id < right.global_id; });
      for (std::size_t index = 0; index < global.size(); ++index)
        if (global[index].global_id != index)
          throw std::runtime_error(
              "scientific state keys are not complete and unique");
      for (int element = 0; element < global_elements; ++element)
      {
        mfem::Array<int> vector_dofs;
        space.GetElementVDofs(element, vector_dofs);
        if (vector_dofs.Size() != scalar_dofs * 4)
          throw std::runtime_error(
              "scientific element map extent is invalid");
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
          for (int component = 0; component < 4; ++component)
          {
            const std::size_t source =
                (static_cast<std::size_t>(element) * scalar_dofs + scalar) *
                    4 +
                component;
            const int encoded = vector_dofs[
                scalar + component * scalar_dofs];
            const int destination = encoded >= 0 ? encoded : -1 - encoded;
            const double sign = encoded >= 0 ? 1.0 : -1.0;
            if (source >= global.size() || destination < 0 ||
                destination >= field.Size())
              throw std::runtime_error("scientific state map is out of range");
            field[destination] = sign *
                global[source].value;
          }
      }

      long double error_squared = 0.0L;
      std::array<long double, 4> numerical_integral{};
      std::array<long double, 4> exact_integral{};
      double minimum_density = std::numeric_limits<double>::infinity();
      double minimum_pressure = std::numeric_limits<double>::infinity();
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
        const mfem::FiniteElement* finite_element = space.GetFE(element);
        mfem::ElementTransformation* transformation =
            mesh.GetElementTransformation(element);
        if (finite_element == nullptr || transformation == nullptr)
          throw std::runtime_error("scientific FE topology is unavailable");
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), 26);
        for (int point = 0; point < rule.GetNPoints(); ++point)
        {
          const mfem::IntegrationPoint& integration = rule.IntPoint(point);
          transformation->SetIntPoint(&integration);
          mfem::Vector physical(2);
          transformation->Transform(integration, physical);
          mfem::Vector numerical(4);
          field.GetVectorValue(element, integration, numerical);
          std::array<double, 4> exact{};
          ManufacturedTestState(physical[0], physical[1], exact);
          const long double weight = static_cast<long double>(
              integration.weight * transformation->Weight());
          for (int component = 0; component < 4; ++component)
          {
            const long double difference =
                static_cast<long double>(numerical[component] -
                                         exact[component]);
            error_squared += weight * difference * difference;
            numerical_integral[static_cast<std::size_t>(component)] +=
                weight * numerical[component];
            exact_integral[static_cast<std::size_t>(component)] +=
                weight * exact[static_cast<std::size_t>(component)];
          }
          const double rho = numerical[0];
          const double pressure = 0.4 *
              (numerical[3] - 0.5 *
                  (numerical[1] * numerical[1] +
                   numerical[2] * numerical[2]) / rho);
          minimum_density = std::min(minimum_density, rho);
          minimum_pressure = std::min(minimum_pressure, pressure);
        }
      }
      evaluation.state_l2_error = std::sqrt(
          static_cast<double>(error_squared));
      evaluation.maximum_conservation_error = 0.0;
      for (std::size_t component = 0; component < 4; ++component)
        evaluation.maximum_conservation_error = std::max(
            evaluation.maximum_conservation_error,
            std::abs(static_cast<double>(numerical_integral[component] -
                                         exact_integral[component])));
      evaluation.minimum_density = minimum_density;
      evaluation.minimum_pressure = minimum_pressure;
    }
    catch (...) {}
  }
  MPI_Bcast(&evaluation.state_l2_error, 1, MPI_DOUBLE, 0, communicator);
  MPI_Bcast(&evaluation.maximum_conservation_error, 1, MPI_DOUBLE, 0,
            communicator);
  MPI_Bcast(&evaluation.minimum_density, 1, MPI_DOUBLE, 0, communicator);
  MPI_Bcast(&evaluation.minimum_pressure, 1, MPI_DOUBLE, 0, communicator);
  evaluation.passed = evaluation.discrete_oracle_match &&
      evaluation.coordinate_oracle_match &&
      evaluation.independent_true_residual &&
      std::isfinite(evaluation.state_l2_error) &&
      evaluation.state_l2_error > 0.0 &&
      evaluation.maximum_conservation_error <= 5.0e-11 &&
      evaluation.minimum_density >= 1.0e-8 &&
      evaluation.minimum_pressure >= 1.0e-8;
  return evaluation;
}

template <typename Selection>
int RunFunctionalTuple(MPI_Comm communicator, const int rank, const int ranks,
                       const int elements_per_direction)
{
  Checks checks{rank, 0};
  const CpuPersistentConfig config =
      MakeCpuFunctionalConfig(elements_per_direction);
  dg_igr_hdg_cpu_test::IndependentOracleConfig oracle_config;
  oracle_config.polynomial_order = cpu_functional_polynomial_order;
  oracle_config.elements_per_direction = elements_per_direction;
  oracle_config.assembly_quadrature_order =
      cpu_functional_assembly_quadrature_order;
  std::array<dg_igr_hdg_cpu_test::IndependentOracleResult,
             cpu_functional_accepted_steps + 1> step_oracles{};
  dg_igr_hdg_cpu_test::IndependentOracleResult& oracle = step_oracles[1];
  checks.Require(dg_igr_hdg_cpu_test::RunIndependentFirstStep(
                     communicator, oracle_config, oracle) && oracle.passed,
                 "functional first-step independent oracle");

  PersistentSmoothEulerCpuAdapter<Selection> uninterrupted;
  checks.Require(uninterrupted.Initialize(config),
                 "functional uninterrupted initialization");
  const CpuRunEvidence& initialized = uninterrupted.evidence();
  const CpuStorageEvidence uninterrupted_storage = initialized.storage;
  const std::uint64_t n = static_cast<std::uint64_t>(elements_per_direction);
  const std::uint64_t expected_interior =
      n * n * 9U * static_cast<std::uint64_t>(cpu_conserved_components);
  const std::uint64_t expected_trace =
      2U * n * n * 3U * static_cast<std::uint64_t>(
          cpu_conserved_components);
  checks.Require(initialized.polynomial_order ==
                         cpu_functional_polynomial_order &&
                     initialized.elements_per_direction ==
                         elements_per_direction &&
                     initialized.assembly_quadrature_order ==
                         cpu_functional_assembly_quadrature_order &&
                     initialized.dt == cpu_backward_euler_dt &&
                     initialized.global_elements == elements_per_direction *
                         elements_per_direction &&
                     initialized.global_owned_interior_coefficients ==
                         expected_interior &&
                     initialized.global_owned_true_trace_coefficients ==
                         expected_trace &&
                     initialized.unique_periodic_faces == 2U * n * n &&
                     initialized.initial_interior_hash != StateHash{} &&
                     initialized.initial_trace_hash != StateHash{} &&
                     initialized.topology_builds == 1 &&
                     initialized.accepted_step_count == 0 &&
                     initialized.storage.allocation_baseline > 0 &&
                     initialized.storage.allocation_events_after_baseline == 0,
                 "functional shape/topology/initial-state evidence");

  std::array<StoredCpuState, cpu_functional_accepted_steps + 1> reference{};
  reference[0] = StoreState(uninterrupted.Snapshot());
  checks.Require(CloseVectors(uninterrupted.Snapshot().interior,
                              oracle.initial_interior,
                              4096.0 *
                                  std::numeric_limits<double>::epsilon(),
                              4096.0 *
                                  std::numeric_limits<double>::epsilon()) &&
                     CloseVectors(uninterrupted.Snapshot().trace,
                                  oracle.initial_true_trace,
                                  4096.0 *
                                      std::numeric_limits<double>::epsilon(),
                                  4096.0 *
                                      std::numeric_limits<double>::epsilon()),
                 "functional initial projections match independent oracle");
  CpuPreparedStep first;
  checks.Require(uninterrupted.PrepareAdvance(first) && first.valid &&
                     first.numerically_ready &&
                     CloseVectors(first.interior, oracle.interior, 1.0e-8,
                                  2.0e-8) &&
                     CloseVectors(first.trace, oracle.true_trace, 1.0e-8,
                                  2.0e-8),
                 "functional production first step matches independent oracle");
  checks.Require(uninterrupted.FinalizePrepared(first).committed,
                 "functional uninterrupted step 1");
  reference[1] = StoreState(uninterrupted.Snapshot());
  const CommittedScientificEvaluation first_scientific =
      EvaluateCommittedScientificState(
          uninterrupted.Snapshot(), elements_per_direction, oracle,
          communicator);
  const CpuAttemptEvidence& functional_first =
      uninterrupted.evidence().accepted.attempt;
  checks.Require(first_scientific.passed && AcceptedStepPasses(
                     uninterrupted.evidence().accepted, 1) &&
                     functional_first.work.complete_factor_setups ==
                         functional_first.work.complete_trace_solves &&
                     functional_first.work.complete_trace_solves ==
                         functional_first.work.accepted_newton_updates + 1,
                 "functional step-1 solver-linked scientific/accounting gates");

  CpuCheckpoint uninterrupted_checkpoint;
  for (std::uint64_t step = 2; step <= cpu_functional_accepted_steps; ++step)
  {
    dg_igr_hdg_cpu_test::IndependentOracleResult& step_oracle =
        step_oracles[static_cast<std::size_t>(step)];
    const StoredCpuState& previous =
        reference[static_cast<std::size_t>(step - 1)];
    checks.Require(dg_igr_hdg_cpu_test::RunIndependentStep(
                       communicator, oracle_config, step,
                       previous.interior, previous.trace, step_oracle) &&
                       step_oracle.passed,
                   "functional current-step independent oracle");
    checks.Require(uninterrupted.Advance().committed,
                   "functional uninterrupted accepted step");
    reference[static_cast<std::size_t>(step)] =
        StoreState(uninterrupted.Snapshot());
    const CommittedScientificEvaluation scientific =
        EvaluateCommittedScientificState(
            uninterrupted.Snapshot(), elements_per_direction, step_oracle,
            communicator);
    const CpuRunEvidence& evidence = uninterrupted.evidence();
    checks.Require(scientific.passed &&
                       evidence.accepted_step_count == step &&
                       AcceptedStepPasses(evidence.accepted, step) &&
                       StorageUnchanged(uninterrupted_storage,
                                        evidence.storage),
                   "functional uninterrupted history and persistent reuse");
    if (step == 2)
    {
      StoredCpuState perturbed = reference[2];
      if (rank == 0 && !perturbed.interior.empty())
        perturbed.interior[0] += 1.0e-4;
      CpuStateSnapshot perturbed_snapshot = uninterrupted.Snapshot();
      perturbed_snapshot.interior = {
          perturbed.interior.data(), perturbed.interior.size(),
          perturbed.interior.capacity()};
      const CommittedScientificEvaluation rejected =
          EvaluateCommittedScientificState(
              perturbed_snapshot, elements_per_direction, step_oracle,
              communicator);
      checks.Require(!rejected.passed &&
                         !rejected.discrete_oracle_match,
                     "later committed-state perturbation fails all-step oracle gate");
    }
    if (step == cpu_functional_checkpoint_step)
    {
      checks.Require(uninterrupted.CaptureCheckpoint(
                         uninterrupted_checkpoint) &&
                         uninterrupted_checkpoint.valid &&
                         uninterrupted_checkpoint.descriptor.coordinates.step ==
                             cpu_functional_checkpoint_step &&
                         uninterrupted_checkpoint.descriptor.polynomial_order ==
                             cpu_functional_polynomial_order &&
                         uninterrupted_checkpoint.descriptor.layout.view() ==
                             cpu_detail::cpu_runtime_layout_identity &&
                         NearTime(uninterrupted_checkpoint.descriptor
                                      .coordinates.time,
                                  0.5) &&
                         ByteEqual(uninterrupted_checkpoint.interior(),
                                   reference[10].interior) &&
                         ByteEqual(uninterrupted_checkpoint.trace(),
                                   reference[10].trace),
                     "functional uninterrupted exact step-10 checkpoint bridge");
    }
  }
  const CpuRunEvidence& uninterrupted_terminal = uninterrupted.evidence();
  checks.Require(uninterrupted_terminal.attempts == 20 &&
                     uninterrupted_terminal.failed_attempts == 0 &&
                     uninterrupted_terminal.retries == 0 &&
                     uninterrupted_terminal.commits == 20 &&
                     uninterrupted_terminal.accepted_step_count == 20 &&
                     uninterrupted.coordinates().step == 20 &&
                     uninterrupted.coordinates().generation == 20 &&
                     uninterrupted.coordinates().cursor == 20 &&
                     NearTime(uninterrupted.coordinates().time, 1.0),
                 "functional uninterrupted terminal census");

  CpuCheckpoint split_checkpoint;
  {
    PersistentSmoothEulerCpuAdapter<Selection> writer;
    checks.Require(writer.Initialize(config),
                   "functional split-writer initialization");
    for (std::uint64_t step = 1; step <= cpu_functional_checkpoint_step;
         ++step)
    {
      checks.Require(writer.Advance().committed,
                     "functional split-writer accepted step");
      const CommittedScientificEvaluation scientific =
          EvaluateCommittedScientificState(
              writer.Snapshot(), elements_per_direction,
              step_oracles[static_cast<std::size_t>(step)], communicator);
      checks.Require(scientific.passed,
                     "functional split-writer all-step scientific gate");
    }
    const CpuStateSnapshot writer_state = writer.Snapshot();
    const bool checkpoint_captured =
        writer.CaptureCheckpoint(split_checkpoint);
    checks.Require(writer.coordinates().step == 10 &&
                       writer.coordinates().generation == 10 &&
                       writer.coordinates().cursor == 10 &&
                       ByteEqual(writer_state.interior,
                                 reference[10].interior) &&
                       ByteEqual(writer_state.trace, reference[10].trace) &&
                       writer.evidence().accepted_step_count == 10 &&
                       checkpoint_captured &&
                       split_checkpoint.valid &&
                       ByteEqual(split_checkpoint.interior(),
                                 reference[10].interior) &&
                       ByteEqual(split_checkpoint.trace(),
                                 reference[10].trace),
                   "functional split writer reproduces exact step-10 bridge");
  }

  PersistentSmoothEulerCpuAdapter<Selection> reader;
  checks.Require(reader.Initialize(config),
                 "functional split-reader initialization");
  const CpuStorageEvidence reader_storage = reader.evidence().storage;
  const RestoreOutcome restored = reader.Restore(split_checkpoint);
  checks.Require(restored.restored && reader.coordinates().step == 10 &&
                     reader.coordinates().generation == 10 &&
                     reader.coordinates().cursor == 10 &&
                     NearTime(reader.coordinates().time, 0.5),
                 "functional split-reader exact step-10 restore");
  for (std::uint64_t step = 11; step <= cpu_functional_accepted_steps; ++step)
  {
    if (step == cpu_functional_retry_target_step)
    {
      const StoredCpuState before = StoreState(reader.Snapshot());
      CpuFailureInjection injection;
      injection.phase = CpuFailurePhase::after_recovery_before_commit;
      injection.target_step = cpu_functional_retry_target_step;
      injection.source_rank = ranks == 1 ? 0 : 1;
      CpuPreparedStep failed_candidate;
      checks.Require(reader.PrepareAdvance(failed_candidate, injection) &&
                         !failed_candidate.numerically_ready,
                     "functional prescribed step-14 failure reaches rollback");
      const AttemptOutcome failure =
          reader.FinalizePrepared(failed_candidate);
      const CpuStateSnapshot rolled_back = reader.Snapshot();
      checks.Require(!failure.committed && failure.retry_available &&
                         ByteEqual(rolled_back.interior, before.interior) &&
                         ByteEqual(rolled_back.trace, before.trace) &&
                         rolled_back.interior_hash == before.interior_hash &&
                         rolled_back.trace_hash == before.trace_hash &&
                         rolled_back.coordinates.step == 13 &&
                         reader.evidence().last_attempt.rollback_byte_equal &&
                         reader.evidence().last_attempt.failure_phase ==
                             CpuFailurePhase::after_recovery_before_commit,
                     "functional step-14 rollback is byte-atomic");
      checks.Require(reader.Retry().committed,
                     "functional prescribed step-14 retry");
    }
    else
    {
      checks.Require(reader.Advance().committed,
                     "functional split-reader accepted step");
    }
    const CpuStateSnapshot actual = reader.Snapshot();
    const CommittedScientificEvaluation scientific =
        EvaluateCommittedScientificState(
            reader.Snapshot(), elements_per_direction,
            step_oracles[static_cast<std::size_t>(step)], communicator);
    const StoredCpuState& expected = reference[static_cast<std::size_t>(step)];
    const CpuRunEvidence& evidence = reader.evidence();
    checks.Require(scientific.passed &&
                       CloseVectors(actual.interior, expected.interior,
                                4096.0 *
                                    std::numeric_limits<double>::epsilon(),
                                4096.0 *
                                    std::numeric_limits<double>::epsilon()) &&
                       CloseVectors(actual.trace, expected.trace,
                                    4096.0 *
                                        std::numeric_limits<double>::epsilon(),
                                    4096.0 *
                                        std::numeric_limits<double>::epsilon()) &&
                       evidence.accepted_step_count == step - 10 &&
                       AcceptedStepPasses(evidence.accepted, step) &&
                       StorageUnchanged(reader_storage, evidence.storage),
                   "functional restart/retry trajectory and reuse equivalence");
  }
  checks.Require(reader.coordinates().step == 20 &&
                     reader.coordinates().generation == 20 &&
                     reader.coordinates().cursor == 21 &&
                     NearTime(reader.coordinates().time, 1.0) &&
                     reader.evidence().attempts == 11 &&
                     reader.evidence().failed_attempts == 1 &&
                     reader.evidence().retries == 1 &&
                     reader.evidence().commits == 10 &&
                     reader.evidence().accepted_step_count == 10,
                 "functional split/retry terminal census");

  int global_failures = 0;
  MPI_Allreduce(&checks.failures, &global_failures, 1, MPI_INT, MPI_SUM,
                communicator);
  if (rank == 0 && global_failures == 0)
  {
    std::cout << "persistent CPU functional leaf passed source candidate "
              << "checks: ranks=" << ranks
              << " p=2 N=" << elements_per_direction << '\n';
  }
  return global_failures == 0 ? 0 : 1;
}

template <typename Selection>
int RunTuple(MPI_Comm communicator, const int rank, const int ranks)
{
  Checks checks{rank, 0};
  dg_igr_hdg_cpu_test::IndependentOracleResult oracle;
  checks.Require(dg_igr_hdg_cpu_test::RunIndependentFirstStep(
                     communicator, oracle) && oracle.passed,
                 "independent mesh/source/flux/Jacobian/signed-map/Schur/"
                 "P^TSP/true-residual/distribute/recovery oracle");

  CheckAsymmetricCollectiveInitializationFaults<Selection>(
      checks, communicator, rank, ranks);
  CheckCollectiveFacadeInitialization<Selection>(
      checks, communicator, rank, ranks);
  CheckAsymmetricCollectiveNumericsFaults<Selection>(
      checks, communicator, rank, ranks);

  PersistentSmoothEulerCpuAdapter<Selection> adapter;
  checks.Require(adapter.Initialize(), "persistent adapter initialization");
  const CpuRunEvidence& initial = adapter.evidence();
  const CpuStorageEvidence initial_storage = initial.storage;
  checks.Require(initial.tuple.model.view() == "euler.compressible.2d" &&
                     initial.tuple.formulation.view() == "hdg.conventional" &&
                     initial.tuple.trace.view() ==
                         "mfem.dg_interface.trace" &&
                     initial.tuple.stabilization.view() == "tau.fixed" &&
                     initial.tuple.time_treatment.view() == "backward_euler" &&
                     initial.tuple.precision.view() == "fp64" &&
                     initial.experiment.view() ==
                         "exp.hdg.euler.conventional.smooth_persistent.w8a.v1",
                 "tuple and experiment IDs must remain frozen");
  checks.Require(initial.storage.interior_size ==
                         initial.storage.interior_capacity &&
                     initial.storage.trace_size ==
                         initial.storage.trace_capacity &&
                     initial.storage.sparse_owned_term_capacity > 0 &&
                     initial.storage.true_diagonal_nnz > 0 &&
                     initial.storage.sparse_owned_term_token != 0 &&
                     initial.storage.topology_generation == 1 &&
                     initial.storage.storage_generation == 1 &&
                     initial.storage.allocation_baseline > 0 &&
                     initial.storage.opaque_mfem_hypre_allocations_unobserved &&
                     initial.storage
                         .opaque_standard_library_control_blocks_unobserved &&
                     initial.storage.allocation_events_after_baseline == 0 &&
                     initial.storage.publish_is_complete_slot_exchange &&
                     initial.storage
                         .candidate_retains_exact_old_state_after_publish &&
                     initial.storage.revert_publish_is_exact_byte_inverse &&
                     initial.storage
                         .publish_and_revert_preserve_disjoint_backing,
                 "complete slots/workspace must be preallocated");

  CpuPreparedStep first;
  checks.Require(adapter.PrepareAdvance(first) && first.valid &&
                     first.numerically_ready &&
                     first.identity.sequence == 1 &&
                     first.candidate_cursor == 1 &&
                     first.entry_coordinates.step == 0 &&
                     NearTime(first.target_coordinates.time, 0.05) &&
                     first.target_coordinates.step == 1 &&
                     first.target_coordinates.generation == 1 &&
                     first.target_coordinates.cursor == 1,
                 "step 1 must expose an attempt-bound precommit candidate");
  checks.Require(CloseVectors(first.interior, oracle.interior, 1.0e-8,
                              2.0e-8) &&
                     CloseVectors(first.trace, oracle.true_trace, 1.0e-8,
                                  2.0e-8),
                 "independent oracle must match the production candidate");
  checks.Require(adapter.FinalizePrepared(first).committed,
                 "step 1 commit");
  const CpuAttemptEvidence& first_accepted =
      adapter.evidence().accepted.attempt;
  checks.Require(oracle.passed && oracle.newton_updates > 1 &&
                     first_accepted.nonlinear_updates ==
                         oracle.newton_updates &&
                     first_accepted.work.complete_factor_setups ==
                         first_accepted.work.complete_trace_solves &&
                     first_accepted.work.complete_trace_solves ==
                         first_accepted.work.accepted_newton_updates + 1 &&
                     first_accepted.finite_gate &&
                     first_accepted.admissibility_gate &&
                     first_accepted.recovery_gate &&
                     first_accepted.nonlinear_gate &&
                     first_accepted.true_residual_gate &&
                     std::isfinite(
                         first_accepted.maximum_true_relative_residual) &&
                     first_accepted.maximum_true_relative_residual <=
                         independent_true_residual_limit,
                 "independent solver state and compact Newton census agree");
  checks.Require(NearTime(adapter.coordinates().time, 0.05) &&
                     adapter.coordinates().step == 1 &&
                     adapter.coordinates().generation == 1 &&
                     adapter.coordinates().cursor == 1,
                 "step 1 coordinates");
  CpuCheckpoint checkpoint;
  checks.Require(adapter.CaptureCheckpoint(checkpoint) && checkpoint.valid &&
                     checkpoint.descriptor.coordinates.step == 1 &&
                     checkpoint.descriptor.polynomial_order ==
                         cpu_smoke_polynomial_order &&
                     checkpoint.descriptor.layout.view() ==
                         cpu_detail::cpu_runtime_layout_identity,
                 "in-memory committed checkpoint capture");

  const CpuStateSnapshot before_failure = adapter.Snapshot();
  const std::vector<double> committed_interior =
      CopyView(before_failure.interior);
  const std::vector<double> committed_trace = CopyView(before_failure.trace);
  CpuFailureInjection prescribed;
  prescribed.phase = CpuFailurePhase::after_recovery_before_commit;
  prescribed.target_step = 2;
  prescribed.source_rank = ranks == 1 ? 0 : 1;
  CpuPreparedStep failed_candidate;
  checks.Require(adapter.PrepareAdvance(failed_candidate, prescribed) &&
                     !failed_candidate.numerically_ready,
                 "prescribed step-2 candidate reaches injected precommit stop");
  const AttemptOutcome failed = adapter.FinalizePrepared(failed_candidate);
  checks.Require(!failed.committed && failed.retry_available,
                 "prescribed step-2 failure is retryable");
  const CpuStateSnapshot after_failure = adapter.Snapshot();
  checks.Require(ByteEqual(after_failure.interior, committed_interior) &&
                     ByteEqual(after_failure.trace, committed_trace) &&
                     after_failure.interior_hash == before_failure.interior_hash &&
                     after_failure.trace_hash == before_failure.trace_hash &&
                     NearTime(after_failure.coordinates.time, 0.05) &&
                     after_failure.coordinates.step == 1 &&
                     after_failure.coordinates.generation == 1 &&
                     after_failure.coordinates.cursor == 1,
                 "failed attempt preserves bytes/hash/coordinates");

  CpuPreparedStep retry;
  checks.Require(adapter.PrepareRetry(retry) && retry.numerically_ready &&
                     retry.identity.sequence == 3 &&
                     retry.candidate_cursor == 3,
                 "single retry identity/cursor");
  checks.Require(adapter.FinalizePrepared(retry).committed,
                 "step-2 retry commit");
  checks.Require(adapter.Advance().committed, "step-3 commit");
  checks.Require(NearTime(adapter.coordinates().time, 0.15) &&
                     adapter.coordinates().step == 3 &&
                     adapter.coordinates().generation == 3 &&
                     adapter.coordinates().cursor == 4,
                 "three commits and four attempts coordinates");

  if (ranks == 2)
  {
    const CpuStateSnapshot identity_state = adapter.Snapshot();
    const std::uint64_t global_identity =
        IndependentRankOrderedStateIdentity(identity_state, false,
                                            communicator);
    const std::uint64_t rank_one_perturbed_identity =
        IndependentRankOrderedStateIdentity(identity_state, true,
                                            communicator);
    checks.Require(global_identity != 0 &&
                       rank_one_perturbed_identity != global_identity,
                   "rank1-only perturbation invalidates global state identity");
  }

  const CpuRunEvidence& terminal = adapter.evidence();
  checks.Require(terminal.attempts == 4 && terminal.failed_attempts == 1 &&
                     terminal.retries == 1 && terminal.commits == 3 &&
                     terminal.accepted.valid &&
                     terminal.accepted.attempt.admissibility_gate &&
                     terminal.accepted.attempt.nonlinear_gate &&
                     terminal.accepted.attempt.true_residual_gate,
                 "status/census/admissibility parity");
  std::array<StateHash, 2> common_interior{};
  std::array<StateHash, 2> common_trace{};
  std::array<StateHash, 2> local_interior_hashes{};
  std::array<StateHash, 2> local_trace_hashes{};
  MPI_Allgather(terminal.accepted.rank_common_interior_hash.bytes.data(), 32,
                MPI_BYTE, common_interior.data(), 32, MPI_BYTE,
                communicator);
  MPI_Allgather(terminal.accepted.rank_common_trace_hash.bytes.data(), 32,
                MPI_BYTE, common_trace.data(), 32, MPI_BYTE, communicator);
  MPI_Allgather(terminal.accepted.interior_hash.bytes.data(), 32, MPI_BYTE,
                local_interior_hashes.data(), 32, MPI_BYTE, communicator);
  MPI_Allgather(terminal.accepted.trace_hash.bytes.data(), 32, MPI_BYTE,
                local_trace_hashes.data(), 32, MPI_BYTE, communicator);
  checks.Require(common_interior[0] == common_interior[ranks - 1] &&
                     common_trace[0] == common_trace[ranks - 1] &&
                     (ranks == 1 ||
                      local_interior_hashes[0] != local_interior_hashes[1] ||
                      local_trace_hashes[0] != local_trace_hashes[1]),
                 "checkpoint identity is rank-common while local hashes remain local");
  const CpuAttemptEvidence& accepted = terminal.accepted.attempt;
  checks.Require(accepted.work.complete_assemblies ==
                         1 + accepted.work.trial_reassemblies &&
                     accepted.work.complete_condensations ==
                         accepted.work.accepted_newton_updates + 1 &&
                     accepted.work.complete_factor_setups ==
                         accepted.work.complete_trace_solves &&
                     accepted.work.complete_factor_setups ==
                         accepted.work.accepted_newton_updates + 1 &&
                     accepted.work.complete_trace_solves ==
                         accepted.work.accepted_newton_updates + 1 &&
                     accepted.work.complete_recoveries ==
                         accepted.work.accepted_newton_updates + 1 &&
                     accepted.candidate_generation != 0 &&
                     accepted.committed_interior_after ==
                         terminal.accepted.interior_hash &&
                     accepted.committed_trace_after ==
                         terminal.accepted.trace_hash,
                 "compact Newton census and committed-state binding");
  const RunTelemetry* telemetry = adapter.contract_telemetry();
  checks.Require(telemetry && telemetry->attempts_started == 4 &&
                     telemetry->attempts_failed == 1 &&
                     telemetry->retries_started == 1 &&
                     telemetry->commits == 3 &&
                     telemetry->attempted_transfers.host_to_device_bytes == 0 &&
                     telemetry->attempted_transfers.device_to_host_bytes == 0 &&
                     telemetry->attempted_transfers.mpi_staging_bytes == 0 &&
                     telemetry->attempted_transfers.allocation_events == 0 &&
                     telemetry->committed_transfers.allocation_events == 0,
                 "contract census and CPU zero-transfer/allocation accounting");
  checks.Require(terminal.storage.committed_storage_token ==
                         initial_storage.committed_storage_token &&
                     terminal.storage.candidate_storage_token ==
                         initial_storage.candidate_storage_token &&
                     terminal.storage.interior_capacity ==
                         initial_storage.interior_capacity &&
                     terminal.storage.trace_capacity ==
                         initial_storage.trace_capacity &&
                     terminal.storage.sparse_owned_term_capacity ==
                         initial_storage.sparse_owned_term_capacity &&
                     terminal.storage.sparse_outgoing_term_capacity ==
                         initial_storage.sparse_outgoing_term_capacity &&
                     terminal.storage.sparse_incoming_slot_capacity ==
                         initial_storage.sparse_incoming_slot_capacity &&
                     terminal.storage.sparse_owned_term_token ==
                         initial_storage.sparse_owned_term_token &&
                     terminal.storage.sparse_outgoing_term_token ==
                         initial_storage.sparse_outgoing_term_token &&
                     terminal.storage.sparse_incoming_slot_token ==
                         initial_storage.sparse_incoming_slot_token &&
                     terminal.storage.allocation_events_after_baseline == 0,
                 "stable storage IDs/capacities and no post-baseline allocation");

  PersistentSmoothEulerCpuAdapter<Selection> restarted;
  checks.Require(restarted.Initialize(), "restart adapter setup");
  const RestoreOutcome restore = restarted.Restore(checkpoint);
  checks.Require(restore.restored && restarted.coordinates().step == 1 &&
                     NearTime(restarted.coordinates().time, 0.05),
                 "in-memory checkpoint restore");
  checks.Require(restarted.Advance().committed &&
                     restarted.Advance().committed,
                 "restored steps 2 and 3");
  const CpuStateSnapshot uninterrupted = adapter.Snapshot();
  const CpuStateSnapshot continued = restarted.Snapshot();
  checks.Require(CloseVectors(continued.interior,
                              CopyView(uninterrupted.interior),
                              4096.0 * std::numeric_limits<double>::epsilon(),
                              4096.0 * std::numeric_limits<double>::epsilon()) &&
                     CloseVectors(continued.trace,
                                  CopyView(uninterrupted.trace),
                                  4096.0 *
                                      std::numeric_limits<double>::epsilon(),
                                  4096.0 *
                                      std::numeric_limits<double>::epsilon()),
                 "uninterrupted/restarted committed state equivalence");

  CheckCanonicalIds(adapter, checks, communicator);
  CheckPhysicalInteriorIds(adapter, checks, communicator);
  CheckCheckpointImportAndIndependence(
      adapter, checkpoint, checks, communicator, rank, ranks);
  CheckProductNegatives<Selection>(checks);
  CheckPhaseRankInjection<Selection>(checks, ranks);
  CheckNaturalFailurePrecedesLaterInjection<Selection>(checks, ranks);
  CheckOuterStageAbort<Selection>(checks, ranks);
  CheckTwoPhaseAndAffineAuthorities<Selection>(checks, ranks);

  int global_failures = 0;
  MPI_Allreduce(&checks.failures, &global_failures, 1, MPI_INT, MPI_SUM,
                communicator);
  if (rank == 0 && global_failures == 0)
  {
    std::cout << "persistent CPU tuple passed source candidate checks: ranks="
              << ranks << '\n';
  }
  return global_failures == 0 ? 0 : 1;
}

int RunSolverLinkedConvergence(const MPI_Comm communicator, const int rank)
{
  Checks checks{rank, 0};
  const std::array<int, 3> meshes{{3, 6, 12}};
  std::array<double, 3> errors{};
  for (std::size_t index = 0; index < meshes.size(); ++index)
  {
    dg_igr_hdg_cpu_test::IndependentOracleConfig oracle_config;
    oracle_config.polynomial_order = cpu_functional_polynomial_order;
    oracle_config.elements_per_direction = meshes[index];
    oracle_config.assembly_quadrature_order =
        cpu_functional_assembly_quadrature_order;
    PersistentSmoothEulerCpuAdapter<SerialSelection> adapter;
    bool all_steps_admitted = adapter.Initialize(
        MakeCpuFunctionalConfig(meshes[index]));
    StoredCpuState previous;
    if (all_steps_admitted) previous = StoreState(adapter.Snapshot());
    CommittedScientificEvaluation evaluation;
    for (std::uint64_t step = 1;
         step <= cpu_functional_accepted_steps && all_steps_admitted; ++step)
    {
      dg_igr_hdg_cpu_test::IndependentOracleResult oracle;
      const bool oracle_ready = step == 1
          ? dg_igr_hdg_cpu_test::RunIndependentFirstStep(
                communicator, oracle_config, oracle)
          : dg_igr_hdg_cpu_test::RunIndependentStep(
                communicator, oracle_config, step, previous.interior,
                previous.trace, oracle);
      const bool advanced = oracle_ready && oracle.passed &&
          adapter.Advance().committed;
      if (advanced)
      {
        evaluation = EvaluateCommittedScientificState(
            adapter.Snapshot(), meshes[index], oracle, communicator);
        const CpuRunEvidence& evidence = adapter.evidence();
        all_steps_admitted = evaluation.passed &&
            evidence.accepted_step_count == step &&
            AcceptedStepPasses(evidence.accepted, step);
        previous = StoreState(adapter.Snapshot());
      }
      else
      {
        all_steps_admitted = false;
      }
    }
    checks.Require(all_steps_admitted,
                   "solver-linked convergence all-step states are admitted");
    errors[index] = evaluation.state_l2_error;
  }
  const double rate_36 = std::log2(errors[0] / errors[1]);
  const double rate_612 = std::log2(errors[1] / errors[2]);
  checks.Require(std::isfinite(errors[0]) &&
                     std::isfinite(errors[1]) &&
                     std::isfinite(errors[2]) && errors[0] > errors[1] &&
                     errors[1] > errors[2] && errors[2] > 0.0 &&
                     rate_36 >= 2.25 && rate_612 >= 2.25,
                 "committed solver N3/N6/N12 spatial order");
  if (rank == 0 && checks.failures == 0)
    std::cout << "solver-linked convergence PASS errors="
              << errors[0] << ',' << errors[1] << ',' << errors[2]
              << " rates=" << rate_36 << ',' << rate_612 << '\n';
  return checks.failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int result = 75;
  int functional_n = 0;
  const bool functional_request = argc == 3 &&
      std::string_view{argv[1]} == "--functional-n";
  const bool convergence_request = argc == 2 &&
      std::string_view{argv[1]} == "--scientific-convergence";
  if (functional_request)
  {
    const std::string_view value{argv[2]};
    if (value == "3") { functional_n = 3; }
    else if (value == "6") { functional_n = 6; }
    else if (value == "12") { functional_n = 12; }
  }
  if (argc == 1 && ranks == 1)
  {
    result = RunTuple<SerialSelection>(MPI_COMM_WORLD, rank, ranks);
  }
  else if (argc == 1 && ranks == 2)
  {
    result = RunTuple<Rank2Selection>(MPI_COMM_WORLD, rank, ranks);
  }
  else if (functional_request && functional_n != 0 && ranks == 1)
  {
    result = RunFunctionalTuple<SerialSelection>(
        MPI_COMM_WORLD, rank, ranks, functional_n);
  }
  else if (functional_request && functional_n == 6 && ranks == 2)
  {
    result = RunFunctionalTuple<Rank2Selection>(
        MPI_COMM_WORLD, rank, ranks, functional_n);
  }
  else if (convergence_request && ranks == 1)
  {
    result = RunSolverLinkedConvergence(MPI_COMM_WORLD, rank);
  }
  else
  {
    result = 64;
  }
  MPI_Finalize();
  return result;
}
