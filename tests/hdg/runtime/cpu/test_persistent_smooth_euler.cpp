#include "independent_oracle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cpu;

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
              "bounded CPU tuple constants");

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
                         adapter.evidence().last_attempt.failure_phase == phase,
                     "phase/rank rollback must preserve committed bytes and IDs");
    }
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
    checks.Require(adapter.Advance().committed,
                   "outer-stage abort prefix step 1");
    const CpuStateSnapshot entry = adapter.Snapshot();
    const std::vector<double> interior = CopyView(entry.interior);
    const std::vector<double> trace = CopyView(entry.trace);
    CpuPreparedStep prepared;
    checks.Require(adapter.PrepareAdvance(prepared) &&
                       prepared.numerically_ready &&
                       prepared.entry_coordinates.step == 1 &&
                       NearTime(prepared.target_coordinates.time, 0.10) &&
                       prepared.target_coordinates.step == 2,
                   "outer stage must receive a qualified numerical snapshot");
    const AttemptOutcome failure = adapter.AbortPrepared(
        prepared, phase, ranks == 1 ? 0 : 1);
    const CpuStateSnapshot exit = adapter.Snapshot();
    checks.Require(!failure.committed && failure.retry_available &&
                       ByteEqual(exit.interior, interior) &&
                       ByteEqual(exit.trace, trace) &&
                       exit.coordinates.step == 1 &&
                       adapter.evidence().last_attempt.failure_phase == phase,
                   "prepublication outer-stage abort must roll back naturally");
  }
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

  PersistentSmoothEulerCpuAdapter<Selection> adapter;
  checks.Require(adapter.Initialize(), "persistent adapter initialization");
  const CpuRunEvidence initial = adapter.evidence();
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
                     initial.storage.topology_generation == 1 &&
                     initial.storage.storage_generation == 1 &&
                     initial.storage.allocation_events_after_baseline == 0,
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
  checks.Require(NearTime(adapter.coordinates().time, 0.05) &&
                     adapter.coordinates().step == 1 &&
                     adapter.coordinates().generation == 1 &&
                     adapter.coordinates().cursor == 1,
                 "step 1 coordinates");
  CpuCheckpoint checkpoint;
  checks.Require(adapter.CaptureCheckpoint(checkpoint) && checkpoint.valid &&
                     checkpoint.descriptor.coordinates.step == 1,
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

  const CpuRunEvidence& terminal = adapter.evidence();
  checks.Require(terminal.attempts == 4 && terminal.failed_attempts == 1 &&
                     terminal.retries == 1 && terminal.commits == 3 &&
                     terminal.accepted.valid &&
                     terminal.accepted.attempt.admissibility_gate &&
                     terminal.accepted.attempt.nonlinear_gate &&
                     terminal.accepted.attempt.true_residual_gate,
                 "status/census/admissibility parity");
  const CpuAttemptEvidence& accepted = terminal.accepted.attempt;
  checks.Require(accepted.work.complete_assemblies ==
                         1 + accepted.work.trial_reassemblies &&
                     accepted.work.complete_condensations ==
                         accepted.work.accepted_newton_updates &&
                     accepted.work.complete_trace_solves ==
                         accepted.work.accepted_newton_updates &&
                     accepted.work.complete_recoveries ==
                         accepted.work.accepted_newton_updates &&
                     accepted.products[0].kind ==
                         DerivedProductKind::residual &&
                     accepted.products[1].kind ==
                         DerivedProductKind::matrix &&
                     accepted.products[2].kind ==
                         DerivedProductKind::factor &&
                     accepted.products[3].kind ==
                         DerivedProductKind::solution &&
                     std::all_of(accepted.products.begin(),
                                 accepted.products.end(),
                                 [&](const CpuProductRecord& product)
                                 {
                                   return product.produced &&
                                       product.attempt_sequence ==
                                           accepted.identity.sequence &&
                                       product.candidate_cursor ==
                                           accepted.candidate_cursor;
                                 }),
                 "Newton census and attempt-qualified product order");
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
                         initial.storage.committed_storage_token &&
                     terminal.storage.candidate_storage_token ==
                         initial.storage.candidate_storage_token &&
                     terminal.storage.interior_capacity ==
                         initial.storage.interior_capacity &&
                     terminal.storage.trace_capacity ==
                         initial.storage.trace_capacity &&
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
  CheckProductNegatives<Selection>(checks);
  CheckPhaseRankInjection<Selection>(checks, ranks);
  CheckOuterStageAbort<Selection>(checks, ranks);

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

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int result = 75;
  if (ranks == 1)
  {
    result = RunTuple<SerialSelection>(MPI_COMM_WORLD, rank, ranks);
  }
  else if (ranks == 2)
  {
    result = RunTuple<Rank2Selection>(MPI_COMM_WORLD, rank, ranks);
  }
  MPI_Finalize();
  return result;
}
