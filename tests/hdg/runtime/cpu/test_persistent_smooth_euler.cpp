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
#include <sstream>
#include <string>
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
static_assert(cpu_maximum_local_interior_coefficients == 144 &&
                  cpu_maximum_local_true_trace_coefficients == 144,
              "p1/N3 rank1/rank2 chain capture capacities are exact");
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

std::string HashHex(const std::array<std::uint8_t, 32>& hash)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string result(hash.size() * 2, '0');
  for (std::size_t index = 0; index < hash.size(); ++index)
  {
    result[2 * index] = digits[hash[index] >> 4U];
    result[2 * index + 1] = digits[hash[index] & 0x0fU];
  }
  return result;
}

constexpr double independent_trajectory_relative_tolerance = 1.0e-8;
constexpr double independent_trajectory_scaled_tolerance = 2.0e-8;
constexpr double independent_true_residual_limit = 1.0e-10;
constexpr double independent_armijo_constant = 1.0e-4;

bool HashNonZero(const std::array<std::uint8_t, 32>& hash) noexcept
{
  return std::any_of(hash.begin(), hash.end(),
                     [](const std::uint8_t byte) { return byte != 0; });
}

double RelativeScalarDifference(const double left,
                                const double right) noexcept
{
  return std::abs(left - right) /
      std::max({std::abs(left), std::abs(right), 1.0e-300});
}

bool CloseTrajectoryScalar(const double left, const double right) noexcept
{
  if (!std::isfinite(left) || !std::isfinite(right)) { return false; }
  const double difference = std::abs(left - right);
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return RelativeScalarDifference(left, right) <=
          independent_trajectory_relative_tolerance &&
      difference <= independent_trajectory_scaled_tolerance * scale;
}

struct VectorComparison
{
  double production_norm = 0.0;
  double oracle_norm = 0.0;
  double difference_norm = 0.0;
  double relative_difference = std::numeric_limits<double>::infinity();
  double maximum_difference = std::numeric_limits<double>::infinity();
  double infinity_scale = 1.0;
  bool size_equal = false;
  bool finite = false;
  bool close = false;
};

VectorComparison CompareVectorData(const double* production,
                                   const std::size_t production_size,
                                   const std::vector<double>& oracle) noexcept
{
  VectorComparison comparison;
  comparison.size_equal = production_size == oracle.size();
  if (!comparison.size_equal || (!production && production_size != 0))
  {
    return comparison;
  }
  double production_squared = 0.0;
  double oracle_squared = 0.0;
  double difference_squared = 0.0;
  double maximum_difference = 0.0;
  double infinity_scale = 1.0;
  bool finite = true;
  for (std::size_t index = 0; index < oracle.size(); ++index)
  {
    finite = finite && std::isfinite(production[index]) &&
        std::isfinite(oracle[index]);
    const double difference = production[index] - oracle[index];
    production_squared += production[index] * production[index];
    oracle_squared += oracle[index] * oracle[index];
    difference_squared += difference * difference;
    maximum_difference = std::max(maximum_difference, std::abs(difference));
    infinity_scale = std::max(
        infinity_scale,
        std::max(std::abs(production[index]), std::abs(oracle[index])));
  }
  comparison.production_norm = std::sqrt(production_squared);
  comparison.oracle_norm = std::sqrt(oracle_squared);
  comparison.difference_norm = std::sqrt(difference_squared);
  comparison.relative_difference = comparison.difference_norm /
      std::max({comparison.production_norm, comparison.oracle_norm, 1.0e-300});
  comparison.maximum_difference = maximum_difference;
  comparison.infinity_scale = infinity_scale;
  comparison.finite = finite && std::isfinite(comparison.relative_difference) &&
      std::isfinite(comparison.maximum_difference);
  comparison.close = comparison.finite &&
      comparison.relative_difference <=
          independent_trajectory_relative_tolerance &&
      comparison.maximum_difference <=
          independent_trajectory_scaled_tolerance * comparison.infinity_scale;
  return comparison;
}

template <std::size_t Capacity>
VectorComparison CompareCaptured(
    const std::array<double, Capacity>& production,
    const std::size_t production_size,
    const std::vector<double>& oracle) noexcept
{
  if (production_size > production.size()) { return {}; }
  return CompareVectorData(production.data(), production_size, oracle);
}

const CpuTrialEvidence* AcceptedProductionTrial(
    const CpuAttemptEvidence& produced,
    const int update,
    int& attempted_trials,
    int& accepted_trials,
    bool& ordered_trial_indices) noexcept
{
  attempted_trials = 0;
  accepted_trials = 0;
  ordered_trial_indices = true;
  const CpuTrialEvidence* accepted = nullptr;
  for (std::size_t index = 0; index < produced.trial_count; ++index)
  {
    const CpuTrialEvidence& trial = produced.trials[index];
    if (trial.newton_iteration != update) { continue; }
    ordered_trial_indices = ordered_trial_indices &&
        trial.trial_index == attempted_trials;
    ++attempted_trials;
    if (trial.accepted)
    {
      ++accepted_trials;
      accepted = &trial;
    }
  }
  return accepted;
}

void ReportIndependentChainDiagnostics(
    const dg_igr_hdg_cpu_test::IndependentOracleResult& oracle,
    const CpuAttemptEvidence& produced,
    const int rank)
{
  const std::size_t oracle_count = oracle.linearizations.size();
  const std::size_t produced_count = produced.linearization_chain_count;
  std::ostringstream output;
  output << "rank " << rank
         << ": independent_oracle flags"
         << " mesh=" << oracle.mesh_rebuilt
         << " analytic_source_flux_jacobian="
         << oracle.analytic_source_flux_jacobian
         << " signed_maps=" << oracle.signed_maps_rebuilt
         << " signed_map_extent=" << oracle.signed_map_extent_valid
         << " signed_map_entries=" << oracle.signed_map_entries_valid
         << " signed_map_permutation="
         << oracle.signed_map_permutation_rebuilt
         << " signed_map_used=" << oracle.signed_map_used
         << " negative_signed_trace_dofs="
         << oracle.negative_signed_trace_dofs
         << " local_schur=" << oracle.local_schur_rebuilt
         << " global_ptsp=" << oracle.global_ptsp_rebuilt
         << " true_residual=" << oracle.true_residual_rebuilt
         << " distribute=" << oracle.distribute_rebuilt
         << " recovery=" << oracle.recovery_rebuilt
         << " admissible=" << oracle.admissible
         << " passed=" << oracle.passed
         << " initial_residual=" << oracle.initial_residual
         << " final_residual=" << oracle.final_residual
         << " maximum_true_residual=" << oracle.maximum_true_residual
         << '\n'
         << "rank " << rank
         << ": independent_chain counts oracle=" << oracle_count
         << " produced=" << produced_count
         << " oracle_steps=" << oracle.newton_steps.size()
         << " oracle_newton_updates=" << oracle.newton_updates
         << " produced_newton_updates=" << produced.nonlinear_updates
         << " production_trial_count=" << produced.trial_count
         << " production_factor_setups="
         << produced.work.complete_factor_setups
         << " production_trace_solves="
         << produced.work.complete_trace_solves
         << " count_equal=" << (oracle_count == produced_count)
         << " oracle_count_matches_updates="
         << (oracle_count == static_cast<std::size_t>(oracle.newton_updates))
         << " produced_count_matches_updates="
         << (produced_count ==
             static_cast<std::size_t>(produced.nonlinear_updates))
         << " multi_update=" << (oracle_count > 1) << '\n';

  const std::size_t common = std::min(oracle_count, produced_count);
  for (std::size_t index = 0; index < common; ++index)
  {
    const dg_igr_hdg_cpu_test::IndependentLinearizationHashes& rebuilt =
        oracle.linearizations[index];
    const CpuLinearizationChain& chain =
        produced.linearization_chains[index];
    const bool residual_equal =
        chain.residual_hash.bytes == rebuilt.residual;
    const bool matrix_equal = chain.matrix_hash.bytes == rebuilt.matrix;
    const bool factor_equal = chain.factor_hash.bytes == rebuilt.factor;
    const bool solution_equal = chain.solution_hash.bytes == rebuilt.solution;
    const bool candidate_equal =
        chain.candidate_state_hash.bytes == rebuilt.candidate;
    const bool solve_before_solution =
        chain.solve_generation < chain.solution_generation;
    const bool solution_before_recovery =
        chain.solution_generation < chain.recovery_generation;
    const bool previous_recovery_before_local = index == 0 ||
        produced.linearization_chains[index - 1].recovery_generation <
            chain.local_generation;
    const bool previous_candidate_before_candidate = index == 0 ||
        produced.linearization_chains[index - 1].candidate_generation <
            chain.candidate_generation;
    const bool factor_setup_census_current = chain.factor_setup_current &&
        chain.matrix_refresh_generation != 0 &&
        chain.matrix_refresh_generation == chain.factor_setup_generation;
    output << "rank " << rank << ": independent_chain[" << index << ']'
           << " residual_equal=" << residual_equal
           << " matrix_equal=" << matrix_equal
           << " factor_equal=" << factor_equal
           << " solution_equal=" << solution_equal
           << " candidate_equal=" << candidate_equal
           << " oracle_hashes_nonzero="
           << (HashNonZero(rebuilt.residual) && HashNonZero(rebuilt.matrix) &&
               HashNonZero(rebuilt.factor) && HashNonZero(rebuilt.solution) &&
               HashNonZero(rebuilt.candidate))
           << " production_hashes_nonzero="
           << (HashNonZero(chain.residual_hash.bytes) &&
               HashNonZero(chain.matrix_hash.bytes) &&
               HashNonZero(chain.factor_hash.bytes) &&
               HashNonZero(chain.solution_hash.bytes) &&
               HashNonZero(chain.candidate_state_hash.bytes))
           << " ordered=" << chain.ordered
           << " affine=" << chain.affine_consumption_verified
           << " solve_before_solution=" << solve_before_solution
           << " solution_before_recovery=" << solution_before_recovery
           << " previous_recovery_before_local="
           << previous_recovery_before_local
           << " previous_candidate_before_candidate="
           << previous_candidate_before_candidate
           << " factor_setup_current=" << chain.factor_setup_current
           << " factor_setup_census_current="
           << factor_setup_census_current
           << " matrix_refresh_generation="
           << chain.matrix_refresh_generation
           << " factor_setup_generation=" << chain.factor_setup_generation
           << " true_relative_residual=" << chain.true_relative_residual
           << " newton_update=" << chain.newton_update
           << " accepted_trial_index=" << chain.accepted_trial_index
           << " captured_sizes={trace_increment:"
           << chain.true_trace_increment_size
           << ",candidate_interior:"
           << chain.accepted_candidate_interior_size
           << ",candidate_trace:"
           << chain.accepted_candidate_true_trace_size << "}"
           << " generations={local:" << chain.local_generation
           << ",condensed:" << chain.condensed_generation
           << ",solve:" << chain.solve_generation
           << ",solution:" << chain.solution_generation
           << ",recovery:" << chain.recovery_generation
           << ",candidate:" << chain.candidate_generation << "}\n"
           << "  oracle.residual=" << HashHex(rebuilt.residual) << '\n'
           << "  produced.residual=" << HashHex(chain.residual_hash.bytes)
           << '\n'
           << "  oracle.matrix=" << HashHex(rebuilt.matrix) << '\n'
           << "  produced.matrix=" << HashHex(chain.matrix_hash.bytes) << '\n'
           << "  oracle.factor=" << HashHex(rebuilt.factor) << '\n'
           << "  produced.factor=" << HashHex(chain.factor_hash.bytes) << '\n'
           << "  oracle.solution=" << HashHex(rebuilt.solution) << '\n'
           << "  produced.solution=" << HashHex(chain.solution_hash.bytes)
           << '\n'
           << "  oracle.candidate=" << HashHex(rebuilt.candidate) << '\n'
           << "  produced.candidate="
           << HashHex(chain.candidate_state_hash.bytes) << '\n';
  }
  for (std::size_t index = common;
       index < std::max(oracle_count, produced_count); ++index)
  {
    output << "rank " << rank << ": independent_chain[" << index << ']'
           << " oracle_present=" << (index < oracle_count)
           << " produced_present=" << (index < produced_count) << '\n';
  }

  const std::size_t trajectory_count = std::min(
      oracle.newton_steps.size(), produced_count);
  for (std::size_t index = 0; index < trajectory_count; ++index)
  {
    const dg_igr_hdg_cpu_test::IndependentNewtonStepEvidence& step =
        oracle.newton_steps[index];
    const CpuLinearizationChain& chain =
        produced.linearization_chains[index];
    const VectorComparison trace_increment = CompareCaptured(
        chain.true_trace_increment, chain.true_trace_increment_size,
        step.true_trace_increment);
    const VectorComparison candidate_interior = CompareCaptured(
        chain.accepted_candidate_interior,
        chain.accepted_candidate_interior_size,
        step.accepted_candidate_interior);
    const VectorComparison candidate_trace = CompareCaptured(
        chain.accepted_candidate_true_trace,
        chain.accepted_candidate_true_trace_size,
        step.accepted_candidate_true_trace);
    int attempted_trials = 0;
    int accepted_trials = 0;
    bool ordered_trial_indices = false;
    const CpuTrialEvidence* trial = AcceptedProductionTrial(
        produced, static_cast<int>(index), attempted_trials,
        accepted_trials, ordered_trial_indices);
    output << "rank " << rank << ": independent_trajectory[" << index << ']'
           << " oracle_update=" << step.update_index
           << " oracle_attempted_trials=" << step.attempted_trials
           << " produced_attempted_trials=" << attempted_trials
           << " accepted_trials=" << accepted_trials
           << " ordered_trial_indices=" << ordered_trial_indices
           << " trace_increment={size_equal:" << trace_increment.size_equal
           << ",finite:" << trace_increment.finite
           << ",close:" << trace_increment.close
           << ",production_norm:" << trace_increment.production_norm
           << ",oracle_norm:" << trace_increment.oracle_norm
           << ",difference_norm:" << trace_increment.difference_norm
           << ",relative:" << trace_increment.relative_difference
           << ",max_difference:" << trace_increment.maximum_difference
           << ",scale:" << trace_increment.infinity_scale << "}"
           << " candidate_interior={size_equal:"
           << candidate_interior.size_equal
           << ",finite:" << candidate_interior.finite
           << ",close:" << candidate_interior.close
           << ",production_norm:" << candidate_interior.production_norm
           << ",oracle_norm:" << candidate_interior.oracle_norm
           << ",difference_norm:" << candidate_interior.difference_norm
           << ",relative:" << candidate_interior.relative_difference
           << ",max_difference:" << candidate_interior.maximum_difference
           << ",scale:" << candidate_interior.infinity_scale << "}"
           << " candidate_trace={size_equal:" << candidate_trace.size_equal
           << ",finite:" << candidate_trace.finite
           << ",close:" << candidate_trace.close
           << ",production_norm:" << candidate_trace.production_norm
           << ",oracle_norm:" << candidate_trace.oracle_norm
           << ",difference_norm:" << candidate_trace.difference_norm
           << ",relative:" << candidate_trace.relative_difference
           << ",max_difference:" << candidate_trace.maximum_difference
           << ",scale:" << candidate_trace.infinity_scale << "}";
    if (trial)
    {
      const double oracle_armijo_boundary =
          (1.0 - independent_armijo_constant * step.accepted_alpha) *
          step.base_residual;
      const double produced_armijo_boundary =
          (1.0 - independent_armijo_constant * trial->alpha) *
          trial->base_residual;
      output << " oracle_trial=" << step.accepted_trial_index
             << " produced_trial=" << trial->trial_index
             << " oracle_alpha=" << step.accepted_alpha
             << " produced_alpha=" << trial->alpha
             << " alpha_difference="
             << std::abs(step.accepted_alpha - trial->alpha)
             << " base_oracle=" << step.base_residual
             << " base_produced=" << trial->base_residual
             << " base_difference="
             << std::abs(step.base_residual - trial->base_residual)
             << " base_relative_difference="
             << RelativeScalarDifference(step.base_residual,
                                         trial->base_residual)
             << " base_close="
             << CloseTrajectoryScalar(step.base_residual,
                                      trial->base_residual)
             << " candidate_oracle=" << step.accepted_candidate_residual
             << " candidate_produced=" << trial->trial_residual
             << " candidate_difference="
             << std::abs(step.accepted_candidate_residual -
                         trial->trial_residual)
             << " candidate_relative_difference="
             << RelativeScalarDifference(step.accepted_candidate_residual,
                                         trial->trial_residual)
             << " candidate_close="
             << CloseTrajectoryScalar(step.accepted_candidate_residual,
                                      trial->trial_residual)
             << " oracle_true_relative=" << step.true_relative_residual
             << " produced_chain_true_relative="
             << chain.true_relative_residual
             << " true_relative_difference="
             << std::abs(step.true_relative_residual -
                         chain.true_relative_residual)
             << " produced_maximum_true_relative="
             << produced.maximum_true_relative_residual
             << " oracle_true_gate="
             << (std::isfinite(step.true_relative_residual) &&
                 step.true_relative_residual <=
                     independent_true_residual_limit)
             << " produced_true_gate="
             << (std::isfinite(produced.maximum_true_relative_residual) &&
                 produced.maximum_true_relative_residual <=
                     independent_true_residual_limit)
             << " oracle_armijo=" << step.armijo_satisfied
             << " oracle_armijo_boundary=" << oracle_armijo_boundary
             << " produced_armijo="
             << (trial->accepted &&
                 trial->trial_residual <= trial->armijo_boundary)
             << " produced_recorded_armijo_boundary="
             << trial->armijo_boundary
             << " produced_rebuilt_armijo_boundary="
             << produced_armijo_boundary;
    }
    else
    {
      output << " produced_accepted_trial=missing";
    }
    output << '\n';
  }
  std::cerr << output.str();
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

struct CpuCheckpointObservation
{
  std::array<unsigned char, sizeof(CheckpointDescriptor)> descriptor_bytes{};
  std::array<unsigned char,
             sizeof(std::array<CpuOwnedAllocationEvent, 4>)> event_bytes{};
  std::vector<double> interior;
  std::vector<double> trace;
  const double* interior_data = nullptr;
  const double* trace_data = nullptr;
  std::size_t interior_capacity = 0;
  std::size_t trace_capacity = 0;
  std::size_t event_count = 0;
  bool valid = false;
};

CpuCheckpointObservation ObserveCheckpoint(const CpuCheckpoint& checkpoint)
{
  CpuCheckpointObservation observation;
  std::memcpy(observation.descriptor_bytes.data(), &checkpoint.descriptor,
              sizeof(checkpoint.descriptor));
  std::memcpy(observation.event_bytes.data(),
              checkpoint.owned_allocation_events.data(),
              sizeof(checkpoint.owned_allocation_events));
  const CpuCoefficientView interior = checkpoint.interior();
  const CpuCoefficientView trace = checkpoint.trace();
  observation.interior = CopyView(interior);
  observation.trace = CopyView(trace);
  observation.interior_data = interior.data;
  observation.trace_data = trace.data;
  observation.interior_capacity = interior.capacity;
  observation.trace_capacity = trace.capacity;
  observation.event_count = checkpoint.owned_allocation_event_count;
  observation.valid = checkpoint.valid;
  return observation;
}

bool CheckpointUnchanged(const CpuCheckpoint& checkpoint,
                         const CpuCheckpointObservation& before)
{
  const CpuCoefficientView interior = checkpoint.interior();
  const CpuCoefficientView trace = checkpoint.trace();
  return checkpoint.valid == before.valid &&
      checkpoint.owned_allocation_event_count == before.event_count &&
      std::memcmp(&checkpoint.descriptor, before.descriptor_bytes.data(),
                  sizeof(checkpoint.descriptor)) == 0 &&
      std::memcmp(checkpoint.owned_allocation_events.data(),
                  before.event_bytes.data(),
                  sizeof(checkpoint.owned_allocation_events)) == 0 &&
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

void SeedInvalidCheckpoint(CpuCheckpoint& checkpoint,
                           const CheckpointDescriptor& descriptor)
{
  checkpoint.descriptor = descriptor;
  checkpoint.descriptor.cursor ^= std::uint64_t{0x5a5a};
  checkpoint.owned_allocation_event_count = 1;
  checkpoint.owned_allocation_events[0] = {
      CpuOwnedAllocationSite::slot_cell, 77, 88, 8, 11, 99};
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
void CheckTwoPhaseAndAffineAuthorities(Checks& checks)
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
  CpuPreparedStep moved_staging{std::move(first_staging)};
  CpuPreparedCommit rejected;
  checks.Require(!first.PrepareCommit(first_staging, rejected) &&
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
                     imported.owned_allocation_event_count == 3 &&
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
                     initial.storage.sparse_owned_term_capacity > 0 &&
                     initial.storage.true_diagonal_nnz > 0 &&
                     initial.storage.sparse_owned_term_token != 0 &&
                     initial.storage.topology_generation == 1 &&
                     initial.storage.storage_generation == 1 &&
                     initial.storage.allocation_baseline > 0 &&
                     initial.storage.owned_allocation_event_count ==
                         initial.storage.allocation_baseline &&
                     initial.storage.selected_owned_allocation_stream_complete &&
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
  for (std::size_t index = 0;
       index < initial.storage.owned_allocation_event_count; ++index)
  {
    const CpuOwnedAllocationEvent& event =
        initial.storage.owned_allocation_events[index];
    checks.Require(event.ordinal == index + 1 && event.bytes > 0 &&
                       event.alignment > 0 && event.capacity > 0 &&
                       event.backing_token != 0,
                   "observed W8 allocation event metadata is exact");
  }

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
  const CpuAttemptEvidence first_accepted =
      adapter.evidence().accepted.attempt;
  bool independent_chain = oracle.passed &&
      oracle.linearizations.size() == oracle.newton_steps.size() &&
      oracle.linearizations.size() ==
          first_accepted.linearization_chain_count &&
      oracle.linearizations.size() ==
          static_cast<std::size_t>(oracle.newton_updates) &&
      oracle.linearizations.size() ==
          static_cast<std::size_t>(first_accepted.nonlinear_updates) &&
      oracle.linearizations.size() > 1 &&
      first_accepted.work.complete_factor_setups ==
          first_accepted.work.complete_trace_solves &&
      first_accepted.work.complete_factor_setups ==
          first_accepted.work.accepted_newton_updates &&
      first_accepted.work.complete_factor_setups ==
          first_accepted.linearization_chain_count &&
      first_accepted.finite_gate && first_accepted.admissibility_gate &&
      first_accepted.recovery_gate && first_accepted.nonlinear_gate &&
      first_accepted.true_residual_gate &&
      std::isfinite(first_accepted.maximum_true_relative_residual) &&
      first_accepted.maximum_true_relative_residual <=
          independent_true_residual_limit;
  bool localized_initial_inputs_exact =
      !oracle.linearizations.empty() &&
      first_accepted.linearization_chain_count != 0;
  bool stale_identity_setup_rejected = localized_initial_inputs_exact;
  std::uint64_t prior_matrix_refresh_generation = 0;
  std::uint64_t prior_factor_setup_generation = 0;
  const std::size_t comparison_count = std::min(
      {oracle.linearizations.size(), oracle.newton_steps.size(),
       first_accepted.linearization_chain_count});
  for (std::size_t index = 0; index < comparison_count; ++index)
  {
    const dg_igr_hdg_cpu_test::IndependentLinearizationHashes& rebuilt =
        oracle.linearizations[index];
    const dg_igr_hdg_cpu_test::IndependentNewtonStepEvidence& step =
        oracle.newton_steps[index];
    const CpuLinearizationChain& produced =
        first_accepted.linearization_chains[index];
    int attempted_trials = 0;
    int accepted_trials = 0;
    bool ordered_trial_indices = false;
    const CpuTrialEvidence* trial = AcceptedProductionTrial(
        first_accepted, static_cast<int>(index), attempted_trials,
        accepted_trials, ordered_trial_indices);
    const VectorComparison trace_increment = CompareCaptured(
        produced.true_trace_increment, produced.true_trace_increment_size,
        step.true_trace_increment);
    const VectorComparison candidate_interior = CompareCaptured(
        produced.accepted_candidate_interior,
        produced.accepted_candidate_interior_size,
        step.accepted_candidate_interior);
    const VectorComparison candidate_trace = CompareCaptured(
        produced.accepted_candidate_true_trace,
        produced.accepted_candidate_true_trace_size,
        step.accepted_candidate_true_trace);
    const bool hashes_retained = HashNonZero(rebuilt.residual) &&
        HashNonZero(rebuilt.matrix) && HashNonZero(rebuilt.factor) &&
        HashNonZero(rebuilt.solution) && HashNonZero(rebuilt.candidate) &&
        HashNonZero(produced.residual_hash.bytes) &&
        HashNonZero(produced.matrix_hash.bytes) &&
        HashNonZero(produced.factor_hash.bytes) &&
        HashNonZero(produced.solution_hash.bytes) &&
        HashNonZero(produced.candidate_state_hash.bytes);
    const bool factor_setup_current = produced.factor_setup_current &&
        produced.matrix_refresh_generation > prior_matrix_refresh_generation &&
        produced.factor_setup_generation > prior_factor_setup_generation &&
        produced.matrix_refresh_generation ==
            produced.factor_setup_generation;
    const bool trial_association = trial &&
        produced.newton_update == static_cast<int>(index) &&
        step.update_index == static_cast<int>(index) &&
        produced.accepted_trial_index == step.accepted_trial_index &&
        trial->newton_iteration == static_cast<int>(index) &&
        trial->trial_index == step.accepted_trial_index &&
        attempted_trials == step.attempted_trials &&
        attempted_trials == step.accepted_trial_index + 1 &&
        accepted_trials == 1 && ordered_trial_indices &&
        trial->alpha == step.accepted_alpha &&
        trial->alpha == std::ldexp(1.0, -trial->trial_index);
    const bool trajectory_scalars = trial &&
        CloseTrajectoryScalar(step.base_residual, trial->base_residual) &&
        CloseTrajectoryScalar(step.accepted_candidate_residual,
                              trial->trial_residual);
    const bool independent_armijo = step.armijo_satisfied &&
        step.accepted_candidate_residual <=
            (1.0 - independent_armijo_constant * step.accepted_alpha) *
                step.base_residual &&
        step.accepted_candidate_residual < step.base_residual;
    const bool production_armijo = trial && trial->accepted &&
        trial->prescreen_passed && trial->reassembled &&
        trial->trial_residual <= trial->armijo_boundary &&
        trial->armijo_boundary ==
            (1.0 - independent_armijo_constant * trial->alpha) *
                trial->base_residual &&
        trial->trial_residual < trial->base_residual;
    const bool true_residuals =
        std::isfinite(step.true_relative_residual) &&
        step.true_relative_residual <= independent_true_residual_limit &&
        std::isfinite(produced.true_relative_residual) &&
        produced.true_relative_residual <= independent_true_residual_limit;
    localized_initial_inputs_exact = localized_initial_inputs_exact &&
        (index != 0 ||
         (produced.residual_hash.bytes == rebuilt.residual &&
          produced.matrix_hash.bytes == rebuilt.matrix &&
          produced.factor_hash.bytes == rebuilt.factor));
    stale_identity_setup_rejected = stale_identity_setup_rejected &&
        (index != 0 || (factor_setup_current && trace_increment.close));
    independent_chain = independent_chain && hashes_retained &&
        factor_setup_current &&
        trace_increment.close && candidate_interior.close &&
        candidate_trace.close && trial_association && trajectory_scalars &&
        independent_armijo && production_armijo && true_residuals &&
        produced.ordered && produced.affine_consumption_verified &&
        produced.solve_generation < produced.solution_generation &&
        produced.solution_generation < produced.recovery_generation &&
        (index == 0 ||
         first_accepted.linearization_chains[index - 1]
                 .recovery_generation < produced.local_generation) &&
        (index == 0 ||
         first_accepted.linearization_chains[index - 1]
                 .candidate_generation < produced.candidate_generation);
    prior_matrix_refresh_generation = produced.matrix_refresh_generation;
    prior_factor_setup_generation = produced.factor_setup_generation;
  }
  checks.Require(localized_initial_inputs_exact,
                 "chain-0 residual/matrix/factor-input hashes localize inputs");
  checks.Require(stale_identity_setup_rejected,
                 "current Schur setup and increment reject identity-primed reuse");
  if (!independent_chain || !localized_initial_inputs_exact ||
      !stale_identity_setup_rejected)
  {
    ReportIndependentChainDiagnostics(oracle, first_accepted, rank);
  }
  checks.Require(independent_chain,
                 "independent numerical Newton chains match with provenance hashes");
  checks.Require(NearTime(adapter.coordinates().time, 0.05) &&
                     adapter.coordinates().step == 1 &&
                     adapter.coordinates().generation == 1 &&
                     adapter.coordinates().cursor == 1,
                 "step 1 coordinates");
  CpuCheckpoint checkpoint;
  checks.Require(adapter.CaptureCheckpoint(checkpoint) && checkpoint.valid &&
                     checkpoint.descriptor.coordinates.step == 1 &&
                     checkpoint.owned_allocation_event_count == 3,
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
  bool chain_coherent = accepted.linearization_chain_count ==
      static_cast<std::size_t>(accepted.nonlinear_updates) &&
      accepted.linearization_chain_count > 1 &&
      accepted.retained_linearization_chain + 1 ==
          accepted.linearization_chain_count;
  std::uint64_t prior_generation = 0;
  std::uint64_t prior_candidate_generation = 0;
  std::uint64_t terminal_prior_factor_setup_generation = 0;
  for (std::size_t index = 0;
       index < accepted.linearization_chain_count; ++index)
  {
    const CpuLinearizationChain& chain = accepted.linearization_chains[index];
    chain_coherent = chain_coherent && chain.ordered &&
        chain.affine_consumption_verified &&
        chain.factor_setup_current &&
        chain.matrix_refresh_generation == chain.factor_setup_generation &&
        chain.factor_setup_generation >
            terminal_prior_factor_setup_generation &&
        chain.true_trace_increment_size ==
            terminal.local_owned_true_trace_coefficients &&
        chain.accepted_candidate_interior_size ==
            terminal.local_owned_interior_coefficients &&
        chain.accepted_candidate_true_trace_size ==
            terminal.local_owned_true_trace_coefficients &&
        chain.local_generation > prior_generation &&
        chain.local_generation < chain.condensed_generation &&
        chain.condensed_generation < chain.solve_generation &&
        chain.solve_generation < chain.solution_generation &&
        chain.solution_generation < chain.recovery_generation &&
        chain.candidate_generation > prior_candidate_generation &&
        chain.residual_hash != StateHash{} && chain.matrix_hash != StateHash{} &&
        chain.factor_hash != StateHash{} && chain.solution_hash != StateHash{} &&
        chain.candidate_state_hash != StateHash{} &&
        (chain.retained_for_publication ==
         (index == accepted.retained_linearization_chain));
    prior_generation = chain.recovery_generation;
    prior_candidate_generation = chain.candidate_generation;
    terminal_prior_factor_setup_generation = chain.factor_setup_generation;
  }
  const CpuLinearizationChain& retained =
      accepted.linearization_chains[accepted.retained_linearization_chain];
  const bool products_bind_retained_chain =
      accepted.products[0].payload_hash == retained.residual_hash &&
      accepted.products[0].linearization_generation ==
          retained.local_generation &&
      accepted.products[1].payload_hash == retained.matrix_hash &&
      accepted.products[1].linearization_generation ==
          retained.condensed_generation &&
      accepted.products[2].payload_hash == retained.factor_hash &&
      accepted.products[2].linearization_generation ==
          retained.solve_generation &&
      accepted.products[3].payload_hash == retained.solution_hash &&
      accepted.products[3].linearization_generation ==
          retained.solution_generation;
  checks.Require(accepted.work.complete_assemblies ==
                         1 + accepted.work.trial_reassemblies &&
                     accepted.work.complete_condensations ==
                         accepted.work.accepted_newton_updates &&
                     accepted.work.complete_factor_setups ==
                         accepted.work.complete_trace_solves &&
                     accepted.work.complete_factor_setups ==
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
                                           accepted.candidate_cursor &&
                                       product.candidate_generation ==
                                           accepted.candidate_generation &&
                                       product.candidate_state_hash ==
                                           accepted.linearization_chains[
                                               accepted
                                                   .retained_linearization_chain]
                                               .candidate_state_hash;
                                 }) && chain_coherent &&
                     products_bind_retained_chain,
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
                     terminal.storage.sparse_owned_term_capacity ==
                         initial.storage.sparse_owned_term_capacity &&
                     terminal.storage.sparse_outgoing_term_capacity ==
                         initial.storage.sparse_outgoing_term_capacity &&
                     terminal.storage.sparse_incoming_slot_capacity ==
                         initial.storage.sparse_incoming_slot_capacity &&
                     terminal.storage.sparse_owned_term_token ==
                         initial.storage.sparse_owned_term_token &&
                     terminal.storage.sparse_outgoing_term_token ==
                         initial.storage.sparse_outgoing_term_token &&
                     terminal.storage.sparse_incoming_slot_token ==
                         initial.storage.sparse_incoming_slot_token &&
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
  CheckCheckpointImportAndIndependence(
      adapter, checkpoint, checks, communicator, rank, ranks);
  CheckProductNegatives<Selection>(checks);
  CheckPhaseRankInjection<Selection>(checks, ranks);
  CheckOuterStageAbort<Selection>(checks, ranks);
  CheckTwoPhaseAndAffineAuthorities<Selection>(checks);

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
