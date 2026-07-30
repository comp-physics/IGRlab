#pragma once

#include <mpi.h>

#include <array>
#include <cstdint>
#include <vector>

namespace dg_igr_hdg_cpu_test
{

struct IndependentOracleConfig
{
  int polynomial_order = 1;
  int elements_per_direction = 3;
  int assembly_quadrature_order = 11;
};

struct IndependentLinearizationHashes
{
  std::array<std::uint8_t, 32> residual{};
  std::array<std::uint8_t, 32> matrix{};
  std::array<std::uint8_t, 32> factor{};
  std::array<std::uint8_t, 32> solution{};
  std::array<std::uint8_t, 32> candidate{};
};

struct IndependentNewtonStepEvidence
{
  int update_index = -1;
  int attempted_trials = 0;
  int accepted_trial_index = -1;
  double base_residual = 0.0;
  double accepted_alpha = 0.0;
  double accepted_candidate_residual = 0.0;
  double true_relative_residual = 0.0;
  std::vector<double> true_trace_increment;
  std::vector<double> accepted_candidate_interior;
  std::vector<double> accepted_candidate_true_trace;
  bool armijo_satisfied = false;
};

struct IndependentOracleResult
{
  std::vector<double> initial_interior;
  std::vector<double> initial_true_trace;
  std::vector<double> interior;
  std::vector<double> true_trace;
  std::vector<IndependentLinearizationHashes> linearizations;
  std::vector<IndependentNewtonStepEvidence> newton_steps;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double maximum_true_residual = 0.0;
  int newton_updates = 0;
  std::uint64_t accepted_step = 0;
  double accepted_time = 0.0;
  std::uint64_t negative_signed_trace_dofs = 0;
  bool mesh_rebuilt = false;
  bool analytic_source_flux_jacobian = false;
  bool signed_map_extent_valid = false;
  bool signed_map_entries_valid = false;
  bool signed_map_permutation_rebuilt = false;
  bool signed_map_used = false;
  bool signed_maps_rebuilt = false;
  bool local_schur_rebuilt = false;
  bool global_ptsp_rebuilt = false;
  bool true_residual_rebuilt = false;
  bool distribute_rebuilt = false;
  bool recovery_rebuilt = false;
  bool admissible = false;
  bool passed = false;
};

/**
 * Independent test-local one-step oracle starting from the supplied previous
 * accepted state. Step one uses the frozen analytic/perturbed initial pair and
 * therefore requires empty previous-state vectors.
 */
bool RunIndependentStep(
    MPI_Comm communicator,
    const IndependentOracleConfig& config,
    std::uint64_t accepted_step,
    const std::vector<double>& previous_interior,
    const std::vector<double>& previous_true_trace,
    IndependentOracleResult& result);

/** Independent test-local first-step oracle; no production helpers. */
bool RunIndependentFirstStep(MPI_Comm communicator,
                             const IndependentOracleConfig& config,
                             IndependentOracleResult& result);

/** Retained accepted p1/N3 smoke entry. */
bool RunIndependentFirstStep(MPI_Comm communicator,
                             IndependentOracleResult& result);

} // namespace dg_igr_hdg_cpu_test
