#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dg_igr_hdg_cpu_test
{

using W8cKnownState = std::array<double, 4>;

W8cKnownState W8cCellState(double x) noexcept;
W8cKnownState W8cInterfaceMean() noexcept;

struct W8cTraceGeometryEvidence final
{
  std::uint64_t faces = 0;
  std::uint64_t face_nodes = 0;
  std::uint64_t true_dofs = 0;
  std::uint64_t element_face_visits = 0;
  std::uint64_t positive_orientations = 0;
  std::uint64_t negative_orientations = 0;
  std::uint64_t negative_face_vdofs = 0;
  std::uint64_t negative_permutation_dofs = 0;
  std::uint64_t right_nodes = 0;
  std::uint64_t left_nodes = 0;
  std::uint64_t mean_nodes = 0;
  std::uint64_t periodic_nodes = 0;
};

/**
 * Rebuild the exact periodic N8/k1 face space locally, including signed face
 * VDofs and per-element orientation permutations, and check every coherent
 * four-component true-trace state against its physical face-node geometry.
 * No production topology, projection, or initialization helper is called.
 */
bool VerifyW8cTraceGeometryAndState(
    MPI_Comm communicator, const double* true_trace,
    std::size_t true_trace_size,
    W8cTraceGeometryEvidence& evidence) noexcept;

W8cKnownState SmoothTransitionV2State(double x) noexcept;

struct SmoothTransitionProjectionEvidence final
{
  std::uint64_t elements = 0;
  std::uint64_t faces = 0;
  std::uint64_t interior_coefficients = 0;
  std::uint64_t trace_coefficients = 0;
  std::uint64_t element_face_visits = 0;
  std::uint64_t positive_orientations = 0;
  std::uint64_t negative_orientations = 0;
  std::uint64_t signed_permuted_trace_entries = 0;
  std::uint64_t permuted_trace_entries = 0;
  std::uint64_t negative_signed_trace_entries = 0;
  std::uint64_t doubly_covered_trace_dofs = 0;
  double density_integral = 0.0;
  double x_momentum_integral = 0.0;
  double y_momentum_integral = 0.0;
  double energy_integral = 0.0;
  double maximum_reflection_defect = 0.0;
  double maximum_half_period_complement_defect = 0.0;
  double maximum_y_invariance_defect = 0.0;
  double maximum_signed_trace_coefficient_defect = 0.0;
  double maximum_projected_trace_value_defect = 0.0;
};

/**
 * Independently rebuild and L2-project the frozen N8/k1 smooth transition on
 * the cell and face spaces at q=14. This oracle does not call production
 * initialization, projection, ownership, or residual helpers.
 */
bool BuildSmoothTransitionV2ProjectionOracle(
    MPI_Comm communicator, std::vector<double>& component_major_interior,
    std::vector<double>& true_trace,
    SmoothTransitionProjectionEvidence& evidence) noexcept;

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
