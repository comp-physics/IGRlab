#pragma once

/**
 * @file numerics/hdg/euler/smooth_euler_device_full.hpp
 * @brief Default-off conventional single-rank CUDA HDG Euler transaction.
 *
 * This verified-experimental interface owns a complete local-assembly,
 * condensation, MFEM/HYPRE trace-solve, recovery, admissibility, and state
 * update transaction.  It is not a supported/default solver, an MPI-device
 * path, or a performance claim.
 */

#include "numerics/hdg/euler/smooth_euler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace dg::hdg::euler
{
enum class FullDeviceStatus : unsigned char
{
  success,
  default_disabled,
  setup_failure,
  projection_failure,
  initial_admissibility_failure,
  assembly_failure,
  condensation_failure,
  trace_assembly_failure,
  trace_solve_failure,
  linear_residual_failure,
  recovery_failure,
  line_search_exhausted,
  iteration_limit,
  error_integration_failure,
  rate_failure,
  injected_failure,
  rollback_failure,
  residency_failure,
  non_finite
};

enum class FullDevicePhase : unsigned char
{
  setup,
  local_assembly,
  condensation,
  trace_assembly,
  trace_solve,
  recovery,
  state_update,
  count
};

struct FullDevicePhaseTelemetry
{
  FullDevicePhase phase = FullDevicePhase::setup;
  FullDeviceStatus status = FullDeviceStatus::setup_failure;
  std::uint64_t generation = 0;
  std::size_t persistent_bytes = 0;
  std::size_t allocation_events = 0;
  std::size_t hypre_refresh_endpoint_growth_events = 0;
  std::size_t hypre_refresh_endpoint_growth_bytes = 0;
  std::size_t hypre_persistent_bytes = 0;
  std::size_t hypre_setup_refresh_events = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_bytes = 0;
  std::size_t synchronization_events = 0;
  std::size_t fallback_events = 0;
  std::size_t managed_device_objects_checked = 0;
  std::size_t managed_device_objects_valid = 0;
  std::size_t app_forall_submissions = 0;
  std::size_t app_forall_device_tickets = 0;
  std::size_t app_forall_host_fallbacks = 0;
  bool reached = false;
  bool device_resident = false;
  bool contemporaneous_audit = false;
  bool app_forall_census_complete = false;
  bool mfem_internal_backend_observable = false;
  bool hypre_internal_backend_observable = false;
};

struct FullDeviceTransferTelemetry
{
  std::size_t persistent_bytes = 0;
  std::size_t allocation_events = 0;
  std::size_t hypre_refresh_endpoint_growth_events = 0;
  std::size_t hypre_refresh_endpoint_growth_bytes = 0;
  std::size_t hypre_persistent_bytes = 0;
  std::size_t hypre_setup_refresh_events = 0;
  std::size_t hypre_shape_warmup_setup_events = 0;
  std::size_t hypre_ilu_configuration_checks = 0;
  std::size_t allocator_snapshot_events = 0;
  std::size_t external_zero_allocation_intervals_marked = 0;
  std::size_t external_allocation_churn_intervals_marked = 0;
  std::size_t external_zero_allocation_interval_sequence = 0;
  std::size_t mfem_allocator_high_watermark_bytes = 0;
  std::size_t hypre_allocator_high_watermark_bytes = 0;
  std::size_t managed_device_objects_checked = 0;
  std::size_t managed_device_objects_valid = 0;
  std::size_t phase_residency_audit_events = 0;
  std::size_t backend_selection_checks = 0;
  std::size_t backend_path_audit_events = 0;
  std::size_t solver_configuration_checks = 0;
  std::size_t host_to_device_bytes = 0;
  std::size_t device_to_host_bytes = 0;
  std::size_t test_only_device_to_host_bytes = 0;
  std::size_t synchronization_events = 0;
  std::size_t fallback_events = 0;
  std::size_t dense_host_trace_inverse_events = 0;
  std::size_t solver_substitution_events = 0;
  std::size_t app_forall_submissions = 0;
  std::size_t app_forall_device_tickets = 0;
  std::size_t app_forall_host_fallbacks = 0;
  std::size_t app_forall_rejected_submissions = 0;
  bool allocator_endpoint_census_complete = false;
  bool whole_transaction_zero_allocation_claimed = false;
  bool hypre_setup_refresh_allocation_churn_reported = false;
  bool external_umpire_event_audit_required = true;
  bool external_umpire_event_results_available_in_process = false;
  bool marked_zero_allocation_scope_excludes_trace_assembly_solve = true;
  bool app_forall_fallback_census_complete = false;
  bool mfem_internal_backend_observable = false;
  bool hypre_internal_backend_observable = false;
  bool fallback_census_complete = false;
  bool cuda_kernel_execution_verified = false;
};

/** Optional, explicitly requested final-boundary data for independent tests. */
struct FullDeviceTestSnapshot
{
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  int global_trace_dofs = 0;
  int assembly_quadrature_order = 0;
  std::vector<int> signed_trace_dofs;
  std::vector<double> old_interior;
  std::vector<double> initial_interior;
  std::vector<double> initial_trace;
  std::vector<double> final_interior;
  std::vector<double> final_trace;
  std::vector<double> first_a;
  std::vector<double> first_b;
  std::vector<double> first_c;
  std::vector<double> first_d;
  std::vector<double> first_f;
  std::vector<double> first_g;
  std::vector<double> first_schur;
  std::vector<double> first_condensed_rhs;
  std::vector<double> first_local_trace_increment;
  std::vector<double> first_recovered_increment;
  std::vector<double> global_residual_r;
  std::vector<double> global_residual_g;
  std::vector<double> global_a;
  std::vector<double> global_b;
  std::vector<double> global_c;
  std::vector<double> global_d;
  std::vector<double> global_f;
  std::vector<double> global_g;
  std::vector<double> global_schur;
  std::vector<double> global_condensed_rhs;
  std::vector<double> global_trace_increment;
  std::vector<double> global_local_trace_increment;
  std::vector<double> global_recovered_increment;
  std::vector<int> actual_true_csr_offsets;
  std::vector<int> actual_true_csr_columns;
  std::vector<double> actual_true_csr_values;
  std::vector<double> actual_true_rhs;
  std::vector<double> final_residual_r;
  std::vector<double> final_residual_g;
  std::vector<double> last_base_interior;
  std::vector<double> last_base_trace;
  std::vector<double> last_delta_interior;
  std::vector<double> last_delta_trace;
  std::vector<double> rollback_entry_interior;
  std::vector<double> rollback_changed_interior;
  std::vector<double> rollback_restored_interior;
  std::vector<double> rollback_entry_trace;
  std::vector<double> rollback_changed_trace;
  std::vector<double> rollback_restored_trace;
  double last_accepted_alpha = 0.0;
  bool global_linear_oracle_staged = false;
};

struct FullDeviceMeshEvidence
{
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int elements = 0;
  int interior_unknowns_per_element = 0;
  int trace_unknowns_per_element = 0;
  int local_trace_unknowns = 0;
  int true_trace_unknowns = 0;
  int nonlinear_iterations = 0;
  int linear_iterations = 0;
  int rejected_trials = 0;
  int effective_hypre_ilu_type = -1;
  int effective_hypre_ilu_level_of_fill = -1;
  int negative_trace_vdof_count = -1;
  int discarded_nonlinear_iterations = 0;
  int discarded_linear_iterations = 0;
  FullDeviceStatus status = FullDeviceStatus::setup_failure;
  FullDevicePhase failure_phase = FullDevicePhase::setup;
  FullDeviceStatus failure_phase_status = FullDeviceStatus::setup_failure;
  std::uint64_t failure_phase_generation = 0;
  std::uint64_t committed_generation = 0;
  std::uint64_t discarded_attempt_generation = 0;
  std::size_t invalidated_product_count = 0;
  std::size_t rejected_consumer_count = 0;
  double initial_residual = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double discarded_accepted_residual =
      std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  double minimum_trace_diagonal_magnitude =
      std::numeric_limits<double>::infinity();
  double maximum_trace_matrix_entry = 0.0;
  double minimum_density = std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
  double maximum_wave_speed = 0.0;
  double volume_relative_l2_qe = std::numeric_limits<double>::infinity();
  double volume_relative_l2_qe4 = std::numeric_limits<double>::infinity();
  double trace_relative_l2_qe = std::numeric_limits<double>::infinity();
  double trace_relative_l2_qe4 = std::numeric_limits<double>::infinity();
  double max_physics_finite_difference_error =
      std::numeric_limits<double>::infinity();
  double constant_state_residual = std::numeric_limits<double>::infinity();
  double local_equation_error = std::numeric_limits<double>::infinity();
  double condensed_equation_error = std::numeric_limits<double>::infinity();
  double schur_action_error = std::numeric_limits<double>::infinity();
  double trace_solution_error = std::numeric_limits<double>::infinity();
  double recovery_error = std::numeric_limits<double>::infinity();
  double state_update_error = std::numeric_limits<double>::infinity();
  bool bynodes_vdim4 = false;
  bool reversed_trace_permutation_observed = false;
  bool shared_face_orientation_action_verified = false;
  bool exact_ptslp_assembly = false;
  bool hypre_matrix_device = false;
  bool hypre_rhs_device = false;
  bool hypre_solution_device = false;
  bool hypre_policy_before_solve_device = false;
  bool hypre_policy_after_solve_device = false;
  bool hypre_gmres = false;
  bool hypre_ilu_type_zero_fill_zero = false;
  bool managed_residency_verified = false;
  bool allocation_observability_reported = false;
  bool phase_audit_append_only = false;
  bool failure_phase_audit_valid = false;
  bool app_forall_fallback_census_complete = false;
  bool mfem_internal_backend_observable = false;
  bool hypre_internal_backend_observable = false;
  bool fallback_census_complete = false;
  bool cuda_kernel_execution_verified = false;
  bool rollback_had_uncommitted_change = false;
  bool rollback_nonfinite_change_observed = false;
  bool rollback_rejected_post_failure_consumption = false;
  bool rollback_restored = false;
  bool pass = false;
  FullDeviceTransferTelemetry telemetry{};
  std::array<FullDevicePhaseTelemetry,
             static_cast<std::size_t>(FullDevicePhase::count)> phases{};
  std::vector<FullDevicePhaseTelemetry> phase_audit_log;
  FullDeviceTestSnapshot test_snapshot{};
};

struct FullDeviceConvergenceEvidence
{
  int polynomial_order = 0;
  std::array<int, 3> mesh_widths{{3, 6, 12}};
  std::array<double, 3> volume_errors{{0.0, 0.0, 0.0}};
  std::array<double, 3> trace_errors{{0.0, 0.0, 0.0}};
  std::array<double, 2> volume_rates{{0.0, 0.0}};
  std::array<double, 2> trace_rates{{0.0, 0.0}};
  double trace_three_level_slope = 0.0;
  bool pass = false;
};

struct FullDeviceRequest
{
  /** The experiment is deliberately unavailable unless explicitly enabled. */
  bool enabled = false;
  bool run_complete_ladder = true;
  bool enable_test_oracle_staging = false;
  /** Stage the bounded p1--p4/N3 all-element first-Newton oracles only. */
  bool enable_global_linear_oracle_staging = false;
  bool inject_failure_after_first_update = false;
  int injected_failure_polynomial_order = 1;
  int injected_failure_elements_per_direction = 3;
  /** Test-only probe: make the uncommitted interior state nonfinite. */
  bool inject_nonfinite_before_rollback = false;
};

struct FullDeviceEvidence
{
  FullDeviceStatus status = FullDeviceStatus::default_disabled;
  std::vector<FullDeviceMeshEvidence> cases;
  std::array<FullDeviceConvergenceEvidence, 4> convergence{};
  bool real_mfem_cuda_path = false;
  bool real_hypre_device_path = false;
  bool default_off = true;
  bool verified_experimental = true;
  bool pass = false;
};

/** Execute the frozen p1--p4 smooth-Euler device experiment. */
bool RunSmoothEulerDeviceFullExperiment(const FullDeviceRequest& request,
                                        FullDeviceEvidence& evidence);

static_assert(!std::is_polymorphic<FullDeviceMeshEvidence>::value,
              "full-device evidence must remain statically dispatched");
static_assert(!std::is_polymorphic<FullDeviceEvidence>::value,
              "full-device experiment must remain statically dispatched");
}  // namespace dg::hdg::euler
