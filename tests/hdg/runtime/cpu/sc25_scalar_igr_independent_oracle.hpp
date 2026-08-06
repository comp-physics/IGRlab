#pragma once

#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dg_igr_hdg_sc25_cpu_test
{

struct IndependentRestrictionEntry final
{
  int element = -1;
  int local_row = -1;
  int local_face = -1;
  int component = -1;
  int oriented_face_dof = -1;
  int canonical_face_dof = -1;
  int signed_carrier_vdof = 0;
  int decoded_carrier_vdof = -1;
  int sign = 0;
  dg::hdg::GlobalIndex global_true_dof = dg::hdg::invalid_global_index;
  dg::hdg::LocalIndex local_true_dof = dg::hdg::invalid_local_index;
  dg::hdg::RankIndex owner_rank = dg::hdg::invalid_rank;
};

struct IndependentC0Evidence final
{
  int checked_elements = 0;
  int checked_restriction_entries = 0;
  int negative_signed_restriction_entries = 0;
  double maximum_trace_action_error = 0.0;
  double true_action_relative_l2 = 0.0;
  double true_rhs_relative_l2 = 0.0;
  double retained_matrix_relative_l2 = 0.0;
  std::uint64_t retained_assembly_generation = 0;
  std::vector<IndependentRestrictionEntry> restriction;
  std::vector<std::string> failures;
};

struct IndependentC1Evidence final
{
  int full_dimension = 0;
  int checked_coordinate_directions = 0;
  double residual_relative_l2 = 0.0;
  double jacobian_relative_l2 = 0.0;
  double maximum_local_block_relative_l2 = 0.0;
  double schur_relative_l2 = 0.0;
  double condensed_rhs_relative_l2 = 0.0;
  double assembled_schur_relative_l2 = 0.0;
  double assembled_rhs_relative_l2 = 0.0;
  double condensed_solve_relative_l2 = 0.0;
  double true_trace_increment_relative_l2 = 0.0;
  double restricted_increment_relative_l2 = 0.0;
  double recovered_increment_relative_l2 = 0.0;
  double full_increment_relative_l2 = 0.0;
  double maximum_directional_discrepancy = 0.0;
  double quadrature_repeat_difference = 0.0;
  std::vector<std::string> failures;
};

struct IndependentTrajectoryEvidence final
{
  int checked_steps = 0;
  int checked_attempts = 0;
  int checked_newton_iterations = 0;
  int checked_line_search_trials = 0;
  int independently_assembled_residuals = 0;
  int checked_conservation_maxima = 0;
  double maximum_integral_discrepancy = 0.0;
  double maximum_conservation_maximum_discrepancy = 0.0;
  double maximum_residual_norm_discrepancy = 0.0;
  double maximum_jacobian_action_relative_l2 = 0.0;
  double maximum_condensed_action_relative_l2 = 0.0;
  double maximum_recovery_relative_l2 = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double minimum_speed_margin = 0.0;
  std::vector<std::string> failures;
};

struct IndependentRank2TrajectoryEvidence final
{
  int checked_local_elements = 0;
  int checked_local_faces = 0;
  int checked_restriction_rows = 0;
  int checked_owner_bound_face_dofs = 0;
  int independently_owned_faces = 0;
  int independently_owned_true_dofs = 0;
  int independently_global_faces = 0;
  int independently_covered_true_dofs = 0;
  int checked_stages = 0;
  int checked_attempts = 0;
  int checked_newton_records = 0;
  int checked_trial_records = 0;
  int checked_vector_records = 0;
  int checked_checkpoint_cadence = 0;
  int independent_rank1_residual_assemblies = 0;
  double maximum_peer_relative_error = 0.0;
  double maximum_operator_relative_error = 0.0;
  double maximum_vector_relative_error = 0.0;
  double maximum_integral_relative_error = 0.0;
  double p_pt_adjoint_relative_error = 0.0;
  double maximum_q12_q14_sensitivity = 0.0;
  std::vector<std::string> failures;
};

/**
 * Independently reconstruct MFEM metadata and elementary algebra.  The oracle
 * consumes public state, never production-owned certification booleans.
 */
bool RunIndependentC0Oracle(
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        foundation,
    IndependentC0Evidence& evidence);

bool RunIndependentC1Oracle(
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        foundation,
    IndependentC1Evidence& evidence);

/** Independently reconstruct trajectory hashes, quadrature admissibility,
 *  conservation integrals, Newton/Armijo decisions, and reuse invariants. */
bool RunIndependentTrajectoryOracle(
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        foundation,
    IndependentTrajectoryEvidence& evidence);

bool EvaluateIndependentAuxiliaryResidualOracle(
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        foundation,
    const dg::hdg::runtime::cpu::sc25::PersistentTrajectoryState& state,
    int quadrature_order,
    dg::hdg::runtime::cpu::sc25::AuxiliaryResidualMetrics& metrics);

bool EvaluateIndependentRank2StateAdmissibilityOracle(
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        foundation,
    const dg::hdg::runtime::cpu::sc25::PersistentRank2TrajectoryState& state,
    int quadrature_order);

bool RunIndependentRank2TrajectoryOracle(
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        rank2,
    dg::hdg::runtime::cpu::sc25::PersistentSc25ScalarIgrCpuFoundation&
        rank1_peer,
    const dg::hdg::runtime::cpu::sc25::PersistentRank2TrajectoryCheckpoint&
        checkpoint10,
    IndependentRank2TrajectoryEvidence& evidence);

bool RehashIndependentRank2Checkpoint(
    dg::hdg::runtime::cpu::sc25::PersistentRank2TrajectoryCheckpoint&
        checkpoint) noexcept;

} // namespace dg_igr_hdg_sc25_cpu_test
