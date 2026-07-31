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

} // namespace dg_igr_hdg_sc25_cpu_test
