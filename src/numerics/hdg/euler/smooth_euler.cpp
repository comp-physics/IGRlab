#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace dg::hdg::euler
{
namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;

bool IsFinite(const double value) noexcept { return std::isfinite(value); }

template <std::size_t Count>
bool IsFinite(const std::array<double, Count>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return IsFinite(value); });
}

PhysicsResult CheckNormal(const std::array<double, dimension>& normal) noexcept
{
  if (!IsFinite(normal) || (normal[0] == 0.0 && normal[1] == 0.0))
  {
    return {PhysicsStatus::invalid_normal};
  }
  return {};
}

PhysicsResult Pressure(const ConservedState& state, double& pressure) noexcept
{
  if (!IsFinite(state)) { return {PhysicsStatus::non_finite}; }
  if (!(state[0] > 0.0)) { return {PhysicsStatus::invalid_density}; }
  const double kinetic = 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                         state[0];
  const double candidate = (gamma - 1.0) * (state[3] - kinetic);
  if (!IsFinite(candidate)) { return {PhysicsStatus::non_finite}; }
  if (!(candidate > 0.0)) { return {PhysicsStatus::invalid_pressure}; }
  pressure = candidate;
  return {};
}

ConservedState ManufacturedDerivative(const double x, const double y,
                                      const int direction)
{
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  PrimitiveState primitive{};
  (void)ManufacturedPrimitive(x, y, primitive);
  const double derivative_rho =
      direction == 0 ? 0.2 * pi * cx * sy : 0.2 * pi * sx * cy;
  const double derivative_u =
      direction == 0 ? -0.1 * pi * sx * sy : 0.1 * pi * cx * cy;
  const double derivative_v =
      direction == 0 ? 0.1 * pi * cx * cy : -0.1 * pi * sx * sy;
  const double derivative_p =
      direction == 0 ? -0.2 * pi * sx * cy : -0.2 * pi * cx * sy;
  const double rho = primitive[0];
  const double u = primitive[1];
  const double v = primitive[2];
  const double derivative_energy = derivative_p / (gamma - 1.0) +
      0.5 * derivative_rho * (u * u + v * v) +
      rho * (u * derivative_u + v * derivative_v);
  return {derivative_rho,
          derivative_rho * u + rho * derivative_u,
          derivative_rho * v + rho * derivative_v,
          derivative_energy};
}

template <typename T>
StridedBlockView<T> View(T* data, const LocalIndex batch, const LocalIndex rows,
                         const LocalIndex columns) noexcept
{
  return {data, batch, rows, columns, rows * columns, columns, 1};
}

bool InterpolateByNodes(const mfem::GridFunction& field,
                        const mfem::Array<int>& signed_vdofs,
                        const mfem::Vector& shape, ConservedState& state)
{
  if (signed_vdofs.Size() != conserved_components * shape.Size()) return false;
  state = {};
  for (int node = 0; node < shape.Size(); ++node)
  {
    for (int component = 0; component < conserved_components; ++component)
    {
      double sign = 1.0;
      const int vdof = DecodeSignedVDof(
          signed_vdofs[ByNodesIndex(node, shape.Size(), component)], sign);
      if (vdof < 0 || vdof >= field.Size()) return false;
      state[component] += shape[node] * sign * field[vdof];
    }
  }
  return IsFinite(state);
}

/**
 * Keeps the accepted byNODES trace layout while presenting contiguous
 * (face-node, conserved-component) blocks to MFEM's BlockILU.  This is an
 * actual similarity permutation of the assembled sparse operator, not a
 * diagonal surrogate: GMRES sees this Solver as its preconditioner.
 */
class PermutedFaceBlockILU final : public mfem::Solver
{
public:
  PermutedFaceBlockILU(const mfem::SparseMatrix& matrix,
                       std::vector<int> permuted_to_global,
                       const int face_block_size)
      : mfem::Solver(matrix.Height()), source_(&matrix),
        permuted_to_global_(std::move(permuted_to_global)),
        permuted_matrix_(matrix.Height())
  {
    if (matrix.Height() != matrix.Width() ||
        static_cast<int>(permuted_to_global_.size()) != matrix.Height() ||
        face_block_size <= 0 || matrix.Height() % face_block_size != 0)
    {
      throw std::invalid_argument("invalid face-block ILU permutation");
    }
    for (int row = 0; row < matrix.Height(); ++row)
      for (int column = 0; column < matrix.Width(); ++column)
      {
        const double value = matrix(permuted_to_global_[row],
                                    permuted_to_global_[column]);
        if (value != 0.0) { permuted_matrix_.Add(row, column, value); }
      }
    permuted_matrix_.Finalize();
    block_ilu_ = std::make_unique<mfem::BlockILU>(
        permuted_matrix_, face_block_size, mfem::BlockILU::Reordering::NONE);
  }

  void SetOperator(const mfem::Operator& op) override
  {
    if (&op != source_) { throw std::invalid_argument("face-block ILU operator changed"); }
  }

  void Mult(const mfem::Vector& input, mfem::Vector& output) const override
  {
    mfem::Vector reordered_rhs(height), reordered_solution(height);
    for (int row = 0; row < height; ++row)
    {
      reordered_rhs[row] = input[permuted_to_global_[row]];
    }
    block_ilu_->Mult(reordered_rhs, reordered_solution);
    output.SetSize(height);
    for (int row = 0; row < height; ++row)
    {
      output[permuted_to_global_[row]] = reordered_solution[row];
    }
  }

private:
  const mfem::SparseMatrix* source_;
  std::vector<int> permuted_to_global_;
  mfem::SparseMatrix permuted_matrix_;
  std::unique_ptr<mfem::BlockILU> block_ilu_;
};

template <typename Scalar>
bool HasExtents(const StridedBlockView<Scalar> view, const LocalIndex batches,
                const LocalIndex rows, const LocalIndex columns) noexcept
{
  return view.data != nullptr && view.batch_count == batches &&
         view.rows == rows && view.columns == columns &&
         view.batch_stride == rows * columns &&
         view.row_stride == columns && view.column_stride == 1;
}

struct TraceSolveDiagnostic
{
  bool primary_attempted = false;
  bool primary_result_reached = false;
  bool primary_converged = false;
  bool primary_true_residual_reached = false;
  int primary_iterations = 0;
  double primary_reported_relative_residual = 0.0;
  double primary_true_relative_residual = 0.0;
  bool correction_attempted = false;
  bool correction_result_reached = false;
  bool correction_converged = false;
  bool correction_true_residual_reached = false;
  int correction_iterations = 0;
  double correction_reported_relative_residual = 0.0;
  double correction_true_relative_residual = 0.0;
};

void AppendReachedBoolean(std::ostringstream& record, const char* key,
                          const bool reached, const bool value)
{
  record << ' ' << key << '=';
  if (reached) record << (value ? 1 : 0);
  else record << "not_reached";
}

void AppendReachedInteger(std::ostringstream& record, const char* key,
                          const bool reached, const int value)
{
  record << ' ' << key << '=';
  if (reached) record << value;
  else record << "not_reached";
}

void AppendReachedDouble(std::ostringstream& record, const char* key,
                         const bool reached, const double value)
{
  record << ' ' << key << '=';
  if (!reached) record << "not_reached";
  else if (!std::isfinite(value)) record << "invalid";
  else record << value;
}

void AppendTraceSolveDiagnostic(std::ostringstream& record,
                                const TraceSolveDiagnostic& diagnostic)
{
  record << " gmres_primary_attempted="
         << (diagnostic.primary_attempted ? 1 : 0);
  AppendReachedBoolean(record, "gmres_primary_converged",
                       diagnostic.primary_result_reached,
                       diagnostic.primary_converged);
  AppendReachedInteger(record, "gmres_primary_iterations",
                       diagnostic.primary_result_reached,
                       diagnostic.primary_iterations);
  AppendReachedDouble(record, "gmres_primary_reported_residual",
                      diagnostic.primary_result_reached,
                      diagnostic.primary_reported_relative_residual);
  AppendReachedDouble(record, "gmres_primary_true_residual",
                      diagnostic.primary_true_residual_reached,
                      diagnostic.primary_true_relative_residual);
  record << " gmres_correction_attempted="
         << (diagnostic.correction_attempted ? 1 : 0);
  AppendReachedBoolean(record, "gmres_correction_converged",
                       diagnostic.correction_result_reached,
                       diagnostic.correction_converged);
  AppendReachedInteger(record, "gmres_correction_iterations",
                       diagnostic.correction_result_reached,
                       diagnostic.correction_iterations);
  AppendReachedDouble(record, "gmres_correction_reported_residual",
                      diagnostic.correction_result_reached,
                      diagnostic.correction_reported_relative_residual);
  AppendReachedDouble(record, "gmres_correction_true_residual",
                      diagnostic.correction_true_residual_reached,
                      diagnostic.correction_true_relative_residual);
}

void AppendSolveFailureDiagnostic(std::ostringstream& record,
                                  const bool newton_reached,
                                  const SolvedMeshEvidence* solved,
                                  const TraceSolveDiagnostic& trace_diagnostic)
{
  const NewtonEvidence* newton =
      newton_reached && solved != nullptr ? &solved->newton : nullptr;
  const LinearSolveEvidence* linear =
      solved != nullptr && !solved->linear_history.empty() ?
      &solved->linear_history.back() : nullptr;
  AppendReachedInteger(record, "solve_status",
                       newton_reached && solved != nullptr,
                       solved == nullptr ? 0 : static_cast<int>(solved->status));
  AppendReachedInteger(record, "newton_status", newton != nullptr,
                       newton == nullptr ? 0 : static_cast<int>(newton->status));
  AppendReachedInteger(record, "linear_status", linear != nullptr,
                       linear == nullptr ? 0 : static_cast<int>(linear->status));
  AppendReachedInteger(record, "nonlinear_iteration", linear != nullptr,
                       linear == nullptr ? 0 : linear->nonlinear_iteration);
  AppendReachedInteger(record, "newton_accepted_updates", newton != nullptr,
                       newton == nullptr ? 0 : newton->nonlinear_iterations);
  const bool residual_reached = newton != nullptr && solved != nullptr &&
      !solved->residual_history.empty();
  const double target = residual_reached ?
      1.0e-10 * (1.0 + newton->initial_residual) : 0.0;
  AppendReachedDouble(record, "nonlinear_initial_residual", residual_reached,
                      residual_reached ? newton->initial_residual : 0.0);
  AppendReachedDouble(record, "nonlinear_final_residual", residual_reached,
                      residual_reached ? newton->final_residual : 0.0);
  AppendReachedDouble(record, "nonlinear_target", residual_reached, target);
  AppendTraceSolveDiagnostic(record, trace_diagnostic);
}

bool QuadraturePairIsConsistent(const ErrorIntegralEvidence& q_e,
                                const ErrorIntegralEvidence& q_e_plus_four)
{
  const double difference = std::abs(q_e.relative_l2 -
                                     q_e_plus_four.relative_l2);
  return std::isfinite(q_e.relative_l2) &&
         std::isfinite(q_e_plus_four.relative_l2) &&
         q_e.relative_l2 >= 1.0e-10 &&
         q_e_plus_four.relative_l2 >= 1.0e-10 &&
         difference / std::max(q_e_plus_four.relative_l2, 1.0e-300) <=
             1.0e-8 &&
         q_e_plus_four.relative_l2 > 100.0 * difference;
}

double ThreeLevelTraceSlope(const std::array<double, 3>& errors)
{
  if (!std::all_of(errors.begin(), errors.end(), [](const double value)
      { return std::isfinite(value) && value > 0.0; }))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return (std::log2(errors[0]) - std::log2(errors[2])) / 2.0;
}

void AppendRateFailureDiagnostic(std::ostringstream& record,
                                 const int polynomial_order,
                                 const ConvergenceEvidence& evidence)
{
  const std::array<double, 3>& errors = evidence.trace_relative_l2;
  const double pair_threshold = static_cast<double>(polynomial_order) + 0.45;
  const double slope_threshold = static_cast<double>(polynomial_order) + 0.50;
  std::array<bool, 3> reached{};
  std::array<bool, 3> quadrature_pair_consistent{};
  bool errors_valid = true;
  bool monotone = true;
  for (std::size_t index = 0; index < errors.size(); ++index)
  {
    reached[index] = evidence.solves[index].internal_consistency;
    quadrature_pair_consistent[index] = reached[index] &&
        QuadraturePairIsConsistent(evidence.solves[index].q_e_trace,
                                   evidence.solves[index].q_e_plus_four_trace);
    errors_valid = errors_valid && reached[index] &&
        std::isfinite(errors[index]) && errors[index] >= 1.0e-10;
    if (index > 0) monotone = monotone && errors[index - 1] > errors[index];
  }
  std::array<double, 2> rates{{
      errors_valid ? std::log2(errors[0] / errors[1]) :
          std::numeric_limits<double>::quiet_NaN(),
      errors_valid ? std::log2(errors[1] / errors[2]) :
          std::numeric_limits<double>::quiet_NaN()}};
  const double slope = errors_valid ? ThreeLevelTraceSlope(errors) :
      std::numeric_limits<double>::quiet_NaN();

  std::ostringstream failures;
  bool has_failure = false;
  const auto fail = [&](const bool failed, const char* name)
  {
    if (!failed) return;
    if (has_failure) failures << '|';
    failures << name;
    has_failure = true;
  };
  fail(!errors_valid, "trace_error_invalid");
  fail(errors_valid && !monotone, "trace_nonmonotone");
  for (std::size_t index = 0; index < quadrature_pair_consistent.size(); ++index)
  {
    fail(!quadrature_pair_consistent[index],
         index == 0 ? "trace_q_pair_0" :
         index == 1 ? "trace_q_pair_1" : "trace_q_pair_2");
  }
  for (std::size_t interval = 0; interval < rates.size(); ++interval)
  {
    fail(!std::isfinite(rates[interval]) || rates[interval] < pair_threshold,
         interval == 0 ? "trace_pair_rate_0" : "trace_pair_rate_1");
    const double volume_left = evidence.volume_relative_l2[interval];
    const double volume_right = evidence.volume_relative_l2[interval + 1];
    const double volume_rate = std::isfinite(volume_left) &&
            std::isfinite(volume_right) && volume_left > 0.0 &&
            volume_right > 0.0 ?
        std::log2(volume_left / volume_right) :
        std::numeric_limits<double>::quiet_NaN();
    fail(!std::isfinite(volume_rate) ||
             volume_rate < static_cast<double>(polynomial_order) + 0.25,
         interval == 0 ? "volume_pair_rate_0" : "volume_pair_rate_1");
  }
  fail(!std::isfinite(slope) || slope < slope_threshold,
       "trace_three_level_slope");

  for (std::size_t index = 0; index < errors.size(); ++index)
  {
    AppendReachedDouble(record,
                        index == 0 ? "trace_qE4_N3" :
                        index == 1 ? "trace_qE4_N6" : "trace_qE4_N12",
                        reached[index], errors[index]);
    record << ' ' << (index == 0 ? "trace_q_pair_N3" :
                      index == 1 ? "trace_q_pair_N6" : "trace_q_pair_N12")
           << '=' << (quadrature_pair_consistent[index] ? 1 : 0);
  }
  for (std::size_t interval = 0; interval < rates.size(); ++interval)
  {
    AppendReachedDouble(record,
                        interval == 0 ? "trace_rate_N3_N6" :
                                        "trace_rate_N6_N12",
                        errors_valid, rates[interval]);
    record << ' ' << (interval == 0 ? "trace_rate_threshold_0" :
                                      "trace_rate_threshold_1")
           << '=' << pair_threshold;
    AppendReachedDouble(record,
                        interval == 0 ? "trace_rate_margin_0" :
                                        "trace_rate_margin_1",
                        errors_valid, rates[interval] - pair_threshold);
  }
  AppendReachedDouble(record, "trace_three_level_slope", errors_valid, slope);
  record << " trace_three_level_threshold=" << slope_threshold;
  AppendReachedDouble(record, "trace_three_level_margin", errors_valid,
                      slope - slope_threshold);
  record << " rate_fail_mask="
         << (has_failure ? failures.str() : "none");
}

void EmitConvergenceFailureDiagnostic(
    const int polynomial_order, const std::size_t mesh_index,
    const int elements_per_direction, const char* phase,
    const ExperimentStatus status, const ConvergenceEvidence& evidence,
    const bool newton_reached, const TraceSolveDiagnostic& trace_diagnostic)
{
  const SolvedMeshEvidence* solved = mesh_index < evidence.solves.size() ?
      &evidence.solves[mesh_index] : nullptr;
  std::ostringstream record;
  record << std::setprecision(17)
         << "DIAG03C case=convergence phase=" << phase
         << " p=" << polynomial_order
         << " mesh_index=" << mesh_index
         << " N=" << elements_per_direction
         << " experiment_status=" << static_cast<int>(status);
  AppendSolveFailureDiagnostic(record, newton_reached, solved,
                               trace_diagnostic);
  if (std::strcmp(phase, "rate") == 0 ||
      std::strcmp(phase, "monotonicity") == 0)
  {
    AppendRateFailureDiagnostic(record, polynomial_order, evidence);
  }
  record << '\n';
  std::cerr << record.str();
}

void EmitRollbackFailureDiagnostic(
    const char* phase, const RollbackAuditEvidence& rollback,
    const NewtonEvidence* newton, const SolvedMeshEvidence* solved,
    const TraceSolveDiagnostic& trace_diagnostic)
{
  std::ostringstream record;
  record << std::setprecision(17)
         << "DIAG03C case=rollback phase=" << phase
         << " p=1 N=3 experiment_status="
         << static_cast<int>(rollback.status);
  AppendSolveFailureDiagnostic(record, newton != nullptr, solved,
                               trace_diagnostic);
  record << " rollback_status=" << static_cast<int>(rollback.status)
         << " rollback_initial_admissibility_reason="
         << static_cast<int>(rollback.initial_admissibility.reason)
         << " rollback_initial_admissibility_passed="
         << (rollback.initial_admissibility.passed ? 1 : 0)
         << " rollback_initial_minimum_density="
         << rollback.initial_admissibility.minimum_density
         << " rollback_initial_minimum_pressure="
         << rollback.initial_admissibility.minimum_pressure
         << " rollback_initial_maximum_wave_speed="
         << rollback.initial_admissibility.maximum_wave_speed
         << " rollback_accepted_updates=" << rollback.accepted_updates
         << " rollback_initial_residual=" << rollback.initial_residual
         << " rollback_residual_after_first="
         << rollback.residual_after_first_accepted
         << " rollback_residual_above_gate="
         << (rollback.residual_after_first_above_convergence ? 1 : 0)
         << " rollback_entry_interior_size=" << rollback.entry_interior.size()
         << " rollback_restored_interior_size="
         << rollback.restored_interior.size()
         << " rollback_entry_trace_size=" << rollback.entry_trace.size()
         << " rollback_restored_trace_size=" << rollback.restored_trace.size()
         << " rollback_entry_interior_hash=" << rollback.entry_interior_hash
         << " rollback_restored_interior_hash="
         << rollback.restored_interior_hash
         << " rollback_entry_trace_hash=" << rollback.entry_trace_hash
         << " rollback_restored_trace_hash=" << rollback.restored_trace_hash
         << " rollback_returned_success="
         << (rollback.returned_success ? 1 : 0)
         << " rollback_interior_byte_equal="
         << (rollback.interior_byte_equal ? 1 : 0)
         << " rollback_trace_byte_equal="
         << (rollback.trace_byte_equal ? 1 : 0)
         << " rollback_local_generation_invalid="
         << (rollback.local_generation_invalid ? 1 : 0)
         << " rollback_condensed_generation_invalid="
         << (rollback.condensed_generation_invalid ? 1 : 0)
         << " rollback_solve_generation_invalid="
         << (rollback.solve_generation_invalid ? 1 : 0)
         << " rollback_local_copy_rejected="
         << (rollback.local_copy_rejected ? 1 : 0)
         << " rollback_rhs_copy_rejected="
         << (rollback.right_hand_side_copy_rejected ? 1 : 0)
         << " rollback_trace_increment_copy_rejected="
         << (rollback.trace_increment_copy_rejected ? 1 : 0)
         << " rollback_condensation_consume_rejected="
         << (rollback.condensation_consume_rejected ? 1 : 0)
         << " rollback_solve_consume_rejected="
         << (rollback.solve_consume_rejected ? 1 : 0)
         << " rollback_recovery_consume_rejected="
         << (rollback.recovery_consume_rejected ? 1 : 0)
         << " rollback_partial_solve_evidence="
         << (rollback.partial_solve_evidence ? 1 : 0)
         << " rollback_partial_convergence_evidence="
         << (rollback.partial_convergence_evidence ? 1 : 0)
         << " rollback_passed=" << (rollback.passed ? 1 : 0)
         << '\n';
  std::cerr << record.str();
}

struct PipelinePhaseState
{
  int batches = 0, interior_dofs = 0, trace_dofs = 0, face_block_size = 0;
  StridedBlockView<double> interior_residual{}, trace_residual{};
  StridedBlockView<double> a{}, b{}, c{}, d{};
  StridedBlockView<double> assembled_system{}, assembled_rhs{}, trace_solution{};
  core::HostReferenceEngine* condensation_engine = nullptr;
  const mfem::FiniteElementSpace* trace_space = nullptr;
  const std::vector<int>* signed_trace_dofs = nullptr;
  const std::vector<int>* trace_permutation = nullptr;
  std::vector<double>* newton_f = nullptr;
  std::vector<double>* newton_g = nullptr;
  std::vector<double>* schur = nullptr;
  std::vector<double>* reduced_rhs = nullptr;
  std::vector<double>* schur_action = nullptr;
  std::vector<double>* recovered = nullptr;
  std::vector<double>* global_matrix = nullptr;
  std::unique_ptr<mfem::SparseMatrix>* trace_matrix = nullptr;
  std::unique_ptr<mfem::SparseMatrix>* trace_solve_matrix = nullptr;
  mfem::Vector* trace_rhs = nullptr;
  mfem::Vector* trace_increment = nullptr;
  std::unique_ptr<PermutedFaceBlockILU>* trace_preconditioner = nullptr;
  int gmres_iterations = 0;
  double reported_relative_residual = 0.0;
  double true_relative_residual = 0.0;
  TraceSolveDiagnostic trace_solve_diagnostic{};
  bool gmres_converged = false, linear_residual_valid = false;
  bool residual_published = false, jacobian_published = false,
       condensed = false, assembled = false, solved = false, recovered_state = false;
};

struct SmoothEulerLocalPhase
{
  PipelinePhaseState* state = nullptr;
  void EvaluateResidual(const LocalResidualRequest<SmoothEulerSpec>&,
                        LocalResidualResult<SmoothEulerSpec>& result)
  {
    result = {};
    state->residual_published =
        HasExtents(state->interior_residual, state->batches,
                   state->interior_dofs, 1) &&
        HasExtents(state->trace_residual, state->batches,
                   state->trace_dofs, 1);
    if (!state->residual_published) { return; }
    // LocalResidualResult is physical Q=(R,G), never the Newton RHS -(R,G).
    result.interior_residual = state->interior_residual;
    result.trace_residual = state->trace_residual;
  }

  void EvaluateJacobian(const LocalJacobianRequest<SmoothEulerSpec>&,
                        LocalJacobianResult<SmoothEulerSpec>& result)
  {
    result = {};
    state->jacobian_published = state->residual_published &&
        HasExtents(state->a, state->batches, state->interior_dofs,
                   state->interior_dofs) &&
        HasExtents(state->b, state->batches, state->interior_dofs,
                   state->trace_dofs) &&
        HasExtents(state->c, state->batches, state->trace_dofs,
                   state->interior_dofs) &&
        HasExtents(state->d, state->batches, state->trace_dofs,
                   state->trace_dofs);
    if (!state->jacobian_published) { return; }
    result.linearization = {state->interior_residual, state->trace_residual,
                            state->a, state->b, state->c, state->d};
  }
};

struct SmoothEulerCondensationPhase
{
  PipelinePhaseState* state = nullptr;
  void Condense(const CondenseRequest<SmoothEulerSpec>& request,
                CondenseResult<SmoothEulerSpec>& result)
  {
    result = {};
    state->condensed = false;
    const auto& linearization = request.linearization;
    if (state->condensation_engine == nullptr || state->newton_f == nullptr ||
        state->newton_g == nullptr || state->schur == nullptr ||
        state->reduced_rhs == nullptr || state->interior_dofs <= 0 ||
        state->trace_dofs <= 0 ||
        state->interior_dofs % conserved_components != 0 ||
        state->trace_dofs % conserved_components != 0 ||
        !HasExtents(linearization.interior_residual, state->batches,
                    state->interior_dofs, 1) ||
        !HasExtents(linearization.trace_residual, state->batches,
                    state->trace_dofs, 1) ||
        !HasExtents(linearization.A, state->batches, state->interior_dofs,
                    state->interior_dofs) ||
        !HasExtents(linearization.B, state->batches, state->interior_dofs,
                    state->trace_dofs) ||
        !HasExtents(linearization.C, state->batches, state->trace_dofs,
                    state->interior_dofs) ||
        !HasExtents(linearization.D, state->batches, state->trace_dofs,
                    state->trace_dofs))
    {
      return;
    }

    // This is the sole numerical boundary into HostReferenceEngine: Pipeline
    // publishes physical Q=(R,G), while Newton condensation consumes f=-R and
    // g=-G.  Recompute those RHS values from the returned phase data.
    state->newton_f->resize(static_cast<std::size_t>(state->batches) *
                            state->interior_dofs);
    state->newton_g->resize(static_cast<std::size_t>(state->batches) *
                            state->trace_dofs);
    for (int batch = 0; batch < state->batches; ++batch)
    {
      for (int row = 0; row < state->interior_dofs; ++row)
      {
        (*state->newton_f)[static_cast<std::size_t>(batch) *
                           state->interior_dofs + row] =
            -linearization.interior_residual(batch, row, 0);
      }
      for (int row = 0; row < state->trace_dofs; ++row)
      {
        (*state->newton_g)[static_cast<std::size_t>(batch) *
                           state->trace_dofs + row] =
            -linearization.trace_residual(batch, row, 0);
      }
    }
    state->schur->assign(static_cast<std::size_t>(state->batches) *
                         state->trace_dofs * state->trace_dofs, 0.0);
    state->reduced_rhs->assign(static_cast<std::size_t>(state->batches) *
                               state->trace_dofs, 0.0);
    const LocalBlockShape shape{state->interior_dofs / conserved_components,
                                state->trace_dofs / conserved_components,
                                conserved_components, conserved_components};
    const core::LocalSystemBatchView input{
        shape, AsConst(linearization.A), AsConst(linearization.B),
        AsConst(linearization.C), AsConst(linearization.D),
        AsConst(View(state->newton_f->data(), state->batches,
                     state->interior_dofs, 1)),
        AsConst(View(state->newton_g->data(), state->batches,
                     state->trace_dofs, 1))};
    if (state->condensation_engine->Condense(
            input, {View(state->schur->data(), state->batches,
                         state->trace_dofs, state->trace_dofs),
                    View(state->reduced_rhs->data(), state->batches,
                         state->trace_dofs, 1)}) != core::Status::success)
    {
      state->schur->clear();
      state->reduced_rhs->clear();
      return;
    }
    result.contribution.schur = AsConst(View(
        state->schur->data(), state->batches, state->trace_dofs,
        state->trace_dofs));
    result.contribution.right_hand_side = AsConst(View(
        state->reduced_rhs->data(), state->batches, state->trace_dofs, 1));
    result.contribution.trace_dof_count = state->trace_dofs;
    state->condensed = true;
  }

  void ApplySchur(const ApplySchurRequest<SmoothEulerSpec>& request,
                  ApplySchurResult<SmoothEulerSpec>& result)
  {
    result = {};
    if (state->schur_action == nullptr ||
        !HasExtents(request.contribution.schur, state->batches,
                    state->trace_dofs, state->trace_dofs) ||
        !HasExtents(state->trace_solution, state->batches,
                    state->trace_dofs, 1))
    {
      return;
    }
    state->schur_action->assign(static_cast<std::size_t>(state->batches) *
                                state->trace_dofs, 0.0);
    for (int batch = 0; batch < state->batches; ++batch)
      for (int row = 0; row < state->trace_dofs; ++row)
        for (int column = 0; column < state->trace_dofs; ++column)
        {
          (*state->schur_action)[static_cast<std::size_t>(batch) *
                                 state->trace_dofs + row] +=
              request.contribution.schur(batch, row, column) *
              state->trace_solution(batch, column, 0);
        }
    result.trace_residual = View(state->schur_action->data(), state->batches,
                                 state->trace_dofs, 1);
  }
};

struct SmoothEulerAssemblyPhase
{
  PipelinePhaseState* state = nullptr;
  void Assemble(const TraceAssemblyRequest<SmoothEulerSpec>& request,
                TraceAssemblyResult<SmoothEulerSpec>& result)
  {
    result = {};
    state->assembled = false;
    const auto& contribution = request.contribution;
    if (state->trace_space == nullptr || state->signed_trace_dofs == nullptr ||
        state->global_matrix == nullptr || state->trace_matrix == nullptr ||
        state->trace_rhs == nullptr || contribution.trace_dof_count != state->trace_dofs ||
        !HasExtents(contribution.schur, state->batches, state->trace_dofs,
                    state->trace_dofs) ||
        !HasExtents(contribution.right_hand_side, state->batches,
                    state->trace_dofs, 1) ||
        state->signed_trace_dofs->size() !=
            static_cast<std::size_t>(state->batches) * state->trace_dofs)
    {
      return;
    }
    try
    {
      const int global_dofs = state->trace_space->GetVSize();
      auto candidate_matrix = std::make_unique<mfem::SparseMatrix>(global_dofs);
      mfem::Vector candidate_rhs(global_dofs);
      candidate_rhs = 0.0;
      for (int element = 0; element < state->batches; ++element)
      {
        mfem::Array<int> dofs(state->trace_dofs);
        mfem::DenseMatrix local_schur(state->trace_dofs);
        for (int row = 0; row < state->trace_dofs; ++row)
        {
          dofs[row] = (*state->signed_trace_dofs)[
              static_cast<std::size_t>(element) * state->trace_dofs + row];
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(dofs[row], sign);
          if (vdof < 0 || vdof >= global_dofs) { return; }
          candidate_rhs[vdof] += sign *
              contribution.right_hand_side(element, row, 0);
          for (int column = 0; column < state->trace_dofs; ++column)
          {
            local_schur(row, column) =
                contribution.schur(element, row, column);
          }
        }
        candidate_matrix->AddSubMatrix(dofs, dofs, local_schur);
      }
      candidate_matrix->Finalize();
      const mfem::SparseMatrix& finalized_matrix = *candidate_matrix;
      state->global_matrix->assign(static_cast<std::size_t>(global_dofs) *
                                   global_dofs, 0.0);
      for (int row = 0; row < global_dofs; ++row)
        for (int column = 0; column < global_dofs; ++column)
        {
          (*state->global_matrix)[static_cast<std::size_t>(row) * global_dofs +
                                  column] = finalized_matrix(row, column);
        }
      *state->trace_matrix = std::move(candidate_matrix);
      state->trace_rhs->SetSize(global_dofs);
      *state->trace_rhs = candidate_rhs;
      result.trace_system = View(state->global_matrix->data(), 1, global_dofs,
                                 global_dofs);
      result.right_hand_side = View(state->trace_rhs->GetData(), 1, global_dofs, 1);
      state->assembled = true;
    }
    catch (const std::exception&)
    {
      state->trace_matrix->reset();
      state->trace_rhs->SetSize(0);
      state->global_matrix->clear();
    }
  }
};

struct SmoothEulerTraceSolvePhase
{
  PipelinePhaseState* state = nullptr;
  void Solve(const TraceSolveRequest<SmoothEulerSpec>& request,
             TraceSolveResult<SmoothEulerSpec>& result)
  {
    result = {};
    state->solved = false;
    state->gmres_iterations = 0;
    state->reported_relative_residual = 0.0;
    state->true_relative_residual = 0.0;
    state->gmres_converged = false;
    state->linear_residual_valid = false;
    state->trace_solve_diagnostic = {};
    if (state->trace_matrix == nullptr || !*state->trace_matrix ||
        state->trace_solve_matrix == nullptr ||
        state->trace_increment == nullptr || state->trace_preconditioner == nullptr ||
        state->trace_permutation == nullptr || state->global_matrix == nullptr)
    {
      return;
    }
    const int global_dofs = (*state->trace_matrix)->Height();
    if (global_dofs <= 0 || (*state->trace_matrix)->Width() != global_dofs ||
        state->face_block_size <= 0 ||
        !HasExtents(request.trace_system, 1, global_dofs, global_dofs) ||
        !HasExtents(request.right_hand_side, 1, global_dofs, 1) ||
        request.trace_system.data != state->global_matrix->data() ||
        state->trace_permutation->size() != static_cast<std::size_t>(global_dofs))
    {
      return;
    }
    try
    {
      // Reconstruct the actual solve operator from the returned assembly
      // view.  This makes TraceAssemblyResult numerical input, not a flag or
      // a post-hoc observer of a separately solved matrix.
      auto candidate_solve_matrix = std::make_unique<mfem::SparseMatrix>(global_dofs);
      for (int row = 0; row < global_dofs; ++row)
        for (int column = 0; column < global_dofs; ++column)
        {
          const double value = request.trace_system(0, row, column);
          if (value != 0.0) candidate_solve_matrix->Add(row, column, value);
        }
      candidate_solve_matrix->Finalize();
      auto candidate_preconditioner = std::make_unique<PermutedFaceBlockILU>(
          *candidate_solve_matrix, *state->trace_permutation,
          state->face_block_size);
      mfem::GMRESSolver gmres;
      gmres.SetPreconditioner(*candidate_preconditioner);
      gmres.SetOperator(*candidate_solve_matrix);
      gmres.SetRelTol(1.0e-10);
      gmres.SetAbsTol(0.0);
      gmres.SetMaxIter(std::max(50, global_dofs));
      gmres.SetPrintLevel(0);
      mfem::Vector rhs(global_dofs), candidate_increment(global_dofs);
      for (int row = 0; row < global_dofs; ++row)
      {
        rhs[row] = request.right_hand_side(0, row, 0);
      }
      candidate_increment = 0.0;
      state->trace_solve_diagnostic.primary_attempted = true;
      gmres.Mult(rhs, candidate_increment);
      state->gmres_iterations = gmres.GetNumIterations();
      state->reported_relative_residual = gmres.GetFinalRelNorm();
      state->gmres_converged = gmres.GetConverged();
      state->trace_solve_diagnostic.primary_result_reached = true;
      state->trace_solve_diagnostic.primary_iterations =
          gmres.GetNumIterations();
      state->trace_solve_diagnostic.primary_reported_relative_residual =
          gmres.GetFinalRelNorm();
      state->trace_solve_diagnostic.primary_converged = gmres.GetConverged();
      if (!state->gmres_converged) { return; }
      mfem::Vector action(global_dofs);
      candidate_solve_matrix->Mult(candidate_increment, action);
      action -= rhs;
      state->true_relative_residual =
          action.Norml2() / std::max(rhs.Norml2(), 1.0e-300);
      state->trace_solve_diagnostic.primary_true_residual_reached = true;
      state->trace_solve_diagnostic.primary_true_relative_residual =
          state->true_relative_residual;
      state->linear_residual_valid =
          std::isfinite(state->true_relative_residual) &&
          state->true_relative_residual <= 1.0e-10;
      if (!std::isfinite(state->true_relative_residual)) { return; }
      if (!state->linear_residual_valid)
      {
        mfem::Vector correction_rhs(global_dofs), correction(global_dofs);
        correction_rhs = action;
        correction_rhs *= -1.0;
        correction = 0.0;
        state->trace_solve_diagnostic.correction_attempted = true;
        gmres.Mult(correction_rhs, correction);
        state->gmres_iterations += gmres.GetNumIterations();
        state->reported_relative_residual = gmres.GetFinalRelNorm();
        state->gmres_converged = gmres.GetConverged();
        state->trace_solve_diagnostic.correction_result_reached = true;
        state->trace_solve_diagnostic.correction_iterations =
            gmres.GetNumIterations();
        state->trace_solve_diagnostic.correction_reported_relative_residual =
            gmres.GetFinalRelNorm();
        state->trace_solve_diagnostic.correction_converged =
            gmres.GetConverged();
        if (!state->gmres_converged) { return; }

        mfem::Vector corrected_increment(global_dofs);
        corrected_increment = candidate_increment;
        corrected_increment += correction;
        mfem::Vector corrected_action(global_dofs);
        candidate_solve_matrix->Mult(corrected_increment, corrected_action);
        corrected_action -= rhs;
        state->true_relative_residual =
            corrected_action.Norml2() / std::max(rhs.Norml2(), 1.0e-300);
        state->trace_solve_diagnostic.correction_true_residual_reached = true;
        state->trace_solve_diagnostic.correction_true_relative_residual =
            state->true_relative_residual;
        state->linear_residual_valid =
            std::isfinite(state->true_relative_residual) &&
            state->true_relative_residual <= 1.0e-10;
        if (!state->linear_residual_valid) { return; }
        candidate_increment = corrected_increment;
      }
      *state->trace_increment = candidate_increment;
      *state->trace_solve_matrix = std::move(candidate_solve_matrix);
      *state->trace_preconditioner = std::move(candidate_preconditioner);
      result.trace_solution = View(state->trace_increment->GetData(), 1,
                                   global_dofs, 1);
      state->solved = true;
    }
    catch (const std::exception&)
    {
      state->trace_increment->SetSize(0);
      state->trace_solve_matrix->reset();
      state->trace_preconditioner->reset();
    }
  }
};

struct SmoothEulerRecoveryPhase
{
  PipelinePhaseState* state = nullptr;
  void Recover(const RecoveryRequest<SmoothEulerSpec>& request,
               RecoveryResult<SmoothEulerSpec>& result)
  {
    result = {};
    state->recovered_state = false;
    if (state->condensation_engine == nullptr || state->signed_trace_dofs == nullptr ||
        state->recovered == nullptr || state->trace_space == nullptr ||
        !HasExtents(request.trace_solution, 1, state->trace_space->GetVSize(), 1) ||
        state->signed_trace_dofs->size() !=
            static_cast<std::size_t>(state->batches) * state->trace_dofs)
    {
      return;
    }
    std::vector<double> local_trace(static_cast<std::size_t>(state->batches) *
                                    state->trace_dofs, 0.0);
    for (int element = 0; element < state->batches; ++element)
      for (int local = 0; local < state->trace_dofs; ++local)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof((*state->signed_trace_dofs)[
            static_cast<std::size_t>(element) * state->trace_dofs + local], sign);
        if (vdof < 0 || vdof >= request.trace_solution.rows) { return; }
        local_trace[static_cast<std::size_t>(element) * state->trace_dofs + local] =
            sign * request.trace_solution(0, vdof, 0);
      }
    std::vector<double> candidate(static_cast<std::size_t>(state->batches) *
                                  state->interior_dofs, 0.0);
    if (state->condensation_engine->Recover(
            AsConst(View(local_trace.data(), state->batches, state->trace_dofs, 1)),
            View(candidate.data(), state->batches, state->interior_dofs, 1)) !=
        core::Status::success)
    {
      return;
    }
    *state->recovered = std::move(candidate);
    result.interior_solution = View(state->recovered->data(), state->batches,
                                    state->interior_dofs, 1);
    state->recovered_state = true;
  }
};

using SmoothEulerPipeline = Pipeline<SmoothEulerSpec, SmoothEulerLocalPhase,
                                    SmoothEulerCondensationPhase,
                                    SmoothEulerAssemblyPhase,
                                    SmoothEulerTraceSolvePhase,
                                    SmoothEulerRecoveryPhase>;

std::vector<double> GridValues(const mfem::GridFunction& field)
{
  return std::vector<double>(field.GetData(), field.GetData() + field.Size());
}

bool AssignGridValues(mfem::GridFunction& field,
                      const std::vector<double>& values)
{
  if (field.Size() != static_cast<int>(values.size())) return false;
  std::copy(values.begin(), values.end(), field.GetData());
  return true;
}

std::uint64_t HashValues(const std::vector<double>& values) noexcept
{
  constexpr std::uint64_t offset = UINT64_C(1469598103934665603);
  constexpr std::uint64_t prime = UINT64_C(1099511628211);
  std::uint64_t hash = offset;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  for (std::size_t byte = 0; byte < values.size() * sizeof(double); ++byte)
  {
    hash ^= bytes[byte];
    hash *= prime;
  }
  return hash;
}

double MaxAbsDifference(const std::vector<double>& first,
                        const std::vector<double>& second) noexcept
{
  if (first.size() != second.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  double result = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index)
  {
    result = std::max(result, std::abs(first[index] - second[index]));
  }
  return result;
}

bool FiniteVector(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool AccumulateRelativeSquares(const std::vector<double>& initial,
                               const std::vector<double>& current,
                               long double& difference_squared,
                               long double& initial_squared) noexcept
{
  if (initial.size() != current.size() || !FiniteVector(initial) ||
      !FiniteVector(current))
  {
    return false;
  }
  for (std::size_t index = 0; index < initial.size(); ++index)
  {
    const long double difference = static_cast<long double>(current[index]) -
                                   initial[index];
    difference_squared += difference * difference;
    initial_squared += static_cast<long double>(initial[index]) * initial[index];
  }
  return true;
}

double RelativeCoefficientChange(const std::vector<double>& initial_interior,
                                 const std::vector<double>& final_interior,
                                 const std::vector<double>& initial_trace,
                                 const std::vector<double>& final_trace) noexcept
{
  long double difference_squared = 0.0L, initial_squared = 0.0L;
  if (!AccumulateRelativeSquares(initial_interior, final_interior,
                                 difference_squared, initial_squared) ||
      !AccumulateRelativeSquares(initial_trace, final_trace,
                                 difference_squared, initial_squared))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const long double denominator = std::max(initial_squared, 1.0e-600L);
  return std::sqrt(static_cast<double>(difference_squared / denominator));
}

double RelativeJacobianChange(const std::vector<double>& initial_a,
                              const std::vector<double>& current_a,
                              const std::vector<double>& initial_b,
                              const std::vector<double>& current_b,
                              const std::vector<double>& initial_c,
                              const std::vector<double>& current_c,
                              const std::vector<double>& initial_d,
                              const std::vector<double>& current_d) noexcept
{
  long double difference_squared = 0.0L, initial_squared = 0.0L;
  if (!AccumulateRelativeSquares(initial_a, current_a, difference_squared,
                                 initial_squared) ||
      !AccumulateRelativeSquares(initial_b, current_b, difference_squared,
                                 initial_squared) ||
      !AccumulateRelativeSquares(initial_c, current_c, difference_squared,
                                 initial_squared) ||
      !AccumulateRelativeSquares(initial_d, current_d, difference_squared,
                                 initial_squared))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const long double denominator = std::max(initial_squared, 1.0e-600L);
  return std::sqrt(static_cast<double>(difference_squared / denominator));
}

double ResidualNorm(const std::vector<double>& interior_residual,
                    const std::vector<double>& trace_residual,
                    const std::vector<int>& signed_trace_dofs,
                    const int global_trace_dofs)
{
  if (trace_residual.size() != signed_trace_dofs.size() ||
      global_trace_dofs <= 0)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  long double squared = 0.0L;
  for (const double value : interior_residual)
  {
    squared += static_cast<long double>(value) * value;
  }
  std::vector<double> unique_trace(static_cast<std::size_t>(global_trace_dofs), 0.0);
  for (std::size_t local = 0; local < trace_residual.size(); ++local)
  {
    double sign = 1.0;
    const int global = DecodeSignedVDof(signed_trace_dofs[local], sign);
    if (global < 0 || global >= global_trace_dofs)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    unique_trace[global] += sign * trace_residual[local];
  }
  for (const double value : unique_trace)
  {
    squared += static_cast<long double>(value) * value;
  }
  return std::sqrt(static_cast<double>(squared));
}

void ObserveAdmissibility(const ConservedState& state,
                          bool& nonfinite, bool& density, bool& pressure,
                          bool& wave_speed, AdmissibilityEvidence& evidence)
{
  if (!IsFinite(state))
  {
    nonfinite = true;
    return;
  }
  evidence.minimum_density = std::min(evidence.minimum_density, state[0]);
  if (!(state[0] >= 1.0e-8))
  {
    density = true;
    return;
  }
  const double kinetic = 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                         state[0];
  const double p = (gamma - 1.0) * (state[3] - kinetic);
  if (!std::isfinite(p))
  {
    nonfinite = true;
    return;
  }
  evidence.minimum_pressure = std::min(evidence.minimum_pressure, p);
  if (!(p >= 1.0e-8))
  {
    pressure = true;
    return;
  }
  const double rho = state[0];
  const double u = state[1] / rho;
  const double v = state[2] / rho;
  const double speed = std::sqrt(u * u + v * v) +
                       std::sqrt(gamma * p / rho);
  if (!std::isfinite(speed))
  {
    nonfinite = true;
    return;
  }
  evidence.maximum_wave_speed = std::max(evidence.maximum_wave_speed, speed);
  if (!(speed < fixed_scalar_tau)) wave_speed = true;
}

bool PrescreenFields(mfem::Mesh& mesh,
                     const mfem::FiniteElementSpace& interior_space,
                     const mfem::FiniteElementSpace& trace_space,
                     const mfem::GridFunction& interior,
                     const mfem::GridFunction& trace,
                     const int quadrature_order,
                     AdmissibilityEvidence& evidence)
{
  evidence = {};
  evidence.minimum_density = std::numeric_limits<double>::infinity();
  evidence.minimum_pressure = std::numeric_limits<double>::infinity();
  bool nonfinite = false, density = false, pressure = false, wave_speed = false;
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* fe = interior_space.GetFE(element);
    mfem::ElementTransformation* transformation =
        mesh.GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) return false;
    mfem::Array<int> interior_vdofs;
    interior_space.GetElementVDofs(element, interior_vdofs);
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& volume_rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < volume_rule.GetNPoints(); ++q)
    {
      fe->CalcShape(volume_rule.IntPoint(q), shape);
      ConservedState state{};
      if (!InterpolateByNodes(interior, interior_vdofs, shape, state))
      {
        nonfinite = true;
      }
      else
      {
        ObserveAdmissibility(state, nonfinite, density, pressure, wave_speed,
                             evidence);
      }
    }

    mfem::Array<int> faces, orientations;
    mesh.GetElementEdges(element, faces, orientations);
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::FaceElementTransformations* face_transform =
          mesh.GetFaceElementTransformations(faces[local_face]);
      if (face_transform == nullptr || face_transform->Face == nullptr ||
          face_transform->Elem1 == nullptr || face_transform->Elem2 == nullptr)
      {
        return false;
      }
      const bool first = face_transform->Elem1No == element;
      if (!first && face_transform->Elem2No != element) return false;
      const mfem::FiniteElement* trace_fe =
          trace_space.GetFaceElement(faces[local_face]);
      if (trace_fe == nullptr) return false;
      const mfem::IntegrationRule& face_rule =
          mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order);
      for (int q = 0; q < face_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& face_point = face_rule.IntPoint(q);
        face_transform->SetAllIntPoints(&face_point);
        const mfem::IntegrationPoint& element_point = first ?
            face_transform->GetElement1IntPoint() :
            face_transform->GetElement2IntPoint();
        fe->CalcShape(element_point, shape);
        ConservedState state{};
        if (!InterpolateByNodes(interior, interior_vdofs, shape, state))
        {
          nonfinite = true;
        }
        else
        {
          ObserveAdmissibility(state, nonfinite, density, pressure, wave_speed,
                               evidence);
        }
      }
    }
  }
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* fe = trace_space.GetFaceElement(face);
    if (fe == nullptr) return false;
    mfem::Array<int> vdofs;
    trace_space.GetFaceVDofs(face, vdofs);
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      fe->CalcShape(rule.IntPoint(q), shape);
      ConservedState state{};
      if (!InterpolateByNodes(trace, vdofs, shape, state))
      {
        nonfinite = true;
      }
      else
      {
        ObserveAdmissibility(state, nonfinite, density, pressure, wave_speed,
                             evidence);
      }
    }
  }
  if (!std::isfinite(evidence.minimum_density)) evidence.minimum_density = 0.0;
  if (!std::isfinite(evidence.minimum_pressure)) evidence.minimum_pressure = 0.0;
  evidence.reason = nonfinite ? TrialRejectionReason::Nonfinite :
      density ? TrialRejectionReason::Density :
      pressure ? TrialRejectionReason::Pressure :
      wave_speed ? TrialRejectionReason::WaveSpeed : TrialRejectionReason::None;
  evidence.passed = evidence.reason == TrialRejectionReason::None;
  return true;
}

} // namespace

int ByNodesIndex(const int scalar_dof, const int scalar_count,
                 const int component) noexcept
{
  return scalar_dof + scalar_count * component;
}

int ComponentFromByNodesIndex(const int vector_dof,
                              const int scalar_count) noexcept
{
  return vector_dof / scalar_count;
}

int ScalarFromByNodesIndex(const int vector_dof,
                           const int scalar_count) noexcept
{
  return vector_dof - scalar_count * ComponentFromByNodesIndex(vector_dof,
                                                                 scalar_count);
}

int DecodeSignedVDof(const int signed_vdof, double& sign) noexcept
{
  return mfem::FiniteElementSpace::DecodeDof(signed_vdof, sign);
}

PhysicsResult ConservativeFromPrimitive(const PrimitiveState& primitive,
                                       ConservedState& conservative) noexcept
{
  if (!IsFinite(primitive)) { return {PhysicsStatus::non_finite}; }
  if (!(primitive[0] > 0.0)) { return {PhysicsStatus::invalid_density}; }
  if (!(primitive[3] > 0.0)) { return {PhysicsStatus::invalid_pressure}; }
  const double rho = primitive[0];
  const double u = primitive[1];
  const double v = primitive[2];
  const double p = primitive[3];
  const ConservedState candidate{{rho, rho * u, rho * v,
      p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)}};
  if (!IsFinite(candidate)) { return {PhysicsStatus::non_finite}; }
  conservative = candidate;
  return {};
}

PhysicsResult PrimitiveFromConservative(const ConservedState& conservative,
                                       PrimitiveState& primitive) noexcept
{
  double pressure = 0.0;
  const PhysicsResult status = Pressure(conservative, pressure);
  if (!status.ok()) { return status; }
  const double rho = conservative[0];
  const PrimitiveState candidate{{rho, conservative[1] / rho,
      conservative[2] / rho, pressure}};
  if (!IsFinite(candidate)) { return {PhysicsStatus::non_finite}; }
  primitive = candidate;
  return {};
}

PhysicsResult NormalFlux(const ConservedState& conservative,
                         const std::array<double, dimension>& normal,
                         FluxVector& flux) noexcept
{
  const PhysicsResult normal_status = CheckNormal(normal);
  if (!normal_status.ok()) { return normal_status; }
  double pressure = 0.0;
  const PhysicsResult pressure_status = Pressure(conservative, pressure);
  if (!pressure_status.ok()) { return pressure_status; }
  const double rho = conservative[0];
  const double momentum_normal = conservative[1] * normal[0] +
                                 conservative[2] * normal[1];
  const FluxVector candidate{{momentum_normal,
      conservative[1] * momentum_normal / rho + pressure * normal[0],
      conservative[2] * momentum_normal / rho + pressure * normal[1],
      (conservative[3] + pressure) * momentum_normal / rho}};
  if (!IsFinite(candidate)) { return {PhysicsStatus::non_finite}; }
  flux = candidate;
  return {};
}

PhysicsResult NormalFluxJacobian(const ConservedState& conservative,
                                 const std::array<double, dimension>& normal,
                                 FluxJacobian& jacobian) noexcept
{
  const PhysicsResult normal_status = CheckNormal(normal);
  if (!normal_status.ok()) { return normal_status; }
  double pressure = 0.0;
  const PhysicsResult pressure_status = Pressure(conservative, pressure);
  if (!pressure_status.ok()) { return pressure_status; }
  const double rho = conservative[0];
  const double mx = conservative[1];
  const double my = conservative[2];
  const double energy = conservative[3];
  const double q = mx * normal[0] + my * normal[1];
  const double pressure_rho = (gamma - 1.0) * 0.5 * (mx * mx + my * my) /
                              (rho * rho);
  const double pressure_mx = -(gamma - 1.0) * mx / rho;
  const double pressure_my = -(gamma - 1.0) * my / rho;
  const double pressure_energy = gamma - 1.0;
  const double enthalpy = energy + pressure;
  const FluxJacobian candidate{{
      {{0.0, normal[0], normal[1], 0.0}},
      {{-mx * q / (rho * rho) + pressure_rho * normal[0],
        q / rho + mx * normal[0] / rho + pressure_mx * normal[0],
        mx * normal[1] / rho + pressure_my * normal[0],
        pressure_energy * normal[0]}},
      {{-my * q / (rho * rho) + pressure_rho * normal[1],
        my * normal[0] / rho + pressure_mx * normal[1],
        q / rho + my * normal[1] / rho + pressure_my * normal[1],
        pressure_energy * normal[1]}},
      {{pressure_rho * q / rho - enthalpy * q / (rho * rho),
        pressure_mx * q / rho + enthalpy * normal[0] / rho,
        pressure_my * q / rho + enthalpy * normal[1] / rho,
        (1.0 + pressure_energy) * q / rho}}}};
  for (const auto& row : candidate)
  {
    if (!IsFinite(row)) { return {PhysicsStatus::non_finite}; }
  }
  jacobian = candidate;
  return {};
}

PhysicsResult ManufacturedPrimitive(const double x, const double y,
                                   PrimitiveState& primitive) noexcept
{
  if (!IsFinite(x) || !IsFinite(y)) { return {PhysicsStatus::non_finite}; }
  const PrimitiveState candidate{{
      1.0 + 0.1 * std::sin(2.0 * pi * x) * std::sin(2.0 * pi * y),
      0.35 + 0.05 * std::cos(2.0 * pi * x) * std::sin(2.0 * pi * y),
      -0.20 + 0.05 * std::sin(2.0 * pi * x) * std::cos(2.0 * pi * y),
      1.0 + 0.1 * std::cos(2.0 * pi * x) * std::cos(2.0 * pi * y)}};
  primitive = candidate;
  return {};
}

PhysicsResult ManufacturedConservative(const double x, const double y,
                                      ConservedState& conservative) noexcept
{
  PrimitiveState primitive{};
  const PhysicsResult status = ManufacturedPrimitive(x, y, primitive);
  return status.ok() ? ConservativeFromPrimitive(primitive, conservative) : status;
}

PhysicsResult ManufacturedSource(const double x, const double y,
                                ConservedState& source) noexcept
{
  ConservedState state{};
  const PhysicsResult state_status = ManufacturedConservative(x, y, state);
  if (!state_status.ok()) { return state_status; }
  FluxJacobian x_jacobian{};
  FluxJacobian y_jacobian{};
  const PhysicsResult x_status = NormalFluxJacobian(state, {{1.0, 0.0}}, x_jacobian);
  const PhysicsResult y_status = NormalFluxJacobian(state, {{0.0, 1.0}}, y_jacobian);
  if (!x_status.ok()) { return x_status; }
  if (!y_status.ok()) { return y_status; }
  const ConservedState dx = ManufacturedDerivative(x, y, 0);
  const ConservedState dy = ManufacturedDerivative(x, y, 1);
  ConservedState candidate{};
  for (int row = 0; row < conserved_components; ++row)
  {
    for (int column = 0; column < conserved_components; ++column)
    {
      candidate[row] += x_jacobian[row][column] * dx[column] +
                        y_jacobian[row][column] * dy[column];
    }
  }
  if (!IsFinite(candidate)) { return {PhysicsStatus::non_finite}; }
  source = candidate;
  return {};
}

double ManufacturedMaximumWaveSpeed() noexcept
{
  // With r=cos(2*pi*x)sin(2*pi*y), s=sin(2*pi*x)cos(2*pi*y), r^2+s^2<=1.
  // Hence |(u,v)| <= |(.35,-.20)| + .05 and p/rho <= 1.1/.9. This is a
  // proved bound, unlike the previous sampled value, and remains below 1.78.
  return std::sqrt(0.35 * 0.35 + 0.20 * 0.20) + 0.05 +
         std::sqrt(gamma * 1.1 / 0.9);
}

struct SmoothEulerExperiment::Implementation
{
  std::unique_ptr<mfem::Mesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::FiniteElementSpace> interior_space;
  std::unique_ptr<mfem::FiniteElementSpace> trace_space;
  std::unique_ptr<mfem::GridFunction> old_interior;
  std::unique_ptr<mfem::GridFunction> old_trace;
  std::unique_ptr<mfem::GridFunction> current_interior;
  std::unique_ptr<mfem::GridFunction> current_trace;
  std::unique_ptr<mfem::SparseMatrix> trace_matrix;
  std::unique_ptr<mfem::SparseMatrix> trace_solve_matrix;
  mfem::Vector trace_rhs, trace_increment;
  std::vector<int> trace_dofs;
  std::vector<double> residual_r, residual_g, a, b, c, d, f, g;
  std::vector<double> schur, reduced_rhs, schur_action, recovered, global_matrix;
  std::unique_ptr<PermutedFaceBlockILU> trace_preconditioner;
  core::HostReferenceEngine condensation;
  int local_interior = 0, local_trace = 0, elements = 0,
      elements_per_direction = 0, trace_face_block_size = 0;
  std::vector<int> trace_permutation;
  std::size_t generation = 0, linearization_generation = 0,
              condensation_generation = 0, solve_generation = 0,
              recovery_generation = 0;
  PipelinePhaseState pipeline_state{};
  TraceSolveDiagnostic last_trace_solve_diagnostic{};
  std::unique_ptr<SmoothEulerPipeline> pipeline;
  SolvedMeshEvidence solved_evidence{};
  bool has_solved_evidence = false;
  bool rollback_probe_mode = false;

  void InvalidateNumericalState()
  {
    ++generation;
    linearization_generation = 0;
    condensation_generation = 0;
    solve_generation = 0;
    recovery_generation = 0;
    residual_r.clear();
    residual_g.clear();
    a.clear();
    b.clear();
    c.clear();
    d.clear();
    f.clear();
    g.clear();
    schur.clear();
    reduced_rhs.clear();
    schur_action.clear();
    recovered.clear();
    global_matrix.clear();
    trace_matrix.reset();
    trace_solve_matrix.reset();
    trace_rhs.SetSize(0);
    trace_increment.SetSize(0);
    trace_preconditioner.reset();
    pipeline_state = {};
  }
};

SmoothEulerExperiment::SmoothEulerExperiment(const int polynomial_order)
    : implementation_(std::make_unique<Implementation>()),
      polynomial_order_(polynomial_order), quadrature_order_(3 * polynomial_order + 8)
{
  if (polynomial_order < 1 || polynomial_order > 4)
  {
    throw std::invalid_argument("smooth Euler experiment requires p in [1,4]");
  }
  implementation_->pipeline = std::make_unique<SmoothEulerPipeline>(
      SmoothEulerLocalPhase{&implementation_->pipeline_state},
      SmoothEulerCondensationPhase{&implementation_->pipeline_state},
      SmoothEulerAssemblyPhase{&implementation_->pipeline_state},
      SmoothEulerTraceSolvePhase{&implementation_->pipeline_state},
      SmoothEulerRecoveryPhase{&implementation_->pipeline_state});
}

SmoothEulerExperiment::~SmoothEulerExperiment() = default;
SmoothEulerExperiment::SmoothEulerExperiment(SmoothEulerExperiment&&) noexcept = default;
SmoothEulerExperiment& SmoothEulerExperiment::operator=(SmoothEulerExperiment&&) noexcept = default;

bool SmoothEulerExperiment::BuildPeriodicMesh(const int elements_per_direction)
{
  if (elements_per_direction < 3) { return false; }
  std::unique_ptr<Implementation> candidate;
  try
  {
    candidate = std::make_unique<Implementation>();
    candidate->pipeline = std::make_unique<SmoothEulerPipeline>(
        SmoothEulerLocalPhase{&candidate->pipeline_state},
        SmoothEulerCondensationPhase{&candidate->pipeline_state},
        SmoothEulerAssemblyPhase{&candidate->pipeline_state},
        SmoothEulerTraceSolvePhase{&candidate->pipeline_state},
        SmoothEulerRecoveryPhase{&candidate->pipeline_state});
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        elements_per_direction, elements_per_direction, mfem::Element::QUADRILATERAL,
        false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations;
    translations.emplace_back(dimension);
    translations.emplace_back(dimension);
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    candidate->mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    if (candidate->mesh->GetNE() <= 0) { return false; }
    for (int face = 0; face < candidate->mesh->GetNumFaces(); ++face)
    {
      int first = -1, second = -1;
      candidate->mesh->GetFaceElements(face, &first, &second);
      if (first < 0 || second < 0 || first == second) { return false; }
    }
    candidate->interior_collection = std::make_unique<mfem::L2_FECollection>(
        polynomial_order_, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    candidate->trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
        polynomial_order_, dimension, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    candidate->interior_space = std::make_unique<mfem::FiniteElementSpace>(
        candidate->mesh.get(), candidate->interior_collection.get(),
        conserved_components, mfem::Ordering::byNODES);
    candidate->trace_space = std::make_unique<mfem::FiniteElementSpace>(
        candidate->mesh.get(), candidate->trace_collection.get(),
        conserved_components, mfem::Ordering::byNODES);
    candidate->elements = candidate->mesh->GetNE();
    candidate->elements_per_direction = elements_per_direction;
    const mfem::FiniteElement* interior_fe = candidate->interior_space->GetFE(0);
    if (interior_fe == nullptr || interior_fe->GetDof() <= 0) { return false; }
    candidate->local_interior = conserved_components * interior_fe->GetDof();
    mfem::Array<int> faces, orientations;
    candidate->mesh->GetElementEdges(0, faces, orientations);
    candidate->local_trace = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const mfem::FiniteElement* face_fe =
          candidate->trace_space->GetFaceElement(faces[local_face]);
      if (face_fe == nullptr || face_fe->GetDof() <= 0) { return false; }
      mfem::Array<int> face_vdofs;
      candidate->trace_space->GetFaceVDofs(faces[local_face], face_vdofs);
      if (face_vdofs.Size() != conserved_components * face_fe->GetDof())
      {
        return false;
      }
      candidate->local_trace += face_vdofs.Size();
    }
    candidate->trace_face_block_size = conserved_components *
        candidate->trace_space->GetFaceElement(faces[0])->GetDof();
    if (candidate->trace_face_block_size <= 0 ||
        candidate->trace_space->GetVSize() % candidate->trace_face_block_size != 0)
    {
      return false;
    }
    candidate->trace_dofs.assign(
        static_cast<std::size_t>(candidate->elements) * candidate->local_trace, 0);
    for (int element = 0; element < candidate->elements; ++element)
    {
      mfem::Array<int> element_faces, element_orientations;
      candidate->mesh->GetElementEdges(element, element_faces, element_orientations);
      if (element_faces.Size() != faces.Size() ||
          element_orientations.Size() != element_faces.Size()) { return false; }
      int offset = 0;
      for (int local_face = 0; local_face < element_faces.Size(); ++local_face)
      {
        mfem::Array<int> face_vdofs;
        candidate->trace_space->GetFaceVDofs(element_faces[local_face], face_vdofs);
        if (face_vdofs.Size() <= 0 || offset + face_vdofs.Size() > candidate->local_trace)
        {
          return false;
        }
        std::copy(face_vdofs.begin(), face_vdofs.end(),
                  candidate->trace_dofs.begin() +
                      static_cast<std::size_t>(element) * candidate->local_trace + offset);
        offset += face_vdofs.Size();
      }
      if (offset != candidate->local_trace) { return false; }
    }
    candidate->trace_permutation.clear();
    std::vector<bool> seen(static_cast<std::size_t>(candidate->trace_space->GetVSize()), false);
    for (int face = 0; face < candidate->mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* face_fe = candidate->trace_space->GetFaceElement(face);
      if (face_fe == nullptr || face_fe->GetDof() <= 0) { return false; }
      const int scalar_dofs = face_fe->GetDof();
      mfem::Array<int> face_vdofs;
      candidate->trace_space->GetFaceVDofs(face, face_vdofs);
      for (int node = 0; node < scalar_dofs; ++node)
        for (int component = 0; component < conserved_components; ++component)
        {
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(
              face_vdofs[ByNodesIndex(node, scalar_dofs, component)], sign);
          if (vdof < 0 || vdof >= candidate->trace_space->GetVSize() || seen[vdof])
          {
            return false;
          }
          seen[vdof] = true;
          candidate->trace_permutation.push_back(vdof);
        }
    }
    if (static_cast<int>(candidate->trace_permutation.size()) !=
            candidate->trace_space->GetVSize() ||
        std::find(seen.begin(), seen.end(), false) != seen.end()) return false;
  }
  catch (const std::exception&)
  {
    return false;
  }
  if (candidate->local_interior <= 0 || candidate->local_trace <= 0) { return false; }
  // Publish only a complete graph: old GridFunctions are destroyed before the
  // old spaces, collections, and mesh when the displaced Implementation dies.
  implementation_.swap(candidate);
  return true;
}

bool SmoothEulerExperiment::ProjectOldStage()
{
  // A reprojection attempt immediately makes any prior local linearization
  // stale.  Candidate fields remain private until the projection succeeds.
  implementation_->InvalidateNumericalState();
  implementation_->has_solved_evidence = false;
  if (!implementation_->mesh || !implementation_->interior_space ||
      !implementation_->trace_space) { return false; }
  auto coefficient = [](const mfem::Vector& point, mfem::Vector& value)
  {
    ConservedState state{};
    if (!ManufacturedConservative(point[0], point[1], state).ok())
    {
      throw std::runtime_error("manufactured state failed");
    }
    value.SetSize(conserved_components);
    for (int component = 0; component < conserved_components; ++component)
    {
      value[component] = state[component];
    }
  };
  std::unique_ptr<mfem::GridFunction> candidate_interior;
  std::unique_ptr<mfem::GridFunction> candidate_trace;
  std::unique_ptr<mfem::GridFunction> candidate_current_interior;
  std::unique_ptr<mfem::GridFunction> candidate_current_trace;
  try
  {
    mfem::VectorFunctionCoefficient exact(conserved_components, coefficient);
    candidate_interior =
        std::make_unique<mfem::GridFunction>(implementation_->interior_space.get());
    candidate_trace =
        std::make_unique<mfem::GridFunction>(implementation_->trace_space.get());
    candidate_interior->ProjectCoefficient(exact);
    // DG_Interface has face finite elements. GetFaceVDofs returns the pinned
    // MFEM canonical face permutation and signed orientation; its local array
    // is byNODES with the face scalar extent, while each GridFunction entry is
    // byNODES with the trace-space scalar extent.
    for (int face = 0; face < implementation_->mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* face_fe =
          implementation_->trace_space->GetFaceElement(face);
      mfem::ElementTransformation* face_transformation =
          implementation_->mesh->GetFaceTransformation(face);
      if (face_fe == nullptr || face_transformation == nullptr) { return false; }
      const int face_scalar_dofs = face_fe->GetDof();
      if (face_scalar_dofs <= 0) { return false; }
      mfem::Array<int> face_vdofs;
      implementation_->trace_space->GetFaceVDofs(face, face_vdofs);
      if (face_vdofs.Size() != conserved_components * face_scalar_dofs) { return false; }
      const mfem::IntegrationRule& nodes = face_fe->GetNodes();
      for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
      {
        mfem::Vector point(dimension);
        face_transformation->Transform(nodes.IntPoint(scalar), point);
        ConservedState state{};
        if (!ManufacturedConservative(point[0], point[1], state).ok()) { return false; }
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = ByNodesIndex(scalar, face_scalar_dofs, component);
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(face_vdofs[local], sign);
          if (vdof < 0 || vdof >= candidate_trace->Size()) { return false; }
          (*candidate_trace)[vdof] = sign * state[component];
        }
      }
    }
    candidate_current_interior =
        std::make_unique<mfem::GridFunction>(implementation_->interior_space.get());
    candidate_current_trace =
        std::make_unique<mfem::GridFunction>(implementation_->trace_space.get());
    *candidate_current_interior = *candidate_interior;
    *candidate_current_trace = *candidate_trace;
  }
  catch (const std::exception&)
  {
    return false;
  }
  implementation_->old_interior = std::move(candidate_interior);
  implementation_->old_trace = std::move(candidate_trace);
  implementation_->current_interior = std::move(candidate_current_interior);
  implementation_->current_trace = std::move(candidate_current_trace);
  return true;
}

bool SmoothEulerExperiment::ProjectPerturbedCurrentStage()
{
  implementation_->InvalidateNumericalState();
  implementation_->has_solved_evidence = false;
  if (!implementation_->mesh || !implementation_->interior_space ||
      !implementation_->trace_space || !implementation_->old_interior ||
      !implementation_->old_trace)
  {
    return false;
  }
  auto perturbed = [](const mfem::Vector& point, mfem::Vector& value)
  {
    PrimitiveState primitive{};
    if (!ManufacturedPrimitive(point[0], point[1], primitive).ok())
    {
      throw std::runtime_error("manufactured primitive failed");
    }
    const double psi_one = std::sin(2.0 * pi * point[0]) *
                           std::cos(2.0 * pi * point[1]);
    const double psi_two = std::cos(2.0 * pi * point[0]) *
                           std::sin(2.0 * pi * point[1]);
    const double psi_three = std::sin(2.0 * pi * (point[0] + point[1]));
    primitive[0] *= 1.0 + 0.04 * psi_one;
    primitive[1] += 0.03 * psi_two;
    primitive[2] -= 0.03 * psi_one;
    primitive[3] *= 1.0 + 0.04 * psi_three;
    ConservedState conservative{};
    if (!ConservativeFromPrimitive(primitive, conservative).ok())
    {
      throw std::runtime_error("perturbed primitive failed");
    }
    value.SetSize(conserved_components);
    for (int component = 0; component < conserved_components; ++component)
    {
      value[component] = conservative[component];
    }
  };
  std::unique_ptr<mfem::GridFunction> candidate_interior;
  std::unique_ptr<mfem::GridFunction> candidate_trace;
  try
  {
    mfem::VectorFunctionCoefficient coefficient(conserved_components, perturbed);
    candidate_interior =
        std::make_unique<mfem::GridFunction>(implementation_->interior_space.get());
    candidate_trace =
        std::make_unique<mfem::GridFunction>(implementation_->trace_space.get());
    candidate_interior->ProjectCoefficient(coefficient);
    for (int face = 0; face < implementation_->mesh->GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* face_fe =
          implementation_->trace_space->GetFaceElement(face);
      mfem::ElementTransformation* face_transformation =
          implementation_->mesh->GetFaceTransformation(face);
      if (face_fe == nullptr || face_transformation == nullptr) return false;
      const int face_scalar_dofs = face_fe->GetDof();
      mfem::Array<int> face_vdofs;
      implementation_->trace_space->GetFaceVDofs(face, face_vdofs);
      if (face_scalar_dofs <= 0 ||
          face_vdofs.Size() != conserved_components * face_scalar_dofs)
      {
        return false;
      }
      const mfem::IntegrationRule& nodes = face_fe->GetNodes();
      for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
      {
        mfem::Vector point(dimension), value(conserved_components);
        face_transformation->Transform(nodes.IntPoint(scalar), point);
        perturbed(point, value);
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = ByNodesIndex(scalar, face_scalar_dofs, component);
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(face_vdofs[local], sign);
          if (vdof < 0 || vdof >= candidate_trace->Size()) return false;
          (*candidate_trace)[vdof] = sign * value[component];
        }
      }
    }
  }
  catch (const std::exception&)
  {
    return false;
  }
  implementation_->current_interior = std::move(candidate_interior);
  implementation_->current_trace = std::move(candidate_trace);
  return true;
}

bool SmoothEulerExperiment::AssembleLinearization()
{
  // An assembly attempt is a new generation.  Invalidate and erase the old
  // snapshot first so every failure path rejects stale data atomically.
  implementation_->InvalidateNumericalState();
  if (!implementation_->mesh || !implementation_->old_interior ||
      !implementation_->old_trace || !implementation_->current_interior ||
      !implementation_->current_trace || implementation_->local_interior <= 0 ||
      implementation_->local_trace <= 0) return false;
  const int ni = implementation_->local_interior;
  const int nt = implementation_->local_trace;
  const int batches = implementation_->elements;
  std::vector<double> candidate_a(static_cast<std::size_t>(batches) * ni * ni,
                                  0.0);
  std::vector<double> candidate_b(static_cast<std::size_t>(batches) * ni * nt, 0.0);
  std::vector<double> candidate_c(static_cast<std::size_t>(batches) * nt * ni, 0.0);
  std::vector<double> candidate_d(static_cast<std::size_t>(batches) * nt * nt, 0.0);
  std::vector<double> candidate_r(static_cast<std::size_t>(batches) * ni, 0.0);
  std::vector<double> candidate_g(static_cast<std::size_t>(batches) * nt, 0.0);
  for (int element = 0; element < batches; ++element)
  {
    const mfem::FiniteElement& fe = *implementation_->interior_space->GetFE(element);
    mfem::ElementTransformation& transformation =
        *implementation_->mesh->GetElementTransformation(element);
    const int scalar_dofs = fe.GetDof();
    mfem::Array<int> interior_vdofs;
    implementation_->interior_space->GetElementVDofs(element, interior_vdofs);
    if (interior_vdofs.Size() != ni) return false;
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe.GetGeomType(), quadrature_order_);
    const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(point_index);
      transformation.SetIntPoint(&point);
      mfem::Vector physical(dimension);
      transformation.Transform(point, physical);
      mfem::Vector shape(scalar_dofs);
      mfem::DenseMatrix derivative(scalar_dofs, dimension);
      fe.CalcShape(point, shape);
      fe.CalcPhysDShape(transformation, derivative);
      ConservedState state{}, old_state{};
      ConservedState source{};
      FluxVector flux_x{};
      FluxVector flux_y{};
      FluxJacobian jacobian_x{};
      FluxJacobian jacobian_y{};
      if (!InterpolateByNodes(*implementation_->current_interior, interior_vdofs,
                              shape, state) ||
          !InterpolateByNodes(*implementation_->old_interior, interior_vdofs,
                              shape, old_state) ||
          !ManufacturedSource(physical[0], physical[1], source).ok() ||
          !NormalFlux(state, {{1.0, 0.0}}, flux_x).ok() ||
          !NormalFlux(state, {{0.0, 1.0}}, flux_y).ok() ||
          !NormalFluxJacobian(state, {{1.0, 0.0}}, jacobian_x).ok() ||
          !NormalFluxJacobian(state, {{0.0, 1.0}}, jacobian_y).ok()) return false;
      const double weight = point.weight * transformation.Weight();
      for (int row = 0; row < scalar_dofs; ++row)
      {
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local_row = ByNodesIndex(row, scalar_dofs, component);
          const double residual = weight *
              (shape[row] * (state[component] - old_state[component]) /
                   backward_euler_dt -
               derivative(row, 0) * flux_x[component] -
               derivative(row, 1) * flux_y[component] -
               shape[row] * source[component]);
          candidate_r[static_cast<std::size_t>(element) * ni + local_row] += residual;
        }
        for (int column = 0; column < scalar_dofs; ++column)
        {
          for (int row_component = 0; row_component < conserved_components; ++row_component)
          {
            const int local_row = ByNodesIndex(row, scalar_dofs, row_component);
            for (int column_component = 0; column_component < conserved_components;
                 ++column_component)
            {
              const int local_column =
                  ByNodesIndex(column, scalar_dofs, column_component);
              const double mass = row_component == column_component ?
                  shape[row] * shape[column] / backward_euler_dt : 0.0;
              const double volume = -(derivative(row, 0) * jacobian_x[row_component][column_component] +
                                      derivative(row, 1) * jacobian_y[row_component][column_component]) *
                                    shape[column];
              candidate_a[a_offset + static_cast<std::size_t>(local_row) * ni + local_column] +=
                  weight * (mass + volume);
            }
          }
        }
      }
    }

    // Every local face is evaluated in the canonical DG_Interface face
    // coordinate.  MFEM's Loc1/Loc2 mapping supplies the element-side point;
    // the shared signed VDof list is retained verbatim for global assembly.
    mfem::Array<int> faces, orientations;
    implementation_->mesh->GetElementEdges(element, faces, orientations);
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* face_transformation =
          implementation_->mesh->GetFaceElementTransformations(face);
      const mfem::FiniteElement* trace_fe =
          implementation_->trace_space->GetFaceElement(face);
      mfem::Array<int> face_vdofs;
      implementation_->trace_space->GetFaceVDofs(face, face_vdofs);
      if (face_transformation == nullptr || face_transformation->Face == nullptr ||
          trace_fe == nullptr || face_vdofs.Size() !=
              conserved_components * trace_fe->GetDof()) return false;
      const bool is_element_one = face_transformation->Elem1No == element;
      if (!is_element_one && face_transformation->Elem2No != element) return false;
      const int trace_scalar_dofs = trace_fe->GetDof();
      const mfem::IntegrationRule& face_rule =
          mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order_);
      mfem::Vector interior_shape(scalar_dofs), trace_shape(trace_scalar_dofs);
      for (int point_index = 0; point_index < face_rule.GetNPoints(); ++point_index)
      {
        const mfem::IntegrationPoint& face_point = face_rule.IntPoint(point_index);
        face_transformation->SetAllIntPoints(&face_point);
        const mfem::IntegrationPoint& element_point = is_element_one ?
            face_transformation->GetElement1IntPoint() :
            face_transformation->GetElement2IntPoint();
        mfem::Vector physical(dimension), scaled_normal(dimension);
        face_transformation->Face->Transform(face_point, physical);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled_normal);
        const double face_measure = scaled_normal.Norml2();
        if (!(face_measure > 0.0) || !std::isfinite(face_measure)) return false;
        const double side = is_element_one ? 1.0 : -1.0;
        const std::array<double, dimension> unit_normal{{
            side * scaled_normal[0] / face_measure,
            side * scaled_normal[1] / face_measure}};
        fe.CalcShape(element_point, interior_shape);
        trace_fe->CalcShape(face_point, trace_shape);
        ConservedState interior_state{}, trace_state{};
        FluxVector trace_flux{};
        FluxJacobian trace_jacobian{};
        if (!InterpolateByNodes(*implementation_->current_interior, interior_vdofs,
                                interior_shape, interior_state) ||
            !InterpolateByNodes(*implementation_->current_trace, face_vdofs,
                                trace_shape, trace_state) ||
            !NormalFlux(trace_state, unit_normal, trace_flux).ok() ||
            !NormalFluxJacobian(trace_state, unit_normal, trace_jacobian).ok()) return false;
        const double weight = face_point.weight * face_measure;
        for (int row = 0; row < scalar_dofs; ++row)
        {
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local_row = ByNodesIndex(row, scalar_dofs, component);
            const double residual = weight * interior_shape[row] *
                (trace_flux[component] + fixed_scalar_tau *
                    (interior_state[component] - trace_state[component]));
            candidate_r[static_cast<std::size_t>(element) * ni + local_row] += residual;
          }
        }
        for (int trace_row = 0; trace_row < trace_scalar_dofs; ++trace_row)
        {
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local_row = trace_offset +
                ByNodesIndex(trace_row, trace_scalar_dofs, component);
            const double residual = weight * trace_shape[trace_row] *
                (trace_flux[component] + fixed_scalar_tau *
                    (interior_state[component] - trace_state[component]));
            candidate_g[static_cast<std::size_t>(element) * nt + local_row] += residual;
          }
        }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          for (int column = 0; column < scalar_dofs; ++column)
          {
            for (int r = 0; r < conserved_components; ++r)
            {
              for (int c = 0; c < conserved_components; ++c)
              {
                const int local_row = ByNodesIndex(row, scalar_dofs, r);
                const int local_column = ByNodesIndex(column, scalar_dofs, c);
                if (r == c)
                {
                  candidate_a[a_offset + static_cast<std::size_t>(local_row) * ni +
                              local_column] += weight * interior_shape[row] *
                                               fixed_scalar_tau * interior_shape[column];
                }
              }
            }
          }
        }
        for (int row = 0; row < scalar_dofs; ++row)
          for (int column = 0; column < trace_scalar_dofs; ++column)
            for (int r = 0; r < conserved_components; ++r)
              for (int c = 0; c < conserved_components; ++c)
              {
                const int local_row = ByNodesIndex(row, scalar_dofs, r);
                const int local_column = trace_offset +
                    ByNodesIndex(column, trace_scalar_dofs, c);
                candidate_b[static_cast<std::size_t>(element) * ni * nt +
                            static_cast<std::size_t>(local_row) * nt + local_column] +=
                    weight * interior_shape[row] * trace_shape[column] *
                    (trace_jacobian[r][c] - (r == c ? fixed_scalar_tau : 0.0));
              }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int column = 0; column < scalar_dofs; ++column)
            for (int r = 0; r < conserved_components; ++r)
            {
              const int local_row = trace_offset + ByNodesIndex(row, trace_scalar_dofs, r);
              const int local_column = ByNodesIndex(column, scalar_dofs, r);
              candidate_c[static_cast<std::size_t>(element) * nt * ni +
                          static_cast<std::size_t>(local_row) * ni + local_column] +=
                  weight * trace_shape[row] * fixed_scalar_tau * interior_shape[column];
            }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int column = 0; column < trace_scalar_dofs; ++column)
            for (int r = 0; r < conserved_components; ++r)
              for (int c = 0; c < conserved_components; ++c)
              {
                const int local_row = trace_offset + ByNodesIndex(row, trace_scalar_dofs, r);
                const int local_column = trace_offset +
                    ByNodesIndex(column, trace_scalar_dofs, c);
                candidate_d[static_cast<std::size_t>(element) * nt * nt +
                            static_cast<std::size_t>(local_row) * nt + local_column] +=
                    weight * trace_shape[row] * trace_shape[column] *
                    (trace_jacobian[r][c] - (r == c ? fixed_scalar_tau : 0.0));
              }
      }
      trace_offset += face_vdofs.Size();
    }
    if (trace_offset != nt) return false;
  }
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  pipeline_state.batches = batches;
  pipeline_state.interior_dofs = ni;
  pipeline_state.trace_dofs = nt;
  pipeline_state.interior_residual = View(candidate_r.data(), batches, ni, 1);
  pipeline_state.trace_residual = View(candidate_g.data(), batches, nt, 1);
  pipeline_state.a = View(candidate_a.data(), batches, ni, ni);
  pipeline_state.b = View(candidate_b.data(), batches, ni, nt);
  pipeline_state.c = View(candidate_c.data(), batches, nt, ni);
  pipeline_state.d = View(candidate_d.data(), batches, nt, nt);
  LocalResidualRequest<SmoothEulerSpec> residual_request;
  LocalResidualResult<SmoothEulerSpec> residual_result;
  LocalJacobianRequest<SmoothEulerSpec> jacobian_request;
  LocalJacobianResult<SmoothEulerSpec> jacobian_result;
  implementation_->pipeline->EvaluateLocalResidual(residual_request, residual_result);
  implementation_->pipeline->EvaluateLocalJacobian(jacobian_request, jacobian_result);
  const auto& returned = jacobian_result.linearization;
  if (!pipeline_state.residual_published || !pipeline_state.jacobian_published ||
      !HasExtents(residual_result.interior_residual, batches, ni, 1) ||
      !HasExtents(residual_result.trace_residual, batches, nt, 1) ||
      returned.interior_residual.data != residual_result.interior_residual.data ||
      returned.trace_residual.data != residual_result.trace_residual.data ||
      !HasExtents(returned.A, batches, ni, ni) ||
      !HasExtents(returned.B, batches, ni, nt) ||
      !HasExtents(returned.C, batches, nt, ni) ||
      !HasExtents(returned.D, batches, nt, nt)) return false;

  // Publish exactly what the static Pipeline returned.  The snapshot keeps
  // f/g for the Newton convention, but Pipeline state remains physical R/G.
  implementation_->residual_r.resize(static_cast<std::size_t>(batches) * ni);
  implementation_->residual_g.resize(static_cast<std::size_t>(batches) * nt);
  implementation_->a.resize(static_cast<std::size_t>(batches) * ni * ni);
  implementation_->b.resize(static_cast<std::size_t>(batches) * ni * nt);
  implementation_->c.resize(static_cast<std::size_t>(batches) * nt * ni);
  implementation_->d.resize(static_cast<std::size_t>(batches) * nt * nt);
  implementation_->f.resize(static_cast<std::size_t>(batches) * ni);
  implementation_->g.resize(static_cast<std::size_t>(batches) * nt);
  for (int batch = 0; batch < batches; ++batch)
  {
    for (int row = 0; row < ni; ++row)
    {
      const double value = residual_result.interior_residual(batch, row, 0);
      implementation_->residual_r[static_cast<std::size_t>(batch) * ni + row] = value;
      implementation_->f[static_cast<std::size_t>(batch) * ni + row] = -value;
      for (int column = 0; column < ni; ++column)
        implementation_->a[static_cast<std::size_t>(batch) * ni * ni +
                           static_cast<std::size_t>(row) * ni + column] =
            returned.A(batch, row, column);
      for (int column = 0; column < nt; ++column)
        implementation_->b[static_cast<std::size_t>(batch) * ni * nt +
                           static_cast<std::size_t>(row) * nt + column] =
            returned.B(batch, row, column);
    }
    for (int row = 0; row < nt; ++row)
    {
      const double value = residual_result.trace_residual(batch, row, 0);
      implementation_->residual_g[static_cast<std::size_t>(batch) * nt + row] = value;
      implementation_->g[static_cast<std::size_t>(batch) * nt + row] = -value;
      for (int column = 0; column < ni; ++column)
        implementation_->c[static_cast<std::size_t>(batch) * nt * ni +
                           static_cast<std::size_t>(row) * ni + column] =
            returned.C(batch, row, column);
      for (int column = 0; column < nt; ++column)
        implementation_->d[static_cast<std::size_t>(batch) * nt * nt +
                           static_cast<std::size_t>(row) * nt + column] =
            returned.D(batch, row, column);
    }
  }
  implementation_->linearization_generation = implementation_->generation;
  pipeline_state.interior_residual = View(implementation_->residual_r.data(),
                                          batches, ni, 1);
  pipeline_state.trace_residual = View(implementation_->residual_g.data(),
                                       batches, nt, 1);
  pipeline_state.a = View(implementation_->a.data(), batches, ni, ni);
  pipeline_state.b = View(implementation_->b.data(), batches, ni, nt);
  pipeline_state.c = View(implementation_->c.data(), batches, nt, ni);
  pipeline_state.d = View(implementation_->d.data(), batches, nt, nt);
  return true;
}

bool SmoothEulerExperiment::CondenseAndAssembleTrace()
{
  const int batches = implementation_->elements, ni = implementation_->local_interior,
            nt = implementation_->local_trace;
  implementation_->condensation_generation = 0;
  implementation_->solve_generation = 0;
  implementation_->recovery_generation = 0;
  implementation_->schur.clear();
  implementation_->reduced_rhs.clear();
  implementation_->schur_action.clear();
  implementation_->global_matrix.clear();
  implementation_->trace_matrix.reset();
  implementation_->trace_solve_matrix.reset();
  implementation_->trace_rhs.SetSize(0);
  implementation_->trace_increment.SetSize(0);
  implementation_->trace_preconditioner.reset();
  implementation_->recovered.clear();
  if (implementation_->linearization_generation == 0 ||
      implementation_->linearization_generation != implementation_->generation ||
      batches <= 0 || ni <= 0 || nt <= 0 ||
      implementation_->residual_r.size() != static_cast<std::size_t>(batches) * ni ||
      implementation_->residual_g.size() != static_cast<std::size_t>(batches) * nt ||
      implementation_->a.size() != static_cast<std::size_t>(batches) * ni * ni ||
      implementation_->b.size() != static_cast<std::size_t>(batches) * ni * nt ||
      implementation_->c.size() != static_cast<std::size_t>(batches) * nt * ni ||
      implementation_->d.size() != static_cast<std::size_t>(batches) * nt * nt ||
      implementation_->f.size() != static_cast<std::size_t>(batches) * ni ||
      implementation_->g.size() != static_cast<std::size_t>(batches) * nt ||
      implementation_->trace_dofs.size() !=
          static_cast<std::size_t>(batches) * nt || !implementation_->trace_space)
  {
    return false;
  }
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  pipeline_state.batches = batches;
  pipeline_state.interior_dofs = ni;
  pipeline_state.trace_dofs = nt;
  pipeline_state.face_block_size = implementation_->trace_face_block_size;
  pipeline_state.interior_residual = View(implementation_->residual_r.data(),
                                          batches, ni, 1);
  pipeline_state.trace_residual = View(implementation_->residual_g.data(),
                                       batches, nt, 1);
  pipeline_state.a = View(implementation_->a.data(), batches, ni, ni);
  pipeline_state.b = View(implementation_->b.data(), batches, ni, nt);
  pipeline_state.c = View(implementation_->c.data(), batches, nt, ni);
  pipeline_state.d = View(implementation_->d.data(), batches, nt, nt);
  pipeline_state.condensation_engine = &implementation_->condensation;
  pipeline_state.trace_space = implementation_->trace_space.get();
  pipeline_state.signed_trace_dofs = &implementation_->trace_dofs;
  pipeline_state.trace_permutation = &implementation_->trace_permutation;
  pipeline_state.newton_f = &implementation_->f;
  pipeline_state.newton_g = &implementation_->g;
  pipeline_state.schur = &implementation_->schur;
  pipeline_state.reduced_rhs = &implementation_->reduced_rhs;
  pipeline_state.schur_action = &implementation_->schur_action;
  pipeline_state.recovered = &implementation_->recovered;
  pipeline_state.global_matrix = &implementation_->global_matrix;
  pipeline_state.trace_matrix = &implementation_->trace_matrix;
  pipeline_state.trace_solve_matrix = &implementation_->trace_solve_matrix;
  pipeline_state.trace_rhs = &implementation_->trace_rhs;
  pipeline_state.trace_increment = &implementation_->trace_increment;
  pipeline_state.trace_preconditioner = &implementation_->trace_preconditioner;
  CondenseRequest<SmoothEulerSpec> condense_request;
  condense_request.linearization = {
      pipeline_state.interior_residual, pipeline_state.trace_residual,
      pipeline_state.a, pipeline_state.b, pipeline_state.c, pipeline_state.d};
  CondenseResult<SmoothEulerSpec> condense_result;
  implementation_->pipeline->CondenseLocalSystem(condense_request, condense_result);
  if (!pipeline_state.condensed ||
      !HasExtents(condense_result.contribution.schur, batches, nt, nt) ||
      !HasExtents(condense_result.contribution.right_hand_side, batches, nt, 1))
  {
    return false;
  }
  TraceAssemblyRequest<SmoothEulerSpec> assembly_request;
  assembly_request.contribution = condense_result.contribution;
  TraceAssemblyResult<SmoothEulerSpec> assembly_result;
  implementation_->pipeline->AssembleTraceSystem(assembly_request, assembly_result);
  const int global_dofs = implementation_->trace_space->GetVSize();
  if (!pipeline_state.assembled || !implementation_->trace_matrix ||
      !HasExtents(assembly_result.trace_system, 1, global_dofs, global_dofs) ||
      !HasExtents(assembly_result.right_hand_side, 1, global_dofs, 1))
  {
    return false;
  }
  // These are the returned assembly products consumed by the later real solve.
  pipeline_state.assembled_system = assembly_result.trace_system;
  pipeline_state.assembled_rhs = assembly_result.right_hand_side;
  implementation_->condensation_generation = implementation_->linearization_generation;
  return true;
}

bool SmoothEulerExperiment::SolveTrace()
{
  if (!implementation_->trace_matrix || implementation_->condensation_generation == 0 ||
      implementation_->condensation_generation != implementation_->linearization_generation ||
      implementation_->linearization_generation != implementation_->generation) return false;
  implementation_->trace_increment.SetSize(0);
  implementation_->recovered.clear();
  implementation_->trace_preconditioner.reset();
  implementation_->trace_solve_matrix.reset();
  implementation_->solve_generation = 0;
  implementation_->recovery_generation = 0;
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  TraceSolveRequest<SmoothEulerSpec> solve_request;
  solve_request.trace_system = AsConst(pipeline_state.assembled_system);
  solve_request.right_hand_side = AsConst(pipeline_state.assembled_rhs);
  TraceSolveResult<SmoothEulerSpec> solve_result;
  implementation_->pipeline->SolveGlobalTraceSystem(solve_request, solve_result);
  implementation_->last_trace_solve_diagnostic =
      pipeline_state.trace_solve_diagnostic;
  const int global_dofs = implementation_->trace_matrix->Height();
  if (!pipeline_state.solved ||
      !HasExtents(solve_result.trace_solution, 1, global_dofs, 1) ||
      solve_result.trace_solution.data != implementation_->trace_increment.GetData())
  {
    return false;
  }
  pipeline_state.trace_solution = solve_result.trace_solution;
  implementation_->solve_generation = implementation_->condensation_generation;
  return true;
}

bool SmoothEulerExperiment::RecoverInterior()
{
  const int batches = implementation_->elements;
  const int ni = implementation_->local_interior;
  if (implementation_->trace_increment.Size() == 0 || implementation_->solve_generation == 0 ||
      implementation_->solve_generation != implementation_->condensation_generation ||
      implementation_->solve_generation != implementation_->generation) return false;
  implementation_->recovered.clear();
  implementation_->recovery_generation = 0;
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  RecoveryRequest<SmoothEulerSpec> recovery_request;
  recovery_request.trace_solution = AsConst(pipeline_state.trace_solution);
  RecoveryResult<SmoothEulerSpec> recovery_result;
  implementation_->pipeline->RecoverInterior(recovery_request, recovery_result);
  if (!pipeline_state.recovered_state ||
      !HasExtents(recovery_result.interior_solution, batches, ni, 1) ||
      recovery_result.interior_solution.data != implementation_->recovered.data())
  {
    return false;
  }
  implementation_->recovery_generation = implementation_->solve_generation;
  return true;
}

bool SmoothEulerExperiment::SolveDampedNewton(NewtonEvidence& evidence)
{
  implementation_->solved_evidence = {};
  implementation_->rollback_probe_mode = false;
  const bool success = SolveNewtonTransaction(
      evidence, &implementation_->solved_evidence);
  implementation_->has_solved_evidence = success;
  return success;
}

bool SmoothEulerExperiment::IntegrateErrorEvidence(
    const int quadrature_order, ErrorIntegralEvidence& volume,
    ErrorIntegralEvidence& trace) const
{
  volume = {};
  trace = {};
  if (!implementation_->mesh || !implementation_->interior_space ||
      !implementation_->trace_space || !implementation_->current_interior ||
      !implementation_->current_trace || quadrature_order <= 0)
  {
    return false;
  }
  long double volume_error = 0.0L, volume_norm = 0.0L, volume_measure = 0.0L;
  for (int element = 0; element < implementation_->mesh->GetNE(); ++element)
  {
    const mfem::FiniteElement* fe = implementation_->interior_space->GetFE(element);
    mfem::ElementTransformation* transformation =
        implementation_->mesh->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr) return false;
    mfem::Array<int> vdofs;
    implementation_->interior_space->GetElementVDofs(element, vdofs);
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      mfem::Vector physical(dimension);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      ConservedState numerical{}, exact{};
      if (!InterpolateByNodes(*implementation_->current_interior, vdofs, shape,
                              numerical) ||
          !ManufacturedConservative(physical[0], physical[1], exact).ok())
      {
        return false;
      }
      const double weight = point.weight * transformation->Weight();
      if (!(weight > 0.0) || !std::isfinite(weight)) return false;
      volume_measure += weight;
      for (int component = 0; component < conserved_components; ++component)
      {
        const long double difference = numerical[component] - exact[component];
        volume_error += weight * difference * difference;
        volume_norm += weight * static_cast<long double>(exact[component]) *
                       exact[component];
      }
    }
  }

  long double trace_error = 0.0L, trace_norm = 0.0L, trace_measure = 0.0L;
  for (int face = 0; face < implementation_->mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* fe = implementation_->trace_space->GetFaceElement(face);
    mfem::ElementTransformation* transformation =
        implementation_->mesh->GetFaceTransformation(face);
    if (fe == nullptr || transformation == nullptr) return false;
    mfem::Array<int> vdofs;
    implementation_->trace_space->GetFaceVDofs(face, vdofs);
    mfem::Vector shape(fe->GetDof());
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    for (int q = 0; q < rule.GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      mfem::Vector scaled_normal(dimension);
      mfem::CalcOrtho(transformation->Jacobian(), scaled_normal);
      const double normal_magnitude = scaled_normal.Norml2();
      if (!(normal_magnitude > 0.0) || !std::isfinite(normal_magnitude))
      {
        return false;
      }
      const std::array<double, dimension> normal{{
          scaled_normal[0] / normal_magnitude,
          scaled_normal[1] / normal_magnitude}};
      if (!IsFinite(normal)) return false;
      mfem::Vector physical(dimension);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      ConservedState numerical{}, exact{};
      if (!InterpolateByNodes(*implementation_->current_trace, vdofs, shape,
                              numerical) ||
          !ManufacturedConservative(physical[0], physical[1], exact).ok())
      {
        return false;
      }
      // CalcOrtho validates and normalizes the physical normal.  The face
      // transformation supplies the unique-face quadrature measure once.
      const double face_measure = transformation->Weight();
      if (!(face_measure > 0.0) || !std::isfinite(face_measure)) return false;
      const double weight = point.weight * face_measure;
      if (!(weight > 0.0) || !std::isfinite(weight)) return false;
      trace_measure += weight;
      for (int component = 0; component < conserved_components; ++component)
      {
        const long double difference = numerical[component] - exact[component];
        trace_error += weight * difference * difference;
        trace_norm += weight * static_cast<long double>(exact[component]) *
                      exact[component];
      }
    }
  }
  auto publish = [](const long double error, const long double norm,
                    const long double measure, const int faces,
                    ErrorIntegralEvidence& output)
  {
    if (!(error > 0.0L) || !(norm > 0.0L) || !(measure > 0.0L)) return false;
    output.error_squared = static_cast<double>(error);
    output.exact_norm_squared = static_cast<double>(norm);
    output.measure = static_cast<double>(measure);
    output.direct_l2 = std::sqrt(output.error_squared);
    output.rms = std::sqrt(output.error_squared / output.measure);
    output.relative_l2 = std::sqrt(output.error_squared / output.exact_norm_squared);
    output.unique_face_count = faces;
    return std::isfinite(output.error_squared) &&
           std::isfinite(output.exact_norm_squared) &&
           std::isfinite(output.measure) && std::isfinite(output.direct_l2) &&
           std::isfinite(output.rms) && std::isfinite(output.relative_l2);
  };
  return publish(volume_error, volume_norm, volume_measure, 0, volume) &&
         publish(trace_error, trace_norm, trace_measure,
                 implementation_->mesh->GetNumFaces(), trace);
}

bool SmoothEulerExperiment::SolveNewtonTransaction(
    NewtonEvidence& evidence, SolvedMeshEvidence* solved_evidence)
{
  evidence = {};
  if (!implementation_->mesh || !implementation_->interior_space ||
      !implementation_->trace_space || !implementation_->old_interior ||
      !implementation_->old_trace || !implementation_->current_interior ||
      !implementation_->current_trace)
  {
    evidence.status = ExperimentStatus::SetupFailure;
    return false;
  }

  SolvedMeshEvidence local_solved{};
  SolvedMeshEvidence& solved = solved_evidence == nullptr ? local_solved : *solved_evidence;
  solved = {};
  solved.polynomial_order = polynomial_order_;
  solved.elements_per_direction = implementation_->elements_per_direction;
  solved.q_a = quadrature_order_;
  solved.q_e = 4 * polynomial_order_ + 12;
  solved.q_e_plus_four = solved.q_e + 4;
  solved.old_interior = GridValues(*implementation_->old_interior);
  solved.old_trace = GridValues(*implementation_->old_trace);
  solved.initial_interior = GridValues(*implementation_->current_interior);
  solved.initial_trace = GridValues(*implementation_->current_trace);
  const std::vector<double> entry_interior = solved.initial_interior;
  const std::vector<double> entry_trace = solved.initial_trace;

  auto fail = [&](const ExperimentStatus status)
  {
    (void)AssignGridValues(*implementation_->current_interior, entry_interior);
    (void)AssignGridValues(*implementation_->current_trace, entry_trace);
    implementation_->InvalidateNumericalState();
    evidence.status = status;
    evidence.admissible = false;
    solved.status = status;
    solved.newton = evidence;
    solved.final_interior.clear();
    solved.final_trace.clear();
    solved.internal_consistency = false;
    return false;
  };

  if (!FiniteVector(solved.old_interior) || !FiniteVector(solved.old_trace) ||
      !FiniteVector(solved.initial_interior) ||
      !FiniteVector(solved.initial_trace))
  {
    return fail(ExperimentStatus::EvidenceConsistencyFailure);
  }

  AdmissibilityEvidence initial_admissibility{};
  const bool initial_screen_evaluated = PrescreenFields(
      *implementation_->mesh, *implementation_->interior_space,
      *implementation_->trace_space, *implementation_->current_interior,
      *implementation_->current_trace, quadrature_order_, initial_admissibility);
  solved.initial_admissibility = initial_admissibility;
  if (!initial_screen_evaluated || !initial_admissibility.passed)
  {
    evidence.minimum_density = initial_admissibility.minimum_density;
    evidence.minimum_pressure = initial_admissibility.minimum_pressure;
    evidence.maximum_wave_speed = initial_admissibility.maximum_wave_speed;
    return fail(ExperimentStatus::InitialAdmissibilityFailure);
  }
  evidence.initial_prescreen_passed = true;
  evidence.minimum_density = initial_admissibility.minimum_density;
  evidence.minimum_pressure = initial_admissibility.minimum_pressure;
  evidence.maximum_wave_speed = initial_admissibility.maximum_wave_speed;

  ++evidence.assembly_attempts;
  if (!AssembleLinearization()) return fail(ExperimentStatus::AssemblyFailure);
  double residual = ResidualNorm(
      implementation_->residual_r, implementation_->residual_g,
      implementation_->trace_dofs, implementation_->trace_space->GetVSize());
  if (!std::isfinite(residual)) return fail(ExperimentStatus::AssemblyFailure);
  evidence.initial_residual = residual;
  evidence.final_residual = residual;
  solved.initial_residual_interior = implementation_->residual_r;
  solved.initial_residual_trace = implementation_->residual_g;
  solved.signed_trace_dofs = implementation_->trace_dofs;
  solved.residual_history.push_back(residual);
  const std::vector<double> initial_a = implementation_->a;
  const std::vector<double> initial_b = implementation_->b;
  const std::vector<double> initial_c = implementation_->c;
  const std::vector<double> initial_d = implementation_->d;
  if (!(residual > 1.0e-8))
  {
    return fail(ExperimentStatus::EvidenceConsistencyFailure);
  }

  // Capture the first element of the exact production assembly generation.
  {
    LocalAssemblyAuditSnapshot& audit = solved.local_assembly;
    audit.polynomial_order = polynomial_order_;
    audit.elements_per_direction = implementation_->elements_per_direction;
    audit.element = 0;
    audit.quadrature_order = quadrature_order_;
    const mfem::FiniteElement* fe = implementation_->interior_space->GetFE(0);
    mfem::Array<int> interior_vdofs;
    implementation_->interior_space->GetElementVDofs(0, interior_vdofs);
    if (fe == nullptr || interior_vdofs.Size() != implementation_->local_interior)
    {
      return fail(ExperimentStatus::EvidenceConsistencyFailure);
    }
    const mfem::IntegrationRule& volume_rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order_);
    audit.volume_quadrature_points = volume_rule.GetNPoints();
    audit.interior_dofs = implementation_->local_interior;
    audit.trace_dofs = implementation_->local_trace;
    audit.old_interior.resize(audit.interior_dofs);
    audit.current_interior.resize(audit.interior_dofs);
    for (int local = 0; local < audit.interior_dofs; ++local)
    {
      double sign = 1.0;
      const int global = DecodeSignedVDof(interior_vdofs[local], sign);
      if (global < 0 || global >= implementation_->old_interior->Size())
      {
        return fail(ExperimentStatus::EvidenceConsistencyFailure);
      }
      audit.old_interior[local] = sign * (*implementation_->old_interior)[global];
      audit.current_interior[local] =
          sign * (*implementation_->current_interior)[global];
    }
    audit.trace.resize(audit.trace_dofs);
    audit.signed_trace_dofs.assign(implementation_->trace_dofs.begin(),
                                   implementation_->trace_dofs.begin() +
                                       audit.trace_dofs);
    for (int local = 0; local < audit.trace_dofs; ++local)
    {
      double sign = 1.0;
      const int global = DecodeSignedVDof(audit.signed_trace_dofs[local], sign);
      if (global < 0 || global >= implementation_->current_trace->Size())
      {
        return fail(ExperimentStatus::EvidenceConsistencyFailure);
      }
      audit.trace[local] = sign * (*implementation_->current_trace)[global];
    }
    audit.residual_interior.assign(implementation_->residual_r.begin(),
                                   implementation_->residual_r.begin() +
                                       audit.interior_dofs);
    audit.residual_trace.assign(implementation_->residual_g.begin(),
                                implementation_->residual_g.begin() +
                                    audit.trace_dofs);
    audit.a.assign(implementation_->a.begin(), implementation_->a.begin() +
        static_cast<std::size_t>(audit.interior_dofs) * audit.interior_dofs);
    audit.b.assign(implementation_->b.begin(), implementation_->b.begin() +
        static_cast<std::size_t>(audit.interior_dofs) * audit.trace_dofs);
    audit.c.assign(implementation_->c.begin(), implementation_->c.begin() +
        static_cast<std::size_t>(audit.trace_dofs) * audit.interior_dofs);
    audit.d.assign(implementation_->d.begin(), implementation_->d.begin() +
        static_cast<std::size_t>(audit.trace_dofs) * audit.trace_dofs);
    mfem::Array<int> faces, orientations;
    implementation_->mesh->GetElementEdges(0, faces, orientations);
    audit.face_quadrature_points = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const mfem::FiniteElement* trace_fe =
          implementation_->trace_space->GetFaceElement(faces[local_face]);
      if (trace_fe == nullptr)
      {
        return fail(ExperimentStatus::EvidenceConsistencyFailure);
      }
      audit.face_quadrature_points +=
          mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order_).GetNPoints();
    }
    audit.interior_reconstruction_id = HashValues(audit.current_interior) ^
        (static_cast<std::uint64_t>(quadrature_order_) << 32);
    audit.trace_reconstruction_id = HashValues(audit.trace) ^
        static_cast<std::uint64_t>(audit.face_quadrature_points);
  }

  const double target = 1.0e-10 * (1.0 + evidence.initial_residual);
  bool captured_first_rejected_trial = false;
  bool captured_first_accepted_trial = false;
  for (int nonlinear_iteration = 0; nonlinear_iteration < 20; ++nonlinear_iteration)
  {
    if (evidence.nonlinear_iterations >= 2 && residual <= target) break;
    implementation_->last_trace_solve_diagnostic = {};
    const std::vector<double> base_a = implementation_->a;
    const std::vector<double> base_b = implementation_->b;
    const std::vector<double> base_c = implementation_->c;
    const std::vector<double> base_d = implementation_->d;
    const std::vector<double> base_interior =
        GridValues(*implementation_->current_interior);
    const std::vector<double> base_trace =
        GridValues(*implementation_->current_trace);

    if (!CondenseAndAssembleTrace())
    {
      return fail(ExperimentStatus::CondensationFailure);
    }
    LinearSolveEvidence linear{};
    linear.nonlinear_iteration = nonlinear_iteration;
    linear.right_hand_side_norm = implementation_->trace_rhs.Norml2();
    if (!SolveTrace())
    {
      linear.gmres_iterations = implementation_->pipeline_state.gmres_iterations;
      linear.reported_relative_residual =
          implementation_->pipeline_state.reported_relative_residual;
      linear.true_relative_residual =
          implementation_->pipeline_state.true_relative_residual;
      linear.status = implementation_->pipeline_state.gmres_converged ?
          ExperimentStatus::LinearResidualFailure : ExperimentStatus::GmresFailure;
      solved.linear_history.push_back(linear);
      return fail(linear.status);
    }
    linear.gmres_iterations = implementation_->pipeline_state.gmres_iterations;
    linear.reported_relative_residual =
        implementation_->pipeline_state.reported_relative_residual;
    linear.true_relative_residual =
        implementation_->pipeline_state.true_relative_residual;
    linear.status = ExperimentStatus::Success;
    solved.linear_history.push_back(linear);
    evidence.linear_iterations += linear.gmres_iterations;
    evidence.trace_relative_residual = linear.true_relative_residual;

    if (implementation_->elements_per_direction == 3 &&
        nonlinear_iteration == 0)
    {
      LinearAuditSnapshot& audit = solved.linear_audit;
      audit.polynomial_order = polynomial_order_;
      audit.elements_per_direction = implementation_->elements_per_direction;
      audit.nonlinear_iteration = nonlinear_iteration;
      audit.rows = implementation_->trace_matrix->Height();
      const int* offsets = implementation_->trace_matrix->GetI();
      const int* columns = implementation_->trace_matrix->GetJ();
      const double* values = implementation_->trace_matrix->GetData();
      if (offsets == nullptr || columns == nullptr || values == nullptr)
      {
        return fail(ExperimentStatus::EvidenceConsistencyFailure);
      }
      audit.row_offsets.assign(offsets, offsets + audit.rows + 1);
      const int nonzeros = audit.row_offsets.back();
      audit.columns.assign(columns, columns + nonzeros);
      audit.values.assign(values, values + nonzeros);
      audit.right_hand_side.assign(implementation_->trace_rhs.GetData(),
          implementation_->trace_rhs.GetData() + implementation_->trace_rhs.Size());
      audit.trace_increment.assign(implementation_->trace_increment.GetData(),
          implementation_->trace_increment.GetData() +
              implementation_->trace_increment.Size());
      audit.true_relative_residual = linear.true_relative_residual;
      solved.has_linear_audit = true;
    }

    if (!RecoverInterior()) return fail(ExperimentStatus::RecoveryFailure);
    std::vector<double> delta_interior(base_interior.size(), 0.0);
    for (int element = 0; element < implementation_->elements; ++element)
    {
      mfem::Array<int> vdofs;
      implementation_->interior_space->GetElementVDofs(element, vdofs);
      if (vdofs.Size() != implementation_->local_interior)
      {
        return fail(ExperimentStatus::RecoveryFailure);
      }
      for (int local = 0; local < vdofs.Size(); ++local)
      {
        double sign = 1.0;
        const int global = DecodeSignedVDof(vdofs[local], sign);
        if (global < 0 || global >= static_cast<int>(delta_interior.size()))
        {
          return fail(ExperimentStatus::RecoveryFailure);
        }
        delta_interior[global] = sign * implementation_->recovered[
            static_cast<std::size_t>(element) * implementation_->local_interior +
            local];
      }
    }
    const std::vector<double> delta_trace(
        implementation_->trace_increment.GetData(),
        implementation_->trace_increment.GetData() +
            implementation_->trace_increment.Size());

    bool accepted = false;
    for (int trial_index = 0; trial_index <= 12; ++trial_index)
    {
      const double alpha = std::ldexp(1.0, -trial_index);
      std::vector<double> trial_interior(base_interior.size());
      std::vector<double> trial_trace(base_trace.size());
      double formula_error = 0.0;
      for (std::size_t index = 0; index < trial_interior.size(); ++index)
      {
        trial_interior[index] = base_interior[index] + alpha * delta_interior[index];
        formula_error = std::max(formula_error, std::abs(
            trial_interior[index] -
            (base_interior[index] + alpha * delta_interior[index])));
      }
      for (std::size_t index = 0; index < trial_trace.size(); ++index)
      {
        trial_trace[index] = base_trace[index] + alpha * delta_trace[index];
        formula_error = std::max(formula_error, std::abs(
            trial_trace[index] - (base_trace[index] + alpha * delta_trace[index])));
      }
      if (!AssignGridValues(*implementation_->current_interior, trial_interior) ||
          !AssignGridValues(*implementation_->current_trace, trial_trace))
      {
        return fail(ExperimentStatus::EvidenceConsistencyFailure);
      }
      implementation_->InvalidateNumericalState();

      TrialEvidence trial{};
      trial.nonlinear_iteration = nonlinear_iteration;
      trial.trial_index = trial_index;
      trial.alpha = alpha;
      trial.status = ExperimentStatus::Success;
      trial.base_residual = residual;
      trial.armijo_boundary = (1.0 - 1.0e-4 * alpha) * residual;
      trial.coefficient_update = std::max(
          MaxAbsDifference(base_interior, trial_interior),
          MaxAbsDifference(base_trace, trial_trace));
      if (!PrescreenFields(*implementation_->mesh, *implementation_->interior_space,
                           *implementation_->trace_space,
                           *implementation_->current_interior,
                           *implementation_->current_trace, quadrature_order_,
                           trial.admissibility))
      {
        return fail(ExperimentStatus::AssemblyFailure);
      }

      TrialAuditSnapshot audit{};
      audit.nonlinear_iteration = nonlinear_iteration;
      audit.trial_index = trial_index;
      audit.alpha = alpha;
      audit.status = ExperimentStatus::Success;
      audit.base_residual = residual;
      audit.armijo_boundary = trial.armijo_boundary;
      audit.coefficient_update = trial.coefficient_update;
      audit.formula_identity_error = formula_error;
      audit.base_interior = base_interior;
      audit.base_trace = base_trace;
      audit.delta_interior = delta_interior;
      audit.delta_trace = delta_trace;
      audit.trial_interior = trial_interior;
      audit.trial_trace = trial_trace;
      audit.admissibility = trial.admissibility;

      if (!trial.admissibility.passed)
      {
        trial.rejection_reason = trial.admissibility.reason;
        trial.generation_discarded = true;
        audit.rejection_reason = trial.rejection_reason;
        audit.generation_discarded = true;
        audit.post_decision_interior = base_interior;
        audit.post_decision_trace = base_trace;
        ++evidence.rejected_trials;
        solved.trial_history.push_back(trial);
        if (!captured_first_rejected_trial)
        {
          solved.trial_audits.push_back(std::move(audit));
          captured_first_rejected_trial = true;
        }
        (void)AssignGridValues(*implementation_->current_interior, base_interior);
        (void)AssignGridValues(*implementation_->current_trace, base_trace);
        implementation_->InvalidateNumericalState();
        continue;
      }

      trial.reassembly_attempted = true;
      audit.reassembly_attempted = true;
      ++evidence.assembly_attempts;
      if (!AssembleLinearization())
      {
        trial.status = ExperimentStatus::AssemblyFailure;
        audit.status = ExperimentStatus::AssemblyFailure;
        solved.trial_history.push_back(trial);
        return fail(ExperimentStatus::AssemblyFailure);
      }
      const double trial_residual = ResidualNorm(
          implementation_->residual_r, implementation_->residual_g,
          implementation_->trace_dofs, implementation_->trace_space->GetVSize());
      if (!std::isfinite(trial_residual))
      {
        return fail(ExperimentStatus::AssemblyFailure);
      }
      trial.residual_evaluated = true;
      trial.trial_residual_present = true;
      trial.trial_residual = trial_residual;
      audit.residual_evaluated = true;
      audit.trial_residual = trial_residual;
      trial.jacobian_change = std::max(
          std::max(MaxAbsDifference(base_a, implementation_->a),
                   MaxAbsDifference(base_b, implementation_->b)),
          std::max(MaxAbsDifference(base_c, implementation_->c),
                   MaxAbsDifference(base_d, implementation_->d)));
      audit.jacobian_change = trial.jacobian_change;

      if (trial_residual > trial.armijo_boundary)
      {
        trial.rejection_reason = TrialRejectionReason::Armijo;
        trial.generation_discarded = true;
        audit.rejection_reason = trial.rejection_reason;
        audit.generation_discarded = true;
        audit.post_decision_interior = base_interior;
        audit.post_decision_trace = base_trace;
        ++evidence.rejected_trials;
        solved.trial_history.push_back(trial);
        if (!captured_first_rejected_trial)
        {
          solved.trial_audits.push_back(std::move(audit));
          captured_first_rejected_trial = true;
        }
        (void)AssignGridValues(*implementation_->current_interior, base_interior);
        (void)AssignGridValues(*implementation_->current_trace, base_trace);
        implementation_->InvalidateNumericalState();
        continue;
      }

      trial.accepted = true;
      trial.generation_published = true;
      audit.accepted = true;
      audit.generation_published = true;
      audit.post_decision_interior = trial_interior;
      audit.post_decision_trace = trial_trace;
      solved.trial_history.push_back(trial);
      if (!captured_first_accepted_trial)
      {
        evidence.relative_jacobian_change = RelativeJacobianChange(
            initial_a, implementation_->a, initial_b, implementation_->b,
            initial_c, implementation_->c, initial_d, implementation_->d);
        solved.trial_audits.push_back(std::move(audit));
        captured_first_accepted_trial = true;
      }
      residual = trial_residual;
      solved.residual_history.push_back(residual);
      ++evidence.nonlinear_iterations;
      evidence.final_residual = residual;
      evidence.minimum_damping =
          std::min(evidence.minimum_damping, alpha);
      evidence.minimum_density = std::min(
          evidence.minimum_density, trial.admissibility.minimum_density);
      evidence.minimum_pressure = std::min(
          evidence.minimum_pressure, trial.admissibility.minimum_pressure);
      evidence.maximum_wave_speed = std::max(
          evidence.maximum_wave_speed, trial.admissibility.maximum_wave_speed);
      evidence.maximum_coefficient_update = std::max(
          evidence.maximum_coefficient_update, trial.coefficient_update);
      evidence.maximum_jacobian_change = std::max(
          evidence.maximum_jacobian_change, trial.jacobian_change);
      if (evidence.nonlinear_iterations == 1)
      {
        evidence.residual_after_first_accepted = residual;
        evidence.residual_after_first_above_convergence =
            std::isfinite(residual) && residual > target;
      }
      accepted = true;
      break;
    }
    if (!accepted) return fail(ExperimentStatus::LineSearchExhausted);
    if (implementation_->rollback_probe_mode &&
        evidence.nonlinear_iterations == 1)
    {
      if (!std::isfinite(evidence.residual_after_first_accepted) ||
          !evidence.residual_after_first_above_convergence)
      {
        return fail(ExperimentStatus::EvidenceConsistencyFailure);
      }
      return fail(ExperimentStatus::IterationLimit);
    }
  }

  if (evidence.nonlinear_iterations < 2 || residual > target)
  {
    return fail(ExperimentStatus::IterationLimit);
  }
  const std::vector<double> final_interior =
      GridValues(*implementation_->current_interior);
  const std::vector<double> final_trace =
      GridValues(*implementation_->current_trace);
  evidence.relative_coefficient_change = RelativeCoefficientChange(
      solved.initial_interior, final_interior, solved.initial_trace, final_trace);
  if (evidence.linear_iterations < 1 ||
      !std::isfinite(evidence.relative_coefficient_change) ||
      !std::isfinite(evidence.relative_jacobian_change) ||
      !(evidence.relative_coefficient_change >= 1.0e-6) ||
      !(evidence.relative_jacobian_change >= 1.0e-8))
  {
    return fail(ExperimentStatus::EvidenceConsistencyFailure);
  }
  evidence.status = ExperimentStatus::Success;
  evidence.final_residual = residual;
  evidence.admissible = true;
  solved.status = ExperimentStatus::Success;
  solved.newton = evidence;
  solved.final_interior = final_interior;
  solved.final_trace = final_trace;
  solved.final_residual_interior = implementation_->residual_r;
  solved.final_residual_trace = implementation_->residual_g;
  return true;
}

bool SmoothEulerExperiment::RunConvergenceLadder(ConvergenceEvidence& evidence)
{
  evidence = {};
  evidence.polynomial_order = polynomial_order_;
  constexpr std::array<int, 3> widths{{3, 6, 12}};
  evidence.mesh_widths = widths;
  auto fail_ladder = [&](const ExperimentStatus status, const char* phase,
                         const std::size_t diagnostic_mesh_index,
                         const bool newton_reached)
  {
    try
    {
      EmitConvergenceFailureDiagnostic(
          polynomial_order_, diagnostic_mesh_index,
          widths[diagnostic_mesh_index], phase, status, evidence,
          newton_reached, implementation_->last_trace_solve_diagnostic);
    }
    catch (...)
    {
    }
    for (auto solve = evidence.solves.rbegin(); solve != evidence.solves.rend();
         ++solve)
    {
      if (implementation_->current_interior && implementation_->current_trace &&
          solve->initial_interior.size() ==
              static_cast<std::size_t>(implementation_->current_interior->Size()) &&
          solve->initial_trace.size() ==
              static_cast<std::size_t>(implementation_->current_trace->Size()))
      {
        (void)AssignGridValues(*implementation_->current_interior,
                               solve->initial_interior);
        (void)AssignGridValues(*implementation_->current_trace,
                               solve->initial_trace);
        break;
      }
    }
    implementation_->InvalidateNumericalState();
    implementation_->has_solved_evidence = false;
    ConvergenceEvidence failed{};
    failed.status = status;
    failed.polynomial_order = polynomial_order_;
    failed.mesh_widths = widths;
    evidence = std::move(failed);
    return false;
  };
  for (std::size_t mesh_index = 0; mesh_index < widths.size(); ++mesh_index)
  {
    implementation_->last_trace_solve_diagnostic = {};
    SolvedMeshEvidence& solved = evidence.solves[mesh_index];
    solved.polynomial_order = polynomial_order_;
    solved.elements_per_direction = widths[mesh_index];
    if (!BuildPeriodicMesh(widths[mesh_index]))
    {
      return fail_ladder(ExperimentStatus::SetupFailure, "mesh", mesh_index,
                         false);
    }
    if (!ProjectOldStage() || !ProjectPerturbedCurrentStage())
    {
      return fail_ladder(ExperimentStatus::ProjectionFailure, "projection",
                         mesh_index, false);
    }
    NewtonEvidence newton{};
    implementation_->rollback_probe_mode = false;
    if (!SolveNewtonTransaction(newton, &solved))
    {
      return fail_ladder(newton.status, "newton", mesh_index, true);
    }
    if (!IntegrateErrorEvidence(solved.q_e, solved.q_e_volume,
                                solved.q_e_trace) ||
        !IntegrateErrorEvidence(solved.q_e_plus_four,
                                solved.q_e_plus_four_volume,
                                solved.q_e_plus_four_trace))
    {
      return fail_ladder(ExperimentStatus::ErrorIntegrationFailure, "qE",
                         mesh_index, true);
    }
    const auto error_pair_is_valid = [](const ErrorIntegralEvidence& q_e,
                                        const ErrorIntegralEvidence& q_e_plus_four)
    {
      const double difference = std::abs(q_e.relative_l2 -
                                         q_e_plus_four.relative_l2);
      return std::isfinite(q_e.relative_l2) &&
             std::isfinite(q_e_plus_four.relative_l2) &&
             q_e.relative_l2 >= 1.0e-10 &&
             q_e_plus_four.relative_l2 >= 1.0e-10 &&
             difference / std::max(q_e_plus_four.relative_l2, 1.0e-300) <=
                 1.0e-8 &&
             q_e_plus_four.relative_l2 > 100.0 * difference;
    };
    const auto error_integral_is_valid = [](const ErrorIntegralEvidence& value)
    {
      if (!std::isfinite(value.error_squared) ||
          !std::isfinite(value.exact_norm_squared) ||
          !std::isfinite(value.measure) || !std::isfinite(value.direct_l2) ||
          !std::isfinite(value.rms) || !std::isfinite(value.relative_l2) ||
          !(value.error_squared > 0.0) || !(value.exact_norm_squared > 0.0) ||
          !(value.measure > 0.0) || !(value.direct_l2 > 0.0) ||
          !(value.rms > 0.0) || !(value.relative_l2 > 0.0))
      {
        return false;
      }
      const double scale = std::max(1.0, value.error_squared);
      return std::abs(value.direct_l2 * value.direct_l2 -
                      value.error_squared) <=
                 128.0 * experiment_epsilon * scale &&
             std::abs(value.rms * value.rms * value.measure -
                      value.error_squared) <=
                 128.0 * experiment_epsilon * scale &&
             std::abs(value.relative_l2 * value.relative_l2 *
                          value.exact_norm_squared - value.error_squared) <=
                 128.0 * experiment_epsilon * scale;
    };
    const double expected_trace_measure = 2.0 * widths[mesh_index];
    const double trace_measure_tolerance = 128.0 * experiment_epsilon *
        std::max(1.0, expected_trace_measure);
    if (!error_integral_is_valid(solved.q_e_volume) ||
        !error_integral_is_valid(solved.q_e_plus_four_volume) ||
        !error_integral_is_valid(solved.q_e_trace) ||
        !error_integral_is_valid(solved.q_e_plus_four_trace) ||
        !error_pair_is_valid(solved.q_e_volume,
                             solved.q_e_plus_four_volume) ||
        !error_pair_is_valid(solved.q_e_trace,
                             solved.q_e_plus_four_trace) ||
        solved.q_e_trace.unique_face_count !=
            2 * widths[mesh_index] * widths[mesh_index] ||
        solved.q_e_plus_four_trace.unique_face_count !=
            2 * widths[mesh_index] * widths[mesh_index] ||
        std::abs(solved.q_e_volume.measure - 1.0) >
            128.0 * experiment_epsilon ||
        std::abs(solved.q_e_plus_four_volume.measure - 1.0) >
            128.0 * experiment_epsilon ||
        std::abs(solved.q_e_trace.measure - expected_trace_measure) >
            trace_measure_tolerance ||
        std::abs(solved.q_e_plus_four_trace.measure - expected_trace_measure) >
            trace_measure_tolerance)
    {
      return fail_ladder(ExperimentStatus::ErrorFloorFailure, "error-floor",
                         mesh_index, true);
    }
    solved.internal_consistency = true;
    evidence.volume_relative_l2[mesh_index] =
        solved.q_e_plus_four_volume.relative_l2;
    evidence.trace_relative_l2[mesh_index] =
        solved.q_e_plus_four_trace.relative_l2;
    implementation_->solved_evidence = solved;
    implementation_->has_solved_evidence = true;
  }
  for (int interval = 0; interval < 2; ++interval)
  {
    if (!(evidence.volume_relative_l2[interval] >
          evidence.volume_relative_l2[interval + 1]) ||
        !(evidence.trace_relative_l2[interval] >
          evidence.trace_relative_l2[interval + 1]))
    {
      return fail_ladder(ExperimentStatus::RateFailure, "monotonicity",
                         widths.size() - 1, true);
    }
    evidence.volume_rates[interval] = std::log2(
        evidence.volume_relative_l2[interval] /
        evidence.volume_relative_l2[interval + 1]);
    evidence.trace_rates[interval] = std::log2(
        evidence.trace_relative_l2[interval] /
        evidence.trace_relative_l2[interval + 1]);
  }
  const double minimum_volume = static_cast<double>(polynomial_order_) + 0.25;
  const double minimum_trace_pair =
      static_cast<double>(polynomial_order_) + 0.45;
  for (int interval = 0; interval < 2; ++interval)
  {
    if (!std::isfinite(evidence.volume_rates[interval]) ||
        !std::isfinite(evidence.trace_rates[interval]) ||
        evidence.volume_rates[interval] < minimum_volume ||
        evidence.trace_rates[interval] < minimum_trace_pair)
    {
      return fail_ladder(ExperimentStatus::RateFailure, "rate",
                         widths.size() - 1, true);
    }
  }
  const double trace_slope = ThreeLevelTraceSlope(evidence.trace_relative_l2);
  const double minimum_trace_slope =
      static_cast<double>(polynomial_order_) + 0.50;
  if (!std::isfinite(trace_slope) || trace_slope < minimum_trace_slope)
  {
    return fail_ladder(ExperimentStatus::RateFailure, "rate",
                       widths.size() - 1, true);
  }
  evidence.status = ExperimentStatus::Success;
  evidence.passed = true;
  return true;
}

const mfem::SparseMatrix* SmoothEulerExperiment::trace_matrix() const noexcept
{
  return implementation_->trace_matrix.get();
}

bool SmoothEulerExperiment::CopyLocalLinearization(
    LocalLinearizationSnapshot& snapshot) const
{
  snapshot = {};
  const int elements = implementation_->elements;
  const int ni = implementation_->local_interior;
  const int nt = implementation_->local_trace;
  if (implementation_->linearization_generation == 0 ||
      implementation_->linearization_generation != implementation_->generation ||
      elements <= 0 || ni <= 0 || nt <= 0 ||
      implementation_->residual_r.size() != static_cast<std::size_t>(elements) * ni ||
      implementation_->residual_g.size() != static_cast<std::size_t>(elements) * nt ||
      implementation_->a.size() != static_cast<std::size_t>(elements) * ni * ni ||
      implementation_->b.size() != static_cast<std::size_t>(elements) * ni * nt ||
      implementation_->c.size() != static_cast<std::size_t>(elements) * nt * ni ||
      implementation_->d.size() != static_cast<std::size_t>(elements) * nt * nt ||
      implementation_->f.size() != static_cast<std::size_t>(elements) * ni ||
      implementation_->g.size() != static_cast<std::size_t>(elements) * nt ||
      implementation_->trace_dofs.size() != static_cast<std::size_t>(elements) * nt)
  {
    return false;
  }
  snapshot.elements = implementation_->elements;
  snapshot.interior_dofs = implementation_->local_interior;
  snapshot.trace_dofs = implementation_->local_trace;
  snapshot.a = implementation_->a;
  snapshot.b = implementation_->b;
  snapshot.c = implementation_->c;
  snapshot.d = implementation_->d;
  snapshot.f = implementation_->f;
  snapshot.g = implementation_->g;
  snapshot.signed_trace_dofs = implementation_->trace_dofs;
  return true;
}

bool SmoothEulerExperiment::CopyTraceIncrement(std::vector<double>& increment) const
{
  increment.clear();
  if (implementation_->solve_generation == 0 ||
      implementation_->solve_generation != implementation_->condensation_generation ||
      implementation_->solve_generation != implementation_->generation ||
      implementation_->trace_increment.Size() == 0) return false;
  increment.assign(implementation_->trace_increment.GetData(),
                   implementation_->trace_increment.GetData() +
                       implementation_->trace_increment.Size());
  return true;
}

bool SmoothEulerExperiment::CopyTraceRightHandSide(
    std::vector<double>& right_hand_side) const
{
  right_hand_side.clear();
  if (!implementation_->trace_matrix || implementation_->condensation_generation == 0 ||
      implementation_->condensation_generation != implementation_->generation ||
      implementation_->trace_rhs.Size() != implementation_->trace_matrix->Height()) return false;
  right_hand_side.assign(implementation_->trace_rhs.GetData(),
                         implementation_->trace_rhs.GetData() +
                             implementation_->trace_rhs.Size());
  return true;
}

const mfem::FiniteElementSpace* SmoothEulerExperiment::trace_space() const noexcept
{
  return implementation_->trace_space.get();
}

const mfem::GridFunction* SmoothEulerExperiment::old_interior() const noexcept
{
  return implementation_->old_interior.get();
}

const mfem::GridFunction* SmoothEulerExperiment::old_trace() const noexcept
{
  return implementation_->old_trace.get();
}

const mfem::GridFunction* SmoothEulerExperiment::current_interior() const noexcept
{
  return implementation_->current_interior.get();
}

const mfem::GridFunction* SmoothEulerExperiment::current_trace() const noexcept
{
  return implementation_->current_trace.get();
}

const SolvedMeshEvidence* SmoothEulerExperiment::solved_evidence() const noexcept
{
  return implementation_->has_solved_evidence ?
      &implementation_->solved_evidence : nullptr;
}

bool RunSmoothEulerExperiment(ExperimentEvidence& evidence)
{
  evidence = {};
  // The probe selector and transaction call remain source-local.  This invokes
  // the same Newton transaction as an ordinary solve; it exposes no public or
  // header-visible one-step control.
  const auto run_rollback_probe = [](SmoothEulerExperiment& experiment,
                                     RollbackAuditEvidence& rollback)
  {
    rollback = {};
    if (experiment.polynomial_order_ != 1 || !experiment.BuildPeriodicMesh(3))
    {
      rollback.status = ExperimentStatus::SetupFailure;
      try
      {
        EmitRollbackFailureDiagnostic(
            "mesh", rollback, nullptr, nullptr, {});
      }
      catch (...)
      {
      }
      return false;
    }
    if (!experiment.ProjectOldStage() ||
        !experiment.ProjectPerturbedCurrentStage())
    {
      rollback.status = ExperimentStatus::ProjectionFailure;
      try
      {
        EmitRollbackFailureDiagnostic(
            "projection", rollback, nullptr, nullptr, {});
      }
      catch (...)
      {
      }
      return false;
    }
    auto& implementation = *experiment.implementation_;
    rollback.entry_interior = GridValues(*implementation.current_interior);
    rollback.entry_trace = GridValues(*implementation.current_trace);
    rollback.entry_interior_hash = HashValues(rollback.entry_interior);
    rollback.entry_trace_hash = HashValues(rollback.entry_trace);
    if (!FiniteVector(rollback.entry_interior) ||
        !FiniteVector(rollback.entry_trace) ||
        !PrescreenFields(*implementation.mesh, *implementation.interior_space,
                         *implementation.trace_space,
                         *implementation.current_interior,
                         *implementation.current_trace,
                         experiment.quadrature_order_,
                         rollback.initial_admissibility) ||
        !rollback.initial_admissibility.passed)
    {
      rollback.status = ExperimentStatus::InitialAdmissibilityFailure;
      try
      {
        EmitRollbackFailureDiagnostic(
            "newton", rollback, nullptr, nullptr, {});
      }
      catch (...)
      {
      }
      return false;
    }

    NewtonEvidence newton{};
    SolvedMeshEvidence private_solved{};
    implementation.rollback_probe_mode = true;
    rollback.returned_success =
        experiment.SolveNewtonTransaction(newton, &private_solved);
    const TraceSolveDiagnostic trace_diagnostic =
        implementation.last_trace_solve_diagnostic;
    implementation.rollback_probe_mode = false;
    rollback.status = newton.status;
    rollback.accepted_updates = newton.nonlinear_iterations;
    rollback.initial_residual = newton.initial_residual;
    rollback.residual_after_first_accepted =
        newton.residual_after_first_accepted;
    rollback.residual_after_first_above_convergence =
        newton.residual_after_first_above_convergence;
    rollback.restored_interior = GridValues(*implementation.current_interior);
    rollback.restored_trace = GridValues(*implementation.current_trace);
    rollback.restored_interior_hash = HashValues(rollback.restored_interior);
    rollback.restored_trace_hash = HashValues(rollback.restored_trace);
    rollback.interior_byte_equal =
        rollback.entry_interior.size() == rollback.restored_interior.size() &&
        std::memcmp(rollback.entry_interior.data(), rollback.restored_interior.data(),
                    rollback.entry_interior.size() * sizeof(double)) == 0;
    rollback.trace_byte_equal =
        rollback.entry_trace.size() == rollback.restored_trace.size() &&
        std::memcmp(rollback.entry_trace.data(), rollback.restored_trace.data(),
                    rollback.entry_trace.size() * sizeof(double)) == 0;
    rollback.local_generation_invalid =
        implementation.linearization_generation == 0;
    rollback.condensed_generation_invalid =
        implementation.condensation_generation == 0;
    rollback.solve_generation_invalid = implementation.solve_generation == 0 &&
        implementation.recovery_generation == 0;
    LocalLinearizationSnapshot local_snapshot;
    std::vector<double> right_hand_side, increment;
    rollback.local_copy_rejected =
        !experiment.CopyLocalLinearization(local_snapshot);
    rollback.right_hand_side_copy_rejected =
        !experiment.CopyTraceRightHandSide(right_hand_side);
    rollback.trace_increment_copy_rejected =
        !experiment.CopyTraceIncrement(increment);
    rollback.condensation_consume_rejected =
        !experiment.CondenseAndAssembleTrace();
    rollback.solve_consume_rejected = !experiment.SolveTrace();
    rollback.recovery_consume_rejected = !experiment.RecoverInterior();
    rollback.partial_solve_evidence = implementation.has_solved_evidence;
    rollback.partial_convergence_evidence = false;
    rollback.passed = !rollback.returned_success &&
        rollback.status == ExperimentStatus::IterationLimit &&
        rollback.initial_admissibility.passed && rollback.accepted_updates == 1 &&
        std::isfinite(rollback.initial_residual) &&
        std::isfinite(rollback.residual_after_first_accepted) &&
        rollback.residual_after_first_above_convergence &&
        rollback.residual_after_first_accepted >
            1.0e-10 * (1.0 + newton.initial_residual) &&
        rollback.interior_byte_equal && rollback.trace_byte_equal &&
        rollback.entry_interior_hash == rollback.restored_interior_hash &&
        rollback.entry_trace_hash == rollback.restored_trace_hash &&
        rollback.local_generation_invalid &&
        rollback.condensed_generation_invalid &&
        rollback.solve_generation_invalid && rollback.local_copy_rejected &&
        rollback.right_hand_side_copy_rejected &&
        rollback.trace_increment_copy_rejected &&
        rollback.condensation_consume_rejected &&
        rollback.solve_consume_rejected && rollback.recovery_consume_rejected &&
        !rollback.partial_solve_evidence &&
        !rollback.partial_convergence_evidence;
    if (!rollback.passed)
    {
      try
      {
        EmitRollbackFailureDiagnostic(
            "newton", rollback, &newton, &private_solved, trace_diagnostic);
      }
      catch (...)
      {
      }
    }
    return rollback.passed;
  };

  SmoothEulerExperiment rollback_experiment(1);
  RollbackAuditEvidence rollback{};
  if (!run_rollback_probe(rollback_experiment, rollback))
  {
    const bool hard_failure = rollback.status != ExperimentStatus::Success &&
        rollback.status != ExperimentStatus::IterationLimit;
    evidence.status = hard_failure ? rollback.status :
        ExperimentStatus::EvidenceConsistencyFailure;
    evidence.rollback = std::move(rollback);
    return false;
  }

  std::vector<ConvergenceEvidence> convergence;
  convergence.reserve(4);
  for (int polynomial_order = 1; polynomial_order <= 4; ++polynomial_order)
  {
    SmoothEulerExperiment experiment(polynomial_order);
    ConvergenceEvidence ladder{};
    if (!experiment.RunConvergenceLadder(ladder))
    {
      evidence.status = ladder.status;
      evidence.rollback = std::move(rollback);
      return false;
    }
    convergence.push_back(std::move(ladder));
  }
  evidence.status = ExperimentStatus::Success;
  evidence.rollback = std::move(rollback);
  evidence.convergence = std::move(convergence);
  evidence.passed = true;
  return true;
}

} // namespace dg::hdg::euler
