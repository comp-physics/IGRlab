#pragma once

/**
 * @file numerics/hdg/euler/smooth_euler.hpp
 * @brief Serial FP64, manufactured smooth-Euler HDG experiment.
 *
 * This is deliberately an unregistered experiment. It is neither a general
 * Euler discretization nor an MPI/device implementation. Its public value
 * types are fixed-size where physics has fixed size, so invalid physics never
 * requires a caller-owned raw output buffer.
 */

#include "numerics/hdg/core/host_reference.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace mfem
{
class FiniteElementSpace;
class GridFunction;
class SparseMatrix;
}

namespace dg::hdg::euler
{
inline constexpr int dimension = 2;
inline constexpr int conserved_components = 4;
inline constexpr double gamma = 1.4;
inline constexpr double fixed_scalar_tau = 2.5;
inline constexpr double backward_euler_dt = 0.05;
inline constexpr double experiment_epsilon =
    std::numeric_limits<double>::epsilon();

// Closed compile-time description: the experiment never chooses a model,
// stabilization, time rule, or backend per quadrature point.
struct SmoothEulerModelTag : ModelTagMarker {};
struct FixedScalarTauTag : StabilizationTagMarker {};
struct BackwardEulerTag : TimeTreatmentTagMarker {};
struct TraceStateName;
struct SerialExecutionTag;
struct HostMemoryTag;
struct NoCommunicationTag;
struct SmoothEulerDevicePolicy
{
  double tau = fixed_scalar_tau;
  double dt = backward_euler_dt;
};
using SmoothEulerTrace = TraceTuple<TracePotential<TraceStateName,
                                                   VectorTrace<conserved_components>>>;
using SmoothEulerSpec = KernelSpec<SmoothEulerModelTag,
                                  SmoothEulerTrace,
                                  FixedScalarTauTag,
                                  BackwardEulerTag,
                                  BackendPolicy<SerialExecutionTag,
                                                HostMemoryTag,
                                                NoCommunicationTag>,
                                  PrecisionPolicy<double, double, double, double, double>,
                                  SmoothEulerDevicePolicy>;
static_assert(IsKernelSpec<SmoothEulerSpec>::value,
              "smooth Euler must remain a closed static HDG KernelSpec");

using ConservedState = std::array<double, conserved_components>;
using PrimitiveState = std::array<double, conserved_components>; // rho,u,v,p
using FluxVector = std::array<double, conserved_components>;
using FluxJacobian =
    std::array<std::array<double, conserved_components>, conserved_components>;

enum class PhysicsStatus : unsigned char
{
  success,
  invalid_density,
  invalid_pressure,
  invalid_normal,
  non_finite
};

/** A value result makes failure atomic: value is untouched by failed helpers. */
struct PhysicsResult
{
  PhysicsStatus status = PhysicsStatus::success;
  bool ok() const noexcept { return status == PhysicsStatus::success; }
};

/** Conservative/primitive conversion with rho,p and all inputs checked. */
PhysicsResult ConservativeFromPrimitive(const PrimitiveState& primitive,
                                       ConservedState& conservative) noexcept;
PhysicsResult PrimitiveFromConservative(const ConservedState& conservative,
                                       PrimitiveState& primitive) noexcept;

/** F(U).n and its analytic 4x4 derivative. Failed calls preserve output. */
PhysicsResult NormalFlux(const ConservedState& conservative,
                         const std::array<double, dimension>& normal,
                         FluxVector& flux) noexcept;
PhysicsResult NormalFluxJacobian(const ConservedState& conservative,
                                 const std::array<double, dimension>& normal,
                                 FluxJacobian& jacobian) noexcept;

/** The frozen periodic manufactured state and its stationary forcing. */
PhysicsResult ManufacturedPrimitive(double x, double y,
                                   PrimitiveState& primitive) noexcept;
PhysicsResult ManufacturedConservative(double x, double y,
                                      ConservedState& conservative) noexcept;
PhysicsResult ManufacturedSource(double x, double y,
                                ConservedState& source) noexcept;
double ManufacturedMaximumWaveSpeed() noexcept;

/** Pinned MFEM Ordering::byNODES helpers used by the serial experiment. */
int ByNodesIndex(int scalar_dof, int scalar_count, int component) noexcept;
int ComponentFromByNodesIndex(int vector_dof, int scalar_count) noexcept;
int ScalarFromByNodesIndex(int vector_dof, int scalar_count) noexcept;
int DecodeSignedVDof(int signed_vdof, double& sign) noexcept;

enum class ExperimentStatus : unsigned char
{
  Success,
  SetupFailure,
  ProjectionFailure,
  InitialAdmissibilityFailure,
  AssemblyFailure,
  CondensationFailure,
  GmresFailure,
  LinearResidualFailure,
  RecoveryFailure,
  LineSearchExhausted,
  IterationLimit,
  ErrorIntegrationFailure,
  ErrorFloorFailure,
  RateFailure,
  EvidenceConsistencyFailure
};

enum class TrialRejectionReason : unsigned char
{
  None,
  Nonfinite,
  Density,
  Pressure,
  WaveSpeed,
  Armijo
};

struct AdmissibilityEvidence
{
  bool passed = false;
  TrialRejectionReason reason = TrialRejectionReason::None;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double maximum_wave_speed = 0.0;
};

struct NewtonEvidence
{
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  int nonlinear_iterations = 0;
  int linear_iterations = 0;
  int assembly_attempts = 0;
  int rejected_trials = 0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double trace_relative_residual = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double minimum_damping = 1.0;
  double maximum_wave_speed = 0.0;
  double maximum_coefficient_update = 0.0;
  double maximum_jacobian_change = 0.0;
  bool initial_prescreen_passed = false;
  bool admissible = false;
};

struct LinearSolveEvidence
{
  int nonlinear_iteration = 0;
  int gmres_iterations = 0;
  double right_hand_side_norm = 0.0;
  double reported_relative_residual = 0.0;
  double true_relative_residual = 0.0;
  ExperimentStatus status = ExperimentStatus::SetupFailure;
};

struct TrialEvidence
{
  int nonlinear_iteration = 0;
  int trial_index = 0;
  double alpha = 0.0;
  double base_residual = 0.0;
  double armijo_boundary = 0.0;
  double trial_residual = 0.0;
  double coefficient_update = 0.0;
  double jacobian_change = 0.0;
  AdmissibilityEvidence admissibility{};
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  TrialRejectionReason rejection_reason = TrialRejectionReason::None;
  bool reassembly_attempted = false;
  bool residual_evaluated = false;
  bool trial_residual_present = false;
  bool accepted = false;
  bool generation_published = false;
  bool generation_discarded = false;
};

struct ErrorIntegralEvidence
{
  double error_squared = 0.0;
  double exact_norm_squared = 0.0;
  double measure = 0.0;
  double direct_l2 = 0.0;
  double rms = 0.0;
  double relative_l2 = 0.0;
  int unique_face_count = 0;
};

/**
 * Read-only local-system capture used by the serial, independent assembly
 * oracles.  The signed trace map deliberately remains in MFEM's native
 * orientation convention; callers must decode it before global insertion.
 */
struct LocalLinearizationSnapshot
{
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::vector<double> a, b, c, d, f, g;
  std::vector<int> signed_trace_dofs;
};

struct LocalAssemblyAuditSnapshot
{
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int element = -1;
  int quadrature_order = 0;
  int volume_quadrature_points = 0;
  int face_quadrature_points = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::vector<double> old_interior;
  std::vector<double> current_interior;
  std::vector<double> trace;
  std::vector<double> residual_interior;
  std::vector<double> residual_trace;
  std::vector<double> a, b, c, d;
  std::vector<int> signed_trace_dofs;
  std::uint64_t interior_reconstruction_id = 0;
  std::uint64_t trace_reconstruction_id = 0;
};

struct LinearAuditSnapshot
{
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int nonlinear_iteration = 0;
  int rows = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;
  std::vector<double> values;
  std::vector<double> right_hand_side;
  std::vector<double> trace_increment;
  double true_relative_residual = 0.0;
};

struct TrialAuditSnapshot
{
  int nonlinear_iteration = 0;
  int trial_index = 0;
  double alpha = 0.0;
  double base_residual = 0.0;
  double armijo_boundary = 0.0;
  double trial_residual = 0.0;
  double coefficient_update = 0.0;
  double jacobian_change = 0.0;
  double formula_identity_error = 0.0;
  std::vector<double> base_interior;
  std::vector<double> base_trace;
  std::vector<double> delta_interior;
  std::vector<double> delta_trace;
  std::vector<double> trial_interior;
  std::vector<double> trial_trace;
  AdmissibilityEvidence admissibility{};
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  TrialRejectionReason rejection_reason = TrialRejectionReason::None;
  bool reassembly_attempted = false;
  bool residual_evaluated = false;
  bool accepted = false;
  bool generation_published = false;
  bool generation_discarded = false;
};

struct SolvedMeshEvidence
{
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int q_a = 0;
  int q_e = 0;
  int q_e_plus_four = 0;
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  NewtonEvidence newton{};
  AdmissibilityEvidence initial_admissibility{};
  std::vector<double> old_interior;
  std::vector<double> initial_interior;
  std::vector<double> final_interior;
  std::vector<double> old_trace;
  std::vector<double> initial_trace;
  std::vector<double> final_trace;
  std::vector<double> initial_residual_interior;
  std::vector<double> initial_residual_trace;
  std::vector<double> final_residual_interior;
  std::vector<double> final_residual_trace;
  std::vector<int> signed_trace_dofs;
  std::vector<double> residual_history;
  std::vector<LinearSolveEvidence> linear_history;
  std::vector<TrialEvidence> trial_history;
  ErrorIntegralEvidence q_e_volume{};
  ErrorIntegralEvidence q_e_plus_four_volume{};
  ErrorIntegralEvidence q_e_trace{};
  ErrorIntegralEvidence q_e_plus_four_trace{};
  LocalAssemblyAuditSnapshot local_assembly{};
  bool has_linear_audit = false;
  LinearAuditSnapshot linear_audit{};
  std::vector<TrialAuditSnapshot> trial_audits;
  bool internal_consistency = false;
};

struct ConvergenceEvidence
{
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  int polynomial_order = 0;
  std::array<int, 3> mesh_widths{{3, 6, 12}};
  std::array<SolvedMeshEvidence, 3> solves{};
  std::array<double, 3> volume_relative_l2{{0.0, 0.0, 0.0}};
  std::array<double, 3> trace_relative_l2{{0.0, 0.0, 0.0}};
  std::array<double, 2> volume_rates{{0.0, 0.0}};
  std::array<double, 2> trace_rates{{0.0, 0.0}};
  bool passed = false;
};

struct RollbackAuditEvidence
{
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  AdmissibilityEvidence initial_admissibility{};
  int accepted_updates = 0;
  std::vector<double> entry_interior;
  std::vector<double> restored_interior;
  std::vector<double> entry_trace;
  std::vector<double> restored_trace;
  std::uint64_t entry_interior_hash = 0;
  std::uint64_t restored_interior_hash = 0;
  std::uint64_t entry_trace_hash = 0;
  std::uint64_t restored_trace_hash = 0;
  bool returned_success = false;
  bool interior_byte_equal = false;
  bool trace_byte_equal = false;
  bool local_generation_invalid = false;
  bool condensed_generation_invalid = false;
  bool solve_generation_invalid = false;
  bool local_copy_rejected = false;
  bool right_hand_side_copy_rejected = false;
  bool trace_increment_copy_rejected = false;
  bool condensation_consume_rejected = false;
  bool solve_consume_rejected = false;
  bool recovery_consume_rejected = false;
  bool partial_solve_evidence = false;
  bool partial_convergence_evidence = false;
  bool passed = false;
};

struct ExperimentEvidence
{
  ExperimentStatus status = ExperimentStatus::SetupFailure;
  RollbackAuditEvidence rollback{};
  std::vector<ConvergenceEvidence> convergence{};
  bool passed = false;
};

/**
 * Serial-only owner for the fixed 2-D experiment. The implementation uses
 * L2(p,2) interiors, DG_Interface(p,2) skeleton unknowns, Ordering::byNODES,
 * a SparseMatrix trace system, and local HostReferenceEngine condensation.
 */
class SmoothEulerExperiment
{
public:
  explicit SmoothEulerExperiment(int polynomial_order);
  ~SmoothEulerExperiment();
  SmoothEulerExperiment(SmoothEulerExperiment&&) noexcept;
  SmoothEulerExperiment& operator=(SmoothEulerExperiment&&) noexcept;
  SmoothEulerExperiment(const SmoothEulerExperiment&) = delete;
  SmoothEulerExperiment& operator=(const SmoothEulerExperiment&) = delete;

  /** Builds and atomically publishes a fresh verified periodic mesh/space graph. */
  bool BuildPeriodicMesh(int elements_per_direction);
  /** Stages then publishes the exact old interior state and single-valued trace. */
  bool ProjectOldStage();
  /** Assembles the current candidate volume/face/forcing local A/B/C/D data. */
  bool AssembleLinearization();
  /** Condenses actual local blocks through HostReferenceEngine and assembles S. */
  bool CondenseAndAssembleTrace();
  /** Solves the current nonsymmetric serial trace candidate. */
  bool SolveTrace();
  /** Recovers local increments through HostReferenceEngine. */
  bool RecoverInterior();
  /** Runs the frozen admissible, damped Newton transaction. */
  bool SolveDampedNewton(NewtonEvidence& evidence);
  /** Runs and validates the frozen 3/6/12 error ladder. */
  bool RunConvergenceLadder(ConvergenceEvidence& evidence);

  int polynomial_order() const noexcept { return polynomial_order_; }
  int quadrature_order() const noexcept { return quadrature_order_; }
  const mfem::FiniteElementSpace* trace_space() const noexcept;
  const mfem::GridFunction* old_interior() const noexcept;
  const mfem::GridFunction* old_trace() const noexcept;
  const mfem::GridFunction* current_interior() const noexcept;
  const mfem::GridFunction* current_trace() const noexcept;
  const SolvedMeshEvidence* solved_evidence() const noexcept;
  const mfem::SparseMatrix* trace_matrix() const noexcept;
  /** Copies assembled blocks without exposing mutable implementation state. */
  bool CopyLocalLinearization(LocalLinearizationSnapshot& snapshot) const;
  /** Copies the assembled trace right-hand side for independent solve checks. */
  bool CopyTraceRightHandSide(std::vector<double>& right_hand_side) const;
  /** Copies the current trace increment only after a successful trace solve. */
  bool CopyTraceIncrement(std::vector<double>& increment) const;

private:
  bool ProjectPerturbedCurrentStage();
  bool SolveNewtonTransaction(NewtonEvidence& evidence,
                              SolvedMeshEvidence* solved_evidence,
                              int maximum_accepted_updates);
  bool IntegrateErrorEvidence(int quadrature_order,
                              ErrorIntegralEvidence& volume,
                              ErrorIntegralEvidence& trace) const;
  bool RunRollbackProbe(RollbackAuditEvidence& evidence);

  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
  int polynomial_order_ = 0;
  int quadrature_order_ = 0;

  friend bool RunSmoothEulerExperiment(ExperimentEvidence& evidence);
};

/** Runs the p=1..4 serial manufactured candidate and returns all evidence. */
bool RunSmoothEulerExperiment(ExperimentEvidence& evidence);

} // namespace dg::hdg::euler
