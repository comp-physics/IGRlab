#pragma once
/**
 * @file numerics/hdg/auxiliary/screened_poisson_cpu.hpp
 * @brief Prescribed-input periodic FP64 HDG screened-Poisson auxiliary.
 *
 * This is the bounded E2-target candidate selected by HDG-AUX-CPU-03.  It is
 * not an accepted E2 auxiliary, a scalar-IGR coupling, a wall operator, a
 * runtime backend, or a supported production method. Inputs are frozen
 * caller-owned fields which are copied before the transaction retains state.
 *
 * Frozen equations and ordering:
 *   rho*z - grad(Sigma) = 0,
 *   Sigma/rho - alpha*div(z) = f,
 *   Rz = (r,rho*z) + (div(r),Sigma) - <r.n,Sigma_hat>,
 *   RSigma = (v,Sigma/rho) + alpha*(grad(v),z)
 *            - alpha*<v,z_hat.n> - (v,f),
 *   G = sum_K alpha*<eta,z_hat_K.n_K> = 0,
 *   z_hat.n = z.n - tau*(Sigma-Sigma_hat),
 *   tau = (k+1)^2/(rho_hat*h_KF), h_KF=|K|/|F|.
 * Local columns are [Sigma,z_x,z_y], residual rows are [Rz_x,Rz_y,RSigma],
 * f_e=[0,0,(v,f)], and g_e=0.
 */
#include "numerics/hdg/hdg_contract.hpp"
#include <mfem.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
namespace dg::hdg::auxiliary
{
inline constexpr int screened_poisson_dimension = 2;
inline constexpr double screened_poisson_primary_alpha = 0.1;
// Closed type-level identity.  None of these tags selects behavior at an
// element, face, or quadrature point.
struct FullyHybridizedScreenedPoissonV1Tag : ModelTagMarker
{};
struct DensityHeightP2TraceStabilizationV2Tag : StabilizationTagMarker
{};
struct FrozenPrescribedSnapshotV1Tag : TimeTreatmentTagMarker
{};
struct PeriodicUnitTorusV1Tag
{};
struct CpuSerialExecutionTag
{};
struct CpuMpiRank2ExecutionTag
{};
struct HostMemoryPolicy
{};
struct NoCommunicationPolicy
{};
struct MpiCommunicationPolicy
{};
struct ScreenedPoissonTraceName;
// Projection-v3 declarations bind the isolated E2 slice to the exact outer
// identity without claiming that the full coupled runtime exists here.
struct HdgPublishedScalarPeriodicSc25ProjectionV3Tag : ModelTagMarker
{};
struct FullyHybridizedScalarPeriodicSc25ProjectionV3Identity
{};
inline constexpr char sc25_projection_v3_id[] =
    "exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3";
inline constexpr char sc25_projection_v3_object_id[] =
    "FullyHybridizedScalarPeriodicSc25ProjectionV3";
inline constexpr char sc25_projection_v3_cpu_serial_tuple_id[] =
    "HDG-SC25-PER-BE-CPU-R1-FP64-v3";
inline constexpr char sc25_projection_v3_cpu_mpi_rank2_tuple_id[] =
    "HDG-SC25-PER-BE-CPU-R2-FP64-v3";
using ScreenedPoissonTrace =
    TraceTuple<TracePotential<ScreenedPoissonTraceName, ScalarTrace>>;
using ScreenedPoissonFp64Policy =
    PrecisionPolicy<double, double, double, double, double>;
using CpuSerialBackendPolicy =
    BackendPolicy<CpuSerialExecutionTag, HostMemoryPolicy,
                  NoCommunicationPolicy>;
using CpuMpiRank2BackendPolicy =
    BackendPolicy<CpuMpiRank2ExecutionTag, HostMemoryPolicy,
                  MpiCommunicationPolicy>;
using CpuSerialKernelSpec =
    KernelSpec<FullyHybridizedScreenedPoissonV1Tag, ScreenedPoissonTrace,
               DensityHeightP2TraceStabilizationV2Tag,
               FrozenPrescribedSnapshotV1Tag, CpuSerialBackendPolicy,
               ScreenedPoissonFp64Policy, PeriodicUnitTorusV1Tag>;
using CpuMpiRank2KernelSpec =
    KernelSpec<FullyHybridizedScreenedPoissonV1Tag, ScreenedPoissonTrace,
               DensityHeightP2TraceStabilizationV2Tag,
               FrozenPrescribedSnapshotV1Tag, CpuMpiRank2BackendPolicy,
               ScreenedPoissonFp64Policy, PeriodicUnitTorusV1Tag>;
static_assert(IsKernelSpec<CpuSerialKernelSpec>::value,
              "serial screened-Poisson must remain a closed KernelSpec");
static_assert(IsKernelSpec<CpuMpiRank2KernelSpec>::value,
              "rank-two screened-Poisson must remain a closed KernelSpec");
static_assert(std::is_same<typename CpuSerialKernelSpec::stabilization_tag,
                           DensityHeightP2TraceStabilizationV2Tag>::value &&
                  std::is_same<
                      typename CpuMpiRank2KernelSpec::stabilization_tag,
                      DensityHeightP2TraceStabilizationV2Tag>::value,
              "projection v3 requires the V2 density-height stabilization tag");
static_assert(std::is_empty<HdgPublishedScalarPeriodicSc25ProjectionV3Tag>::value &&
                  std::is_trivially_copyable<
                      HdgPublishedScalarPeriodicSc25ProjectionV3Tag>::value &&
                  std::is_empty<
                      FullyHybridizedScalarPeriodicSc25ProjectionV3Identity>::value,
              "projection-v3 outer and object identities must remain static markers");
static_assert(std::is_empty<PeriodicUnitTorusV1Tag>::value &&
                  std::is_trivially_copyable<PeriodicUnitTorusV1Tag>::value,
              "the periodic domain marker must be empty and trivially copyable");
enum class StatusCode : std::uint8_t
{
  Uninitialized,
  Ready,
  Success,
  InvalidArgument,
  UnsupportedCommunicator,
  AllocationFailure,
  SetupFailure,
  FactorizationFailure,
  TrueAssemblyFailure,
  TraceSolveFailure,
  DistributionFailure,
  RecoveryFailure,
  PublicationFailure,
  InjectedFailure
};
enum class AttemptPhase : std::uint8_t
{
  Setup,
  Factorization,
  TrueAssembly,
  TraceSolve,
  Distribution,
  Recovery,
  Publication,
  Count
};
enum class PhaseDisposition : std::uint8_t
{
  NotReached,
  Passed,
  Failed
};
struct Status
{
  StatusCode code = StatusCode::Uninitialized;
  AttemptPhase phase = AttemptPhase::Setup;
  int origin_rank = -1;
  constexpr bool succeeded() const noexcept
  {
    return code == StatusCode::Ready || code == StatusCode::Success;
  }
};
constexpr bool operator==(const Status left, const Status right) noexcept
{
  return left.code == right.code && left.phase == right.phase &&
      left.origin_rank == right.origin_rank;
}
constexpr bool operator!=(const Status left, const Status right) noexcept
{
  return !(left == right);
}
struct PhaseDiagnostic
{
  AttemptPhase phase = AttemptPhase::Setup;
  PhaseDisposition disposition = PhaseDisposition::NotReached;
  bool selected = false;
  bool entered = false;
  bool completed = false;
  bool passed = false;
};
struct SolverEvidence
{
  // HYPRE 3.1 getters return the process-wide error accumulator after writing
  // their output. Preserve both that aggregate and every raw getter return.
  int gmres_configuration_query_status = -1;
  int gmres_krylov_dimension_query_return = -1;
  int gmres_relative_tolerance_query_return = -1;
  int gmres_absolute_tolerance_query_return = -1;
  int gmres_maximum_iterations_query_return = -1;
  int gmres_setup_status = -1;
  int gmres_solve_status = -1;
  int gmres_convergence_query_status = -1;
  int gmres_converged = -1;
  int gmres_iterations = 0;
  int effective_gmres_krylov_dimension = -1;
  int effective_gmres_max_iterations = -1;
  std::uint64_t effective_gmres_relative_tolerance_bits = 0;
  std::uint64_t effective_gmres_absolute_tolerance_bits = 0;
  double gmres_reported_residual = 0.0;
  double true_residual_norm = 0.0;
  double true_right_hand_side_norm = 0.0;
  double maximum_local_recovery_residual = 0.0;
  double quadrature_repeat_relative_change = 0.0;
  int effective_ilu_type = -1;
  int effective_ilu_level_of_fill = -1;
  int effective_ilu_local_reordering = -1;
  bool exact_gmres_configuration = false;
  bool exact_ilu_configuration = false;
};
struct AttemptDiagnostic
{
  Status status{};
  std::uint64_t attempt_generation = 0;
  std::uint64_t input_generation = 0;
  std::array<PhaseDiagnostic,
             static_cast<std::size_t>(AttemptPhase::Count)> phases{};
  SolverEvidence solver{};
};
struct ScreenedPoissonPolicy
{
  double alpha = screened_poisson_primary_alpha;
};
/** Nonowning solve input.  Solve validates exact identity and copies bytes. */
struct PrescribedSnapshot
{
  const mfem::ParGridFunction* rho = nullptr;
  const mfem::ParGridFunction* rho_hat = nullptr;
  const mfem::ParGridFunction* source = nullptr;
  std::uint64_t generation = 0;
};
struct FailureInjection
{
  AttemptPhase phase = AttemptPhase::Count;
  int rank = -1;
  // Test-only collective control. The production value remains the pinned 300.
  bool force_trace_nonconvergence = false;
};
struct TraceRestrictionEntry
{
  int element = -1;
  int row = -1;
  int signed_local_vdof = 0;
  int decoded_local_vdof = -1;
  int sign = 0;
  GlobalIndex global_true_dof = invalid_global_index;
  LocalIndex local_true_dof = invalid_local_index;
  RankIndex owner_rank = invalid_rank;
};
/** Rank-local element batches, all row-major and element-major. */
struct LocalBlockSnapshots
{
  int local_elements = 0;
  int scalar_interior_dofs = 0;
  int interior_unknowns = 0;
  int trace_unknowns = 0;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> d;
  std::vector<double> f;
  std::vector<double> g;
  std::vector<double> schur;
  std::vector<double> condensed_right_hand_side;
  std::vector<TraceRestrictionEntry> restriction;
};
struct TopologyCensus
{
  int rank = -1;
  int ranks = 0;
  int local_elements = 0;
  int global_elements = 0;
  int local_trace_dofs = 0;
  int owned_true_trace_dofs = 0;
  std::int64_t global_true_trace_dofs = 0;
  std::size_t owned_trace_entries = 0;
  std::size_t ghost_trace_entries = 0;
  int negative_restriction_rows = 0;
};
/** Immutable after publication through the owning solver. */
struct AuxiliarySolution
{
  int polynomial_degree = 0;
  int elements_per_direction = 0;
  double alpha = 0.0;
  std::uint64_t input_generation = 0;
  std::uint64_t solution_generation = 0;
  // Per element: Sigma[scalar], then z[zx scalar..., zy scalar...].
  std::vector<double> sigma;
  std::vector<double> z;
  // MFEM local canonical storage and this rank's true-DOF partition.
  std::vector<double> local_sigma_hat;
  std::vector<double> owned_true_sigma_hat;
  TopologyCensus census;
  SolverEvidence solver;
  LocalBlockSnapshots blocks;
};
struct ScreenedPoissonCpuTestAccess;
/**
 * Collective, failure-atomic owner of one fixed periodic topology.
 * Invalid MPI handles, divergent collective lifecycles, and MPI collective
 * failures themselves are process-fatal exclusions, not recoverable Status
 * outcomes.
 */
class FullyHybridizedScreenedPoissonCpu
{
public:
  ~FullyHybridizedScreenedPoissonCpu() noexcept;
  FullyHybridizedScreenedPoissonCpu(
      FullyHybridizedScreenedPoissonCpu&&) noexcept;
  FullyHybridizedScreenedPoissonCpu& operator=(
      FullyHybridizedScreenedPoissonCpu&&) = delete;
  FullyHybridizedScreenedPoissonCpu(
      const FullyHybridizedScreenedPoissonCpu&) = delete;
  FullyHybridizedScreenedPoissonCpu& operator=(
      const FullyHybridizedScreenedPoissonCpu&) = delete;
  static Status Create(
      MPI_Comm communicator, CpuSerialExecutionTag,
      PeriodicUnitTorusV1Tag, int degree, int elements_per_direction,
      ScreenedPoissonPolicy policy,
      std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
      AttemptDiagnostic& diagnostic) noexcept;
  static Status Create(
      MPI_Comm communicator, CpuMpiRank2ExecutionTag,
      PeriodicUnitTorusV1Tag, int degree, int elements_per_direction,
      ScreenedPoissonPolicy policy,
      std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
      AttemptDiagnostic& diagnostic) noexcept;
  Status Solve(const PrescribedSnapshot& snapshot,
               const FailureInjection& injection = {}) noexcept;
  Status status() const noexcept;
  const AttemptDiagnostic& last_attempt() const noexcept;
  std::uint64_t generation() const noexcept;
  int degree() const noexcept;
  int elements_per_direction() const noexcept;
  double alpha() const noexcept;
  const AuxiliarySolution* solution() const noexcept;
  const LocalBlockSnapshots* blocks() const noexcept;
  const mfem::ParMesh* mesh() const noexcept;
  const mfem::ParFiniteElementSpace* volume_space() const noexcept;
  const mfem::ParFiniteElementSpace* flux_space() const noexcept;
  const mfem::ParFiniteElementSpace* trace_space() const noexcept;
  /**
   * Apply the immutable assembled true-trace operator from the last successful
   * publication and return its matching assembled right-hand side. This
   * collective observation seam never changes solver state or output on
   * failure.
   */
  bool ApplyPublishedTraceOperator(
      const std::vector<double>& owned_true_input,
      std::vector<double>& owned_true_action,
      std::vector<double>& owned_true_right_hand_side) const noexcept;
private:
  struct Implementation;
  explicit FullyHybridizedScreenedPoissonCpu(
      std::unique_ptr<Implementation> implementation) noexcept;
  static Status CreateInternal(
      MPI_Comm communicator, std::uint8_t mode, int degree,
      int elements_per_direction, ScreenedPoissonPolicy policy,
      bool admit_two_by_two_test_mesh,
      std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
      AttemptDiagnostic& diagnostic) noexcept;
  std::unique_ptr<Implementation> implementation_;
  friend struct ScreenedPoissonCpuTestAccess;
};
static_assert(!std::is_polymorphic<FullyHybridizedScreenedPoissonCpu>::value,
              "screened-Poisson owner must not use virtual dispatch");
} // namespace dg::hdg::auxiliary
