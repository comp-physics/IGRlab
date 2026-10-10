#include "numerics/hdg/auxiliary/screened_poisson_cpu.hpp"
#include "screened_poisson_h1_reference.hpp"
#include <mfem.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace dg::hdg::auxiliary
{
/** The only bypass of the public N={8,16,32} gate. */
struct ScreenedPoissonCpuTestAccess
{
  static Status CreateTwoByTwo(
      MPI_Comm communicator, const int degree,
      const ScreenedPoissonPolicy policy,
      std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& output,
      AttemptDiagnostic& diagnostic) noexcept
  {
    return FullyHybridizedScreenedPoissonCpu::CreateInternal(
        communicator, 0, degree, 2, policy, true, output, diagnostic);
  }
};
} // namespace dg::hdg::auxiliary
namespace
{
using dg::hdg::auxiliary::AttemptDiagnostic;
using dg::hdg::auxiliary::AttemptPhase;
using dg::hdg::auxiliary::AuxiliarySolution;
using dg::hdg::auxiliary::CpuMpiRank2ExecutionTag;
using dg::hdg::auxiliary::CpuSerialExecutionTag;
using dg::hdg::auxiliary::FailureInjection;
using dg::hdg::auxiliary::FullyHybridizedScreenedPoissonCpu;
using dg::hdg::auxiliary::LocalBlockSnapshots;
using dg::hdg::auxiliary::PhaseDiagnostic;
using dg::hdg::auxiliary::PhaseDisposition;
using dg::hdg::auxiliary::PeriodicUnitTorusV1Tag;
using dg::hdg::auxiliary::PrescribedSnapshot;
using dg::hdg::auxiliary::ScreenedPoissonCpuTestAccess;
using dg::hdg::auxiliary::ScreenedPoissonPolicy;
using dg::hdg::auxiliary::Status;
using dg::hdg::auxiliary::StatusCode;
using dg::hdg::auxiliary::testing::H1ReferenceRequest;
using dg::hdg::auxiliary::testing::H1ReferenceSolution;
using dg::hdg::auxiliary::testing::SolveH1Reference;
using dg::hdg::GlobalIndex;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double comparison_tolerance = 2.0e-11;
constexpr double direction_nonvacuity = 1.0e-8;
bool IsFinite(const double value) { return std::isfinite(value); }
bool CheckProjectionV3Identity()
{
  using namespace dg::hdg::auxiliary;
  return std::string(sc25_projection_v3_id) ==
          "exp.hdg.igr.traced_scalar.sc25.periodic.projection.v3" &&
      std::string(sc25_projection_v3_object_id) ==
          "FullyHybridizedScalarPeriodicSc25ProjectionV3" &&
      std::string(sc25_projection_v3_cpu_serial_tuple_id) ==
          "HDG-SC25-PER-BE-CPU-R1-FP64-v3" &&
      std::string(sc25_projection_v3_cpu_mpi_rank2_tuple_id) ==
          "HDG-SC25-PER-BE-CPU-R2-FP64-v3" &&
      !std::is_same<HdgPublishedScalarPeriodicSc25ProjectionV3Tag,
                    FullyHybridizedScreenedPoissonV1Tag>::value &&
      std::is_same<typename CpuSerialKernelSpec::stabilization_tag,
                   DensityHeightP2TraceStabilizationV2Tag>::value &&
      std::is_same<typename CpuMpiRank2KernelSpec::stabilization_tag,
                   DensityHeightP2TraceStabilizationV2Tag>::value;
}
bool GlobalAll(MPI_Comm communicator, const bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator);
  return output == 1;
}
double GlobalNorm(MPI_Comm communicator, const std::vector<double>& values)
{
  long double local = 0.0L;
  for (const double value : values)
  {
    local += static_cast<long double>(value) * value;
  }
  double local_double = static_cast<double>(local);
  double global = 0.0;
  MPI_Allreduce(&local_double, &global, 1, MPI_DOUBLE, MPI_SUM, communicator);
  return std::sqrt(global);
}
bool ExactPositiveZero(const std::vector<double>& values)
{
  const double zero = 0.0;
  return std::all_of(values.begin(), values.end(), [&](const double value)
                     { return std::memcmp(&value, &zero,
                                          sizeof(value)) == 0; });
}
bool ExactZero(const std::vector<double>& values)
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return value == 0.0; });
}
double RelativeDifference(const std::vector<double>& left,
                          const std::vector<double>& right)
{
  if (left.size() != right.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  long double difference = 0.0L;
  long double reference = 0.0L;
  for (std::size_t index = 0; index < left.size(); ++index)
  {
    const long double delta = static_cast<long double>(left[index]) -
        right[index];
    difference += delta * delta;
    reference += static_cast<long double>(right[index]) * right[index];
  }
  return std::sqrt(static_cast<double>(difference)) /
      std::max(1.0, std::sqrt(static_cast<double>(reference)));
}
int Decode(const int signed_dof, int& sign)
{
  sign = signed_dof < 0 ? -1 : 1;
  return signed_dof < 0 ? -1 - signed_dof : signed_dof;
}
struct CategoryCensus
{
  std::string name;
  int required = 0;
  int selected = 0;
  int entered = 0;
  int completed = 0;
  int passed = 0;
  int failed = 0;
  int skipped = 0;
  int not_reached = 0;
};
class TestCensus
{
public:
  explicit TestCensus(MPI_Comm communicator) : communicator_(communicator) {}
  void Require(const std::string& category, const int count)
  {
    categories_[category].name = category;
    categories_[category].required = count;
  }
  template <typename Callable>
  void Run(const std::string& category, const std::string& id,
           Callable&& callable)
  {
    CategoryCensus& census = categories_[category];
    ++census.selected;
    ++census.entered;
    bool local = false;
    try { local = static_cast<bool>(callable()); }
    catch (...) { local = false; }
    const bool passed = GlobalAll(communicator_, local);
    ++census.completed;
    if (passed) { ++census.passed; }
    else
    {
      ++census.failed;
      int rank = 0;
      MPI_Comm_rank(communicator_, &rank);
      if (rank == 0)
      {
        std::cerr << "FAIL row=" << id << " category=" << category << '\n';
      }
    }
  }
  bool Complete() const
  {
    bool result = true;
    int rank = 0;
    MPI_Comm_rank(communicator_, &rank);
    for (const auto& item : categories_)
    {
      const CategoryCensus& census = item.second;
      const bool exact = census.required == census.selected &&
          census.selected == census.entered &&
          census.entered == census.completed &&
          census.completed == census.passed && census.failed == 0 &&
          census.skipped == 0 && census.not_reached == 0;
      result = exact && result;
      if (rank == 0)
      {
        std::cout << "CENSUS category=" << census.name
                  << " required=" << census.required
                  << " selected=" << census.selected
                  << " entered=" << census.entered
                  << " completed=" << census.completed
                  << " passed=" << census.passed
                  << " failed=" << census.failed
                  << " skipped=" << census.skipped
                  << " not_reached=" << census.not_reached << '\n';
      }
    }
    return GlobalAll(communicator_, result);
  }
private:
  MPI_Comm communicator_;
  std::map<std::string, CategoryCensus> categories_;
};
using FieldFunction = std::function<double(double, double)>;
bool ProjectVolume(mfem::ParFiniteElementSpace& space,
                   mfem::ParGridFunction& field,
                   const FieldFunction& function)
{
  mfem::FunctionCoefficient coefficient(
      [&function](const mfem::Vector& point)
      { return function(point[0], point[1]); });
  field.ProjectCoefficient(coefficient);
  const double* const data = field.HostRead();
  if (data == nullptr) { return false; }
  for (int index = 0; index < field.Size(); ++index)
    if (!IsFinite(data[index])) { return false; }
  return field.ParFESpace() == &space;
}
bool ProjectTrace(mfem::ParFiniteElementSpace& space,
                  mfem::ParGridFunction& field,
                  const FieldFunction& function)
{
  mfem::ParMesh* const mesh = space.GetParMesh();
  if (mesh == nullptr) { return false; }
  field = 0.0;
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element =
        space.GetFaceElement(face);
    mfem::ElementTransformation* const transformation =
        mesh->GetFaceTransformation(face);
    if (finite_element == nullptr || transformation == nullptr) { return false; }
    mfem::Array<int> signed_dofs;
    space.GetFaceVDofs(face, signed_dofs);
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    if (signed_dofs.Size() != finite_element->GetDof() ||
        nodes.GetNPoints() != finite_element->GetDof())
    {
      return false;
    }
    for (int dof = 0; dof < signed_dofs.Size(); ++dof)
    {
      mfem::Vector point(2);
      transformation->Transform(nodes.IntPoint(dof), point);
      int sign = 0;
      const int decoded = Decode(signed_dofs[dof], sign);
      if (decoded < 0 || decoded >= field.Size()) { return false; }
      field[decoded] = static_cast<double>(sign) * function(point[0], point[1]);
    }
  }
  return true;
}
enum class PublicBaselineStage
{
  NotReached,
  CreateStatus,
  CreatePostcondition,
  ProjectRho,
  ProjectSource,
  ProjectTrace,
  SolveStatus,
  SolutionPublication,
  GmresPostcondition,
  CheckZeroSigma,
  CheckZeroZ,
  CheckZeroTrace,
  CheckZeroH1,
  CheckZeroPassed
};
const char* PublicBaselineStageName(const PublicBaselineStage stage)
{
  static constexpr std::array<const char*, 14> names{{
      "not_reached", "create_status", "create_postcondition", "project_rho",
      "project_source", "project_trace", "solve_status",
      "solution_publication", "gmres_postcondition", "check_zero_sigma",
      "check_zero_z", "check_zero_trace", "check_zero_h1",
      "check_zero_passed"}};
  return names[static_cast<std::size_t>(stage)];
}
int PublicBaselineStageResult(const PublicBaselineStage actual,
                              const PublicBaselineStage predicate)
{
  if (actual < predicate) { return -1; }
  return actual == predicate ? 0 : 1;
}
struct PublicBaselineObservation
{
  bool selected = false;
  PublicBaselineStage stage = PublicBaselineStage::NotReached;
  int rank = -1;
  int ranks = 0;
  int degree = 0;
  int n = 0;
};
struct SolveBundle
{
  std::unique_ptr<FullyHybridizedScreenedPoissonCpu> solver;
  std::unique_ptr<mfem::ParGridFunction> rho;
  std::unique_ptr<mfem::ParGridFunction> rho_hat;
  std::unique_ptr<mfem::ParGridFunction> source;
  Status create_status{};
  Status solve_status{};
  AttemptDiagnostic create_diagnostic{};
  PublicBaselineObservation public_baseline{};
};
void PrintAttemptDiagnostic(const char* label,
                            const AttemptDiagnostic& diagnostic);
void EmitPublicBaselineDiagnostic(const SolveBundle& solve)
{
  const PublicBaselineObservation& observation = solve.public_baseline;
  if (!observation.selected) { return; }
  const auto checked = [](const bool reached, const bool passed)
  { return reached ? (passed ? 1 : 0) : -1; };
  const bool create_ready = solve.create_status.code == StatusCode::Ready;
  const bool have_solver = solve.solver != nullptr;
  const bool create_status_match =
      have_solver && solve.solver->status() == solve.create_status;
  const bool create_generation_one =
      have_solver && solve.solver->generation() == 1;
  const bool create_solution_null =
      have_solver && solve.solver->solution() == nullptr;
  const bool create_blocks_null =
      have_solver && solve.solver->blocks() == nullptr;
  const bool create_mesh_nonnull =
      have_solver && solve.solver->mesh() != nullptr;
  const bool create_volume_nonnull =
      have_solver && solve.solver->volume_space() != nullptr;
  const bool create_flux_nonnull =
      have_solver && solve.solver->flux_space() != nullptr;
  const bool create_trace_nonnull =
      have_solver && solve.solver->trace_space() != nullptr;
  const bool solve_called =
      observation.stage >= PublicBaselineStage::SolveStatus;
  const AuxiliarySolution* const solution =
      have_solver ? solve.solver->solution() : nullptr;
  const auto* const solver_evidence =
      solution != nullptr ? &solution->solver : nullptr;
  const bool exact_zero_completion = solver_evidence != nullptr &&
      solver_evidence->gmres_setup_status == -1 &&
      solver_evidence->gmres_solve_status == -1 &&
      solver_evidence->gmres_convergence_query_status == -1 &&
      solver_evidence->gmres_converged == -1 &&
      solver_evidence->gmres_iterations == 0 &&
      solver_evidence->gmres_reported_residual == 0.0 &&
      solver_evidence->true_right_hand_side_norm == 0.0 &&
      solver_evidence->true_residual_norm == 0.0;
  const auto solver_checked = [&](const bool passed)
  { return checked(solver_evidence != nullptr, passed); };
  const int rho_projection = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::ProjectRho);
  const int source_projection = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::ProjectSource);
  const int trace_projection = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::ProjectTrace);
  const int check_zero_sigma = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::CheckZeroSigma);
  const int check_zero_z = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::CheckZeroZ);
  const int check_zero_trace = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::CheckZeroTrace);
  const int check_zero_h1 = PublicBaselineStageResult(
      observation.stage, PublicBaselineStage::CheckZeroH1);
  std::ostringstream stream;
  stream << "AUX_PUBLIC_BASELINE"
         << " rank=" << observation.rank
         << " ranks=" << observation.ranks
         << " degree=" << observation.degree
         << " n=" << observation.n
         << " stage=" << PublicBaselineStageName(observation.stage)
         << " create_called=1"
         << " create_status_code="
         << static_cast<int>(solve.create_status.code)
         << " create_status_phase="
         << static_cast<int>(solve.create_status.phase)
         << " create_status_origin=" << solve.create_status.origin_rank
         << " create_status_ready=" << create_ready
         << " create_solver_nonnull=" << checked(create_ready, have_solver)
         << " create_solver_status_match="
         << checked(have_solver, create_status_match)
         << " create_generation_one="
         << checked(have_solver, create_generation_one)
         << " create_solution_null="
         << checked(have_solver, create_solution_null)
         << " create_blocks_null="
         << checked(have_solver, create_blocks_null)
         << " create_mesh_nonnull="
         << checked(have_solver, create_mesh_nonnull)
         << " create_volume_space_nonnull="
         << checked(have_solver, create_volume_nonnull)
         << " create_flux_space_nonnull="
         << checked(have_solver, create_flux_nonnull)
         << " create_trace_space_nonnull="
         << checked(have_solver, create_trace_nonnull)
         << " rho_projection_called=" << (rho_projection >= 0)
         << " rho_projection_passed=" << rho_projection
         << " source_projection_called=" << (source_projection >= 0)
         << " source_projection_passed=" << source_projection
         << " trace_projection_called=" << (trace_projection >= 0)
         << " trace_projection_passed=" << trace_projection
         << " solve_called=" << solve_called
         << " solve_status_code="
         << static_cast<int>(solve.solve_status.code)
         << " solve_status_phase="
         << static_cast<int>(solve.solve_status.phase)
         << " solve_status_origin=" << solve.solve_status.origin_rank
         << " solution_published="
         << checked(solve_called, solution != nullptr)
         << " solution_null=" << checked(solve_called, solution == nullptr)
         << " gmres_setup_zero=" << solver_checked(
             solver_evidence != nullptr && solver_evidence->gmres_setup_status == 0)
         << " gmres_solve_zero=" << solver_checked(
             solver_evidence != nullptr && solver_evidence->gmres_solve_status == 0)
         << " gmres_convergence_query_zero="
         << solver_checked(solver_evidence != nullptr &&
             solver_evidence->gmres_convergence_query_status == 0)
         << " gmres_converged_one=" << solver_checked(
             solver_evidence != nullptr && solver_evidence->gmres_converged == 1)
         << " gmres_krylov_dimension_50=" << solver_checked(
             solver_evidence != nullptr &&
                 solver_evidence->effective_gmres_krylov_dimension == 50)
         << " gmres_max_iterations_300=" << solver_checked(
             solver_evidence != nullptr &&
                 solver_evidence->effective_gmres_max_iterations == 300)
         << " exact_gmres_configuration=" << solver_checked(
             solver_evidence != nullptr &&
                 solver_evidence->exact_gmres_configuration)
         << " exact_zero_rhs_completion=" << solver_checked(
             exact_zero_completion)
         << " check_zero_sigma_called=" << (check_zero_sigma >= 0)
         << " check_zero_sigma_passed=" << check_zero_sigma
         << " check_zero_z_called=" << (check_zero_z >= 0)
         << " check_zero_z_passed=" << check_zero_z
         << " check_zero_trace_called=" << (check_zero_trace >= 0)
         << " check_zero_trace_passed=" << check_zero_trace
         << " check_zero_h1_called=" << (check_zero_h1 >= 0)
         << " check_zero_h1_passed=" << check_zero_h1;
  const bool solve_attempt_available =
      solve_called && solve.solver != nullptr;
  stream << " solve_attempt_available=" << solve_attempt_available;
  std::cerr << stream.str() << '\n';
  const std::string suffix = "-r" + std::to_string(observation.rank) +
      "-of" + std::to_string(observation.ranks);
  const std::string create_label = "public-create" + suffix;
  PrintAttemptDiagnostic(create_label.c_str(), solve.create_diagnostic);
  if (solve_attempt_available)
  {
    const std::string solve_label = "public-solve" + suffix;
    PrintAttemptDiagnostic(solve_label.c_str(), solve.solver->last_attempt());
  }
}
bool CreateSolver(MPI_Comm communicator, const int ranks, const int degree,
                  const int n, const double alpha,
                  std::unique_ptr<FullyHybridizedScreenedPoissonCpu>& solver,
                  Status& status, AttemptDiagnostic& diagnostic)
{
  if (ranks == 1)
  {
    status = FullyHybridizedScreenedPoissonCpu::Create(
        communicator, CpuSerialExecutionTag{}, PeriodicUnitTorusV1Tag{},
        degree, n, ScreenedPoissonPolicy{alpha}, solver, diagnostic);
  }
  else
  {
    status = FullyHybridizedScreenedPoissonCpu::Create(
        communicator, CpuMpiRank2ExecutionTag{}, PeriodicUnitTorusV1Tag{},
        degree, n, ScreenedPoissonPolicy{alpha}, solver, diagnostic);
  }
  return status.code == StatusCode::Ready && solver != nullptr &&
      solver->status() == status && solver->generation() == 1 &&
      solver->solution() == nullptr && solver->blocks() == nullptr &&
      solver->mesh() != nullptr && solver->volume_space() != nullptr &&
      solver->flux_space() != nullptr && solver->trace_space() != nullptr;
}
bool SolveFields(MPI_Comm communicator, const int ranks, const int degree,
                 const int n, const double alpha,
                 const FieldFunction& rho_function,
                 const FieldFunction& rho_hat_function,
                 const FieldFunction& source_function, SolveBundle& output,
                 const FailureInjection& injection = {})
{
  SolveBundle candidate;
  static int public_baselines_claimed = 0;
  if (public_baselines_claimed < 2 && degree == 1 && n == 8)
  {
    ++public_baselines_claimed;
    candidate.public_baseline.selected = true;
    candidate.public_baseline.ranks = ranks;
    candidate.public_baseline.degree = degree;
    candidate.public_baseline.n = n;
    MPI_Comm_rank(communicator, &candidate.public_baseline.rank);
  }
  PublicBaselineObservation* const observation =
      candidate.public_baseline.selected ? &candidate.public_baseline : nullptr;
  if (!CreateSolver(communicator, ranks, degree, n, alpha, candidate.solver,
                    candidate.create_status, candidate.create_diagnostic))
  {
    if (observation != nullptr)
    {
      observation->stage = candidate.create_status.code == StatusCode::Ready ?
          PublicBaselineStage::CreatePostcondition :
          PublicBaselineStage::CreateStatus;
      EmitPublicBaselineDiagnostic(candidate);
    }
    return false;
  }
  candidate.rho = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(candidate.solver->volume_space()));
  candidate.source = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(candidate.solver->volume_space()));
  candidate.rho_hat = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(candidate.solver->trace_space()));
  bool rho_projected = false;
  bool source_projected = false;
  bool trace_projected = false;
  if (!(rho_projected = ProjectVolume(
            *const_cast<mfem::ParFiniteElementSpace*>(
                candidate.solver->volume_space()),
            *candidate.rho, rho_function)) ||
      !(source_projected = ProjectVolume(
            *const_cast<mfem::ParFiniteElementSpace*>(
                candidate.solver->volume_space()),
            *candidate.source, source_function)) ||
      !(trace_projected = ProjectTrace(
            *const_cast<mfem::ParFiniteElementSpace*>(
                candidate.solver->trace_space()),
            *candidate.rho_hat, rho_hat_function)))
  {
    if (observation != nullptr)
    {
      observation->stage = !rho_projected ? PublicBaselineStage::ProjectRho :
          (!source_projected ? PublicBaselineStage::ProjectSource :
                               PublicBaselineStage::ProjectTrace);
      EmitPublicBaselineDiagnostic(candidate);
    }
    return false;
  }
  candidate.solve_status = candidate.solver->Solve(
      {candidate.rho.get(), candidate.rho_hat.get(), candidate.source.get(), 1},
      injection);
  const auto* const solution = candidate.solver->solution();
  const bool exact_configuration = solution != nullptr &&
      solution->solver.effective_gmres_krylov_dimension == 50 &&
      solution->solver.effective_gmres_max_iterations == 300 &&
      solution->solver.exact_gmres_configuration;
  const bool converged_gmres_completion = exact_configuration &&
      solution->solver.gmres_setup_status == 0 &&
      solution->solver.gmres_solve_status == 0 &&
      solution->solver.gmres_convergence_query_status == 0 &&
      solution->solver.gmres_converged == 1;
  const bool exact_zero_rhs_completion = exact_configuration &&
      !injection.force_trace_nonconvergence &&
      solution->solver.gmres_setup_status == -1 &&
      solution->solver.gmres_solve_status == -1 &&
      solution->solver.gmres_convergence_query_status == -1 &&
      solution->solver.gmres_converged == -1 &&
      solution->solver.gmres_iterations == 0 &&
      solution->solver.gmres_reported_residual == 0.0 &&
      solution->solver.true_right_hand_side_norm == 0.0 &&
      solution->solver.true_residual_norm == 0.0 &&
      ExactPositiveZero(solution->owned_true_sigma_hat) &&
      ExactZero(solution->local_sigma_hat);
  const bool true_residual_passed = solution != nullptr &&
      solution->solver.true_residual_norm <= 1.0e-14 +
          1.0e-12 * solution->solver.true_right_hand_side_norm;
  if (candidate.solve_status.code != StatusCode::Success ||
      (!converged_gmres_completion && !exact_zero_rhs_completion) ||
      !true_residual_passed)
  {
    if (observation != nullptr)
    {
      observation->stage =
          candidate.solve_status.code != StatusCode::Success ?
              PublicBaselineStage::SolveStatus :
          solution == nullptr ? PublicBaselineStage::SolutionPublication :
                                PublicBaselineStage::GmresPostcondition;
      EmitPublicBaselineDiagnostic(candidate);
    }
    const std::string label = "solve-fields-r" +
        std::to_string(candidate.public_baseline.rank) + "-of" +
        std::to_string(ranks) + "-k" + std::to_string(degree) + "-n" +
        std::to_string(n);
    PrintAttemptDiagnostic(label.c_str(), candidate.solver->last_attempt());
    return false;
  }
  output = std::move(candidate);
  return true;
}
bool SameBytes(const AuxiliarySolution& first,
               const AuxiliarySolution& second)
{
  const auto same_double = [](const double left, const double right)
  {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
  };
  const auto same = [](const std::vector<double>& left,
                       const std::vector<double>& right)
  {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(left.data(), right.data(),
                                    left.size() * sizeof(double)) == 0);
  };
  const auto same_restriction = [](const auto& left, const auto& right)
  {
    if (left.size() != right.size()) { return false; }
    for (std::size_t index = 0; index < left.size(); ++index)
    {
      const auto& a = left[index];
      const auto& b = right[index];
      if (a.element != b.element || a.row != b.row ||
          a.signed_local_vdof != b.signed_local_vdof ||
          a.decoded_local_vdof != b.decoded_local_vdof ||
          a.sign != b.sign || a.global_true_dof != b.global_true_dof ||
          a.local_true_dof != b.local_true_dof ||
          a.owner_rank != b.owner_rank)
      {
        return false;
      }
    }
    return true;
  };
  const auto& a = first.census;
  const auto& b = second.census;
  const auto& sa = first.solver;
  const auto& sb = second.solver;
  return first.polynomial_degree == second.polynomial_degree &&
      first.elements_per_direction == second.elements_per_direction &&
      same_double(first.alpha, second.alpha) &&
      first.input_generation == second.input_generation &&
      first.solution_generation == second.solution_generation &&
      a.rank == b.rank && a.ranks == b.ranks &&
      a.local_elements == b.local_elements &&
      a.global_elements == b.global_elements &&
      a.local_trace_dofs == b.local_trace_dofs &&
      a.owned_true_trace_dofs == b.owned_true_trace_dofs &&
      a.global_true_trace_dofs == b.global_true_trace_dofs &&
      a.owned_trace_entries == b.owned_trace_entries &&
      a.ghost_trace_entries == b.ghost_trace_entries &&
      a.negative_restriction_rows == b.negative_restriction_rows &&
      sa.gmres_configuration_query_status ==
          sb.gmres_configuration_query_status &&
      sa.gmres_krylov_dimension_query_return ==
          sb.gmres_krylov_dimension_query_return &&
      sa.gmres_relative_tolerance_query_return ==
          sb.gmres_relative_tolerance_query_return &&
      sa.gmres_absolute_tolerance_query_return ==
          sb.gmres_absolute_tolerance_query_return &&
      sa.gmres_maximum_iterations_query_return ==
          sb.gmres_maximum_iterations_query_return &&
      sa.gmres_setup_status == sb.gmres_setup_status &&
      sa.gmres_solve_status == sb.gmres_solve_status &&
      sa.gmres_convergence_query_status ==
          sb.gmres_convergence_query_status &&
      sa.gmres_converged == sb.gmres_converged &&
      sa.gmres_iterations == sb.gmres_iterations &&
      sa.effective_gmres_krylov_dimension ==
          sb.effective_gmres_krylov_dimension &&
      sa.effective_gmres_max_iterations ==
          sb.effective_gmres_max_iterations &&
      sa.effective_gmres_relative_tolerance_bits ==
          sb.effective_gmres_relative_tolerance_bits &&
      sa.effective_gmres_absolute_tolerance_bits ==
          sb.effective_gmres_absolute_tolerance_bits &&
      same_double(sa.gmres_reported_residual,
                  sb.gmres_reported_residual) &&
      same_double(sa.true_residual_norm, sb.true_residual_norm) &&
      same_double(sa.true_right_hand_side_norm,
                  sb.true_right_hand_side_norm) &&
      same_double(sa.maximum_local_recovery_residual,
                  sb.maximum_local_recovery_residual) &&
      same_double(sa.quadrature_repeat_relative_change,
                  sb.quadrature_repeat_relative_change) &&
      sa.effective_ilu_type == sb.effective_ilu_type &&
      sa.effective_ilu_level_of_fill == sb.effective_ilu_level_of_fill &&
      sa.effective_ilu_local_reordering ==
          sb.effective_ilu_local_reordering &&
      sa.exact_gmres_configuration == sb.exact_gmres_configuration &&
      sa.exact_ilu_configuration == sb.exact_ilu_configuration &&
      first.blocks.local_elements == second.blocks.local_elements &&
      first.blocks.scalar_interior_dofs ==
          second.blocks.scalar_interior_dofs &&
      first.blocks.interior_unknowns == second.blocks.interior_unknowns &&
      first.blocks.trace_unknowns == second.blocks.trace_unknowns &&
      same(first.sigma, second.sigma) && same(first.z, second.z) &&
      same(first.local_sigma_hat, second.local_sigma_hat) &&
      same(first.owned_true_sigma_hat, second.owned_true_sigma_hat) &&
      same(first.blocks.a, second.blocks.a) &&
      same(first.blocks.b, second.blocks.b) &&
      same(first.blocks.c, second.blocks.c) &&
      same(first.blocks.d, second.blocks.d) &&
      same(first.blocks.f, second.blocks.f) &&
      same(first.blocks.g, second.blocks.g) &&
      same(first.blocks.schur, second.blocks.schur) &&
      same(first.blocks.condensed_right_hand_side,
           second.blocks.condensed_right_hand_side) &&
      same_restriction(first.blocks.restriction,
                       second.blocks.restriction);
}
bool PhaseAudit(const AttemptDiagnostic& diagnostic,
                const AttemptPhase failed_phase, const int origin)
{
  if (diagnostic.status.phase != failed_phase ||
      diagnostic.status.origin_rank != origin)
  {
    return false;
  }
  for (std::size_t index = 0; index < diagnostic.phases.size(); ++index)
  {
    const auto& phase = diagnostic.phases[index];
    if (!phase.selected || phase.phase != static_cast<AttemptPhase>(index))
    {
      return false;
    }
    if (index < static_cast<std::size_t>(failed_phase) &&
        (!phase.entered || !phase.completed || !phase.passed ||
         phase.disposition != PhaseDisposition::Passed))
    {
      return false;
    }
    if (index == static_cast<std::size_t>(failed_phase) &&
        (!phase.entered || !phase.completed || phase.passed ||
         phase.disposition != PhaseDisposition::Failed))
    {
      return false;
    }
    if (index > static_cast<std::size_t>(failed_phase) &&
        (phase.entered || phase.completed || phase.passed ||
         phase.disposition !=
             PhaseDisposition::NotReached))
    {
      return false;
    }
  }
  return true;
}
bool SuccessfulPhaseAudit(const AttemptDiagnostic& diagnostic)
{
  if (diagnostic.status.code != StatusCode::Success ||
      diagnostic.status.phase != AttemptPhase::Publication ||
      diagnostic.status.origin_rank != -1)
  {
    return false;
  }
  return std::all_of(diagnostic.phases.begin(), diagnostic.phases.end(),
                     [](const PhaseDiagnostic& phase)
  {
    return phase.selected && phase.entered && phase.completed &&
        phase.passed && phase.disposition == PhaseDisposition::Passed;
  });
}
} // namespace
namespace
{
bool CheckConstantH1Comparison(MPI_Comm communicator, const SolveBundle& solve,
                               const double rho_value,
                               const double source_value,
                               const double expected_sigma,
                               double* const hdg_z_l2 = nullptr)
{
  H1ReferenceRequest request;
  request.degree = 1;
  request.elements_per_direction = 8;
  request.alpha = solve.solver->alpha();
  request.rho = [rho_value](const mfem::Vector&) { return rho_value; };
  request.source = [source_value](const mfem::Vector&)
  { return source_value; };
  H1ReferenceSolution h1;
  if (!SolveH1Reference(request, h1)) { return false; }
  const AuxiliarySolution& hdg = *solve.solver->solution();
  const mfem::ParFiniteElementSpace& hdg_space =
      *solve.solver->volume_space();
  mfem::ParMesh& hdg_mesh =
      *const_cast<mfem::ParMesh*>(solve.solver->mesh());
  const int scalar = hdg.blocks.scalar_interior_dofs;
  std::array<long double, 5> sums{};
  for (int element = 0; element < hdg_mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* const finite_element =
        hdg_space.GetFE(element);
    mfem::ElementTransformation* const transformation =
        hdg_mesh.GetElementTransformation(element);
    if (finite_element == nullptr || transformation == nullptr ||
        finite_element->GetDof() != scalar)
    {
      return false;
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), 16);
    mfem::Vector shape(scalar);
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      const mfem::IntegrationPoint& ip = rule.IntPoint(point_index);
      transformation->SetIntPoint(&ip);
      finite_element->CalcShape(ip, shape);
      double sigma_hdg = 0.0;
      double zx_hdg = 0.0;
      double zy_hdg = 0.0;
      const std::size_t sigma_offset =
          static_cast<std::size_t>(element) * scalar;
      const std::size_t z_offset =
          static_cast<std::size_t>(element) * 2 * scalar;
      for (int dof = 0; dof < scalar; ++dof)
      {
        sigma_hdg += shape[dof] * hdg.sigma[sigma_offset + dof];
        zx_hdg += shape[dof] * hdg.z[z_offset + dof];
        zy_hdg += shape[dof] * hdg.z[z_offset + scalar + dof];
      }
      mfem::Vector physical(2);
      transformation->Transform(ip, physical);
      const int ex = std::min(7, static_cast<int>(
          std::floor(physical[0] * 8.0)));
      const int ey = std::min(7, static_cast<int>(
          std::floor(physical[1] * 8.0)));
      const int h1_element = ex + 8 * ey;
      mfem::IntegrationPoint h1_ip;
      h1_ip.Set2(physical[0] * 8.0 - ex, physical[1] * 8.0 - ey);
      mfem::ElementTransformation* const h1_transformation =
          h1.mesh->GetElementTransformation(h1_element);
      if (h1_transformation == nullptr) { return false; }
      h1_transformation->SetIntPoint(&h1_ip);
      const double sigma_h1 = h1.full_solution->GetValue(h1_element, h1_ip);
      mfem::Vector gradient(2);
      h1.full_solution->GetGradient(*h1_transformation, gradient);
      const double weight = ip.weight * transformation->Weight();
      sums[0] += weight * (sigma_hdg - sigma_h1) *
          (sigma_hdg - sigma_h1);
      sums[1] += weight * sigma_h1 * sigma_h1;
      sums[2] += weight * (sigma_h1 - expected_sigma) *
          (sigma_h1 - expected_sigma);
      sums[3] += weight * (zx_hdg * zx_hdg + zy_hdg * zy_hdg);
      sums[4] += weight * (gradient[0] * gradient[0] +
                           gradient[1] * gradient[1]);
    }
  }
  std::array<double, 5> local{}, global{};
  for (std::size_t index = 0; index < sums.size(); ++index)
  {
    local[index] = static_cast<double>(sums[index]);
  }
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(global.size()),
                MPI_DOUBLE, MPI_SUM, communicator);
  const double h1_scale = std::max(1.0, std::sqrt(global[1]));
  const double sigma_cross = std::sqrt(global[0]) / h1_scale;
  const double sigma_exact = std::sqrt(global[2]) /
      std::max(1.0, std::abs(expected_sigma));
  const double z_hdg = std::sqrt(global[3]);
  const double z_h1 = std::sqrt(global[4]);
  if (hdg_z_l2 != nullptr) { *hdg_z_l2 = z_hdg; }
  const bool passed = sigma_cross <= 2.0e-12 &&
      sigma_exact <= 2.0e-12 && z_hdg <= 2.0e-12 && z_h1 <= 2.0e-12;
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  if (!passed && rank == 0)
  {
    std::cerr << "AUX_CONST_H1 rho=" << rho_value
              << " source=" << source_value
              << " expected_sigma=" << expected_sigma
              << " sigma_cross=" << sigma_cross
              << " sigma_exact=" << sigma_exact
              << " z_hdg=" << z_hdg
              << " z_h1=" << z_h1 << '\n';
  }
  return passed;
}
bool CheckZero(MPI_Comm communicator, const int ranks)
{
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, 1, 8, 0.1,
                   [](double, double) { return 1.2; },
                   [](double, double) { return 1.1; },
                   [](double, double) { return 0.0; }, solve))
  {
    return false;
  }
  const AuxiliarySolution& solution = *solve.solver->solution();
  const bool exact_zero_completion =
      solution.solver.gmres_setup_status == -1 &&
      solution.solver.gmres_solve_status == -1 &&
      solution.solver.gmres_convergence_query_status == -1 &&
      solution.solver.gmres_converged == -1 &&
      solution.solver.gmres_iterations == 0 &&
      solution.solver.gmres_reported_residual == 0.0 &&
      solution.solver.true_right_hand_side_norm == 0.0 &&
      solution.solver.true_residual_norm == 0.0 &&
      ExactPositiveZero(solution.owned_true_sigma_hat) &&
      ExactZero(solution.local_sigma_hat);
  bool sigma_passed = false;
  bool z_passed = false;
  bool trace_passed = false;
  bool h1_passed = false;
  bool passed = exact_zero_completion &&
      (sigma_passed = GlobalNorm(communicator, solution.sigma) <= 2.0e-12) &&
      (z_passed = GlobalNorm(communicator, solution.z) <= 2.0e-12) &&
      (trace_passed = GlobalNorm(
          communicator, solution.owned_true_sigma_hat) <= 2.0e-12) &&
      (h1_passed = CheckConstantH1Comparison(
          communicator, solve, 1.2, 0.0, 0.0));
  if (solve.public_baseline.selected)
  {
    PublicBaselineObservation* const observation = &solve.public_baseline;
    observation->stage = !sigma_passed ? PublicBaselineStage::CheckZeroSigma :
        !z_passed ? PublicBaselineStage::CheckZeroZ :
        !trace_passed ? PublicBaselineStage::CheckZeroTrace :
        !h1_passed ? PublicBaselineStage::CheckZeroH1 :
                     PublicBaselineStage::CheckZeroPassed;
    EmitPublicBaselineDiagnostic(solve);
  }
  return passed;
}
bool CheckConstant(MPI_Comm communicator, const int ranks,
                   const double alpha)
{
  const double expected = 8.0 * alpha;
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, 1, 8, alpha,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.0; },
                   [expected](double, double) { return expected; }, solve))
  {
    return false;
  }
  const AuxiliarySolution& solution = *solve.solver->solution();
  const auto solved_field_error = [expected](
      const std::vector<double>& values)
  {
    long double square = 0.0L;
    for (const double value : values)
    {
      const long double difference =
          static_cast<long double>(value) - expected;
      square += difference * difference;
    }
    return std::sqrt(static_cast<double>(square)) /
        std::max(1.0, std::abs(expected));
  };
  const double sigma_error = solved_field_error(solution.sigma);
  const double local_trace_error = solved_field_error(
      solution.local_sigma_hat);
  const double owned_trace_error = solved_field_error(
      solution.owned_true_sigma_hat);
  const bool sigma = sigma_error <= 2.0e-12;
  const bool local_trace = local_trace_error <= 2.0e-12;
  const bool owned_trace = owned_trace_error <= 2.0e-12;
  double z_norm = std::numeric_limits<double>::infinity();
  const bool h1 = CheckConstantH1Comparison(
      communicator, solve, 1.0, expected, expected, &z_norm);
  const bool passed = sigma && local_trace && owned_trace &&
      z_norm <= 2.0e-12 && h1;
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  if (rank == 0 && alpha <= 1.0e-4)
  {
    const double rhs_norm = solution.solver.true_right_hand_side_norm;
    const int rhs_scale_exponent = rhs_norm > 0.0 ?
        -std::ilogb(rhs_norm) : 0;
    const double scaled_rhs_norm = std::scalbn(
        rhs_norm, rhs_scale_exponent);
    std::cerr << "AUX_RHS_SCALING alpha=" << alpha
              << " exponent=" << rhs_scale_exponent
              << " original_rhs_norm=" << rhs_norm
              << " scaled_rhs_norm=" << scaled_rhs_norm
              << " setup=" << solution.solver.gmres_setup_status
              << " solve=" << solution.solver.gmres_solve_status
              << " convergence_query="
              << solution.solver.gmres_convergence_query_status
              << " converged=" << solution.solver.gmres_converged
              << " iterations=" << solution.solver.gmres_iterations
              << " reported_residual="
              << solution.solver.gmres_reported_residual
              << " true_residual="
              << solution.solver.true_residual_norm << '\n';
  }
  if (!passed && rank == 0)
  {
    std::cerr << "AUX_CONST alpha=" << alpha
              << " expected=" << expected
              << " sigma=" << sigma
              << " sigma_error=" << sigma_error
              << " local_trace=" << local_trace
              << " local_trace_error=" << local_trace_error
              << " owned_trace=" << owned_trace
              << " owned_trace_error=" << owned_trace_error
              << " z_norm=" << z_norm
              << " gmres_iterations=" << solution.solver.gmres_iterations
              << " true_residual=" << solution.solver.true_residual_norm
              << " rhs_norm=" << solution.solver.true_right_hand_side_norm
              << " h1=" << h1 << '\n';
  }
  return passed;
}
bool CheckManufacturedDifferential()
{
  const double alpha = 0.1;
  double maximum = 0.0;
  for (int point = 0; point < 41; ++point)
  {
    const double x = static_cast<double>(point) / 40.0;
    const double sigma = std::sin(2.0 * pi * x);
    const double sigma_xx = -4.0 * pi * pi * sigma;
    const double source = (1.0 + 4.0 * alpha * pi * pi) * sigma;
    maximum = std::max(maximum,
                       std::abs(sigma - alpha * sigma_xx - source));
  }
  return maximum <= 1.0e-12;
}
struct ComparisonMetrics
{
  double sigma_hdg = 0.0;
  double sigma_h1 = 0.0;
  double sigma_cross = 0.0;
  double z_hdg = 0.0;
  double z_h1 = 0.0;
  double z_analytic = 0.0;
  double z_cross = 0.0;
  H1ReferenceSolution h1;
};
bool ComputeComparison(MPI_Comm communicator, const int ranks,
                       const int degree, const int n,
                       ComparisonMetrics& metrics,
                       const bool corrupt_h1_dof = false,
                       const bool corrupt_analytic_constitutive_z = false)
{
  constexpr double alpha = 0.1;
  constexpr double analytic_constitutive_mutation = 1.0e-4;
  const FieldFunction exact = [](const double x, const double)
  { return std::sin(2.0 * pi * x); };
  const FieldFunction source = [](const double x, const double)
  { return (1.0 + 0.1 * 4.0 * pi * pi) * std::sin(2.0 * pi * x); };
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, degree, n, alpha,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.0; }, source, solve))
  {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::cerr << "AUX_COMPARISON rank=" << rank << " ranks=" << ranks
              << " degree=" << degree << " n=" << n
              << " stage=hdg_solve\n";
    return false;
  }
  H1ReferenceRequest request;
  request.degree = degree;
  request.elements_per_direction = n;
  request.alpha = alpha;
  request.rho = [](const mfem::Vector&) { return 1.0; };
  request.source = [](const mfem::Vector& point)
  { return (1.0 + 0.1 * 4.0 * pi * pi) * std::sin(2.0 * pi * point[0]); };
  if (!SolveH1Reference(request, metrics.h1))
  {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::cerr << "AUX_COMPARISON rank=" << rank << " ranks=" << ranks
              << " degree=" << degree << " n=" << n
              << " stage=h1_solve\n";
    return false;
  }
  if (corrupt_h1_dof)
  {
    mfem::Vector true_dofs;
    metrics.h1.full_solution->GetTrueDofs(true_dofs);
    if (true_dofs.Size() == 0) { return false; }
    true_dofs[0] += 2.0 * std::max(1.0, true_dofs.Normlinf());
    metrics.h1.full_solution->SetFromTrueDofs(true_dofs);
  }
  const AuxiliarySolution& hdg = *solve.solver->solution();
  const mfem::ParFiniteElementSpace& space = *solve.solver->volume_space();
  mfem::ParMesh& mesh = *const_cast<mfem::ParMesh*>(solve.solver->mesh());
  const int scalar = hdg.blocks.scalar_interior_dofs;
  std::array<long double, 12> sums{};
  const int quadrature_order = 4 * degree + 12;
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* const fe = space.GetFE(element);
    mfem::ElementTransformation* const transformation =
        mesh.GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr || fe->GetDof() != scalar)
    {
      return false;
    }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        fe->GetGeomType(), quadrature_order);
    mfem::Vector shape(scalar);
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      const mfem::IntegrationPoint& ip = rule.IntPoint(point_index);
      transformation->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      double sigma_hdg = 0.0, zx_hdg = 0.0, zy_hdg = 0.0;
      const std::size_t sigma_offset = static_cast<std::size_t>(element) * scalar;
      const std::size_t z_offset = static_cast<std::size_t>(element) * 2 * scalar;
      for (int dof = 0; dof < scalar; ++dof)
      {
        sigma_hdg += shape[dof] * hdg.sigma[sigma_offset + dof];
        zx_hdg += shape[dof] * hdg.z[z_offset + dof];
        zy_hdg += shape[dof] * hdg.z[z_offset + scalar + dof];
      }
      mfem::Vector physical(2);
      transformation->Transform(ip, physical);
      const int ex = std::min(n - 1,
          static_cast<int>(std::floor(physical[0] * n)));
      const int ey = std::min(n - 1,
          static_cast<int>(std::floor(physical[1] * n)));
      const int h1_element = ex + n * ey;
      mfem::IntegrationPoint h1_ip;
      h1_ip.Set2(physical[0] * n - ex, physical[1] * n - ey);
      mfem::ElementTransformation* const h1_transformation =
          metrics.h1.mesh->GetElementTransformation(h1_element);
      h1_transformation->SetIntPoint(&h1_ip);
      const double sigma_h1 = metrics.h1.full_solution->GetValue(
          h1_element, h1_ip);
      mfem::Vector gradient(2);
      metrics.h1.full_solution->GetGradient(*h1_transformation, gradient);
      const double sigma_exact = exact(physical[0], physical[1]);
      // Legitimate flux truth: rho=1 and rho*z-grad(Sigma)=0 imply
      // z=grad(Sigma), evaluated independently at the physical point.
      const double zx_exact = 2.0 * pi * std::cos(2.0 * pi * physical[0]);
      const double zy_exact = 0.0;
      // Evaluate the exact-field gradient and the card's constitutive z
      // independently. Neither path calls production nor reuses zx_exact.
      const std::array<double, 2> gradient_sigma_exact{{
          2.0 * pi * std::cos(2.0 * pi * physical[0]), 0.0}};
      const double rho_exact = 1.0;
      std::array<double, 2> constitutive_z_exact{{
          (2.0 * pi / rho_exact) *
              std::cos(2.0 * pi * physical[0]),
          0.0}};
      if (corrupt_analytic_constitutive_z)
      {
        constitutive_z_exact[0] += analytic_constitutive_mutation;
      }
      const double weight = ip.weight * transformation->Weight();
      sums[0] += weight * (sigma_hdg - sigma_exact) *
          (sigma_hdg - sigma_exact);
      sums[1] += weight * (sigma_h1 - sigma_exact) *
          (sigma_h1 - sigma_exact);
      sums[2] += weight * (sigma_hdg - sigma_h1) * (sigma_hdg - sigma_h1);
      sums[3] += weight * (zx_hdg - zx_exact) * (zx_hdg - zx_exact) +
          weight * (zy_hdg - zy_exact) * (zy_hdg - zy_exact);
      sums[4] += weight * (gradient[0] - zx_exact) *
          (gradient[0] - zx_exact) +
          weight * (gradient[1] - zy_exact) *
              (gradient[1] - zy_exact);
      sums[5] += weight * (zx_hdg - gradient[0]) *
          (zx_hdg - gradient[0]) +
          weight * (zy_hdg - gradient[1]) * (zy_hdg - gradient[1]);
      sums[6] += weight * sigma_exact * sigma_exact;
      sums[7] += weight * (zx_exact * zx_exact + zy_exact * zy_exact);
      sums[8] += weight * sigma_h1 * sigma_h1;
      sums[9] += weight * (gradient[0] * gradient[0] +
                           gradient[1] * gradient[1]);
      const double analytic_x =
          rho_exact * constitutive_z_exact[0] - gradient_sigma_exact[0];
      const double analytic_y =
          rho_exact * constitutive_z_exact[1] - gradient_sigma_exact[1];
      sums[10] += weight * (analytic_x * analytic_x +
                            analytic_y * analytic_y);
      sums[11] += weight *
          (gradient_sigma_exact[0] * gradient_sigma_exact[0] +
           gradient_sigma_exact[1] * gradient_sigma_exact[1]);
    }
  }
  std::array<double, 12> local{}, global{};
  for (std::size_t i = 0; i < sums.size(); ++i)
  {
    local[i] = static_cast<double>(sums[i]);
  }
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(global.size()),
                MPI_DOUBLE, MPI_SUM, communicator);
  const double sigma_scale = std::max(1.0, std::sqrt(global[6]));
  const double z_scale = std::max(1.0, std::sqrt(global[7]));
  metrics.sigma_hdg = std::sqrt(global[0]) / sigma_scale;
  metrics.sigma_h1 = std::sqrt(global[1]) / sigma_scale;
  metrics.sigma_cross = std::sqrt(global[2]) /
      std::max(1.0, std::sqrt(global[8]));
  metrics.z_hdg = std::sqrt(global[3]) / z_scale;
  metrics.z_h1 = std::sqrt(global[4]) / z_scale;
  metrics.z_analytic = std::sqrt(global[10]) /
      std::max(1.0, std::sqrt(global[11]));
  metrics.z_cross = std::sqrt(global[5]) /
      std::max(1.0, std::sqrt(global[9]));
  const bool passed = IsFinite(metrics.sigma_hdg) &&
      IsFinite(metrics.sigma_h1) &&
      IsFinite(metrics.sigma_cross) && IsFinite(metrics.z_hdg) &&
      IsFinite(metrics.z_h1) && IsFinite(metrics.z_analytic) &&
      metrics.z_analytic <= 1.0e-12 && IsFinite(metrics.z_cross);
  if (!passed && !corrupt_analytic_constitutive_z)
  {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::cerr << "AUX_COMPARISON rank=" << rank << " ranks=" << ranks
              << " degree=" << degree << " n=" << n
              << " stage=metrics"
              << " sigma_hdg=" << metrics.sigma_hdg
              << " sigma_h1=" << metrics.sigma_h1
              << " sigma_cross=" << metrics.sigma_cross
              << " z_hdg=" << metrics.z_hdg
              << " z_h1=" << metrics.z_h1
              << " z_analytic=" << metrics.z_analytic
              << " z_cross=" << metrics.z_cross << '\n';
  }
  return passed;
}
bool NonIncreasing(const std::array<double, 3>& values)
{
  return values[1] <= values[0] && values[2] <= values[1];
}
bool HierarchyGate(const std::array<double, 3>& values, const int degree,
                   const bool gradient)
{
  const double coefficient = gradient ? 8.0 : 4.0;
  const int power = gradient ? degree : degree + 1;
  for (int index = 0; index < 3; ++index)
  {
    const int n = 8 << index;
    if (!IsFinite(values[index]) || values[index] < 0.0 ||
        values[index] > std::max(5.0e-11,
            coefficient * std::pow(static_cast<double>(n), -power)))
    {
      return false;
    }
  }
  if (!NonIncreasing(values)) { return false; }
  if (values[0] == 0.0 || values[2] == 0.0 ||
      values[0] <= 5.0e-11 || values[2] <= 5.0e-11)
  {
    return true;
  }
  const double slope = (std::log2(values[0]) - std::log2(values[2])) / 2.0;
  return slope >= (gradient ? degree - 0.25 : degree + 0.25);
}
bool AllComputed(const std::array<bool, 3>& computed)
{
  return std::all_of(computed.begin(), computed.end(),
                     [](const bool value) { return value; });
}
bool HdgHierarchyAccepted(
    const std::array<ComparisonMetrics, 3>& metrics,
    const std::array<bool, 3>& computed, const int degree)
{
  const std::array<double, 3> sigma_hdg{{metrics[0].sigma_hdg,
      metrics[1].sigma_hdg, metrics[2].sigma_hdg}};
  const std::array<double, 3> z_hdg{{metrics[0].z_hdg,
      metrics[1].z_hdg, metrics[2].z_hdg}};
  return AllComputed(computed) &&
      HierarchyGate(sigma_hdg, degree, false) &&
      HierarchyGate(z_hdg, degree, true);
}
bool H1HierarchyAccepted(
    const std::array<ComparisonMetrics, 3>& metrics,
    const std::array<bool, 3>& computed, const int degree)
{
  const std::array<double, 3> sigma_h1{{metrics[0].sigma_h1,
      metrics[1].sigma_h1, metrics[2].sigma_h1}};
  const std::array<double, 3> z_h1{{metrics[0].z_h1,
      metrics[1].z_h1, metrics[2].z_h1}};
  return AllComputed(computed) &&
      HierarchyGate(sigma_h1, degree, false) &&
      HierarchyGate(z_h1, degree, true);
}
bool CrossHierarchyAccepted(
    const std::array<ComparisonMetrics, 3>& metrics,
    const std::array<bool, 3>& computed, const int degree)
{
  const std::array<double, 3> sigma_cross{{metrics[0].sigma_cross,
      metrics[1].sigma_cross, metrics[2].sigma_cross}};
  const std::array<double, 3> z_cross{{metrics[0].z_cross,
      metrics[1].z_cross, metrics[2].z_cross}};
  return AllComputed(computed) &&
      HierarchyGate(sigma_cross, degree, false) &&
      HierarchyGate(z_cross, degree, true);
}
bool AnalyticZHierarchyAccepted(
    const std::array<ComparisonMetrics, 3>& metrics,
    const std::array<bool, 3>& computed)
{
  return AllComputed(computed) &&
      std::all_of(metrics.begin(), metrics.end(),
                  [](const ComparisonMetrics& value)
                  { return value.z_analytic <= 1.0e-12; });
}
bool ComparisonHierarchyAccepted(
    const std::array<ComparisonMetrics, 3>& metrics,
    const std::array<bool, 3>& computed, const int degree)
{
  return HdgHierarchyAccepted(metrics, computed, degree) &&
      H1HierarchyAccepted(metrics, computed, degree) &&
      CrossHierarchyAccepted(metrics, computed, degree) &&
      AnalyticZHierarchyAccepted(metrics, computed);
}
bool H1CorruptedDofRejected()
{
  std::array<ComparisonMetrics, 3> baseline;
  std::array<bool, 3> baseline_computed{};
  for (int level = 0; level < 3; ++level)
  {
    baseline_computed[level] = ComputeComparison(
        MPI_COMM_SELF, 1, 1, 8 << level, baseline[level]);
  }
  if (!ComparisonHierarchyAccepted(baseline, baseline_computed, 1))
  {
    return false;
  }
  std::array<ComparisonMetrics, 3> corrupted;
  std::array<bool, 3> corrupted_computed{};
  for (int level = 0; level < 3; ++level)
  {
    corrupted_computed[level] = ComputeComparison(
        MPI_COMM_SELF, 1, 1, 8 << level, corrupted[level], level == 0);
  }
  const bool computed = std::all_of(
      corrupted_computed.begin(), corrupted_computed.end(),
      [](const bool value) { return value; });
  const bool hierarchy_rejected =
      !ComparisonHierarchyAccepted(corrupted, corrupted_computed, 1);
  const bool cross_rejected =
      corrupted[0].sigma_cross > baseline[0].sigma_cross + 1.0e-8;
  if (!(computed && hierarchy_rejected && cross_rejected))
  {
    std::cerr << "AUX_H1_MUTATION computed=" << computed
              << " hierarchy_rejected=" << hierarchy_rejected
              << " cross_rejected=" << cross_rejected
              << " baseline_cross=" << baseline[0].sigma_cross
              << " corrupted_cross=" << corrupted[0].sigma_cross << '\n';
  }
  return computed && hierarchy_rejected && cross_rejected;
}
bool AnalyticConstitutiveZMutationRejected()
{
  ComparisonMetrics baseline, mutated;
  const bool baseline_accepted = ComputeComparison(
      MPI_COMM_SELF, 1, 1, 8, baseline);
  const bool mutation_accepted = ComputeComparison(
      MPI_COMM_SELF, 1, 1, 8, mutated, false, true);
  return baseline_accepted && !mutation_accepted &&
      baseline.z_analytic <= 1.0e-12 &&
      IsFinite(mutated.z_analytic) && mutated.z_analytic > 1.0e-8;
}
using FaceNodeKey = std::tuple<int, int, int, std::int64_t>;
FaceNodeKey GeometricFaceNodeKey(mfem::ElementTransformation& transformation,
                                 const mfem::IntegrationPoint& node,
                                 const int n)
{
  mfem::Vector point(2);
  mfem::Vector center_point(2);
  mfem::IntegrationPoint center;
  center.Set1w(0.5, 1.0);
  transformation.Transform(node, point);
  transformation.Transform(center, center_point);
  const double x = point[0] - std::floor(point[0]);
  const double y = point[1] - std::floor(point[1]);
  const double center_x = center_point[0] - std::floor(center_point[0]);
  const double center_y = center_point[1] - std::floor(center_point[1]);
  transformation.SetIntPoint(&center);
  const mfem::DenseMatrix& jacobian = transformation.Jacobian();
  const bool vertical = std::abs(jacobian(1, 0)) >
      std::abs(jacobian(0, 0));
  const double fixed = vertical ? center_x : center_y;
  const double varying = vertical ? y : x;
  const int line = static_cast<int>(std::llround(fixed * n)) % n;
  const double center_varying = vertical ? center_y : center_x;
  const double scaled_varying = center_varying * n;
  int segment = static_cast<int>(std::floor(scaled_varying));
  if (segment == n) { segment = 0; }
  const std::int64_t node_key = std::llround(varying * 1000000000000.0);
  return {vertical ? 0 : 1, line, segment, node_key};
}
int CartesianElementOwner(const int ex, const int n, const int ranks)
{
  return ranks == 1 ? 0 : (ex < n / 2 ? 0 : 1);
}
int MinimumAdjacentOwner(const FaceNodeKey& key, const int n,
                         const int ranks)
{
  const int axis = std::get<0>(key);
  const int line = std::get<1>(key);
  const int segment = std::get<2>(key);
  if (axis == 0)
  {
    const int left = CartesianElementOwner((line + n - 1) % n, n, ranks);
    const int right = CartesianElementOwner(line, n, ranks);
    return std::min(left, right);
  }
  return CartesianElementOwner(segment, n, ranks);
}
struct CanonicalTraceView
{
  std::map<FaceNodeKey, double> values;
  std::map<int, FaceNodeKey> local_true_keys;
  std::map<FaceNodeKey, int> key_to_local_true;
  std::map<std::tuple<int, int, int>, int> face_node_counts;
};
bool BuildCanonicalTraceView(const FullyHybridizedScreenedPoissonCpu& solver,
                             const AuxiliarySolution& solution,
                             const int n, CanonicalTraceView& view)
{
  const mfem::ParMesh* const mesh = solver.mesh();
  const mfem::ParFiniteElementSpace* const space = solver.trace_space();
  if (mesh == nullptr || space == nullptr ||
      solution.local_sigma_hat.size() !=
          static_cast<std::size_t>(space->GetVSize()))
  {
    return false;
  }
  view = {};
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element =
        space->GetFaceElement(face);
    mfem::IsoparametricTransformation transformation;
    mesh->GetFaceTransformation(face, &transformation);
    mfem::Array<int> signed_dofs;
    space->GetFaceVDofs(face, signed_dofs);
    if (finite_element == nullptr ||
        signed_dofs.Size() != finite_element->GetDof())
    {
      return false;
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    for (int dof = 0; dof < finite_element->GetDof(); ++dof)
    {
      const FaceNodeKey key = GeometricFaceNodeKey(
          transformation, nodes.IntPoint(dof), n);
      int sign = 0;
      const int decoded = Decode(signed_dofs[dof], sign);
      if (decoded < 0 || decoded >= space->GetVSize()) { return false; }
      const double value = static_cast<double>(sign) *
          solution.local_sigma_hat[static_cast<std::size_t>(decoded)];
      const auto inserted_value = view.values.emplace(key, value);
      if (!inserted_value.second &&
          std::abs(inserted_value.first->second - value) >
              64.0 * std::numeric_limits<double>::epsilon() *
                  std::max(1.0, std::abs(value)))
      {
        return false;
      }
      const int local_true = space->GetLocalTDofNumber(decoded);
      if (local_true >= 0)
      {
        const auto inserted_local = view.local_true_keys.emplace(
            local_true, key);
        const auto inserted_key = view.key_to_local_true.emplace(
            key, local_true);
        if ((!inserted_local.second && inserted_local.first->second != key) ||
            (!inserted_key.second && inserted_key.first->second != local_true))
        {
          return false;
        }
      }
    }
  }
  for (const auto& item : view.values)
  {
    ++view.face_node_counts[{std::get<0>(item.first),
                             std::get<1>(item.first),
                             std::get<2>(item.first)}];
  }
  return true;
}
bool ValidateIndependentOwnership(MPI_Comm communicator, const int ranks,
                                  const int rank, const int degree,
                                  const int n,
                                  const CanonicalTraceView& view,
                                  const AuxiliarySolution& solution)
{
  if ((ranks != 1 && ranks != 2) || rank < 0 || rank >= ranks) { return false; }
  const int nodes_per_face = degree + 1;
  if (!std::all_of(view.face_node_counts.begin(),
                   view.face_node_counts.end(),
                   [nodes_per_face](const auto& item)
                   { return item.second == nodes_per_face; }))
  {
    return false;
  }
  std::map<FaceNodeKey, int> expected_owned;
  for (const auto& item : view.values)
  {
    if (MinimumAdjacentOwner(item.first, n, ranks) == rank)
    {
      expected_owned.emplace(item.first, rank);
    }
  }
  const int expected_local = ranks == 1 ? 2 * n * n * nodes_per_face :
      n * (rank == 0 ? n + 1 : n - 1) * nodes_per_face;
  int local_owned = static_cast<int>(view.key_to_local_true.size());
  int global_owned = 0;
  MPI_Allreduce(&local_owned, &global_owned, 1, MPI_INT, MPI_SUM,
                communicator);
  if (local_owned != expected_local ||
      global_owned != 2 * n * n * nodes_per_face ||
      expected_owned.size() != static_cast<std::size_t>(expected_local) ||
      view.local_true_keys.size() != expected_owned.size() ||
      solution.owned_true_sigma_hat.size() != expected_owned.size())
  {
    return false;
  }
  for (int local_true = 0; local_true < expected_local; ++local_true)
  {
    const auto actual = view.local_true_keys.find(local_true);
    if (actual == view.local_true_keys.end() ||
        expected_owned.count(actual->second) != 1)
    {
      return false;
    }
  }
  return std::all_of(expected_owned.begin(), expected_owned.end(),
                     [&](const auto& item)
                     { return view.key_to_local_true.count(item.first) == 1; });
}
bool OrientationAndCount(MPI_Comm communicator, const int ranks,
                         const int degree)
{
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, degree, 8, 0.1,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.1; },
                   [](double, double) { return 0.25; }, solve))
  {
    return false;
  }
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  const AuxiliarySolution& solution = *solve.solver->solution();
  CanonicalTraceView view;
  return BuildCanonicalTraceView(*solve.solver, solution, 8, view) &&
      ValidateIndependentOwnership(communicator, ranks, rank, degree, 8,
                                   view, solution);
}
bool OppositeSidePermutation(const int degree)
{
  SolveBundle solve;
  if (!SolveFields(MPI_COMM_SELF, 1, degree, 8, 0.1,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.1; },
                   [](double, double) { return 0.25; }, solve))
  {
    return false;
  }
  struct FaceRecord
  {
    int element = -1;
    int orientation = 0;
    std::vector<GlobalIndex> columns;
  };
  std::map<std::vector<GlobalIndex>, std::vector<FaceRecord>> shared_faces;
  const auto& blocks = solve.solver->solution()->blocks;
  const int face_dofs = degree + 1;
  for (int element = 0; element < blocks.local_elements; ++element)
  {
    mfem::Array<int> faces, orientations;
    solve.solver->mesh()->GetElementEdges(element, faces, orientations);
    if (faces.Size() != 4 || orientations.Size() != faces.Size())
    {
      return false;
    }
    for (int local_face = 0; local_face < 4; ++local_face)
    {
      FaceRecord record;
      record.element = element;
      record.orientation = orientations[local_face];
      record.columns.resize(static_cast<std::size_t>(face_dofs));
      for (int local_dof = 0; local_dof < face_dofs; ++local_dof)
      {
        const int row = local_face * face_dofs + local_dof;
        const auto& entry = blocks.restriction[
            static_cast<std::size_t>(element) * blocks.trace_unknowns + row];
        if (entry.element != element || entry.row != row ||
            (record.orientation != 1 && record.orientation != -1))
        {
          return false;
        }
        record.columns[static_cast<std::size_t>(local_dof)] =
            entry.global_true_dof;
      }
      std::vector<GlobalIndex> key(record.columns);
      std::sort(key.begin(), key.end());
      shared_faces[key].push_back(std::move(record));
    }
  }
  int opposite_pairs = 0;
  for (const auto& item : shared_faces)
  {
    const auto& records = item.second;
    if (records.size() != 2) { return false; }
    const auto& first = records[0];
    const auto& second = records[1];
    if (first.orientation == second.orientation) { continue; }
    ++opposite_pairs;
    const FaceRecord& positive = first.orientation > 0 ? first : second;
    const FaceRecord& negative = first.orientation < 0 ? first : second;
    bool reversed = face_dofs > 1 && positive.columns != negative.columns;
    for (int dof = 0; dof < face_dofs; ++dof)
    {
      const int opposite = face_dofs - 1 - dof;
      reversed = reversed &&
          positive.columns[static_cast<std::size_t>(dof)] ==
              negative.columns[static_cast<std::size_t>(opposite)];
    }
    if (!reversed) { return false; }
  }
  return opposite_pairs > 0;
}
using InteriorNodeKey = std::tuple<std::int64_t, std::int64_t, int>;
bool BuildCanonicalInteriorView(
    const FullyHybridizedScreenedPoissonCpu& solver,
    const AuxiliarySolution& solution,
    std::map<InteriorNodeKey, double>& values)
{
  const mfem::ParMesh* const mesh = solver.mesh();
  const mfem::ParFiniteElementSpace* const space = solver.volume_space();
  if (mesh == nullptr || space == nullptr) { return false; }
  values.clear();
  const int scalar = solution.blocks.scalar_interior_dofs;
  if (scalar <= 0 || solution.sigma.size() !=
          static_cast<std::size_t>(mesh->GetNE()) * scalar ||
      solution.z.size() !=
          static_cast<std::size_t>(mesh->GetNE()) * 2 * scalar)
  {
    return false;
  }
  for (int element = 0; element < mesh->GetNE(); ++element)
  {
    const mfem::FiniteElement* const finite_element = space->GetFE(element);
    mfem::IsoparametricTransformation transformation;
    mesh->GetElementTransformation(element, &transformation);
    if (finite_element == nullptr || finite_element->GetDof() != scalar)
    {
      return false;
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    for (int dof = 0; dof < scalar; ++dof)
    {
      mfem::Vector point(2);
      transformation.Transform(nodes.IntPoint(dof), point);
      const auto x = std::llround(point[0] * 1000000000000.0);
      const auto y = std::llround(point[1] * 1000000000000.0);
      const std::size_t sigma_offset =
          static_cast<std::size_t>(element) * scalar;
      const std::size_t z_offset =
          static_cast<std::size_t>(element) * 2 * scalar;
      const std::array<double, 3> components{{
          solution.sigma[sigma_offset + dof],
          solution.z[z_offset + dof],
          solution.z[z_offset + scalar + dof]}};
      for (int component = 0; component < 3; ++component)
      {
        if (!values.emplace(InteriorNodeKey{x, y, component},
                            components[static_cast<std::size_t>(component)])
                 .second)
        {
          return false;
        }
      }
    }
  }
  return true;
}
template <typename Key>
double CanonicalMapDifference(const std::map<Key, double>& candidate,
                              const std::map<Key, double>& reference)
{
  if (candidate.empty() || candidate.size() > reference.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  long double difference = 0.0L;
  long double norm = 0.0L;
  for (const auto& item : candidate)
  {
    const auto found = reference.find(item.first);
    if (found == reference.end())
    {
      return std::numeric_limits<double>::infinity();
    }
    const long double delta = static_cast<long double>(item.second) -
        found->second;
    difference += delta * delta;
    norm += static_cast<long double>(found->second) * found->second;
  }
  return std::sqrt(static_cast<double>(difference)) /
      std::max(1.0, std::sqrt(static_cast<double>(norm)));
}
bool MapOwnedTrueVector(const CanonicalTraceView& view,
                        const std::vector<double>& vector,
                        std::map<FaceNodeKey, double>& mapped)
{
  if (view.local_true_keys.size() != vector.size()) { return false; }
  mapped.clear();
  for (const auto& item : view.local_true_keys)
  {
    if (item.first < 0 ||
        static_cast<std::size_t>(item.first) >= vector.size() ||
        !mapped.emplace(item.second,
                        vector[static_cast<std::size_t>(item.first)]).second)
    {
      return false;
    }
  }
  return true;
}
bool BuildCanonicalProbe(const CanonicalTraceView& view,
                         const std::map<FaceNodeKey, int>& ordinals,
                         std::vector<double>& probe)
{
  probe.assign(view.local_true_keys.size(), 0.0);
  const double scale = std::max(1.0, static_cast<double>(ordinals.size()));
  for (const auto& item : view.local_true_keys)
  {
    const auto ordinal = ordinals.find(item.second);
    if (ordinal == ordinals.end() || item.first < 0 ||
        static_cast<std::size_t>(item.first) >= probe.size())
    {
      return false;
    }
    const double magnitude = static_cast<double>(ordinal->second + 1) / scale;
    probe[static_cast<std::size_t>(item.first)] =
        (ordinal->second % 2 == 0 ? 1.0 : -1.0) * magnitude;
  }
  return true;
}
bool TraceParity(MPI_Comm communicator, const int world_rank,
                 const int degree)
{
  SolveBundle parallel, serial;
  const FieldFunction rho = [](double x, double y)
  { return 1.0 + 0.02 * std::sin(2.0 * pi * x) *
      std::cos(2.0 * pi * y); };
  const FieldFunction rho_hat = [](double x, double y)
  { return 1.05 + 0.01 * std::cos(2.0 * pi * x) *
      std::sin(2.0 * pi * y); };
  const FieldFunction source = [](double x, double y)
  { return 0.2 + 0.03 * std::cos(2.0 * pi * x) *
      std::cos(2.0 * pi * y); };
  if (!SolveFields(communicator, 2, degree, 8, 0.1, rho, rho_hat, source,
                   parallel) ||
      !SolveFields(MPI_COMM_SELF, 1, degree, 8, 0.1, rho, rho_hat, source,
                   serial))
  {
    return false;
  }
  const AuxiliarySolution& p = *parallel.solver->solution();
  const AuxiliarySolution& s = *serial.solver->solution();
  CanonicalTraceView parallel_trace, serial_trace;
  std::map<InteriorNodeKey, double> parallel_interior, serial_interior;
  if (!BuildCanonicalTraceView(*parallel.solver, p, 8, parallel_trace) ||
      !BuildCanonicalTraceView(*serial.solver, s, 8, serial_trace) ||
      !ValidateIndependentOwnership(communicator, 2, world_rank, degree, 8,
                                    parallel_trace, p) ||
      !ValidateIndependentOwnership(MPI_COMM_SELF, 1, 0, degree, 8,
                                    serial_trace, s) ||
      !BuildCanonicalInteriorView(*parallel.solver, p, parallel_interior) ||
      !BuildCanonicalInteriorView(*serial.solver, s, serial_interior))
  {
    return false;
  }
  std::map<FaceNodeKey, int> ordinals;
  int ordinal = 0;
  for (const auto& item : serial_trace.values)
  {
    ordinals.emplace(item.first, ordinal++);
  }
  std::vector<double> parallel_probe, serial_probe;
  std::vector<double> parallel_action, parallel_rhs;
  std::vector<double> serial_action, serial_rhs;
  std::map<FaceNodeKey, double> parallel_solution_map, serial_solution_map;
  if (!BuildCanonicalProbe(parallel_trace, ordinals, parallel_probe) ||
      !BuildCanonicalProbe(serial_trace, ordinals, serial_probe) ||
      !MapOwnedTrueVector(parallel_trace, p.owned_true_sigma_hat,
                          parallel_solution_map) ||
      !MapOwnedTrueVector(serial_trace, s.owned_true_sigma_hat,
                          serial_solution_map) ||
      !parallel.solver->ApplyPublishedTraceOperator(
          parallel_probe, parallel_action, parallel_rhs) ||
      !serial.solver->ApplyPublishedTraceOperator(
          serial_probe, serial_action, serial_rhs))
  {
    return false;
  }
  std::map<FaceNodeKey, double> parallel_action_map, parallel_rhs_map;
  std::map<FaceNodeKey, double> serial_action_map, serial_rhs_map;
  return MapOwnedTrueVector(parallel_trace, parallel_action,
                            parallel_action_map) &&
      MapOwnedTrueVector(parallel_trace, parallel_rhs, parallel_rhs_map) &&
      MapOwnedTrueVector(serial_trace, serial_action, serial_action_map) &&
      MapOwnedTrueVector(serial_trace, serial_rhs, serial_rhs_map) &&
      CanonicalMapDifference(parallel_solution_map, serial_solution_map) <=
          5.0e-11 &&
      CanonicalMapDifference(parallel_interior, serial_interior) <= 5.0e-11 &&
      CanonicalMapDifference(parallel_action_map, serial_action_map) <=
          5.0e-11 &&
      CanonicalMapDifference(parallel_rhs_map, serial_rhs_map) <= 5.0e-11;
}
bool ProjectionV3RankTwoCensus(MPI_Comm communicator, const int world_rank)
{
  SolveBundle solve;
  if (!SolveFields(communicator, 2, 1, 8, 0.1,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.0; },
                   [](double, double) { return 0.0; }, solve))
  {
    return false;
  }
  const AuxiliarySolution& solution = *solve.solver->solution();
  CanonicalTraceView view;
  if (!BuildCanonicalTraceView(*solve.solver, solution, 8, view))
  {
    return false;
  }
  // The isolated operator owns the Sigmahat block. Reconstruct the full
  // projection-v3 tuple census independently by adjoining the four Uhat
  // component blocks on the same canonical face-node ownership map.
  const std::int64_t local =
      static_cast<std::int64_t>(view.local_true_keys.size()) * 5;
  std::int64_t global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT64_T, MPI_SUM, communicator);
  const std::int64_t expected_local = world_rank == 0 ? 720 : 560;
  return local == expected_local && global == 1280 &&
      view.local_true_keys.size() ==
          static_cast<std::size_t>(world_rank == 0 ? 144 : 112);
}
bool InvalidAlpha(MPI_Comm communicator, const int ranks, const double alpha)
{
  std::unique_ptr<FullyHybridizedScreenedPoissonCpu> solver;
  AttemptDiagnostic diagnostic;
  Status status = ranks == 1 ?
      FullyHybridizedScreenedPoissonCpu::Create(
          communicator, CpuSerialExecutionTag{}, PeriodicUnitTorusV1Tag{},
          1, 8, ScreenedPoissonPolicy{alpha}, solver, diagnostic) :
      FullyHybridizedScreenedPoissonCpu::Create(
          communicator, CpuMpiRank2ExecutionTag{}, PeriodicUnitTorusV1Tag{},
          1, 8, ScreenedPoissonPolicy{alpha}, solver, diagnostic);
  return status.code == StatusCode::InvalidArgument &&
      status.phase == AttemptPhase::Setup && status.origin_rank == 0 &&
      solver == nullptr && diagnostic.status == status &&
      diagnostic.phases[0].selected && diagnostic.phases[0].entered &&
      diagnostic.phases[0].completed && !diagnostic.phases[0].passed &&
      std::none_of(diagnostic.phases.begin() + 1,
                   diagnostic.phases.end(), [](const auto& phase)
                   { return phase.selected || phase.entered ||
                       phase.completed || phase.passed; });
}
bool CreateMismatch(MPI_Comm communicator, const int rank, const int kind)
{
  std::unique_ptr<FullyHybridizedScreenedPoissonCpu> solver;
  AttemptDiagnostic diagnostic;
  Status status;
  if (kind == 0 && rank == 1)
  {
    status = FullyHybridizedScreenedPoissonCpu::Create(
        communicator, CpuSerialExecutionTag{}, PeriodicUnitTorusV1Tag{},
        2, 8, ScreenedPoissonPolicy{0.1}, solver, diagnostic);
  }
  else
  {
    const int degree = kind == 1 && rank == 1 ? 3 : 2;
    const int n = kind == 2 && rank == 1 ? 16 : 8;
    const std::uint64_t alpha_bits = kind == 3 && rank == 1 ?
        UINT64_C(0x3fb999999999999b) : UINT64_C(0x3fb999999999999a);
    double alpha = 0.0;
    std::memcpy(&alpha, &alpha_bits, sizeof(alpha));
    status = FullyHybridizedScreenedPoissonCpu::Create(
        communicator, CpuMpiRank2ExecutionTag{}, PeriodicUnitTorusV1Tag{},
        degree, n, ScreenedPoissonPolicy{alpha}, solver, diagnostic);
  }
  return status.code == StatusCode::InvalidArgument &&
      status.phase == AttemptPhase::Setup && status.origin_rank == 1 &&
      solver == nullptr && diagnostic.status == status &&
      diagnostic.phases[0].selected && diagnostic.phases[0].entered &&
      diagnostic.phases[0].completed && !diagnostic.phases[0].passed &&
      std::none_of(diagnostic.phases.begin() + 1,
                   diagnostic.phases.end(), [](const auto& phase)
                   { return phase.selected || phase.entered ||
                       phase.completed || phase.passed; });
}
bool InjectedFailure(MPI_Comm communicator, const int ranks,
                     const AttemptPhase phase, const int injection_rank,
                     bool* const successful_reentry = nullptr)
{
  if (successful_reentry != nullptr) { *successful_reentry = false; }
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, 1, 8, 0.1,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.1; },
                   [](double x, double y)
                   { return 0.2 + 0.01 * std::sin(2.0 * pi * x) *
                       std::cos(2.0 * pi * y); }, solve))
  {
    return false;
  }
  const AuxiliarySolution baseline = *solve.solver->solution();
  const Status baseline_status = solve.solver->status();
  const std::uint64_t baseline_generation = solve.solver->generation();
  const AuxiliarySolution* const baseline_solution = solve.solver->solution();
  const mfem::ParMesh* const baseline_mesh = solve.solver->mesh();
  const mfem::ParFiniteElementSpace* const baseline_volume =
      solve.solver->volume_space();
  const mfem::ParFiniteElementSpace* const baseline_trace =
      solve.solver->trace_space();
  const Status failed = solve.solver->Solve(
      {solve.rho.get(), solve.rho_hat.get(), solve.source.get(), 2},
      {phase, injection_rank});
  const bool rolled_back = failed.code == StatusCode::InjectedFailure &&
      PhaseAudit(solve.solver->last_attempt(), phase, injection_rank) &&
      solve.solver->status() == baseline_status &&
      solve.solver->generation() == baseline_generation &&
      solve.solver->solution() == baseline_solution &&
      solve.solver->mesh() == baseline_mesh &&
      solve.solver->volume_space() == baseline_volume &&
      solve.solver->trace_space() == baseline_trace &&
      SameBytes(baseline, *solve.solver->solution());
  if (!rolled_back) { return false; }
  const Status retried_status = solve.solver->Solve(
      {solve.rho.get(), solve.rho_hat.get(), solve.source.get(), 3});
  if (retried_status.code != StatusCode::Success ||
      !SuccessfulPhaseAudit(solve.solver->last_attempt()) ||
      solve.solver->generation() != baseline_generation + 1 ||
      solve.solver->solution() == nullptr)
  {
    return false;
  }
  AuxiliarySolution retried = *solve.solver->solution();
  // Input/publication generation fields are transaction metadata; all
  // numerical bytes and solver evidence must match the uninterrupted solve.
  retried.input_generation = baseline.input_generation;
  retried.solution_generation = baseline.solution_generation;
  const bool matched = SameBytes(baseline, retried);
  if (successful_reentry != nullptr) { *successful_reentry = matched; }
  return matched;
}
bool ExhaustedTraceSolveIsAtomic(MPI_Comm communicator, const int ranks)
{
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, 4, 8, 0.1,
                   [](double x, double y)
                   { return 1.0 + 0.12 * std::sin(2.0 * pi * x) *
                       std::cos(4.0 * pi * y); },
                   [](double x, double y)
                   { return 1.1 + 0.08 * std::cos(4.0 * pi * x) *
                       std::sin(2.0 * pi * y); },
                   [](double x, double y)
                   { return 0.2 + std::sin(6.0 * pi * x) *
                       std::cos(4.0 * pi * y); }, solve))
  {
    return false;
  }
  const AuxiliarySolution baseline = *solve.solver->solution();
  const Status baseline_status = solve.solver->status();
  const std::uint64_t baseline_generation = solve.solver->generation();
  const AuxiliarySolution* const baseline_solution = solve.solver->solution();
  const mfem::ParMesh* const baseline_mesh = solve.solver->mesh();
  const mfem::ParFiniteElementSpace* const baseline_volume =
      solve.solver->volume_space();
  const mfem::ParFiniteElementSpace* const baseline_trace =
      solve.solver->trace_space();
  std::vector<double> probe(baseline.owned_true_sigma_hat.size(), 0.0);
  for (std::size_t index = 0; index < probe.size(); ++index)
  {
    probe[index] = (index % 2 == 0 ? 1.0 : -1.0) *
        static_cast<double>(index + 1) /
        std::max(1.0, static_cast<double>(probe.size()));
  }
  std::vector<double> baseline_action, baseline_rhs;
  if (!solve.solver->ApplyPublishedTraceOperator(
          probe, baseline_action, baseline_rhs))
  {
    return false;
  }
  FailureInjection injection;
  injection.force_trace_nonconvergence = true;
  const Status failed = solve.solver->Solve(
      {solve.rho.get(), solve.rho_hat.get(), solve.source.get(), 2},
      injection);
  const AttemptDiagnostic& attempt = solve.solver->last_attempt();
  std::vector<double> preserved_action, preserved_rhs;
  const bool rolled_back = failed.code == StatusCode::TraceSolveFailure &&
      failed.phase == AttemptPhase::TraceSolve && failed.origin_rank == 0 &&
      PhaseAudit(attempt, AttemptPhase::TraceSolve, 0) &&
      attempt.solver.gmres_setup_status == 0 &&
      attempt.solver.gmres_solve_status == HYPRE_ERROR_CONV &&
      attempt.solver.gmres_convergence_query_status == HYPRE_ERROR_CONV &&
      attempt.solver.gmres_converged == 0 &&
      attempt.solver.gmres_iterations == 1 &&
      attempt.solver.effective_gmres_krylov_dimension == 50 &&
      attempt.solver.effective_gmres_max_iterations == 1 &&
      attempt.solver.exact_gmres_configuration &&
      solve.solver->status() == baseline_status &&
      solve.solver->generation() == baseline_generation &&
      solve.solver->solution() == baseline_solution &&
      solve.solver->mesh() == baseline_mesh &&
      solve.solver->volume_space() == baseline_volume &&
      solve.solver->trace_space() == baseline_trace &&
      SameBytes(baseline, *solve.solver->solution()) &&
      solve.solver->ApplyPublishedTraceOperator(
          probe, preserved_action, preserved_rhs) &&
      RelativeDifference(preserved_action, baseline_action) == 0.0 &&
      RelativeDifference(preserved_rhs, baseline_rhs) == 0.0;
  return rolled_back;
}
bool ExactZeroTraceNonconvergenceIsAtomic(MPI_Comm communicator,
                                          const int ranks)
{
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, 1, 8, 0.1,
                   [](double, double) { return 1.2; },
                   [](double, double) { return 1.1; },
                   [](double, double) { return 0.0; }, solve))
  {
    return false;
  }
  const AuxiliarySolution baseline = *solve.solver->solution();
  const Status baseline_status = solve.solver->status();
  const std::uint64_t baseline_generation = solve.solver->generation();
  const AuxiliarySolution* const baseline_solution = solve.solver->solution();
  FailureInjection injection;
  injection.force_trace_nonconvergence = true;
  const Status failed = solve.solver->Solve(
      {solve.rho.get(), solve.rho_hat.get(), solve.source.get(), 2},
      injection);
  const AttemptDiagnostic& attempt = solve.solver->last_attempt();
  const bool rolled_back = failed.code == StatusCode::TraceSolveFailure &&
      failed.phase == AttemptPhase::TraceSolve && failed.origin_rank == 0 &&
      PhaseAudit(attempt, AttemptPhase::TraceSolve, 0) &&
      attempt.solver.gmres_setup_status == 0 &&
      attempt.solver.gmres_solve_status == 0 &&
      attempt.solver.gmres_convergence_query_status == 0 &&
      attempt.solver.gmres_converged == 0 &&
      attempt.solver.gmres_iterations == 0 &&
      attempt.solver.gmres_reported_residual == 0.0 &&
      solve.solver->status() == baseline_status &&
      solve.solver->generation() == baseline_generation &&
      solve.solver->solution() == baseline_solution &&
      SameBytes(baseline, *solve.solver->solution());
  return rolled_back;
}
bool AsymmetricRank2SnapshotFailure(MPI_Comm communicator, const int rank)
{
  SolveBundle solve;
  if (!SolveFields(communicator, 2, 1, 8, 0.1,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.1; },
                   [](double, double) { return 0.2; }, solve))
  {
    return false;
  }
  const AuxiliarySolution baseline = *solve.solver->solution();
  const Status baseline_status = solve.solver->status();
  const std::uint64_t baseline_generation = solve.solver->generation();
  const AuxiliarySolution* const baseline_solution = solve.solver->solution();
  auto rho = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(solve.solver->volume_space()));
  *rho = *solve.rho;
  if (rank == 1) { (*rho)[0] = 0.0; }
  const Status failed = solve.solver->Solve(
      {rho.get(), solve.rho_hat.get(), solve.source.get(), 2});
  const bool rolled_back = failed.code == StatusCode::InvalidArgument &&
      failed.phase == AttemptPhase::Setup && failed.origin_rank == 1 &&
      PhaseAudit(solve.solver->last_attempt(), AttemptPhase::Setup, 1) &&
      solve.solver->status() == baseline_status &&
      solve.solver->generation() == baseline_generation &&
      solve.solver->solution() == baseline_solution &&
      SameBytes(baseline, *solve.solver->solution());
  if (!rolled_back) { return false; }
  const Status retried_status = solve.solver->Solve(
      {solve.rho.get(), solve.rho_hat.get(), solve.source.get(), 3});
  if (retried_status.code != StatusCode::Success ||
      !SuccessfulPhaseAudit(solve.solver->last_attempt()) ||
      solve.solver->generation() != baseline_generation + 1 ||
      solve.solver->solution() == nullptr)
  {
    return false;
  }
  AuxiliarySolution retried = *solve.solver->solution();
  retried.input_generation = baseline.input_generation;
  retried.solution_generation = baseline.solution_generation;
  return SameBytes(baseline, retried);
}
enum class InvalidSnapshotKind
{
  RhoZero,
  RhoHatNegative,
  SourceNan,
  ForeignSpace
};
bool InvalidSnapshot(MPI_Comm communicator, const int ranks,
                     const InvalidSnapshotKind kind)
{
  SolveBundle solve;
  if (!SolveFields(communicator, ranks, 1, 8, 0.1,
                   [](double, double) { return 1.0; },
                   [](double, double) { return 1.1; },
                   [](double, double) { return 0.2; }, solve))
  {
    return false;
  }
  const AuxiliarySolution baseline = *solve.solver->solution();
  const Status baseline_status = solve.solver->status();
  const std::uint64_t baseline_generation = solve.solver->generation();
  const AuxiliarySolution* const baseline_solution = solve.solver->solution();
  const mfem::ParMesh* const baseline_mesh = solve.solver->mesh();
  const mfem::ParFiniteElementSpace* const baseline_volume =
      solve.solver->volume_space();
  const mfem::ParFiniteElementSpace* const baseline_trace =
      solve.solver->trace_space();
  auto rho = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(solve.solver->volume_space()));
  auto source = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(solve.solver->volume_space()));
  auto rho_hat = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(solve.solver->trace_space()));
  *rho = *solve.rho;
  *source = *solve.source;
  *rho_hat = *solve.rho_hat;
  std::unique_ptr<mfem::L2_FECollection> foreign_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> foreign_space;
  std::unique_ptr<mfem::ParGridFunction> foreign;
  if (kind == InvalidSnapshotKind::RhoZero) { (*rho)[0] = 0.0; }
  if (kind == InvalidSnapshotKind::RhoHatNegative) { (*rho_hat)[0] = -1.0; }
  if (kind == InvalidSnapshotKind::SourceNan)
  {
    (*source)[0] = std::numeric_limits<double>::quiet_NaN();
  }
  if (kind == InvalidSnapshotKind::ForeignSpace)
  {
    foreign_collection = std::make_unique<mfem::L2_FECollection>(
        1, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
    foreign_space = std::make_unique<mfem::ParFiniteElementSpace>(
        const_cast<mfem::ParMesh*>(solve.solver->mesh()),
        foreign_collection.get(), 1, mfem::Ordering::byNODES);
    foreign = std::make_unique<mfem::ParGridFunction>(foreign_space.get());
    const double* input = rho->HostRead();
    double* output = foreign->HostWrite();
    std::copy(input, input + rho->Size(), output);
  }
  const PrescribedSnapshot snapshot{
      kind == InvalidSnapshotKind::ForeignSpace ? foreign.get() : rho.get(),
      rho_hat.get(), source.get(), 2};
  const Status failed = solve.solver->Solve(snapshot);
  return failed.code == StatusCode::InvalidArgument &&
      failed.phase == AttemptPhase::Setup && failed.origin_rank == 0 &&
      solve.solver->status() == baseline_status &&
      solve.solver->generation() == baseline_generation &&
      solve.solver->solution() == baseline_solution &&
      solve.solver->mesh() == baseline_mesh &&
      solve.solver->volume_space() == baseline_volume &&
      solve.solver->trace_space() == baseline_trace &&
      SameBytes(baseline, *solve.solver->solution());
}
} // namespace
namespace
{
struct DenseFactor
{
  int size = 0;
  std::vector<double> lu;
  std::vector<int> pivots;
};
bool FactorDense(const std::vector<double>& matrix, const int size,
                 DenseFactor& output)
{
  if (size <= 0 || matrix.size() !=
          static_cast<std::size_t>(size) * size)
  {
    return false;
  }
  output.size = size;
  output.lu = matrix;
  output.pivots.resize(static_cast<std::size_t>(size));
  for (int column = 0; column < size; ++column)
  {
    int pivot = column;
    double maximum = std::abs(output.lu[
        static_cast<std::size_t>(column) * size + column]);
    for (int row = column + 1; row < size; ++row)
    {
      const double value = std::abs(output.lu[
          static_cast<std::size_t>(row) * size + column]);
      if (value > maximum) { maximum = value; pivot = row; }
    }
    if (!IsFinite(maximum) || !(maximum > 0.0)) { return false; }
    output.pivots[static_cast<std::size_t>(column)] = pivot;
    if (pivot != column)
    {
      for (int entry = 0; entry < size; ++entry)
      {
        std::swap(output.lu[static_cast<std::size_t>(column) * size + entry],
                  output.lu[static_cast<std::size_t>(pivot) * size + entry]);
      }
    }
    const double diagonal = output.lu[
        static_cast<std::size_t>(column) * size + column];
    for (int row = column + 1; row < size; ++row)
    {
      double& multiplier = output.lu[
          static_cast<std::size_t>(row) * size + column];
      multiplier /= diagonal;
      for (int entry = column + 1; entry < size; ++entry)
      {
        output.lu[static_cast<std::size_t>(row) * size + entry] -=
            multiplier * output.lu[
                static_cast<std::size_t>(column) * size + entry];
      }
    }
  }
  return true;
}
bool SolveDense(const DenseFactor& factor, std::vector<double>& values,
                const int right_hand_sides)
{
  const int n = factor.size;
  if (n <= 0 || right_hand_sides <= 0 || values.size() !=
          static_cast<std::size_t>(n) * right_hand_sides)
  {
    return false;
  }
  for (int column = 0; column < n; ++column)
  {
    const int pivot = factor.pivots[static_cast<std::size_t>(column)];
    if (pivot != column)
      for (int rhs = 0; rhs < right_hand_sides; ++rhs)
      {
        std::swap(values[static_cast<std::size_t>(column) * right_hand_sides + rhs],
                  values[static_cast<std::size_t>(pivot) * right_hand_sides + rhs]);
      }
  }
  for (int column = 0; column < n; ++column)
  {
    for (int row = column + 1; row < n; ++row)
      for (int rhs = 0; rhs < right_hand_sides; ++rhs)
      {
        values[static_cast<std::size_t>(row) * right_hand_sides + rhs] -=
            factor.lu[static_cast<std::size_t>(row) * n + column] *
            values[static_cast<std::size_t>(column) * right_hand_sides + rhs];
      }
  }
  for (int row = n - 1; row >= 0; --row)
    for (int rhs = 0; rhs < right_hand_sides; ++rhs)
    {
      double value = values[static_cast<std::size_t>(row) * right_hand_sides + rhs];
      for (int column = row + 1; column < n; ++column)
      {
        value -= factor.lu[static_cast<std::size_t>(row) * n + column] *
            values[static_cast<std::size_t>(column) * right_hand_sides + rhs];
      }
      values[static_cast<std::size_t>(row) * right_hand_sides + rhs] =
          value / factor.lu[static_cast<std::size_t>(row) * n + row];
    }
  return std::all_of(values.begin(), values.end(), IsFinite);
}
struct IndependentOracle
{
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> volume_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> volume_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::unique_ptr<mfem::ParGridFunction> rho;
  std::unique_ptr<mfem::ParGridFunction> rho_hat;
  std::unique_ptr<mfem::ParGridFunction> source;
  int elements = 4;
  int ni = 12;
  int nt = 8;
  int true_dofs = 16;
  std::vector<double> a, b, c, d, f, g, schur, rhs;
  std::vector<double> restriction; // element-major 8x16
  std::vector<double> full_matrix;
  std::vector<double> full_rhs;
  std::vector<double> full_solution;
  std::vector<double> global_schur;
  std::vector<double> global_rhs;
  std::vector<double> condensed_solution;
  std::vector<double> recovered;
  double full_condensed_relative_difference =
      std::numeric_limits<double>::infinity();
};
struct FrozenRestrictionIncidence
{
  int element;
  int row;
  int element_orientation;
  int face;
  int face_orientation;
  int geometric_dof;
  int torus_dof;
  int sign;
};
// Frozen integer oracle for the four quads
//   e0=(0,1,4,3), e1=(1,2,5,4),
//   e2=(3,4,7,6), e3=(4,5,8,7).
// Element-local block rows apply the element-edge orientation to MFEM's
// face-oriented VDof order. In 2-D, GetFaceVDofs has already applied the
// face-edge orientation, so the geometric endpoint order is their composition.
// Faces 2, 3, 6, 8, 9, and 11 have negative face orientation in this mesh.
// DG_Interface(VALUE) contributes no orientation sign, so all expected signs
// remain +1. No coordinate, floating-point key, associative container, or
// production numbering algorithm enters P_e.
constexpr std::array<FrozenRestrictionIncidence, 32>
    frozen_restriction_incidence{{
        {0, 0,  1,  0,  1,  0,  8, 1},
        {0, 1,  1,  0,  1,  1,  9, 1},
        {0, 2,  1,  1,  1,  2,  4, 1},
        {0, 3,  1,  1,  1,  3,  5, 1},
        {0, 4, -1,  2, -1,  4, 12, 1},
        {0, 5, -1,  2, -1,  5, 13, 1},
        {0, 6, -1,  3, -1,  6,  0, 1},
        {0, 7, -1,  3, -1,  7,  1, 1},
        {1, 0,  1,  4,  1,  8, 10, 1},
        {1, 1,  1,  4,  1,  9, 11, 1},
        {1, 2,  1,  5,  1, 10,  0, 1},
        {1, 3,  1,  5,  1, 11,  1, 1},
        {1, 4, -1,  6, -1, 12, 14, 1},
        {1, 5, -1,  6, -1, 13, 15, 1},
        {1, 6, -1,  1,  1,  3,  5, 1},
        {1, 7, -1,  1,  1,  2,  4, 1},
        {2, 0,  1,  2, -1,  5, 13, 1},
        {2, 1,  1,  2, -1,  4, 12, 1},
        {2, 2,  1,  7,  1, 14,  6, 1},
        {2, 3,  1,  7,  1, 15,  7, 1},
        {2, 4, -1,  8, -1, 16,  8, 1},
        {2, 5, -1,  8, -1, 17,  9, 1},
        {2, 6, -1,  9, -1, 18,  2, 1},
        {2, 7, -1,  9, -1, 19,  3, 1},
        {3, 0,  1,  6, -1, 13, 15, 1},
        {3, 1,  1,  6, -1, 12, 14, 1},
        {3, 2,  1, 10,  1, 20,  2, 1},
        {3, 3,  1, 10,  1, 21,  3, 1},
        {3, 4, -1, 11, -1, 22, 10, 1},
        {3, 5, -1, 11, -1, 23, 11, 1},
        {3, 6, -1,  7,  1, 15,  7, 1},
        {3, 7, -1,  7,  1, 14,  6, 1}}};
std::unique_ptr<mfem::ParMesh> BuildOracleMesh()
{
  // Keep the four affine cells as a valid MFEM mesh. The independent oracle
  // imposes the unit-torus equivalence with the frozen P_e incidence table;
  // a periodic four-quad MFEM mesh would require unsupported vertex-pair edges.
  mfem::Mesh cartesian = mfem::Mesh::MakeCartesian2D(
      2, 2, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<int> partition(static_cast<std::size_t>(cartesian.GetNE()), 0);
  return std::make_unique<mfem::ParMesh>(MPI_COMM_SELF, cartesian,
                                         partition.data());
}
bool BuildIndependentRestriction(IndependentOracle& oracle)
{
  oracle.restriction.assign(
      static_cast<std::size_t>(oracle.elements) * oracle.nt *
          oracle.true_dofs,
      0.0);
  std::vector<int> row_counts(32, 0);
  std::vector<int> geometric_counts(24, 0);
  std::vector<int> geometric_columns(24, -1);
  std::vector<int> column_counts(
      static_cast<std::size_t>(oracle.true_dofs), 0);
  for (const FrozenRestrictionIncidence& entry :
       frozen_restriction_incidence)
  {
    if (entry.element < 0 || entry.element >= oracle.elements ||
        entry.row < 0 || entry.row >= oracle.nt ||
        (entry.element_orientation != -1 && entry.element_orientation != 1) ||
        entry.face < 0 || entry.face >= oracle.mesh->GetNumFaces() ||
        (entry.face_orientation != -1 && entry.face_orientation != 1) ||
        entry.geometric_dof < 0 || entry.geometric_dof >= 24 ||
        entry.torus_dof < 0 || entry.torus_dof >= oracle.true_dofs ||
        (entry.sign != -1 && entry.sign != 1))
    {
      return false;
    }
    const int local_face = entry.row / 2;
    const int local_face_dof = entry.row % 2;
    mfem::Array<int> faces, element_orientations;
    oracle.mesh->GetElementEdges(entry.element, faces, element_orientations);
    mfem::Array<int> face_edges, face_orientations;
    oracle.mesh->GetFaceEdges(entry.face, face_edges, face_orientations);
    const int composed_orientation =
        entry.element_orientation * entry.face_orientation;
    const int expected_geometric_dof = 2 * entry.face +
        (composed_orientation > 0 ? local_face_dof : 1 - local_face_dof);
    if (faces.Size() != 4 || element_orientations.Size() != 4 ||
        faces[local_face] != entry.face ||
        element_orientations[local_face] != entry.element_orientation ||
        face_edges.Size() != 1 || face_orientations.Size() != 1 ||
        face_edges[0] != entry.face ||
        face_orientations[0] != entry.face_orientation ||
        expected_geometric_dof != entry.geometric_dof)
    {
      return false;
    }
    const std::size_t offset =
        (static_cast<std::size_t>(entry.element) * oracle.nt + entry.row) *
        oracle.true_dofs + entry.torus_dof;
    if (oracle.restriction[offset] != 0.0) { return false; }
    oracle.restriction[offset] = static_cast<double>(entry.sign);
    ++row_counts[static_cast<std::size_t>(entry.element * oracle.nt +
                                         entry.row)];
    int& geometric_column =
        geometric_columns[static_cast<std::size_t>(entry.geometric_dof)];
    if (geometric_column >= 0 && geometric_column != entry.torus_dof)
    {
      return false;
    }
    geometric_column = entry.torus_dof;
    ++geometric_counts[static_cast<std::size_t>(entry.geometric_dof)];
    ++column_counts[static_cast<std::size_t>(entry.torus_dof)];
  }
  return oracle.mesh && oracle.trace_space &&
      oracle.mesh->GetNE() == 4 && oracle.mesh->GetNumFaces() == 12 &&
      oracle.trace_space->GetVSize() == 24 &&
      std::all_of(row_counts.begin(), row_counts.end(),
                  [](const int count) { return count == 1; }) &&
      std::all_of(geometric_counts.begin(), geometric_counts.end(),
                  [](const int count) { return count == 1 || count == 2; }) &&
      std::count(geometric_counts.begin(), geometric_counts.end(), 1) == 16 &&
      std::count(geometric_counts.begin(), geometric_counts.end(), 2) == 8 &&
      std::all_of(column_counts.begin(), column_counts.end(),
                  [](const int count) { return count == 2; });
}
bool Interpolate(const mfem::Vector& values,
                 const mfem::Array<int>& signed_dofs,
                 const mfem::Vector& shape, double& result)
{
  if (signed_dofs.Size() != shape.Size()) { return false; }
  result = 0.0;
  for (int dof = 0; dof < shape.Size(); ++dof)
  {
    int sign = 0;
    const int decoded = Decode(signed_dofs[dof], sign);
    if (decoded < 0 || decoded >= values.Size()) { return false; }
    result += static_cast<double>(sign) * values[decoded] * shape[dof];
  }
  return IsFinite(result);
}
double IndependentMeasure(mfem::ElementTransformation& transformation,
                          const mfem::Geometry::Type geometry,
                          const int order)
{
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(geometry, order);
  double result = 0.0;
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    const mfem::IntegrationPoint& ip = rule.IntPoint(point);
    transformation.SetIntPoint(&ip);
    result += ip.weight * transformation.Weight();
  }
  return result;
}
bool AssembleIndependentOracle(IndependentOracle& oracle,
                               const int stabilization_mutation = 0)
{
  constexpr int scalar = 4;
  constexpr int face_scalar = 2;
  constexpr int order = 12;
  oracle.a.assign(4 * 12 * 12, 0.0);
  oracle.b.assign(4 * 12 * 8, 0.0);
  oracle.c.assign(4 * 8 * 12, 0.0);
  oracle.d.assign(4 * 8 * 8, 0.0);
  oracle.f.assign(4 * 12, 0.0);
  oracle.g.assign(4 * 8, 0.0);
  for (int element = 0; element < 4; ++element)
  {
    const mfem::FiniteElement* const fe = oracle.volume_space->GetFE(element);
    mfem::ElementTransformation* const transformation =
        oracle.mesh->GetElementTransformation(element);
    mfem::Array<int> volume_dofs;
    oracle.volume_space->GetElementVDofs(element, volume_dofs);
    if (fe == nullptr || transformation == nullptr || fe->GetDof() != scalar)
    {
      return false;
    }
    const auto index = [](const int component, const int dof)
    { return component * scalar + dof; };
    const std::size_t ao = static_cast<std::size_t>(element) * 12 * 12;
    const std::size_t bo = static_cast<std::size_t>(element) * 12 * 8;
    const std::size_t co = static_cast<std::size_t>(element) * 8 * 12;
    const std::size_t doff = static_cast<std::size_t>(element) * 8 * 8;
    const std::size_t fo = static_cast<std::size_t>(element) * 12;
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, order);
    mfem::Vector shape(scalar);
    mfem::DenseMatrix derivative(scalar, 2);
    for (int point = 0; point < volume_rule.GetNPoints(); ++point)
    {
      const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
      transformation->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*transformation, derivative);
      double rho = 0.0, source = 0.0;
      if (!Interpolate(*oracle.rho, volume_dofs, shape, rho) ||
          !Interpolate(*oracle.source, volume_dofs, shape, source) ||
          !(rho > 0.0))
      {
        return false;
      }
      const double weight = ip.weight * transformation->Weight();
      for (int row = 0; row < scalar; ++row)
      {
        oracle.f[fo + index(2, row)] += weight * shape[row] * source;
        for (int column = 0; column < scalar; ++column)
        {
          oracle.a[ao + index(0, row) * 12 + index(1, column)] +=
              weight * shape[row] * rho * shape[column];
          oracle.a[ao + index(1, row) * 12 + index(2, column)] +=
              weight * shape[row] * rho * shape[column];
          oracle.a[ao + index(0, row) * 12 + index(0, column)] +=
              weight * derivative(row, 0) * shape[column];
          oracle.a[ao + index(1, row) * 12 + index(0, column)] +=
              weight * derivative(row, 1) * shape[column];
          oracle.a[ao + index(2, row) * 12 + index(0, column)] +=
              weight * shape[row] * shape[column] / rho;
          oracle.a[ao + index(2, row) * 12 + index(1, column)] +=
              0.1 * weight * derivative(row, 0) * shape[column];
          oracle.a[ao + index(2, row) * 12 + index(2, column)] +=
              0.1 * weight * derivative(row, 1) * shape[column];
        }
      }
    }
    const double element_measure = IndependentMeasure(
        *transformation, mfem::Geometry::SQUARE, order);
    mfem::Array<int> faces, orientations;
    oracle.mesh->GetElementEdges(element, faces, orientations);
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* const face_transformation =
          oracle.mesh->GetFaceElementTransformations(face);
      const mfem::FiniteElement* const trace_fe =
          oracle.trace_space->GetFaceElement(face);
      mfem::Array<int> trace_dofs;
      oracle.trace_space->GetFaceVDofs(face, trace_dofs);
      mfem::ElementTransformation* const face_only =
          oracle.mesh->GetFaceTransformation(face);
      const double face_measure = IndependentMeasure(
          *face_only, mfem::Geometry::SEGMENT, order);
      const double height = element_measure / face_measure;
      const bool first = face_transformation->Elem1No == element;
      if (!first && face_transformation->Elem2No != element) { return false; }
      const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
          mfem::Geometry::SEGMENT, order);
      mfem::Vector interior_shape(scalar), trace_shape(face_scalar);
      mfem::Vector oriented_trace_shape(face_scalar);
      for (int point = 0; point < face_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
        face_transformation->SetAllIntPoints(&ip);
        const mfem::IntegrationPoint& element_ip = first ?
            face_transformation->GetElement1IntPoint() :
            face_transformation->GetElement2IntPoint();
        fe->CalcShape(element_ip, interior_shape);
        trace_fe->CalcShape(ip, trace_shape);
        const int orientation = orientations[local_face];
        if (orientation != 1 && orientation != -1) { return false; }
        for (int dof = 0; dof < face_scalar; ++dof)
        {
          // Independent segment endpoint reversal, deliberately not MFEM's
          // DofOrderForOrientation implementation.
          const int canonical = orientation > 0 ? dof :
              face_scalar - 1 - dof;
          oriented_trace_shape[dof] = trace_shape[canonical];
        }
        mfem::Vector scaled(2);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled);
        const double measure = face_transformation->Face->Weight();
        const double norm = scaled.Norml2();
        if (std::abs(norm - measure) >
            256.0 * std::numeric_limits<double>::epsilon())
        {
          return false;
        }
        const double side = first ? 1.0 : -1.0;
        const double normal[2] = {side * scaled[0] / norm,
                                  side * scaled[1] / norm};
        double rho_hat = 0.0;
        if (!Interpolate(*oracle.rho_hat, trace_dofs, trace_shape, rho_hat) ||
            !(rho_hat > 0.0))
        {
          return false;
        }
        double tau_density = rho_hat;
        if (stabilization_mutation == 3 &&
            (!Interpolate(*oracle.rho, volume_dofs, interior_shape,
                          tau_density) || !(tau_density > 0.0)))
        {
          return false;
        }
        const double tau = stabilization_mutation == 2 ? 16.0 / height :
            4.0 / (tau_density * height);
        const double weight = ip.weight * measure;
        // Mutation 4 is the exact rejected v2 flux sign.  It exists only in
        // this independent oracle so the v3 sign gate cannot pass by merely
        // reproducing production assembly.
        const double penalty_sign = stabilization_mutation == 4 ? -1.0 : 1.0;
        for (int row = 0; row < scalar; ++row)
        {
          for (int column = 0; column < scalar; ++column)
          {
            oracle.a[ao + index(2, row) * 12 + index(0, column)] +=
                penalty_sign * 0.1 * weight * interior_shape[row] * tau *
                interior_shape[column];
            oracle.a[ao + index(2, row) * 12 + index(1, column)] -=
                0.1 * weight * interior_shape[row] * normal[0] *
                interior_shape[column];
            oracle.a[ao + index(2, row) * 12 + index(2, column)] -=
                0.1 * weight * interior_shape[row] * normal[1] *
                interior_shape[column];
          }
          for (int column = 0; column < face_scalar; ++column)
          {
            const int trace_column = trace_offset + column;
            oracle.b[bo + index(0, row) * 8 + trace_column] -=
                weight * interior_shape[row] * normal[0] *
                oriented_trace_shape[column];
            oracle.b[bo + index(1, row) * 8 + trace_column] -=
                weight * interior_shape[row] * normal[1] *
                oriented_trace_shape[column];
            oracle.b[bo + index(2, row) * 8 + trace_column] -=
                penalty_sign * 0.1 * weight * interior_shape[row] * tau *
                oriented_trace_shape[column];
          }
        }
        for (int row = 0; row < face_scalar; ++row)
        {
          const int trace_row = trace_offset + row;
          for (int column = 0; column < scalar; ++column)
          {
            oracle.c[co + trace_row * 12 + index(0, column)] -=
                penalty_sign * 0.1 * weight * oriented_trace_shape[row] * tau *
                interior_shape[column];
            oracle.c[co + trace_row * 12 + index(1, column)] +=
                0.1 * weight * oriented_trace_shape[row] * normal[0] *
                interior_shape[column];
            oracle.c[co + trace_row * 12 + index(2, column)] +=
                0.1 * weight * oriented_trace_shape[row] * normal[1] *
                interior_shape[column];
          }
          for (int column = 0; column < face_scalar; ++column)
          {
            oracle.d[doff + trace_row * 8 + trace_offset + column] +=
                penalty_sign * 0.1 * weight * oriented_trace_shape[row] * tau *
                oriented_trace_shape[column];
          }
        }
      }
      trace_offset += face_scalar;
    }
  }
  return true;
}
bool CondenseIndependentOracle(IndependentOracle& oracle)
{
  oracle.schur.assign(4 * 8 * 8, 0.0);
  oracle.rhs.assign(4 * 8, 0.0);
  oracle.recovered.assign(4 * 12, 0.0);
  std::vector<DenseFactor> factors(4);
  for (int element = 0; element < 4; ++element)
  {
    const std::size_t ao = static_cast<std::size_t>(element) * 12 * 12;
    const std::size_t bo = static_cast<std::size_t>(element) * 12 * 8;
    const std::size_t co = static_cast<std::size_t>(element) * 8 * 12;
    const std::size_t doff = static_cast<std::size_t>(element) * 8 * 8;
    const std::size_t fo = static_cast<std::size_t>(element) * 12;
    const std::size_t go = static_cast<std::size_t>(element) * 8;
    std::vector<double> local_a(oracle.a.begin() + ao,
                                oracle.a.begin() + ao + 12 * 12);
    if (!FactorDense(local_a, 12, factors[static_cast<std::size_t>(element)]))
    {
      return false;
    }
    std::vector<double> inverse_b(oracle.b.begin() + bo,
                                  oracle.b.begin() + bo + 12 * 8);
    std::vector<double> inverse_f(oracle.f.begin() + fo,
                                  oracle.f.begin() + fo + 12);
    if (!SolveDense(factors[static_cast<std::size_t>(element)], inverse_b, 8) ||
        !SolveDense(factors[static_cast<std::size_t>(element)], inverse_f, 1))
    {
      return false;
    }
    for (int row = 0; row < 8; ++row)
    {
      double value = oracle.g[go + row];
      for (int k = 0; k < 12; ++k)
      {
        value -= oracle.c[co + row * 12 + k] * inverse_f[k];
      }
      oracle.rhs[go + row] = value;
      for (int column = 0; column < 8; ++column)
      {
        double entry = oracle.d[doff + row * 8 + column];
        for (int k = 0; k < 12; ++k)
        {
          entry -= oracle.c[co + row * 12 + k] *
              inverse_b[k * 8 + column];
        }
        oracle.schur[doff + row * 8 + column] = entry;
      }
    }
  }
  oracle.global_schur.assign(16 * 16, 0.0);
  oracle.global_rhs.assign(16, 0.0);
  for (int element = 0; element < 4; ++element)
  {
    const std::size_t so = static_cast<std::size_t>(element) * 8 * 8;
    const std::size_t ro = static_cast<std::size_t>(element) * 8;
    const std::size_t po = static_cast<std::size_t>(element) * 8 * 16;
    for (int true_row = 0; true_row < 16; ++true_row)
      for (int local_row = 0; local_row < 8; ++local_row)
      {
        const double scatter = oracle.restriction[
            po + local_row * 16 + true_row];
        oracle.global_rhs[static_cast<std::size_t>(true_row)] +=
            scatter * oracle.rhs[ro + local_row];
        for (int local_column = 0; local_column < 8; ++local_column)
          for (int true_column = 0; true_column < 16; ++true_column)
          {
            oracle.global_schur[
                static_cast<std::size_t>(true_row) * 16 + true_column] +=
                scatter * oracle.schur[
                    so + local_row * 8 + local_column] *
                oracle.restriction[
                    po + local_column * 16 + true_column];
          }
      }
  }
  DenseFactor condensed_factor;
  if (!FactorDense(oracle.global_schur, 16, condensed_factor)) { return false; }
  oracle.condensed_solution = oracle.global_rhs;
  if (!SolveDense(condensed_factor, oracle.condensed_solution, 1))
  {
    return false;
  }
  oracle.full_matrix.assign(64 * 64, 0.0);
  oracle.full_rhs.assign(64, 0.0);
  for (int element = 0; element < 4; ++element)
  {
    const int interior_begin = element * 12;
    const std::size_t ao = static_cast<std::size_t>(element) * 12 * 12;
    const std::size_t bo = static_cast<std::size_t>(element) * 12 * 8;
    const std::size_t co = static_cast<std::size_t>(element) * 8 * 12;
    const std::size_t doff = static_cast<std::size_t>(element) * 8 * 8;
    const std::size_t po = static_cast<std::size_t>(element) * 8 * 16;
    for (int row = 0; row < 12; ++row)
    {
      oracle.full_rhs[interior_begin + row] =
          oracle.f[static_cast<std::size_t>(element) * 12 + row];
      for (int column = 0; column < 12; ++column)
      {
        oracle.full_matrix[static_cast<std::size_t>(interior_begin + row) * 64 +
                           interior_begin + column] =
            oracle.a[ao + row * 12 + column];
      }
      for (int local_trace = 0; local_trace < 8; ++local_trace)
        for (int true_column = 0; true_column < 16; ++true_column)
        {
          oracle.full_matrix[static_cast<std::size_t>(interior_begin + row) * 64 +
                             48 + true_column] +=
              oracle.b[bo + row * 8 + local_trace] *
              oracle.restriction[po + local_trace * 16 + true_column];
        }
    }
    for (int true_row = 0; true_row < 16; ++true_row)
    {
      for (int local_row = 0; local_row < 8; ++local_row)
      {
        const double scatter = oracle.restriction[
            po + local_row * 16 + true_row];
        if (scatter == 0.0) { continue; }
        for (int column = 0; column < 12; ++column)
        {
          oracle.full_matrix[static_cast<std::size_t>(48 + true_row) * 64 +
                             interior_begin + column] +=
              scatter * oracle.c[co + local_row * 12 + column];
        }
        for (int local_column = 0; local_column < 8; ++local_column)
          for (int true_column = 0; true_column < 16; ++true_column)
          {
            oracle.full_matrix[static_cast<std::size_t>(48 + true_row) * 64 +
                               48 + true_column] +=
                scatter * oracle.d[doff + local_row * 8 + local_column] *
                oracle.restriction[po + local_column * 16 + true_column];
          }
      }
    }
  }
  DenseFactor full_factor;
  if (!FactorDense(oracle.full_matrix, 64, full_factor)) { return false; }
  oracle.full_solution = oracle.full_rhs;
  if (!SolveDense(full_factor, oracle.full_solution, 1)) { return false; }
  oracle.full_condensed_relative_difference = RelativeDifference(
      oracle.condensed_solution,
      std::vector<double>(oracle.full_solution.begin() + 48,
                          oracle.full_solution.end()));
  if (oracle.full_condensed_relative_difference >
      comparison_tolerance)
  {
    return false;
  }
  for (int element = 0; element < 4; ++element)
  {
    std::vector<double> local_rhs(12);
    const std::size_t bo = static_cast<std::size_t>(element) * 12 * 8;
    const std::size_t po = static_cast<std::size_t>(element) * 8 * 16;
    for (int row = 0; row < 12; ++row)
    {
      local_rhs[row] = oracle.f[static_cast<std::size_t>(element) * 12 + row];
      for (int local_trace = 0; local_trace < 8; ++local_trace)
        for (int true_column = 0; true_column < 16; ++true_column)
      {
        local_rhs[row] -= oracle.b[bo + row * 8 + local_trace] *
            oracle.restriction[po + local_trace * 16 + true_column] *
            oracle.full_solution[48 + true_column];
      }
    }
    if (!SolveDense(factors[static_cast<std::size_t>(element)], local_rhs, 1))
    {
      return false;
    }
    std::copy(local_rhs.begin(), local_rhs.end(),
              oracle.recovered.begin() + element * 12);
  }
  return true;
}
bool BuildIndependentOracle(IndependentOracle& oracle,
                            const int stabilization_mutation = 0)
{
  oracle.mesh = BuildOracleMesh();
  oracle.volume_collection = std::make_unique<mfem::L2_FECollection>(
      1, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
  oracle.trace_collection =
      std::make_unique<mfem::DG_Interface_FECollection>(
          1, 2, mfem::FiniteElement::VALUE,
          mfem::BasisType::GaussLegendre);
  oracle.volume_space = std::make_unique<mfem::ParFiniteElementSpace>(
      oracle.mesh.get(), oracle.volume_collection.get(), 1,
      mfem::Ordering::byNODES);
  oracle.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
      oracle.mesh.get(), oracle.trace_collection.get(), 1,
      mfem::Ordering::byNODES);
  oracle.rho = std::make_unique<mfem::ParGridFunction>(oracle.volume_space.get());
  oracle.source =
      std::make_unique<mfem::ParGridFunction>(oracle.volume_space.get());
  oracle.rho_hat =
      std::make_unique<mfem::ParGridFunction>(oracle.trace_space.get());
  const FieldFunction rho = [](const double x, const double y)
  { return 1.0 + 0.08 * std::sin(2.0 * pi * x) +
      0.06 * std::cos(2.0 * pi * y); };
  const FieldFunction rho_hat = [](const double x, const double y)
  { return 1.10 + 0.04 * std::cos(2.0 * pi * x) -
      0.03 * std::sin(2.0 * pi * y); };
  const FieldFunction source = [](const double x, const double y)
  { return 0.25 + 0.07 * std::cos(2.0 * pi * x) *
      std::sin(2.0 * pi * y); };
  return ProjectVolume(*oracle.volume_space, *oracle.rho, rho) &&
      ProjectVolume(*oracle.volume_space, *oracle.source, source) &&
      ProjectTrace(*oracle.trace_space, *oracle.rho_hat, rho_hat) &&
      BuildIndependentRestriction(oracle) &&
      AssembleIndependentOracle(oracle, stabilization_mutation) &&
      CondenseIndependentOracle(oracle);
}
bool CheckProductionRestriction(const LocalBlockSnapshots& production,
                                const IndependentOracle& oracle)
{
  if (production.restriction.size() != 4U * 8U) { return false; }
  for (std::size_t index = 0; index < production.restriction.size(); ++index)
  {
    const auto& entry = production.restriction[index];
    const FrozenRestrictionIncidence& expected =
        frozen_restriction_incidence[index];
    if (entry.element < 0 || entry.element >= 4 || entry.row < 0 ||
        entry.row >= 8 || entry.global_true_dof < 0 ||
        entry.global_true_dof >= 16 || entry.element != expected.element ||
        entry.row != expected.row ||
        entry.signed_local_vdof != expected.geometric_dof ||
        entry.decoded_local_vdof != expected.geometric_dof ||
        entry.sign != expected.sign ||
        entry.global_true_dof != expected.torus_dof ||
        entry.local_true_dof != expected.torus_dof || entry.owner_rank != 0)
    {
      return false;
    }
    const std::size_t offset =
        (static_cast<std::size_t>(entry.element) * 8 + entry.row) * 16;
    int nonzeros = 0;
    for (int column = 0; column < 16; ++column)
    {
      const double value = oracle.restriction[offset + column];
      if (value != 0.0)
      {
        ++nonzeros;
        if (column != entry.global_true_dof || value != entry.sign)
        {
          return false;
        }
      }
    }
    if (nonzeros != 1) { return false; }
  }
  return true;
}
struct IsolatedComparisonDiagnostics
{
  const char* stage = "not-started";
  int rho_projected = 0;
  int rho_hat_projected = 0;
  int source_projected = 0;
  int higher_order_bypass_rejected = 0;
  int solve_called = 0;
  int oracle_ready = 0;
  int published_operator_ready = 0;
  int failure_atomic = 0;
  int restriction_exact = 0;
  double a = std::numeric_limits<double>::quiet_NaN();
  double b = std::numeric_limits<double>::quiet_NaN();
  double c = std::numeric_limits<double>::quiet_NaN();
  double d = std::numeric_limits<double>::quiet_NaN();
  double f = std::numeric_limits<double>::quiet_NaN();
  double g = std::numeric_limits<double>::quiet_NaN();
  double schur = std::numeric_limits<double>::quiet_NaN();
  double condensed_rhs = std::numeric_limits<double>::quiet_NaN();
  double full_residual = std::numeric_limits<double>::quiet_NaN();
  double owned_trace = std::numeric_limits<double>::quiet_NaN();
  double operator_action = std::numeric_limits<double>::quiet_NaN();
  double operator_rhs = std::numeric_limits<double>::quiet_NaN();
  double element_trace = std::numeric_limits<double>::quiet_NaN();
  double recovery = std::numeric_limits<double>::quiet_NaN();
};
void PrintAttemptDiagnostic(const char* const label,
                            const AttemptDiagnostic& diagnostic)
{
  std::cerr << "AUX_ISO_ATTEMPT label=" << label
            << " status_code=" << static_cast<int>(diagnostic.status.code)
            << " status_phase=" << static_cast<int>(diagnostic.status.phase)
            << " origin_rank=" << diagnostic.status.origin_rank
            << " attempt_generation=" << diagnostic.attempt_generation
            << " input_generation=" << diagnostic.input_generation << '\n';
  for (std::size_t index = 0; index < diagnostic.phases.size(); ++index)
  {
    const PhaseDiagnostic& phase = diagnostic.phases[index];
    std::cerr << "AUX_ISO_PHASE label=" << label
              << " index=" << index
              << " phase=" << static_cast<int>(phase.phase)
              << " disposition=" << static_cast<int>(phase.disposition)
              << " selected=" << phase.selected
              << " entered=" << phase.entered
              << " completed=" << phase.completed
              << " passed=" << phase.passed << '\n';
  }
  const auto& solver = diagnostic.solver;
  std::ostringstream message;
  message.precision(std::numeric_limits<double>::max_digits10);
  message << "AUX_ISO_GMRES label=" << label
          << " config_query=" << solver.gmres_configuration_query_status
          << " kdim_query_return="
          << solver.gmres_krylov_dimension_query_return
          << " relative_tolerance_query_return="
          << solver.gmres_relative_tolerance_query_return
          << " absolute_tolerance_query_return="
          << solver.gmres_absolute_tolerance_query_return
          << " max_iterations_query_return="
          << solver.gmres_maximum_iterations_query_return
          << " setup=" << solver.gmres_setup_status
          << " solve=" << solver.gmres_solve_status
          << " convergence_query="
          << solver.gmres_convergence_query_status
          << " converged=" << solver.gmres_converged
          << " iterations=" << solver.gmres_iterations
          << " kdim=" << solver.effective_gmres_krylov_dimension
          << " max_iterations=" << solver.effective_gmres_max_iterations
          << " relative_tolerance_bits="
          << solver.effective_gmres_relative_tolerance_bits
          << " absolute_tolerance_bits="
          << solver.effective_gmres_absolute_tolerance_bits
          << " reported_residual=" << solver.gmres_reported_residual
          << " true_residual=" << solver.true_residual_norm
          << " rhs_norm=" << solver.true_right_hand_side_norm
          << " recovery_residual="
          << solver.maximum_local_recovery_residual
          << " quadrature_repeat="
          << solver.quadrature_repeat_relative_change
          << " ilu_type=" << solver.effective_ilu_type
          << " ilu_fill=" << solver.effective_ilu_level_of_fill
          << " ilu_reordering=" << solver.effective_ilu_local_reordering
          << " exact_gmres=" << solver.exact_gmres_configuration
          << " exact_ilu=" << solver.exact_ilu_configuration << '\n';
  std::cerr << message.str();
}
void PrintFrozenRestrictionDiagnostic(const LocalBlockSnapshots* const blocks)
{
  std::array<int, 24> geometric_counts{};
  std::array<int, 16> torus_counts{};
  std::array<int, 24> actual_geometric_counts{};
  std::array<int, 16> actual_torus_counts{};
  int invalid_actual_indices = 0;
  int mismatches = 0;
  for (std::size_t index = 0;
       index < frozen_restriction_incidence.size(); ++index)
  {
    const FrozenRestrictionIncidence& expected =
        frozen_restriction_incidence[index];
    ++geometric_counts[static_cast<std::size_t>(expected.geometric_dof)];
    ++torus_counts[static_cast<std::size_t>(expected.torus_dof)];
    const auto* const actual =
        blocks != nullptr && index < blocks->restriction.size() ?
        &blocks->restriction[index] : nullptr;
    const bool match = actual != nullptr &&
        actual->element == expected.element && actual->row == expected.row &&
        actual->signed_local_vdof == expected.geometric_dof &&
        actual->decoded_local_vdof == expected.geometric_dof &&
        actual->sign == expected.sign &&
        actual->global_true_dof == expected.torus_dof &&
        actual->local_true_dof == expected.torus_dof &&
        actual->owner_rank == 0;
    if (!match) { ++mismatches; }
    if (actual != nullptr)
    {
      if (actual->decoded_local_vdof >= 0 &&
          actual->decoded_local_vdof < 24)
      {
        ++actual_geometric_counts[static_cast<std::size_t>(
            actual->decoded_local_vdof)];
      }
      else { ++invalid_actual_indices; }
      if (actual->global_true_dof >= 0 && actual->global_true_dof < 16)
      {
        ++actual_torus_counts[static_cast<std::size_t>(
            actual->global_true_dof)];
      }
      else { ++invalid_actual_indices; }
    }
    std::cerr << "AUX_ISO_PE incidence=" << index
              << " expected_element=" << expected.element
              << " expected_row=" << expected.row
              << " expected_element_orientation="
              << expected.element_orientation
              << " expected_face=" << expected.face
              << " expected_face_orientation="
              << expected.face_orientation
              << " expected_geometric_dof=" << expected.geometric_dof
              << " expected_column=" << expected.torus_dof
              << " expected_sign=" << expected.sign
              << " actual_element=" << (actual ? actual->element : -1)
              << " actual_row=" << (actual ? actual->row : -1)
              << " actual_signed_dof="
              << (actual ? actual->signed_local_vdof : -1)
              << " actual_geometric_dof="
              << (actual ? actual->decoded_local_vdof : -1)
              << " actual_column="
              << (actual ? actual->global_true_dof : -1)
              << " actual_local_column="
              << (actual ? actual->local_true_dof : -1)
              << " actual_owner="
              << (actual ? actual->owner_rank : -1)
              << " actual_sign=" << (actual ? actual->sign : 0)
              << " match=" << match << '\n';
  }
  std::cerr << "AUX_ISO_PE_CENSUS expected_incidences=32"
            << " expected_geometric_dofs=24"
            << " expected_boundary_geometric_multiplicity_one="
            << std::count(geometric_counts.begin(), geometric_counts.end(), 1)
            << " expected_interior_geometric_multiplicity_two="
            << std::count(geometric_counts.begin(), geometric_counts.end(), 2)
            << " expected_torus_dofs=16"
            << " expected_torus_multiplicity_two="
            << std::count(torus_counts.begin(), torus_counts.end(), 2)
            << " actual_entries="
            << (blocks ? static_cast<long long>(blocks->restriction.size()) : -1)
            << " actual_geometric_dofs="
            << std::count_if(actual_geometric_counts.begin(),
                             actual_geometric_counts.end(),
                             [](const int count) { return count > 0; })
            << " actual_boundary_geometric_multiplicity_one="
            << std::count(actual_geometric_counts.begin(),
                          actual_geometric_counts.end(), 1)
            << " actual_interior_geometric_multiplicity_two="
            << std::count(actual_geometric_counts.begin(),
                          actual_geometric_counts.end(), 2)
            << " actual_torus_dofs="
            << std::count_if(actual_torus_counts.begin(),
                             actual_torus_counts.end(),
                             [](const int count) { return count > 0; })
            << " actual_torus_multiplicity_two="
            << std::count(actual_torus_counts.begin(),
                          actual_torus_counts.end(), 2)
            << " invalid_actual_indices=" << invalid_actual_indices
            << " mismatches=" << mismatches << '\n';
}
void PrintIsolatedDiagnostic(const SolveBundle& production,
                             const IndependentOracle& oracle,
                             const IsolatedComparisonDiagnostics& comparison)
{
  const AuxiliarySolution* const solution =
      production.solver ? production.solver->solution() : nullptr;
  const LocalBlockSnapshots* const blocks =
      solution ? &solution->blocks : nullptr;
  const mfem::ParMesh* const mesh =
      production.solver ? production.solver->mesh() : nullptr;
  const mfem::ParFiniteElementSpace* const trace_space =
      production.solver ? production.solver->trace_space() : nullptr;
  std::cerr << "AUX_ISO_DIAGNOSTIC stage=" << comparison.stage
            << " create_code=" << static_cast<int>(production.create_status.code)
            << " solve_called=" << comparison.solve_called
            << " solve_code=" << static_cast<int>(production.solve_status.code)
            << " rho_projected=" << comparison.rho_projected
            << " rho_hat_projected=" << comparison.rho_hat_projected
            << " source_projected=" << comparison.source_projected
            << " higher_order_bypass_rejected="
            << comparison.higher_order_bypass_rejected
            << " solution_available=" << (solution != nullptr)
            << " oracle_ready=" << comparison.oracle_ready
            << " published_operator_ready="
            << comparison.published_operator_ready << '\n';
  PrintAttemptDiagnostic("create", production.create_diagnostic);
  if (production.solver)
  {
    PrintAttemptDiagnostic("solve", production.solver->last_attempt());
  }
  std::cerr << "AUX_ISO_TOPOLOGY expected_elements=4 expected_faces=12"
            << " expected_local_trace=24 expected_owned_true=16"
            << " expected_global_true=16 expected_owned_entries=16"
            << " expected_ghost_entries=0 actual_elements="
            << (mesh ? mesh->GetNE() : -1)
            << " actual_faces=" << (mesh ? mesh->GetNumFaces() : -1)
            << " actual_local_trace="
            << (trace_space ? trace_space->GetVSize() : -1)
            << " actual_census_local_trace="
            << (solution ? solution->census.local_trace_dofs : -1)
            << " actual_owned_true="
            << (solution ? solution->census.owned_true_trace_dofs : -1)
            << " actual_global_true="
            << (solution ? solution->census.global_true_trace_dofs : -1)
            << " actual_owned_entries="
            << (solution ?
                static_cast<long long>(solution->census.owned_trace_entries) : -1)
            << " actual_ghost_entries="
            << (solution ?
                static_cast<long long>(solution->census.ghost_trace_entries) : -1)
            << " actual_negative_rows="
            << (solution ? solution->census.negative_restriction_rows : -1)
            << '\n';
  PrintFrozenRestrictionDiagnostic(blocks);
  std::cerr << "AUX_ISO_SHAPES expected_interior=12 expected_trace=8"
            << " expected_block_elements=4 expected_owned_solution=16"
            << " expected_local_solution=24 actual_interior="
            << (blocks ? blocks->interior_unknowns : -1)
            << " actual_trace=" << (blocks ? blocks->trace_unknowns : -1)
            << " actual_block_elements="
            << (blocks ? blocks->local_elements : -1)
            << " actual_owned_solution="
            << (solution ?
                static_cast<long long>(solution->owned_true_sigma_hat.size()) : -1)
            << " actual_local_solution="
            << (solution ?
                static_cast<long long>(solution->local_sigma_hat.size()) : -1)
            << '\n';
  std::cerr << "AUX_ISO_ORACLE expected_a=576 expected_b=384"
            << " expected_c=384 expected_d=256 expected_f=48 expected_g=32"
            << " expected_restriction=512 expected_full_matrix=4096"
            << " expected_full_rhs=64 expected_full_solution=64"
            << " expected_global_schur=256 expected_global_rhs=16"
            << " expected_condensed_solution=16 oracle_mesh="
            << (oracle.mesh != nullptr)
            << " oracle_trace_space=" << (oracle.trace_space != nullptr)
            << " actual_oracle_faces="
            << (oracle.mesh ? oracle.mesh->GetNumFaces() : -1)
            << " actual_oracle_local_trace="
            << (oracle.trace_space ? oracle.trace_space->GetVSize() : -1)
            << " actual_a=" << oracle.a.size()
            << " actual_b=" << oracle.b.size()
            << " actual_c=" << oracle.c.size()
            << " actual_d=" << oracle.d.size()
            << " actual_f=" << oracle.f.size()
            << " actual_g=" << oracle.g.size()
            << " actual_restriction=" << oracle.restriction.size()
            << " actual_full_matrix=" << oracle.full_matrix.size()
            << " actual_full_rhs=" << oracle.full_rhs.size()
            << " actual_full_solution=" << oracle.full_solution.size()
            << " actual_global_schur=" << oracle.global_schur.size()
            << " actual_global_rhs=" << oracle.global_rhs.size()
            << " actual_condensed_solution="
            << oracle.condensed_solution.size() << '\n';
  std::cerr << "AUX_ISO_COMPARISON tolerance=" << comparison_tolerance
            << " a=" << comparison.a << " b=" << comparison.b
            << " c=" << comparison.c << " d=" << comparison.d
            << " f=" << comparison.f << " g=" << comparison.g
            << " schur=" << comparison.schur
            << " condensed_rhs=" << comparison.condensed_rhs
            << " independent_full_vs_condensed="
            << oracle.full_condensed_relative_difference
            << " independent_full_residual=" << comparison.full_residual
            << " restriction_exact=" << comparison.restriction_exact
            << " owned_trace=" << comparison.owned_trace
            << " operator_action=" << comparison.operator_action
            << " operator_rhs=" << comparison.operator_rhs
            << " failure_atomic=" << comparison.failure_atomic
            << " element_trace=" << comparison.element_trace
            << " recovery=" << comparison.recovery << '\n';
}
std::vector<double> ProductionRecovered(const AuxiliarySolution& solution)
{
  std::vector<double> result(4 * 12, 0.0);
  for (int element = 0; element < 4; ++element)
    for (int dof = 0; dof < 4; ++dof)
    {
      result[element * 12 + dof] = solution.sigma[element * 4 + dof];
      result[element * 12 + 4 + dof] = solution.z[element * 8 + dof];
      result[element * 12 + 8 + dof] = solution.z[element * 8 + 4 + dof];
    }
  return result;
}
std::vector<double> DenseAction(const std::vector<double>& matrix,
                                int size,
                                const std::vector<double>& value);
bool CheckIsolatedFixture(SolveBundle& production,
                          IndependentOracle& oracle)
{
  IsolatedComparisonDiagnostics comparison;
  std::unique_ptr<FullyHybridizedScreenedPoissonCpu> solver;
  production.create_status = ScreenedPoissonCpuTestAccess::CreateTwoByTwo(
      MPI_COMM_SELF, 1, ScreenedPoissonPolicy{0.1}, solver,
      production.create_diagnostic);
  production.solver = std::move(solver);
  std::unique_ptr<FullyHybridizedScreenedPoissonCpu> rejected_solver;
  AttemptDiagnostic rejected_diagnostic;
  const Status rejected_status =
      ScreenedPoissonCpuTestAccess::CreateTwoByTwo(
          MPI_COMM_SELF, 2, ScreenedPoissonPolicy{0.1}, rejected_solver,
          rejected_diagnostic);
  comparison.higher_order_bypass_rejected =
      rejected_status.code == StatusCode::InvalidArgument &&
      rejected_solver == nullptr ? 1 : 0;
  comparison.oracle_ready = BuildIndependentOracle(oracle) ? 1 : 0;
  if (production.create_status.code != StatusCode::Ready ||
      !production.solver)
  {
    comparison.stage = "create";
    PrintIsolatedDiagnostic(production, oracle, comparison);
    return false;
  }
  production.rho = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(production.solver->volume_space()));
  production.source = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(production.solver->volume_space()));
  production.rho_hat = std::make_unique<mfem::ParGridFunction>(
      const_cast<mfem::ParFiniteElementSpace*>(production.solver->trace_space()));
  const FieldFunction rho = [](const double x, const double y)
  { return 1.0 + 0.08 * std::sin(2.0 * pi * x) +
      0.06 * std::cos(2.0 * pi * y); };
  const FieldFunction rho_hat = [](const double x, const double y)
  { return 1.10 + 0.04 * std::cos(2.0 * pi * x) -
      0.03 * std::sin(2.0 * pi * y); };
  const FieldFunction source = [](const double x, const double y)
  { return 0.25 + 0.07 * std::cos(2.0 * pi * x) *
      std::sin(2.0 * pi * y); };
  comparison.rho_projected = ProjectVolume(
      *const_cast<mfem::ParFiniteElementSpace*>(
          production.solver->volume_space()), *production.rho, rho) ? 1 : 0;
  comparison.source_projected = ProjectVolume(
      *const_cast<mfem::ParFiniteElementSpace*>(
          production.solver->volume_space()), *production.source, source) ?
      1 : 0;
  comparison.rho_hat_projected = ProjectTrace(
      *const_cast<mfem::ParFiniteElementSpace*>(
          production.solver->trace_space()), *production.rho_hat, rho_hat) ?
      1 : 0;
  if (comparison.rho_projected && comparison.source_projected &&
      comparison.rho_hat_projected)
  {
    comparison.solve_called = 1;
    production.solve_status = production.solver->Solve(
        {production.rho.get(), production.rho_hat.get(),
         production.source.get(), 1});
  }
  if (!comparison.rho_projected || !comparison.source_projected ||
      !comparison.rho_hat_projected || !comparison.solve_called ||
      production.solve_status.code != StatusCode::Success ||
      production.solver->solution() == nullptr || !comparison.oracle_ready)
  {
    comparison.stage = !comparison.rho_projected ||
            !comparison.source_projected || !comparison.rho_hat_projected ?
        "projection" :
        (!comparison.oracle_ready ? "independent-oracle" : "solve");
    PrintIsolatedDiagnostic(production, oracle, comparison);
    return false;
  }
  const AuxiliarySolution& solution = *production.solver->solution();
  const LocalBlockSnapshots& blocks = solution.blocks;
  const std::vector<double>& oracle_true_trace = oracle.condensed_solution;
  std::vector<double> production_element_trace(4 * 8, 0.0);
  std::vector<double> oracle_element_trace(4 * 8, 0.0);
  bool restriction_indices_ready = blocks.restriction.size() == 32;
  for (const auto& entry : blocks.restriction)
  {
    if (entry.element < 0 || entry.element >= 4 || entry.row < 0 ||
        entry.row >= 8 || entry.decoded_local_vdof < 0 ||
        static_cast<std::size_t>(entry.decoded_local_vdof) >=
            solution.local_sigma_hat.size())
    {
      restriction_indices_ready = false;
      continue;
    }
    const std::size_t row = static_cast<std::size_t>(entry.element) * 8 +
        entry.row;
    production_element_trace[row] = static_cast<double>(entry.sign) *
        solution.local_sigma_hat[
            static_cast<std::size_t>(entry.decoded_local_vdof)];
    for (int column = 0; column < 16; ++column)
    {
      oracle_element_trace[row] += oracle.restriction[row * 16 + column] *
          oracle.condensed_solution[static_cast<std::size_t>(column)];
    }
  }
  std::vector<double> full_residual(64, 0.0);
  const bool full_system_ready = oracle.full_matrix.size() == 64U * 64U &&
      oracle.full_solution.size() == 64U && oracle.full_rhs.size() == 64U;
  if (full_system_ready)
  {
    for (int row = 0; row < 64; ++row)
      for (int column = 0; column < 64; ++column)
      {
        full_residual[static_cast<std::size_t>(row)] +=
            oracle.full_matrix[static_cast<std::size_t>(row) * 64 + column] *
            oracle.full_solution[static_cast<std::size_t>(column)];
      }
    for (int row = 0; row < 64; ++row)
    {
      full_residual[static_cast<std::size_t>(row)] -=
          oracle.full_rhs[static_cast<std::size_t>(row)];
    }
  }
  std::vector<double> probe(16, 0.0);
  for (int dof = 0; dof < 16; ++dof)
  {
    probe[static_cast<std::size_t>(dof)] =
        (dof % 2 == 0 ? 1.0 : -1.0) *
        static_cast<double>(dof + 1) / 16.0;
  }
  const std::vector<double> oracle_action = DenseAction(
      oracle.global_schur, 16, probe);
  std::vector<double> production_action, production_rhs;
  const bool operator_applied =
      production.solver->ApplyPublishedTraceOperator(
          probe, production_action, production_rhs);
  comparison.published_operator_ready = operator_applied &&
      production_action.size() == 16 && production_rhs.size() == 16 ? 1 : 0;
  std::vector<double> failure_action{17.0};
  std::vector<double> failure_rhs{19.0};
  comparison.failure_atomic = !production.solver->ApplyPublishedTraceOperator(
      std::vector<double>(15, 0.0), failure_action, failure_rhs) &&
      failure_action == std::vector<double>{17.0} &&
      failure_rhs == std::vector<double>{19.0} ? 1 : 0;
  comparison.a = RelativeDifference(blocks.a, oracle.a);
  comparison.b = RelativeDifference(blocks.b, oracle.b);
  comparison.c = RelativeDifference(blocks.c, oracle.c);
  comparison.d = RelativeDifference(blocks.d, oracle.d);
  comparison.f = RelativeDifference(blocks.f, oracle.f);
  comparison.g = RelativeDifference(blocks.g, oracle.g);
  comparison.schur = RelativeDifference(blocks.schur, oracle.schur);
  comparison.condensed_rhs = RelativeDifference(
      blocks.condensed_right_hand_side, oracle.rhs);
  comparison.restriction_exact =
      CheckProductionRestriction(blocks, oracle) ? 1 : 0;
  comparison.full_residual = full_system_ready ?
      GlobalNorm(MPI_COMM_SELF, full_residual) :
      std::numeric_limits<double>::infinity();
  comparison.owned_trace = RelativeDifference(
      solution.owned_true_sigma_hat, oracle_true_trace);
  comparison.operator_action = comparison.published_operator_ready ?
      RelativeDifference(production_action, oracle_action) :
      std::numeric_limits<double>::infinity();
  comparison.operator_rhs = comparison.published_operator_ready ?
      RelativeDifference(production_rhs, oracle.global_rhs) :
      std::numeric_limits<double>::infinity();
  comparison.element_trace = restriction_indices_ready ?
      RelativeDifference(production_element_trace, oracle_element_trace) :
      std::numeric_limits<double>::infinity();
  const bool recovery_ready = solution.sigma.size() == 16U &&
      solution.z.size() == 32U && oracle.recovered.size() == 48U;
  comparison.recovery = recovery_ready ? RelativeDifference(
      ProductionRecovered(solution), oracle.recovered) :
      std::numeric_limits<double>::infinity();
  const bool passed =
      comparison.higher_order_bypass_rejected &&
      blocks.interior_unknowns == 12 && blocks.trace_unknowns == 8 &&
      blocks.local_elements == 4 && solution.owned_true_sigma_hat.size() == 16 &&
      solution.local_sigma_hat.size() == 24 &&
      production.solver->mesh()->GetNumFaces() == 12 &&
      production.solver->trace_space()->GetVSize() == 24 &&
      solution.census.local_trace_dofs == 24 &&
      solution.census.owned_true_trace_dofs == 16 &&
      solution.census.global_true_trace_dofs == 16 &&
      solution.census.owned_trace_entries == 16 &&
      solution.census.ghost_trace_entries == 0 && oracle.mesh &&
      oracle.mesh->GetNumFaces() == 12 && oracle.trace_space &&
      oracle.trace_space->GetVSize() == 24 &&
      oracle.a.size() == 4U * 12U * 12U &&
      oracle.b.size() == 4U * 12U * 8U &&
      oracle.c.size() == 4U * 8U * 12U &&
      oracle.d.size() == 4U * 8U * 8U && oracle.f.size() == 4U * 12U &&
      oracle.g.size() == 4U * 8U &&
      oracle.restriction.size() == 4U * 8U * 16U &&
      oracle.full_matrix.size() == 64U * 64U &&
      oracle.full_rhs.size() == 64U && oracle.full_solution.size() == 64U &&
      oracle.global_schur.size() == 16U * 16U &&
      oracle.global_rhs.size() == 16U &&
      oracle.condensed_solution.size() == 16U &&
      oracle.full_condensed_relative_difference <= comparison_tolerance &&
      comparison.a <= comparison_tolerance &&
      comparison.b <= comparison_tolerance &&
      comparison.c <= comparison_tolerance &&
      comparison.d <= comparison_tolerance &&
      comparison.f <= comparison_tolerance &&
      comparison.g <= comparison_tolerance &&
      comparison.schur <= comparison_tolerance &&
      comparison.condensed_rhs <= comparison_tolerance &&
      comparison.restriction_exact &&
      comparison.full_residual <= comparison_tolerance &&
      comparison.owned_trace <= comparison_tolerance &&
      comparison.published_operator_ready &&
      comparison.operator_action <= comparison_tolerance &&
      comparison.operator_rhs <= comparison_tolerance &&
      comparison.failure_atomic &&
      comparison.element_trace <= comparison_tolerance &&
      comparison.recovery <= comparison_tolerance;
  if (!passed)
  {
    comparison.stage = "predicate";
    PrintIsolatedDiagnostic(production, oracle, comparison);
  }
  return passed;
}
std::vector<double> DenseAction(const std::vector<double>& matrix,
                                const int size,
                                const std::vector<double>& value)
{
  std::vector<double> result(static_cast<std::size_t>(size), 0.0);
  for (int row = 0; row < size; ++row)
    for (int column = 0; column < size; ++column)
    {
      result[row] += matrix[static_cast<std::size_t>(row) * size + column] *
          value[column];
    }
  return result;
}
bool EvaluateIndependentWeakResidual(const IndependentOracle& oracle,
                                     const std::vector<double>& state,
                                     std::vector<double>& residual)
{
  constexpr int scalar = 4;
  constexpr int face_scalar = 2;
  constexpr int order = 12;
  constexpr double alpha = 0.1;
  if (state.size() != 64 || !oracle.mesh || !oracle.volume_space ||
      !oracle.trace_space || !oracle.rho || !oracle.rho_hat ||
      !oracle.source)
  {
    return false;
  }
  residual.assign(64, 0.0);
  const auto index = [](const int component, const int dof)
  { return component * scalar + dof; };
  for (int element = 0; element < oracle.elements; ++element)
  {
    const mfem::FiniteElement* const fe =
        oracle.volume_space->GetFE(element);
    mfem::ElementTransformation* const transformation =
        oracle.mesh->GetElementTransformation(element);
    mfem::Array<int> volume_dofs;
    oracle.volume_space->GetElementVDofs(element, volume_dofs);
    if (fe == nullptr || transformation == nullptr || fe->GetDof() != scalar)
    {
      return false;
    }
    const int interior_begin = element * 12;
    const std::size_t po = static_cast<std::size_t>(element) * 8 * 16;
    std::vector<double> element_trace(8, 0.0);
    for (int local_row = 0; local_row < 8; ++local_row)
      for (int true_column = 0; true_column < 16; ++true_column)
      {
        element_trace[static_cast<std::size_t>(local_row)] +=
            oracle.restriction[po + local_row * 16 + true_column] *
            state[48 + true_column];
      }
    std::vector<double> local_trace_residual(8, 0.0);
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, order);
    mfem::Vector shape(scalar);
    mfem::DenseMatrix derivative(scalar, 2);
    for (int point = 0; point < volume_rule.GetNPoints(); ++point)
    {
      const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
      transformation->SetIntPoint(&ip);
      fe->CalcShape(ip, shape);
      fe->CalcPhysDShape(*transformation, derivative);
      double rho = 0.0, source = 0.0;
      if (!Interpolate(*oracle.rho, volume_dofs, shape, rho) ||
          !Interpolate(*oracle.source, volume_dofs, shape, source) ||
          !(rho > 0.0))
      {
        return false;
      }
      double sigma = 0.0, zx = 0.0, zy = 0.0;
      for (int dof = 0; dof < scalar; ++dof)
      {
        sigma += shape[dof] * state[interior_begin + index(0, dof)];
        zx += shape[dof] * state[interior_begin + index(1, dof)];
        zy += shape[dof] * state[interior_begin + index(2, dof)];
      }
      const double weight = ip.weight * transformation->Weight();
      for (int row = 0; row < scalar; ++row)
      {
        residual[interior_begin + index(0, row)] += weight *
            (shape[row] * rho * zx + derivative(row, 0) * sigma);
        residual[interior_begin + index(1, row)] += weight *
            (shape[row] * rho * zy + derivative(row, 1) * sigma);
        residual[interior_begin + index(2, row)] += weight *
            (shape[row] * sigma / rho + alpha * derivative(row, 0) * zx +
             alpha * derivative(row, 1) * zy - shape[row] * source);
      }
    }
    const double element_measure = IndependentMeasure(
        *transformation, mfem::Geometry::SQUARE, order);
    mfem::Array<int> faces, orientations;
    oracle.mesh->GetElementEdges(element, faces, orientations);
    if (faces.Size() != 4 || orientations.Size() != faces.Size())
    {
      return false;
    }
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* const face_transformation =
          oracle.mesh->GetFaceElementTransformations(face);
      mfem::ElementTransformation* const face_only =
          oracle.mesh->GetFaceTransformation(face);
      const mfem::FiniteElement* const trace_fe =
          oracle.trace_space->GetFaceElement(face);
      mfem::Array<int> trace_dofs;
      oracle.trace_space->GetFaceVDofs(face, trace_dofs);
      if (face_transformation == nullptr ||
          face_transformation->Face == nullptr || face_only == nullptr ||
          trace_fe == nullptr || trace_fe->GetDof() != face_scalar ||
          trace_dofs.Size() != face_scalar)
      {
        return false;
      }
      const bool first = face_transformation->Elem1No == element;
      if (!first && face_transformation->Elem2No != element) { return false; }
      const double face_measure = IndependentMeasure(
          *face_only, mfem::Geometry::SEGMENT, order);
      const double height = element_measure / face_measure;
      const int orientation = orientations[local_face];
      if (!(element_measure > 0.0) || !(face_measure > 0.0) ||
          !(height > 0.0) || (orientation != 1 && orientation != -1))
      {
        return false;
      }
      const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
          mfem::Geometry::SEGMENT, order);
      mfem::Vector interior_shape(scalar), trace_shape(face_scalar);
      mfem::Vector oriented_trace_shape(face_scalar);
      for (int point = 0; point < face_rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& ip = face_rule.IntPoint(point);
        face_transformation->SetAllIntPoints(&ip);
        const mfem::IntegrationPoint& element_ip = first ?
            face_transformation->GetElement1IntPoint() :
            face_transformation->GetElement2IntPoint();
        fe->CalcShape(element_ip, interior_shape);
        trace_fe->CalcShape(ip, trace_shape);
        for (int dof = 0; dof < face_scalar; ++dof)
        {
          const int canonical = orientation > 0 ? dof :
              face_scalar - 1 - dof;
          oriented_trace_shape[dof] = trace_shape[canonical];
        }
        mfem::Vector scaled(2);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled);
        const double norm = scaled.Norml2();
        const double measure = face_transformation->Face->Weight();
        const double geometry_tolerance = 256.0 *
            std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::max(norm, measure));
        if (!(norm > 0.0) || !(measure > 0.0) ||
            std::abs(norm - measure) > geometry_tolerance)
        {
          return false;
        }
        const double side = first ? 1.0 : -1.0;
        const double normal[2] = {side * scaled[0] / norm,
                                  side * scaled[1] / norm};
        double rho_hat = 0.0;
        if (!Interpolate(*oracle.rho_hat, trace_dofs, trace_shape, rho_hat) ||
            !(rho_hat > 0.0))
        {
          return false;
        }
        double sigma = 0.0, zx = 0.0, zy = 0.0, sigma_hat = 0.0;
        for (int dof = 0; dof < scalar; ++dof)
        {
          sigma += interior_shape[dof] *
              state[interior_begin + index(0, dof)];
          zx += interior_shape[dof] *
              state[interior_begin + index(1, dof)];
          zy += interior_shape[dof] *
              state[interior_begin + index(2, dof)];
        }
        for (int dof = 0; dof < face_scalar; ++dof)
        {
          sigma_hat += oriented_trace_shape[dof] *
              element_trace[static_cast<std::size_t>(trace_offset + dof)];
        }
        const double tau = 4.0 / (rho_hat * height);
        const double z_hat_normal = zx * normal[0] + zy * normal[1] -
            tau * (sigma - sigma_hat);
        const double weight = ip.weight * measure;
        for (int row = 0; row < scalar; ++row)
        {
          residual[interior_begin + index(0, row)] -= weight *
              interior_shape[row] * normal[0] * sigma_hat;
          residual[interior_begin + index(1, row)] -= weight *
              interior_shape[row] * normal[1] * sigma_hat;
          residual[interior_begin + index(2, row)] -= alpha * weight *
              interior_shape[row] * z_hat_normal;
        }
        for (int row = 0; row < face_scalar; ++row)
        {
          local_trace_residual[static_cast<std::size_t>(trace_offset + row)] +=
              alpha * weight * oriented_trace_shape[row] * z_hat_normal;
        }
      }
      trace_offset += face_scalar;
    }
    if (trace_offset != 8) { return false; }
    for (int true_row = 0; true_row < 16; ++true_row)
      for (int local_row = 0; local_row < 8; ++local_row)
      {
        residual[48 + true_row] +=
            oracle.restriction[po + local_row * 16 + true_row] *
            local_trace_residual[static_cast<std::size_t>(local_row)];
      }
  }
  return std::all_of(residual.begin(), residual.end(), IsFinite);
}
bool CheckResidualDirection(const IndependentOracle& weak_oracle,
                            const IndependentOracle& jacobian_oracle,
                            const int kind)
{
  std::vector<double> y(64), direction(64, 0.0);
  for (int index = 0; index < 64; ++index)
  {
    y[index] = 0.01 * std::sin(0.17 * (index + 1));
    const bool selected = kind == 0 || (kind == 1 && index < 48) ||
        (kind == 2 && index >= 48);
    if (selected)
    {
      direction[index] = (index % 2 == 0 ? 1.0 : -1.0) *
          static_cast<double>(index + 1) / 64.0;
    }
  }
  constexpr double epsilon = 0.25;
  std::vector<double> plus(y), minus(y);
  for (int index = 0; index < 64; ++index)
  {
    plus[index] += epsilon * direction[index];
    minus[index] -= epsilon * direction[index];
  }
  std::vector<double> plus_residual, minus_residual;
  if (!EvaluateIndependentWeakResidual(weak_oracle, plus, plus_residual) ||
      !EvaluateIndependentWeakResidual(weak_oracle, minus, minus_residual))
  {
    return false;
  }
  std::vector<double> centered(64, 0.0);
  for (int index = 0; index < 64; ++index)
  {
    centered[index] = (plus_residual[index] - minus_residual[index]) /
        (2.0 * epsilon);
  }
  const std::vector<double> action = DenseAction(
      jacobian_oracle.full_matrix, 64, direction);
  return GlobalNorm(MPI_COMM_SELF, action) >= direction_nonvacuity &&
      RelativeDifference(centered, action) <= comparison_tolerance;
}
bool IndependentResidualRejectsMatrix(const IndependentOracle& weak_oracle,
                                      const IndependentOracle& candidate)
{
  for (int direction = 0; direction < 3; ++direction)
  {
    if (!CheckResidualDirection(weak_oracle, candidate, direction))
    {
      return true;
    }
  }
  return false;
}

double IndependentAuxiliaryEnergy(const IndependentOracle& oracle,
                                  const std::vector<double>& state)
{
  if (oracle.full_matrix.size() != 64U * 64U || state.size() != 64U)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<double> residual(64, 0.0);
  for (int row = 0; row < 64; ++row)
    for (int column = 0; column < 64; ++column)
    {
      residual[static_cast<std::size_t>(row)] +=
          oracle.full_matrix[static_cast<std::size_t>(row) * 64 + column] *
          state[static_cast<std::size_t>(column)];
    }
  double energy = 0.0;
  for (int element = 0; element < 4; ++element)
  {
    const int begin = element * 12;
    for (int dof = 0; dof < 4; ++dof)
    {
      energy += 0.1 * state[static_cast<std::size_t>(begin + 4 + dof)] *
          residual[static_cast<std::size_t>(begin + dof)];
      energy += 0.1 * state[static_cast<std::size_t>(begin + 8 + dof)] *
          residual[static_cast<std::size_t>(begin + 4 + dof)];
      energy += state[static_cast<std::size_t>(begin + dof)] *
          residual[static_cast<std::size_t>(begin + 8 + dof)];
    }
  }
  for (int dof = 0; dof < 16; ++dof)
  {
    energy += state[static_cast<std::size_t>(48 + dof)] *
        residual[static_cast<std::size_t>(48 + dof)];
  }
  return energy;
}

bool CheckV3AuxiliarySignIdentity(const IndependentOracle& v3)
{
  IndependentOracle rejected_v2;
  if (!BuildIndependentOracle(rejected_v2, 4) ||
      rejected_v2.a.size() != v3.a.size() ||
      rejected_v2.b.size() != v3.b.size() ||
      rejected_v2.c.size() != v3.c.size() ||
      rejected_v2.d.size() != v3.d.size() ||
      rejected_v2.restriction != v3.restriction)
  {
    return false;
  }
  const auto delta_bilinear = [](const std::vector<double>& accepted,
                                 const std::vector<double>& rejected,
                                 const std::size_t offset, const int rows,
                                 const int columns,
                                 const std::vector<double>& left,
                                 const std::vector<double>& right)
  {
    double value = 0.0;
    for (int row = 0; row < rows; ++row)
      for (int column = 0; column < columns; ++column)
      {
        value += left[static_cast<std::size_t>(row)] * 0.5 *
            (accepted[offset + static_cast<std::size_t>(row) * columns +
                      column] -
             rejected[offset + static_cast<std::size_t>(row) * columns +
                      column]) *
            right[static_cast<std::size_t>(column)];
      }
    return value;
  };
  double a_energy = 0.0;
  double b_energy = 0.0;
  double c_energy = 0.0;
  double d_energy = 0.0;
  const std::vector<double> true_constant(16, 1.0);
  for (int element = 0; element < 4; ++element)
  {
    std::vector<double> sigma_columns(12, 0.0);
    std::fill(sigma_columns.begin(), sigma_columns.begin() + 4, 1.0);
    std::vector<double> sigma_rows(12, 0.0);
    std::fill(sigma_rows.begin() + 8, sigma_rows.end(), 1.0);
    std::vector<double> local_trace(8, 0.0);
    const std::size_t po = static_cast<std::size_t>(element) * 8 * 16;
    for (int row = 0; row < 8; ++row)
      for (int column = 0; column < 16; ++column)
      {
        local_trace[static_cast<std::size_t>(row)] +=
            v3.restriction[po + static_cast<std::size_t>(row) * 16 +
                           column] *
            true_constant[static_cast<std::size_t>(column)];
      }
    a_energy += delta_bilinear(
        v3.a, rejected_v2.a, static_cast<std::size_t>(element) * 12 * 12,
        12, 12, sigma_rows, sigma_columns);
    b_energy += delta_bilinear(
        v3.b, rejected_v2.b, static_cast<std::size_t>(element) * 12 * 8,
        12, 8, sigma_rows, local_trace);
    c_energy += delta_bilinear(
        v3.c, rejected_v2.c, static_cast<std::size_t>(element) * 8 * 12,
        8, 12, local_trace, sigma_columns);
    d_energy += delta_bilinear(
        v3.d, rejected_v2.d, static_cast<std::size_t>(element) * 8 * 8,
        8, 8, local_trace, local_trace);
  }
  const double block_scale = std::max(
      {1.0, std::abs(a_energy), std::abs(b_energy), std::abs(c_energy),
       std::abs(d_energy)});
  const double block_tolerance = 64.0 *
      std::numeric_limits<double>::epsilon() * block_scale;
  const bool block_signs = a_energy > 0.0 && b_energy < 0.0 &&
      c_energy < 0.0 && d_energy > 0.0 &&
      std::abs(b_energy - c_energy) <= block_tolerance &&
      std::abs(a_energy + b_energy + c_energy + d_energy) <=
          block_tolerance;
  std::vector<double> jump_state(64, 0.0);
  for (int element = 0; element < 4; ++element)
    for (int dof = 0; dof < 4; ++dof)
    {
      jump_state[static_cast<std::size_t>(element * 12 + dof)] = 1.0;
    }
  std::fill(jump_state.begin() + 48, jump_state.end(), -8.0);
  const double accepted_energy = IndependentAuxiliaryEnergy(v3, jump_state);
  const double rejected_energy = IndependentAuxiliaryEnergy(
      rejected_v2, jump_state);
  const double common_energy = 0.5 * (accepted_energy + rejected_energy);
  const double penalty_energy = 0.5 * (accepted_energy - rejected_energy);
  const double identity_scale = std::max(
      {1.0, std::abs(accepted_energy), std::abs(rejected_energy),
       std::abs(common_energy), std::abs(penalty_energy)});
  const double identity_tolerance = 64.0 *
      std::numeric_limits<double>::epsilon() * identity_scale;
  const double solution_difference = RelativeDifference(
      rejected_v2.full_solution, v3.full_solution);
  const bool residual_rejected = IndependentResidualRejectsMatrix(
      v3, rejected_v2);
  const bool passed = block_signs && IsFinite(accepted_energy) &&
      IsFinite(rejected_energy) && common_energy > 0.0 &&
      penalty_energy > common_energy && accepted_energy > 0.0 &&
      rejected_energy < 0.0 &&
      std::abs(accepted_energy - common_energy - penalty_energy) <=
          identity_tolerance &&
      solution_difference > 1.0e-8 && residual_rejected;
  if (!passed)
  {
    std::cerr << "AUX_V3_SIGN block_signs=" << block_signs
              << " a=" << a_energy << " b=" << b_energy
              << " c=" << c_energy << " d=" << d_energy
              << " accepted_energy=" << accepted_energy
              << " rejected_energy=" << rejected_energy
              << " common_energy=" << common_energy
              << " penalty_energy=" << penalty_energy
              << " solution_difference=" << solution_difference
              << " residual_rejected=" << residual_rejected << '\n';
  }
  return passed;
}

bool OracleMutation(const IndependentOracle& baseline, const int kind)
{
  if (kind == 0 || kind == 1 || kind == 4 || kind == 6)
  {
    IndependentOracle mutated;
    mutated.elements = baseline.elements;
    mutated.ni = baseline.ni;
    mutated.nt = baseline.nt;
    mutated.true_dofs = baseline.true_dofs;
    mutated.a = baseline.a;
    mutated.b = baseline.b;
    mutated.c = baseline.c;
    mutated.d = baseline.d;
    mutated.f = baseline.f;
    mutated.g = baseline.g;
    mutated.restriction = baseline.restriction;
    if (kind == 0)
    {
      const auto entry = std::find_if(
          mutated.restriction.begin(), mutated.restriction.end(),
          [](const double value) { return value != 0.0; });
      if (entry == mutated.restriction.end()) { return false; }
      *entry *= -1.0;
    }
    else if (kind == 1) { mutated.b[8 * 8] += 1.0e-3; } // B_0(8,0)
    else if (kind == 4)
    {
      int old_column = -1;
      for (int column = 0; column < 16; ++column)
        if (mutated.restriction[static_cast<std::size_t>(column)] != 0.0)
        {
          old_column = column;
          break;
        }
      if (old_column < 0) { return false; }
      const double sign = mutated.restriction[
          static_cast<std::size_t>(old_column)];
      mutated.restriction[static_cast<std::size_t>(old_column)] = 0.0;
      mutated.restriction[static_cast<std::size_t>(
          (old_column + 1) % 16)] = sign;
    }
    else
    {
      int reversed_element = -1;
      int reversed_face = -1;
      for (int element = 0; element < baseline.elements; ++element)
      {
        mfem::Array<int> faces, orientations;
        baseline.mesh->GetElementEdges(element, faces, orientations);
        for (int local_face = 0; local_face < orientations.Size(); ++local_face)
          if (orientations[local_face] < 0)
          {
            reversed_element = element;
            reversed_face = local_face;
            break;
          }
        if (reversed_element >= 0) { break; }
      }
      if (reversed_element < 0) { return false; }
      const std::size_t first_row =
          (static_cast<std::size_t>(reversed_element) * 8 +
           reversed_face * 2) * 16;
      for (int column = 0; column < 16; ++column)
      {
        std::swap(mutated.restriction[first_row + column],
                  mutated.restriction[first_row + 16 + column]);
      }
    }
    if (!CondenseIndependentOracle(mutated)) { return false; }
    return RelativeDifference(mutated.full_solution,
                              baseline.full_solution) > 1.0e-8 &&
        IndependentResidualRejectsMatrix(baseline, mutated);
  }
  if (kind == 2 || kind == 3)
  {
    IndependentOracle mutated;
    return BuildIndependentOracle(mutated, kind) &&
        RelativeDifference(mutated.full_solution,
                           baseline.full_solution) > 1.0e-8 &&
        IndependentResidualRejectsMatrix(baseline, mutated);
  }
  if (kind == 5) { return H1CorruptedDofRejected(); }
  return kind == 7 && AnalyticConstitutiveZMutationRejected();
}
} // namespace
static_assert(std::is_nothrow_move_constructible<
                  FullyHybridizedScreenedPoissonCpu>::value,
              "owner move construction must be noexcept");
static_assert(!std::is_move_assignable<
                  FullyHybridizedScreenedPoissonCpu>::value,
              "communicator-owning move assignment must remain deleted");
static_assert(!std::is_copy_constructible<
                  FullyHybridizedScreenedPoissonCpu>::value &&
                  !std::is_copy_assignable<
                      FullyHybridizedScreenedPoissonCpu>::value,
              "owner must remain move-only");
int RunScreenedPoissonTests(const int rank, const int ranks,
                            const bool isolated_nonconvergence_evidence)
{
  // Rejected-candidate recovery census: the stable packet plus independent
  // face-permutation, wrong-permutation, asymmetric-precheck, successful
  // post-injection reentry, and driver-attested isolated nonconvergence.
  constexpr int serial_census_rows =
      1 + 1 + 1 + 3 + 6 + 12 + 12 + 12 + 12 + 4 + 4 + 12 + 5 + 7 + 7 +
      1;
  constexpr int rank_two_census_rows =
      serial_census_rows + 4 + 4 + 14 + 14 + 4 + 1 + 1;
  static_assert(serial_census_rows == 100, "serial census freeze changed");
  static_assert(rank_two_census_rows == 142, "rank-two census freeze changed");
  TestCensus census(MPI_COMM_WORLD);
  census.Require("projection_v3_identity", 1);
  census.Require("isolated_full_condensed", 1);
  census.Require("auxiliary_v3_sign", 1);
  census.Require("residual_jacobian_directions", 3);
  census.Require("analytic_rows", 6);
  census.Require("hdg_hierarchy", 12);
  census.Require("h1_oracle_hierarchy", 12);
  census.Require("cross_hierarchy", 12);
  census.Require("analytic_z_hierarchy", 12);
  census.Require("serial_trace_orientation", 4);
  census.Require("orientation_permutation", 4);
  census.Require("negative_mutations", 12);
  census.Require("alpha_invalid", 5);
  census.Require("serial_phase_injection", 7);
  census.Require("serial_successful_reentry", 7);
  census.Require("trace_nonconvergence_reentry", 1);
  census.Require("rank2_trace_count", ranks == 2 ? 4 : 0);
  census.Require("rank2_serial_parity", ranks == 2 ? 4 : 0);
  census.Require("rank2_phase_injection", ranks == 2 ? 14 : 0);
  census.Require("rank2_successful_reentry", ranks == 2 ? 14 : 0);
  census.Require("create_tuple_mismatch", ranks == 2 ? 4 : 0);
  census.Require("rank2_asymmetric_precheck", ranks == 2 ? 1 : 0);
  census.Require("rank2_projection_v3_census", ranks == 2 ? 1 : 0);
  SolveBundle isolated_production;
  IndependentOracle isolated_oracle;
  bool isolated_ready = false;
  census.Run("projection_v3_identity", "HDG-SC25-V3-IDENTITY-01",
             []() { return CheckProjectionV3Identity(); });
  census.Run("isolated_full_condensed", "AUX-ISO-PRESCRIBED-2X2-K1-01",
             [&]()
  {
    isolated_ready = CheckIsolatedFixture(isolated_production,
                                           isolated_oracle);
    return isolated_ready;
  });
  census.Run("auxiliary_v3_sign", "HDG-SC25-AUX-SIGN-V3-01",
             [&]()
             { return isolated_ready &&
                   CheckV3AuxiliarySignIdentity(isolated_oracle); });
  const char* direction_ids[3] = {
      "AUX-ISO-JAC-DIRECTION-ALL-01",
      "AUX-ISO-JAC-DIRECTION-INTERIOR-01",
      "AUX-ISO-JAC-DIRECTION-TRACE-01"};
  for (int direction = 0; direction < 3; ++direction)
  {
    census.Run("residual_jacobian_directions", direction_ids[direction],
               [&, direction]()
               { return isolated_ready &&
                   CheckResidualDirection(isolated_oracle, isolated_oracle,
                                          direction); });
  }
  census.Run("analytic_rows", "HDG-SC25-ZERO-01",
             [&]() { return CheckZero(MPI_COMM_WORLD, ranks); });
  census.Run("analytic_rows", "HDG-SC25-CONST-01",
             [&]() { return CheckConstant(MPI_COMM_WORLD, ranks, 0.1); });
  const double alpha_values[3] = {1.0e-2, 1.0e-4, 1.0e-6};
  for (int index = 0; index < 3; ++index)
  {
    census.Run("analytic_rows",
               "HDG-SC25-ALPHA0-01/" + std::to_string(index),
               [&, index]()
               { return CheckConstant(MPI_COMM_WORLD, ranks,
                                      alpha_values[index]); });
  }
  census.Run("analytic_rows", "HDG-SC25-MANUF-DIFFERENTIAL-01",
             []() { return CheckManufacturedDifferential(); });
  for (int degree = 1; degree <= 4; ++degree)
  {
    std::array<ComparisonMetrics, 3> metrics;
    std::array<bool, 3> computed{};
    for (int level = 0; level < 3; ++level)
    {
      computed[level] = ComputeComparison(MPI_COMM_WORLD, ranks, degree,
                                          8 << level, metrics[level]);
    }
    const bool hdg_hierarchy = HdgHierarchyAccepted(
        metrics, computed, degree);
    const bool h1_hierarchy = H1HierarchyAccepted(
        metrics, computed, degree);
    const bool cross_hierarchy = CrossHierarchyAccepted(
        metrics, computed, degree);
    const bool analytic_z_hierarchy = AnalyticZHierarchyAccepted(
        metrics, computed);
    for (int level = 0; level < 3; ++level)
    {
      const int n = 8 << level;
      census.Run("hdg_hierarchy",
                 "HDG-SC25-MANUF-01/k" + std::to_string(degree) + "/N" +
                     std::to_string(n),
                 [&, level, hdg_hierarchy]()
                 { return computed[level] && hdg_hierarchy; });
      census.Run("h1_oracle_hierarchy",
                 "HDG-SC25-MANUF-H1-ORACLE-01/k" +
                     std::to_string(degree) + "/N" +
                     std::to_string(n),
                 [&, level, h1_hierarchy]()
      {
        const auto& evidence = metrics[level].h1.evidence;
        return computed[level] && h1_hierarchy &&
            evidence.sparse_symmetry_ratio <= 1.0e-13 &&
            evidence.full_residual_norm <= 1.0e-14 +
                1.0e-12 * evidence.full_right_hand_side_norm &&
            evidence.partial_residual_norm <= 1.0e-14 +
                1.0e-12 * evidence.partial_right_hand_side_norm &&
            evidence.full_probe_rayleigh > 0.0 &&
            evidence.partial_probe_rayleigh > 0.0 &&
            !evidence.deterministic_probe.empty() &&
            evidence.full_probe_action.size() ==
                evidence.deterministic_probe.size() &&
            evidence.partial_probe_action.size() ==
                evidence.deterministic_probe.size() &&
            evidence.full_true_solution_probe.size() ==
                evidence.deterministic_probe.size() &&
            evidence.partial_true_solution_probe.size() ==
                evidence.deterministic_probe.size() &&
            evidence.h1_quadrature_relative_change <= 1.0e-12;
      });
      census.Run("cross_hierarchy",
                 "HDG-SC25-MANUF-CROSS-01/k" +
                     std::to_string(degree) + "/N" + std::to_string(n),
                 [&, level, cross_hierarchy]()
                 { return computed[level] && cross_hierarchy; });
      census.Run("analytic_z_hierarchy",
                 "HDG-SC25-MANUF-Z-ANALYTIC-01/k" +
                     std::to_string(degree) + "/N" + std::to_string(n),
                 [&, level, analytic_z_hierarchy]()
                 { return computed[level] && analytic_z_hierarchy; });
    }
  }
  for (int degree = 1; degree <= 4; ++degree)
  {
    census.Run("serial_trace_orientation",
               "AUX-TRACE-SERIAL-ORIENTATION/k" + std::to_string(degree),
               [degree]()
               { return OrientationAndCount(MPI_COMM_SELF, 1, degree); });
    census.Run("orientation_permutation",
               "AUX-TRACE-OPPOSITE-REVERSAL/k" + std::to_string(degree),
               [degree]() { return OppositeSidePermutation(degree); });
    if (ranks == 2)
    {
      census.Run("rank2_trace_count",
                 "AUX-TRACE-RANK2-COUNT/k" + std::to_string(degree),
                 [degree]()
                 { return OrientationAndCount(MPI_COMM_WORLD, 2, degree); });
      census.Run("rank2_serial_parity",
                 "AUX-TRACE-RANK2-PARITY/k" + std::to_string(degree),
                 [&, degree]()
                 { return TraceParity(MPI_COMM_WORLD, rank, degree); });
    }
  }
  if (ranks == 2)
  {
    census.Run("rank2_projection_v3_census",
               "HDG-SC25-PERIODIC-01/OWNERSHIP-720-560-1280",
               [&, rank]()
               { return ProjectionV3RankTwoCensus(MPI_COMM_WORLD, rank); });
  }
  const char* oracle_mutations[8] = {
      "MUT-SIGNED-P", "MUT-B-TERM", "MUT-OLD-TAU",
      "MUT-RHO-FOR-RHOHAT", "MUT-TRACE-DOF", "MUT-H1-DOF",
      "MUT-WRONG-PERMUTATION", "MUT-ANALYTIC-CONSTITUTIVE-Z"};
  for (int mutation = 0; mutation < 8; ++mutation)
  {
    census.Run("negative_mutations", oracle_mutations[mutation],
               [&, mutation]()
               { return isolated_ready &&
                   OracleMutation(isolated_oracle, mutation); });
  }
  const InvalidSnapshotKind invalid_snapshots[4] = {
      InvalidSnapshotKind::RhoZero, InvalidSnapshotKind::RhoHatNegative,
      InvalidSnapshotKind::SourceNan, InvalidSnapshotKind::ForeignSpace};
  const char* invalid_snapshot_ids[4] = {
      "MUT-RHO-ZERO", "MUT-RHOHAT-NEGATIVE", "MUT-F-NAN",
      "MUT-FOREIGN-SAME-SIZE-FES"};
  for (int mutation = 0; mutation < 4; ++mutation)
  {
    census.Run("negative_mutations", invalid_snapshot_ids[mutation],
               [&, mutation]()
               { return InvalidSnapshot(MPI_COMM_WORLD, ranks,
                                        invalid_snapshots[mutation]); });
  }
  const double invalid_alpha_values[5] = {
      0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  for (int invalid = 0; invalid < 5; ++invalid)
  {
    census.Run("alpha_invalid",
               "CREATE-ALPHA-INVALID/" + std::to_string(invalid),
               [&, invalid]()
               { return InvalidAlpha(MPI_COMM_WORLD, ranks,
                                     invalid_alpha_values[invalid]); });
  }
  std::array<bool, static_cast<std::size_t>(AttemptPhase::Count)>
      serial_reentry{};
  for (int phase = 0; phase < static_cast<int>(AttemptPhase::Count); ++phase)
  {
    census.Run("serial_phase_injection",
               "AUX-INJECT-SERIAL/phase" + std::to_string(phase),
               [&, phase]()
               { return InjectedFailure(MPI_COMM_SELF, 1,
                     static_cast<AttemptPhase>(phase), 0,
                     &serial_reentry[static_cast<std::size_t>(phase)]); });
  }
  for (int phase = 0; phase < static_cast<int>(AttemptPhase::Count); ++phase)
  {
    census.Run("serial_successful_reentry",
               "AUX-REENTRY-SERIAL/phase" + std::to_string(phase),
               [&, phase]()
               { return serial_reentry[static_cast<std::size_t>(phase)]; });
  }
  if (ranks == 2)
  {
    std::array<std::array<bool,
        static_cast<std::size_t>(AttemptPhase::Count)>, 2> rank2_reentry{};
    census.Run("rank2_asymmetric_precheck",
               "AUX-RANK2-ASYMMETRIC-RHO-INVALID-REENTRY",
               [&, rank]()
               { return AsymmetricRank2SnapshotFailure(MPI_COMM_WORLD,
                                                       rank); });
    for (int origin = 0; origin < 2; ++origin)
      for (int phase = 0; phase < static_cast<int>(AttemptPhase::Count);
           ++phase)
      {
        census.Run("rank2_phase_injection",
                   "AUX-INJECT-RANK2/rank" + std::to_string(origin) +
                   "/phase" + std::to_string(phase),
                   [&, origin, phase]()
                   { return InjectedFailure(MPI_COMM_WORLD, 2,
                         static_cast<AttemptPhase>(phase), origin,
                         &rank2_reentry[static_cast<std::size_t>(origin)]
                                       [static_cast<std::size_t>(phase)]); });
      }
    for (int origin = 0; origin < 2; ++origin)
      for (int phase = 0; phase < static_cast<int>(AttemptPhase::Count);
           ++phase)
      {
        census.Run("rank2_successful_reentry",
                   "AUX-REENTRY-RANK2/rank" + std::to_string(origin) +
                       "/phase" + std::to_string(phase),
                   [&, origin, phase]()
                   { return rank2_reentry[static_cast<std::size_t>(origin)]
                                           [static_cast<std::size_t>(phase)]; });
      }
    for (int mismatch = 0; mismatch < 4; ++mismatch)
    {
      census.Run("create_tuple_mismatch",
                 "CREATE-TUPLE-MISMATCH/" + std::to_string(mismatch),
                 [&, mismatch]()
                 { return CreateMismatch(MPI_COMM_WORLD, rank, mismatch); });
    }
  }
  census.Run("trace_nonconvergence_reentry",
             ranks == 1 ? "AUX-GMRES-EXHAUSTED-SERIAL-01" :
                          "AUX-GMRES-EXHAUSTED-RANK2-01",
             [&]() { return isolated_nonconvergence_evidence; });
  return census.Complete() ? 0 : 1;
}
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  mfem::Hypre::Init();
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int result = 2;
  const std::string mode = argc == 2 ? argv[1] : std::string{};
  if ((ranks == 1 || ranks == 2) &&
      mode == "--isolated-negative-nonzero")
  {
    result = ExhaustedTraceSolveIsAtomic(MPI_COMM_WORLD, ranks) ? 0 : 1;
    if (rank == 0)
    {
      std::cout << (result == 0 ? "PASS" : "FAIL")
                << " HDG-AUX-CPU-03 isolated nonzero nonconvergence"
                << " rollback\n";
    }
  }
  else if ((ranks == 1 || ranks == 2) &&
           mode == "--isolated-negative-zero")
  {
    result = ExactZeroTraceNonconvergenceIsAtomic(
        MPI_COMM_WORLD, ranks) ? 0 : 1;
    if (rank == 0)
    {
      std::cout << (result == 0 ? "PASS" : "FAIL")
                << " HDG-AUX-CPU-03 isolated zero-RHS nonconvergence"
                << " rollback\n";
    }
  }
  else if ((ranks == 1 || ranks == 2) &&
           mode == "--positive-with-isolated-negatives")
  {
    result = RunScreenedPoissonTests(rank, ranks, true);
  }
  else if (rank == 0)
  {
    std::cerr << "FAIL HDG-AUX-CPU-03 requires one or two ranks and one"
                 " recognized isolated-test mode\n";
  }
  if (rank == 0 && mode == "--positive-with-isolated-negatives")
  {
    std::cout << (result == 0 ? "PASS" : "FAIL")
              << " HDG-AUX-CPU-03 source/runtime census\n";
  }
  mfem::Hypre::Finalize();
  MPI_Finalize();
  return result;
}
