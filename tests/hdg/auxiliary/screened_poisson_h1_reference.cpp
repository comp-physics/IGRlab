#include "screened_poisson_h1_reference.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
namespace dg::hdg::auxiliary::testing
{
namespace
{
constexpr double h1_relative_solver_tolerance = 1.0e-13;
constexpr double h1_absolute_solver_tolerance = 1.0e-15;
constexpr int h1_maximum_iterations = 5000;
constexpr double h1_truth_absolute_tolerance = 1.0e-14;
constexpr double h1_truth_relative_tolerance = 1.0e-12;
constexpr double h1_symmetry_tolerance = 1.0e-13;
constexpr double h1_quadrature_tolerance = 1.0e-12;
bool IsFinite(const double value) noexcept { return std::isfinite(value); }
double VectorNorm(MPI_Comm communicator, const mfem::Vector& value)
{
  const double square = mfem::InnerProduct(communicator, value, value);
  return square >= 0.0 && IsFinite(square) ? std::sqrt(square) :
      std::numeric_limits<double>::infinity();
}
double RelativeDifference(MPI_Comm communicator, const mfem::Vector& left,
                          const mfem::Vector& right)
{
  if (left.Size() != right.Size())
  {
    return std::numeric_limits<double>::infinity();
  }
  mfem::Vector difference(left);
  difference -= right;
  return VectorNorm(communicator, difference) /
      std::max(1.0, VectorNorm(communicator, right));
}
std::unique_ptr<mfem::ParMesh> BuildPeriodicMesh(const int n)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[0][1] = 0.0;
  translations[1][0] = 0.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  std::vector<int> partition(static_cast<std::size_t>(periodic.GetNE()), 0);
  return std::make_unique<mfem::ParMesh>(MPI_COMM_SELF, periodic,
                                         partition.data());
}
double SparseSymmetryRatio(mfem::HypreParMatrix& matrix)
{
  if (matrix.GetComm() != MPI_COMM_SELF ||
      matrix.GetGlobalNumRows() != matrix.GetGlobalNumCols() ||
      matrix.GetGlobalNumRows() > std::numeric_limits<int>::max())
  {
    return std::numeric_limits<double>::infinity();
  }
  mfem::SparseMatrix merged;
  matrix.MergeDiagAndOffd(merged);
  const int* const rows = merged.GetI();
  const int* const columns = merged.GetJ();
  const double* const data = merged.GetData();
  if (rows == nullptr || columns == nullptr || data == nullptr)
  {
    return std::numeric_limits<double>::infinity();
  }
  // Sparse transpose comparison: merge the stored CSR rows into one ordered
  // coordinate map and compare only the union of stored entries.  No dense
  // matrix or basis-vector sweep is formed.
  using Coordinate = std::pair<int, int>;
  std::map<Coordinate, long double> entries;
  for (int row = 0; row < merged.Height(); ++row)
    for (int entry = rows[row]; entry < rows[row + 1]; ++entry)
    {
      entries[{row, columns[entry]}] += data[entry];
    }
  long double matrix_square = 0.0L;
  long double difference_square = 0.0L;
  std::map<Coordinate, bool> visited;
  for (const auto& item : entries)
  {
    if (visited[item.first]) { continue; }
    const Coordinate transpose{item.first.second, item.first.first};
    const auto found = entries.find(transpose);
    const long double value = item.second;
    const long double opposite = found == entries.end() ? 0.0L :
        found->second;
    matrix_square += value * value;
    if (transpose != item.first) { matrix_square += opposite * opposite; }
    const long double delta = value - opposite;
    difference_square += delta * delta;
    if (transpose != item.first) { difference_square += delta * delta; }
    visited[item.first] = true;
    visited[transpose] = true;
  }
  return std::sqrt(static_cast<double>(difference_square)) /
      std::max(1.0, std::sqrt(static_cast<double>(matrix_square)));
}
struct SolveArtifacts
{
  mfem::Vector true_solution;
  mfem::Vector true_right_hand_side;
  mfem::Vector deterministic_probe;
  mfem::Vector probe_action;
  int iterations = 0;
  double residual_norm = 0.0;
  double rhs_norm = 0.0;
  double rayleigh = 0.0;
  double symmetry = 0.0;
};
bool ValidRequest(const H1ReferenceRequest& request) noexcept
{
  return request.degree >= 1 && request.degree <= 4 &&
      (request.elements_per_direction == 8 ||
       request.elements_per_direction == 16 ||
       request.elements_per_direction == 32 ||
       request.elements_per_direction == 2) &&
      IsFinite(request.alpha) && request.alpha > 0.0 &&
      static_cast<bool>(request.rho) && static_cast<bool>(request.source);
}
bool BuildCoefficients(const H1ReferenceRequest& request,
                       bool& coefficient_valid,
                       std::unique_ptr<mfem::FunctionCoefficient>& inverse,
                       std::unique_ptr<mfem::FunctionCoefficient>& source)
{
  coefficient_valid = true;
  inverse = std::make_unique<mfem::FunctionCoefficient>(
      [&request, &coefficient_valid](const mfem::Vector& point)
      {
        const double rho = request.rho(point);
        if (!IsFinite(rho) || !(rho > 0.0))
        {
          coefficient_valid = false;
          return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / rho;
      });
  source = std::make_unique<mfem::FunctionCoefficient>(
      [&request, &coefficient_valid](const mfem::Vector& point)
      {
        const double value = request.source(point);
        if (!IsFinite(value))
        {
          coefficient_valid = false;
          return std::numeric_limits<double>::quiet_NaN();
        }
        return value;
      });
  return true;
}
void ConfigureCg(mfem::CGSolver& solver)
{
  solver.SetRelTol(h1_relative_solver_tolerance);
  solver.SetAbsTol(h1_absolute_solver_tolerance);
  solver.SetMaxIter(h1_maximum_iterations);
  solver.SetPrintLevel(0);
  solver.iterative_mode = false;
}
void FillProbe(mfem::Vector& probe)
{
  for (int index = 0; index < probe.Size(); ++index)
  {
    probe[index] = std::sin(0.37 * static_cast<double>(index + 1)) +
        0.25 * std::cos(0.19 * static_cast<double>(index + 1));
  }
}
std::vector<double> RetainVector(const mfem::Vector& value)
{
  const double* const data = value.HostRead();
  return data == nullptr ? std::vector<double>{} :
      std::vector<double>(data, data + value.Size());
}
bool FinalizeArtifacts(const mfem::Operator& operation,
                       const mfem::Vector& true_solution,
                       const mfem::Vector& right_hand_side,
                       SolveArtifacts& artifacts)
{
  mfem::Vector residual(right_hand_side.Size());
  operation.Mult(true_solution, residual);
  residual -= right_hand_side;
  artifacts.true_solution = true_solution;
  artifacts.true_right_hand_side = right_hand_side;
  artifacts.residual_norm = VectorNorm(MPI_COMM_SELF, residual);
  artifacts.rhs_norm = VectorNorm(MPI_COMM_SELF, right_hand_side);
  mfem::Vector probe(true_solution.Size());
  FillProbe(probe);
  artifacts.deterministic_probe = probe;
  artifacts.probe_action.SetSize(probe.Size());
  operation.Mult(probe, artifacts.probe_action);
  artifacts.rayleigh = mfem::InnerProduct(MPI_COMM_SELF, probe,
                                           artifacts.probe_action);
  return IsFinite(artifacts.residual_norm) && IsFinite(artifacts.rhs_norm) &&
      IsFinite(artifacts.rayleigh) && artifacts.rayleigh > 0.0 &&
      artifacts.residual_norm <= h1_truth_absolute_tolerance +
          h1_truth_relative_tolerance * artifacts.rhs_norm;
}
bool SolveFull(const H1ReferenceRequest& request,
               mfem::ParFiniteElementSpace& space,
               const int quadrature_order, mfem::ParGridFunction& solution,
               SolveArtifacts& artifacts)
{
  bool coefficient_valid = true;
  std::unique_ptr<mfem::FunctionCoefficient> inverse, source;
  BuildCoefficients(request, coefficient_valid, inverse, source);
  mfem::ConstantCoefficient alpha(request.alpha);
  mfem::ProductCoefficient diffusion(alpha, *inverse);
  mfem::ParBilinearForm form(&space);
  auto* const mass = new mfem::MassIntegrator(*inverse);
  auto* const stiffness = new mfem::DiffusionIntegrator(diffusion);
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mfem::Geometry::SQUARE, quadrature_order);
  mass->SetIntRule(&rule);
  stiffness->SetIntRule(&rule);
  form.AddDomainIntegrator(mass);
  form.AddDomainIntegrator(stiffness);
  mfem::ParLinearForm right_hand_side(&space);
  auto* const source_integrator = new mfem::DomainLFIntegrator(*source);
  source_integrator->SetIntRule(&rule);
  right_hand_side.AddDomainIntegrator(source_integrator);
  form.Assemble();
  form.Finalize();
  right_hand_side.Assemble();
  if (!coefficient_valid) { return false; }
  solution = 0.0;
  mfem::Array<int> essential;
  mfem::OperatorHandle operation;
  mfem::Vector true_solution, true_rhs;
  form.FormLinearSystem(essential, solution, right_hand_side, operation,
                        true_solution, true_rhs);
  mfem::HypreParMatrix* const matrix =
      operation.As<mfem::HypreParMatrix>();
  if (matrix == nullptr) { return false; }
  mfem::HypreBoomerAMG preconditioner(*matrix);
  preconditioner.SetPrintLevel(0);
  mfem::CGSolver solver(MPI_COMM_SELF);
  ConfigureCg(solver);
  solver.SetPreconditioner(preconditioner);
  solver.SetOperator(*matrix);
  true_solution = 0.0;
  solver.Mult(true_rhs, true_solution);
  artifacts.iterations = solver.GetNumIterations();
  artifacts.symmetry = SparseSymmetryRatio(*matrix);
  if (!solver.GetConverged() || artifacts.iterations < 0 ||
      !IsFinite(artifacts.symmetry) ||
      artifacts.symmetry > h1_symmetry_tolerance ||
      !FinalizeArtifacts(*matrix, true_solution, true_rhs, artifacts))
  {
    return false;
  }
  form.RecoverFEMSolution(true_solution, right_hand_side, solution);
  return true;
}
bool SolvePartial(const H1ReferenceRequest& request,
                  mfem::ParFiniteElementSpace& space,
                  const int quadrature_order,
                  mfem::ParGridFunction& solution,
                  SolveArtifacts& artifacts)
{
  bool coefficient_valid = true;
  std::unique_ptr<mfem::FunctionCoefficient> inverse, source;
  BuildCoefficients(request, coefficient_valid, inverse, source);
  mfem::ConstantCoefficient alpha(request.alpha);
  mfem::ProductCoefficient diffusion(alpha, *inverse);
  mfem::ParBilinearForm form(&space);
  form.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
  auto* const mass = new mfem::MassIntegrator(*inverse);
  auto* const stiffness = new mfem::DiffusionIntegrator(diffusion);
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mfem::Geometry::SQUARE, quadrature_order);
  mass->SetIntRule(&rule);
  stiffness->SetIntRule(&rule);
  form.AddDomainIntegrator(mass);
  form.AddDomainIntegrator(stiffness);
  mfem::ParLinearForm right_hand_side(&space);
  auto* const source_integrator = new mfem::DomainLFIntegrator(*source);
  source_integrator->SetIntRule(&rule);
  right_hand_side.AddDomainIntegrator(source_integrator);
  form.Assemble();
  right_hand_side.Assemble();
  if (!coefficient_valid) { return false; }
  solution = 0.0;
  mfem::Array<int> essential;
  mfem::OperatorHandle operation;
  mfem::Vector true_solution, true_rhs;
  form.FormLinearSystem(essential, solution, right_hand_side, operation,
                        true_solution, true_rhs);
  if (operation.Ptr() == nullptr) { return false; }
  mfem::OperatorJacobiSmoother preconditioner(form, essential);
  mfem::CGSolver solver(MPI_COMM_SELF);
  ConfigureCg(solver);
  solver.SetPreconditioner(preconditioner);
  solver.SetOperator(*operation.Ptr());
  true_solution = 0.0;
  solver.Mult(true_rhs, true_solution);
  artifacts.iterations = solver.GetNumIterations();
  if (!solver.GetConverged() || artifacts.iterations < 0 ||
      !FinalizeArtifacts(*operation.Ptr(), true_solution, true_rhs,
                         artifacts))
  {
    return false;
  }
  form.RecoverFEMSolution(true_solution, right_hand_side, solution);
  return true;
}
} // namespace
bool SolveH1Reference(const H1ReferenceRequest& request,
                      H1ReferenceSolution& output) noexcept
{
  if (!ValidRequest(request)) { return false; }
  try
  {
    H1ReferenceSolution candidate;
    candidate.mesh = BuildPeriodicMesh(request.elements_per_direction);
    candidate.collection = std::make_unique<mfem::H1_FECollection>(
        request.degree, 2, mfem::BasisType::GaussLobatto);
    candidate.space = std::make_unique<mfem::ParFiniteElementSpace>(
        candidate.mesh.get(), candidate.collection.get(), 1,
        mfem::Ordering::byNODES);
    candidate.full_solution = std::make_unique<mfem::ParGridFunction>(
        candidate.space.get());
    candidate.partial_solution = std::make_unique<mfem::ParGridFunction>(
        candidate.space.get());
    candidate.repeat_solution = std::make_unique<mfem::ParGridFunction>(
        candidate.space.get());
    SolveArtifacts full, partial, repeat;
    const int quadrature_order = 2 * request.degree + 4;
    if (!SolveFull(request, *candidate.space, quadrature_order,
                   *candidate.full_solution, full) ||
        !SolvePartial(request, *candidate.space, quadrature_order,
                      *candidate.partial_solution, partial) ||
        !SolveFull(request, *candidate.space, quadrature_order + 1,
                   *candidate.repeat_solution, repeat))
    {
      return false;
    }
    candidate.evidence.full_iterations = full.iterations;
    candidate.evidence.partial_iterations = partial.iterations;
    candidate.evidence.full_residual_norm = full.residual_norm;
    candidate.evidence.full_right_hand_side_norm = full.rhs_norm;
    candidate.evidence.partial_residual_norm = partial.residual_norm;
    candidate.evidence.partial_right_hand_side_norm = partial.rhs_norm;
    candidate.evidence.sparse_symmetry_ratio = full.symmetry;
    candidate.evidence.full_probe_rayleigh = full.rayleigh;
    candidate.evidence.partial_probe_rayleigh = partial.rayleigh;
    candidate.evidence.full_partial_action_relative_difference =
        RelativeDifference(MPI_COMM_SELF, full.probe_action,
                           partial.probe_action);
    candidate.evidence.full_partial_solution_relative_difference =
        RelativeDifference(MPI_COMM_SELF, full.true_solution,
                           partial.true_solution);
    candidate.evidence.h1_quadrature_relative_change =
        RelativeDifference(MPI_COMM_SELF, full.true_solution,
                           repeat.true_solution);
    candidate.evidence.deterministic_probe =
        RetainVector(full.deterministic_probe);
    candidate.evidence.full_probe_action = RetainVector(full.probe_action);
    candidate.evidence.partial_probe_action =
        RetainVector(partial.probe_action);
    candidate.evidence.full_true_solution_probe =
        RetainVector(full.true_solution);
    candidate.evidence.partial_true_solution_probe =
        RetainVector(partial.true_solution);
    if (!IsFinite(candidate.evidence.full_partial_action_relative_difference) ||
        !IsFinite(candidate.evidence.full_partial_solution_relative_difference) ||
        !IsFinite(candidate.evidence.h1_quadrature_relative_change) ||
        candidate.evidence.full_partial_action_relative_difference > 1.0e-11 ||
        candidate.evidence.full_partial_solution_relative_difference > 1.0e-11 ||
        candidate.evidence.h1_quadrature_relative_change >
            h1_quadrature_tolerance)
    {
      return false;
    }
    output = std::move(candidate);
    return true;
  }
  catch (...)
  {
    return false;
  }
}
} // namespace dg::hdg::auxiliary::testing
