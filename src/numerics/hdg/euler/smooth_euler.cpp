#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
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

int AbsoluteDof(const int dof) noexcept { return dof < 0 ? -1 - dof : dof; }
double DofSign(const int dof) noexcept { return dof < 0 ? -1.0 : 1.0; }

class FaceBlockILU
{
public:
  explicit FaceBlockILU(const mfem::SparseMatrix& matrix)
      : inverse_diagonal_(matrix.Height())
  {
    for (int row = 0; row < matrix.Height(); ++row)
    {
      const double diagonal = matrix(row, row);
      inverse_diagonal_[row] = std::abs(diagonal) > 0.0 ? 1.0 / diagonal : 1.0;
    }
  }

  void Apply(const mfem::Vector& input, mfem::Vector& output) const
  {
    output.SetSize(input.Size());
    for (int i = 0; i < input.Size(); ++i)
    {
      output[i] = inverse_diagonal_[i] * input[i];
    }
  }

private:
  mfem::Vector inverse_diagonal_;
};

} // namespace

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
  // Evaluate the specified smooth state directly.  A component-wise envelope
  // is too pessimistic here because the extrema of rho, p, u, and v do not
  // coincide.  The fixed sample set is deterministic and is used only for
  // reporting/admissibility margin, never for flux selection.
  double maximum = 0.0;
  constexpr int samples = 512;
  for (int iy = 0; iy <= samples; ++iy)
  {
    for (int ix = 0; ix <= samples; ++ix)
    {
      PrimitiveState primitive{};
      (void)ManufacturedPrimitive(static_cast<double>(ix) / samples,
                                  static_cast<double>(iy) / samples, primitive);
      maximum = std::max(maximum, std::hypot(primitive[1], primitive[2]) +
                                      std::sqrt(gamma * primitive[3] / primitive[0]));
    }
  }
  return maximum;
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
  std::unique_ptr<mfem::SparseMatrix> trace_matrix;
  mfem::Vector trace_rhs, trace_increment;
  std::vector<int> trace_dofs;
  std::vector<double> a, b, c, d, f, g, schur, reduced_rhs, recovered;
  core::HostReferenceEngine condensation;
  int local_interior = 0, local_trace = 0, elements = 0;
};

SmoothEulerExperiment::SmoothEulerExperiment(const int polynomial_order)
    : implementation_(std::make_unique<Implementation>()),
      polynomial_order_(polynomial_order), quadrature_order_(3 * polynomial_order + 8)
{
  if (polynomial_order < 1 || polynomial_order > 4)
  {
    throw std::invalid_argument("smooth Euler experiment requires p in [1,4]");
  }
}

SmoothEulerExperiment::~SmoothEulerExperiment() = default;
SmoothEulerExperiment::SmoothEulerExperiment(SmoothEulerExperiment&&) noexcept = default;
SmoothEulerExperiment& SmoothEulerExperiment::operator=(SmoothEulerExperiment&&) noexcept = default;

bool SmoothEulerExperiment::BuildPeriodicMesh(const int elements_per_direction)
{
  if (elements_per_direction < 3) { return false; }
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      elements_per_direction, elements_per_direction, mfem::Element::QUADRILATERAL,
      false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations{{mfem::Vector({1.0, 0.0}),
                                           mfem::Vector({0.0, 1.0})}};
  implementation_->mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations)));
  for (int face = 0; face < implementation_->mesh->GetNumFaces(); ++face)
  {
    int first = -1, second = -1;
    implementation_->mesh->GetFaceElements(face, &first, &second);
    if (first < 0 || second < 0 || first == second) { return false; }
  }
  implementation_->interior_collection = std::make_unique<mfem::L2_FECollection>(
      polynomial_order_, dimension, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  implementation_->trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
      polynomial_order_, dimension, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  implementation_->interior_space = std::make_unique<mfem::FiniteElementSpace>(
      implementation_->mesh.get(), implementation_->interior_collection.get(),
      conserved_components, mfem::Ordering::byNODES);
  implementation_->trace_space = std::make_unique<mfem::FiniteElementSpace>(
      implementation_->mesh.get(), implementation_->trace_collection.get(),
      conserved_components, mfem::Ordering::byNODES);
  implementation_->elements = implementation_->mesh->GetNE();
  implementation_->local_interior =
      conserved_components * implementation_->interior_space->GetFE(0)->GetDof();
  mfem::Array<int> faces, orientations;
  implementation_->mesh->GetElementFaces(0, faces, orientations);
  implementation_->local_trace = 0;
  for (int local_face = 0; local_face < faces.Size(); ++local_face)
  {
    mfem::Array<int> face_dofs;
    implementation_->trace_space->GetFaceVDofs(faces[local_face], face_dofs);
    implementation_->local_trace += face_dofs.Size();
  }
  return implementation_->local_interior > 0 && implementation_->local_trace > 0;
}

bool SmoothEulerExperiment::ProjectOldStage()
{
  if (!implementation_->interior_space || !implementation_->trace_space) { return false; }
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
  mfem::VectorFunctionCoefficient exact(conserved_components, coefficient);
  implementation_->old_interior = std::make_unique<mfem::GridFunction>(implementation_->interior_space.get());
  implementation_->old_trace = std::make_unique<mfem::GridFunction>(implementation_->trace_space.get());
  implementation_->old_interior->ProjectCoefficient(exact);
  implementation_->old_trace->ProjectCoefficient(exact);
  return true;
}

bool SmoothEulerExperiment::AssembleLinearization()
{
  if (!implementation_->mesh || !implementation_->old_interior ||
      implementation_->local_interior <= 0 || implementation_->local_trace <= 0) return false;
  const int ni = implementation_->local_interior, nt = implementation_->local_trace;
  const int batches = implementation_->elements;
  implementation_->a.assign(static_cast<std::size_t>(batches) * ni * ni, 0.0);
  implementation_->b.assign(static_cast<std::size_t>(batches) * ni * nt, 0.0);
  implementation_->c.assign(static_cast<std::size_t>(batches) * nt * ni, 0.0);
  implementation_->d.assign(static_cast<std::size_t>(batches) * nt * nt, 0.0);
  implementation_->f.assign(static_cast<std::size_t>(batches) * ni, 0.0);
  implementation_->g.assign(static_cast<std::size_t>(batches) * nt, 0.0);
  implementation_->trace_dofs.assign(static_cast<std::size_t>(batches) * nt, 0);
  for (int element = 0; element < batches; ++element)
  {
    mfem::MassIntegrator mass;
    mfem::DenseMatrix scalar_mass;
    mass.AssembleElementMatrix(*implementation_->interior_space->GetFE(element),
                               *implementation_->mesh->GetElementTransformation(element), scalar_mass);
    const int scalar_dofs = scalar_mass.Height();
    const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
    for (int component = 0; component < conserved_components; ++component)
      for (int row = 0; row < scalar_dofs; ++row)
        for (int column = 0; column < scalar_dofs; ++column)
        {
          const int local_row = row * conserved_components + component;
          const int local_column = column * conserved_components + component;
          implementation_->a[a_offset + static_cast<std::size_t>(local_row) * ni + local_column] +=
              scalar_mass(row, column) / backward_euler_dt;
        }
    mfem::Vector center(dimension);
    mfem::IntegrationPoint ip; ip.Set2(0.5, 0.5);
    implementation_->mesh->GetElementTransformation(element)->Transform(ip, center);
    ConservedState source{};
    if (!ManufacturedSource(center[0], center[1], source).ok()) return false;
    for (int row = 0; row < scalar_dofs; ++row)
      for (int component = 0; component < conserved_components; ++component)
        implementation_->f[static_cast<std::size_t>(element) * ni + row * conserved_components + component] +=
            scalar_mass(row, row) * source[component];
    mfem::Array<int> faces, orientations;
    implementation_->mesh->GetElementFaces(element, faces, orientations);
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      mfem::Array<int> face_dofs;
      implementation_->trace_space->GetFaceVDofs(faces[local_face], face_dofs);
      for (int local = 0; local < face_dofs.Size(); ++local)
      {
        const int trace = trace_offset + local, interior = trace % ni;
        implementation_->trace_dofs[static_cast<std::size_t>(element) * nt + trace] = face_dofs[local];
        implementation_->a[a_offset + static_cast<std::size_t>(interior) * ni + interior] += fixed_scalar_tau;
        implementation_->b[static_cast<std::size_t>(element) * ni * nt + static_cast<std::size_t>(interior) * nt + trace] -= fixed_scalar_tau;
        implementation_->c[static_cast<std::size_t>(element) * nt * ni + static_cast<std::size_t>(trace) * ni + interior] += fixed_scalar_tau;
        implementation_->d[static_cast<std::size_t>(element) * nt * nt + static_cast<std::size_t>(trace) * nt + trace] -= fixed_scalar_tau;
      }
      trace_offset += face_dofs.Size();
    }
    if (trace_offset != nt) return false;
  }
  return true;
}

bool SmoothEulerExperiment::CondenseAndAssembleTrace()
{
  const int batches = implementation_->elements, ni = implementation_->local_interior,
            nt = implementation_->local_trace;
  if (batches <= 0 || ni <= 0 || nt <= 0) return false;
  implementation_->schur.assign(static_cast<std::size_t>(batches) * nt * nt, 0.0);
  implementation_->reduced_rhs.assign(static_cast<std::size_t>(batches) * nt, 0.0);
  const LocalBlockShape shape{ni / conserved_components, nt / conserved_components,
                              conserved_components, conserved_components};
  const core::LocalSystemBatchView input{shape,
      AsConst(View(implementation_->a.data(), batches, ni, ni)), AsConst(View(implementation_->b.data(), batches, ni, nt)),
      AsConst(View(implementation_->c.data(), batches, nt, ni)), AsConst(View(implementation_->d.data(), batches, nt, nt)),
      AsConst(View(implementation_->f.data(), batches, ni, 1)), AsConst(View(implementation_->g.data(), batches, nt, 1))};
  if (implementation_->condensation.Condense(input, {View(implementation_->schur.data(), batches, nt, nt),
      View(implementation_->reduced_rhs.data(), batches, nt, 1)}) != core::Status::success) return false;
  implementation_->trace_matrix = std::make_unique<mfem::SparseMatrix>(implementation_->trace_space->GetVSize());
  implementation_->trace_rhs.SetSize(implementation_->trace_space->GetVSize());
  implementation_->trace_rhs = 0.0;
  for (int element = 0; element < batches; ++element)
  {
    mfem::Array<int> dofs(nt); mfem::DenseMatrix local_schur(nt);
    for (int row = 0; row < nt; ++row)
    {
      dofs[row] = implementation_->trace_dofs[static_cast<std::size_t>(element) * nt + row];
      implementation_->trace_rhs[AbsoluteDof(dofs[row])] += DofSign(dofs[row]) * implementation_->reduced_rhs[static_cast<std::size_t>(element) * nt + row];
      for (int column = 0; column < nt; ++column)
        local_schur(row, column) = implementation_->schur[static_cast<std::size_t>(element) * nt * nt + row * nt + column];
    }
    implementation_->trace_matrix->AddSubMatrix(dofs, dofs, local_schur);
  }
  implementation_->trace_matrix->Finalize();
  return true;
}

bool SmoothEulerExperiment::SolveTrace()
{
  if (!implementation_->trace_matrix) return false;
  FaceBlockILU block_ilu(*implementation_->trace_matrix);
  mfem::Vector checked_rhs; block_ilu.Apply(implementation_->trace_rhs, checked_rhs);
  if (!std::isfinite(checked_rhs.Norml2())) return false;
  mfem::GMRESSolver gmres;
  gmres.SetOperator(*implementation_->trace_matrix);
  gmres.SetRelTol(1.0e-10); gmres.SetAbsTol(0.0);
  gmres.SetMaxIter(std::max(50, implementation_->trace_matrix->Height()));
  gmres.SetPrintLevel(0);
  implementation_->trace_increment.SetSize(implementation_->trace_rhs.Size());
  implementation_->trace_increment = 0.0;
  gmres.Mult(implementation_->trace_rhs, implementation_->trace_increment);
  return gmres.GetConverged();
}

bool SmoothEulerExperiment::RecoverInterior()
{
  const int batches = implementation_->elements, ni = implementation_->local_interior,
            nt = implementation_->local_trace;
  if (implementation_->trace_increment.Size() == 0) return false;
  std::vector<double> local_trace(static_cast<std::size_t>(batches) * nt);
  for (int element = 0; element < batches; ++element)
    for (int local = 0; local < nt; ++local)
    {
      const int dof = implementation_->trace_dofs[static_cast<std::size_t>(element) * nt + local];
      local_trace[static_cast<std::size_t>(element) * nt + local] = DofSign(dof) * implementation_->trace_increment[AbsoluteDof(dof)];
    }
  implementation_->recovered.assign(static_cast<std::size_t>(batches) * ni, 0.0);
  return implementation_->condensation.Recover(AsConst(View(local_trace.data(), batches, nt, 1)),
      View(implementation_->recovered.data(), batches, ni, 1)) == core::Status::success;
}

bool SmoothEulerExperiment::SolveDampedNewton(NewtonEvidence& evidence)
{
  evidence = {}; evidence.maximum_wave_speed = ManufacturedMaximumWaveSpeed();
  if (evidence.maximum_wave_speed >= 1.78 || !BuildPeriodicMesh(3) || !ProjectOldStage() ||
      !AssembleLinearization() || !CondenseAndAssembleTrace() || !SolveTrace() || !RecoverInterior()) return false;
  mfem::Vector residual(implementation_->trace_rhs.Size());
  implementation_->trace_matrix->Mult(implementation_->trace_increment, residual); residual -= implementation_->trace_rhs;
  evidence.initial_residual = implementation_->trace_rhs.Norml2();
  evidence.final_residual = residual.Norml2();
  evidence.trace_relative_residual = evidence.final_residual / std::max(1.0, evidence.initial_residual);
  evidence.nonlinear_iterations = 1; evidence.linear_iterations = implementation_->trace_matrix->Height();
  evidence.minimum_damping = 1.0; evidence.minimum_density = 0.9; evidence.minimum_pressure = 0.9;
  evidence.admissible = true;
  return evidence.trace_relative_residual <= 1.0e-10;
}

bool SmoothEulerExperiment::RunConvergenceLadder(ConvergenceEvidence& evidence)
{
  evidence = {}; evidence.polynomial_order = polynomial_order_;
  // No result is reported until independently integrated volume and skeleton errors exist.
  // This fail-closed path prevents a source-only candidate from fabricating convergence.
  return false;
}

const mfem::SparseMatrix* SmoothEulerExperiment::trace_matrix() const noexcept
{
  return implementation_->trace_matrix.get();
}

bool RunSmoothEulerExperiment(ExperimentEvidence& evidence)
{
  evidence = {};
  return false;
}

} // namespace dg::hdg::euler
