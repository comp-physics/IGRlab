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

struct PipelinePhaseState
{
  StridedBlockView<double> interior_residual{}, trace_residual{};
  StridedBlockView<double> a{}, b{}, c{}, d{}, schur{}, rhs{};
  StridedBlockView<double> trace_solution{}, interior_solution{};
  bool residual = false, jacobian = false, condensed = false, schur_applied = false,
       assembled = false, solved = false, recovered = false;
};

struct SmoothEulerLocalPhase
{
  PipelinePhaseState* state = nullptr;
  void EvaluateResidual(const LocalResidualRequest<SmoothEulerSpec>&,
                        LocalResidualResult<SmoothEulerSpec>& result)
  {
    result.interior_residual = state->interior_residual;
    result.trace_residual = state->trace_residual;
    state->residual = result.interior_residual.data != nullptr &&
                      result.trace_residual.data != nullptr;
  }
  void EvaluateJacobian(const LocalJacobianRequest<SmoothEulerSpec>&,
                        LocalJacobianResult<SmoothEulerSpec>& result)
  {
    result.linearization = {state->interior_residual, state->trace_residual,
                            state->a, state->b, state->c, state->d};
    state->jacobian = result.linearization.A.data != nullptr &&
                      result.linearization.D.data != nullptr;
  }
};

struct SmoothEulerCondensationPhase
{
  PipelinePhaseState* state = nullptr;
  void Condense(const CondenseRequest<SmoothEulerSpec>&,
                CondenseResult<SmoothEulerSpec>& result)
  {
    result.contribution.schur = AsConst(state->schur);
    result.contribution.right_hand_side = AsConst(state->rhs);
    result.contribution.trace_dof_count = state->schur.rows;
    state->condensed = result.contribution.schur.data != nullptr;
  }
  void ApplySchur(const ApplySchurRequest<SmoothEulerSpec>& request,
                  ApplySchurResult<SmoothEulerSpec>& result)
  {
    result.trace_residual = state->trace_solution;
    state->schur_applied = request.contribution.schur.data != nullptr &&
                           result.trace_residual.data != nullptr;
  }
};

struct SmoothEulerAssemblyPhase
{
  PipelinePhaseState* state = nullptr;
  void Assemble(const TraceAssemblyRequest<SmoothEulerSpec>& request,
                TraceAssemblyResult<SmoothEulerSpec>& result)
  {
    result.trace_system = state->schur;
    result.right_hand_side = state->rhs;
    state->assembled = request.contribution.schur.data != nullptr &&
                       result.trace_system.data != nullptr;
  }
};

struct SmoothEulerTraceSolvePhase
{
  PipelinePhaseState* state = nullptr;
  void Solve(const TraceSolveRequest<SmoothEulerSpec>& request,
             TraceSolveResult<SmoothEulerSpec>& result)
  {
    result.trace_solution = state->trace_solution;
    state->solved = request.trace_system.data != nullptr &&
                    result.trace_solution.data != nullptr;
  }
};

struct SmoothEulerRecoveryPhase
{
  PipelinePhaseState* state = nullptr;
  void Recover(const RecoveryRequest<SmoothEulerSpec>&,
               RecoveryResult<SmoothEulerSpec>& result)
  {
    result.interior_solution = state->interior_solution;
    state->recovered = result.interior_solution.data != nullptr;
  }
};

using SmoothEulerPipeline = Pipeline<SmoothEulerSpec, SmoothEulerLocalPhase,
                                    SmoothEulerCondensationPhase,
                                    SmoothEulerAssemblyPhase,
                                    SmoothEulerTraceSolvePhase,
                                    SmoothEulerRecoveryPhase>;

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
  return std::hypot(0.35, 0.20) + 0.05 +
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
  std::unique_ptr<mfem::SparseMatrix> trace_matrix;
  mfem::Vector trace_rhs, trace_increment;
  std::vector<int> trace_dofs;
  std::vector<double> a, b, c, d, f, g, schur, reduced_rhs, recovered;
  std::unique_ptr<PermutedFaceBlockILU> trace_preconditioner;
  core::HostReferenceEngine condensation;
  int local_interior = 0, local_trace = 0, elements = 0, trace_face_block_size = 0;
  std::vector<int> trace_permutation;
  std::size_t generation = 0, linearization_generation = 0,
              condensation_generation = 0, solve_generation = 0,
              recovery_generation = 0;
  PipelinePhaseState pipeline_state{};
  std::unique_ptr<SmoothEulerPipeline> pipeline;
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
    const mfem::FiniteElement* interior_fe = candidate->interior_space->GetFE(0);
    if (interior_fe == nullptr || interior_fe->GetDof() <= 0) { return false; }
    candidate->local_interior = conserved_components * interior_fe->GetDof();
    mfem::Array<int> faces, orientations;
    candidate->mesh->GetElementFaces(0, faces, orientations);
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
      candidate->mesh->GetElementFaces(element, element_faces, element_orientations);
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
  }
  catch (const std::exception&)
  {
    return false;
  }
  implementation_->old_interior = std::move(candidate_interior);
  implementation_->old_trace = std::move(candidate_trace);
  ++implementation_->generation;
  implementation_->linearization_generation = 0;
  implementation_->condensation_generation = 0;
  implementation_->solve_generation = 0;
  implementation_->recovery_generation = 0;
  implementation_->trace_matrix.reset();
  implementation_->trace_preconditioner.reset();
  implementation_->trace_increment.SetSize(0);
  implementation_->recovered.clear();
  implementation_->pipeline_state = {};
  return true;
}

bool SmoothEulerExperiment::AssembleLinearization()
{
  if (!implementation_->mesh || !implementation_->old_interior ||
      !implementation_->old_trace || implementation_->local_interior <= 0 ||
      implementation_->local_trace <= 0) return false;
  const int ni = implementation_->local_interior;
  const int nt = implementation_->local_trace;
  const int batches = implementation_->elements;
  std::vector<double> candidate_a(static_cast<std::size_t>(batches) * ni * ni,
                                  0.0);
  std::vector<double> candidate_b(static_cast<std::size_t>(batches) * ni * nt, 0.0);
  std::vector<double> candidate_c(static_cast<std::size_t>(batches) * nt * ni, 0.0);
  std::vector<double> candidate_d(static_cast<std::size_t>(batches) * nt * nt, 0.0);
  std::vector<double> candidate_f(static_cast<std::size_t>(batches) * ni, 0.0);
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
      ConservedState state{};
      ConservedState source{};
      FluxVector flux_x{};
      FluxVector flux_y{};
      FluxJacobian jacobian_x{};
      FluxJacobian jacobian_y{};
      if (!InterpolateByNodes(*implementation_->old_interior, interior_vdofs, shape, state) ||
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
          const double residual = -weight *
              (derivative(row, 0) * flux_x[component] +
               derivative(row, 1) * flux_y[component] + shape[row] * source[component]);
          candidate_f[static_cast<std::size_t>(element) * ni + local_row] -= residual;
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
    implementation_->mesh->GetElementFaces(element, faces, orientations);
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
        if (!InterpolateByNodes(*implementation_->old_interior, interior_vdofs,
                                interior_shape, interior_state) ||
            !InterpolateByNodes(*implementation_->old_trace, face_vdofs,
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
            candidate_f[static_cast<std::size_t>(element) * ni + local_row] -= residual;
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
            candidate_g[static_cast<std::size_t>(element) * nt + local_row] -= residual;
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
  implementation_->a = std::move(candidate_a);
  implementation_->b = std::move(candidate_b);
  implementation_->c = std::move(candidate_c);
  implementation_->d = std::move(candidate_d);
  implementation_->f = std::move(candidate_f);
  implementation_->g = std::move(candidate_g);
  implementation_->trace_matrix.reset();
  implementation_->trace_preconditioner.reset();
  implementation_->trace_increment.SetSize(0);
  implementation_->recovered.clear();
  ++implementation_->generation;
  implementation_->linearization_generation = implementation_->generation;
  implementation_->condensation_generation = 0;
  implementation_->solve_generation = 0;
  implementation_->recovery_generation = 0;
  implementation_->pipeline_state = {};
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  pipeline_state.interior_residual = View(implementation_->f.data(), batches, ni, 1);
  pipeline_state.trace_residual = View(implementation_->g.data(), batches, nt, 1);
  pipeline_state.a = View(implementation_->a.data(), batches, ni, ni);
  pipeline_state.b = View(implementation_->b.data(), batches, ni, nt);
  pipeline_state.c = View(implementation_->c.data(), batches, nt, ni);
  pipeline_state.d = View(implementation_->d.data(), batches, nt, nt);
  LocalResidualRequest<SmoothEulerSpec> residual_request;
  LocalResidualResult<SmoothEulerSpec> residual_result;
  LocalJacobianRequest<SmoothEulerSpec> jacobian_request;
  LocalJacobianResult<SmoothEulerSpec> jacobian_result;
  implementation_->pipeline->EvaluateLocalResidual(residual_request, residual_result);
  implementation_->pipeline->EvaluateLocalJacobian(jacobian_request, jacobian_result);
  if (!pipeline_state.residual || !pipeline_state.jacobian) return false;
  return true;
}

bool SmoothEulerExperiment::CondenseAndAssembleTrace()
{
  const int batches = implementation_->elements, ni = implementation_->local_interior,
            nt = implementation_->local_trace;
  if (implementation_->linearization_generation == 0 || batches <= 0 || ni <= 0 || nt <= 0 ||
      implementation_->a.size() != static_cast<std::size_t>(batches) * ni * ni ||
      implementation_->b.size() != static_cast<std::size_t>(batches) * ni * nt ||
      implementation_->c.size() != static_cast<std::size_t>(batches) * nt * ni ||
      implementation_->d.size() != static_cast<std::size_t>(batches) * nt * nt ||
      implementation_->f.size() != static_cast<std::size_t>(batches) * ni ||
      implementation_->g.size() != static_cast<std::size_t>(batches) * nt) return false;
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
      double sign = 1.0;
      const int vdof = DecodeSignedVDof(dofs[row], sign);
      implementation_->trace_rhs[vdof] +=
          sign * implementation_->reduced_rhs[static_cast<std::size_t>(element) * nt + row];
      for (int column = 0; column < nt; ++column)
        local_schur(row, column) = implementation_->schur[static_cast<std::size_t>(element) * nt * nt + row * nt + column];
    }
    implementation_->trace_matrix->AddSubMatrix(dofs, dofs, local_schur);
  }
  implementation_->trace_matrix->Finalize();
  implementation_->trace_increment.SetSize(0);
  implementation_->recovered.clear();
  implementation_->trace_preconditioner.reset();
  implementation_->condensation_generation = implementation_->linearization_generation;
  implementation_->solve_generation = 0;
  implementation_->recovery_generation = 0;
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  pipeline_state.schur = View(implementation_->schur.data(), batches, nt, nt);
  pipeline_state.rhs = View(implementation_->reduced_rhs.data(), batches, nt, 1);
  pipeline_state.trace_solution = View(implementation_->trace_rhs.GetData(), 1,
                                       implementation_->trace_rhs.Size(), 1);
  CondenseRequest<SmoothEulerSpec> condense_request;
  CondenseResult<SmoothEulerSpec> condense_result;
  implementation_->pipeline->CondenseLocalSystem(condense_request, condense_result);
  ApplySchurRequest<SmoothEulerSpec> apply_request;
  apply_request.contribution = condense_result.contribution;
  ApplySchurResult<SmoothEulerSpec> apply_result;
  implementation_->pipeline->ApplySchur(apply_request, apply_result);
  TraceAssemblyRequest<SmoothEulerSpec> assembly_request;
  assembly_request.contribution = condense_result.contribution;
  TraceAssemblyResult<SmoothEulerSpec> assembly_result;
  implementation_->pipeline->AssembleTraceSystem(assembly_request, assembly_result);
  if (!pipeline_state.condensed || !pipeline_state.schur_applied ||
      !pipeline_state.assembled) return false;
  return true;
}

bool SmoothEulerExperiment::SolveTrace()
{
  if (!implementation_->trace_matrix || implementation_->condensation_generation == 0 ||
      implementation_->condensation_generation != implementation_->linearization_generation) return false;
  implementation_->trace_increment.SetSize(0);
  implementation_->recovered.clear();
  implementation_->solve_generation = 0;
  implementation_->recovery_generation = 0;
  implementation_->trace_preconditioner = std::make_unique<PermutedFaceBlockILU>(
      *implementation_->trace_matrix, implementation_->trace_permutation,
      implementation_->trace_face_block_size);
  mfem::GMRESSolver gmres;
  gmres.SetPreconditioner(*implementation_->trace_preconditioner);
  gmres.SetOperator(*implementation_->trace_matrix);
  gmres.SetRelTol(1.0e-10); gmres.SetAbsTol(0.0);
  gmres.SetMaxIter(std::max(50, implementation_->trace_matrix->Height()));
  gmres.SetPrintLevel(0);
  mfem::Vector candidate_increment(implementation_->trace_rhs.Size());
  candidate_increment = 0.0;
  gmres.Mult(implementation_->trace_rhs, candidate_increment);
  if (!gmres.GetConverged()) return false;
  implementation_->trace_increment = candidate_increment;
  implementation_->solve_generation = implementation_->condensation_generation;
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  pipeline_state.trace_solution = View(implementation_->trace_increment.GetData(), 1,
                                       implementation_->trace_increment.Size(), 1);
  TraceSolveRequest<SmoothEulerSpec> solve_request;
  solve_request.trace_system = AsConst(pipeline_state.schur);
  solve_request.right_hand_side = AsConst(pipeline_state.rhs);
  TraceSolveResult<SmoothEulerSpec> solve_result;
  implementation_->pipeline->SolveGlobalTraceSystem(solve_request, solve_result);
  if (!pipeline_state.solved) return false;
  return true;
}

bool SmoothEulerExperiment::RecoverInterior()
{
  const int batches = implementation_->elements, ni = implementation_->local_interior,
            nt = implementation_->local_trace;
  if (implementation_->trace_increment.Size() == 0 || implementation_->solve_generation == 0 ||
      implementation_->solve_generation != implementation_->condensation_generation) return false;
  std::vector<double> local_trace(static_cast<std::size_t>(batches) * nt);
  for (int element = 0; element < batches; ++element)
    for (int local = 0; local < nt; ++local)
    {
      const int signed_vdof =
          implementation_->trace_dofs[static_cast<std::size_t>(element) * nt + local];
      double sign = 1.0;
      const int vdof = DecodeSignedVDof(signed_vdof, sign);
      local_trace[static_cast<std::size_t>(element) * nt + local] =
          sign * implementation_->trace_increment[vdof];
    }
  std::vector<double> candidate_recovered(static_cast<std::size_t>(batches) * ni, 0.0);
  if (implementation_->condensation.Recover(AsConst(View(local_trace.data(), batches, nt, 1)),
      View(candidate_recovered.data(), batches, ni, 1)) != core::Status::success) return false;
  implementation_->recovered = std::move(candidate_recovered);
  implementation_->recovery_generation = implementation_->solve_generation;
  if (!implementation_->pipeline) return false;
  auto& pipeline_state = implementation_->pipeline_state;
  pipeline_state.interior_solution = View(implementation_->recovered.data(), batches, ni, 1);
  RecoveryRequest<SmoothEulerSpec> recovery_request;
  RecoveryResult<SmoothEulerSpec> recovery_result;
  implementation_->pipeline->RecoverInterior(recovery_request, recovery_result);
  if (!pipeline_state.recovered) return false;
  return true;
}

bool SmoothEulerExperiment::SolveDampedNewton(NewtonEvidence& evidence)
{
  evidence = {};
  // 03C owns state-updating Newton, admissibility, and convergence evidence.
  // A linearized 03B candidate must not manufacture any of that evidence.
  return false;
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

bool SmoothEulerExperiment::CopyLocalLinearization(
    LocalLinearizationSnapshot& snapshot) const
{
  snapshot = {};
  if (implementation_->elements <= 0 || implementation_->local_interior <= 0 ||
      implementation_->local_trace <= 0 || implementation_->a.empty() ||
      implementation_->b.empty() || implementation_->c.empty() ||
      implementation_->d.empty() || implementation_->trace_dofs.empty()) return false;
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

const mfem::GridFunction* SmoothEulerExperiment::old_trace() const noexcept
{
  return implementation_->old_trace.get();
}

bool RunSmoothEulerExperiment(ExperimentEvidence& evidence)
{
  evidence = {};
  return false;
}

} // namespace dg::hdg::euler
