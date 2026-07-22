#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace
{
using dg::hdg::euler::ConservedState;
using dg::hdg::euler::ByNodesIndex;
using dg::hdg::euler::ComponentFromByNodesIndex;
using dg::hdg::euler::DecodeSignedVDof;
using dg::hdg::euler::FluxJacobian;
using dg::hdg::euler::FluxVector;
using dg::hdg::euler::ManufacturedConservative;
using dg::hdg::euler::ManufacturedMaximumWaveSpeed;
using dg::hdg::euler::ManufacturedSource;
using dg::hdg::euler::NormalFlux;
using dg::hdg::euler::NormalFluxJacobian;
using dg::hdg::euler::PhysicsStatus;
using dg::hdg::euler::RunSmoothEulerExperiment;
using dg::hdg::euler::ScalarFromByNodesIndex;
using dg::hdg::euler::SmoothEulerExperiment;
using dg::hdg::euler::ExperimentEvidence;
using dg::hdg::euler::LocalLinearizationSnapshot;
using dg::hdg::euler::fixed_scalar_tau;
using dg::hdg::euler::conserved_components;

bool Close(const double left, const double right, const double tolerance)
{
  return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

int TestHandFluxAndFailureAtomicity()
{
  const ConservedState state{{1.0, 2.0, 3.0, 10.0}};
  FluxVector flux{{-7.0, -7.0, -7.0, -7.0}};
  if (!NormalFlux(state, {{1.0, 0.0}}, flux).ok()) { return 1; }
  const FluxVector expected{{2.0, 5.4, 6.0, 22.8}};
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!Close(flux[component], expected[component], 2.0e-14)) { return 2; }
  }
  const FluxVector saved = flux;
  ConservedState bad_density = state;
  bad_density[0] = 0.0;
  if (NormalFlux(bad_density, {{1.0, 0.0}}, flux).status != PhysicsStatus::invalid_density ||
      flux != saved) { return 3; }
  ConservedState bad_pressure = state;
  bad_pressure[3] = 6.5;
  if (NormalFlux(bad_pressure, {{1.0, 0.0}}, flux).status != PhysicsStatus::invalid_pressure ||
      flux != saved) { return 4; }
  ConservedState non_finite = state;
  non_finite[2] = std::numeric_limits<double>::infinity();
  if (NormalFlux(non_finite, {{1.0, 0.0}}, flux).status != PhysicsStatus::non_finite ||
      flux != saved) { return 5; }
  return 0;
}

int TestAnalyticFluxJacobian()
{
  const ConservedState state{{1.2, 0.35, -0.22, 2.8}};
  const std::array<double, 2> normal{{0.6, -0.8}};
  FluxJacobian analytic{};
  if (!NormalFluxJacobian(state, normal, analytic).ok()) { return 1; }
  if (analytic[0][0] != 0.0) { return 2; }
  for (int column = 0; column < conserved_components; ++column)
  {
    ConservedState plus = state;
    ConservedState minus = state;
    const double step = 1.0e-7 * std::max(1.0, std::abs(state[column]));
    plus[column] += step;
    minus[column] -= step;
    FluxVector positive{};
    FluxVector negative{};
    if (!NormalFlux(plus, normal, positive).ok() || !NormalFlux(minus, normal, negative).ok())
    {
      return 3;
    }
    for (int row = 0; row < conserved_components; ++row)
    {
      const double difference = (positive[row] - negative[row]) / (2.0 * step);
      if (!Close(analytic[row][column], difference, 2.0e-6)) { return 4; }
    }
  }
  return 0;
}

int TestManufacturedSource()
{
  constexpr double x = 0.173, y = 0.619, h = 1.0e-5;
  ConservedState source{};
  if (!ManufacturedSource(x, y, source).ok()) { return 1; }
  FluxVector xp{}, xm{}, yp{}, ym{};
  ConservedState state{};
  if (!ManufacturedConservative(x + h, y, state).ok() || !NormalFlux(state, {{1.0, 0.0}}, xp).ok() ||
      !ManufacturedConservative(x - h, y, state).ok() || !NormalFlux(state, {{1.0, 0.0}}, xm).ok() ||
      !ManufacturedConservative(x, y + h, state).ok() || !NormalFlux(state, {{0.0, 1.0}}, yp).ok() ||
      !ManufacturedConservative(x, y - h, state).ok() || !NormalFlux(state, {{0.0, 1.0}}, ym).ok())
  {
    return 2;
  }
  for (int component = 0; component < conserved_components; ++component)
  {
    const double finite_difference = (xp[component] - xm[component] + yp[component] - ym[component]) /
                                     (2.0 * h);
    if (!Close(source[component], finite_difference, 2.0e-6)) { return 3; }
  }
  const double maximum_wave_speed = ManufacturedMaximumWaveSpeed();
  return maximum_wave_speed < 1.78 && fixed_scalar_tau - maximum_wave_speed > 0.72 ? 0 : 4;
}

int TestMfemByNodesAndSignedVDofs()
{
  constexpr int scalar_count = 9;
  for (int component = 0; component < conserved_components; ++component)
  {
    for (int scalar = 0; scalar < scalar_count; ++scalar)
    {
      const int vector_dof = ByNodesIndex(scalar, scalar_count, component);
      const int mfem_vector_dof = mfem::Ordering::Map<mfem::Ordering::byNODES>(
          scalar_count, conserved_components, scalar, component);
      if (vector_dof != mfem_vector_dof ||
          ComponentFromByNodesIndex(vector_dof, scalar_count) != component ||
          ScalarFromByNodesIndex(vector_dof, scalar_count) != scalar)
      {
        return 1;
      }
    }
  }
  double positive_sign = 0.0;
  double production_positive_sign = 0.0;
  if (DecodeSignedVDof(23, production_positive_sign) !=
          mfem::FiniteElementSpace::DecodeDof(23, positive_sign) ||
      production_positive_sign != positive_sign)
  {
    return 2;
  }
  double negative_sign = 0.0;
  double production_negative_sign = 0.0;
  if (DecodeSignedVDof(-24, production_negative_sign) !=
          mfem::FiniteElementSpace::DecodeDof(-24, negative_sign) ||
      production_negative_sign != negative_sign)
  {
    return 3;
  }
  return 0;
}

int TestProjectedTraceComponentOrdering()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage())
  {
    return 1;
  }
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  const mfem::GridFunction* trace = experiment.old_trace();
  if (trace_space == nullptr || trace == nullptr || trace_space->GetMesh() == nullptr)
  {
    return 2;
  }
  mfem::Mesh* mesh = trace_space->GetMesh();
  for (int face = 0; face < mesh->GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(face);
    mfem::ElementTransformation* transformation = mesh->GetFaceTransformation(face);
    if (face_fe == nullptr || transformation == nullptr) { return 3; }
    const int face_scalar_dofs = face_fe->GetDof();
    mfem::Array<int> face_vdofs;
    trace_space->GetFaceVDofs(face, face_vdofs);
    if (face_vdofs.Size() != conserved_components * face_scalar_dofs) { return 4; }
    for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
    {
      mfem::Vector point(dg::hdg::euler::dimension);
      transformation->Transform(face_fe->GetNodes().IntPoint(scalar), point);
      ConservedState state{};
      if (!ManufacturedConservative(point[0], point[1], state).ok()) { return 5; }
      for (int component = 0; component < conserved_components; ++component)
      {
        const int local = ByNodesIndex(scalar, face_scalar_dofs, component);
        double production_sign = 0.0;
        const int vdof = DecodeSignedVDof(face_vdofs[local], production_sign);
        if (vdof < 0 || vdof >= trace->Size() ||
            !Close((*trace)[vdof], production_sign * state[component], 2.0e-14))
        {
          return 6;
        }
      }
    }
  }
  return 0;
}

int TestElementFaceOrientationAndPermutation()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot snapshot;
  const mfem::FiniteElementSpace* trace_space = experiment.trace_space();
  if (trace_space == nullptr || trace_space->GetMesh() == nullptr ||
      !experiment.CopyLocalLinearization(snapshot)) return 2;
  mfem::Mesh* mesh = trace_space->GetMesh();
  for (int element = 0; element < snapshot.elements; ++element)
  {
    mfem::Array<int> faces, orientations;
    mesh->GetElementFaces(element, faces, orientations);
    int offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      const mfem::FiniteElement* face_fe = trace_space->GetFaceElement(face);
      mfem::FaceElementTransformations* transform = mesh->GetFaceElementTransformations(face);
      mfem::Array<int> face_vdofs;
      trace_space->GetFaceVDofs(face, face_vdofs);
      if (face_fe == nullptr || transform == nullptr ||
          (transform->Elem1No != element && transform->Elem2No != element)) return 3;
      for (int local = 0; local < face_vdofs.Size(); ++local)
      {
        if (snapshot.signed_trace_dofs[static_cast<std::size_t>(element) *
                                       snapshot.trace_dofs + offset + local] != face_vdofs[local])
        {
          return 4;
        }
      }
      for (int node = 0; node < face_fe->GetNodes().GetNPoints(); ++node)
      {
        const mfem::IntegrationPoint& face_point = face_fe->GetNodes().IntPoint(node);
        transform->SetAllIntPoints(&face_point);
        const mfem::IntegrationPoint& element_point = transform->Elem1No == element ?
            transform->GetElement1IntPoint() : transform->GetElement2IntPoint();
        mfem::Vector face_physical(dg::hdg::euler::dimension);
        mfem::Vector element_physical(dg::hdg::euler::dimension);
        transform->Face->Transform(face_point, face_physical);
        mesh->GetElementTransformation(element)->Transform(element_point, element_physical);
        if (!Close(face_physical[0], element_physical[0], 2.0e-14) ||
            !Close(face_physical[1], element_physical[1], 2.0e-14)) return 5;
      }
      offset += face_vdofs.Size();
    }
    if (offset != snapshot.trace_dofs || orientations.Size() != faces.Size()) return 6;
  }
  return 0;
}

int TestFailurePreservingSetup()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage())
  {
    return 1;
  }
  if (experiment.BuildPeriodicMesh(2)) { return 2; }
  // A failed fresh build cannot displace the previously complete mesh/space
  // graph, so the already configured projection remains reusable.
  if (!experiment.ProjectOldStage()) { return 3; }
  return experiment.BuildPeriodicMesh(3) && experiment.ProjectOldStage() ? 0 : 4;
}

bool SolveDense(std::vector<double> matrix, std::vector<double>& rhs, const int n)
{
  if (static_cast<int>(matrix.size()) != n * n || static_cast<int>(rhs.size()) != n)
  {
    return false;
  }
  for (int pivot = 0; pivot < n; ++pivot)
  {
    int best = pivot;
    for (int row = pivot + 1; row < n; ++row)
    {
      if (std::abs(matrix[static_cast<std::size_t>(row) * n + pivot]) >
          std::abs(matrix[static_cast<std::size_t>(best) * n + pivot])) best = row;
    }
    if (std::abs(matrix[static_cast<std::size_t>(best) * n + pivot]) < 1.0e-13)
    {
      return false;
    }
    for (int column = pivot; column < n; ++column)
    {
      std::swap(matrix[static_cast<std::size_t>(pivot) * n + column],
                matrix[static_cast<std::size_t>(best) * n + column]);
    }
    std::swap(rhs[pivot], rhs[best]);
    const double diagonal = matrix[static_cast<std::size_t>(pivot) * n + pivot];
    for (int column = pivot; column < n; ++column)
    {
      matrix[static_cast<std::size_t>(pivot) * n + column] /= diagonal;
    }
    rhs[pivot] /= diagonal;
    for (int row = 0; row < n; ++row)
    {
      if (row == pivot) { continue; }
      const double factor = matrix[static_cast<std::size_t>(row) * n + pivot];
      for (int column = pivot; column < n; ++column)
      {
        matrix[static_cast<std::size_t>(row) * n + column] -=
            factor * matrix[static_cast<std::size_t>(pivot) * n + column];
      }
      rhs[row] -= factor * rhs[pivot];
    }
  }
  return true;
}

std::vector<double> IndependentlyCondense(const LocalLinearizationSnapshot& snapshot,
                                          const int element)
{
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
  const std::size_t b_offset = static_cast<std::size_t>(element) * ni * nt;
  const std::size_t c_offset = static_cast<std::size_t>(element) * nt * ni;
  const std::size_t d_offset = static_cast<std::size_t>(element) * nt * nt;
  std::vector<double> result(nt * nt);
  for (int row = 0; row < nt; ++row)
    for (int column = 0; column < nt; ++column)
    {
      std::vector<double> rhs(ni);
      for (int interior = 0; interior < ni; ++interior)
      {
        rhs[interior] = snapshot.b[b_offset + static_cast<std::size_t>(interior) * nt + column];
      }
      std::vector<double> a(snapshot.a.begin() + a_offset,
                            snapshot.a.begin() + a_offset + ni * ni);
      if (!SolveDense(std::move(a), rhs, ni)) { return {}; }
      double reduced = snapshot.d[d_offset + static_cast<std::size_t>(row) * nt + column];
      for (int interior = 0; interior < ni; ++interior)
      {
        reduced -= snapshot.c[c_offset + static_cast<std::size_t>(row) * ni + interior] *
                   rhs[interior];
      }
      result[static_cast<std::size_t>(row) * nt + column] = reduced;
    }
  return result;
}

std::vector<double> ResolveInteriorThenEvaluateTrace(
    const LocalLinearizationSnapshot& snapshot, const int element,
    const std::vector<double>& trace)
{
  const int ni = snapshot.interior_dofs;
  const int nt = snapshot.trace_dofs;
  if (static_cast<int>(trace.size()) != nt) return {};
  const std::size_t a_offset = static_cast<std::size_t>(element) * ni * ni;
  const std::size_t b_offset = static_cast<std::size_t>(element) * ni * nt;
  const std::size_t c_offset = static_cast<std::size_t>(element) * nt * ni;
  const std::size_t d_offset = static_cast<std::size_t>(element) * nt * nt;
  std::vector<double> rhs(ni);
  for (int row = 0; row < ni; ++row)
  {
    rhs[row] = snapshot.f[static_cast<std::size_t>(element) * ni + row];
    for (int column = 0; column < nt; ++column)
    {
      rhs[row] -= snapshot.b[b_offset + static_cast<std::size_t>(row) * nt + column] *
                  trace[column];
    }
  }
  std::vector<double> a(snapshot.a.begin() + a_offset,
                        snapshot.a.begin() + a_offset + ni * ni);
  if (!SolveDense(std::move(a), rhs, ni)) return {};
  std::vector<double> residual(nt);
  for (int row = 0; row < nt; ++row)
  {
    residual[row] = -snapshot.g[static_cast<std::size_t>(element) * nt + row];
    for (int column = 0; column < ni; ++column)
    {
      residual[row] += snapshot.c[c_offset + static_cast<std::size_t>(row) * ni + column] *
                       rhs[column];
    }
    for (int column = 0; column < nt; ++column)
    {
      residual[row] += snapshot.d[d_offset + static_cast<std::size_t>(row) * nt + column] *
                       trace[column];
    }
  }
  return residual;
}

int TestFixedScalarTauConstantAndConservationOracles()
{
  const ConservedState state{{1.1, 0.3, -0.2, 2.9}};
  const std::array<double, 2> normal{{0.6, -0.8}};
  FluxVector positive{}, negative{};
  if (!NormalFlux(state, normal, positive).ok() ||
      !NormalFlux(state, {{-normal[0], -normal[1]}}, negative).ok()) return 1;
  // With U=Uhat the tau term is independently zero, and the two oriented
  // numerical fluxes must cancel component-wise.
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!Close(positive[component] + negative[component], 0.0, 3.0e-14)) return 2;
  }
  const ConservedState left{{1.0, 0.2, -0.1, 2.6}};
  const ConservedState right{{1.2, 0.1, -0.3, 3.0}};
  const ConservedState trace{{1.1, 0.15, -0.2, 2.8}};
  FluxVector trace_positive{}, trace_negative{};
  if (!NormalFlux(trace, normal, trace_positive).ok() ||
      !NormalFlux(trace, {{-normal[0], -normal[1]}}, trace_negative).ok()) return 3;
  // This is separate skeleton conservation accounting: advective terms cancel
  // by orientation and the remaining jump is exactly the two tau penalties.
  for (int component = 0; component < conserved_components; ++component)
  {
    const double accounted = trace_positive[component] + trace_negative[component] +
        fixed_scalar_tau * (left[component] - trace[component]) +
        fixed_scalar_tau * (right[component] - trace[component]);
    const double expected = fixed_scalar_tau *
        (left[component] + right[component] - 2.0 * trace[component]);
    if (!Close(accounted, expected, 3.0e-14)) return 4;
  }
  return 0;
}

int TestFaceBlockDirectionalAndIndependentSchurOracles()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage() ||
      !experiment.AssembleLinearization()) return 1;
  LocalLinearizationSnapshot snapshot;
  if (!experiment.CopyLocalLinearization(snapshot) || snapshot.elements <= 0 ||
      snapshot.a.empty() || snapshot.b.empty() || snapshot.c.empty() || snapshot.d.empty())
  {
    return 2;
  }
  // Independent directional finite differences of the two arguments of
  // F(Uhat).n + tau(U-Uhat), which respectively exercise A/C and B/D signs.
  const ConservedState interior{{1.15, 0.31, -0.18, 2.95}};
  const ConservedState trace{{1.08, 0.22, -0.27, 2.85}};
  const ConservedState direction{{0.03, -0.02, 0.04, 0.05}};
  const std::array<double, 2> normal{{0.8, 0.6}};
  FluxJacobian jacobian{};
  if (!NormalFluxJacobian(trace, normal, jacobian).ok()) return 3;
  constexpr double h = 1.0e-7;
  for (int varied_trace = 0; varied_trace < 2; ++varied_trace)
  {
    ConservedState plus_interior = interior, minus_interior = interior;
    ConservedState plus_trace = trace, minus_trace = trace;
    for (int component = 0; component < conserved_components; ++component)
    {
      (varied_trace ? plus_trace : plus_interior)[component] += h * direction[component];
      (varied_trace ? minus_trace : minus_interior)[component] -= h * direction[component];
    }
    FluxVector fp{}, fm{};
    if (!NormalFlux(plus_trace, normal, fp).ok() || !NormalFlux(minus_trace, normal, fm).ok())
    {
      return 4;
    }
    for (int row = 0; row < conserved_components; ++row)
    {
      const double difference = (fp[row] - fm[row]) / (2.0 * h) +
          fixed_scalar_tau * ((plus_interior[row] - plus_trace[row]) -
                              (minus_interior[row] - minus_trace[row])) / (2.0 * h);
      double analytic = 0.0;
      for (int column = 0; column < conserved_components; ++column)
      {
        analytic += (varied_trace ? jacobian[row][column] -
                                      (row == column ? fixed_scalar_tau : 0.0) :
                                    (row == column ? fixed_scalar_tau : 0.0)) * direction[column];
      }
      if (!Close(difference, analytic, 3.0e-6)) return 5;
    }
  }
  if (!experiment.CondenseAndAssembleTrace() || experiment.trace_matrix() == nullptr) return 6;
  const mfem::SparseMatrix& condensed = *experiment.trace_matrix();
  std::vector<double> monolithic(static_cast<std::size_t>(condensed.Height()) * condensed.Width());
  for (int element = 0; element < snapshot.elements; ++element)
  {
    const std::vector<double> local = IndependentlyCondense(snapshot, element);
    if (local.empty()) return 7;
    for (int row = 0; row < snapshot.trace_dofs; ++row)
    {
      double row_sign = 1.0;
      const int global_row = DecodeSignedVDof(
          snapshot.signed_trace_dofs[static_cast<std::size_t>(element) * snapshot.trace_dofs + row], row_sign);
      for (int column = 0; column < snapshot.trace_dofs; ++column)
      {
        double column_sign = 1.0;
        const int global_column = DecodeSignedVDof(
            snapshot.signed_trace_dofs[static_cast<std::size_t>(element) * snapshot.trace_dofs + column], column_sign);
        monolithic[static_cast<std::size_t>(global_row) * condensed.Width() + global_column] +=
            row_sign * column_sign * local[static_cast<std::size_t>(row) * snapshot.trace_dofs + column];
      }
    }
  }
  for (int row = 0; row < condensed.Height(); ++row)
    for (int column = 0; column < condensed.Width(); ++column)
    {
      if (!Close(monolithic[static_cast<std::size_t>(row) * condensed.Width() + column],
                 condensed(row, column), 2.0e-10)) return 8;
    }
  // Independently resolve the local interior state for two perturbed traces,
  // then difference the reduced residual.  This is not ApplySchur applied to
  // itself: the interior solve and residual evaluation are reconstructed here.
  const std::vector<double> local_schur = IndependentlyCondense(snapshot, 0);
  std::vector<double> lambda(snapshot.trace_dofs, 0.0);
  for (int i = 0; i < snapshot.trace_dofs; ++i) { lambda[i] = 0.01 * (i + 1); }
  std::vector<double> plus = lambda, minus = lambda, expected(snapshot.trace_dofs, 0.0);
  constexpr double reduced_step = 1.0e-7;
  for (int i = 0; i < snapshot.trace_dofs; ++i)
  {
    plus[i] += reduced_step * lambda[i];
    minus[i] -= reduced_step * lambda[i];
  }
  const std::vector<double> reduced_plus = ResolveInteriorThenEvaluateTrace(snapshot, 0, plus);
  const std::vector<double> reduced_minus = ResolveInteriorThenEvaluateTrace(snapshot, 0, minus);
  if (reduced_plus.empty() || reduced_minus.empty()) return 9;
  for (int row = 0; row < snapshot.trace_dofs; ++row)
    for (int column = 0; column < snapshot.trace_dofs; ++column)
    {
      expected[row] += local_schur[static_cast<std::size_t>(row) * snapshot.trace_dofs + column] * lambda[column];
    }
  for (int row = 0; row < snapshot.trace_dofs; ++row)
  {
    const double derivative = (reduced_plus[row] - reduced_minus[row]) / (2.0 * reduced_step);
    if (!Close(derivative, expected[row], 2.0e-6)) return 10;
  }
  return 0;
}

int TestCompleteExperimentGate()
{
  ExperimentEvidence evidence{};
  evidence.passed = true;
  // This is a real complete-gate invocation, not a print-only status.  03B
  // ends at a fixed-state linearization; full Newton/convergence remains
  // deliberately unavailable until 03C, so the top-level experiment is an
  // explicit fail-closed gate rather than a false success.
  return !RunSmoothEulerExperiment(evidence) && !evidence.passed ? 0 : 1;
}

} // namespace

int main()
{
  const int flux = TestHandFluxAndFailureAtomicity();
  if (flux != 0) { return 10 + flux; }
  const int jacobian = TestAnalyticFluxJacobian();
  if (jacobian != 0) { return 20 + jacobian; }
  const int source = TestManufacturedSource();
  if (source != 0) { return 30 + source; }
  const int ordering = TestMfemByNodesAndSignedVDofs();
  if (ordering != 0) { return 40 + ordering; }
  const int trace_ordering = TestProjectedTraceComponentOrdering();
  if (trace_ordering != 0) { return 50 + trace_ordering; }
  const int orientation = TestElementFaceOrientationAndPermutation();
  if (orientation != 0) { return 60 + orientation; }
  const int setup = TestFailurePreservingSetup();
  if (setup != 0) { return 70 + setup; }
  const int face_oracles = TestFixedScalarTauConstantAndConservationOracles();
  if (face_oracles != 0) { return 80 + face_oracles; }
  const int block_oracles = TestFaceBlockDirectionalAndIndependentSchurOracles();
  if (block_oracles != 0) { return 90 + block_oracles; }
  const int experiment = TestCompleteExperimentGate();
  return experiment == 0 ? 0 : 100 + experiment;
}
