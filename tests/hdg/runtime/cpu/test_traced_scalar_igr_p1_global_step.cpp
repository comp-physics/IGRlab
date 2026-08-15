#include "numerics/hdg/runtime/cpu/detail/traced_scalar_igr_p1_global_step.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace
{
namespace detail = dg::hdg::runtime::cpu::detail;
namespace local = dg::hdg::runtime::cpu::traced_scalar_igr_p1;
using Layout = dg::hdg::runtime::cpu::BorrowedP1DGInterfaceLayout;

bool AllRanks(const MPI_Comm communicator, const bool local_ok)
{
  const int source = local_ok ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

bool Require(const MPI_Comm communicator, const bool local_ok,
             const char* const stage)
{
  if (AllRanks(communicator, local_ok)) { return true; }
  int rank = -1;
  MPI_Comm_rank(communicator, &rank);
  if (!local_ok)
  {
    std::cerr << "traced_scalar_igr_p1_global_step: " << stage
              << " failed on rank " << rank << '\n';
  }
  return false;
}

bool Finite(const mfem::Vector& values)
{
  const double* const data = values.HostRead();
  if (data == nullptr && values.Size() != 0) { return false; }
  for (int index = 0; index < values.Size(); ++index)
  {
    if (!std::isfinite(data[index])) { return false; }
  }
  return true;
}

double NormSquared(const MPI_Comm communicator, const mfem::Vector& values)
{
  return mfem::InnerProduct(communicator, values, values);
}

mfem::ParMesh MakeMesh(const MPI_Comm communicator)
{
  int ranks = 1;
  MPI_Comm_size(communicator, &ranks);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  const std::vector<int> partition{0, ranks == 1 ? 0 : 1};
  return mfem::ParMesh(communicator, serial, partition.data());
}

bool FillTrueTrace(const mfem::ParFiniteElementSpace& space, mfem::Vector& values,
                   const double perturbation)
{
  if (values.Size() != space.GetTrueVSize()) { return false; }
  values = 0.0;
  for (int local_dof = 0; local_dof < space.GetVSize(); ++local_dof)
  {
    const int local_true = space.GetLocalTDofNumber(local_dof);
    if (local_true < 0) { continue; }
    const int scalar_vsize = space.GetVSize() / 5;
    if (scalar_vsize <= 0) { return false; }
    const int component = local_dof / scalar_vsize;
    const double base = component == 0 ? 1.0 : component == 1 ? 0.14 :
        component == 2 ? -0.03 : component == 3 ? 2.8 : 0.85;
    values[local_true] = base + perturbation * static_cast<double>(local_dof + 1);
  }
  return true;
}

bool FillEquilibriumTrueTrace(const mfem::ParFiniteElementSpace& space,
                              mfem::Vector& values)
{
  if (values.Size() != space.GetTrueVSize()) { return false; }
  values = 0.0;
  for (int local_dof = 0; local_dof < space.GetVSize(); ++local_dof)
  {
    const int local_true = space.GetLocalTDofNumber(local_dof);
    if (local_true < 0) { continue; }
    const int scalar_vsize = space.GetVSize() / 5;
    if (scalar_vsize <= 0) { return false; }
    const int component = local_dof / scalar_vsize;
    values[local_true] = component == 0 ? 1.0 : component == 1 ? 0.1 :
        component == 2 ? -0.02 : component == 3 ? 2.6 : 0.0;
  }
  return true;
}

bool FillInterior(const Layout& layout, mfem::Vector& current,
                  mfem::Vector& previous)
{
  current.SetSize(layout.InteriorLocalVSize());
  previous.SetSize(layout.InteriorLocalVSize());
  current = 0.0; previous = 0.0;
  for (int element = 0; element < layout.LocalElementCount(); ++element)
  {
    mfem::Vector current_element(28), previous_element(28);
    for (int component = 0; component < 7; ++component)
    {
      for (int dof = 0; dof < 4; ++dof)
      {
        const double spatial = 0.001 * static_cast<double>(element * 4 + dof);
        double value = component == 0 ? 1.0 + spatial : component == 1 ?
            0.12 + spatial : component == 2 ? -0.04 + spatial : component == 3 ?
            2.75 + spatial : component == 4 ? 0.86 + spatial :
            (component == 5 ? 0.015 : -0.009) + spatial;
        current_element[component * 4 + dof] = value;
        previous_element[component * 4 + dof] = value -
            (component < 4 ? 3.0e-4 * static_cast<double>(component + 1) : 0.0);
      }
    }
    if (!layout.ScatterInteriorElement(element, current_element, current) ||
        !layout.ScatterInteriorElement(element, previous_element, previous))
    {
      return false;
    }
  }
  return true;
}

bool FillEquilibriumInterior(const Layout& layout, mfem::Vector& values)
{
  values.SetSize(layout.InteriorLocalVSize());
  values = 0.0;
  for (int element = 0; element < layout.LocalElementCount(); ++element)
  {
    mfem::Vector element_values(28);
    element_values = 0.0;
    for (int dof = 0; dof < 4; ++dof)
    {
      element_values[dof] = 1.0;
      element_values[4 + dof] = 0.1;
      element_values[8 + dof] = -0.02;
      element_values[12 + dof] = 2.6;
    }
    if (!layout.ScatterInteriorElement(element, element_values, values))
    {
      return false;
    }
  }
  return true;
}

bool FillSource(const mfem::ParFiniteElementSpace& space, mfem::Vector& values,
                const double slope)
{
  if (space.GetVDim() != 4 || space.GetOrdering() != mfem::Ordering::byNODES ||
      values.Size() != space.GetVSize())
  {
    return false;
  }
  const int scalar_vsize = space.GetVSize() / 4;
  if (scalar_vsize <= 0) { return false; }
  for (int vdof = 0; vdof < values.Size(); ++vdof)
  {
    const int component = vdof / scalar_vsize;
    const int node = vdof % scalar_vsize;
    values[vdof] = (component == 3 ? 0.015 : 0.0) + slope *
        static_cast<double>((node % 4) - 1.5) * static_cast<double>(component + 1);
  }
  return true;
}

bool BuildAffineQuads(const mfem::ParMesh& mesh,
                      std::vector<local::AffineQuad>& quads)
{
  quads.resize(static_cast<std::size_t>(mesh.GetNE()));
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    mfem::Array<int> vertices;
    mesh.GetElementVertices(element, vertices);
    if (vertices.Size() != 4) { return false; }
    for (int corner = 0; corner < 4; ++corner)
    {
      const double* const point = mesh.GetVertex(vertices[corner]);
      if (point == nullptr) { return false; }
      quads[static_cast<std::size_t>(element)].vertices[
          static_cast<std::size_t>(corner)] = {{point[0], point[1]}};
    }
  }
  return true;
}

int DecodeSigned(const int signed_vdof, double& sign)
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  sign = signed_vdof < 0 ? -1.0 : 1.0;
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

bool SampleSource(const mfem::ParFiniteElementSpace& space,
                  const mfem::Vector& coefficients, const int element,
                  const mfem::IntegrationRule& rule, std::vector<double>& samples)
{
  samples.assign(static_cast<std::size_t>(rule.GetNPoints()) * 4, 0.0);
  mfem::Array<int> vdofs;
  space.GetElementVDofs(element, vdofs);
  const mfem::FiniteElement* const finite_element = space.GetFE(element);
  if (finite_element == nullptr || finite_element->GetDof() != 4 ||
      vdofs.Size() != 16) { return false; }
  mfem::Vector shape(4);
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    finite_element->CalcShape(rule.IntPoint(point), shape);
    for (int component = 0; component < 4; ++component)
    {
      for (int dof = 0; dof < 4; ++dof)
      {
        double sign = 1.0;
        const int vdof = DecodeSigned(vdofs[component * 4 + dof], sign);
        if (vdof < 0 || vdof >= coefficients.Size()) { return false; }
        samples[static_cast<std::size_t>(point) * 4 + component] +=
            shape[dof] * sign * coefficients[vdof];
      }
    }
  }
  for (const double value : samples)
  {
    if (!std::isfinite(value)) { return false; }
  }
  return true;
}

bool CheckLocalRecovery(const MPI_Comm communicator, const Layout& layout,
                        const std::vector<local::AffineQuad>& quads,
                        const mfem::ParFiniteElementSpace& source_space,
                        const mfem::Vector& source,
                        const detail::TracedScalarIgrP1GlobalStepOutput& output)
{
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (trace_space == nullptr) { return false; }
  mfem::Vector current_true(layout.TraceTrueVSize()), previous_true(layout.TraceTrueVSize());
  mfem::Vector current_interior, previous_interior, current_local(layout.TraceLocalVSize()),
      previous_local(layout.TraceLocalVSize()), local_increment(layout.TraceLocalVSize());
  mfem::Vector local_schur_residual(layout.TraceLocalVSize());
  local_schur_residual = 0.0;
  bool ok = FillTrueTrace(*trace_space, current_true, 2.0e-5) &&
      FillTrueTrace(*trace_space, previous_true, 0.0) &&
      FillInterior(layout, current_interior, previous_interior) &&
      layout.DistributeTrace(current_true, current_local) &&
      layout.DistributeTrace(previous_true, previous_local) &&
      layout.DistributeTrace(output.trace_increment, local_increment);
  if (!AllRanks(communicator, ok)) { return false; }
  const local::Parameters parameters{1.4, 0.8, 1.1, 0.05, 1.0e-12, 12};
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(mfem::Geometry::SQUARE, 12);
  for (int element = 0; element < layout.LocalElementCount() && ok; ++element)
  {
    mfem::Vector now(28), old(28), trace(40), old_trace(40), trace_increment(40),
        returned_interior(28), recovered_interior(28);
    std::vector<double> samples, schur(40 * 40), reduced(40);
    ok = layout.GatherInteriorElement(element, current_interior, now) &&
        layout.GatherInteriorElement(element, previous_interior, old) &&
        layout.GatherTraceElement(element, current_local, trace) &&
        layout.GatherTraceElement(element, previous_local, old_trace) &&
        layout.GatherTraceElement(element, local_increment, trace_increment) &&
        layout.GatherInteriorElement(element, output.interior_increment, returned_interior) &&
        SampleSource(source_space, source, element, rule, samples);
    local::LocalOperator operator_data;
    if (ok)
    {
      const local::ElementState state{now.HostRead(), old.HostRead(), trace.HostRead(),
          old_trace.HostRead(), {samples.data(), static_cast<std::size_t>(rule.GetNPoints())}};
      ok = local::Assemble(quads[static_cast<std::size_t>(element)], parameters, state,
                           operator_data) == local::Status::success;
    }
    dg::hdg::core::HostReferenceEngine condensation;
    auto mutable_view = [](double* values, const int rows, const int columns) {
      return dg::hdg::StridedBlockView<double>{values, 1, rows, columns,
                                                rows * columns, columns, 1};
    };
    if (ok)
    {
      ok = condensation.Condense(local::HostReferenceInput(operator_data),
          {mutable_view(schur.data(), 40, 40), mutable_view(reduced.data(), 40, 1)})
          == dg::hdg::core::Status::success;
    }
    if (ok)
    {
      const dg::hdg::StridedBlockView<const double> trace_view{
          trace_increment.HostRead(), 1, 40, 1, 40, 1, 1};
      ok = condensation.Recover(trace_view,
          mutable_view(recovered_interior.HostWrite(), 28, 1)) ==
          dg::hdg::core::Status::success;
    }
    mfem::Array<int> vdofs;
    trace_space->GetElementVDofs(element, vdofs);
    if (vdofs.Size() != 40) { ok = false; }
    for (int row = 0; row < 40 && ok; ++row)
    {
      double sign = 1.0;
      const int vdof = DecodeSigned(vdofs[row], sign);
      double action = -reduced[static_cast<std::size_t>(row)];
      for (int column = 0; column < 40; ++column)
      {
        action += schur[static_cast<std::size_t>(row) * 40 + column] *
            trace_increment[column];
      }
      ok = vdof >= 0 && vdof < local_schur_residual.Size() &&
          std::isfinite(action);
      if (ok) { local_schur_residual[vdof] += sign * action; }
    }
    for (int index = 0; index < 28 && ok; ++index)
    {
      ok = std::abs(recovered_interior[index] - returned_interior[index]) <=
          1.0e-11 * std::max({1.0, std::abs(recovered_interior[index]),
                               std::abs(returned_interior[index])});
    }
  }
  mfem::Vector true_schur_residual(layout.TraceTrueVSize());
  trace_space->Dof_TrueDof_Matrix()->MultTranspose(local_schur_residual,
                                                    true_schur_residual);
  const double oracle_norm = std::sqrt(NormSquared(communicator,
                                                   true_schur_residual));
  ok = ok && std::isfinite(oracle_norm) &&
      std::abs(oracle_norm - output.linear_residual_norm) <=
          1.0e-11 * std::max({1.0, oracle_norm, output.linear_residual_norm});
  return AllRanks(communicator, ok);
}

bool Run(const MPI_Comm communicator, const Layout& layout,
         const std::vector<local::AffineQuad>& quads,
         const mfem::ParFiniteElementSpace& source_space,
         const mfem::Vector& source, const double dt,
         detail::TracedScalarIgrP1GlobalStepOutput& output)
{
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (trace_space == nullptr) { return false; }
  mfem::Vector current_trace(layout.TraceTrueVSize());
  mfem::Vector previous_trace(layout.TraceTrueVSize());
  mfem::Vector current_interior;
  mfem::Vector previous_interior;
  bool ok = FillTrueTrace(*trace_space, current_trace, 2.0e-5) &&
      FillTrueTrace(*trace_space, previous_trace, 0.0) &&
      FillInterior(layout, current_interior, previous_interior);
  if (!AllRanks(communicator, ok)) { return false; }
  local::Parameters parameters{};
  parameters.dt = dt; parameters.alpha = 0.8; parameters.beta = 1.1;
  parameters.quadrature_order = 12;
  const detail::TracedScalarIgrP1GlobalStepInput input{
      communicator, &layout, quads.data(), static_cast<int>(quads.size()),
      parameters, &current_trace, &previous_trace, &current_interior,
      &previous_interior, &source_space, &source};
  return detail::ComputeTracedScalarIgrP1GlobalStep(input, output);
}

bool RunZeroReducedTraceRhs(
    const MPI_Comm communicator, const Layout& layout,
    const std::vector<local::AffineQuad>& quads,
    detail::TracedScalarIgrP1GlobalStepOutput& output)
{
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (trace_space == nullptr) { return false; }
  mfem::Vector trace(layout.TraceTrueVSize());
  mfem::Vector interior;
  bool ok = FillEquilibriumTrueTrace(*trace_space, trace) &&
      FillEquilibriumInterior(layout, interior);
  if (!AllRanks(communicator, ok)) { return false; }
  local::Parameters parameters{};
  parameters.alpha = 0.8; parameters.beta = 1.1; parameters.dt = 0.05;
  parameters.quadrature_order = 12;
  mfem::Vector previous_trace(trace);
  mfem::Vector previous_interior(interior);
  const detail::TracedScalarIgrP1GlobalStepInput distinct_input{
      communicator, &layout, quads.data(), static_cast<int>(quads.size()),
      parameters, &trace, &previous_trace, &interior, &previous_interior,
      nullptr, nullptr};
  return detail::ComputeTracedScalarIgrP1GlobalStep(distinct_input, output);
}

bool Check(const MPI_Comm communicator)
{
  mfem::ParMesh mesh = MakeMesh(communicator);
  Layout layout(mesh);
  mfem::L2_FECollection source_collection(
      1, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace source_space(&mesh, &source_collection, 4,
                                            mfem::Ordering::byNODES);
  std::vector<local::AffineQuad> quads;
  bool ok = layout.IsUsable() && BuildAffineQuads(mesh, quads);
  if (!Require(communicator, ok, "layout/geometry setup")) { return false; }

  mfem::Vector source_a(source_space.GetVSize());
  mfem::Vector source_b(source_space.GetVSize());
  ok = FillSource(source_space, source_a, 2.0e-3) &&
      FillSource(source_space, source_b, -2.0e-3);
  if (!Require(communicator, ok, "nonconstant source setup")) { return false; }
  mfem::Vector ones(source_space.GetVSize());
  ones = 1.0;
  const double mean_a = mfem::InnerProduct(communicator, source_a, ones);
  const double mean_b = mfem::InnerProduct(communicator, source_b, ones);
  ok = std::abs(mean_a - mean_b) < 1.0e-12;
  if (!Require(communicator, ok, "equal source means")) { return false; }

  detail::TracedScalarIgrP1GlobalStepOutput first, second, changed_dt;
  ok = Run(communicator, layout, quads, source_space, source_a, 0.05, first) &&
      Run(communicator, layout, quads, source_space, source_b, 0.05, second) &&
      Run(communicator, layout, quads, source_space, source_a, 0.08, changed_dt);
  if (!Require(communicator, ok, "nonconstant/dt global steps")) { return false; }
  const double trace_difference = NormSquared(communicator, first.trace_increment) +
      NormSquared(communicator, first.interior_increment);
  mfem::Vector source_delta(first.trace_increment);
  source_delta -= second.trace_increment;
  const double source_difference = NormSquared(communicator, source_delta);
  source_delta = first.trace_increment;
  source_delta -= changed_dt.trace_increment;
  const double dt_difference = NormSquared(communicator, source_delta);
  ok = Finite(first.trace_increment) && Finite(first.interior_increment) &&
      trace_difference > 1.0e-22 && source_difference > 1.0e-24 &&
      dt_difference > 1.0e-24 && std::isfinite(first.initial_residual_norm) &&
      std::isfinite(first.reduced_rhs_norm) && std::isfinite(first.linear_residual_norm) &&
      first.reduced_rhs_norm > 0.0 &&
      first.linear_residual_norm <= first.reduced_rhs_norm * 1.0e-10;
  if (!Require(communicator, ok, "increment/residual predicates")) { return false; }

  ok = CheckLocalRecovery(communicator, layout, quads, source_space, source_a,
                          first);
  if (!Require(communicator, ok, "local recovery/action oracle")) { return false; }

  detail::TracedScalarIgrP1GlobalStepOutput zero_rhs;
  ok = RunZeroReducedTraceRhs(communicator, layout, quads, zero_rhs);
  const double zero_trace_norm = NormSquared(communicator,
                                             zero_rhs.trace_increment);
  const double zero_interior_norm = NormSquared(communicator,
                                                zero_rhs.interior_increment);
  ok = ok && std::isfinite(zero_rhs.reduced_rhs_norm) &&
      zero_rhs.reduced_rhs_norm <= 1.0e-150 &&
      zero_rhs.linear_residual_norm <= 1.0e-150 &&
      zero_trace_norm <= 1.0e-300 && zero_interior_norm <= 1.0e-300;
  if (!Require(communicator, ok, "zero reduced trace rhs")) { return false; }

  detail::TracedScalarIgrP1GlobalStepOutput preserved;
  preserved.trace_increment.SetSize(1); preserved.trace_increment[0] = 73.0;
  preserved.interior_increment.SetSize(1); preserved.interior_increment[0] = -19.0;
  preserved.initial_residual_norm = 11.0;
  preserved.reduced_rhs_norm = 13.0;
  preserved.linear_residual_norm = 17.0;
  const detail::TracedScalarIgrP1GlobalStepInput invalid{
      communicator, &layout, quads.data(), static_cast<int>(quads.size()), {},
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  ok = !detail::ComputeTracedScalarIgrP1GlobalStep(invalid, preserved) &&
      preserved.trace_increment.Size() == 1 && preserved.trace_increment[0] == 73.0 &&
      preserved.interior_increment.Size() == 1 && preserved.interior_increment[0] == -19.0 &&
      preserved.initial_residual_norm == 11.0 &&
      preserved.reduced_rhs_norm == 13.0 && preserved.linear_residual_norm == 17.0;
  if (!Require(communicator, ok, "invalid input preservation")) { return false; }

  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  mfem::Vector aliased_trace(layout.TraceTrueVSize());
  mfem::Vector aliased_current_interior, aliased_previous_interior;
  ok = trace_space != nullptr && FillTrueTrace(*trace_space, aliased_trace, 0.0) &&
      FillInterior(layout, aliased_current_interior, aliased_previous_interior);
  if (!Require(communicator, ok, "trace alias setup")) { return false; }
  local::Parameters alias_parameters{};
  alias_parameters.dt = 0.05; alias_parameters.alpha = 0.8;
  alias_parameters.beta = 1.1; alias_parameters.quadrature_order = 12;
  const detail::TracedScalarIgrP1GlobalStepInput aliased_trace_input{
      communicator, &layout, quads.data(), static_cast<int>(quads.size()),
      alias_parameters, &aliased_trace, &aliased_trace,
      &aliased_current_interior, &aliased_previous_interior, nullptr, nullptr};
  ok = !detail::ComputeTracedScalarIgrP1GlobalStep(aliased_trace_input, preserved) &&
      preserved.trace_increment.Size() == 1 && preserved.trace_increment[0] == 73.0 &&
      preserved.interior_increment.Size() == 1 && preserved.interior_increment[0] == -19.0 &&
      preserved.initial_residual_norm == 11.0 &&
      preserved.reduced_rhs_norm == 13.0 && preserved.linear_residual_norm == 17.0;
  return Require(communicator, ok, "trace alias preservation");
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session session(argc, argv);
  return Check(MPI_COMM_WORLD) ? 0 : 1;
}
