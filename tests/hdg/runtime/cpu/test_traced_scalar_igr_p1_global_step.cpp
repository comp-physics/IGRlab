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

mfem::ParMesh MakePeriodicMesh(const MPI_Comm communicator)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(communicator, periodic);
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

bool FillSource(const mfem::ParFiniteElementSpace& space, mfem::Vector& values,
                const double slope)
{
  if (space.GetVDim() != 4 || space.GetOrdering() != mfem::Ordering::byNODES ||
      values.Size() != space.GetVSize())
  {
    return false;
  }
  const int scalar_vsize = space.GetVSize() / 4;
  if (scalar_vsize == 0) { return true; }
  for (int vdof = 0; vdof < values.Size(); ++vdof)
  {
    const int component = vdof / scalar_vsize;
    const int node = vdof % scalar_vsize;
    values[vdof] = (component == 3 ? 0.015 : 0.0) + slope *
        static_cast<double>((node % 4) - 1.5) * static_cast<double>(component + 1);
  }
  return true;
}

bool BuildAffineQuads(mfem::ParMesh& mesh,
                      std::vector<local::AffineQuad>& quads)
{
  quads.resize(static_cast<std::size_t>(mesh.GetNE()));
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    mfem::ElementTransformation* const transformation =
        mesh.GetElementTransformation(element);
    if (transformation == nullptr) { return false; }
    constexpr double corners[4][2]{{0.0, 0.0}, {1.0, 0.0},
                                   {1.0, 1.0}, {0.0, 1.0}};
    for (int corner = 0; corner < 4; ++corner)
    {
      mfem::IntegrationPoint reference;
      reference.Set2(corners[corner][0], corners[corner][1]);
      mfem::Vector point(2);
      transformation->Transform(reference, point);
      if (point.Size() != 2 || !std::isfinite(point[0]) ||
          !std::isfinite(point[1]))
      {
        return false;
      }
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
  if (coefficients.Size() != space.GetVSize()) { return false; }
  mfem::ParGridFunction source_field(
      const_cast<mfem::ParFiniteElementSpace*>(&space));
  source_field = coefficients;
  mfem::Vector value(4);
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    source_field.GetVectorValue(element, rule.IntPoint(point), value);
    if (value.Size() != 4) { return false; }
    for (int component = 0; component < 4; ++component)
    {
      samples[static_cast<std::size_t>(point) * 4 + component] = value[component];
    }
  }
  for (const double value : samples)
  {
    if (!std::isfinite(value)) { return false; }
  }
  return true;
}

struct RawUncondensedResidual final
{
  double norm = 0.0;
  double element_trace_norm_squared = 0.0;
  double true_trace_norm_squared = 0.0;
};

bool ComputeRawUncondensedResidual(
    const MPI_Comm communicator, const Layout& layout,
    const std::vector<local::AffineQuad>& quads,
    const mfem::ParFiniteElementSpace* const source_space,
    const mfem::Vector* const source, const local::Parameters& parameters,
    const mfem::Vector& current_trace, const mfem::Vector& previous_trace,
    const mfem::Vector& current_interior, const mfem::Vector& previous_interior,
    RawUncondensedResidual& output)
{
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (trace_space == nullptr || static_cast<int>(quads.size()) !=
      layout.LocalElementCount())
  {
    return false;
  }
  mfem::Vector current_local(layout.TraceLocalVSize()),
      previous_local(layout.TraceLocalVSize()), local_raw(layout.TraceLocalVSize()),
      true_raw(layout.TraceTrueVSize());
  bool ok = layout.DistributeTrace(current_trace, current_local) &&
      layout.DistributeTrace(previous_trace, previous_local);
  if (!AllRanks(communicator, ok)) { return false; }
  local_raw = 0.0; true_raw = 0.0;
  std::vector<double> interior;
  double local_element_trace_sum = 0.0;
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(
      mfem::Geometry::SQUARE, parameters.quadrature_order);
  for (int element = 0; element < layout.LocalElementCount() && ok; ++element)
  {
    mfem::Vector now(28), old(28), trace(40), old_trace(40);
    std::vector<double> samples;
    ok = layout.GatherInteriorElement(element, current_interior, now) &&
        layout.GatherInteriorElement(element, previous_interior, old) &&
        layout.GatherTraceElement(element, current_local, trace) &&
        layout.GatherTraceElement(element, previous_local, old_trace);
    if (ok && source != nullptr)
    {
      ok = source_space != nullptr && SampleSource(*source_space, *source,
                                                    element, rule, samples);
    }
    local::LocalOperator operator_data;
    if (ok)
    {
      const local::ElementState state{now.HostRead(), old.HostRead(),
          trace.HostRead(), old_trace.HostRead(),
          {samples.empty() ? nullptr : samples.data(),
           samples.empty() ? 0U : static_cast<std::size_t>(rule.GetNPoints())}};
      ok = local::Assemble(quads[static_cast<std::size_t>(element)], parameters,
                           state, operator_data) == local::Status::success;
    }
    mfem::Array<int> vdofs;
    if (ok) { trace_space->GetElementVDofs(element, vdofs); ok = vdofs.Size() == 40; }
    for (int row = 0; row < 40 && ok; ++row)
    {
      double sign = 1.0;
      const int vdof = DecodeSigned(vdofs[row], sign);
      ok = vdof >= 0 && vdof < local_raw.Size();
      if (ok) { local_raw[vdof] += sign * operator_data.g[row]; }
    }
    if (ok)
    {
      interior.insert(interior.end(), operator_data.f.begin(), operator_data.f.end());
      // Preserve the per-element raw trace residual before face assembly.
      for (const double value : operator_data.g)
      {
        local_element_trace_sum += value * value;
      }
    }
  }
  if (!AllRanks(communicator, ok) || !Finite(local_raw)) { return false; }
  trace_space->Dof_TrueDof_Matrix()->MultTranspose(local_raw, true_raw);
  if (!AllRanks(communicator, Finite(true_raw))) { return false; }

  double local_maximum = 0.0;
  double local_sum = 0.0;
  for (const double value : interior) { local_maximum = std::max(local_maximum, std::abs(value)); }
  for (int index = 0; index < true_raw.Size(); ++index)
  {
    local_maximum = std::max(local_maximum, std::abs(true_raw[index]));
  }
  double maximum = 0.0;
  if (MPI_Allreduce(&local_maximum, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                    communicator) != MPI_SUCCESS || !std::isfinite(maximum))
  {
    return false;
  }
  if (maximum > 0.0)
  {
    for (const double value : interior)
    {
      const double scaled = value / maximum; local_sum += scaled * scaled;
    }
    for (int index = 0; index < true_raw.Size(); ++index)
    {
      const double scaled = true_raw[index] / maximum; local_sum += scaled * scaled;
    }
  }
  double global_sum = 0.0;
  if (MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    communicator) != MPI_SUCCESS || !std::isfinite(global_sum))
  {
    return false;
  }
  output.norm = maximum == 0.0 ? 0.0 : maximum * std::sqrt(global_sum);
  output.true_trace_norm_squared = NormSquared(communicator, true_raw);
  if (!std::isfinite(local_element_trace_sum) ||
      MPI_Allreduce(&local_element_trace_sum, &output.element_trace_norm_squared,
                    1, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS)
  {
    return false;
  }
  return std::isfinite(output.norm) &&
      std::isfinite(output.element_trace_norm_squared) &&
      std::isfinite(output.true_trace_norm_squared);
}

bool FillUniformTrace(mfem::ParMesh& mesh,
                      mfem::ParFiniteElementSpace& space,
                      mfem::Vector& values)
{
  if (values.Size() != space.GetTrueVSize()) { return false; }
  const double state[5]{1.0, 0.12, -0.04, 2.75, 0.86};
  mfem::ParGridFunction local(&space);
  local = 0.0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element = space.GetFaceElement(face);
    mfem::Array<int> vdofs;
    space.GetFaceVDofs(face, vdofs);
    if (finite_element == nullptr || finite_element->GetDof() <= 0 ||
        vdofs.Size() != finite_element->GetDof() * 5)
    {
      return false;
    }
    for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
    {
      for (int component = 0; component < 5; ++component)
      {
        double sign = 1.0;
        const int vdof = DecodeSigned(
            vdofs[component * finite_element->GetDof() + scalar], sign);
        if (vdof < 0 || vdof >= local.Size()) { return false; }
        local[vdof] = sign * state[component];
      }
    }
  }
  local.ParallelProject(values);
  local.Distribute(values);
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    const mfem::FiniteElement* const finite_element = space.GetFaceElement(face);
    mfem::Array<int> vdofs;
    space.GetFaceVDofs(face, vdofs);
    if (finite_element == nullptr || vdofs.Size() != finite_element->GetDof() * 5)
    {
      return false;
    }
    for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
    {
      for (int component = 0; component < 5; ++component)
      {
        double sign = 1.0;
        const int vdof = DecodeSigned(
            vdofs[component * finite_element->GetDof() + scalar], sign);
        if (vdof < 0 || vdof >= local.Size() ||
            std::abs(sign * local[vdof] - state[component]) > 1.0e-14)
        {
          return false;
        }
      }
    }
  }
  return Finite(values);
}

bool FillUniformInterior(const Layout& layout, mfem::Vector& values)
{
  values.SetSize(layout.InteriorLocalVSize()); values = 0.0;
  const double state[7]{1.0, 0.12, -0.04, 2.75, 0.86, 0.0, 0.0};
  for (int element = 0; element < layout.LocalElementCount(); ++element)
  {
    mfem::Vector local_values(28);
    for (int component = 0; component < 7; ++component)
    {
      for (int dof = 0; dof < 4; ++dof) { local_values[component * 4 + dof] = state[component]; }
    }
    if (!layout.ScatterInteriorElement(element, local_values, values)) { return false; }
  }
  return true;
}

bool CheckUniformTraceGather(const Layout& layout, const mfem::Vector& values)
{
  const double state[5]{1.0, 0.12, -0.04, 2.75, 0.86};
  for (int element = 0; element < layout.LocalElementCount(); ++element)
  {
    mfem::Vector gathered(40);
    if (!layout.GatherTraceElement(element, values, gathered)) { return false; }
    for (int component = 0; component < 5; ++component)
    {
      for (int face_dof = 0; face_dof < 8; ++face_dof)
      {
        if (std::abs(gathered[component * 8 + face_dof] - state[component]) >
            1.0e-14)
        {
          return false;
        }
      }
    }
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

bool CheckExactZeroReducedRhsRecovery()
{
  constexpr int interior = 28;
  constexpr int trace = 40;
  std::vector<double> a(static_cast<std::size_t>(interior) * interior, 0.0);
  std::vector<double> b(static_cast<std::size_t>(interior) * trace, 0.0);
  std::vector<double> c(static_cast<std::size_t>(trace) * interior, 0.0);
  std::vector<double> d(static_cast<std::size_t>(trace) * trace, 0.0);
  std::vector<double> f(interior), g(trace, 0.0), schur(trace * trace),
      reduced(trace), recovered(interior), zero_trace(trace, 0.0);
  for (int index = 0; index < interior; ++index)
  {
    a[static_cast<std::size_t>(index) * interior + index] = 1.0;
    f[static_cast<std::size_t>(index)] = 0.125 * static_cast<double>(index + 1);
  }
  for (int index = 0; index < trace; ++index)
  {
    d[static_cast<std::size_t>(index) * trace + index] = 1.0;
  }
  auto constant = [](const std::vector<double>& values, const int rows,
                     const int columns) {
    return dg::hdg::StridedBlockView<const double>{values.data(), 1, rows,
        columns, rows * columns, columns, 1};
  };
  auto writable = [](std::vector<double>& values, const int rows,
                     const int columns) {
    return dg::hdg::StridedBlockView<double>{values.data(), 1, rows, columns,
        rows * columns, columns, 1};
  };
  dg::hdg::core::HostReferenceEngine condensation;
  const dg::hdg::core::LocalSystemBatchView system{
      {4, 8, 7, 5}, constant(a, interior, interior),
      constant(b, interior, trace), constant(c, trace, interior),
      constant(d, trace, trace), constant(f, interior, 1), constant(g, trace, 1)};
  if (condensation.Condense(system, {writable(schur, trace, trace),
                                     writable(reduced, trace, 1)}) !=
      dg::hdg::core::Status::success)
  {
    return false;
  }
  if (std::any_of(reduced.begin(), reduced.end(),
                  [](const double value) { return value != 0.0; }))
  {
    return false;
  }
  const dg::hdg::StridedBlockView<const double> trace_view{
      zero_trace.data(), 1, trace, 1, trace, 1, 1};
  if (condensation.Recover(trace_view, writable(recovered, interior, 1)) !=
      dg::hdg::core::Status::success)
  {
    return false;
  }
  return recovered == f;
}

bool CheckTraceSolveRecoverySeam(const MPI_Comm communicator, const Layout& layout)
{
  const mfem::ParFiniteElementSpace* const space = layout.TraceSpace();
  const int elements = layout.LocalElementCount();
  if (space == nullptr) { return false; }
  std::vector<double> a(static_cast<std::size_t>(elements) * 28 * 28, 0.0);
  std::vector<double> b(static_cast<std::size_t>(elements) * 28 * 40, 0.0);
  std::vector<double> c(static_cast<std::size_t>(elements) * 40 * 28, 0.0);
  std::vector<double> d(static_cast<std::size_t>(elements) * 40 * 40, 0.0);
  std::vector<double> f(static_cast<std::size_t>(elements) * 28, 0.0);
  std::vector<double> g(static_cast<std::size_t>(elements) * 40, 0.0);
  for (int element = 0; element < elements; ++element)
  {
    for (int i = 0; i < 28; ++i)
    {
      a[static_cast<std::size_t>(element) * 28 * 28 + i * 28 + i] = 1.0;
      f[static_cast<std::size_t>(element) * 28 + i] = 0.125 * (i + 1);
    }
    for (int i = 0; i < 40; ++i)
    {
      d[static_cast<std::size_t>(element) * 40 * 40 + i * 40 + i] = 1.0;
    }
  }
  auto cv = [](const std::vector<double>& values, int rows, int cols, int batches) {
    return dg::hdg::StridedBlockView<const double>{values.data(), batches, rows,
        cols, rows * cols, cols, 1};
  };
  auto v = [](std::vector<double>& values, int rows, int cols, int batches) {
    return dg::hdg::StridedBlockView<double>{values.data(), batches, rows, cols,
        rows * cols, cols, 1};
  };
  std::vector<double> schur(static_cast<std::size_t>(elements) * 40 * 40);
  std::vector<double> reduced(static_cast<std::size_t>(elements) * 40);
  dg::hdg::core::HostReferenceEngine condensation;
  bool ok = elements == 0 || condensation.Condense({{4,8,7,5},cv(a,28,28,elements),cv(b,28,40,elements),cv(c,40,28,elements),cv(d,40,40,elements),cv(f,28,1,elements),cv(g,40,1,elements)}, {v(schur,40,40,elements),v(reduced,40,1,elements)}) == dg::hdg::core::Status::success;
  if (!AllRanks(communicator, ok)) { return false; }
  mfem::SparseMatrix local_matrix(layout.TraceLocalVSize());
  for (int element = 0; element < elements; ++element)
  {
    mfem::Array<int> vdofs; space->GetElementVDofs(element, vdofs);
    mfem::DenseMatrix identity(40); identity = 0.0;
    for (int i = 0; i < 40; ++i) { identity(i,i) = 1.0; }
    local_matrix.AddSubMatrix(vdofs, vdofs, identity);
  }
  local_matrix.Finalize();
  mfem::ParBilinearForm form(const_cast<mfem::ParFiniteElementSpace*>(space));
  std::unique_ptr<mfem::HypreParMatrix> matrix(form.ParallelAssemble(&local_matrix));
  mfem::Vector zero(layout.TraceTrueVSize()); zero = 0.0;
  detail::TracedScalarIgrP1TraceSolveOutput exact;
  const detail::TracedScalarIgrP1TraceSolveInput exact_input{communicator,&layout,space,matrix.get(),&zero,&condensation};
  ok = matrix != nullptr && detail::SolveAndRecoverTracedScalarIgrP1Trace(exact_input, exact) &&
      NormSquared(communicator, exact.trace_increment) == 0.0 &&
      NormSquared(communicator, exact.interior_increment) > 0.0;
  if (!AllRanks(communicator, ok)) { return false; }
  mfem::Vector tiny(layout.TraceTrueVSize()); tiny = 1.0e-151;
  detail::TracedScalarIgrP1TraceSolveOutput tiny_output;
  const detail::TracedScalarIgrP1TraceSolveInput tiny_input{communicator,&layout,space,matrix.get(),&tiny,&condensation};
  ok = detail::SolveAndRecoverTracedScalarIgrP1Trace(tiny_input, tiny_output) &&
      NormSquared(communicator, tiny_output.trace_increment) > 0.0 &&
      Finite(tiny_output.trace_increment) && Finite(tiny_output.interior_increment);
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
  return detail::ComputeTracedScalarIgrP1GlobalStep(input, output) ==
      detail::TracedScalarIgrP1GlobalStepResult::success;
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

  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  mfem::Vector raw_current_trace(layout.TraceTrueVSize()),
      raw_previous_trace(layout.TraceTrueVSize()), raw_current_interior,
      raw_previous_interior;
  local::Parameters raw_parameters{};
  raw_parameters.dt = 0.05; raw_parameters.alpha = 0.8;
  raw_parameters.beta = 1.1; raw_parameters.quadrature_order = 12;
  RawUncondensedResidual raw_nonconstant;
  ok = trace_space != nullptr &&
      FillTrueTrace(*trace_space, raw_current_trace, 2.0e-5) &&
      FillTrueTrace(*trace_space, raw_previous_trace, 0.0) &&
      FillInterior(layout, raw_current_interior, raw_previous_interior) &&
      ComputeRawUncondensedResidual(communicator, layout, quads, &source_space,
          &source_a, raw_parameters, raw_current_trace, raw_previous_trace,
          raw_current_interior, raw_previous_interior, raw_nonconstant) &&
      std::abs(raw_nonconstant.norm - first.initial_residual_norm) <=
          1.0e-11 * std::max({1.0, raw_nonconstant.norm,
                               first.initial_residual_norm});
  if (!Require(communicator, ok, "independent [f;T^T g] initial norm"))
  {
    return false;
  }

  mfem::ParMesh periodic_mesh = MakePeriodicMesh(communicator);
  Layout periodic_layout(periodic_mesh);
  std::vector<local::AffineQuad> periodic_quads;
  mfem::ParFiniteElementSpace* const periodic_trace =
      const_cast<mfem::ParFiniteElementSpace*>(periodic_layout.TraceSpace());
  mfem::Vector periodic_current_trace(periodic_layout.TraceTrueVSize()),
      periodic_previous_trace(periodic_layout.TraceTrueVSize()),
      periodic_current_interior, periodic_previous_interior;
  RawUncondensedResidual periodic_raw;
  ok = periodic_layout.IsUsable() && periodic_trace != nullptr &&
      BuildAffineQuads(periodic_mesh, periodic_quads);
  if (!Require(communicator, ok, "periodic layout/geometry setup")) { return false; }
  ok = FillUniformTrace(periodic_mesh, *periodic_trace, periodic_current_trace) &&
      FillUniformTrace(periodic_mesh, *periodic_trace, periodic_previous_trace) &&
      CheckUniformTraceGather(periodic_layout, periodic_current_trace) &&
      CheckUniformTraceGather(periodic_layout, periodic_previous_trace) &&
      FillUniformInterior(periodic_layout, periodic_current_interior) &&
      FillUniformInterior(periodic_layout, periodic_previous_interior);
  if (!Require(communicator, ok, "periodic oriented trace setup")) { return false; }
  ok = ComputeRawUncondensedResidual(communicator, periodic_layout, periodic_quads,
      nullptr, nullptr, raw_parameters, periodic_current_trace,
      periodic_previous_trace, periodic_current_interior,
      periodic_previous_interior, periodic_raw);
  if (!Require(communicator, ok, "periodic raw residual setup")) { return false; }
  ok = periodic_raw.element_trace_norm_squared > 0.0 &&
      periodic_raw.true_trace_norm_squared <= 1.0e-26 *
          periodic_raw.element_trace_norm_squared;
  if (!Require(communicator, ok, "periodic raw trace cancellation")) { return false; }

  ok = CheckLocalRecovery(communicator, layout, quads, source_space, source_a,
                          first);
  if (!Require(communicator, ok, "local recovery/action oracle")) { return false; }

  ok = CheckExactZeroReducedRhsRecovery();
  if (!Require(communicator, ok, "exact zero reduced rhs recovery")) { return false; }

  ok = CheckTraceSolveRecoverySeam(communicator, layout);
  if (!Require(communicator, ok, "trace solve/recovery seam")) { return false; }

  detail::TracedScalarIgrP1GlobalStepOutput preserved;
  preserved.trace_increment.SetSize(1); preserved.trace_increment[0] = 73.0;
  preserved.interior_increment.SetSize(1); preserved.interior_increment[0] = -19.0;
  preserved.initial_residual_norm = 11.0;
  preserved.reduced_rhs_norm = 13.0;
  preserved.linear_residual_norm = 17.0;
  const detail::TracedScalarIgrP1GlobalStepInput invalid{
      communicator, &layout, quads.data(), static_cast<int>(quads.size()), {},
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  ok = detail::ComputeTracedScalarIgrP1GlobalStep(invalid, preserved) ==
          detail::TracedScalarIgrP1GlobalStepResult::failure &&
      preserved.trace_increment.Size() == 1 && preserved.trace_increment[0] == 73.0 &&
      preserved.interior_increment.Size() == 1 && preserved.interior_increment[0] == -19.0 &&
      preserved.initial_residual_norm == 11.0 &&
      preserved.reduced_rhs_norm == 13.0 && preserved.linear_residual_norm == 17.0;
  if (!Require(communicator, ok, "invalid input preservation")) { return false; }

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
  ok = detail::ComputeTracedScalarIgrP1GlobalStep(aliased_trace_input, preserved) ==
          detail::TracedScalarIgrP1GlobalStepResult::failure &&
      preserved.trace_increment.Size() == 1 && preserved.trace_increment[0] == 73.0 &&
      preserved.interior_increment.Size() == 1 && preserved.interior_increment[0] == -19.0 &&
      preserved.initial_residual_norm == 11.0 &&
      preserved.reduced_rhs_norm == 13.0 && preserved.linear_residual_norm == 17.0;
  if (!Require(communicator, ok, "trace alias preservation")) { return false; }

  mfem::Vector classified_current_trace(layout.TraceTrueVSize()),
      classified_previous_trace(layout.TraceTrueVSize());
  ok = trace_space != nullptr &&
      FillTrueTrace(*trace_space, classified_current_trace, 0.0) &&
      FillTrueTrace(*trace_space, classified_previous_trace, 0.0);
  if (!Require(communicator, ok, "classified-result trace setup")) { return false; }
  mfem::Vector inadmissible_interior(aliased_current_interior);
  if (layout.LocalElementCount() > 0)
  {
    mfem::Vector first_element(28);
    ok = layout.GatherInteriorElement(0, inadmissible_interior, first_element);
    for (int dof = 0; dof < 4 && ok; ++dof) { first_element[dof] = 0.0; }
    if (ok)
    {
      ok = layout.ScatterInteriorElement(0, first_element, inadmissible_interior);
    }
  }
  if (!Require(communicator, ok, "inadmissible trial setup")) { return false; }
  const detail::TracedScalarIgrP1GlobalStepInput inadmissible_input{
      communicator, &layout, quads.data(), static_cast<int>(quads.size()),
      alias_parameters, &classified_current_trace, &classified_previous_trace,
      &inadmissible_interior, &aliased_previous_interior, nullptr, nullptr};
  ok = detail::ComputeTracedScalarIgrP1GlobalStep(inadmissible_input, preserved) ==
          detail::TracedScalarIgrP1GlobalStepResult::inadmissible_trial &&
      preserved.trace_increment.Size() == 1 && preserved.trace_increment[0] == 73.0 &&
      preserved.interior_increment.Size() == 1 && preserved.interior_increment[0] == -19.0 &&
      preserved.initial_residual_norm == 11.0 &&
      preserved.reduced_rhs_norm == 13.0 && preserved.linear_residual_norm == 17.0;
  if (!Require(communicator, ok, "inadmissible trial preservation")) { return false; }

  std::vector<local::AffineQuad> structurally_invalid_quads = quads;
  if (!structurally_invalid_quads.empty())
  {
    structurally_invalid_quads.front().vertices[2] =
        structurally_invalid_quads.front().vertices[1];
  }
  const detail::TracedScalarIgrP1GlobalStepInput structural_input{
      communicator, &layout, structurally_invalid_quads.data(),
      static_cast<int>(structurally_invalid_quads.size()), alias_parameters,
      &classified_current_trace, &classified_previous_trace, &aliased_current_interior,
      &aliased_previous_interior, nullptr, nullptr};
  ok = detail::ComputeTracedScalarIgrP1GlobalStep(structural_input, preserved) ==
          detail::TracedScalarIgrP1GlobalStepResult::failure &&
      preserved.trace_increment.Size() == 1 && preserved.trace_increment[0] == 73.0 &&
      preserved.interior_increment.Size() == 1 && preserved.interior_increment[0] == -19.0 &&
      preserved.initial_residual_norm == 11.0 &&
      preserved.reduced_rhs_norm == 13.0 && preserved.linear_residual_norm == 17.0;
  return Require(communicator, ok, "structural failure preservation");
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session session(argc, argv);
  return Check(MPI_COMM_WORLD) ? 0 : 1;
}
