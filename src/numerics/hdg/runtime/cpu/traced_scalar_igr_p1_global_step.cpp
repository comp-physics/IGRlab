#include "numerics/hdg/runtime/cpu/detail/traced_scalar_igr_p1_global_step.hpp"

#include "numerics/hdg/core/host_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu::detail
{
namespace
{
namespace local = traced_scalar_igr_p1;

bool Finite(const double value) noexcept { return std::isfinite(value); }

bool AllFinite(const mfem::Vector& values) noexcept
{
  if (values.Size() == 0) { return true; }
  const double* const data = values.HostRead();
  return data != nullptr && std::all_of(data, data + values.Size(), Finite);
}

bool AllFinite(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(), Finite);
}

bool CollectiveAll(const MPI_Comm communicator, const bool local_ok) noexcept
{
  int source = local_ok ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

bool CopyVector(const mfem::Vector& source, mfem::Vector& destination) noexcept
{
  if (source.Size() != destination.Size()) { return false; }
  const double* const from = source.HostRead();
  double* const to = destination.HostWrite();
  if ((from == nullptr || to == nullptr) && source.Size() != 0) { return false; }
  if (source.Size() != 0)
  {
    std::memcpy(to, from, static_cast<std::size_t>(source.Size()) * sizeof(double));
  }
  return true;
}

int DecodeSignedVDof(const int signed_vdof, double& sign) noexcept
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  sign = signed_vdof < 0 ? -1.0 : 1.0;
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

template <class T>
StridedBlockView<const T> ConstView(const std::vector<T>& values,
                                    const int batches, const int rows,
                                    const int columns) noexcept
{
  return {values.data(), batches, rows, columns, rows * columns, columns, 1};
}

template <class T>
StridedBlockView<T> View(std::vector<T>& values, const int batches,
                         const int rows, const int columns) noexcept
{
  return {values.data(), batches, rows, columns, rows * columns, columns, 1};
}

bool SourceIsZero(const TracedScalarIgrP1GlobalStepInput& input) noexcept
{
  return input.static_euler_rhs_coefficients == nullptr ||
      input.static_euler_rhs_coefficients->Size() == 0;
}

bool ValidParameters(const local::Parameters& parameters) noexcept
{
  return Finite(parameters.gamma) && parameters.gamma > 1.0 &&
      Finite(parameters.alpha) && parameters.alpha > 0.0 &&
      Finite(parameters.beta) && parameters.beta > 0.0 &&
      Finite(parameters.dt) && parameters.dt > 0.0 &&
      Finite(parameters.density_floor) && parameters.density_floor > 0.0 &&
      parameters.quadrature_order >= 2 && parameters.quadrature_order <= 32;
}

bool ValidInput(const TracedScalarIgrP1GlobalStepInput& input) noexcept
{
  if (input.communicator == MPI_COMM_NULL || input.layout == nullptr ||
      !input.layout->IsUsable() || input.current_true_trace == nullptr ||
      input.previous_true_trace == nullptr || input.current_interior == nullptr ||
      input.previous_interior == nullptr ||
      input.current_interior == input.previous_interior ||
      input.current_true_trace->Size() != input.layout->TraceTrueVSize() ||
      input.previous_true_trace->Size() != input.layout->TraceTrueVSize() ||
      input.current_interior->Size() != input.layout->InteriorLocalVSize() ||
      input.previous_interior->Size() != input.layout->InteriorLocalVSize() ||
      !AllFinite(*input.current_true_trace) ||
      !AllFinite(*input.previous_true_trace) || !AllFinite(*input.current_interior) ||
      !AllFinite(*input.previous_interior) ||
      !ValidParameters(input.parameters) ||
      input.affine_quad_count != input.layout->LocalElementCount() ||
      (input.affine_quad_count > 0 && input.affine_quads == nullptr))
  {
    return false;
  }
  if (SourceIsZero(input)) { return true; }
  const mfem::ParFiniteElementSpace* const source_space =
      input.static_euler_rhs_space;
  const mfem::Vector& source = *input.static_euler_rhs_coefficients;
  const mfem::ParFiniteElementSpace* const interior_space =
      input.layout->InteriorSpace();
  return source_space != nullptr && interior_space != nullptr &&
      source_space->GetParMesh() == interior_space->GetParMesh() &&
      source_space->GetVDim() == 4 &&
      source_space->GetOrdering() == mfem::Ordering::byNODES &&
      source.Size() == source_space->GetVSize() && AllFinite(source);
}

bool FillElementSource(const mfem::ParFiniteElementSpace* source_space,
                       const mfem::Vector* source_coefficients,
                       const int element, const mfem::IntegrationRule& rule,
                       double* samples) noexcept
{
  if (source_coefficients == nullptr || source_coefficients->Size() == 0)
  {
    std::fill(samples, samples + static_cast<std::size_t>(rule.GetNPoints()) * 4,
              0.0);
    return true;
  }
  if (source_space == nullptr || samples == nullptr ||
      source_space->GetOrdering() != mfem::Ordering::byNODES)
  {
    return false;
  }
  mfem::Array<int> vdofs;
  source_space->GetElementVDofs(element, vdofs);
  const mfem::FiniteElement* const finite_element = source_space->GetFE(element);
  if (finite_element == nullptr || finite_element->GetDof() != 4 ||
      vdofs.Size() != 16)
  {
    return false;
  }
  const double* const coefficients = source_coefficients->HostRead();
  const int* const signed_vdofs = vdofs.HostRead();
  if (coefficients == nullptr || signed_vdofs == nullptr) { return false; }
  mfem::Vector shape(4);
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    finite_element->CalcShape(rule.IntPoint(point), shape);
    for (int component = 0; component < 4; ++component)
    {
      double value = 0.0;
      for (int dof = 0; dof < 4; ++dof)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(signed_vdofs[dof * 4 + component], sign);
        if (vdof < 0 || vdof >= source_coefficients->Size()) { return false; }
        value += shape[dof] * sign * coefficients[vdof];
      }
      if (!Finite(value)) { return false; }
      samples[static_cast<std::size_t>(point) * 4 + component] = value;
    }
  }
  return true;
}

bool GlobalNorm(const MPI_Comm communicator, const double local_squared,
                double& norm) noexcept
{
  if (!Finite(local_squared) || local_squared < 0.0) { return false; }
  double global_squared = 0.0;
  if (MPI_Allreduce(&local_squared, &global_squared, 1, MPI_DOUBLE, MPI_SUM,
                    communicator) != MPI_SUCCESS ||
      !Finite(global_squared) || global_squared < 0.0)
  {
    return false;
  }
  norm = std::sqrt(global_squared);
  return Finite(norm);
}

} // namespace

bool ComputeTracedScalarIgrP1GlobalStep(
    const TracedScalarIgrP1GlobalStepInput& input,
    TracedScalarIgrP1GlobalStepOutput& output) noexcept
{
  // The communicator has been admitted by the caller.  This initial local
  // check is deliberately before every opaque collective phase.
  if (input.communicator == MPI_COMM_NULL) { return false; }
  if (!CollectiveAll(input.communicator, ValidInput(input))) { return false; }

  const BorrowedP1DGInterfaceLayout& layout = *input.layout;
  const int elements = layout.LocalElementCount();
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (!CollectiveAll(input.communicator, trace_space != nullptr)) { return false; }

  mfem::Vector current_local_trace;
  mfem::Vector previous_local_trace;
  std::vector<double> a, b, c, d, f, g, source_samples;
  std::vector<double> schur, reduced, recovered;
  double local_initial_squared = 0.0;
  bool local_ok = true;
  try
  {
    current_local_trace.SetSize(layout.TraceLocalVSize());
    previous_local_trace.SetSize(layout.TraceLocalVSize());
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, input.parameters.quadrature_order);
    const std::size_t count = static_cast<std::size_t>(elements);
    const std::size_t points = static_cast<std::size_t>(volume_rule.GetNPoints());
    local_ok = volume_rule.GetNPoints() > 0 &&
        count <= std::numeric_limits<std::size_t>::max() / (28 * 28) &&
        count <= std::numeric_limits<std::size_t>::max() / (40 * 40) &&
        count <= std::numeric_limits<std::size_t>::max() / (points * 4);
    if (local_ok)
    {
      a.reserve(count * 28 * 28); b.reserve(count * 28 * 40);
      c.reserve(count * 40 * 28); d.reserve(count * 40 * 40);
      f.reserve(count * 28); g.reserve(count * 40);
      source_samples.resize(count * points * 4, 0.0);
      schur.resize(count * 40 * 40); reduced.resize(count * 40);
      recovered.resize(count * 28);
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  // Both trace distributions are collective through the existing exact-space
  // exchange plan, so all local allocation/shape checks have happened above.
  local_ok = layout.DistributeTrace(*input.current_true_trace,
                                    current_local_trace) &&
      layout.DistributeTrace(*input.previous_true_trace, previous_local_trace);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  try
  {
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, input.parameters.quadrature_order);
    for (int element = 0; element < elements && local_ok; ++element)
    {
      mfem::Vector current_interior(28), previous_interior(28), trace(40),
          previous_trace(40);
      local_ok = layout.GatherInteriorElement(element, *input.current_interior,
                                              current_interior) &&
          layout.GatherInteriorElement(element, *input.previous_interior,
                                      previous_interior) &&
          layout.GatherTraceElement(element, current_local_trace, trace) &&
          layout.GatherTraceElement(element, previous_local_trace, previous_trace) &&
          FillElementSource(input.static_euler_rhs_space,
                            input.static_euler_rhs_coefficients, element,
                            volume_rule,
                            source_samples.data() +
                                static_cast<std::size_t>(element) *
                                    static_cast<std::size_t>(volume_rule.GetNPoints()) * 4);
      local::LocalOperator operator_data;
      if (local_ok)
      {
        const local::ElementState state{
            current_interior.HostRead(), previous_interior.HostRead(),
            trace.HostRead(), previous_trace.HostRead(),
            {source_samples.data() + static_cast<std::size_t>(element) *
                 static_cast<std::size_t>(volume_rule.GetNPoints()) * 4,
             static_cast<std::size_t>(volume_rule.GetNPoints())}};
        local_ok = local::Assemble(input.affine_quads[element], input.parameters,
                                   state, operator_data) == local::Status::success;
      }
      if (local_ok)
      {
        a.insert(a.end(), operator_data.A.begin(), operator_data.A.end());
        b.insert(b.end(), operator_data.B.begin(), operator_data.B.end());
        c.insert(c.end(), operator_data.C.begin(), operator_data.C.end());
        d.insert(d.end(), operator_data.D.begin(), operator_data.D.end());
        f.insert(f.end(), operator_data.f.begin(), operator_data.f.end());
        g.insert(g.end(), operator_data.g.begin(), operator_data.g.end());
        for (const double value : operator_data.interior_residual)
        {
          local_initial_squared += value * value;
        }
        for (const double value : operator_data.trace_residual)
        {
          local_initial_squared += value * value;
        }
        local_ok = Finite(local_initial_squared);
      }
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  core::HostReferenceEngine condensation;
  if (elements > 0)
  {
    const core::LocalSystemBatchView system{
        {4, 8, 7, 5}, ConstView(a, elements, 28, 28),
        ConstView(b, elements, 28, 40), ConstView(c, elements, 40, 28),
        ConstView(d, elements, 40, 40), ConstView(f, elements, 28, 1),
        ConstView(g, elements, 40, 1)};
    // Condense the complete local batch once.  Recover below then uses these
    // exact retained LU factors rather than a second factorization.
    local_ok = condensation.Condense(
        system, {View(schur, elements, 40, 40), View(reduced, elements, 40, 1)})
        == core::Status::success && AllFinite(schur) && AllFinite(reduced);
  }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  std::unique_ptr<mfem::SparseMatrix> local_matrix;
  mfem::Vector local_rhs;
  mfem::Vector true_rhs_values;
  try
  {
    local_matrix = std::make_unique<mfem::SparseMatrix>(layout.TraceLocalVSize());
    local_rhs.SetSize(layout.TraceLocalVSize()); local_rhs = 0.0;
    true_rhs_values.SetSize(layout.TraceTrueVSize()); true_rhs_values = 0.0;
    for (int element = 0; element < elements && local_ok; ++element)
    {
      mfem::Array<int> vdofs;
      trace_space->GetElementVDofs(element, vdofs);
      local_ok = vdofs.Size() == 40;
      mfem::DenseMatrix local_schur(40);
      for (int row = 0; row < 40 && local_ok; ++row)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(vdofs[row], sign);
        local_ok = vdof >= 0 && vdof < local_rhs.Size();
        if (!local_ok) { break; }
        local_rhs[vdof] += sign * reduced[static_cast<std::size_t>(element) * 40 + row];
        // DenseMatrix storage is not a contract for the row-major Schur
        // vector; assign every entry explicitly to preclude a transpose.
        for (int column = 0; column < 40; ++column)
        {
          local_schur(row, column) = schur[
              static_cast<std::size_t>(element) * 40 * 40 +
              static_cast<std::size_t>(row) * 40 + column];
        }
      }
      if (local_ok) { local_matrix->AddSubMatrix(vdofs, vdofs, local_schur); }
    }
    if (local_ok) { local_matrix->Finalize(); }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  // The opaque parallel assembly is entered only after all ranks have a fully
  // formed local matrix.  Do not catch this collective operation.
  mfem::ParBilinearForm parallel_form(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
  std::unique_ptr<mfem::HypreParMatrix> true_matrix(
      parallel_form.ParallelAssemble(local_matrix.get()));
  local_ok = true_matrix != nullptr &&
      true_matrix->Height() == layout.TraceTrueVSize() &&
      true_matrix->Width() == layout.TraceTrueVSize() &&
      true_matrix->GetGlobalNumRows() == trace_space->GlobalTrueVSize() &&
      true_matrix->GetGlobalNumCols() == trace_space->GlobalTrueVSize() &&
      true_matrix->GetGlobalNumRows() > 0 &&
      true_matrix->GetGlobalNumRows() <= std::numeric_limits<int>::max();
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  trace_space->Dof_TrueDof_Matrix()->MultTranspose(local_rhs, true_rhs_values);
  if (!CollectiveAll(input.communicator, AllFinite(true_rhs_values))) { return false; }

  double initial_norm = 0.0;
  if (!GlobalNorm(input.communicator, local_initial_squared, initial_norm)) { return false; }
  const double rhs_squared = mfem::InnerProduct(input.communicator,
                                                true_rhs_values, true_rhs_values);
  if (!CollectiveAll(input.communicator, Finite(rhs_squared) && rhs_squared > 1.0e-300))
  {
    return false;
  }
  const double reduced_norm = std::sqrt(rhs_squared);

  mfem::HypreParVector true_rhs(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
  mfem::HypreParVector true_increment(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
  local_ok = CopyVector(true_rhs_values, true_rhs);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  true_increment = 0.0;

  // All solver objects are constructed and dimension-checked before the
  // collective HYPRE solve.  ILU/GMRES configuration matches the accepted
  // CPU HDG solve policy, while the independently recomputed true residual is
  // the success condition.
  mfem::HypreILU ilu;
  ilu.SetType(0); ilu.SetLevelOfFill(0); ilu.SetMaxIter(1); ilu.SetTol(0.0);
  ilu.SetLocalReordering(0); ilu.SetPrintLevel(0);
  mfem::HypreGMRES gmres(input.communicator);
  const int global_size = static_cast<int>(true_matrix->GetGlobalNumRows());
  gmres.SetTol(1.0e-12); gmres.SetAbsTol(0.0);
  gmres.SetMaxIter(std::max(100, global_size));
  gmres.SetKDim(std::min(100, std::max(20, global_size)));
  gmres.SetLogging(0); gmres.SetPrintLevel(0);
  gmres.SetPreconditioner(ilu); gmres.SetOperator(*true_matrix);
  gmres.Mult(true_rhs, true_increment);

  mfem::HypreParVector true_residual(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
  true_matrix->Mult(true_increment, true_residual);
  true_residual -= true_rhs;
  const double residual_squared = mfem::InnerProduct(input.communicator,
                                                     true_residual, true_residual);
  local_ok = AllFinite(true_increment) && Finite(residual_squared) &&
      residual_squared >= 0.0 && residual_squared <= rhs_squared * 1.0e-20;
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  const double linear_norm = std::sqrt(residual_squared);

  mfem::Vector next_trace;
  mfem::Vector local_trace_increment;
  mfem::Vector next_interior;
  try
  {
    next_trace.SetSize(layout.TraceTrueVSize());
    local_trace_increment.SetSize(layout.TraceLocalVSize());
    next_interior.SetSize(layout.InteriorLocalVSize());
    local_ok = CopyVector(true_increment, next_trace);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  // Enter the exchange only after every rank has its destination storage.
  local_ok = layout.DistributeTrace(next_trace, local_trace_increment);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  try
  {
    next_interior = 0.0;
    if (local_ok && elements > 0)
    {
      std::vector<double> local_trace_values(static_cast<std::size_t>(elements) * 40);
      for (int element = 0; element < elements && local_ok; ++element)
      {
        mfem::Vector element_trace(40);
        local_ok = layout.GatherTraceElement(element, local_trace_increment,
                                             element_trace);
        if (local_ok)
        {
          std::memcpy(local_trace_values.data() + static_cast<std::size_t>(element) * 40,
                      element_trace.HostRead(), 40 * sizeof(double));
        }
      }
      if (local_ok)
      {
        local_ok = condensation.Recover(
            ConstView(local_trace_values, elements, 40, 1),
            View(recovered, elements, 28, 1)) == core::Status::success;
      }
      for (int element = 0; element < elements && local_ok; ++element)
      {
        mfem::Vector element_interior(28);
        std::memcpy(element_interior.HostWrite(),
                    recovered.data() + static_cast<std::size_t>(element) * 28,
                    28 * sizeof(double));
        local_ok = layout.ScatterInteriorElement(element, element_interior,
                                                 next_interior);
      }
    }
    local_ok = local_ok && AllFinite(next_trace) && AllFinite(next_interior);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }

  output.trace_increment.Swap(next_trace);
  output.interior_increment.Swap(next_interior);
  output.initial_residual_norm = initial_norm;
  output.reduced_rhs_norm = reduced_norm;
  output.linear_residual_norm = linear_norm;
  return true;
}

} // namespace dg::hdg::runtime::cpu::detail
