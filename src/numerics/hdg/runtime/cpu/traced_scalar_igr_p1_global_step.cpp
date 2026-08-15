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

TracedScalarIgrP1GlobalStepResult CollectiveStepResult(
    const MPI_Comm communicator,
    const TracedScalarIgrP1GlobalStepResult local_result) noexcept
{
  const int local_code = static_cast<int>(local_result);
  int global_code = static_cast<int>(TracedScalarIgrP1GlobalStepResult::failure);
  if (MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX,
                    communicator) != MPI_SUCCESS ||
      global_code < static_cast<int>(TracedScalarIgrP1GlobalStepResult::success) ||
      global_code > static_cast<int>(TracedScalarIgrP1GlobalStepResult::failure))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  return static_cast<TracedScalarIgrP1GlobalStepResult>(global_code);
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
      input.current_true_trace == input.previous_true_trace ||
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
      source_space->GetVSize() % 4 == 0 &&
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
      source_space->GetOrdering() != mfem::Ordering::byNODES ||
      source_space->GetVSize() % 4 != 0)
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
        // MFEM byNODES is component-blocked: [component][scalar dof].
        const int vdof = DecodeSignedVDof(signed_vdofs[component * 4 + dof], sign);
        if (vdof < 0 || vdof >= source_coefficients->Size()) { return false; }
        value += shape[dof] * sign * coefficients[vdof];
      }
      if (!Finite(value)) { return false; }
      samples[static_cast<std::size_t>(point) * 4 + component] = value;
    }
  }
  return true;
}

bool MaxAbsAndNorm(const MPI_Comm communicator, const mfem::Vector& values,
                   double& maximum, double& norm) noexcept
{
  const double* const data = values.HostRead();
  bool local_ok = data != nullptr || values.Size() == 0;
  double local_maximum = 0.0;
  for (int index = 0; index < values.Size() && local_ok; ++index)
  {
    local_ok = Finite(data[index]);
    if (!local_ok) { break; }
    local_maximum = std::max(local_maximum, std::abs(data[index]));
  }
  if (!CollectiveAll(communicator, local_ok) ||
      MPI_Allreduce(&local_maximum, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                    communicator) != MPI_SUCCESS || !Finite(maximum) ||
      maximum < 0.0)
  {
    return false;
  }
  if (maximum == 0.0) { norm = 0.0; return true; }
  double local_sum = 0.0;
  for (int index = 0; index < values.Size(); ++index)
  {
    const double scaled = data[index] / maximum;
    local_sum += scaled * scaled;
  }
  double global_sum = 0.0;
  if (!CollectiveAll(communicator, Finite(local_sum)) ||
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    communicator) != MPI_SUCCESS || !Finite(global_sum) ||
      global_sum <= 0.0)
  {
    return false;
  }
  norm = maximum * std::sqrt(global_sum);
  return Finite(norm);
}

/** Scale-safe norm of all broken interior values and owned assembled trace values. */
bool MaxAbsAndNorm(const MPI_Comm communicator, const std::vector<double>& interior,
                   const mfem::Vector& true_trace, double& maximum,
                   double& norm) noexcept
{
  const double* const trace = true_trace.HostRead();
  bool local_ok = (trace != nullptr || true_trace.Size() == 0) &&
      AllFinite(interior);
  double local_maximum = 0.0;
  for (const double value : interior)
  {
    local_maximum = std::max(local_maximum, std::abs(value));
  }
  for (int index = 0; index < true_trace.Size() && local_ok; ++index)
  {
    local_ok = Finite(trace[index]);
    if (!local_ok) { break; }
    local_maximum = std::max(local_maximum, std::abs(trace[index]));
  }
  if (!CollectiveAll(communicator, local_ok) ||
      MPI_Allreduce(&local_maximum, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                    communicator) != MPI_SUCCESS || !Finite(maximum) ||
      maximum < 0.0)
  {
    return false;
  }
  if (maximum == 0.0) { norm = 0.0; return true; }
  double local_sum = 0.0;
  for (const double value : interior)
  {
    const double scaled = value / maximum;
    local_sum += scaled * scaled;
  }
  for (int index = 0; index < true_trace.Size(); ++index)
  {
    const double scaled = trace[index] / maximum;
    local_sum += scaled * scaled;
  }
  double global_sum = 0.0;
  if (!CollectiveAll(communicator, Finite(local_sum)) ||
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    communicator) != MPI_SUCCESS || !Finite(global_sum) ||
      global_sum <= 0.0)
  {
    return false;
  }
  norm = maximum * std::sqrt(global_sum);
  return Finite(norm);
}

} // namespace

bool SolveAndRecoverTracedScalarIgrP1Trace(
    const TracedScalarIgrP1TraceSolveInput& input,
    TracedScalarIgrP1TraceSolveOutput& output) noexcept
{
  if (input.communicator == MPI_COMM_NULL) { return false; }
  bool local_input_ok = input.layout != nullptr && input.trace_space != nullptr &&
      input.true_matrix != nullptr && input.true_rhs_values != nullptr &&
      input.condensation != nullptr;
  if (local_input_ok)
  {
    local_input_ok = input.layout->IsUsable() &&
        input.trace_space == input.layout->TraceSpace() &&
        input.true_rhs_values->Size() == input.layout->TraceTrueVSize() &&
        input.true_matrix->Height() == input.layout->TraceTrueVSize() &&
        input.true_matrix->Width() == input.layout->TraceTrueVSize() &&
        input.true_matrix->GetGlobalNumRows() == input.trace_space->GlobalTrueVSize() &&
        input.true_matrix->GetGlobalNumCols() == input.trace_space->GlobalTrueVSize();
  }
  if (!CollectiveAll(input.communicator, local_input_ok)) { return false; }
  const int elements = input.layout->LocalElementCount();
  if (!CollectiveAll(input.communicator, elements == 0 ||
      (input.condensation->ready() && input.condensation->batch_count() ==
       static_cast<std::size_t>(elements)))) { return false; }

  double rhs_maximum = 0.0;
  double rhs_norm = 0.0;
  if (!MaxAbsAndNorm(input.communicator, *input.true_rhs_values, rhs_maximum,
                     rhs_norm) || !CollectiveAll(input.communicator, Finite(rhs_norm)))
  {
    return false;
  }
  const bool exact_zero_rhs = rhs_maximum == 0.0;
  if (!exact_zero_rhs && rhs_maximum < std::numeric_limits<double>::min())
  {
    return false;
  }
  mfem::HypreParVector trace_solution(
      const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
  trace_solution = 0.0;
  double linear_norm = 0.0;

  if (!exact_zero_rhs)
  {
    mfem::Vector scaled_values(input.layout->TraceTrueVSize());
    const double* const rhs = input.true_rhs_values->HostRead();
    double* const scaled = scaled_values.HostWrite();
    bool local_ok = (rhs != nullptr && scaled != nullptr) || scaled_values.Size() == 0;
    for (int index = 0; index < scaled_values.Size() && local_ok; ++index)
    {
      scaled[index] = rhs[index] / rhs_maximum;
      local_ok = Finite(scaled[index]);
    }
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    mfem::HypreParVector scaled_rhs(
        const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
    mfem::HypreParVector scaled_solution(
        const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
    local_ok = CopyVector(scaled_values, scaled_rhs);
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    scaled_solution = 0.0;
    mfem::HypreILU ilu;
    ilu.SetType(0); ilu.SetLevelOfFill(0); ilu.SetMaxIter(1); ilu.SetTol(0.0);
    ilu.SetLocalReordering(0); ilu.SetPrintLevel(0);
    mfem::HypreGMRES gmres(input.communicator);
    const int global_size = static_cast<int>(input.true_matrix->GetGlobalNumRows());
    gmres.SetTol(1.0e-12); gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(std::max(100, global_size));
    gmres.SetKDim(std::min(100, std::max(20, global_size)));
    gmres.SetLogging(0); gmres.SetPrintLevel(0);
    gmres.SetPreconditioner(ilu); gmres.SetOperator(*input.true_matrix);
    gmres.Mult(scaled_rhs, scaled_solution);
    mfem::HypreParVector scaled_residual(
        const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
    input.true_matrix->Mult(scaled_solution, scaled_residual);
    scaled_residual -= scaled_rhs;
    double residual_maximum = 0.0;
    double residual_norm = 0.0;
    if (!MaxAbsAndNorm(input.communicator, scaled_residual, residual_maximum,
                       residual_norm)) { return false; }
    local_ok = AllFinite(scaled_solution) && residual_norm <= 1.0e-10;
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    const double* const scaled_solution_data = scaled_solution.HostRead();
    double* const solution = trace_solution.HostWrite();
    local_ok = (scaled_solution_data != nullptr && solution != nullptr) ||
        trace_solution.Size() == 0;
    for (int index = 0; index < trace_solution.Size() && local_ok; ++index)
    {
      solution[index] = rhs_maximum * scaled_solution_data[index];
      local_ok = Finite(solution[index]);
    }
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    linear_norm = rhs_maximum * residual_norm;
    if (!Finite(linear_norm)) { return false; }
  }

  mfem::Vector next_trace;
  mfem::Vector local_trace;
  mfem::Vector next_interior;
  std::vector<double> recovered;
  bool local_ok = true;
  try
  {
    next_trace.SetSize(input.layout->TraceTrueVSize());
    local_trace.SetSize(input.layout->TraceLocalVSize());
    next_interior.SetSize(input.layout->InteriorLocalVSize());
    recovered.resize(static_cast<std::size_t>(elements) * 28);
    local_ok = CopyVector(trace_solution, next_trace);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  local_ok = input.layout->DistributeTrace(next_trace, local_trace);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  try
  {
    next_interior = 0.0;
    if (elements > 0)
    {
      std::vector<double> local_trace_values(static_cast<std::size_t>(elements) * 40);
      for (int element = 0; element < elements && local_ok; ++element)
      {
        mfem::Vector gathered(40);
        local_ok = input.layout->GatherTraceElement(element, local_trace, gathered);
        if (local_ok) { std::memcpy(local_trace_values.data() + static_cast<std::size_t>(element) * 40, gathered.HostRead(), 40 * sizeof(double)); }
      }
      if (local_ok) { local_ok = input.condensation->Recover(ConstView(local_trace_values, elements, 40, 1), View(recovered, elements, 28, 1)) == core::Status::success; }
      for (int element = 0; element < elements && local_ok; ++element)
      {
        mfem::Vector element_interior(28);
        std::memcpy(element_interior.HostWrite(), recovered.data() + static_cast<std::size_t>(element) * 28, 28 * sizeof(double));
        local_ok = input.layout->ScatterInteriorElement(element, element_interior, next_interior);
      }
    }
    local_ok = local_ok && AllFinite(next_trace) && AllFinite(next_interior);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  output.trace_increment.Swap(next_trace);
  output.interior_increment.Swap(next_interior);
  output.reduced_rhs_norm = rhs_norm;
  output.linear_residual_norm = linear_norm;
  return true;
}

TracedScalarIgrP1GlobalStepResult ComputeTracedScalarIgrP1GlobalStep(
    const TracedScalarIgrP1GlobalStepInput& input,
    TracedScalarIgrP1GlobalStepOutput& output) noexcept
{
  // The communicator has been admitted by the caller.  This initial local
  // check is deliberately before every opaque collective phase.
  if (input.communicator == MPI_COMM_NULL)
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  if (!CollectiveAll(input.communicator, ValidInput(input)))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  const BorrowedP1DGInterfaceLayout& layout = *input.layout;
  const int elements = layout.LocalElementCount();
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (!CollectiveAll(input.communicator, trace_space != nullptr))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  mfem::Vector current_local_trace;
  mfem::Vector previous_local_trace;
  std::vector<double> a, b, c, d, f, g, source_samples;
  std::vector<double> schur, reduced;
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
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  // Both trace distributions are collective through the existing exact-space
  // exchange plan, so all local allocation/shape checks have happened above.
  local_ok = layout.DistributeTrace(*input.current_true_trace,
                                    current_local_trace) &&
      layout.DistributeTrace(*input.previous_true_trace, previous_local_trace);
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  TracedScalarIgrP1GlobalStepResult local_result =
      TracedScalarIgrP1GlobalStepResult::success;
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
        const local::Status assembly = local::Assemble(
            input.affine_quads[element], input.parameters, state, operator_data);
        if (assembly == local::Status::inadmissible_state)
        {
          local_result = TracedScalarIgrP1GlobalStepResult::inadmissible_trial;
          local_ok = false;
        }
        else
        {
          local_ok = assembly == local::Status::success;
        }
      }
      if (local_ok)
      {
        a.insert(a.end(), operator_data.A.begin(), operator_data.A.end());
        b.insert(b.end(), operator_data.B.begin(), operator_data.B.end());
        c.insert(c.end(), operator_data.C.begin(), operator_data.C.end());
        d.insert(d.end(), operator_data.D.begin(), operator_data.D.end());
        f.insert(f.end(), operator_data.f.begin(), operator_data.f.end());
        g.insert(g.end(), operator_data.g.begin(), operator_data.g.end());
        local_ok = AllFinite(operator_data.interior_residual) &&
            AllFinite(operator_data.trace_residual);
      }
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (local_result == TracedScalarIgrP1GlobalStepResult::success && !local_ok)
  {
    local_result = TracedScalarIgrP1GlobalStepResult::failure;
  }
  const TracedScalarIgrP1GlobalStepResult assembly_result =
      CollectiveStepResult(input.communicator, local_result);
  if (assembly_result != TracedScalarIgrP1GlobalStepResult::success)
  {
    return assembly_result;
  }

  mfem::Vector raw_local_trace;
  mfem::Vector raw_true_trace;
  try
  {
    raw_local_trace.SetSize(layout.TraceLocalVSize()); raw_local_trace = 0.0;
    raw_true_trace.SetSize(layout.TraceTrueVSize()); raw_true_trace = 0.0;
    for (int element = 0; element < elements && local_ok; ++element)
    {
      mfem::Array<int> vdofs;
      trace_space->GetElementVDofs(element, vdofs);
      local_ok = vdofs.Size() == 40;
      for (int row = 0; row < 40 && local_ok; ++row)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(vdofs[row], sign);
        local_ok = vdof >= 0 && vdof < raw_local_trace.Size();
        if (local_ok)
        {
          raw_local_trace[vdof] += sign *
              g[static_cast<std::size_t>(element) * 40 + row];
          local_ok = Finite(raw_local_trace[vdof]);
        }
      }
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  trace_space->Dof_TrueDof_Matrix()->MultTranspose(raw_local_trace,
                                                    raw_true_trace);
  if (!CollectiveAll(input.communicator, AllFinite(raw_true_trace)))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  double initial_maximum = 0.0;
  double initial_norm = 0.0;
  if (!MaxAbsAndNorm(input.communicator, f, raw_true_trace, initial_maximum,
                     initial_norm))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

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
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

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
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

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
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  trace_space->Dof_TrueDof_Matrix()->MultTranspose(local_rhs, true_rhs_values);
  if (!CollectiveAll(input.communicator, AllFinite(true_rhs_values)))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  TracedScalarIgrP1TraceSolveOutput solved;
  const TracedScalarIgrP1TraceSolveInput solve_input{
      input.communicator, &layout, trace_space, true_matrix.get(),
      &true_rhs_values, &condensation};
  if (!SolveAndRecoverTracedScalarIgrP1Trace(solve_input, solved))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  output.trace_increment.Swap(solved.trace_increment);
  output.interior_increment.Swap(solved.interior_increment);
  output.initial_residual_norm = initial_norm;
  output.reduced_rhs_norm = solved.reduced_rhs_norm;
  output.linear_residual_norm = solved.linear_residual_norm;
  return TracedScalarIgrP1GlobalStepResult::success;
}

} // namespace dg::hdg::runtime::cpu::detail
