#include "euler_committed_observable_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace dg_igr_hdg_reference_transfer_test
{
namespace
{

constexpr int components = 4;
constexpr int scalar_p1_dofs = 4;
constexpr double gamma_minus_one = 0.4;
constexpr double pi = 3.141592653589793238462643383279502884;

bool All(const MPI_Comm communicator, const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      output == 1;
}

int ByNodes(const int scalar, const int scalar_dofs,
            const int component) noexcept
{
  return scalar + scalar_dofs * component;
}

bool Pressure(const std::array<double, components>& state,
              double& pressure) noexcept
{
  const double density = state[0];
  if (!std::isfinite(density) || !(density > 0.0)) return false;
  const double momentum_squared = state[1] * state[1] + state[2] * state[2];
  pressure = gamma_minus_one *
      (state[3] - 0.5 * momentum_squared / density);
  return std::isfinite(pressure);
}

std::array<double, components> SmoothTransition(const double x) noexcept
{
  const double width = 1.0 / 8.0;
  const double kappa = std::atanh(0.8) / std::sin(pi * width);
  const double chi = 0.5 * (1.0 - std::tanh(kappa * std::cos(2.0 * pi * x)));
  const double density = 0.125 + 0.875 * chi;
  const double pressure = 0.1 + 0.9 * chi;
  return {{density, 0.0, 0.0, pressure / gamma_minus_one}};
}

bool SolveMass(std::vector<double> matrix,
               std::array<std::vector<double>, components>& rhs,
               const int size) noexcept
{
  if (size <= 0 || matrix.size() != static_cast<std::size_t>(size * size))
    return false;
  for (const auto& values : rhs)
    if (values.size() != static_cast<std::size_t>(size)) return false;
  for (int column = 0; column < size; ++column)
  {
    int pivot = column;
    for (int row = column + 1; row < size; ++row)
      if (std::abs(matrix[static_cast<std::size_t>(row * size + column)]) >
          std::abs(matrix[static_cast<std::size_t>(pivot * size + column)]))
        pivot = row;
    const double diagonal = matrix[static_cast<std::size_t>(pivot * size + column)];
    if (!std::isfinite(diagonal) || !(std::abs(diagonal) > 0.0)) return false;
    if (pivot != column)
    {
      for (int entry = column; entry < size; ++entry)
        std::swap(matrix[static_cast<std::size_t>(pivot * size + entry)],
                  matrix[static_cast<std::size_t>(column * size + entry)]);
      for (auto& values : rhs)
        std::swap(values[static_cast<std::size_t>(pivot)],
                  values[static_cast<std::size_t>(column)]);
    }
    for (int row = column + 1; row < size; ++row)
    {
      const double factor = matrix[static_cast<std::size_t>(row * size + column)] /
          matrix[static_cast<std::size_t>(column * size + column)];
      matrix[static_cast<std::size_t>(row * size + column)] = 0.0;
      for (int entry = column + 1; entry < size; ++entry)
        matrix[static_cast<std::size_t>(row * size + entry)] -=
            factor * matrix[static_cast<std::size_t>(column * size + entry)];
      for (auto& values : rhs)
        values[static_cast<std::size_t>(row)] -=
            factor * values[static_cast<std::size_t>(column)];
    }
  }
  for (auto& values : rhs)
    for (int row = size - 1; row >= 0; --row)
    {
      double value = values[static_cast<std::size_t>(row)];
      for (int column = row + 1; column < size; ++column)
        value -= matrix[static_cast<std::size_t>(row * size + column)] *
            values[static_cast<std::size_t>(column)];
      values[static_cast<std::size_t>(row)] = value /
          matrix[static_cast<std::size_t>(row * size + row)];
    }
  return true;
}

template <typename Sampler>
bool ReduceObservables(const MPI_Comm communicator, const mfem::ParMesh& mesh,
                       const int quadrature_order, Sampler&& sample,
                       EulerCommittedObservableOracle& output) noexcept
{
  std::array<double, components> local_integrals{};
  std::array<double, components> local_squares{};
  double local_density = std::numeric_limits<double>::infinity();
  double local_pressure = std::numeric_limits<double>::infinity();
  bool local_valid = quadrature_order > 0;
  try
  {
    for (int element = 0; element < mesh.GetNE() && local_valid; ++element)
    {
      const mfem::FiniteElement* finite_element =
          sample.ElementFiniteElement(element);
      mfem::ElementTransformation* transformation =
          mesh.GetElementTransformation(element);
      if (!finite_element || !transformation) { local_valid = false; break; }
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), quadrature_order);
      for (int point = 0; point < rule.GetNPoints() && local_valid; ++point)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(point);
        transformation->SetIntPoint(&ip);
        std::array<double, components> state{};
        local_valid = sample.Value(element, ip, state);
        double pressure = 0.0;
        local_valid = local_valid && Pressure(state, pressure);
        if (!local_valid) break;
        const double weight = ip.weight * transformation->Weight();
        local_valid = std::isfinite(weight);
        for (int component = 0; component < components && local_valid;
             ++component)
        {
          local_valid = std::isfinite(state[component]);
          local_integrals[static_cast<std::size_t>(component)] +=
              weight * state[component];
          local_squares[static_cast<std::size_t>(component)] +=
              weight * state[component] * state[component];
        }
        local_density = std::min(local_density, state[0]);
        local_pressure = std::min(local_pressure, pressure);
      }
    }
  }
  catch (...)
  {
    local_valid = false;
  }
  if (!All(communicator, local_valid)) return false;

  std::array<double, components> integrals{};
  std::array<double, components> squares{};
  double density = 0.0, pressure = 0.0;
  if (MPI_Allreduce(local_integrals.data(), integrals.data(), components,
                    MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS ||
      MPI_Allreduce(local_squares.data(), squares.data(), components,
                    MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_density, &density, 1, MPI_DOUBLE, MPI_MIN,
                    communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_pressure, &pressure, 1, MPI_DOUBLE, MPI_MIN,
                    communicator) != MPI_SUCCESS)
    return false;
  output = {};
  output.available = std::all_of(integrals.begin(), integrals.end(),
                                 [](const double value) { return std::isfinite(value); }) &&
      std::all_of(squares.begin(), squares.end(),
                  [](const double value) { return std::isfinite(value) && value >= 0.0; }) &&
      std::isfinite(density) && std::isfinite(pressure);
  output.quadrature_order = quadrature_order;
  output.conserved_integrals = integrals;
  for (int component = 0; component < components; ++component)
    output.conserved_l2_norms[static_cast<std::size_t>(component)] =
        std::sqrt(squares[static_cast<std::size_t>(component)]);
  output.minimum_density = density;
  output.minimum_pressure = pressure;
  return output.available;
}

} // namespace

bool MakeN8PeriodicMesh(const MPI_Comm communicator,
                        std::unique_ptr<mfem::ParMesh>& output) noexcept
{
  output.reset();
  int ranks = 0;
  if (communicator == MPI_COMM_NULL ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      (ranks != 1 && ranks != 2)) return false;
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        8, 8, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(2));
    translations[0][0] = 1.0;
    translations[1][1] = 1.0;
    mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    std::vector<int> partition(static_cast<std::size_t>(periodic.GetNE()), 0);
    if (ranks == 2)
      for (int element = 0; element < periodic.GetNE(); ++element)
        partition[static_cast<std::size_t>(element)] = element % 8 < 4 ? 0 : 1;
    output = std::make_unique<mfem::ParMesh>(communicator, periodic,
                                             partition.data());
  }
  catch (...)
  {
    output.reset();
  }
  return All(communicator, output != nullptr);
}

bool ProjectN8SmoothTransition(
    mfem::ParMesh& mesh, std::vector<double>& component_major_interior,
    std::vector<double>& true_trace) noexcept
{
  component_major_interior.clear();
  true_trace.clear();
  try
  {
    mfem::L2_FECollection volume_collection(
        n8_polynomial_order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::DG_Interface_FECollection trace_collection(
        n8_polynomial_order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace volume_space(
        &mesh, &volume_collection, components, mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, components, mfem::Ordering::byNODES);
    mfem::ParGridFunction volume(&volume_space);
    mfem::ParGridFunction trace(&trace_space);
    volume = 0.0;
    trace = 0.0;
    for (int element = 0; element < mesh.GetNE(); ++element)
    {
      const mfem::FiniteElement* finite_element = volume_space.GetFE(element);
      mfem::ElementTransformation* transformation = mesh.GetElementTransformation(element);
      if (!finite_element || !transformation) return false;
      const int dofs = finite_element->GetDof();
      std::vector<double> mass(static_cast<std::size_t>(dofs * dofs), 0.0);
      std::array<std::vector<double>, components> rhs;
      for (auto& values : rhs) values.assign(static_cast<std::size_t>(dofs), 0.0);
      mfem::Vector shape(dofs), point(2);
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), n8_projection_quadrature_order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        transformation->SetIntPoint(&ip);
        transformation->Transform(ip, point);
        finite_element->CalcShape(ip, shape);
        const auto state = SmoothTransition(point[0]);
        const double weight = ip.weight * transformation->Weight();
        for (int row = 0; row < dofs; ++row)
        {
          for (int column = 0; column < dofs; ++column)
            mass[static_cast<std::size_t>(row * dofs + column)] +=
                weight * shape[row] * shape[column];
          for (int component = 0; component < components; ++component)
            rhs[static_cast<std::size_t>(component)][static_cast<std::size_t>(row)] +=
                weight * shape[row] * state[static_cast<std::size_t>(component)];
        }
      }
      if (!SolveMass(mass, rhs, dofs)) return false;
      mfem::Vector local(components * dofs);
      for (int component = 0; component < components; ++component)
        for (int scalar = 0; scalar < dofs; ++scalar)
          local[ByNodes(scalar, dofs, component)] =
              rhs[static_cast<std::size_t>(component)][static_cast<std::size_t>(scalar)];
      mfem::Array<int> vdofs;
      mfem::DofTransformation dof_transformation(
          volume_space.GetVDim(), volume_space.GetOrdering());
      volume_space.GetElementVDofs(element, vdofs, dof_transformation);
      if (vdofs.Size() != local.Size()) return false;
      dof_transformation.TransformPrimal(local);
      volume.SetSubVector(vdofs, local);
    }
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      const mfem::FiniteElement* finite_element = trace_space.GetFaceElement(face);
      mfem::ElementTransformation* transformation = mesh.GetFaceTransformation(face);
      if (!finite_element || !transformation) return false;
      const int dofs = finite_element->GetDof();
      std::vector<double> mass(static_cast<std::size_t>(dofs * dofs), 0.0);
      std::array<std::vector<double>, components> rhs;
      for (auto& values : rhs) values.assign(static_cast<std::size_t>(dofs), 0.0);
      mfem::Vector shape(dofs), point(2);
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), n8_projection_quadrature_order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        transformation->SetIntPoint(&ip);
        transformation->Transform(ip, point);
        finite_element->CalcShape(ip, shape);
        const auto state = SmoothTransition(point[0]);
        const double weight = ip.weight * transformation->Weight();
        for (int row = 0; row < dofs; ++row)
        {
          for (int column = 0; column < dofs; ++column)
            mass[static_cast<std::size_t>(row * dofs + column)] +=
                weight * shape[row] * shape[column];
          for (int component = 0; component < components; ++component)
            rhs[static_cast<std::size_t>(component)][static_cast<std::size_t>(row)] +=
                weight * shape[row] * state[static_cast<std::size_t>(component)];
        }
      }
      if (!SolveMass(mass, rhs, dofs)) return false;
      mfem::Array<int> vdofs;
      trace_space.GetFaceVDofs(face, vdofs);
      if (vdofs.Size() != components * dofs) return false;
      for (int component = 0; component < components; ++component)
        for (int scalar = 0; scalar < dofs; ++scalar)
        {
          double sign = 1.0;
          const int dof = mfem::FiniteElementSpace::DecodeDof(
              vdofs[ByNodes(scalar, dofs, component)], sign);
          if (dof < 0 || dof >= trace.Size()) return false;
          trace[dof] = sign * rhs[static_cast<std::size_t>(component)]
                                  [static_cast<std::size_t>(scalar)];
        }
    }
    mfem::Vector projected(trace_space.GetTrueVSize());
    trace.ParallelProject(projected);
    const double* interior = volume.HostRead();
    const double* trace_values = projected.HostRead();
    if ((volume.Size() != 0 && interior == nullptr) ||
        (projected.Size() != 0 && trace_values == nullptr)) return false;
    if (volume.Size() != 0)
      component_major_interior.assign(interior, interior + volume.Size());
    if (projected.Size() != 0)
      true_trace.assign(trace_values, trace_values + projected.Size());
  }
  catch (...)
  {
    return false;
  }
  return All(mesh.GetComm(), true);
}

bool EvaluateCommittedEulerObservables(
    const mfem::ParGridFunction& interior, const int quadrature_order,
    EulerCommittedObservableOracle& output) noexcept
{
  const mfem::ParFiniteElementSpace* space = interior.ParFESpace();
  const mfem::ParMesh* mesh = space ? space->GetParMesh() : nullptr;
  if (!space || !mesh || space->GetVDim() != components) return false;
  struct Sampler final
  {
    const mfem::ParGridFunction& field;
    const mfem::ParFiniteElementSpace& space;
    const mfem::FiniteElement* ElementFiniteElement(const int element) const
    { return space.GetFE(element); }
    bool Value(const int element, const mfem::IntegrationPoint& point,
               std::array<double, components>& state) const
    {
      for (int component = 0; component < components; ++component)
        state[static_cast<std::size_t>(component)] =
            field.GetValue(element, point, component + 1);
      return true;
    }
  } sampler{interior, *space};
  return ReduceObservables(mesh->GetComm(), *mesh, quadrature_order, sampler,
                           output);
}

bool EvaluateSc25ElementMajorObservables(
    const mfem::ParMesh& mesh, const std::vector<double>& element_major_u,
    const int quadrature_order, EulerCommittedObservableOracle& output) noexcept
{
  if (element_major_u.size() != static_cast<std::size_t>(mesh.GetNE()) *
          components * scalar_p1_dofs) return false;
  mfem::L2_FECollection collection(n8_polynomial_order, 2,
      mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(const_cast<mfem::ParMesh*>(&mesh),
      &collection, components, mfem::Ordering::byNODES);
  struct Sampler final
  {
    const mfem::ParFiniteElementSpace& space;
    const std::vector<double>& coefficients;
    const mfem::FiniteElement* ElementFiniteElement(const int element) const
    { return space.GetFE(element); }
    bool Value(const int element, const mfem::IntegrationPoint& point,
               std::array<double, components>& state) const
    {
      const mfem::FiniteElement* finite_element = space.GetFE(element);
      if (!finite_element || finite_element->GetDof() != scalar_p1_dofs)
        return false;
      mfem::Vector shape(scalar_p1_dofs);
      finite_element->CalcShape(point, shape);
      for (int component = 0; component < components; ++component)
      {
        double value = 0.0;
        for (int scalar = 0; scalar < scalar_p1_dofs; ++scalar)
          value += shape[scalar] * coefficients[
              (static_cast<std::size_t>(element) * components + component) *
                  scalar_p1_dofs + scalar];
        state[static_cast<std::size_t>(component)] = value;
      }
      return true;
    }
  } sampler{space, element_major_u};
  return ReduceObservables(mesh.GetComm(), mesh, quadrature_order, sampler,
                           output);
}

bool CompareGenericFieldToSc25ElementMajor(
    const mfem::ParGridFunction& generic,
    const std::vector<double>& element_major_u,
    double& maximum_absolute_error, double& maximum_relative_error) noexcept
{
  maximum_absolute_error = 0.0;
  maximum_relative_error = 0.0;
  const mfem::ParFiniteElementSpace* space = generic.ParFESpace();
  const mfem::ParMesh* mesh = space ? space->GetParMesh() : nullptr;
  if (!space || !mesh || space->GetVDim() != components ||
      element_major_u.size() != static_cast<std::size_t>(mesh->GetNE()) *
          components * scalar_p1_dofs) return false;
  bool local_valid = true;
  for (int element = 0; element < mesh->GetNE() && local_valid; ++element)
  {
    const mfem::FiniteElement* finite_element = space->GetFE(element);
    mfem::Array<int> vdofs;
    space->GetElementVDofs(element, vdofs);
    local_valid = finite_element && finite_element->GetDof() == scalar_p1_dofs &&
        vdofs.Size() == components * scalar_p1_dofs;
    for (int component = 0; component < components && local_valid; ++component)
      for (int scalar = 0; scalar < scalar_p1_dofs; ++scalar)
      {
        double sign = 1.0;
        const int dof = mfem::FiniteElementSpace::DecodeDof(
            vdofs[ByNodes(scalar, scalar_p1_dofs, component)], sign);
        if (dof < 0 || dof >= generic.Size()) { local_valid = false; break; }
        const double actual = sign * generic[dof];
        const double expected = element_major_u[
            (static_cast<std::size_t>(element) * components + component) *
                scalar_p1_dofs + scalar];
        const double absolute = std::abs(actual - expected);
        maximum_absolute_error = std::max(maximum_absolute_error, absolute);
        maximum_relative_error = std::max(maximum_relative_error,
            absolute / std::max({1.0, std::abs(actual), std::abs(expected)}));
      }
  }
  if (!All(mesh->GetComm(), local_valid)) return false;
  double global_absolute = 0.0, global_relative = 0.0;
  return MPI_Allreduce(&maximum_absolute_error, &global_absolute, 1,
                       MPI_DOUBLE, MPI_MAX, mesh->GetComm()) == MPI_SUCCESS &&
      MPI_Allreduce(&maximum_relative_error, &global_relative, 1,
                    MPI_DOUBLE, MPI_MAX, mesh->GetComm()) == MPI_SUCCESS &&
      ((maximum_absolute_error = global_absolute),
       (maximum_relative_error = global_relative), true);
}

} // namespace dg_igr_hdg_reference_transfer_test
