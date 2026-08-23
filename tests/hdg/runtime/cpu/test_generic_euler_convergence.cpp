#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;

constexpr int dimension = 2;
constexpr int components = 4;
constexpr double gamma = 1.4;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double spatial_order_floor = 1.5;
constexpr double temporal_order_floor = 0.8;
constexpr double temporal_order_ceiling = 1.2;

constexpr std::uint32_t cpu_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cpu) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

/**
 * Constant velocity and pressure reduce the Euler equations for this family
 * to scalar advection of density.  A wavevector perpendicular to velocity is
 * therefore an exact steady solution; a general wavevector is translated at
 * velocity dot wavevector.  Both spellings remain smooth and admissible.
 */
struct EntropyWave
{
  std::array<double, dimension> velocity{{0.0, 0.0}};
  std::array<double, dimension> wavevector{{1.0, 0.0}};
  double density_amplitude = 0.05;
  double pressure = 1.0;

  void Evaluate(const double x, const double y, const double time,
                std::array<double, components>& conservative) const noexcept
  {
    const double advection_rate = velocity[0] * wavevector[0] +
        velocity[1] * wavevector[1];
    const double phase = 2.0 * pi *
        (wavevector[0] * x + wavevector[1] * y - advection_rate * time);
    const double density = 1.0 + density_amplitude * std::sin(phase);
    const double kinetic_energy = 0.5 * density *
        (velocity[0] * velocity[0] + velocity[1] * velocity[1]);
    conservative = {{density,
                     density * velocity[0],
                     density * velocity[1],
                     pressure / (gamma - 1.0) + kinetic_energy}};
  }
};

struct BoundaryAudit
{
  int calls = 0;
};

bool PeriodicExterior(const void* opaque,
                      const dg::hdg::generic::BoundaryPointView& point,
                      const double* interior, double* exterior) noexcept
{
  auto* const audit = static_cast<BoundaryAudit*>(const_cast<void*>(opaque));
  if (audit == nullptr || point.space_dimension != dimension ||
      point.position == nullptr || point.outward_normal == nullptr ||
      interior == nullptr || exterior == nullptr)
  {
    return false;
  }
  ++audit->calls;
  for (int component = 0; component < components; ++component)
    exterior[component] = interior[component];
  return true;
}

bool CollectiveAll(const bool local, const MPI_Comm communicator)
{
  const int source = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

mfem::ParMesh MakePeriodicMesh(const int elements_per_direction,
                               const MPI_Comm communicator)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      elements_per_direction, elements_per_direction,
      mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(communicator, periodic);
}

bool ProjectInitialState(const EntropyWave& wave,
                         mfem::ParGridFunction& initial)
{
  auto exact = [&wave](const mfem::Vector& point, mfem::Vector& value)
  {
    std::array<double, components> state{};
    wave.Evaluate(point[0], point[1], 0.0, state);
    value.SetSize(components);
    for (int component = 0; component < components; ++component)
      value[component] = state[static_cast<std::size_t>(component)];
  };
  try
  {
    mfem::VectorFunctionCoefficient coefficient(components, exact);
    initial.ProjectCoefficient(coefficient);
  }
  catch (...)
  {
    return false;
  }
  return true;
}

EulerHdgRequest MakeRequest(const mfem::ParMesh& mesh, const int order,
                            const mfem::ParGridFunction& initial,
                            const mfem::ParGridFunction& source,
                            BoundaryAudit& boundary)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = cpu_capabilities;
  request.initial_state = {
      initial.HostRead(), static_cast<std::size_t>(initial.Size()), components};
  request.initial_true_trace = {nullptr, 0, components};
  request.source = {
      source.HostRead(), static_cast<std::size_t>(source.Size()), components};
  request.boundary = {
      &boundary, PeriodicExterior, {nullptr, 0, 0, 0.0}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  return request;
}

bool Advance(const mfem::ParMesh& mesh, const int order,
             const mfem::ParGridFunction& initial,
             const mfem::ParGridFunction& source, const int steps,
             const double dt, mfem::ParGridFunction& committed)
{
  BoundaryAudit boundary{};
  const EulerHdgRequest request =
      MakeRequest(mesh, order, initial, source, boundary);
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    if (session.Advance(dt).status != GenericEulerCpuSessionStatus::advanced)
      return false;
  }
  const auto coordinates = session.coordinates();
  const auto diagnostics = session.diagnostics();
  const double expected_time = static_cast<double>(steps) * dt;
  const double time_scale = std::max(1.0, std::abs(expected_time));
  const bool local = session.ExportCommittedInterior(committed) &&
      boundary.calls == 0 && coordinates.step ==
          static_cast<std::uint64_t>(steps) &&
      coordinates.generation == static_cast<std::uint64_t>(steps) &&
      std::abs(coordinates.time - expected_time) <=
          64.0 * std::numeric_limits<double>::epsilon() * time_scale &&
      diagnostics.available && diagnostics.wave_generation ==
          coordinates.generation &&
      diagnostics.coordinates.step == coordinates.step &&
      std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed > 0.0;
  return CollectiveAll(local, mesh.GetComm());
}

double GlobalExactL2Error(const mfem::ParGridFunction& field,
                          const EntropyWave& wave, const double time)
{
  const auto* const space = field.ParFESpace();
  auto* const mesh = space == nullptr
      ? nullptr
      : const_cast<mfem::ParMesh*>(space->GetParMesh());
  if (space == nullptr || mesh == nullptr) return
      std::numeric_limits<double>::infinity();

  bool valid = true;
  long double local_error_squared = 0.0L;
  try
  {
    for (int element = 0; element < mesh->GetNE(); ++element)
    {
      const mfem::FiniteElement* const finite_element = space->GetFE(element);
      mfem::ElementTransformation* const transformation =
          mesh->GetElementTransformation(element);
      if (finite_element == nullptr || transformation == nullptr)
      {
        valid = false;
        break;
      }
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), 2 * finite_element->GetOrder() + 8);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& integration = rule.IntPoint(point);
        transformation->SetIntPoint(&integration);
        mfem::Vector physical(dimension);
        mfem::Vector numerical(components);
        transformation->Transform(integration, physical);
        field.GetVectorValue(element, integration, numerical);
        std::array<double, components> exact{};
        wave.Evaluate(physical[0], physical[1], time, exact);
        const double weight = integration.weight * transformation->Weight();
        if (numerical.Size() != components || !(weight > 0.0) ||
            !std::isfinite(weight))
        {
          valid = false;
          break;
        }
        for (int component = 0; component < components; ++component)
        {
          const long double difference =
              static_cast<long double>(numerical[component]) -
              exact[static_cast<std::size_t>(component)];
          local_error_squared +=
              static_cast<long double>(weight) * difference * difference;
        }
      }
      if (!valid) break;
    }
  }
  catch (...)
  {
    valid = false;
  }

  if (!CollectiveAll(valid, mesh->GetComm()))
    return std::numeric_limits<double>::infinity();
  long double global_error_squared = 0.0L;
  if (MPI_Allreduce(&local_error_squared, &global_error_squared, 1,
                    MPI_LONG_DOUBLE, MPI_SUM, mesh->GetComm()) != MPI_SUCCESS ||
      !(global_error_squared > 0.0L))
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(static_cast<double>(global_error_squared));
}

double GlobalDifferenceL2(const mfem::ParGridFunction& left,
                          const mfem::ParGridFunction& right)
{
  const auto* const space = left.ParFESpace();
  auto* const mesh = space == nullptr
      ? nullptr
      : const_cast<mfem::ParMesh*>(space->GetParMesh());
  if (space == nullptr || mesh == nullptr || right.ParFESpace() != space)
    return std::numeric_limits<double>::infinity();

  bool valid = true;
  long double local_difference_squared = 0.0L;
  try
  {
    for (int element = 0; element < mesh->GetNE(); ++element)
    {
      const mfem::FiniteElement* const finite_element = space->GetFE(element);
      mfem::ElementTransformation* const transformation =
          mesh->GetElementTransformation(element);
      if (finite_element == nullptr || transformation == nullptr)
      {
        valid = false;
        break;
      }
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), 2 * finite_element->GetOrder() + 8);
      for (int point = 0; point < rule.GetNPoints(); ++point)
      {
        const mfem::IntegrationPoint& integration = rule.IntPoint(point);
        transformation->SetIntPoint(&integration);
        mfem::Vector left_value(components);
        mfem::Vector right_value(components);
        left.GetVectorValue(element, integration, left_value);
        right.GetVectorValue(element, integration, right_value);
        const double weight = integration.weight * transformation->Weight();
        if (left_value.Size() != components ||
            right_value.Size() != components || !(weight > 0.0) ||
            !std::isfinite(weight))
        {
          valid = false;
          break;
        }
        for (int component = 0; component < components; ++component)
        {
          const long double difference =
              static_cast<long double>(left_value[component]) -
              right_value[component];
          local_difference_squared +=
              static_cast<long double>(weight) * difference * difference;
        }
      }
      if (!valid) break;
    }
  }
  catch (...)
  {
    valid = false;
  }

  if (!CollectiveAll(valid, mesh->GetComm()))
    return std::numeric_limits<double>::infinity();
  long double global_difference_squared = 0.0L;
  if (MPI_Allreduce(&local_difference_squared, &global_difference_squared, 1,
                    MPI_LONG_DOUBLE, MPI_SUM, mesh->GetComm()) != MPI_SUCCESS ||
      !(global_difference_squared > 0.0L))
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(static_cast<double>(global_difference_squared));
}

double HalvingOrder(const double coarse, const double fine) noexcept
{
  if (!(coarse > 0.0) || !(fine > 0.0) || !std::isfinite(coarse) ||
      !std::isfinite(fine))
  {
    return -std::numeric_limits<double>::infinity();
  }
  return std::log(coarse / fine) / std::log(2.0);
}

bool CheckSmoothSpatialConvergence(const MPI_Comm communicator)
{
  // velocity dot wavevector is exactly zero: the analytic solution is steady,
  // so the fixed final time introduces no Backward-Euler truncation error.
  const EntropyWave wave{{{0.5, 0.0}}, {{0.0, 1.0}}, 0.1, 1.0};
  constexpr int order = 1;
  constexpr double final_time = 0.02;
  const std::array<int, 3> mesh_ladder{{3, 6, 12}};
  std::array<double, 3> errors{};

  for (std::size_t level = 0; level < mesh_ladder.size(); ++level)
  {
    mfem::ParMesh mesh = MakePeriodicMesh(mesh_ladder[level], communicator);
    mfem::L2_FECollection collection(
        order, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, components, mfem::Ordering::byNODES);
    mfem::ParGridFunction initial(&space);
    mfem::ParGridFunction source(&space);
    mfem::ParGridFunction committed(&space);
    source = 0.0;
    if (!ProjectInitialState(wave, initial) ||
        !Advance(mesh, order, initial, source, 1, final_time, committed))
    {
      return false;
    }
    errors[level] = GlobalExactL2Error(committed, wave, final_time);
  }

  const double coarse_order = HalvingOrder(errors[0], errors[1]);
  const double fine_order = HalvingOrder(errors[1], errors[2]);
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  if (rank == 0)
  {
    std::cout << std::setprecision(17)
              << "generic CPU smooth spatial errors: "
              << errors[0] << ' ' << errors[1] << ' ' << errors[2]
              << "; observed orders: " << coarse_order << ' '
              << fine_order << "; required floor: "
              << spatial_order_floor << '\n';
  }
  const bool local = std::isfinite(errors[0]) && std::isfinite(errors[1]) &&
      std::isfinite(errors[2]) && errors[0] > errors[1] &&
      errors[1] > errors[2] && coarse_order >= spatial_order_floor &&
      fine_order >= spatial_order_floor;
  return CollectiveAll(local, communicator);
}

bool CheckBackwardEulerTemporalConvergence(const MPI_Comm communicator)
{
  const EntropyWave wave{{{0.5, 0.25}}, {{1.0, 1.0}}, 0.05, 1.0};
  constexpr int order = 3;
  constexpr int elements_per_direction = 4;
  constexpr double final_time = 0.08;
  const std::array<int, 3> step_ladder{{2, 4, 8}};

  mfem::ParMesh mesh =
      MakePeriodicMesh(elements_per_direction, communicator);
  mfem::L2_FECollection collection(
      order, dimension, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction initial(&space);
  mfem::ParGridFunction source(&space);
  mfem::ParGridFunction coarse(&space);
  mfem::ParGridFunction medium(&space);
  mfem::ParGridFunction fine(&space);
  source = 0.0;
  if (!ProjectInitialState(wave, initial)) return false;

  const std::array<mfem::ParGridFunction*, 3> solutions{{
      &coarse, &medium, &fine}};
  std::array<double, 3> exact_errors{};
  for (std::size_t level = 0; level < step_ladder.size(); ++level)
  {
    const double dt = final_time /
        static_cast<double>(step_ladder[level]);
    if (!Advance(mesh, order, initial, source, step_ladder[level], dt,
                 *solutions[level]))
    {
      return false;
    }
    exact_errors[level] =
        GlobalExactL2Error(*solutions[level], wave, final_time);
  }

  const double coarse_difference = GlobalDifferenceL2(coarse, medium);
  const double fine_difference = GlobalDifferenceL2(medium, fine);
  const double observed_order =
      HalvingOrder(coarse_difference, fine_difference);
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  if (rank == 0)
  {
    std::cout << std::setprecision(17)
              << "generic CPU Backward-Euler exact errors: "
              << exact_errors[0] << ' ' << exact_errors[1] << ' '
              << exact_errors[2] << "; self differences: "
              << coarse_difference << ' ' << fine_difference
              << "; observed order: " << observed_order
              << "; required interval: [" << temporal_order_floor << ", "
              << temporal_order_ceiling << "]\n";
  }
  const bool local = std::isfinite(exact_errors[0]) &&
      std::isfinite(exact_errors[1]) && std::isfinite(exact_errors[2]) &&
      exact_errors[2] < exact_errors[0] &&
      std::isfinite(coarse_difference) && std::isfinite(fine_difference) &&
      coarse_difference > fine_difference &&
      observed_order >= temporal_order_floor &&
      observed_order <= temporal_order_ceiling;
  return CollectiveAll(local, communicator);
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  const bool spatial = CheckSmoothSpatialConvergence(MPI_COMM_WORLD);
  const bool temporal =
      CheckBackwardEulerTemporalConvergence(MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
  {
    std::cout << "generic CPU convergence gate: spatial=" << spatial
              << " temporal=" << temporal << '\n';
  }
  MPI_Finalize();
  return spatial && temporal ? 0 : 1;
}
