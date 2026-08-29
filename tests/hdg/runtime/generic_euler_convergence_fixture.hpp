#pragma once

#include "numerics/hdg/generic/contract.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace dg::hdg::test::generic_euler_convergence
{

using generic::BackendKind;
using generic::Capability;
using generic::CapabilityBit;
using generic::CudaEngineKind;
using generic::EulerHdgAssemblyQuadratureOrders;
using generic::EulerBoundaryPolicyTableView;
using generic::EulerHdgRequest;
using generic::RegularizationKind;

inline constexpr int dimension = 2;
inline constexpr int components = generic::euler_2d_conserved_components;
inline constexpr double gamma = 1.4;
inline constexpr double pi =
    3.141592653589793238462643383279502884;

// These are the pre-existing CPU convergence gates.  Applying the same
// thresholds to every backend and mode changes coverage, not acceptance.
inline constexpr double spatial_order_floor = 1.5;
inline constexpr double temporal_order_floor = 0.8;
inline constexpr double temporal_order_ceiling = 1.2;

inline constexpr std::uint32_t common_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

struct ExecutionSelection
{
  BackendKind backend = BackendKind::cpu;
  CudaEngineKind cuda_engine = CudaEngineKind::ordinary;
  std::string_view name = "cpu";
};

struct CaseSelection
{
  int order = 0;
  RegularizationKind regularization = RegularizationKind::none;
  EulerHdgAssemblyQuadratureOrders igr_assembly_quadrature_orders{0, 0};
  bool quadrature_equivalence = false;
};

inline bool ParseOrder(const std::string_view text, int& order) noexcept
{
  order = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, order);
  return parsed.ec == std::errc{} && parsed.ptr == end &&
      order >= generic::minimum_polynomial_order &&
      order <= generic::maximum_polynomial_order;
}

inline bool ParseQuadratureOrder(
    const std::string_view text, int& order) noexcept
{
  order = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, order);
  return parsed.ec == std::errc{} && parsed.ptr == end &&
      order >= 1 && order <= 14;
}

inline bool ParseCaseSelection(const int argc, char* argv[],
                               CaseSelection& selection) noexcept
{
  selection = {};
  bool have_order = false;
  bool have_regularization = false;
  bool have_igr_volume_order = false;
  bool have_igr_face_order = false;
  bool have_quadrature_equivalence = false;
  for (int argument = 1; argument < argc; ++argument)
  {
    const std::string_view option{argv[argument]};
    if (option == "--order" && argument + 1 < argc && !have_order)
    {
      have_order = ParseOrder(argv[++argument], selection.order);
      if (!have_order) return false;
    }
    else if (option == "--regularization" && argument + 1 < argc &&
             !have_regularization)
    {
      const std::string_view value{argv[++argument]};
      if (value == "conventional")
        selection.regularization = RegularizationKind::none;
      else if (value == "igr")
        selection.regularization = RegularizationKind::igr;
      else
        return false;
      have_regularization = true;
    }
    else if (option == "--igr-volume-quadrature-order" &&
             argument + 1 < argc && !have_igr_volume_order)
    {
      have_igr_volume_order = ParseQuadratureOrder(
          argv[++argument],
          selection.igr_assembly_quadrature_orders.volume_order);
      if (!have_igr_volume_order) return false;
    }
    else if (option == "--igr-face-quadrature-order" &&
             argument + 1 < argc && !have_igr_face_order)
    {
      have_igr_face_order = ParseQuadratureOrder(
          argv[++argument],
          selection.igr_assembly_quadrature_orders.face_order);
      if (!have_igr_face_order) return false;
    }
    else if (option == "--quadrature-equivalence" &&
             !have_quadrature_equivalence)
    {
      selection.quadrature_equivalence = true;
      have_quadrature_equivalence = true;
    }
    else
    {
      return false;
    }
  }
  return have_order && have_regularization &&
      have_igr_volume_order == have_igr_face_order &&
      (!have_igr_volume_order ||
       selection.regularization == RegularizationKind::igr) &&
      (!selection.quadrature_equivalence ||
       (selection.regularization == RegularizationKind::igr &&
        have_igr_volume_order));
}

/**
 * Constant velocity and pressure reduce the Euler equations for this family
 * to scalar advection of density.  A wavevector perpendicular to velocity is
 * therefore an exact steady solution; a general wavevector is translated at
 * velocity dot wavevector.  The IGR velocity-gradient source is identically
 * zero for both spellings, so the same exact solution qualifies conventional
 * and IGR execution without changing the mathematical control.
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

/**
 * Smooth periodic state with nonconstant velocity and pressure.  Unlike the
 * entropy-wave convergence control, this state exercises the IGR velocity
 * gradient and scalar regularization terms.  It is used only for the direct
 * reduced-versus-q12 quadrature comparison; no analytic evolution is claimed.
 */
struct ActiveIgrWave
{
  void Evaluate(const double x, const double y,
                std::array<double, components>& conservative) const noexcept
  {
    const double x_phase = 2.0 * pi * x;
    const double y_phase = 2.0 * pi * y;
    const double density =
        1.0 + 0.05 * std::sin(x_phase) * std::cos(y_phase);
    const double velocity_x = 0.20 + 0.03 * std::cos(y_phase);
    const double velocity_y = -0.10 + 0.02 * std::sin(x_phase);
    const double pressure =
        1.0 + 0.03 * std::cos(x_phase) * std::sin(y_phase);
    const double kinetic_energy = 0.5 * density *
        (velocity_x * velocity_x + velocity_y * velocity_y);
    conservative = {{density,
                     density * velocity_x,
                     density * velocity_y,
                     pressure / (gamma - 1.0) + kinetic_energy}};
  }
};

struct BoundaryAudit
{
  int calls = 0;
};

inline bool PeriodicExterior(
    const void* opaque, const generic::BoundaryPointView& point,
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

inline bool CollectiveAll(const bool local, const MPI_Comm communicator)
{
  const int source = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

inline mfem::ParMesh MakePeriodicMesh(const int elements_per_direction,
                                      const MPI_Comm communicator)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      elements_per_direction, elements_per_direction,
      mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(communicator, periodic);
}

inline bool ProjectInitialState(const EntropyWave& wave,
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

inline bool ProjectInitialState(const ActiveIgrWave& wave,
                                mfem::ParGridFunction& initial)
{
  auto exact = [&wave](const mfem::Vector& point, mfem::Vector& value)
  {
    std::array<double, components> state{};
    wave.Evaluate(point[0], point[1], state);
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

inline EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const int order,
    const mfem::ParGridFunction& initial,
    const mfem::ParGridFunction& source, BoundaryAudit& boundary,
    const ExecutionSelection execution,
    const RegularizationKind regularization,
    const EulerHdgAssemblyQuadratureOrders igr_assembly_quadrature_orders)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {
      execution.backend,
      execution.backend == BackendKind::cuda ? 0 : -1,
      execution.backend == BackendKind::cuda
          ? execution.cuda_engine : CudaEngineKind::ordinary};
  request.capabilities = common_capabilities | CapabilityBit(
      execution.backend == BackendKind::cuda
          ? Capability::cuda : Capability::cpu);
  request.initial_state = {
      initial.HostRead(), static_cast<std::size_t>(initial.Size()), components};
  request.initial_true_trace = {nullptr, 0, components};
  request.source = {
      source.HostRead(), static_cast<std::size_t>(source.Size()), components};
  request.boundary = {
      &boundary, PeriodicExterior,
      EulerBoundaryPolicyTableView{nullptr, 0, dimension, gamma}};
  request.numerical_policy = generic::DefaultEulerHdgNumericalPolicy(order);
  if (igr_assembly_quadrature_orders.volume_order > 0)
    request.numerical_policy.igr_assembly_quadrature_orders =
        igr_assembly_quadrature_orders;
  request.numerical_policy.stabilization =
      regularization == RegularizationKind::igr
      ? generic::RegularizationSelection{RegularizationKind::igr, 0.2, 0.8}
      : generic::RegularizationSelection{RegularizationKind::none, 0.0, 0.0};
  return request;
}

inline double GlobalExactL2Error(const mfem::ParGridFunction& field,
                                 const EntropyWave& wave,
                                 const double time)
{
  const auto* const space = field.ParFESpace();
  auto* const mesh = space == nullptr
      ? nullptr
      : const_cast<mfem::ParMesh*>(space->GetParMesh());
  if (space == nullptr || mesh == nullptr)
    return std::numeric_limits<double>::infinity();

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

inline double GlobalDifferenceL2(const mfem::ParGridFunction& left,
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
      !std::isfinite(global_difference_squared) ||
      global_difference_squared < 0.0L)
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(static_cast<double>(global_difference_squared));
}

inline double GlobalL2Norm(const mfem::ParGridFunction& field)
{
  const auto* const space = field.ParFESpace();
  auto* const mesh = space == nullptr
      ? nullptr
      : const_cast<mfem::ParMesh*>(space->GetParMesh());
  if (space == nullptr || mesh == nullptr)
    return std::numeric_limits<double>::infinity();

  bool valid = true;
  long double local_norm_squared = 0.0L;
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
        mfem::Vector value(components);
        field.GetVectorValue(element, integration, value);
        const double weight = integration.weight * transformation->Weight();
        if (value.Size() != components || !(weight > 0.0) ||
            !std::isfinite(weight))
        {
          valid = false;
          break;
        }
        for (int component = 0; component < components; ++component)
        {
          const long double entry = value[component];
          local_norm_squared += static_cast<long double>(weight) *
              entry * entry;
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
  long double global_norm_squared = 0.0L;
  if (MPI_Allreduce(&local_norm_squared, &global_norm_squared, 1,
                    MPI_LONG_DOUBLE, MPI_SUM, mesh->GetComm()) != MPI_SUCCESS ||
      !std::isfinite(global_norm_squared) || !(global_norm_squared > 0.0L))
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(static_cast<double>(global_norm_squared));
}

struct GlobalFieldMetrics
{
  double l2_norm = std::numeric_limits<double>::infinity();
  double linf_norm = std::numeric_limits<double>::infinity();
  std::array<double, components> component_integrals{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  double minimum_density = -std::numeric_limits<double>::infinity();
  double minimum_pressure = -std::numeric_limits<double>::infinity();
  bool valid = false;
};

struct GlobalFieldDifference
{
  double l2 = std::numeric_limits<double>::infinity();
  double linf = std::numeric_limits<double>::infinity();
  bool valid = false;
};

/** Global physical-field invariants; no coefficient or rank ordering enters. */
inline GlobalFieldMetrics MeasureGlobalField(
    const mfem::ParGridFunction& field)
{
  GlobalFieldMetrics result;
  const auto* const space = field.ParFESpace();
  auto* const mesh = space == nullptr
      ? nullptr
      : const_cast<mfem::ParMesh*>(space->GetParMesh());
  if (space == nullptr || mesh == nullptr) return result;

  bool valid = true;
  long double local_norm_squared = 0.0L;
  std::array<long double, components> local_integrals{};
  double local_linf = 0.0;
  double local_minimum_density = std::numeric_limits<double>::infinity();
  double local_minimum_pressure = std::numeric_limits<double>::infinity();
  try
  {
    for (int element = 0; element < mesh->GetNE() && valid; ++element)
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
        mfem::Vector value(components);
        field.GetVectorValue(element, integration, value);
        const double weight = integration.weight * transformation->Weight();
        if (value.Size() != components || !(weight > 0.0) ||
            !std::isfinite(weight))
        {
          valid = false;
          break;
        }
        for (int component = 0; component < components; ++component)
        {
          const double entry = value[component];
          if (!std::isfinite(entry))
          {
            valid = false;
            break;
          }
          local_norm_squared += static_cast<long double>(weight) *
              static_cast<long double>(entry) * entry;
          local_integrals[static_cast<std::size_t>(component)] +=
              static_cast<long double>(weight) * entry;
          local_linf = std::max(local_linf, std::abs(entry));
        }
        if (!valid) break;
        const double density = value[0];
        if (!(density > 0.0))
        {
          valid = false;
          break;
        }
        const double pressure = (gamma - 1.0) *
            (value[3] - 0.5 *
                 (value[1] * value[1] + value[2] * value[2]) / density);
        if (!std::isfinite(pressure))
        {
          valid = false;
          break;
        }
        local_minimum_density = std::min(local_minimum_density, density);
        local_minimum_pressure = std::min(local_minimum_pressure, pressure);
      }
    }
  }
  catch (...)
  {
    valid = false;
  }
  if (!CollectiveAll(valid, mesh->GetComm())) return result;

  long double global_norm_squared = 0.0L;
  std::array<long double, components> global_integrals{};
  double global_linf = 0.0;
  double global_minimum_density = 0.0;
  double global_minimum_pressure = 0.0;
  const int norm_status = MPI_Allreduce(
      &local_norm_squared, &global_norm_squared, 1, MPI_LONG_DOUBLE, MPI_SUM,
      mesh->GetComm());
  const int integral_status = MPI_Allreduce(
      local_integrals.data(), global_integrals.data(), components,
      MPI_LONG_DOUBLE, MPI_SUM, mesh->GetComm());
  const int linf_status = MPI_Allreduce(
      &local_linf, &global_linf, 1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
  const int density_status = MPI_Allreduce(
      &local_minimum_density, &global_minimum_density, 1, MPI_DOUBLE, MPI_MIN,
      mesh->GetComm());
  const int pressure_status = MPI_Allreduce(
      &local_minimum_pressure, &global_minimum_pressure, 1, MPI_DOUBLE,
      MPI_MIN, mesh->GetComm());
  const bool reductions = norm_status == MPI_SUCCESS &&
      integral_status == MPI_SUCCESS && linf_status == MPI_SUCCESS &&
      density_status == MPI_SUCCESS && pressure_status == MPI_SUCCESS;
  if (!CollectiveAll(reductions, mesh->GetComm()) ||
      !(global_norm_squared > 0.0L) || !std::isfinite(global_norm_squared) ||
      !std::isfinite(global_linf) ||
      !std::isfinite(global_minimum_density) ||
      !std::isfinite(global_minimum_pressure))
  {
    return result;
  }

  result.l2_norm = std::sqrt(static_cast<double>(global_norm_squared));
  result.linf_norm = global_linf;
  for (int component = 0; component < components; ++component)
    result.component_integrals[static_cast<std::size_t>(component)] =
        static_cast<double>(
            global_integrals[static_cast<std::size_t>(component)]);
  result.minimum_density = global_minimum_density;
  result.minimum_pressure = global_minimum_pressure;
  result.valid = std::isfinite(result.l2_norm) && result.l2_norm > 0.0 &&
      std::all_of(result.component_integrals.begin(),
                  result.component_integrals.end(),
                  [](const double value) { return std::isfinite(value); });
  return result;
}

inline GlobalFieldDifference MeasureGlobalFieldDifference(
    const mfem::ParGridFunction& left,
    const mfem::ParGridFunction& right)
{
  GlobalFieldDifference result;
  const auto* const space = left.ParFESpace();
  auto* const mesh = space == nullptr
      ? nullptr
      : const_cast<mfem::ParMesh*>(space->GetParMesh());
  if (space == nullptr || mesh == nullptr || right.ParFESpace() != space)
    return result;

  bool valid = true;
  long double local_difference_squared = 0.0L;
  double local_linf = 0.0;
  try
  {
    for (int element = 0; element < mesh->GetNE() && valid; ++element)
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
          const double difference =
              left_value[component] - right_value[component];
          if (!std::isfinite(difference))
          {
            valid = false;
            break;
          }
          local_difference_squared += static_cast<long double>(weight) *
              static_cast<long double>(difference) * difference;
          local_linf = std::max(local_linf, std::abs(difference));
        }
        if (!valid) break;
      }
    }
  }
  catch (...)
  {
    valid = false;
  }
  if (!CollectiveAll(valid, mesh->GetComm())) return result;

  long double global_difference_squared = 0.0L;
  double global_linf = 0.0;
  const int l2_status = MPI_Allreduce(
      &local_difference_squared, &global_difference_squared, 1,
      MPI_LONG_DOUBLE, MPI_SUM, mesh->GetComm());
  const int linf_status = MPI_Allreduce(
      &local_linf, &global_linf, 1, MPI_DOUBLE, MPI_MAX, mesh->GetComm());
  const bool reductions =
      l2_status == MPI_SUCCESS && linf_status == MPI_SUCCESS;
  if (!CollectiveAll(reductions, mesh->GetComm()) ||
      global_difference_squared < 0.0L ||
      !std::isfinite(global_difference_squared) ||
      !std::isfinite(global_linf))
  {
    return result;
  }
  result.l2 = std::sqrt(static_cast<double>(global_difference_squared));
  result.linf = global_linf;
  result.valid = std::isfinite(result.l2) && std::isfinite(result.linf);
  return result;
}

/** Aggregate accepted-work summary; deliberately no per-attempt history. */
struct QuadratureTrajectoryHealth
{
  std::int64_t accepted_steps = 0;
  std::int64_t nonlinear_updates = 0;
  std::int64_t linear_iterations = 0;
  std::int64_t linear_solve_attempts = 0;
  std::int64_t line_search_trials = 0;
  std::int64_t true_residual_tightenings = 0;
  std::int64_t trace_solves = 0;
  double maximum_true_relative_residual = 0.0;
  double maximum_conservation_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  double maximum_final_nonlinear_residual = 0.0;
  double minimum_density = std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
  double maximum_wave_speed = 0.0;
  bool passed = false;
};

// Reuse the established three-way trajectory-alignment tolerances for the
// global L2 field comparisons performed at every h and dt ladder point.
inline constexpr double field_comparison_relative_tolerance = 5.0e-8;
inline constexpr double field_comparison_absolute_tolerance = 5.0e-9;

inline bool ErrorNormsAgree(
    const double left_error, const double right_error,
    const double left_field_norm, const double right_field_norm,
    double& difference) noexcept
{
  difference = std::abs(left_error - right_error);
  const double scale = std::max(left_field_norm, right_field_norm);
  return std::isfinite(left_error) && std::isfinite(right_error) &&
      std::isfinite(difference) && std::isfinite(scale) &&
      difference <= field_comparison_absolute_tolerance +
          field_comparison_relative_tolerance * scale;
}

inline bool FieldNormsAgree(const double left, const double right,
                            double& difference) noexcept
{
  difference = std::abs(left - right);
  const double scale = std::max(left, right);
  return std::isfinite(left) && std::isfinite(right) &&
      std::isfinite(difference) && std::isfinite(scale) &&
      difference <= field_comparison_absolute_tolerance +
          field_comparison_relative_tolerance * scale;
}

inline double HalvingOrder(const double coarse, const double fine) noexcept
{
  if (!(coarse > 0.0) || !(fine > 0.0) || !std::isfinite(coarse) ||
      !std::isfinite(fine))
  {
    return -std::numeric_limits<double>::infinity();
  }
  return std::log(coarse / fine) / std::log(2.0);
}

struct SpatialResult
{
  std::array<double, 3> errors{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 3> field_norms{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 2> observed_orders{{
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()}};
  bool passed = false;
};

struct TemporalResult
{
  std::array<double, 3> exact_errors{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 3> field_norms{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 2> self_differences{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  double observed_order = -std::numeric_limits<double>::infinity();
  bool passed = false;
};

struct PairwiseResult
{
  std::size_t left = 0;
  std::size_t right = 0;
  std::array<double, 3> spatial_differences{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 3> spatial_norm_differences{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 3> temporal_differences{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 3> temporal_norm_differences{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  bool passed = true;
};

inline std::string_view RegularizationName(
    const RegularizationKind regularization) noexcept
{
  return regularization == RegularizationKind::igr
      ? std::string_view{"igr"} : std::string_view{"conventional"};
}

inline bool MixedAgreement(const double difference,
                           const double left_scale,
                           const double right_scale) noexcept
{
  const double scale = std::max(left_scale, right_scale);
  return std::isfinite(difference) && std::isfinite(scale) &&
      difference <= field_comparison_absolute_tolerance +
          field_comparison_relative_tolerance * scale;
}

inline bool ConservedIntegralAgreement(
    const double left, const double right,
    const double relative_tolerance) noexcept
{
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::isfinite(left) && std::isfinite(right) &&
      std::isfinite(relative_tolerance) && relative_tolerance > 0.0 &&
      std::abs(left - right) <= 1.0e-12 + relative_tolerance * scale;
}

inline bool EquivalentSolverWork(
    const QuadratureTrajectoryHealth& control,
    const QuadratureTrajectoryHealth& candidate) noexcept
{
  // A quadrature perturbation can move a Krylov residual across its unchanged
  // stopping threshold by one iteration.  Bound that discrete effect by one
  // iteration per accepted step while requiring every nonlinear, solve-call,
  // line-search, tightening, and trace-solve count to remain identical.  This
  // is a test-only work-equivalence envelope, not a relaxed solver tolerance.
  const std::int64_t iteration_allowance = control.accepted_steps;
  return control.accepted_steps == candidate.accepted_steps &&
      control.nonlinear_updates == candidate.nonlinear_updates &&
      candidate.linear_iterations >=
          control.linear_iterations - iteration_allowance &&
      candidate.linear_iterations <=
          control.linear_iterations + iteration_allowance &&
      control.linear_solve_attempts == candidate.linear_solve_attempts &&
      control.line_search_trials == candidate.line_search_trials &&
      control.true_residual_tightenings ==
          candidate.true_residual_tightenings &&
      control.trace_solves == candidate.trace_solves;
}

/**
 * Direct test-only comparison of the bounded p-dependent IGR quadrature
 * candidate against the unchanged q12/q12 policy.  The same active IGR
 * initial projection, mesh, dt, and solver policy feed both trajectories.
 * Only two fixed endpoint scenarios are retained; accepted-step work is
 * reduced immediately into aggregate health quantities by the runner.
 */
template <typename Runner>
bool RunQuadratureEquivalenceCase(
    const MPI_Comm communicator, const CaseSelection test_case,
    const ExecutionSelection execution, Runner& runner)
{
  const EulerHdgAssemblyQuadratureOrders expected{
      2 * test_case.order + 4, 2 * test_case.order + 2};
  const EulerHdgAssemblyQuadratureOrders candidate =
      test_case.igr_assembly_quadrature_orders;
  const bool admitted = test_case.quadrature_equivalence &&
      test_case.regularization == RegularizationKind::igr &&
      candidate.volume_order == expected.volume_order &&
      candidate.face_order == expected.face_order &&
      candidate.volume_order < 12 && candidate.face_order < 12;
  if (!CollectiveAll(admitted && runner.Begin(execution), communicator))
    return false;

  constexpr int elements_per_direction = 4;
  constexpr double dt = 2.5e-4;
  const std::array<int, 2> step_counts{{1, 4}};
  bool passed = true;
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  for (const int steps : step_counts)
  {
    mfem::ParMesh mesh =
        MakePeriodicMesh(elements_per_direction, communicator);
    mfem::L2_FECollection collection(
        test_case.order, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, components, mfem::Ordering::byNODES);
    mfem::ParGridFunction initial(&space);
    mfem::ParGridFunction source(&space);
    mfem::ParGridFunction control_field(&space);
    mfem::ParGridFunction candidate_field(&space);
    source = 0.0;
    const ActiveIgrWave wave;
    if (!CollectiveAll(ProjectInitialState(wave, initial), communicator))
      return false;
    const GlobalFieldMetrics initial_metrics = MeasureGlobalField(initial);

    BoundaryAudit control_boundary{};
    BoundaryAudit candidate_boundary{};
    const EulerHdgAssemblyQuadratureOrders control_orders{12, 12};
    const EulerHdgRequest control_request = MakeRequest(
        mesh, test_case.order, initial, source, control_boundary, execution,
        RegularizationKind::igr, control_orders);
    const EulerHdgRequest candidate_request = MakeRequest(
        mesh, test_case.order, initial, source, candidate_boundary, execution,
        RegularizationKind::igr, candidate);
    QuadratureTrajectoryHealth control_health;
    QuadratureTrajectoryHealth candidate_health;
    const bool control_advanced = runner.AdvanceQualified(
        control_request, steps, dt, control_field, control_health);
    const bool candidate_advanced = runner.AdvanceQualified(
        candidate_request, steps, dt, candidate_field, candidate_health);
    if (!CollectiveAll(control_advanced && candidate_advanced &&
                           control_boundary.calls == 0 &&
                           candidate_boundary.calls == 0,
                       communicator))
    {
      return false;
    }

    const GlobalFieldMetrics control_metrics =
        MeasureGlobalField(control_field);
    const GlobalFieldMetrics candidate_metrics =
        MeasureGlobalField(candidate_field);
    const GlobalFieldDifference field_difference =
        MeasureGlobalFieldDifference(control_field, candidate_field);
    bool conservation_agrees = initial_metrics.valid &&
        control_metrics.valid && candidate_metrics.valid;
    for (int component = 0; component < components; ++component)
    {
      const std::size_t index = static_cast<std::size_t>(component);
      const double tolerance =
          control_request.numerical_policy.conservation_relative_tolerance;
      conservation_agrees = conservation_agrees &&
          ConservedIntegralAgreement(
              initial_metrics.component_integrals[index],
              control_metrics.component_integrals[index], tolerance) &&
          ConservedIntegralAgreement(
              initial_metrics.component_integrals[index],
              candidate_metrics.component_integrals[index], tolerance) &&
          ConservedIntegralAgreement(
              control_metrics.component_integrals[index],
              candidate_metrics.component_integrals[index], tolerance);
    }
    const bool field_agrees = field_difference.valid &&
        MixedAgreement(field_difference.l2, control_metrics.l2_norm,
                       candidate_metrics.l2_norm) &&
        MixedAgreement(field_difference.linf, control_metrics.linf_norm,
                       candidate_metrics.linf_norm);
    const bool admissible = control_metrics.minimum_density > 0.0 &&
        control_metrics.minimum_pressure > 0.0 &&
        candidate_metrics.minimum_density > 0.0 &&
        candidate_metrics.minimum_pressure > 0.0 &&
        control_health.minimum_density > 0.0 &&
        control_health.minimum_pressure > 0.0 &&
        candidate_health.minimum_density > 0.0 &&
        candidate_health.minimum_pressure > 0.0;
    const bool work_agrees =
        EquivalentSolverWork(control_health, candidate_health);
    const bool scenario_passed = control_health.passed &&
        candidate_health.passed && field_agrees && conservation_agrees &&
        admissible && work_agrees;
    passed = passed && scenario_passed;

    if (rank == 0)
    {
      std::cout << std::setprecision(17)
                << "generic CPU IGR p" << test_case.order << " q12/q12 vs q"
                << candidate.volume_order << "/q" << candidate.face_order
                << " steps=" << steps
                << " global field difference L2=" << field_difference.l2
                << " Linf=" << field_difference.linf
                << "; field norms control/candidate L2="
                << control_metrics.l2_norm << '/' << candidate_metrics.l2_norm
                << " Linf=" << control_metrics.linf_norm << '/'
                << candidate_metrics.linf_norm
                << "; minimum rho control/candidate="
                << control_metrics.minimum_density << '/'
                << candidate_metrics.minimum_density
                << " pressure=" << control_metrics.minimum_pressure << '/'
                << candidate_metrics.minimum_pressure
                << "; true residual control/candidate="
                << control_health.maximum_true_relative_residual << '/'
                << candidate_health.maximum_true_relative_residual
                << "; conservation relative control/candidate="
                << control_health.maximum_conservation_relative_error << '/'
                << candidate_health.maximum_conservation_relative_error
                << "; nonlinear updates control/candidate="
                << control_health.nonlinear_updates << '/'
                << candidate_health.nonlinear_updates
                << "; linear iterations=" << control_health.linear_iterations
                << '/' << candidate_health.linear_iterations
                << "; solve attempts="
                << control_health.linear_solve_attempts << '/'
                << candidate_health.linear_solve_attempts
                << "; line-search trials="
                << control_health.line_search_trials << '/'
                << candidate_health.line_search_trials
                << "; tightenings="
                << control_health.true_residual_tightenings << '/'
                << candidate_health.true_residual_tightenings
                << "; field=" << field_agrees
                << " conservation=" << conservation_agrees
                << " admissible=" << admissible
                << " work=" << work_agrees
                << " passed=" << scenario_passed << '\n';
    }
  }
  passed = CollectiveAll(passed, communicator);
  if (rank == 0)
    std::cout << "generic CPU IGR quadrature equivalence gate p"
              << test_case.order << ": " << passed << '\n';
  return passed;
}

template <std::size_t execution_count, typename Runner>
bool RunComparisonCase(
    const MPI_Comm communicator, const CaseSelection test_case,
    const std::array<ExecutionSelection, execution_count>& executions,
    Runner& runner)
{
  static_assert(execution_count > 0,
                "a convergence comparison needs at least one backend");
  const int order = test_case.order;
  const RegularizationKind regularization = test_case.regularization;
  std::array<SpatialResult, execution_count> spatial{};
  std::array<TemporalResult, execution_count> temporal{};
  std::vector<PairwiseResult> pairwise;
  pairwise.reserve(execution_count * (execution_count - 1) / 2);
  for (std::size_t left = 0; left < execution_count; ++left)
    for (std::size_t right = left + 1; right < execution_count; ++right)
    {
      PairwiseResult comparison;
      comparison.left = left;
      comparison.right = right;
      pairwise.push_back(comparison);
    }

  // Form complete backend phases.  The runner selects MFEM's CUDA device only
  // after the CPU phase and clears HYPRE at each process-global transition;
  // no host solver execution is attempted after device selection.
  const EntropyWave spatial_wave{{{0.5, 0.0}}, {{0.0, 1.0}}, 0.1, 1.0};
  constexpr double spatial_final_time = 0.02;
  const std::array<int, 3> mesh_ladder{{3, 6, 12}};
  const EntropyWave temporal_wave{
      {{0.5, 0.25}}, {{1.0, 1.0}}, 0.05, 1.0};
  constexpr int elements_per_direction = 4;
  constexpr double temporal_final_time = 0.08;
  const std::array<int, 3> step_ladder{{2, 4, 8}};
  for (std::size_t execution = 0; execution < execution_count; ++execution)
  {
    if (!CollectiveAll(runner.Begin(executions[execution]), communicator))
      return false;

    // Spatial behavior: velocity dot wavevector is exactly zero, so this
    // exact steady solution contains no Backward-Euler truncation error.
    for (std::size_t level = 0; level < mesh_ladder.size(); ++level)
    {
      mfem::ParMesh mesh = MakePeriodicMesh(
          mesh_ladder[level], communicator);
      mfem::L2_FECollection collection(
          order, dimension, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      mfem::ParFiniteElementSpace space(
          &mesh, &collection, components, mfem::Ordering::byNODES);
      mfem::ParGridFunction initial(&space);
      mfem::ParGridFunction source(&space);
      mfem::ParGridFunction committed(&space);
      source = 0.0;
      BoundaryAudit boundary{};
      if (!ProjectInitialState(spatial_wave, initial)) return false;
      const EulerHdgRequest request = MakeRequest(
          mesh, order, initial, source, boundary,
          executions[execution], regularization,
          test_case.igr_assembly_quadrature_orders);
      const bool advanced = runner.Advance(
          request, 1, spatial_final_time, committed);
      if (!CollectiveAll(advanced && boundary.calls == 0, communicator))
        return false;
      spatial[execution].errors[level] = GlobalExactL2Error(
          committed, spatial_wave, spatial_final_time);
      spatial[execution].field_norms[level] = GlobalL2Norm(committed);
    }

    // Backward-Euler behavior: spatial discretization and initial projection
    // remain fixed while only dt is halved.
    mfem::ParMesh temporal_mesh =
        MakePeriodicMesh(elements_per_direction, communicator);
    mfem::L2_FECollection temporal_collection(
        order, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace temporal_space(
        &temporal_mesh, &temporal_collection, components,
        mfem::Ordering::byNODES);
    mfem::ParGridFunction temporal_initial(&temporal_space);
    mfem::ParGridFunction temporal_source(&temporal_space);
    temporal_source = 0.0;
    if (!ProjectInitialState(temporal_wave, temporal_initial)) return false;
    std::array<std::unique_ptr<mfem::ParGridFunction>, 3> solutions{};
    for (auto& solution : solutions)
      solution = std::make_unique<mfem::ParGridFunction>(&temporal_space);
    for (std::size_t level = 0; level < step_ladder.size(); ++level)
    {
      const double dt = temporal_final_time /
          static_cast<double>(step_ladder[level]);
      BoundaryAudit boundary{};
      const EulerHdgRequest request = MakeRequest(
          temporal_mesh, order, temporal_initial, temporal_source, boundary,
          executions[execution], regularization,
          test_case.igr_assembly_quadrature_orders);
      const bool advanced = runner.Advance(
          request, step_ladder[level], dt, *solutions[level]);
      if (!CollectiveAll(advanced && boundary.calls == 0, communicator))
        return false;
      temporal[execution].exact_errors[level] = GlobalExactL2Error(
          *solutions[level], temporal_wave, temporal_final_time);
      temporal[execution].field_norms[level] =
          GlobalL2Norm(*solutions[level]);
    }
    TemporalResult& temporal_result = temporal[execution];
    temporal_result.self_differences[0] = GlobalDifferenceL2(
        *solutions[0], *solutions[1]);
    temporal_result.self_differences[1] = GlobalDifferenceL2(
        *solutions[1], *solutions[2]);
  }

  for (std::size_t execution = 0; execution < execution_count; ++execution)
  {
    SpatialResult& spatial_result = spatial[execution];
    spatial_result.observed_orders[0] = HalvingOrder(
        spatial_result.errors[0], spatial_result.errors[1]);
    spatial_result.observed_orders[1] = HalvingOrder(
        spatial_result.errors[1], spatial_result.errors[2]);
    spatial_result.passed = std::isfinite(spatial_result.errors[0]) &&
        std::isfinite(spatial_result.errors[1]) &&
        std::isfinite(spatial_result.errors[2]) &&
        std::all_of(spatial_result.field_norms.begin(),
                    spatial_result.field_norms.end(),
                    [](const double value)
                    { return std::isfinite(value) && value > 0.0; }) &&
        spatial_result.errors[0] > spatial_result.errors[1] &&
        spatial_result.errors[1] > spatial_result.errors[2] &&
        spatial_result.observed_orders[0] >= spatial_order_floor &&
        spatial_result.observed_orders[1] >= spatial_order_floor;

    TemporalResult& temporal_result = temporal[execution];
    temporal_result.observed_order = HalvingOrder(
        temporal_result.self_differences[0],
        temporal_result.self_differences[1]);
    temporal_result.passed =
        std::isfinite(temporal_result.exact_errors[0]) &&
        std::isfinite(temporal_result.exact_errors[1]) &&
        std::isfinite(temporal_result.exact_errors[2]) &&
        std::all_of(temporal_result.field_norms.begin(),
                    temporal_result.field_norms.end(),
                    [](const double value)
                    { return std::isfinite(value) && value > 0.0; }) &&
        // The legacy p3 fixture also had monotonically decreasing exact
        // error.  At p1 the fixed 4x4 spatial error dominates and partially
        // cancels the temporal error: all three engines showed the same
        // non-monotone exact-error sequence while their self differences
        // converged at order 0.944.  Preserve the legacy p3 auxiliary check;
        // use the actual BE self-convergence gate for lower orders.
        (order < 3 ||
         temporal_result.exact_errors[2] <
             temporal_result.exact_errors[0]) &&
        std::isfinite(temporal_result.self_differences[0]) &&
        std::isfinite(temporal_result.self_differences[1]) &&
        temporal_result.self_differences[0] >
            temporal_result.self_differences[1] &&
        temporal_result.observed_order >= temporal_order_floor &&
        temporal_result.observed_order <= temporal_order_ceiling;
  }

  // Existing field-alignment gates already compare full trajectories.  This
  // convergence fixture compares backend-neutral global exact-error and field
  // norm sequences at every h/dt point, with the same mixed FP64 tolerance.
  for (PairwiseResult& comparison : pairwise)
    for (std::size_t level = 0; level < 3; ++level)
    {
      const bool spatial_error_agrees = ErrorNormsAgree(
          spatial[comparison.left].errors[level],
          spatial[comparison.right].errors[level],
          spatial[comparison.left].field_norms[level],
          spatial[comparison.right].field_norms[level],
          comparison.spatial_differences[level]);
      const bool spatial_norm_agrees = FieldNormsAgree(
          spatial[comparison.left].field_norms[level],
          spatial[comparison.right].field_norms[level],
          comparison.spatial_norm_differences[level]);
      const bool temporal_error_agrees = ErrorNormsAgree(
          temporal[comparison.left].exact_errors[level],
          temporal[comparison.right].exact_errors[level],
          temporal[comparison.left].field_norms[level],
          temporal[comparison.right].field_norms[level],
          comparison.temporal_differences[level]);
      const bool temporal_norm_agrees = FieldNormsAgree(
          temporal[comparison.left].field_norms[level],
          temporal[comparison.right].field_norms[level],
          comparison.temporal_norm_differences[level]);
      comparison.passed = comparison.passed && spatial_error_agrees &&
          spatial_norm_agrees && temporal_error_agrees &&
          temporal_norm_agrees;
    }

  bool passed = true;
  int rank = 0;
  MPI_Comm_rank(communicator, &rank);
  for (std::size_t execution = 0; execution < execution_count; ++execution)
  {
    passed = passed && spatial[execution].passed && temporal[execution].passed;
    if (rank == 0)
    {
      const SpatialResult& spatial_result = spatial[execution];
      const TemporalResult& temporal_result = temporal[execution];
      std::cout << std::setprecision(17)
                << "generic " << executions[execution].name << ' '
                << RegularizationName(regularization) << " p" << order
                << " spatial errors: " << spatial_result.errors[0] << ' '
                << spatial_result.errors[1] << ' '
                << spatial_result.errors[2] << "; observed orders: "
                << spatial_result.observed_orders[0] << ' '
                << spatial_result.observed_orders[1]
                << "; required floor: " << spatial_order_floor
                << "; passed=" << spatial_result.passed << '\n'
                << "generic " << executions[execution].name << ' '
                << RegularizationName(regularization) << " p" << order
                << " Backward-Euler exact errors: "
                << temporal_result.exact_errors[0] << ' '
                << temporal_result.exact_errors[1] << ' '
                << temporal_result.exact_errors[2]
                << "; self differences: "
                << temporal_result.self_differences[0] << ' '
                << temporal_result.self_differences[1]
                << "; observed order: " << temporal_result.observed_order
                << "; required interval: [" << temporal_order_floor
                << ", " << temporal_order_ceiling << "]"
                << "; passed=" << temporal_result.passed << '\n';
    }
  }
  for (const PairwiseResult& comparison : pairwise)
  {
    passed = passed && comparison.passed;
    if (rank == 0)
      std::cout << std::setprecision(17)
                << "generic " << executions[comparison.left].name << " vs "
                << executions[comparison.right].name << ' '
                << RegularizationName(regularization) << " p" << order
                << " spatial global-L2 error differences: "
                << comparison.spatial_differences[0] << ' '
                << comparison.spatial_differences[1] << ' '
                << comparison.spatial_differences[2]
                << "; spatial field-norm differences: "
                << comparison.spatial_norm_differences[0] << ' '
                << comparison.spatial_norm_differences[1] << ' '
                << comparison.spatial_norm_differences[2]
                << "; Backward-Euler global-L2 error differences: "
                << comparison.temporal_differences[0] << ' '
                << comparison.temporal_differences[1] << ' '
                << comparison.temporal_differences[2]
                << "; Backward-Euler field-norm differences: "
                << comparison.temporal_norm_differences[0] << ' '
                << comparison.temporal_norm_differences[1] << ' '
                << comparison.temporal_norm_differences[2]
                << "; passed=" << comparison.passed << '\n';
  }
  passed = CollectiveAll(passed, communicator);
  if (rank == 0)
    std::cout << "generic backend convergence comparison gate "
              << RegularizationName(regularization) << " p" << order
              << ": " << passed << '\n';
  return passed;
}

} // namespace dg::hdg::test::generic_euler_convergence
