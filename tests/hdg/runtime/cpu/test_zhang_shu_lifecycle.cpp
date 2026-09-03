#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/euler/zhang_shu_u4.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace dg::hdg::runtime::cpu::detail
{

struct CpuQualificationTestAccess
{
  static bool AuditLastConservation(
      const CpuNumericsCore& core, double& maximum_error,
      double& maximum_relative_error, bool& gate) noexcept
  {
    return core.AuditLastBorrowedEulerConservation(
        maximum_error, maximum_relative_error, gate);
  }

  static bool AuditCommittedClosure(
      CpuNumericsCore& core, double& residual, double& tolerance) noexcept
  {
    return core.AuditCommittedScalarClosure(residual, tolerance);
  }
};

} // namespace dg::hdg::runtime::cpu::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgPositivityLimiterKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::EulerHdgZhangShuPointSet;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cpu::CpuAttemptEvidence;
using dg::hdg::runtime::cpu::CpuAttemptStatus;
using dg::hdg::runtime::cpu::CpuStateSnapshot;
using dg::hdg::runtime::cpu::detail::CpuNumericsCore;

constexpr int components = 4;
constexpr int order = 1;
constexpr int q1_scalar_dofs = 4;
constexpr std::uint32_t cpu_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cpu) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

struct BoundaryContext
{
  int calls = 0;
};

bool PeriodicExterior(const void* opaque,
                      const dg::hdg::generic::BoundaryPointView& point,
                      const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(
      const_cast<void*>(opaque));
  if (context == nullptr || point.space_dimension != 2 ||
      interior == nullptr || exterior == nullptr)
  {
    return false;
  }
  ++context->calls;
  std::copy_n(interior, components, exterior);
  return true;
}

mfem::ParMesh MakePeriodicMesh(const int elements_per_direction = 3)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      elements_per_direction, elements_per_direction,
      mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[0][1] = 0.0;
  translations[1][0] = 0.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakePhysicalMesh(const int elements_per_direction = 3)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      elements_per_direction, elements_per_direction,
      mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_WORLD, mesh);
}

int DecodeDof(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

struct BorrowedFields
{
  std::vector<double> initial{};
  std::vector<double> source{};
  std::size_t scalar_count = 0;
  int elements = 0;
};

BorrowedFields MakeFields(
    mfem::ParMesh& mesh,
    const std::array<std::array<double, q1_scalar_dofs>, components>& values)
{
  BorrowedFields fields;
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  fields.elements = mesh.GetNE();
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() / components);
  fields.initial.assign(static_cast<std::size_t>(space.GetVSize()), 0.0);
  fields.source.assign(fields.initial.size(), 0.0);
  std::vector<unsigned char> assigned(fields.initial.size(), 0);
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* const finite_element = space.GetFE(element);
    mfem::Array<int> vdofs;
    space.GetElementVDofs(element, vdofs);
    if (finite_element == nullptr ||
        finite_element->GetDof() != q1_scalar_dofs ||
        vdofs.Size() != components * q1_scalar_dofs)
    {
      return {};
    }
    for (int component = 0; component < components; ++component)
      for (int scalar = 0; scalar < q1_scalar_dofs; ++scalar)
      {
        double sign = 1.0;
        const int dof = DecodeDof(
            vdofs[component * q1_scalar_dofs + scalar], sign);
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= fields.initial.size())
        {
          return {};
        }
        const std::size_t index = static_cast<std::size_t>(dof);
        fields.initial[index] = sign * values[component][scalar];
        assigned[index] = 1;
      }
  }
  if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end())
    return {};
  return fields;
}

BorrowedFields MakeStationaryContactFields(
    mfem::ParMesh& mesh, const int elements_per_direction)
{
  BorrowedFields fields;
  if (elements_per_direction <= 0 ||
      mesh.GetNE() != elements_per_direction * elements_per_direction)
  {
    return fields;
  }
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  fields.elements = mesh.GetNE();
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() / components);
  fields.initial.assign(static_cast<std::size_t>(space.GetVSize()), 0.0);
  fields.source.assign(fields.initial.size(), 0.0);
  std::vector<unsigned char> assigned(fields.initial.size(), 0);
  constexpr double density_amplitude = 0.75;
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* const finite_element = space.GetFE(element);
    mfem::ElementTransformation* const transformation =
        mesh.GetElementTransformation(element);
    mfem::Array<int> vdofs;
    space.GetElementVDofs(element, vdofs);
    if (!finite_element || !transformation ||
        finite_element->GetDof() != q1_scalar_dofs ||
        vdofs.Size() != components * q1_scalar_dofs)
    {
      return {};
    }
    mfem::IntegrationPoint reference_center;
    reference_center.Set2(0.5, 0.5);
    mfem::Vector physical_center(2);
    transformation->Transform(reference_center, physical_center);
    const int element_x = std::clamp(
        static_cast<int>(std::floor(
            physical_center[0] * elements_per_direction)),
        0, elements_per_direction - 1);
    const int element_y = std::clamp(
        static_cast<int>(std::floor(
            physical_center[1] * elements_per_direction)),
        0, elements_per_direction - 1);
    const double parity = (element_x + element_y) % 2 == 0 ? 1.0 : -1.0;
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    if (nodes.GetNPoints() != q1_scalar_dofs) return {};
    for (int scalar = 0; scalar < q1_scalar_dofs; ++scalar)
    {
      const mfem::IntegrationPoint& point = nodes.IntPoint(scalar);
      const double density = 1.0 + density_amplitude * parity *
          (1.0 - 2.0 * point.x) * (1.0 - 2.0 * point.y);
      const std::array<double, components> state{
          density, 0.0, 0.0, 2.5};
      for (int component = 0; component < components; ++component)
      {
        double sign = 1.0;
        const int dof = DecodeDof(
            vdofs[component * q1_scalar_dofs + scalar], sign);
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= fields.initial.size())
        {
          return {};
        }
        const std::size_t index = static_cast<std::size_t>(dof);
        fields.initial[index] = sign * state[component];
        assigned[index] = 1;
      }
    }
  }
  if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end())
    return {};
  return fields;
}

constexpr int hidden_corner_order = 3;
constexpr int hidden_corner_scalar_dofs = 16;
constexpr int hidden_corner_quadrature_order = 12;
constexpr int hidden_corner_pressure_bisections = 24;
constexpr double hidden_corner_pressure_delta = 1.0e-2;
constexpr double hidden_corner_density_floor = 1.0e-8;
constexpr double hidden_corner_pressure_floor = 1.0e-6;

BorrowedFields MakeHiddenCornerPressureFields(mfem::ParMesh& mesh)
{
  BorrowedFields fields;
  mfem::L2_FECollection collection(
      hidden_corner_order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  fields.elements = mesh.GetNE();
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() / components);
  fields.initial.assign(static_cast<std::size_t>(space.GetVSize()), 0.0);
  fields.source.assign(fields.initial.size(), 0.0);
  std::vector<unsigned char> assigned(fields.initial.size(), 0);
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    const mfem::FiniteElement* const finite_element = space.GetFE(element);
    mfem::Array<int> vdofs;
    space.GetElementVDofs(element, vdofs);
    if (!finite_element ||
        finite_element->GetDof() != hidden_corner_scalar_dofs ||
        vdofs.Size() != components * hidden_corner_scalar_dofs)
    {
      return {};
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    if (nodes.GetNPoints() != hidden_corner_scalar_dofs) return {};
    for (int scalar = 0; scalar < hidden_corner_scalar_dofs; ++scalar)
    {
      const mfem::IntegrationPoint& point = nodes.IntPoint(scalar);
      const double pressure =
          point.x + point.y - hidden_corner_pressure_delta;
      const std::array<double, components> state{
          1.0, 0.0, 0.0, pressure / 0.4};
      for (int component = 0; component < components; ++component)
      {
        double sign = 1.0;
        const int dof = DecodeDof(
            vdofs[component * hidden_corner_scalar_dofs + scalar], sign);
        if (dof < 0 ||
            static_cast<std::size_t>(dof) >= fields.initial.size())
        {
          return {};
        }
        const std::size_t index = static_cast<std::size_t>(dof);
        fields.initial[index] = sign * state[component];
        assigned[index] = 1;
      }
    }
  }
  if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end())
    return {};
  return fields;
}

struct HiddenCornerAudit
{
  std::array<double, components> mean{};
  double minimum_selected_pressure =
      std::numeric_limits<double>::infinity();
  double minimum_vertex_pressure =
      std::numeric_limits<double>::infinity();
  bool selected_points_admissible = false;
  bool vertices_admissible = false;
};

bool AuditHiddenCornerField(
    mfem::ParMesh& mesh, const std::vector<double>& coefficients,
    HiddenCornerAudit& audit)
{
  audit = {};
  bool ready = mesh.GetNE() == 1;
  try
  {
    mfem::L2_FECollection collection(
        hidden_corner_order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, components, mfem::Ordering::byNODES);
    const mfem::FiniteElement* const finite_element =
        ready ? space.GetFE(0) : nullptr;
    mfem::ElementTransformation* const transformation =
        ready ? mesh.GetElementTransformation(0) : nullptr;
    mfem::Array<int> vdofs;
    if (ready) space.GetElementVDofs(0, vdofs);
    ready = ready && finite_element && transformation &&
        finite_element->GetDof() == hidden_corner_scalar_dofs &&
        vdofs.Size() == components * hidden_corner_scalar_dofs &&
        coefficients.size() == static_cast<std::size_t>(space.GetVSize());

    const auto evaluate = [&](const mfem::IntegrationPoint& point,
                              std::array<double, components>& state)
    {
      if (!ready) return false;
      mfem::Vector shape(hidden_corner_scalar_dofs);
      finite_element->CalcShape(point, shape);
      state = {};
      for (int scalar = 0; scalar < hidden_corner_scalar_dofs; ++scalar)
        for (int component = 0; component < components; ++component)
        {
          double sign = 1.0;
          const int dof = DecodeDof(
              vdofs[component * hidden_corner_scalar_dofs + scalar], sign);
          if (dof < 0 ||
              static_cast<std::size_t>(dof) >= coefficients.size())
          {
            return false;
          }
          state[component] += sign * shape[scalar] *
              coefficients[static_cast<std::size_t>(dof)];
        }
      return std::all_of(
          state.begin(), state.end(),
          [](const double value) { return std::isfinite(value); });
    };
    const auto admit = [&](const mfem::IntegrationPoint& point,
                           double& minimum_pressure)
    {
      std::array<double, components> state{};
      double pressure = std::numeric_limits<double>::quiet_NaN();
      const bool pressure_ready = evaluate(point, state) &&
          dg::hdg::euler::zhang_shu::Pressure(
              state.data(), 1.4, pressure);
      if (pressure_ready)
        minimum_pressure = std::min(minimum_pressure, pressure);
      return pressure_ready && state[0] >= hidden_corner_density_floor &&
          pressure >= hidden_corner_pressure_floor;
    };

    bool selected = ready;
    double measure = 0.0;
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, hidden_corner_quadrature_order);
    for (int q = 0; q < volume_rule.GetNPoints() && ready; ++q)
    {
      const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
      std::array<double, components> state{};
      transformation->SetIntPoint(&point);
      const double weight = point.weight * transformation->Weight();
      ready = std::isfinite(weight) && weight > 0.0 &&
          evaluate(point, state);
      if (!ready) break;
      measure += weight;
      for (int component = 0; component < components; ++component)
        audit.mean[component] += weight * state[component];
      selected = admit(point, audit.minimum_selected_pressure) && selected;
    }
    ready = ready && std::isfinite(measure) && measure > 0.0;
    for (double& value : audit.mean)
    {
      if (ready) value /= measure;
      ready = ready && std::isfinite(value);
    }

    const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
        mfem::Geometry::SEGMENT, hidden_corner_quadrature_order);
    for (int q = 0; q < face_rule.GetNPoints() && ready; ++q)
    {
      const double s = face_rule.IntPoint(q).x;
      std::array<mfem::IntegrationPoint, 4> points{};
      points[0].Set2(s, 0.0);
      points[1].Set2(1.0, s);
      points[2].Set2(s, 1.0);
      points[3].Set2(0.0, s);
      for (const mfem::IntegrationPoint& point : points)
        selected = admit(point, audit.minimum_selected_pressure) && selected;
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    ready = ready && nodes.GetNPoints() == hidden_corner_scalar_dofs;
    for (int scalar = 0; scalar < nodes.GetNPoints() && ready; ++scalar)
      selected = admit(
          nodes.IntPoint(scalar), audit.minimum_selected_pressure) && selected;

    const mfem::IntegrationRule* const vertices =
        mfem::Geometries.GetVertices(mfem::Geometry::SQUARE);
    bool vertices_admissible = ready && vertices &&
        vertices->GetNPoints() == 4;
    for (int vertex = 0;
         vertices && vertex < vertices->GetNPoints() && ready; ++vertex)
    {
      vertices_admissible = admit(
          vertices->IntPoint(vertex), audit.minimum_vertex_pressure) &&
          vertices_admissible;
    }
    audit.selected_points_admissible = selected;
    audit.vertices_admissible = vertices_admissible;
  }
  catch (...)
  {
    ready = false;
  }
  return ready &&
      std::isfinite(audit.minimum_selected_pressure) &&
      std::isfinite(audit.minimum_vertex_pressure);
}

std::array<std::array<double, q1_scalar_dofs>, components>
ConstantState()
{
  std::array<std::array<double, q1_scalar_dofs>, components> values{};
  for (int scalar = 0; scalar < q1_scalar_dofs; ++scalar)
  {
    values[0][scalar] = 1.0;
    values[1][scalar] = 0.05;
    values[2][scalar] = -0.02;
    values[3][scalar] = 2.60;
  }
  return values;
}

EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const BorrowedFields& fields,
    BoundaryContext& boundary, const bool igr,
    const int polynomial_order = order)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = polynomial_order;
  request.backend = {BackendKind::cpu, -1, CudaEngineKind::ordinary};
  request.capabilities = cpu_capabilities;
  request.initial_state = {
      fields.initial.empty() ? nullptr : fields.initial.data(),
      fields.initial.size(), components};
  request.initial_true_trace = {nullptr, 0, components};
  request.source = {
      fields.source.empty() ? nullptr : fields.source.data(),
      fields.source.size(), components};
  request.boundary = {
      &boundary, PeriodicExterior, {nullptr, 0, 0, 0.0}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(polynomial_order);
  if (igr)
  {
    request.numerical_policy.stabilization =
        {RegularizationKind::igr, 0.2, 0.4};
  }
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  return request;
}

bool SameBytes(const std::vector<double>& left,
               const std::vector<double>& right) noexcept
{
  return left.size() == right.size() &&
      (left.empty() ||
       std::memcmp(left.data(), right.data(),
                   left.size() * sizeof(double)) == 0);
}

struct SuccessfulStep
{
  CpuAttemptEvidence evidence{};
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::vector<double> interior{};
  std::vector<double> trace{};
  std::vector<double> theta_rho{};
  std::vector<double> theta_p{};
  bool conservation_audited = false;
  bool conservation_gate = false;
  double conservation_error = std::numeric_limits<double>::infinity();
  double conservation_relative_error =
      std::numeric_limits<double>::infinity();
  bool closure_audited = false;
  double closure_residual = std::numeric_limits<double>::infinity();
  double closure_tolerance = std::numeric_limits<double>::infinity();
  bool strict_algebraic_rejected = false;
  double normalized_algebraic_correction =
      std::numeric_limits<double>::infinity();
};

bool RunSuccessfulStep(const EulerHdgRequest& request,
                       const BorrowedFields& fields,
                       const bool audit_closure,
                       SuccessfulStep& result,
                       const double dt = 1.0e-8,
                       const double relative_tolerance = 1.0e-5,
                       const double relaxed_algebraic_fraction = 0.25)
{
  result = {};
  CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return false;

  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(dt, result.coordinates, generation) ||
      generation == 0)
  {
    const CpuAttemptEvidence& evidence = core.evidence().last_attempt;
    std::cerr << "CPU Zhang--Shu prepare failed: status="
              << static_cast<int>(evidence.status) << " phase="
              << static_cast<int>(evidence.failure_phase) << " calls="
              << evidence.work.positivity_limiter_calls << " active="
              << evidence.work.positivity_limiter_active_calls
              << " projections="
              << evidence.work.positivity_algebraic_projections
              << " rebuilds="
              << evidence.work.common_shift_operator_rebuilds << '\n';
    return false;
  }
  const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      assessment_policy{1.0e-8, relative_tolerance, 0.25};
  dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
  bool assessed = core.AssessPreparedBorrowedEuler(
      generation, assessment_policy, assessment);
  const bool strict_primary_gates = assessed && assessment.finite_gate &&
      assessment.solver_gate && assessment.residual_gate &&
      assessment.admissibility_gate &&
      assessment.algebraic_closure_gate;
  result.strict_algebraic_rejected = strict_primary_gates &&
      !assessment.algebraic_correction_gate;
  if (result.strict_algebraic_rejected &&
      relaxed_algebraic_fraction > 0.25)
  {
    const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy relaxed_policy{
        1.0e-8, relative_tolerance, relaxed_algebraic_fraction};
    assessed = core.AssessPreparedBorrowedEuler(
        generation, relaxed_policy, assessment);
  }
  if (!assessed || !assessment.finite_gate || !assessment.solver_gate ||
      !assessment.residual_gate || !assessment.admissibility_gate ||
      !assessment.algebraic_closure_gate ||
      !assessment.algebraic_correction_gate)
  {
    std::cerr << "CPU Zhang--Shu assessment failed: finite="
              << assessment.finite_gate << " solver="
              << assessment.solver_gate << " residual="
              << assessment.residual_gate << " admissible="
              << assessment.admissibility_gate << " closure="
              << assessment.algebraic_closure_gate << " algebraic="
              << assessment.algebraic_correction_gate << " differential="
              << assessment.normalized_differential_error
              << " algebraic_norm="
              << assessment.normalized_algebraic_correction << '\n';
    static_cast<void>(core.AbortBorrowedEuler(generation,
                                               result.coordinates));
    return false;
  }
  result.normalized_algebraic_correction =
      assessment.normalized_algebraic_correction;
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(
          generation, result.coordinates, maximum_wave_speed, stable_dt) ||
      !std::isfinite(maximum_wave_speed) || !(maximum_wave_speed > 0.0) ||
      !std::isfinite(stable_dt) || !(stable_dt > 0.0))
  {
    return false;
  }

  result.evidence = core.evidence().last_attempt;
  result.conservation_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditLastConservation(
              core, result.conservation_error,
              result.conservation_relative_error,
              result.conservation_gate);
  if (!result.conservation_audited || !result.conservation_gate ||
      !std::isfinite(result.conservation_error) ||
      !std::isfinite(result.conservation_relative_error))
  {
    return false;
  }
  result.interior.assign(fields.initial.size(), 0.0);
  const bool limiter_enabled =
      request.numerical_policy.positivity_limiter.kind ==
          EulerHdgPositivityLimiterKind::zhang_shu;
  if (limiter_enabled)
  {
    result.theta_rho.assign(static_cast<std::size_t>(fields.elements), 0.0);
    result.theta_p.assign(static_cast<std::size_t>(fields.elements), 0.0);
  }
  if (!core.ExportCommittedFields(
          {result.interior.data(), nullptr, fields.scalar_count,
           limiter_enabled ? result.theta_rho.data() : nullptr,
           limiter_enabled ? result.theta_p.data() : nullptr,
           limiter_enabled ? static_cast<std::size_t>(fields.elements) : 0}))
  {
    return false;
  }
  const CpuStateSnapshot snapshot = core.Snapshot(result.coordinates);
  if ((snapshot.trace.data == nullptr && snapshot.trace.size != 0) ||
      snapshot.trace.size == 0)
  {
    return false;
  }
  result.trace.assign(snapshot.trace.data,
                      snapshot.trace.data + snapshot.trace.size);
  if (audit_closure)
  {
    result.closure_audited =
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditCommittedClosure(core, result.closure_residual,
                                  result.closure_tolerance);
  }
  return true;
}

bool CheckDefaultOffAndInactiveIdentity()
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  const BorrowedFields fields = MakeFields(mesh, ConstantState());
  if (fields.initial.empty()) return false;

  BoundaryContext off_boundary{};
  const EulerHdgRequest off_request =
      MakeRequest(mesh, fields, off_boundary, false);
  SuccessfulStep off{};
  if (!RunSuccessfulStep(off_request, fields, false, off)) return false;

  BoundaryContext inactive_boundary{};
  EulerHdgRequest inactive_request =
      MakeRequest(mesh, fields, inactive_boundary, false);
  inactive_request.numerical_policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      1.0e-8,
      1.0e-8,
      12};
  SuccessfulStep inactive{};
  if (!RunSuccessfulStep(inactive_request, fields, false, inactive))
    return false;

  const auto zero_limiter_work = [](const CpuAttemptEvidence& evidence)
  {
    return evidence.work.positivity_limiter_calls == 0 &&
        evidence.work.positivity_limiter_active_calls == 0 &&
        evidence.work.positivity_limited_elements == 0 &&
        evidence.work.positivity_algebraic_projections == 0 &&
        evidence.work.common_shift_operator_rebuilds == 0 &&
        evidence.maximum_positivity_limiter_correction_norm == 0.0 &&
        evidence.maximum_positivity_algebraic_correction_norm == 0.0;
  };
  const auto inactive_limiter_work = [](const CpuAttemptEvidence& evidence)
  {
    return evidence.work.positivity_limiter_calls == 6 &&
        evidence.work.positivity_limiter_active_calls == 0 &&
        evidence.work.positivity_limited_elements == 0 &&
        evidence.work.positivity_algebraic_projections == 0 &&
        evidence.work.common_shift_operator_rebuilds == 0 &&
        evidence.maximum_positivity_limiter_correction_norm == 0.0 &&
        evidence.maximum_positivity_algebraic_correction_norm == 0.0;
  };
  const bool neutral_diagnostics =
      inactive.theta_rho.size() ==
          static_cast<std::size_t>(fields.elements) &&
      inactive.theta_p.size() == static_cast<std::size_t>(fields.elements) &&
      std::all_of(inactive.theta_rho.begin(), inactive.theta_rho.end(),
                  [](const double value) { return value == 1.0; }) &&
      std::all_of(inactive.theta_p.begin(), inactive.theta_p.end(),
                  [](const double value) { return value == 1.0; });
  return off.evidence.committed && inactive.evidence.committed &&
      off.conservation_audited && off.conservation_gate &&
      inactive.conservation_audited && inactive.conservation_gate &&
      zero_limiter_work(off.evidence) &&
      inactive_limiter_work(inactive.evidence) &&
      off.theta_rho.empty() && off.theta_p.empty() &&
      neutral_diagnostics &&
      SameBytes(off.interior, inactive.interior) &&
      SameBytes(off.trace, inactive.trace) &&
      off.coordinates.time == inactive.coordinates.time &&
      off.coordinates.step == inactive.coordinates.step &&
      off.coordinates.generation == inactive.coordinates.generation &&
      off_boundary.calls == 0 && inactive_boundary.calls == 0;
}

bool CheckCumulativeAcceptedHistory()
{
  constexpr int elements_per_direction = 4;
  constexpr double density_floor = 0.80;
  constexpr double first_dt = 1.0e-8;
  constexpr double second_dt = 1.0e-16;
  const auto fail = [](const char* const stage)
  {
    std::cerr << "CPU Zhang--Shu cumulative-history failure: "
              << stage << '\n';
    return false;
  };
  mfem::ParMesh mesh = MakePeriodicMesh(elements_per_direction);
  const BorrowedFields fields =
      MakeStationaryContactFields(mesh, elements_per_direction);
  if (fields.initial.empty() || fields.elements <= 0 ||
      fields.scalar_count == 0 ||
      *std::min_element(fields.initial.begin(),
                        fields.initial.begin() +
                            static_cast<std::ptrdiff_t>(fields.scalar_count)) >=
          density_floor)
  {
    return fail("fixture");
  }

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(mesh, fields, boundary, false);
  request.numerical_policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      density_floor,
      0.10,
      12};
  CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return fail("initialize");

  dg::hdg::runtime::StateCoordinates coordinates{};
  std::vector<double> scratch(fields.initial.size(), 0.0);
  const std::size_t element_count =
      static_cast<std::size_t>(fields.elements);
  const auto export_committed = [&](std::vector<double>& theta_rho,
                                    std::vector<double>& theta_p)
  {
    theta_rho.assign(element_count, 0.0);
    theta_p.assign(element_count, 0.0);
    return core.ExportCommittedFields(
        {scratch.data(), nullptr, fields.scalar_count,
         theta_rho.data(), theta_p.data(), element_count});
  };
  const auto export_prepared = [&](const std::uint64_t generation,
                                   std::vector<double>& theta_rho,
                                   std::vector<double>& theta_p)
  {
    theta_rho.assign(element_count, 0.0);
    theta_p.assign(element_count, 0.0);
    return core.ExportPreparedFields(
        generation,
        {scratch.data(), nullptr, fields.scalar_count,
         theta_rho.data(), theta_p.data(), element_count});
  };
  const auto neutral = [](const std::vector<double>& values)
  {
    return std::all_of(values.begin(), values.end(),
                       [](const double value) { return value == 1.0; });
  };
  const auto valid = [](const std::vector<double>& values)
  {
    return std::all_of(values.begin(), values.end(),
                       [](const double value)
                       {
                         return std::isfinite(value) && value >= 0.0 &&
                             value <= 1.0;
                       });
  };
  const auto assess_for_commit = [&](const std::uint64_t generation)
  {
    constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy policy{
        1.0e-8, 1.0e-4, 0.25};
    dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
    return core.AssessPreparedBorrowedEuler(
               generation, policy, assessment) &&
        assessment.finite_gate && assessment.solver_gate &&
        assessment.residual_gate && assessment.admissibility_gate &&
        assessment.algebraic_closure_gate &&
        assessment.algebraic_correction_gate;
  };
  const auto commit = [&](const std::uint64_t generation)
  {
    double maximum_wave_speed = 0.0;
    double stable_dt = 0.0;
    return core.CommitBorrowedEuler(
               generation, coordinates, maximum_wave_speed, stable_dt) &&
        std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
        std::isfinite(stable_dt) && stable_dt > 0.0;
  };

  std::vector<double> initial_theta_rho;
  std::vector<double> initial_theta_p;
  if (!export_committed(initial_theta_rho, initial_theta_p) ||
      !neutral(initial_theta_rho) || !neutral(initial_theta_p))
  {
    return fail("initial-neutral-export");
  }

  // First expose an active candidate and abort it.  Prepared export may see
  // the attempt, but committed export must remain the neutral initial history.
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(first_dt, coordinates, generation) ||
      generation == 0)
  {
    return fail("aborted-active-prepare");
  }
  std::vector<double> aborted_theta_rho;
  std::vector<double> aborted_theta_p;
  std::vector<double> committed_during_abort_rho;
  std::vector<double> committed_during_abort_p;
  const bool aborted_candidate_active =
      export_prepared(generation, aborted_theta_rho, aborted_theta_p) &&
      std::any_of(aborted_theta_rho.begin(), aborted_theta_rho.end(),
                  [](const double value) { return value < 1.0; }) &&
      valid(aborted_theta_rho) && valid(aborted_theta_p);
  const bool committed_hidden_while_prepared =
      export_committed(committed_during_abort_rho,
                       committed_during_abort_p) &&
      neutral(committed_during_abort_rho) &&
      neutral(committed_during_abort_p);
  const bool aborted = core.AbortBorrowedEuler(generation, coordinates);
  const CpuAttemptEvidence aborted_evidence = core.evidence().last_attempt;
  const bool aborted_work_active =
      aborted_evidence.work.positivity_limiter_calls == 6 &&
      aborted_evidence.work.positivity_limiter_active_calls >= 1;
  if (!aborted_candidate_active || !committed_hidden_while_prepared ||
      !aborted || !aborted_work_active)
  {
    std::cerr << "CPU Zhang--Shu cumulative-history abort evidence: calls="
              << aborted_evidence.work.positivity_limiter_calls
              << " active="
              << aborted_evidence.work.positivity_limiter_active_calls
              << " candidate_active=" << aborted_candidate_active
              << " committed_hidden="
              << committed_hidden_while_prepared
              << " aborted=" << aborted << '\n';
    return fail("aborted-active-visibility");
  }
  std::vector<double> committed_after_abort_rho;
  std::vector<double> committed_after_abort_p;
  if (!export_committed(committed_after_abort_rho,
                        committed_after_abort_p) ||
      !neutral(committed_after_abort_rho) ||
      !neutral(committed_after_abort_p) || coordinates.step != 0 ||
      coordinates.time != 0.0)
  {
    return fail("post-abort-neutral-export");
  }

  // Retry the same active step and commit it.  The accepted minima become the
  // cumulative committed history.
  generation = 0;
  if (!core.PrepareBorrowedEuler(first_dt, coordinates, generation) ||
      generation == 0)
  {
    return fail("accepted-active-prepare");
  }
  std::vector<double> first_candidate_rho;
  std::vector<double> first_candidate_p;
  const bool first_prepared_export = export_prepared(
      generation, first_candidate_rho, first_candidate_p);
  const bool first_retry_matches = first_prepared_export &&
      SameBytes(aborted_theta_rho, first_candidate_rho) &&
      SameBytes(aborted_theta_p, first_candidate_p);
  const bool first_assessed = assess_for_commit(generation);
  const bool first_committed = first_assessed && commit(generation);
  const CpuAttemptEvidence first_evidence = core.evidence().last_attempt;
  if (!first_prepared_export ||
      first_evidence.work.positivity_limiter_calls != 6 ||
      first_evidence.work.positivity_limiter_active_calls < 1 ||
      !first_retry_matches || !first_assessed || !first_committed)
  {
    std::cerr << "CPU Zhang--Shu cumulative-history first commit: calls="
              << first_evidence.work.positivity_limiter_calls
              << " active="
              << first_evidence.work.positivity_limiter_active_calls
              << " export=" << first_prepared_export
              << " retry_match=" << first_retry_matches
              << " assessed=" << first_assessed
              << " committed=" << first_committed << '\n';
    return fail("accepted-active-commit");
  }
  std::vector<double> first_history_rho;
  std::vector<double> first_history_p;
  if (!export_committed(first_history_rho, first_history_p) ||
      !SameBytes(first_candidate_rho, first_history_rho) ||
      !SameBytes(first_candidate_p, first_history_p) ||
      !valid(first_history_rho) || !valid(first_history_p) ||
      std::none_of(first_history_rho.begin(), first_history_rho.end(),
                   [](const double value) { return value < 1.0; }))
  {
    return fail("accepted-active-history");
  }

  // The limited checkerboard is a continuous stationary contact.  Use a
  // roundoff-safe positive step so its next accepted RODAS attempt exercises
  // all six limiter calls without a reconstruction-sized perturbation at the
  // floor.  Candidate diagnostics must nevertheless retain the prior accepted
  // minima, and committing this inactive step must preserve them.
  generation = 0;
  if (!core.PrepareBorrowedEuler(second_dt, coordinates, generation) ||
      generation == 0)
  {
    return fail("accepted-inactive-prepare");
  }
  std::vector<double> second_candidate_rho;
  std::vector<double> second_candidate_p;
  std::vector<double> committed_during_second_rho;
  std::vector<double> committed_during_second_p;
  const bool second_prepared_export = export_prepared(
      generation, second_candidate_rho, second_candidate_p);
  const bool second_candidate_retained = second_prepared_export &&
      SameBytes(first_history_rho, second_candidate_rho) &&
      SameBytes(first_history_p, second_candidate_p);
  const bool second_committed_export = export_committed(
      committed_during_second_rho, committed_during_second_p);
  const bool prior_commit_retained = second_committed_export &&
      SameBytes(first_history_rho, committed_during_second_rho) &&
      SameBytes(first_history_p, committed_during_second_p);
  const bool second_assessed = assess_for_commit(generation);
  const bool second_committed = second_assessed && commit(generation);
  const CpuAttemptEvidence second_evidence = core.evidence().last_attempt;
  const bool second_work_inactive =
      second_evidence.work.positivity_limiter_calls == 6 &&
      second_evidence.work.positivity_limiter_active_calls == 0 &&
      second_evidence.work.positivity_limited_elements == 0 &&
      second_evidence.work.positivity_algebraic_projections == 0;
  if (!second_work_inactive || !second_candidate_retained ||
      !prior_commit_retained || !second_assessed || !second_committed)
  {
    std::cerr << "CPU Zhang--Shu cumulative-history second commit: calls="
              << second_evidence.work.positivity_limiter_calls
              << " active="
              << second_evidence.work.positivity_limiter_active_calls
              << " limited="
              << second_evidence.work.positivity_limited_elements
              << " projections="
              << second_evidence.work.positivity_algebraic_projections
              << " candidate_retained=" << second_candidate_retained
              << " prior_commit_retained=" << prior_commit_retained
              << " assessed=" << second_assessed
              << " committed=" << second_committed << '\n';
    return fail("accepted-inactive-commit");
  }
  std::vector<double> final_history_rho;
  std::vector<double> final_history_p;
  const bool final_ready =
      export_committed(final_history_rho, final_history_p) &&
      SameBytes(first_history_rho, final_history_rho) &&
      SameBytes(first_history_p, final_history_p) &&
      coordinates.step == 2 && coordinates.generation == 2 &&
      coordinates.time == first_dt + second_dt && boundary.calls == 0;
  return final_ready || fail("final-committed-history");
}

bool CheckActiveAdmissibleMeanAndIgrClosure()
{
  // A physical boundary is intentional: entry limiting changes boundary
  // point values even though it preserves each cell mean, so the frozen
  // boundary-Jacobian conservation witness must use the admitted base state.
  mfem::ParMesh mesh = MakePhysicalMesh();
  auto values = ConstantState();
  for (int scalar = 0; scalar < q1_scalar_dofs; ++scalar)
  {
    values[0][scalar] = scalar % 2 == 0 ? 0.75 : 1.25;
  }
  const BorrowedFields fields = MakeFields(mesh, values);
  if (fields.initial.empty() || fields.elements <= 0) return false;

  constexpr double density_floor = 0.80;
  constexpr double pressure_floor = 0.10;
  const double cell_density_mean =
      0.25 * (values[0][0] + values[0][1] +
              values[0][2] + values[0][3]);
  const bool input_witness = cell_density_mean > density_floor &&
      *std::min_element(values[0].begin(), values[0].end()) < density_floor;

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(mesh, fields, boundary, true);
  request.numerical_policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      density_floor,
      pressure_floor,
      12};
  SuccessfulStep step{};
  if (!input_witness || !RunSuccessfulStep(request, fields, true, step))
    return false;

  const CpuAttemptEvidence& evidence = step.evidence;
  const bool diagnostics_active =
      step.theta_rho.size() == static_cast<std::size_t>(fields.elements) &&
      step.theta_p.size() == static_cast<std::size_t>(fields.elements) &&
      std::any_of(step.theta_rho.begin(), step.theta_rho.end(),
                  [](const double value) { return value < 1.0; }) &&
      std::all_of(step.theta_rho.begin(), step.theta_rho.end(),
                  [](const double value)
                  { return std::isfinite(value) && value >= 0.0 && value <= 1.0; }) &&
      std::all_of(step.theta_p.begin(), step.theta_p.end(),
                  [](const double value)
                  { return std::isfinite(value) && value >= 0.0 && value <= 1.0; });
  bool consumer_coefficients_admissible =
      step.interior.size() == fields.initial.size();
  for (std::size_t scalar = 0;
       scalar < fields.scalar_count && consumer_coefficients_admissible;
       ++scalar)
  {
    const double density = step.interior[scalar];
    const double momentum_x =
        step.interior[fields.scalar_count + scalar];
    const double momentum_y =
        step.interior[2 * fields.scalar_count + scalar];
    const double energy =
        step.interior[3 * fields.scalar_count + scalar];
    const double pressure = 0.4 *
        (energy - 0.5 *
            (momentum_x * momentum_x + momentum_y * momentum_y) / density);
    consumer_coefficients_admissible = std::isfinite(density) &&
        density >= density_floor && std::isfinite(pressure) &&
        pressure >= pressure_floor;
  }
  return evidence.committed && evidence.status == CpuAttemptStatus::success &&
      evidence.work.positivity_limiter_calls == 6 &&
      evidence.work.positivity_limiter_active_calls >= 1 &&
      evidence.work.positivity_limiter_active_calls <=
          evidence.work.positivity_limiter_calls &&
      evidence.work.positivity_limited_elements >=
          static_cast<std::uint64_t>(fields.elements) &&
      evidence.work.positivity_algebraic_projections ==
          evidence.work.positivity_limiter_active_calls &&
      evidence.work.common_shift_operator_builds ==
          1 + evidence.work.common_shift_operator_rebuilds &&
      evidence.work.common_shift_operator_reuses == 3 &&
      evidence.maximum_positivity_limiter_correction_norm > 0.0 &&
      std::isfinite(
          evidence.maximum_positivity_algebraic_correction_norm) &&
      evidence.maximum_positivity_algebraic_correction_norm > 0.0 &&
      diagnostics_active &&
      evidence.finite_gate && evidence.admissibility_gate &&
      evidence.recovery_gate && evidence.nonlinear_gate &&
      evidence.true_residual_gate && consumer_coefficients_admissible &&
      step.closure_audited && std::isfinite(step.closure_residual) &&
      std::isfinite(step.closure_tolerance) &&
      step.closure_residual <= step.closure_tolerance &&
      boundary.calls > 0;
}

bool CheckActiveCausalStageAndIgrSeeding()
{
  constexpr double dt = 1.0e-6;
  constexpr double density_floor = 0.999;
  constexpr double source_amplitude = 2500.0;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeFields(mesh, ConstantState());
  if (fields.initial.empty() || fields.source.size() != fields.initial.size())
    return false;

  // The Q1 Gauss--Legendre weights are equal, so this density source has
  // exactly zero cell mean.  Entry and the first two causal stage predictors
  // remain coherent, while a later predictor crosses the selected-point
  // floor and exercises limited-U4 projection from a coherent algebraic seed.
  bool source_ready = true;
  try
  {
    mfem::L2_FECollection collection(
        order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, components, mfem::Ordering::byNODES);
    for (int element = 0; element < mesh.GetNE() && source_ready; ++element)
    {
      const mfem::FiniteElement* const finite_element = space.GetFE(element);
      mfem::Array<int> vdofs;
      space.GetElementVDofs(element, vdofs);
      source_ready = finite_element != nullptr &&
          finite_element->GetDof() == q1_scalar_dofs &&
          vdofs.Size() == components * q1_scalar_dofs;
      for (int scalar = 0;
           scalar < q1_scalar_dofs && source_ready; ++scalar)
      {
        double sign = 1.0;
        const int dof = DecodeDof(vdofs[scalar], sign);
        source_ready = dof >= 0 &&
            static_cast<std::size_t>(dof) < fields.source.size();
        if (source_ready)
        {
          const double value = scalar % 2 == 0
              ? -source_amplitude : source_amplitude;
          fields.source[static_cast<std::size_t>(dof)] = sign * value;
        }
      }
    }
  }
  catch (...)
  {
    source_ready = false;
  }
  int local_source_ready = source_ready ? 1 : 0;
  int global_source_ready = 0;
  if (MPI_Allreduce(&local_source_ready, &global_source_ready, 1, MPI_INT,
                    MPI_MIN, mesh.GetComm()) != MPI_SUCCESS ||
      global_source_ready != 1)
  {
    return false;
  }

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(mesh, fields, boundary, true);
  request.numerical_policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      density_floor,
      1.0e-8,
      12};
  SuccessfulStep step{};
  if (!RunSuccessfulStep(
          request, fields, true, step, dt, 1.0e-2, 0.25))
    return false;

  const CpuAttemptEvidence& evidence = step.evidence;
  const bool diagnostics_active =
      step.theta_rho.size() == static_cast<std::size_t>(fields.elements) &&
      step.theta_p.size() == static_cast<std::size_t>(fields.elements) &&
      (std::any_of(step.theta_rho.begin(), step.theta_rho.end(),
                   [](const double value) { return value < 1.0; }) ||
       std::any_of(step.theta_p.begin(), step.theta_p.end(),
                   [](const double value) { return value < 1.0; }));
  return evidence.committed && evidence.status == CpuAttemptStatus::success &&
      evidence.work.positivity_limiter_calls == 6 &&
      evidence.work.positivity_limiter_active_calls >= 1 &&
      evidence.work.positivity_algebraic_projections ==
          evidence.work.positivity_limiter_active_calls &&
      evidence.work.common_shift_operator_rebuilds >= 1 &&
      evidence.work.common_shift_operator_builds ==
          1 + evidence.work.common_shift_operator_rebuilds &&
      evidence.work.common_shift_operator_reuses == 3 &&
      !step.strict_algebraic_rejected &&
      std::isfinite(step.normalized_algebraic_correction) &&
      step.normalized_algebraic_correction >= 0.0 &&
      step.normalized_algebraic_correction <= 0.25 &&
      evidence.maximum_positivity_limiter_correction_norm > 0.0 &&
      diagnostics_active &&
      std::isfinite(evidence.maximum_positivity_algebraic_correction_norm) &&
      evidence.finite_gate && evidence.admissibility_gate &&
      evidence.recovery_gate && evidence.nonlinear_gate &&
      evidence.true_residual_gate && step.conservation_audited &&
      step.conservation_gate && step.closure_audited &&
      std::isfinite(step.closure_residual) &&
      step.closure_residual <= step.closure_tolerance;
}

bool CheckP3ConsumerVertexPressureAndMeanPreservation()
{
  mfem::ParMesh mesh = MakePhysicalMesh(1);
  const BorrowedFields fields = MakeHiddenCornerPressureFields(mesh);
  HiddenCornerAudit input{};
  if (fields.initial.empty() || fields.elements != 1 ||
      fields.scalar_count != hidden_corner_scalar_dofs ||
      !AuditHiddenCornerField(mesh, fields.initial, input))
  {
    return false;
  }

  double mean_pressure = 0.0;
  const bool hidden_corner_witness = input.selected_points_admissible &&
      !input.vertices_admissible &&
      input.minimum_selected_pressure > hidden_corner_pressure_floor &&
      input.minimum_vertex_pressure < hidden_corner_pressure_floor &&
      dg::hdg::euler::zhang_shu::Pressure(
          input.mean.data(), 1.4, mean_pressure) &&
      mean_pressure > hidden_corner_pressure_floor;

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, fields, boundary, false, hidden_corner_order);
  request.numerical_policy.conventional_assembly_quadrature_order =
      hidden_corner_quadrature_order;
  request.numerical_policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      hidden_corner_density_floor,
      hidden_corner_pressure_floor,
      hidden_corner_pressure_bisections};
  SuccessfulStep step{};
  const bool advanced = hidden_corner_witness && RunSuccessfulStep(
      request, fields, false, step, 1.0e-16, 1.0e-5, 0.25);
  HiddenCornerAudit admitted{};
  const bool output_audited = advanced &&
      AuditHiddenCornerField(mesh, step.interior, admitted);

  bool mean_preserved = output_audited;
  for (int component = 0; component < components && mean_preserved;
       ++component)
  {
    const double scale = std::max(
        {1.0, std::abs(input.mean[component]),
         std::abs(admitted.mean[component])});
    mean_preserved = std::abs(
        input.mean[component] - admitted.mean[component]) <= 2.0e-12 * scale;
  }

  const bool theta_ready = step.theta_rho.size() == 1 &&
      step.theta_p.size() == 1 && step.theta_rho[0] == 1.0 &&
      std::isfinite(step.theta_p[0]) && step.theta_p[0] >= 0.0 &&
      step.theta_p[0] < 1.0;
  const double exact_pressure_theta = theta_ready
      ? (mean_pressure - hidden_corner_pressure_floor) /
          (mean_pressure - input.minimum_vertex_pressure)
      : 0.0;
  const double bisection_width = std::ldexp(
      1.0, -hidden_corner_pressure_bisections);
  const bool conservative_pressure_bracket = theta_ready &&
      step.theta_p[0] <= exact_pressure_theta + 64.0 *
          std::numeric_limits<double>::epsilon() &&
      step.theta_p[0] >= exact_pressure_theta - bisection_width - 1.0e-12;

  const CpuAttemptEvidence& evidence = step.evidence;
  const bool passed = advanced && output_audited &&
      admitted.selected_points_admissible && admitted.vertices_admissible &&
      admitted.minimum_vertex_pressure >= hidden_corner_pressure_floor &&
      mean_preserved && conservative_pressure_bracket &&
      evidence.committed && evidence.status == CpuAttemptStatus::success &&
      evidence.work.positivity_limiter_calls == 6 &&
      evidence.work.positivity_limiter_active_calls >= 1 &&
      evidence.work.positivity_limited_elements >= 1 &&
      evidence.work.positivity_algebraic_projections ==
          evidence.work.positivity_limiter_active_calls &&
      evidence.maximum_positivity_limiter_correction_norm > 0.0 &&
      evidence.finite_gate && evidence.admissibility_gate &&
      evidence.recovery_gate && evidence.true_residual_gate;
  if (!passed)
  {
    std::cerr << "CPU Zhang--Shu p3 consumer-vertex failure: witness="
              << hidden_corner_witness << " advanced=" << advanced
              << " output_audited=" << output_audited
              << " input_selected=" << input.selected_points_admissible
              << " input_vertex=" << input.vertices_admissible
              << " input_vertex_p=" << input.minimum_vertex_pressure
              << " output_vertex=" << admitted.vertices_admissible
              << " output_vertex_p=" << admitted.minimum_vertex_pressure
              << " mean_preserved=" << mean_preserved
              << " theta_rho="
              << (step.theta_rho.empty() ? -1.0 : step.theta_rho[0])
              << " theta_p="
              << (step.theta_p.empty() ? -1.0 : step.theta_p[0])
              << " exact_theta=" << exact_pressure_theta
              << " calls=" << evidence.work.positivity_limiter_calls
              << " active="
              << evidence.work.positivity_limiter_active_calls << '\n';
  }
  return passed;
}

bool CheckInadmissibleMeanRejectsWithoutRepair()
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  auto values = ConstantState();
  values[0].fill(0.70);
  const BorrowedFields fields = MakeFields(mesh, values);
  if (fields.initial.empty()) return false;

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(mesh, fields, boundary, false);
  request.numerical_policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      0.80,
      0.10,
      12};
  CpuNumericsCore core;
  dg::hdg::runtime::StateCoordinates coordinates{};
  if (!core.InitializeBorrowedEuler(request)) return false;
  const CpuStateSnapshot before = core.Snapshot(coordinates);
  if (before.interior.data == nullptr || before.trace.data == nullptr)
    return false;
  const std::vector<double> interior_before(
      before.interior.data, before.interior.data + before.interior.size);
  const std::vector<double> trace_before(
      before.trace.data, before.trace.data + before.trace.size);

  std::uint64_t generation = 99;
  const bool first_rejected =
      !core.PrepareBorrowedEuler(1.0e-8, coordinates, generation) &&
      generation == 0;
  const CpuStateSnapshot after_first = core.Snapshot(coordinates);
  const bool first_unchanged =
      std::equal(interior_before.begin(), interior_before.end(),
                 after_first.interior.data) &&
      std::equal(trace_before.begin(), trace_before.end(),
                 after_first.trace.data);

  generation = 100;
  const bool retry_rejected =
      !core.PrepareBorrowedEuler(1.0e-8, coordinates, generation) &&
      generation == 0;
  const CpuStateSnapshot after_retry = core.Snapshot(coordinates);
  const bool retry_unchanged =
      std::equal(interior_before.begin(), interior_before.end(),
                 after_retry.interior.data) &&
      std::equal(trace_before.begin(), trace_before.end(),
                 after_retry.trace.data);

  return first_rejected && retry_rejected && first_unchanged &&
      retry_unchanged &&
      coordinates.time == 0.0 && coordinates.step == 0 &&
      coordinates.generation == 0 && coordinates.cursor == 0 &&
      coordinates.dt == 0.0 && boundary.calls == 0;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS || ranks != 1)
    return 2;

  const bool off_identity = CheckDefaultOffAndInactiveIdentity();
  const bool cumulative_history = CheckCumulativeAcceptedHistory();
  const bool active_closure = CheckActiveAdmissibleMeanAndIgrClosure();
  const bool active_stage = CheckActiveCausalStageAndIgrSeeding();
  const bool p3_consumer_vertices =
      CheckP3ConsumerVertexPressureAndMeanPreservation();
  const bool bad_mean = CheckInadmissibleMeanRejectsWithoutRepair();
  if (!(off_identity && cumulative_history && active_closure &&
        active_stage && p3_consumer_vertices && bad_mean))
  {
    std::cerr << "HDG Zhang--Shu lifecycle checks failed: off_identity="
              << off_identity
              << " cumulative_history=" << cumulative_history
              << " active_closure=" << active_closure
              << " active_stage=" << active_stage
              << " p3_consumer_vertices=" << p3_consumer_vertices
              << " bad_mean=" << bad_mean << '\n';
    return 1;
  }
  std::cout << "HDG Zhang--Shu CPU lifecycle checks passed\n";
  return 0;
}
