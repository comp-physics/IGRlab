#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu::detail
{

struct CpuQualificationTestAccess
{
  static bool SameCheckpointState(
      const GenericEulerCpuCheckpoint& left,
      const GenericEulerCpuCheckpoint& right) noexcept
  {
    return left.SameStateForTests(right);
  }

  static bool AuditLastConservation(
      const CpuNumericsCore& core, double& maximum_error,
      double& maximum_relative_error, bool& gate) noexcept
  {
    return core.AuditLastBorrowedEulerConservation(
        maximum_error, maximum_relative_error, gate);
  }

  static bool AuditPreparedRodasEndpointConservation(
      CpuNumericsCore& core, double& maximum_error,
      double& maximum_relative_error) noexcept
  {
    return core.AuditPreparedRodasEndpointConservation(
        maximum_error, maximum_relative_error);
  }

  static bool VerifyAssemblerJacobian(
      CpuNumericsCore& core, const double relative_tolerance,
      std::array<double, 6>& relative_errors) noexcept
  {
    return core.AuditBorrowedEulerAssemblerJacobian(
        relative_tolerance, relative_errors);
  }

  static bool VerifyRodasAlgebraicBlock(
      CpuNumericsCore& core, double& algebraic_residual,
      double& correction_norm) noexcept
  {
    return core.AuditRodasAlgebraicBlockSolvability(
        algebraic_residual, correction_norm);
  }

  static bool AuditPreparedRodasStageEquations(
      CpuNumericsCore& core,
      std::array<double, 4>& absolute_residuals,
      std::array<double, 4>& relative_residuals,
      std::array<double, 4>& counterfactual_relative_residuals,
      double& maximum_zero_mass_entry) noexcept
  {
    return core.AuditPreparedRodasStageEquations(
        absolute_residuals, relative_residuals,
        counterfactual_relative_residuals, maximum_zero_mass_entry);
  }

  static bool VerifyLlfPolicy(
      CpuNumericsCore& core, double& base_stabilization,
      double& refreshed_stabilization,
      double& frozen_jacobian_relative_error,
      double& refreshed_candidate_relative_difference) noexcept
  {
    return core.AuditBorrowedEulerLlfPolicy(
        base_stabilization, refreshed_stabilization,
        frozen_jacobian_relative_error,
        refreshed_candidate_relative_difference);
  }

  static bool ArmExactZeroInitialResidual(CpuNumericsCore& core) noexcept
  {
    return core.ArmExactZeroInitialResidualForTests();
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
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cpu::ExecuteGenericEulerHdgCpu;
using dg::hdg::runtime::cpu::GenericEulerCpuCheckpoint;
using dg::hdg::runtime::cpu::GenericEulerCpuPreparedStep;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionResult;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionCoordinates;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::runtime::cpu::RestartGenericEulerHdgCpu;

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
  bool fail = false;
};

bool NonperiodicExterior(const void* opaque,
                         const dg::hdg::generic::BoundaryPointView& point,
                         const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(const_cast<void*>(opaque));
  if (context == nullptr || context->fail || point.space_dimension != 2 ||
      point.position == nullptr || point.outward_normal == nullptr ||
      point.boundary_attribute <= 0)
  {
    return false;
  }
  ++context->calls;
  for (int component = 0; component < 4; ++component)
  {
    exterior[component] = interior[component];
  }
  // A semantic nonperiodic boundary: reflect normal momentum at the wall.
  const double normal_velocity = interior[1] * point.outward_normal[0] +
      interior[2] * point.outward_normal[1];
  exterior[1] -= 2.0 * normal_velocity * point.outward_normal[0];
  exterior[2] -= 2.0 * normal_velocity * point.outward_normal[1];
  return true;
}

bool FixedExteriorForAssemblerJacobianForTests(
    const void* opaque,
    const dg::hdg::generic::BoundaryPointView& point,
    const double*, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(const_cast<void*>(opaque));
  if (context == nullptr || context->fail || point.space_dimension != 2 ||
      point.position == nullptr || point.outward_normal == nullptr ||
      point.boundary_attribute <= 0 || exterior == nullptr)
  {
    return false;
  }
  ++context->calls;
  // BE/PBE deliberately freeze the exterior-state callback (Picard
  // linearization).  Keep this legacy audit callback independent of the
  // perturbed state while still exercising every physical face row; the
  // separate RODAS3 audit below uses state-dependent boundary maps.
  exterior[0] = 1.1;
  exterior[1] = 0.03 * point.outward_normal[0];
  exterior[2] = 0.03 * point.outward_normal[1];
  exterior[3] = 2.9;
  return true;
}

bool SmoothAutonomousExteriorForRodasJacobianForTests(
    const void* opaque,
    const dg::hdg::generic::BoundaryPointView& point,
    const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(const_cast<void*>(opaque));
  if (context == nullptr || context->fail || point.space_dimension != 2 ||
      point.position == nullptr || point.outward_normal == nullptr ||
      point.boundary_attribute <= 0 || interior == nullptr ||
      exterior == nullptr)
  {
    return false;
  }
  ++context->calls;
  const double rho = interior[0];
  const double mx = interior[1];
  const double my = interior[2];
  const double energy = interior[3];
  exterior[0] = 0.95 + 0.05 * rho + 0.01 * rho * rho;
  exterior[1] = 0.10 * mx - 0.05 * my +
      0.02 * rho * point.outward_normal[0];
  exterior[2] = 0.10 * my - 0.05 * mx +
      0.02 * rho * point.outward_normal[1];
  exterior[3] = 2.50 + 0.10 * energy +
      0.01 * (mx * mx + my * my);
  return std::all_of(exterior, exterior + 4,
                     [](const double value) { return std::isfinite(value); });
}

bool CheckBoundaryExteriorSemantics()
{
  BoundaryContext context{};
  const std::array<double, 2> position{{0.25, 0.75}};
  const std::array<double, 2> normal{{1.0, 0.0}};
  const std::array<double, 4> interior{{1.2, 0.15, -0.04, 2.7}};
  std::array<double, 4> exterior{};
  const dg::hdg::generic::BoundaryPointView point{
      position.data(), normal.data(), 2, 1};
  return NonperiodicExterior(&context, point, interior.data(), exterior.data()) &&
      context.calls == 1 && exterior[0] == interior[0] &&
      exterior[1] == -interior[1] && exterior[2] == interior[2] &&
      exterior[3] == interior[3];
}

mfem::ParMesh MakeBorrowedMesh(const MPI_Comm communicator = MPI_COMM_WORLD)
{
  int ranks = 1;
  MPI_Comm_size(communicator, &ranks);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      6, 3, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()));
  for (int element = 0; element < serial.GetNE(); ++element)
  {
    partition[static_cast<std::size_t>(element)] = element % ranks;
  }
  return mfem::ParMesh(communicator, serial, partition.data());
}

mfem::ParMesh MakePeriodicBorrowedMesh(
    const MPI_Comm communicator = MPI_COMM_WORLD)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[0][1] = 0.0;
  translations[1][0] = 0.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(communicator, periodic);
}

mfem::ParMesh MakeRankZeroOwnedBorrowedMesh(
    const MPI_Comm communicator = MPI_COMM_WORLD)
{
  // Deliberately use the frozen contract's globally nonempty one-cell case:
  // rank zero owns its only cell and all other ranks have NE == 0.
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  const std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()), 0);
  return mfem::ParMesh(communicator, serial, partition.data());
}

mfem::ParMesh MakePeriodicRankZeroOwnedBorrowedMesh(
    const MPI_Comm communicator = MPI_COMM_WORLD)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  std::vector<int> partition(static_cast<std::size_t>(periodic.GetNE()), 0);
  return mfem::ParMesh(communicator, periodic, partition.data());
}

struct ExactBorrowedEulerFields
{
  std::vector<double> initial;
  std::vector<double> source;

  ExactBorrowedEulerFields(const mfem::ParMesh& mesh, const int order,
                           const std::array<double, 8>& initial_pattern,
                           const std::array<double, 8>& source_pattern)
  {
    mfem::L2_FECollection collection(
        order, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        const_cast<mfem::ParMesh*>(&mesh), &collection, 4,
        mfem::Ordering::byNODES);
    const int scalar_dofs = space.GetVSize() / 4;
    initial.resize(static_cast<std::size_t>(space.GetVSize()));
    source.resize(static_cast<std::size_t>(space.GetVSize()));
    for (int component = 0; component < 4; ++component)
    {
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        const std::size_t destination = static_cast<std::size_t>(
            component * scalar_dofs + scalar);
        const std::size_t pattern = static_cast<std::size_t>(2 * component +
                                                              scalar % 2);
        initial[destination] = initial_pattern[pattern];
        source[destination] = source_pattern[pattern];
      }
    }
  }
};

EulerHdgRequest Request(const mfem::ParMesh& mesh, const int order,
                        BoundaryContext& boundary,
                        const ExactBorrowedEulerFields& fields,
                        const EulerHdgTimeTreatment time_treatment =
                            EulerHdgTimeTreatment::backward_euler)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cpu, -1, CudaEngineKind::ordinary};
  request.capabilities = cpu_capabilities;
  request.initial_state = {fields.initial.empty() ? nullptr : fields.initial.data(),
                           fields.initial.size(), 4};
  request.initial_true_trace = {nullptr, 0, 4};
  request.source = {fields.source.empty() ? nullptr : fields.source.data(),
                    fields.source.size(), 4};
  request.boundary = {
      &boundary, NonperiodicExterior, {nullptr, 0, 0, 0.0}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  request.time_treatment = time_treatment;
  return request;
}

bool MatchesCommittedCflDiagnostic(
    const dg::hdg::generic::HdgEulerCommittedDiagnostics& diagnostics,
    mfem::ParMesh& mesh, const int polynomial_order)
{
  double local_hmin = std::numeric_limits<double>::infinity();
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    local_hmin = std::min(local_hmin, mesh.GetElementSize(element, 1));
  }
  double hmin = std::numeric_limits<double>::infinity();
  if (MPI_Allreduce(&local_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN,
                    mesh.GetComm()) != MPI_SUCCESS ||
      !(hmin > 0.0) || !std::isfinite(hmin))
  {
    return false;
  }
  const double denominator = 2.0 * polynomial_order + 1.0;
  const double expected = diagnostics.maximum_wave_speed > 0.0
      ? hmin / (denominator * diagnostics.maximum_wave_speed)
      : hmin / denominator;
  const double scale = std::max({1.0, std::abs(expected),
                                 std::abs(diagnostics.backend_stable_dt)});
  return std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed >= 0.0 &&
      std::isfinite(diagnostics.backend_stable_dt) &&
      diagnostics.backend_stable_dt > 0.0 &&
      std::abs(diagnostics.backend_stable_dt - expected) <= 1.0e-12 * scale;
}

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  int source = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

bool SameSessionCoordinates(
    const dg::hdg::runtime::cpu::GenericEulerCpuSessionCoordinates& left,
    const dg::hdg::runtime::cpu::GenericEulerCpuSessionCoordinates& right)
    noexcept
{
  return dg::hdg::generic::DoubleBits(left.time) ==
          dg::hdg::generic::DoubleBits(right.time) &&
      left.step == right.step && left.generation == right.generation &&
      dg::hdg::generic::DoubleBits(left.dt) ==
          dg::hdg::generic::DoubleBits(right.dt);
}

bool LinearWorkAccountingExact(
    const dg::hdg::runtime::cpu::CpuAttemptEvidence& attempt) noexcept
{
  return attempt.linear_iterations >= 0 &&
      attempt.linear_solve_attempts >= 0 &&
      attempt.newton_correction_linear_iterations >= 0 &&
      attempt.final_coherent_linear_iterations >= 0 &&
      attempt.scalar_closure_linear_iterations >= 0 &&
      attempt.scalar_closure_linear_solve_attempts >= 0 &&
      attempt.scalar_closure_true_residual_tightenings >= 0 &&
      attempt.true_residual_tightenings >= 0 &&
      attempt.linear_iterations ==
          attempt.newton_correction_linear_iterations +
              attempt.final_coherent_linear_iterations &&
      attempt.linear_solve_attempts ==
          static_cast<int>(attempt.work.complete_trace_solves) +
              attempt.true_residual_tightenings &&
      attempt.scalar_closure_linear_iterations <=
          attempt.final_coherent_linear_iterations &&
      attempt.scalar_closure_linear_solve_attempts <=
          attempt.linear_solve_attempts &&
      attempt.scalar_closure_true_residual_tightenings <=
          attempt.true_residual_tightenings &&
      attempt.scalar_closure_linear_solve_attempts ==
          static_cast<int>(attempt.work.complete_scalar_closure_solves) +
              attempt.scalar_closure_true_residual_tightenings;
}

bool AdmissibleEulerCoefficients(const std::vector<double>& state) noexcept
{
  if (state.size() % 4 != 0) return false;
  const std::size_t scalar_count = state.size() / 4;
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
  {
    const double density = state[scalar];
    const double momentum_x = state[scalar_count + scalar];
    const double momentum_y = state[2 * scalar_count + scalar];
    const double energy = state[3 * scalar_count + scalar];
    if (!std::isfinite(density) || !(density > 0.0) ||
        !std::isfinite(momentum_x) || !std::isfinite(momentum_y) ||
        !std::isfinite(energy))
    {
      return false;
    }
    const double pressure = 0.4 *
        (energy - 0.5 *
             (momentum_x * momentum_x + momentum_y * momentum_y) / density);
    if (!std::isfinite(pressure) || !(pressure > 0.0)) return false;
  }
  return true;
}

bool SameEulerCoefficientsWithinRoundoff(
    const std::vector<double>& left,
    const std::vector<double>& right) noexcept
{
  if (left.size() != right.size()) return false;
  constexpr double multiplier = 128.0;
  for (std::size_t index = 0; index < left.size(); ++index)
  {
    const double scale = std::max({1.0, std::abs(left[index]),
                                   std::abs(right[index])});
    if (!std::isfinite(left[index]) || !std::isfinite(right[index]) ||
        std::abs(left[index] - right[index]) >
            multiplier * std::numeric_limits<double>::epsilon() * scale)
    {
      return false;
    }
  }
  return true;
}

double MaximumScaledDifference(const std::vector<double>& left,
                               const std::vector<double>& right) noexcept
{
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
  {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index]))
      return std::numeric_limits<double>::infinity();
    const double scale = std::max({1.0, std::abs(left[index]),
                                   std::abs(right[index])});
    maximum = std::max(maximum,
                       std::abs(left[index] - right[index]) / scale);
  }
  return maximum;
}

double QuadratureScaledU4Difference(
    const mfem::ParMesh& mesh, const int order,
    const std::vector<double>& left, const std::vector<double>& right,
    const double absolute_tolerance, const double relative_tolerance) noexcept
{
  if (order < 1 || left.size() != right.size() ||
      !std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
      !std::isfinite(relative_tolerance) || relative_tolerance < 0.0 ||
      (absolute_tolerance == 0.0 && relative_tolerance == 0.0))
  {
    return std::numeric_limits<double>::infinity();
  }
  double local_squared = 0.0;
  double local_measure = 0.0;
  bool local_valid = true;
  const auto collective_all = [&mesh](const bool value) noexcept
  {
    const int local = value ? 1 : 0;
    int global = 0;
    return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                         mesh.GetComm()) == MPI_SUCCESS && global == 1;
  };
  try
  {
    mfem::L2_FECollection collection(
        order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        const_cast<mfem::ParMesh*>(&mesh), &collection, 4,
        mfem::Ordering::byNODES);
    local_valid = left.size() == static_cast<std::size_t>(space.GetVSize());
    mfem::Array<int> vdofs;
    mfem::Vector shape;
    for (int element = 0; element < mesh.GetNE() && local_valid; ++element)
    {
      const mfem::FiniteElement* const finite_element = space.GetFE(element);
      mfem::ElementTransformation* const transformation =
          const_cast<mfem::ParMesh&>(mesh).GetElementTransformation(element);
      local_valid = finite_element != nullptr && transformation != nullptr;
      if (!local_valid) break;
      const int scalar_dofs = finite_element->GetDof();
      shape.SetSize(scalar_dofs);
      space.GetElementVDofs(element, vdofs);
      local_valid = vdofs.Size() == 4 * scalar_dofs;
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), 2 * order + 6);
      for (int q = 0; q < rule.GetNPoints() && local_valid; ++q)
      {
        const mfem::IntegrationPoint& point = rule.IntPoint(q);
        transformation->SetIntPoint(&point);
        finite_element->CalcShape(point, shape);
        const double weight = point.weight * transformation->Weight();
        local_valid = std::isfinite(weight) && weight > 0.0;
        if (!local_valid) break;
        local_measure += weight;
        for (int component = 0; component < 4 && local_valid; ++component)
        {
          double before = 0.0;
          double after = 0.0;
          for (int scalar = 0; scalar < scalar_dofs; ++scalar)
          {
            const int signed_dof = vdofs[component * scalar_dofs + scalar];
            const int dof = signed_dof < 0 ? -1 - signed_dof : signed_dof;
            const double sign = signed_dof < 0 ? -1.0 : 1.0;
            local_valid = dof >= 0 &&
                static_cast<std::size_t>(dof) < left.size() &&
                std::isfinite(shape[scalar]);
            if (!local_valid) break;
            before += shape[scalar] * sign *
                left[static_cast<std::size_t>(dof)];
            after += shape[scalar] * sign *
                right[static_cast<std::size_t>(dof)];
          }
          const double scale = absolute_tolerance + relative_tolerance *
              std::max(std::abs(before), std::abs(after));
          local_valid = local_valid && std::isfinite(before) &&
              std::isfinite(after) && std::isfinite(scale) && scale > 0.0;
          if (local_valid)
          {
            const double normalized = (before - after) / scale;
            local_squared += weight * normalized * normalized;
            local_valid = std::isfinite(local_squared) &&
                std::isfinite(local_measure);
          }
        }
      }
    }
  }
  catch (...) { local_valid = false; }
  if (!collective_all(local_valid))
    return std::numeric_limits<double>::infinity();
  double global_squared = 0.0;
  double global_measure = 0.0;
  const bool reduced = MPI_Allreduce(
          &local_squared, &global_squared, 1, MPI_DOUBLE, MPI_SUM,
          mesh.GetComm()) == MPI_SUCCESS &&
      MPI_Allreduce(&local_measure, &global_measure, 1, MPI_DOUBLE, MPI_SUM,
                    mesh.GetComm()) == MPI_SUCCESS;
  if (!collective_all(reduced && std::isfinite(global_squared) &&
                      global_squared >= 0.0 &&
                      std::isfinite(global_measure) &&
                      global_measure > 0.0))
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(global_squared / (4.0 * global_measure));
}

struct OneStepEulerState
{
  std::vector<double> interior_u4;
  std::vector<double> trace_uhat4;
  double temporal_error_norm = std::numeric_limits<double>::infinity();
  double normalized_temporal_error =
      std::numeric_limits<double>::infinity();
  double normalized_algebraic_correction =
      std::numeric_limits<double>::infinity();
};

bool AssessPreparedRodas3(
    dg::hdg::runtime::cpu::detail::CpuNumericsCore& core,
    const std::uint64_t generation,
    dg::hdg::generic::HdgEulerPreparedAssessment* const captured = nullptr)
{
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy policy{
      1.0e-8, 1.0e-5, 0.25};
  dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
  const bool assessed = core.AssessPreparedBorrowedEuler(
      generation, policy, assessment);
  if (assessed && captured != nullptr) *captured = assessment;
  return assessed && assessment.algebraic_correction_gate;
}

bool RunEulerSteps(
    const mfem::ParMesh& mesh, const ExactBorrowedEulerFields& fields,
    BoundaryContext& boundary, const EulerHdgTimeTreatment time_treatment,
    const int order, const bool igr, const double dt,
    OneStepEulerState& result, const int accepted_steps = 1)
{
  if (accepted_steps <= 0) return false;
  EulerHdgRequest request = Request(
      mesh, order, boundary, fields, time_treatment);
  if (igr)
    request.numerical_policy.stabilization =
        {RegularizationKind::igr, 0.2, 0.4};
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return false;

  dg::hdg::runtime::StateCoordinates coordinates{};
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  dg::hdg::generic::HdgEulerPreparedAssessment last_assessment{};
  for (int step = 0; step < accepted_steps; ++step)
  {
    std::uint64_t generation = 0;
    if (!core.PrepareBorrowedEuler(dt, coordinates, generation) ||
        generation == 0 ||
        (time_treatment == EulerHdgTimeTreatment::rodas3 &&
         !AssessPreparedRodas3(core, generation, &last_assessment)) ||
        !core.CommitBorrowedEuler(generation, coordinates,
                                  maximum_wave_speed, stable_dt))
    {
      return false;
    }
  }
  result.temporal_error_norm =
      core.evidence().last_attempt.temporal_error_norm;
  if (time_treatment == EulerHdgTimeTreatment::rodas3)
  {
    result.normalized_temporal_error =
        last_assessment.normalized_differential_error;
    result.normalized_algebraic_correction =
        last_assessment.normalized_algebraic_correction;
  }

  result.interior_u4.assign(fields.initial.size(), 0.0);
  if (!core.ExportCommittedInterior(result.interior_u4.data(),
                                    result.interior_u4.size()))
  {
    return false;
  }
  const dg::hdg::runtime::cpu::CpuStateSnapshot snapshot =
      core.Snapshot(coordinates);
  const std::size_t trace_components = igr ? 5U : 4U;
  if (snapshot.trace.size == 0 || snapshot.trace.data == nullptr ||
      snapshot.trace.size % trace_components != 0)
  {
    return false;
  }
  const std::size_t scalar_trace =
      snapshot.trace.size / trace_components;
  result.trace_uhat4.assign(
      snapshot.trace.data, snapshot.trace.data + 4 * scalar_trace);

  double maximum_conservation_error =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::quiet_NaN();
  bool conservation_gate = false;
  const bool conservation_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditLastConservation(
              core, maximum_conservation_error,
              maximum_conservation_relative_error, conservation_gate);
  const auto& attempt = core.evidence().last_attempt;
  return conservation_audited && conservation_gate && attempt.committed &&
      attempt.status == dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate &&
      attempt.minimum_density > 0.0 && attempt.minimum_pressure > 0.0 &&
      std::isfinite(attempt.maximum_true_relative_residual) &&
      attempt.maximum_true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      std::isfinite(maximum_conservation_error) &&
      maximum_conservation_error >= 0.0 &&
      std::isfinite(maximum_conservation_relative_error) &&
      maximum_conservation_relative_error <=
          request.numerical_policy.conservation_relative_tolerance &&
      AdmissibleEulerCoefficients(result.interior_u4) &&
      coordinates.step == static_cast<std::uint64_t>(accepted_steps) &&
      coordinates.generation == static_cast<std::uint64_t>(accepted_steps) &&
      std::abs(coordinates.time - accepted_steps * dt) <=
          64.0 * std::numeric_limits<double>::epsilon() *
              std::max(1.0, std::abs(accepted_steps * dt)) &&
      coordinates.dt == dt &&
      std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
      std::isfinite(stable_dt) && stable_dt > 0.0;
}

struct BorrowedMeshSnapshot
{
  int elements = 0;
  int vertices = 0;
  int faces = 0;
  std::vector<int> attributes;
  MPI_Comm communicator = MPI_COMM_NULL;
};

BorrowedMeshSnapshot SnapshotBorrowedMesh(const mfem::ParMesh& mesh)
{
  BorrowedMeshSnapshot snapshot{};
  snapshot.elements = mesh.GetNE();
  snapshot.vertices = mesh.GetNV();
  snapshot.faces = mesh.GetNumFaces();
  snapshot.communicator = mesh.GetComm();
  snapshot.attributes.reserve(static_cast<std::size_t>(mesh.GetNE()));
  for (int element = 0; element < mesh.GetNE(); ++element)
    snapshot.attributes.push_back(mesh.GetAttribute(element));
  return snapshot;
}

bool SameBorrowedMesh(const mfem::ParMesh& mesh,
                      const BorrowedMeshSnapshot& before,
                      const MPI_Comm requested_communicator)
{
  int mesh_same = MPI_UNEQUAL;
  int request_same = MPI_UNEQUAL;
  if (MPI_Comm_compare(before.communicator, mesh.GetComm(), &mesh_same) !=
          MPI_SUCCESS ||
      MPI_Comm_compare(requested_communicator, mesh.GetComm(),
                       &request_same) != MPI_SUCCESS ||
      mesh.GetNE() != before.elements || mesh.GetNV() != before.vertices ||
      mesh.GetNumFaces() != before.faces ||
      mesh_same != MPI_IDENT ||
      (request_same != MPI_IDENT && request_same != MPI_CONGRUENT))
  {
    return false;
  }
  for (int element = 0; element < mesh.GetNE(); ++element)
    if (mesh.GetAttribute(element) !=
        before.attributes[static_cast<std::size_t>(element)]) return false;
  return true;
}

int ByNodes(const int scalar, const int scalar_dofs,
            const int component) noexcept
{
  return scalar + scalar_dofs * component;
}

bool CheckBorrowedTraceRestrictsInterior()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  for (int order = 1; order <= 3; ++order)
  {
    mfem::L2_FECollection interior_collection(
        order, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace interior_space(
        &mesh, &interior_collection, 4, mfem::Ordering::byNODES);
    mfem::ParGridFunction nonuniform_interior(&interior_space);
    auto field = [](const mfem::Vector& point, mfem::Vector& value)
    {
      const double density = 1.0 + 0.05 * point[0] + 0.02 * point[1];
      const double velocity_x = 0.2 + 0.03 * point[0];
      const double velocity_y = -0.1 + 0.02 * point[1];
      const double pressure = 1.0 + 0.03 * point[0] - 0.02 * point[1];
      value.SetSize(4);
      value[0] = density;
      value[1] = density * velocity_x;
      value[2] = density * velocity_y;
      value[3] = pressure / 0.4 +
          0.5 * density * (velocity_x * velocity_x + velocity_y * velocity_y);
    };
    mfem::VectorFunctionCoefficient coefficient(4, field);
    nonuniform_interior.ProjectCoefficient(coefficient);
    const double* const interior_values = nonuniform_interior.HostRead();
    if (interior_values == nullptr && nonuniform_interior.Size() != 0)
      return false;
    std::vector<double> initial;
    if (nonuniform_interior.Size() != 0)
      initial.assign(interior_values,
                     interior_values + nonuniform_interior.Size());
    const std::array<double, 4> zero_local_initial{{1.0, 0.2, -0.1, 2.5}};

    mfem::DG_Interface_FECollection trace_collection(
        order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, 4, mfem::Ordering::byNODES);
    mfem::ParGridFunction expected_local_trace(&trace_space);
    mfem::Vector expected_true_trace(trace_space.GetTrueVSize());
    expected_local_trace = 0.0;
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      mfem::FaceElementTransformations* const transformations =
          mesh.GetFaceElementTransformations(face);
      const mfem::FiniteElement* const trace_element =
          trace_space.GetFaceElement(face);
      if (!transformations || !trace_element || transformations->Elem1No < 0 ||
          transformations->Elem1No >= mesh.GetNE())
      {
        return false;
      }
      const mfem::FiniteElement* const interior_element =
          interior_space.GetFE(transformations->Elem1No);
      mfem::Array<int> interior_vdofs;
      mfem::Array<int> trace_vdofs;
      interior_space.GetElementVDofs(transformations->Elem1No, interior_vdofs);
      trace_space.GetFaceVDofs(face, trace_vdofs);
      if (!interior_element ||
          interior_vdofs.Size() != 4 * interior_element->GetDof() ||
          trace_vdofs.Size() != 4 * trace_element->GetDof())
      {
        return false;
      }
      mfem::Vector shape(interior_element->GetDof());
      const mfem::IntegrationRule& nodes = trace_element->GetNodes();
      for (int scalar = 0; scalar < trace_element->GetDof(); ++scalar)
      {
        const mfem::IntegrationPoint& point = nodes.IntPoint(scalar);
        transformations->SetAllIntPoints(&point);
        interior_element->CalcShape(transformations->GetElement1IntPoint(),
                                    shape);
        for (int component = 0; component < 4; ++component)
        {
          double restricted = 0.0;
          for (int interior_scalar = 0;
               interior_scalar < interior_element->GetDof(); ++interior_scalar)
          {
            double sign = 1.0;
            const int dof = mfem::FiniteElementSpace::DecodeDof(
                interior_vdofs[ByNodes(interior_scalar,
                                       interior_element->GetDof(), component)],
                sign);
            if (dof < 0 || dof >= nonuniform_interior.Size()) return false;
            restricted += shape[interior_scalar] * sign * interior_values[dof];
          }
          double sign = 1.0;
          const int dof = mfem::FiniteElementSpace::DecodeDof(
              trace_vdofs[ByNodes(scalar, trace_element->GetDof(), component)],
              sign);
          if (dof < 0 || dof >= expected_local_trace.Size()) return false;
          expected_local_trace[dof] = sign * restricted;
        }
      }
    }
    expected_local_trace.ParallelProject(expected_true_trace);

    BoundaryContext boundary{};
    EulerHdgRequest request{};
    request.mesh = &mesh;
    request.communicator = mesh.GetComm();
    request.polynomial_order = order;
    request.backend = {BackendKind::cpu, -1, CudaEngineKind::ordinary};
    request.capabilities = cpu_capabilities;
    request.initial_state = initial.empty()
        ? dg::hdg::generic::EulerFieldView{zero_local_initial.data(),
                                           zero_local_initial.size(), 4}
        : dg::hdg::generic::EulerFieldView{initial.data(), initial.size(), 4};
    request.initial_true_trace = {nullptr, 0, 4};
    request.source = {nullptr, 0, 4};
    request.boundary = {
        &boundary, NonperiodicExterior, {nullptr, 0, 0, 0.0}};
    request.numerical_policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
    if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
        dg::hdg::generic::ContractStatus::accepted)
    {
      return false;
    }
    dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
    if (!core.InitializeBorrowedEuler(request)) return false;
    if (core.evidence().tuple.formulation.view() !=
            "hdg.conventional.v1" ||
        core.evidence().tuple.stabilization.view() !=
            dg::hdg::runtime::borrowed_euler_llf_stabilization_id)
    {
      return false;
    }
    const dg::hdg::runtime::cpu::CpuStateSnapshot snapshot = core.Snapshot({});
    if (snapshot.trace.size != static_cast<std::size_t>(expected_true_trace.Size()) ||
        (snapshot.trace.data == nullptr && snapshot.trace.size != 0))
    {
      return false;
    }
    const double* const expected = expected_true_trace.HostRead();
    if (expected == nullptr && expected_true_trace.Size() != 0) return false;
    for (int index = 0; index < expected_true_trace.Size(); ++index)
      if (snapshot.trace.data[index] != expected[index]) return false;
  }
  return true;
}

bool CheckOrdersAndDeterminism()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  // EulerFieldView is component-major: two repeat samples per component.
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{0.03, 0.04, 0.01, 0.02,
                                       0.00, 0.00, 0.05, 0.06}};
  for (int order = 1; order <= 3; ++order)
  {
    BoundaryContext context{};
    ExactBorrowedEulerFields fields(mesh, order, initial, source);
    EulerHdgRequest request = Request(mesh, order, context, fields);
    const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
    GenericEulerHdgCpuSession session;
    const auto initialized = session.Initialize(request);
    if (initialized.status != GenericEulerCpuSessionStatus::initialized ||
        initialized.coordinates.step != 0 || initialized.coordinates.time != 0.0)
    {
      std::cerr << "generic session initialization failed for p" << order
                << '\n';
      return false;
    }
    mfem::L2_FECollection collection(
        order, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace interior_space(
        &mesh, &collection, 4, mfem::Ordering::byNODES);
    mfem::ParGridFunction committed(&interior_space);
    if (!session.ExportCommittedInterior(committed)) return false;
    std::vector<double> first(static_cast<std::size_t>(committed.Size()));
    if (!session.ExportCommittedInterior({first.data(), first.size()})) return false;

    const auto first_step = session.Advance(0.005);
    const auto second_step = session.Advance(0.011);
    if (first_step.status != GenericEulerCpuSessionStatus::advanced ||
        second_step.status != GenericEulerCpuSessionStatus::advanced ||
        second_step.coordinates.step != 2 ||
        second_step.coordinates.generation != 2 ||
        second_step.coordinates.dt != 0.011 ||
        second_step.coordinates.time != 0.016 || context.calls == 0)
    {
      std::cerr << "generic unequal-dt session advance failed for p" << order
                << '\n';
      return false;
    }
    std::vector<double> second(first.size());
    if (!session.ExportCommittedInterior({second.data(), second.size()})) return false;
    bool changed = false;
    for (std::size_t index = 0; index < first.size(); ++index)
      changed = changed || first[index] != second[index];
    if (!changed) return false;

    mfem::ParFiniteElementSpace wrong_layout_space(
        &mesh, &collection, 4, mfem::Ordering::byVDIM);
    mfem::ParGridFunction wrong_layout(&wrong_layout_space);
    wrong_layout = 7.0;
    const double* const wrong_values = wrong_layout.HostRead();
    bool wrong_unchanged = wrong_values != nullptr || wrong_layout.Size() == 0;
    for (int index = 0; index < wrong_layout.Size(); ++index)
      wrong_unchanged = wrong_unchanged && wrong_values[index] == 7.0;
    if (session.ExportCommittedInterior(wrong_layout) || !wrong_unchanged)
    {
      return false;
    }
    if (!session.ExportCommittedInterior(committed)) return false;
    const double* committed_values = committed.HostRead();
    for (int index = 0; index < committed.Size(); ++index)
      if (second[static_cast<std::size_t>(index)] != committed_values[index])
        return false;

    const auto coordinates_before_failure = session.coordinates();
    const std::vector<double> committed_before_failure = second;
    context.fail = true;
    if (session.Advance(0.031).status != GenericEulerCpuSessionStatus::rejected ||
        session.coordinates().step != coordinates_before_failure.step ||
        session.coordinates().time != coordinates_before_failure.time ||
        !session.ExportCommittedInterior({second.data(), second.size()}))
    {
      return false;
    }
    if (second != committed_before_failure) return false;
    if (session.Advance(0.0).status != GenericEulerCpuSessionStatus::rejected ||
        session.coordinates().step != coordinates_before_failure.step ||
        session.coordinates().time != coordinates_before_failure.time)
    {
      return false;
    }
    if (!SameBorrowedMesh(mesh, before, request.communicator)) return false;
  }
  return true;
}

bool CheckPreparedSessionLifecycle()
{
  const dg::hdg::generic::HdgEulerCommittedDiagnostics unavailable{};
  if (dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(unavailable, 0))
  {
    return false;
  }
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{0.03, 0.04, 0.01, 0.02,
                                       0.00, 0.00, 0.05, 0.06}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  GenericEulerHdgCpuSession session;
  if (session.Initialize(Request(mesh, 1, boundary, fields)).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  const std::uint32_t required_capabilities =
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::prepared_commit) |
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::prepared_output) |
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::committed_cfl_diagnostics) |
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::generic_borrowed_inputs);
  const std::uint32_t assessment_capability =
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::
              prepared_embedded_assessment);
  if (!dg::hdg::generic::HasHdgEulerSessionCapabilities(
          session.capabilities(), required_capabilities) ||
      dg::hdg::generic::HasHdgEulerSessionCapabilities(
          session.capabilities(), assessment_capability) ||
      !dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
          session.diagnostics(), session.capabilities()) ||
      !MatchesCommittedCflDiagnostic(session.diagnostics(), mesh, 1))
  {
    return false;
  }
  auto stale_generation = session.diagnostics();
  stale_generation.wave_generation = 1;
  if (dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
          stale_generation, session.capabilities()))
  {
    return false;
  }

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const auto initial_diagnostics = session.diagnostics();
  if (ranks > 1)
  {
    GenericEulerCpuPreparedStep divergent_proposal;
    const double rank_divergent_dt = rank == 0 ? 0.005 : 0.006;
    if (session.Prepare(rank_divergent_dt, divergent_proposal).status !=
            GenericEulerCpuSessionStatus::rejected ||
        session.coordinates().generation != 0 ||
        session.diagnostics().maximum_wave_speed !=
            initial_diagnostics.maximum_wave_speed ||
        session.diagnostics().backend_stable_dt !=
            initial_diagnostics.backend_stable_dt)
    {
      return false;
    }
  }

  mfem::L2_FECollection collection(
      1, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(&mesh, &collection, 4,
                                    mfem::Ordering::byNODES);
  mfem::ParGridFunction committed(&space);
  mfem::ParGridFunction prepared(&space);
  if (!session.ExportCommittedInterior(committed)) return false;
  std::vector<double> before(static_cast<std::size_t>(committed.Size()));
  if (!session.ExportCommittedInterior({before.data(), before.size()})) return false;
  const auto coordinates_before = session.coordinates();
  const auto diagnostics_before = session.diagnostics();

  GenericEulerCpuPreparedStep first;
  if (session.Prepare(0.005, first).status !=
          GenericEulerCpuSessionStatus::prepared ||
      session.Prepare(0.005, first).status !=
          GenericEulerCpuSessionStatus::rejected ||
      session.coordinates().step != coordinates_before.step ||
      session.coordinates().generation != coordinates_before.generation ||
      session.diagnostics().coordinates.generation !=
          diagnostics_before.coordinates.generation ||
      !session.ExportPreparedInterior(first, prepared) ||
      !session.ExportCommittedInterior(committed))
  {
    return false;
  }
  std::vector<double> prepared_values(
      static_cast<std::size_t>(prepared.Size()));
  std::vector<double> committed_values(
      static_cast<std::size_t>(committed.Size()));
  if (!session.ExportPreparedInterior(
          first, {prepared_values.data(), prepared_values.size()}) ||
      !session.ExportCommittedInterior(
          {committed_values.data(), committed_values.size()}) ||
      committed_values != before)
  {
    return false;
  }

  // A different rank-local token must be rejected collectively with neither
  // committed bytes nor coordinates changed; the valid authority survives.
  if (ranks > 1)
  {
    const auto divergent_operation = rank == 0
        ? session.Commit(std::move(first))
        : session.Abort(std::move(first));
    if (divergent_operation.status != GenericEulerCpuSessionStatus::rejected ||
        session.coordinates().step != coordinates_before.step ||
        !session.ExportCommittedInterior(
            {committed_values.data(), committed_values.size()}) ||
        committed_values != before)
    {
      return false;
    }
  }
  GenericEulerCpuPreparedStep stale;
  const auto mismatched = rank == 0 ? session.Commit(std::move(stale))
                                    : session.Commit(std::move(first));
  if (mismatched.status != GenericEulerCpuSessionStatus::rejected ||
      session.coordinates().step != coordinates_before.step ||
      !session.ExportCommittedInterior(
          {committed_values.data(), committed_values.size()}) ||
      committed_values != before)
  {
    return false;
  }
  if (session.Abort(std::move(first)).status !=
          GenericEulerCpuSessionStatus::aborted ||
      session.coordinates().step != coordinates_before.step ||
      session.coordinates().generation != coordinates_before.generation ||
      session.diagnostics().coordinates.generation !=
          diagnostics_before.coordinates.generation ||
      session.diagnostics().maximum_wave_speed !=
          diagnostics_before.maximum_wave_speed ||
      session.diagnostics().backend_stable_dt !=
          diagnostics_before.backend_stable_dt ||
      !session.ExportCommittedInterior(
          {committed_values.data(), committed_values.size()}) ||
      committed_values != before)
  {
    return false;
  }

  GenericEulerCpuPreparedStep second;
  if (session.Prepare(0.007, second).status !=
          GenericEulerCpuSessionStatus::prepared ||
      session.Commit(std::move(second)).status !=
          GenericEulerCpuSessionStatus::committed ||
      session.coordinates().step != 1 || session.coordinates().generation != 1 ||
      session.coordinates().dt != 0.007 || session.coordinates().time != 0.007 ||
      !dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
          session.diagnostics(), session.capabilities()) ||
      session.diagnostics().coordinates.generation != 1 ||
      session.diagnostics().wave_generation != 1 ||
      !MatchesCommittedCflDiagnostic(session.diagnostics(), mesh, 1))
  {
    return false;
  }
  const auto committed_coordinates = session.coordinates();
  if (session.Commit(std::move(second)).status !=
          GenericEulerCpuSessionStatus::rejected ||
      session.coordinates().step != committed_coordinates.step ||
      session.coordinates().generation != committed_coordinates.generation ||
      session.Advance(0.011).status != GenericEulerCpuSessionStatus::advanced ||
      session.coordinates().step != 2 || session.coordinates().generation != 2 ||
      session.diagnostics().coordinates.generation != 2 ||
      session.diagnostics().wave_generation != 2 ||
      !MatchesCommittedCflDiagnostic(session.diagnostics(), mesh, 1))
  {
    return false;
  }
  return true;
}

bool CheckCommittedExportLifecycle()
{
  GenericEulerHdgCpuSession uninitialized;
  std::array<double, 4> unavailable{{1.0, 2.0, 3.0, 4.0}};
  const auto unavailable_before = unavailable;
  if (uninitialized.ExportCommittedInterior(
          {unavailable.data(), unavailable.size()}) ||
      unavailable != unavailable_before)
  {
    return false;
  }

  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                        0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                       0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  GenericEulerHdgCpuSession session;
  if (session.Initialize(Request(mesh, 1, boundary, fields)).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }

  const auto before_coordinates = session.coordinates();
  std::vector<double> committed_before(fields.initial.size());
  if (!session.ExportCommittedInterior(
          {committed_before.data(), committed_before.size()}))
  {
    return false;
  }

  GenericEulerCpuPreparedStep prepared;
  if (session.Prepare(0.005, prepared).status !=
          GenericEulerCpuSessionStatus::prepared ||
      session.coordinates().generation != before_coordinates.generation)
  {
    return false;
  }
  std::vector<double> committed_during_prepare(committed_before.size());
  if (!session.ExportCommittedInterior(
          {committed_during_prepare.data(), committed_during_prepare.size()}) ||
      committed_during_prepare != committed_before ||
      session.Abort(std::move(prepared)).status !=
          GenericEulerCpuSessionStatus::aborted)
  {
    return false;
  }

  if (session.Advance(0.007).status != GenericEulerCpuSessionStatus::advanced)
  {
    return false;
  }
  std::vector<double> committed_after_first_export(committed_before.size());
  if (!session.ExportCommittedInterior(
          {committed_after_first_export.data(),
           committed_after_first_export.size()}))
  {
    return false;
  }
  const auto after_coordinates = session.coordinates();
  std::vector<double> committed_after_second_export(committed_before.size());
  return after_coordinates.step == 1 &&
      after_coordinates.generation == 1 && after_coordinates.dt == 0.007 &&
      after_coordinates.time == 0.007 &&
      session.ExportCommittedInterior(
          {committed_after_second_export.data(),
           committed_after_second_export.size()}) &&
      committed_after_second_export == committed_after_first_export &&
      session.coordinates().step == after_coordinates.step &&
      session.coordinates().generation == after_coordinates.generation;
}

bool CheckCommittedExportZeroOwnedRank()
{
  mfem::ParMesh mesh = MakeRankZeroOwnedBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  GenericEulerHdgCpuSession session;
  if (session.Initialize(Request(mesh, 1, boundary, fields)).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  std::vector<double> committed(fields.initial.size(),
                                std::numeric_limits<double>::quiet_NaN());
  const bool exported = session.ExportCommittedInterior(
      {committed.empty() ? nullptr : committed.data(), committed.size()});
  return All(exported && committed == fields.initial);
}

bool CheckDefaultNumericalPolicyAdaptiveLinearCap()
{
  // p3 on this deliberately nontrivial mesh has a true trace larger than the
  // historical literal 100 cap.  The default policy must preserve the older
  // size-derived bound, while a positive request stays literal.
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 3, initial, source);
  EulerHdgRequest request = Request(mesh, 3, boundary, fields);
  const auto default_policy = request.numerical_policy;
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (default_policy.maximum_linear_iterations != 0 ||
      default_policy.newton_correction_linear_relative_tolerance != 1.0e-12 ||
      default_policy.final_coherent_linear_relative_tolerance != 1.0e-12 ||
      default_policy.true_residual_tightening_linear_relative_tolerance !=
          1.0e-12 ||
      default_policy.igr_assembly_quadrature_orders.volume_order != 12 ||
      default_policy.igr_assembly_quadrature_orders.face_order != 12 ||
      !core.InitializeBorrowedEuler(request))
  {
    return false;
  }
  const int global_true_trace = static_cast<int>(
      core.evidence().global_owned_true_trace_coefficients);
  if (global_true_trace <= 100 ||
      dg::hdg::generic::ResolveEulerHdgMaximumLinearIterations(
          default_policy.maximum_linear_iterations, global_true_trace) !=
          global_true_trace)
  {
    return false;
  }
  constexpr int explicit_cap = 37;
  request.numerical_policy.maximum_linear_iterations = explicit_cap;
  return dg::hdg::generic::ValidEulerHdgNumericalPolicy(
             request.numerical_policy) &&
      dg::hdg::generic::ResolveEulerHdgMaximumLinearIterations(
          request.numerical_policy.maximum_linear_iterations,
          global_true_trace) == explicit_cap;
}

bool CheckPeriodicAnnotationsRemainTraceCouplings()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{};
  BoundaryContext context{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, context, fields);
  request.numerical_policy.conventional_assembly_quadrature_order = 10;
  request.numerical_policy.maximum_linear_iterations = 64;
  request.numerical_policy.newton_correction_linear_relative_tolerance =
      1.0e-11;
  request.numerical_policy.final_coherent_linear_relative_tolerance =
      1.0e-11;
  request.numerical_policy.
      true_residual_tightening_linear_relative_tolerance = 1.0e-11;
  request.numerical_policy.true_residual_relative_tolerance = 1.0e-9;
  GenericEulerHdgCpuSession session;
  return session.Initialize(request).status ==
      GenericEulerCpuSessionStatus::initialized &&
      session.Advance(0.001).status ==
      GenericEulerCpuSessionStatus::advanced &&
      context.calls == 0;
}

bool CheckNaturalFreeStreamPreservation()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> free_stream{{1.0, 1.0, 0.20, 0.20,
                                           -0.10, -0.10, 2.60, 2.60}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, free_stream, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request) ||
      core.evidence().tuple.formulation.view() !=
          "hdg.conventional.v1" ||
      core.evidence().tuple.stabilization.view() !=
          dg::hdg::runtime::borrowed_euler_llf_stabilization_id)
  {
    return false;
  }

  std::vector<double> before(fields.initial.size(), 0.0);
  std::vector<double> candidate(fields.initial.size(), 0.0);
  std::vector<double> committed(fields.initial.size(), 0.0);
  if (!core.ExportCommittedInterior(before.data(), before.size())) return false;

  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation) ||
      generation == 0 ||
      !core.ExportPreparedInterior(
          generation, candidate.data(), candidate.size()) ||
      candidate != before)
  {
    return false;
  }
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(
          generation, coordinates, maximum_wave_speed, stable_dt) ||
      !core.ExportCommittedInterior(committed.data(), committed.size()))
  {
    return false;
  }

  const auto& attempt = core.evidence().last_attempt;
  return committed == before && attempt.committed &&
      LinearWorkAccountingExact(attempt) &&
      attempt.newton_correction_linear_iterations == 0 &&
      attempt.true_residual_tightenings == 0 &&
      attempt.status == dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
      attempt.nonlinear_updates == 0 &&
      attempt.initial_nonlinear_residual <=
          request.numerical_policy.nonlinear_relative_tolerance *
              (1.0 + attempt.initial_nonlinear_residual) &&
      attempt.final_nonlinear_residual ==
          attempt.initial_nonlinear_residual &&
      attempt.committed_interior_before == attempt.committed_interior_after &&
      attempt.committed_trace_before == attempt.committed_trace_after &&
      attempt.work.complete_assemblies == 1 &&
      attempt.work.trial_reassemblies == 0 &&
      attempt.work.accepted_newton_updates == 0 &&
      attempt.work.complete_condensations == 1 &&
      attempt.work.complete_factor_setups == 1 &&
      attempt.work.complete_trace_solves == 1 &&
      attempt.work.complete_recoveries == 1 &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate &&
      attempt.maximum_true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      coordinates.step == 1 && coordinates.generation == 1 &&
      coordinates.time == 0.001 && std::isfinite(maximum_wave_speed) &&
      maximum_wave_speed > 0.0 && std::isfinite(stable_dt) &&
      stable_dt > 0.0 && boundary.calls == 0;
}

bool CheckRodas3CommonShiftFreeStream()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> free_stream{{1.0, 1.0, 0.20, 0.20,
                                           -0.10, -0.10, 2.60, 2.60}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, free_stream, source);
  EulerHdgRequest request = Request(
      mesh, 1, boundary, fields, EulerHdgTimeTreatment::rodas3);
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request) ||
      core.evidence().tuple.time_treatment.view() !=
          "rodas3-common-gamma.v1")
  {
    std::cerr << "RODAS3 free-stream initialization failed\n";
    return false;
  }
  std::vector<double> before(fields.initial.size(), 0.0);
  std::vector<double> after(fields.initial.size(), 0.0);
  if (!core.ExportCommittedInterior(before.data(), before.size())) return false;
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  constexpr double dt = 1.0e-3;
  if (!core.PrepareBorrowedEuler(dt, coordinates, generation) ||
      generation == 0)
  {
    const auto& failed = core.evidence().last_attempt;
    std::cerr << "RODAS3 free-stream prepare failed status="
              << static_cast<int>(failed.status) << " phase="
              << static_cast<int>(failed.failure_phase) << " assemblies="
              << failed.work.complete_assemblies << " condensations="
              << failed.work.complete_condensations << " solves="
              << failed.work.complete_trace_solves << " recoveries="
              << failed.work.complete_recoveries << " builds="
              << failed.work.common_shift_operator_builds << " reuses="
              << failed.work.common_shift_operator_reuses << " rhs_refreshes="
              << failed.work.condensed_rhs_refreshes << '\n';
    return false;
  }
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!AssessPreparedRodas3(core, generation) ||
      !core.CommitBorrowedEuler(generation, coordinates,
                                maximum_wave_speed, stable_dt) ||
      !core.ExportCommittedInterior(after.data(), after.size()))
  {
    std::cerr << "RODAS3 free-stream commit/export failed\n";
    return false;
  }
  const auto& attempt = core.evidence().last_attempt;
  const bool passed = after == before && attempt.committed &&
      attempt.status == dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
      attempt.work.common_shift_operator_builds == 1 &&
      attempt.work.common_shift_operator_reuses == 3 &&
      attempt.work.condensed_rhs_refreshes == 3 &&
      attempt.work.complete_factor_setups <
          attempt.work.complete_trace_solves &&
      attempt.work.complete_trace_solves >= 4 &&
      attempt.work.complete_recoveries ==
          attempt.work.complete_trace_solves &&
      attempt.temporal_error_norm <=
          64.0 * std::numeric_limits<double>::epsilon() &&
      attempt.final_algebraic_correction_norm == 0.0 &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate && boundary.calls == 0 &&
      coordinates.step == 1 && coordinates.generation == 1 &&
      coordinates.time == dt && coordinates.dt == dt &&
      std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
      std::isfinite(stable_dt) && stable_dt > 0.0;
  if (!passed)
  {
    std::cerr << "RODAS3 free-stream evidence mismatch state_equal="
              << (after == before) << " status="
              << static_cast<int>(attempt.status) << " committed="
              << attempt.committed << " builds="
              << attempt.work.common_shift_operator_builds << " reuses="
              << attempt.work.common_shift_operator_reuses << " rhs_refreshes="
              << attempt.work.condensed_rhs_refreshes << " setups="
              << attempt.work.complete_factor_setups << " solves="
              << attempt.work.complete_trace_solves << " recoveries="
              << attempt.work.complete_recoveries << " temporal_error="
              << attempt.temporal_error_norm << " algebraic_correction="
              << attempt.final_algebraic_correction_norm << " gates="
              << attempt.finite_gate << attempt.admissibility_gate
              << attempt.recovery_gate << attempt.nonlinear_gate
              << attempt.true_residual_gate << " coordinates="
              << coordinates.time << ',' << coordinates.step << ','
              << coordinates.generation << ',' << coordinates.dt << '\n';
  }
  return passed;
}

bool CheckRodas3NontrivialDaeLifecycle()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks != 1 && ranks != 2) return true;

  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  std::array<std::array<BoundaryContext, 2>, 3> boundaries{};
  constexpr double dt = 1.0e-4;

  for (int order = 1; order <= 3; ++order)
  {
    ExactBorrowedEulerFields fields(mesh, order, initial, source);
    for (int regularized = 0; regularized < 2; ++regularized)
    {
      BoundaryContext& boundary =
          boundaries[static_cast<std::size_t>(order - 1)]
                    [static_cast<std::size_t>(regularized)];
      EulerHdgRequest request = Request(
          mesh, order, boundary, fields,
          EulerHdgTimeTreatment::rodas3);
      if (regularized != 0)
        request.numerical_policy.stabilization =
            {RegularizationKind::igr, 0.2, 0.4};

      dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
      if (!core.InitializeBorrowedEuler(request)) return false;
      std::vector<double> committed_before(fields.initial.size(), 0.0);
      if (!core.ExportCommittedInterior(committed_before.data(),
                                        committed_before.size()))
      {
        return false;
      }

      dg::hdg::runtime::StateCoordinates coordinates{};
      std::uint64_t first_generation = 0;
      if (!core.PrepareBorrowedEuler(dt, coordinates, first_generation) ||
          first_generation == 0)
      {
        return false;
      }
      std::vector<double> first_candidate(fields.initial.size(), 0.0);
      if (!core.ExportPreparedInterior(first_generation,
                                       first_candidate.data(),
                                       first_candidate.size()) ||
          first_candidate == committed_before ||
          !AdmissibleEulerCoefficients(first_candidate) ||
          !core.AbortBorrowedEuler(first_generation, coordinates))
      {
        return false;
      }
      std::vector<double> after_abort(fields.initial.size(), 0.0);
      if (!core.ExportCommittedInterior(
              after_abort.data(), after_abort.size()) ||
          after_abort != committed_before ||
          !core.evidence().last_attempt.rollback_byte_equal)
      {
        return false;
      }

      std::uint64_t retry_generation = 0;
      if (!core.PrepareBorrowedEuler(dt, coordinates, retry_generation) ||
          retry_generation == 0)
      {
        return false;
      }
      std::vector<double> retry_candidate(fields.initial.size(), 0.0);
      if (!core.ExportPreparedInterior(retry_generation,
                                       retry_candidate.data(),
                                       retry_candidate.size()) ||
          retry_candidate != first_candidate)
      {
        return false;
      }
      double maximum_wave_speed = 0.0;
      double stable_dt = 0.0;
      if (!AssessPreparedRodas3(core, retry_generation) ||
          !core.CommitBorrowedEuler(retry_generation, coordinates,
                                    maximum_wave_speed, stable_dt))
      {
        return false;
      }

      double closure_residual = 0.0;
      double closure_tolerance = 0.0;
      const bool closure_ready = regularized == 0 ||
          dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
              AuditCommittedClosure(core, closure_residual,
                                    closure_tolerance);
      const auto& attempt = core.evidence().last_attempt;
      const bool accepted = attempt.committed &&
        attempt.status ==
            dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
        attempt.work.common_shift_operator_builds == 1 &&
        attempt.work.common_shift_operator_reuses == 3 &&
        attempt.work.condensed_rhs_refreshes == 3 &&
        attempt.work.complete_factor_setups <
            attempt.work.complete_trace_solves &&
        attempt.work.complete_recoveries ==
            attempt.work.complete_trace_solves &&
        std::isfinite(attempt.temporal_error_norm) &&
        attempt.temporal_error_norm > 0.0 &&
        std::isfinite(attempt.final_algebraic_residual) &&
        std::isfinite(attempt.final_algebraic_correction_norm) &&
        attempt.finite_gate && attempt.admissibility_gate &&
        attempt.recovery_gate && attempt.nonlinear_gate &&
        attempt.true_residual_gate && closure_ready &&
        (regularized == 0 ||
         (std::isfinite(closure_residual) &&
          std::isfinite(closure_tolerance) &&
          closure_residual <= closure_tolerance)) &&
        coordinates.time == dt && coordinates.step == 1 &&
        coordinates.generation == 1 && coordinates.dt == dt &&
        std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
        std::isfinite(stable_dt) && stable_dt > 0.0 &&
        boundary.calls == 0;
      if (!accepted)
      {
        std::cerr << "RODAS3 nontrivial DAE lifecycle failed order="
                << order << " igr=" << regularized << " status="
                << static_cast<int>(attempt.status) << " builds="
                << attempt.work.common_shift_operator_builds << " reuses="
                << attempt.work.common_shift_operator_reuses
                << " rhs_refreshes="
                << attempt.work.condensed_rhs_refreshes << " setups="
                << attempt.work.complete_factor_setups << " solves="
                << attempt.work.complete_trace_solves << " recoveries="
                << attempt.work.complete_recoveries << " error="
                << attempt.temporal_error_norm << " algebraic_residual="
                << attempt.final_algebraic_residual
                << " algebraic_correction="
                << attempt.final_algebraic_correction_norm << " closure="
                << closure_residual << '/' << closure_tolerance << '\n';
        return false;
      }
    }
  }
  return true;
}

bool CheckRodas3IgrInitialScalarClosure()
{
  constexpr int order = 1;
  constexpr int igr_interior_components = 7;
  constexpr int igr_trace_components = 5;
  const auto zero_coordinates =
      [](const dg::hdg::runtime::StateCoordinates& coordinates) noexcept
  {
    return coordinates.time == 0.0 && coordinates.step == 0 &&
        coordinates.generation == 0 && coordinates.cursor == 0 &&
        coordinates.dt == 0.0;
  };
  const auto zero_work = [](const auto& work) noexcept
  {
    return work.complete_assemblies == 0 &&
        work.complete_condensations == 0 &&
        work.complete_factor_setups == 0 &&
        work.complete_trace_solves == 0 &&
        work.complete_scalar_closure_solves == 0 &&
        work.complete_recoveries == 0 && work.trial_reassemblies == 0 &&
        work.accepted_newton_updates == 0 && work.phase_entries == 0 &&
        work.phase_completions == 0 &&
        work.common_shift_operator_builds == 0 &&
        work.common_shift_operator_reuses == 0 &&
        work.condensed_rhs_refreshes == 0;
  };
  const auto same_bytes = [](const double* const left,
                             const double* const right,
                             const std::size_t count) noexcept
  {
    return count == 0 ||
        (left != nullptr && right != nullptr &&
         std::memcmp(left, right, count * sizeof(double)) == 0);
  };

  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.00, 1.08, 0.16, 0.09,
                                      -0.03, 0.04, 2.70, 2.88}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  ExactBorrowedEulerFields fields(mesh, order, initial, source);

  // Construct a coherent Uhat4 and then supply it explicitly, so the IGR
  // initialization below proves preservation of caller-owned trace bytes.
  BoundaryContext reference_boundary{};
  dg::hdg::runtime::cpu::detail::CpuNumericsCore reference;
  const EulerHdgRequest reference_request = Request(
      mesh, order, reference_boundary, fields);
  if (!reference.InitializeBorrowedEuler(reference_request)) return false;
  const auto reference_snapshot = reference.Snapshot({});
  if (reference_snapshot.trace.size == 0 ||
      reference_snapshot.trace.size % 4 != 0 ||
      reference_snapshot.trace.data == nullptr)
  {
    return false;
  }
  std::vector<double> caller_trace(
      reference_snapshot.trace.data,
      reference_snapshot.trace.data + reference_snapshot.trace.size);

  BoundaryContext boundary{};
  EulerHdgRequest request = Request(
      mesh, order, boundary, fields, EulerHdgTimeTreatment::rodas3);
  request.numerical_policy.stabilization =
      {RegularizationKind::igr, 0.2, 0.4};
  request.initial_true_trace = {
      caller_trace.data(), caller_trace.size(), 4};
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return false;

  dg::hdg::runtime::StateCoordinates coordinates{};
  const auto initialized = core.Snapshot(coordinates);
  const bool layout_ready = initialized.interior.data != nullptr &&
      initialized.trace.data != nullptr &&
      initialized.interior.size % igr_interior_components == 0 &&
      initialized.trace.size % igr_trace_components == 0;
  const std::size_t interior_scalar = layout_ready
      ? initialized.interior.size / igr_interior_components : 0;
  const std::size_t trace_scalar = layout_ready
      ? initialized.trace.size / igr_trace_components : 0;
  const bool caller_u4_preserved = layout_ready &&
      fields.initial.size() == 4 * interior_scalar &&
      same_bytes(initialized.interior.data, fields.initial.data(),
                 fields.initial.size());
  const bool caller_uhat4_preserved = layout_ready &&
      caller_trace.size() == 4 * trace_scalar &&
      same_bytes(initialized.trace.data, caller_trace.data(),
                 caller_trace.size());

  // The three trailing volume components are sigma and the two entries of z;
  // the trailing trace component is sigma-hat.  Each algebraic field must be
  // finite and globally nontrivial after the t=0 closure solve.
  bool local_auxiliary_finite = layout_ready;
  std::array<double, 3> local_auxiliary_maximum{};
  if (layout_ready)
  {
    for (std::size_t scalar = 0; scalar < interior_scalar; ++scalar)
    {
      const double sigma = initialized.interior.data[
          4 * interior_scalar + scalar];
      local_auxiliary_finite =
          local_auxiliary_finite && std::isfinite(sigma);
      if (std::isfinite(sigma))
        local_auxiliary_maximum[0] =
            std::max(local_auxiliary_maximum[0], std::abs(sigma));
      for (int component = 5; component < igr_interior_components;
           ++component)
      {
        const double z = initialized.interior.data[
            static_cast<std::size_t>(component) * interior_scalar + scalar];
        local_auxiliary_finite =
            local_auxiliary_finite && std::isfinite(z);
        if (std::isfinite(z))
          local_auxiliary_maximum[1] =
              std::max(local_auxiliary_maximum[1], std::abs(z));
      }
    }
    for (std::size_t scalar = 0; scalar < trace_scalar; ++scalar)
    {
      const double sigma_hat =
          initialized.trace.data[4 * trace_scalar + scalar];
      local_auxiliary_finite =
          local_auxiliary_finite && std::isfinite(sigma_hat);
      if (std::isfinite(sigma_hat))
        local_auxiliary_maximum[2] =
            std::max(local_auxiliary_maximum[2], std::abs(sigma_hat));
    }
  }
  std::array<double, 3> global_auxiliary_maximum{};
  const bool auxiliary_reduced = MPI_Allreduce(
      local_auxiliary_maximum.data(), global_auxiliary_maximum.data(),
      static_cast<int>(global_auxiliary_maximum.size()), MPI_DOUBLE, MPI_MAX,
      mesh.GetComm()) == MPI_SUCCESS;
  const bool auxiliary_finite = All(local_auxiliary_finite, mesh.GetComm());
  const bool auxiliary_nontrivial = auxiliary_reduced &&
      std::all_of(global_auxiliary_maximum.begin(),
                  global_auxiliary_maximum.end(),
                  [](const double value)
                  { return std::isfinite(value) && value > 0.0; });

  std::vector<double> initialized_interior(
      initialized.interior.data,
      initialized.interior.data + initialized.interior.size);
  std::vector<double> initialized_trace(
      initialized.trace.data,
      initialized.trace.data + initialized.trace.size);
  double closure_residual = std::numeric_limits<double>::quiet_NaN();
  double closure_tolerance = std::numeric_limits<double>::quiet_NaN();
  const bool closure_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditCommittedClosure(
              core, closure_residual, closure_tolerance) &&
      std::isfinite(closure_residual) && closure_residual >= 0.0 &&
      std::isfinite(closure_tolerance) && closure_tolerance > 0.0 &&
      closure_residual <= closure_tolerance;
  const auto after_audit = core.Snapshot(coordinates);
  const bool audit_preserved_state =
      after_audit.interior.size == initialized_interior.size() &&
      after_audit.trace.size == initialized_trace.size() &&
      same_bytes(after_audit.interior.data, initialized_interior.data(),
                 initialized_interior.size()) &&
      same_bytes(after_audit.trace.data, initialized_trace.data(),
                 initialized_trace.size());

  const auto& initial_evidence = core.evidence();
  const bool zero_initial_accounting = zero_coordinates(coordinates) &&
      zero_coordinates(initialized.coordinates) &&
      zero_coordinates(after_audit.coordinates) &&
      initial_evidence.accepted_step_count == 0 &&
      initial_evidence.attempts == 0 &&
      initial_evidence.failed_attempts == 0 &&
      initial_evidence.retries == 0 && initial_evidence.commits == 0 &&
      !initial_evidence.accepted.valid &&
      initial_evidence.last_attempt.status ==
          dg::hdg::runtime::cpu::CpuAttemptStatus::ready &&
      initial_evidence.last_attempt.candidate_cursor == 0 &&
      initial_evidence.last_attempt.nonlinear_updates == 0 &&
      initial_evidence.last_attempt.linear_iterations == 0 &&
      initial_evidence.last_attempt.linear_solve_attempts == 0 &&
      initial_evidence.last_attempt.scalar_closure_linear_iterations == 0 &&
      initial_evidence.last_attempt.scalar_closure_linear_solve_attempts == 0 &&
      zero_coordinates(initial_evidence.last_attempt.entry_coordinates) &&
      zero_coordinates(initial_evidence.last_attempt.exit_coordinates) &&
      zero_work(initial_evidence.last_attempt.work);

  std::uint64_t generation = 0;
  const bool prepared =
      core.PrepareBorrowedEuler(1.0e-4, coordinates, generation) &&
      generation != 0;
  const auto committed_while_prepared = core.Snapshot(coordinates);
  const auto candidate = core.CandidateSnapshot(coordinates);
  const bool prepare_is_transactional = prepared &&
      zero_coordinates(coordinates) &&
      same_bytes(committed_while_prepared.interior.data,
                 initialized_interior.data(), initialized_interior.size()) &&
      same_bytes(committed_while_prepared.trace.data,
                 initialized_trace.data(), initialized_trace.size()) &&
      candidate.interior.size == initialized_interior.size() &&
      candidate.trace.size == initialized_trace.size();
  const bool aborted = prepared &&
      core.AbortBorrowedEuler(generation, coordinates);
  const auto after_abort = core.Snapshot(coordinates);
  const auto candidate_after_abort = core.CandidateSnapshot(coordinates);
  double rollback_closure_residual =
      std::numeric_limits<double>::quiet_NaN();
  double rollback_closure_tolerance =
      std::numeric_limits<double>::quiet_NaN();
  const bool rollback_closure_audited = aborted &&
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditCommittedClosure(
              core, rollback_closure_residual,
              rollback_closure_tolerance) &&
      std::isfinite(rollback_closure_residual) &&
      std::isfinite(rollback_closure_tolerance) &&
      rollback_closure_residual <= rollback_closure_tolerance;
  const auto& abort_evidence = core.evidence();
  const bool abort_is_transactional = aborted &&
      zero_coordinates(coordinates) &&
      zero_coordinates(after_abort.coordinates) &&
      after_abort.interior.size == initialized_interior.size() &&
      after_abort.trace.size == initialized_trace.size() &&
      same_bytes(after_abort.interior.data, initialized_interior.data(),
                 initialized_interior.size()) &&
      same_bytes(after_abort.trace.data, initialized_trace.data(),
                 initialized_trace.size()) &&
      candidate_after_abort.interior.size == initialized_interior.size() &&
      candidate_after_abort.trace.size == initialized_trace.size() &&
      same_bytes(candidate_after_abort.interior.data,
                 initialized_interior.data(), initialized_interior.size()) &&
      same_bytes(candidate_after_abort.trace.data,
                 initialized_trace.data(), initialized_trace.size()) &&
      abort_evidence.attempts == 1 &&
      abort_evidence.failed_attempts == 1 &&
      abort_evidence.accepted_step_count == 0 &&
      abort_evidence.commits == 0 && !abort_evidence.accepted.valid &&
      !abort_evidence.last_attempt.committed &&
      abort_evidence.last_attempt.rollback_byte_equal &&
      abort_evidence.last_attempt.status ==
          dg::hdg::runtime::cpu::CpuAttemptStatus::runtime_failure;

  const bool passed = caller_u4_preserved && caller_uhat4_preserved &&
      auxiliary_finite && auxiliary_nontrivial && closure_audited &&
      audit_preserved_state && zero_initial_accounting &&
      prepare_is_transactional && rollback_closure_audited &&
      abort_is_transactional && reference_boundary.calls == 0 &&
      boundary.calls == 0;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(mesh.GetComm(), &rank);
    std::cerr << "RODAS3 IGR initial scalar closure failed rank=" << rank
              << " U4=" << caller_u4_preserved
              << " Uhat4=" << caller_uhat4_preserved
              << " auxiliary_finite=" << auxiliary_finite
              << " auxiliary_nontrivial=" << auxiliary_nontrivial
              << " auxiliary_maxima=" << global_auxiliary_maximum[0] << ','
              << global_auxiliary_maximum[1] << ','
              << global_auxiliary_maximum[2]
              << " closure=" << closure_residual << '/'
              << closure_tolerance
              << " audit_state=" << audit_preserved_state
              << " zero_accounting=" << zero_initial_accounting
              << " prepared=" << prepared
              << " prepare_transaction=" << prepare_is_transactional
              << " aborted=" << aborted
              << " rollback_closure=" << rollback_closure_residual << '/'
              << rollback_closure_tolerance
              << " abort_transaction=" << abort_is_transactional << '\n';
  }
  return passed;
}

bool CheckRodas3PreparedAssessmentAuthority()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  EulerHdgRequest request = Request(
      mesh, 2, boundary, fields, EulerHdgTimeTreatment::rodas3);
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  const auto assessment_capability =
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::
              prepared_embedded_assessment);
  if (!dg::hdg::generic::HasHdgEulerSessionCapabilities(
          session.capabilities(), assessment_capability))
  {
    return false;
  }

  GenericEulerCpuPreparedStep prepared;
  constexpr double dt = 1.0e-4;
  if (session.Prepare(dt, prepared).status !=
          GenericEulerCpuSessionStatus::prepared ||
      session.Commit(std::move(prepared)).status !=
          GenericEulerCpuSessionStatus::rejected ||
      session.coordinates().step != 0)
  {
    return false;
  }

  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy policy{
      1.0e-8, 1.0e-5, 0.25};
  dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
  int rank = -1;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) return false;
  const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      rank_local_policy = rank == 0
      ? dg::hdg::generic::HdgEulerPreparedAssessmentPolicy{0.0, 0.0, 0.25}
      : policy;
  // A rank-local malformed policy must reject collectively before any peer
  // enters the core policy-consensus collectives.  The live prepared token
  // remains valid for a subsequent coherent assessment.
  if (session.AssessPrepared(prepared, rank_local_policy, assessment))
  {
    return false;
  }
  const bool assessed = session.AssessPrepared(
      prepared, policy, assessment);
  const bool gates = assessment.finite_gate && assessment.solver_gate &&
      assessment.residual_gate && assessment.admissibility_gate &&
      assessment.algebraic_closure_gate &&
      assessment.algebraic_correction_gate;
  if (!assessed || !gates ||
      !dg::hdg::generic::IsValidHdgEulerPreparedAssessment(assessment) ||
      session.Commit(std::move(prepared)).status !=
          GenericEulerCpuSessionStatus::committed)
  {
    return false;
  }
  if (session.coordinates().step != 1 ||
      session.coordinates().generation != 1 ||
      session.coordinates().time != dt || session.coordinates().dt != dt)
  {
    return false;
  }

  // Exercise the core boundary directly so no facade-only token state can
  // mask a publication bypass.  A failed replacement assessment must revoke
  // a prior passing policy for the same still-live generation.
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.InitializeBorrowedEuler(request) ||
      !core.PrepareBorrowedEuler(dt, coordinates, generation) ||
      generation == 0)
  {
    return false;
  }
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (core.CommitBorrowedEuler(generation, coordinates,
                               maximum_wave_speed, stable_dt))
  {
    return false;
  }
  dg::hdg::generic::HdgEulerPreparedAssessment core_assessment{};
  if (!core.AssessPreparedBorrowedEuler(
          generation, policy, core_assessment) ||
      !core_assessment.algebraic_correction_gate)
  {
    return false;
  }
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      invalid_replacement{0.0, 0.0, 0.25};
  if (core.AssessPreparedBorrowedEuler(
          generation, invalid_replacement, core_assessment) ||
      core.CommitBorrowedEuler(generation, coordinates,
                               maximum_wave_speed, stable_dt))
  {
    return false;
  }
  if (!core.AssessPreparedBorrowedEuler(
          generation, policy, core_assessment) ||
      !core_assessment.algebraic_correction_gate ||
      !core.CommitBorrowedEuler(generation, coordinates,
                                maximum_wave_speed, stable_dt))
  {
    return false;
  }
  return coordinates.step == 1 &&
      coordinates.generation == 1 && coordinates.time == dt &&
      coordinates.dt == dt &&
      session.coordinates().generation == 1 &&
      std::isfinite(maximum_wave_speed) && std::isfinite(stable_dt) &&
      stable_dt > 0.0;
}

bool CheckRodas3FixedTimeThirdOrder()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks != 1) return true;

  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  std::array<BoundaryContext, 3> boundaries{};
  OneStepEulerState coarse;
  OneStepEulerState medium;
  OneStepEulerState fine;
  constexpr double final_time = 8.0e-4;
  constexpr int coarse_steps = 4;
  const bool coarse_ok = RunEulerSteps(
      mesh, fields, boundaries[0], EulerHdgTimeTreatment::rodas3,
      2, false, final_time / coarse_steps, coarse, coarse_steps);
  const bool medium_ok = RunEulerSteps(
      mesh, fields, boundaries[1], EulerHdgTimeTreatment::rodas3,
      2, false, final_time / (2 * coarse_steps), medium, 2 * coarse_steps);
  const bool fine_ok = RunEulerSteps(
      mesh, fields, boundaries[2], EulerHdgTimeTreatment::rodas3,
      2, false, final_time / (4 * coarse_steps), fine, 4 * coarse_steps);
  if (!coarse_ok || !medium_ok || !fine_ok)
  {
    std::cerr << "RODAS3 fixed-time order run failed coarse=" << coarse_ok
              << " medium=" << medium_ok << " fine=" << fine_ok << '\n';
    return false;
  }

  const double u4_coarse = MaximumScaledDifference(
      coarse.interior_u4, medium.interior_u4);
  const double u4_fine = MaximumScaledDifference(
      medium.interior_u4, fine.interior_u4);
  const double uhat_coarse = MaximumScaledDifference(
      coarse.trace_uhat4, medium.trace_uhat4);
  const double uhat_fine = MaximumScaledDifference(
      medium.trace_uhat4, fine.trace_uhat4);
  constexpr double assessment_absolute_tolerance = 1.0e-8;
  constexpr double assessment_relative_tolerance = 1.0e-5;
  const double weighted_u4_coarse = QuadratureScaledU4Difference(
      mesh, 2, coarse.interior_u4, medium.interior_u4,
      assessment_absolute_tolerance, assessment_relative_tolerance);
  const double weighted_u4_fine = QuadratureScaledU4Difference(
      mesh, 2, medium.interior_u4, fine.interior_u4,
      assessment_absolute_tolerance, assessment_relative_tolerance);
  constexpr double roundoff_floor =
      8192.0 * std::numeric_limits<double>::epsilon();
  const auto third_order = [roundoff_floor](
      const double coarse_difference, const double fine_difference) noexcept
  {
    const double ratio = coarse_difference /
        std::max(fine_difference, roundoff_floor);
    return std::isfinite(coarse_difference) &&
        std::isfinite(fine_difference) &&
        coarse_difference > 32.0 * roundoff_floor &&
        fine_difference > roundoff_floor &&
        ratio >= 6.0 && ratio <= 10.0;
  };
  const auto estimator_third_order = [roundoff_floor](
      const double coarse_estimate, const double fine_estimate) noexcept
  {
    const double ratio = coarse_estimate /
        std::max(fine_estimate, roundoff_floor);
    return std::isfinite(coarse_estimate) &&
        std::isfinite(fine_estimate) &&
        coarse_estimate > 32.0 * roundoff_floor &&
        fine_estimate > 4.0 * roundoff_floor &&
        ratio >= 6.0 && ratio <= 10.0;
  };
  const auto finite_positive = [](const double value) noexcept
  {
    return std::isfinite(value) && value > 0.0;
  };
  const double coarse_effectivity =
      coarse.normalized_temporal_error / weighted_u4_coarse;
  const double fine_effectivity =
      medium.normalized_temporal_error / weighted_u4_fine;
  const double effectivity_ratio = coarse_effectivity / fine_effectivity;
  const bool normalized_estimator_calibrated =
      estimator_third_order(coarse.normalized_temporal_error,
                            medium.normalized_temporal_error) &&
      estimator_third_order(medium.normalized_temporal_error,
                            fine.normalized_temporal_error) &&
      estimator_third_order(weighted_u4_coarse, weighted_u4_fine) &&
      finite_positive(coarse_effectivity) && finite_positive(fine_effectivity) &&
      std::isfinite(effectivity_ratio) && effectivity_ratio >= 0.2 &&
      effectivity_ratio <= 5.0 &&
      coarse.normalized_algebraic_correction <= 0.25 &&
      medium.normalized_algebraic_correction <= 0.25 &&
      fine.normalized_algebraic_correction <= 0.25;
  // On one fixed spatial discretization, both Richardson differences and the
  // principal/embedded one-step defect remain resolved above the numerical
  // floor.  A halved step therefore gives the expected factor-eight decrease
  // before spatial or nonlinear-solve error can mask the temporal behavior.
  const bool accepted = third_order(u4_coarse, u4_fine) &&
      third_order(uhat_coarse, uhat_fine) &&
      estimator_third_order(coarse.temporal_error_norm,
                            medium.temporal_error_norm) &&
      estimator_third_order(medium.temporal_error_norm,
                            fine.temporal_error_norm) &&
      normalized_estimator_calibrated &&
      std::all_of(boundaries.begin(), boundaries.end(),
                  [](const BoundaryContext& boundary)
                  { return boundary.calls == 0; });
  int rank = -1;
  if (MPI_Comm_rank(mesh.GetComm(), &rank) == MPI_SUCCESS && rank == 0)
  {
    std::cout << "[cpu-rodas3-estimator] normalized="
              << coarse.normalized_temporal_error << ','
              << medium.normalized_temporal_error << ','
              << fine.normalized_temporal_error
              << " weighted_richardson=" << weighted_u4_coarse << ','
              << weighted_u4_fine << " effectivity="
              << coarse_effectivity << ',' << fine_effectivity
              << " effectivity_ratio=" << effectivity_ratio
              << " passed=" << accepted << '\n';
  }
  if (!accepted)
  {
    std::cerr << "RODAS3 fixed-time order failed U4_coarse=" << u4_coarse
              << " U4_fine=" << u4_fine << " U4_ratio="
              << u4_coarse / std::max(u4_fine, roundoff_floor)
              << " Uhat4_coarse=" << uhat_coarse
              << " Uhat4_fine=" << uhat_fine << " Uhat4_ratio="
              << uhat_coarse / std::max(uhat_fine, roundoff_floor)
              << " estimator=" << coarse.temporal_error_norm << ' '
              << medium.temporal_error_norm << ' '
              << fine.temporal_error_norm << " estimator_ratios="
              << coarse.temporal_error_norm /
                    std::max(medium.temporal_error_norm, roundoff_floor)
              << ' ' << medium.temporal_error_norm /
                    std::max(fine.temporal_error_norm, roundoff_floor)
              << " normalized_estimator="
              << coarse.normalized_temporal_error << ' '
              << medium.normalized_temporal_error << ' '
              << fine.normalized_temporal_error
              << " weighted_Richardson=" << weighted_u4_coarse << ' '
              << weighted_u4_fine << " effectivity="
              << coarse_effectivity << ' ' << fine_effectivity
              << " effectivity_ratio=" << effectivity_ratio
              << " normalized_algebraic="
              << coarse.normalized_algebraic_correction << ' '
              << medium.normalized_algebraic_correction << ' '
              << fine.normalized_algebraic_correction
              << " roundoff_floor=" << roundoff_floor << '\n';
  }
  return accepted;
}

bool CheckRodas3PhysicalBoundaryConservation()
{
  constexpr std::array<double, 8> initial{{
      1.10, 1.12, 0.08, 0.09, -0.04, -0.03, 2.90, 2.95}};
  constexpr std::array<double, 8> source{{
      0.01, 0.02, -0.01, 0.00, 0.00, 0.01, 0.03, 0.04}};
  constexpr double dt = 1.0e-4;
  bool local = true;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int regularized = 0; regularized < 2; ++regularized)
  {
    mfem::ParMesh mesh = MakeBorrowedMesh();
    BoundaryContext boundary{};
    ExactBorrowedEulerFields fields(mesh, 2, initial, source);
    EulerHdgRequest request = Request(
        mesh, 2, boundary, fields, EulerHdgTimeTreatment::rodas3);
    request.boundary = {
        &boundary, SmoothAutonomousExteriorForRodasJacobianForTests,
        {nullptr, 0, 0, 0.0}};
    if (regularized != 0)
      request.numerical_policy.stabilization =
          {RegularizationKind::igr, 0.2, 0.4};

    dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
    dg::hdg::runtime::StateCoordinates coordinates{};
    std::uint64_t generation = 0;
    double endpoint_error = std::numeric_limits<double>::infinity();
    double endpoint_relative_error =
        std::numeric_limits<double>::infinity();
    bool prepared = core.InitializeBorrowedEuler(request) &&
        core.PrepareBorrowedEuler(dt, coordinates, generation) &&
        generation != 0;
    const bool endpoint_audited = prepared &&
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditPreparedRodasEndpointConservation(
                core, endpoint_error, endpoint_relative_error);
    double maximum_wave_speed = 0.0;
    double stable_dt = 0.0;
    const bool committed = endpoint_audited &&
        AssessPreparedRodas3(core, generation) &&
        core.CommitBorrowedEuler(generation, coordinates,
                                 maximum_wave_speed, stable_dt);
    double rodas_error = std::numeric_limits<double>::infinity();
    double rodas_relative_error = std::numeric_limits<double>::infinity();
    bool conservation_gate = false;
    const bool audited = committed &&
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditLastConservation(
                core, rodas_error, rodas_relative_error,
                conservation_gate);
    int global_boundary_calls = 0;
    const bool calls_reduced = MPI_Allreduce(
        &boundary.calls, &global_boundary_calls, 1, MPI_INT, MPI_SUM,
        mesh.GetComm()) == MPI_SUCCESS;
    const auto& attempt = core.evidence().last_attempt;
    // The endpoint expression is not merely numerically different: it would
    // reject this step under the requested production tolerance, while the
    // stage-weighted ROW identity accepts it.  This avoids a fixture-specific
    // magic lower bound.
    const bool verified = audited && conservation_gate && calls_reduced &&
        global_boundary_calls > 0 && attempt.committed &&
        attempt.status ==
            dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
        std::isfinite(rodas_error) && rodas_error >= 0.0 &&
        std::isfinite(rodas_relative_error) &&
        rodas_relative_error <=
            request.numerical_policy.conservation_relative_tolerance &&
        std::isfinite(endpoint_error) && endpoint_error > 0.0 &&
        std::isfinite(endpoint_relative_error) &&
        endpoint_relative_error >
            request.numerical_policy.conservation_relative_tolerance &&
        endpoint_relative_error > rodas_relative_error &&
        coordinates.step == 1 && coordinates.generation == 1 &&
        coordinates.time == dt && coordinates.dt == dt &&
        std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
        std::isfinite(stable_dt) && stable_dt > 0.0;
    if (!verified && rank == 0)
    {
      std::cerr << "RODAS3 physical-boundary conservation failed igr="
                << regularized << " prepared=" << prepared
                << " endpoint_audited=" << endpoint_audited
                << " committed=" << committed << " audited=" << audited
                << " gate=" << conservation_gate << " ROW="
                << rodas_error << '/' << rodas_relative_error
                << " endpoint=" << endpoint_error << '/'
                << endpoint_relative_error << " tolerance="
                << request.numerical_policy.conservation_relative_tolerance
                << " boundary_calls=" << global_boundary_calls << '\n';
    }
    local = local && verified;
  }
  return All(local);
}

bool CheckRodas3SingularMassStageEquationsForTests()
{
  constexpr std::array<double, 8> initial{{
      1.10, 1.12, 0.08, 0.09, -0.04, -0.03, 2.90, 2.95}};
  constexpr std::array<double, 8> source{{
      0.01, 0.02, -0.01, 0.00, 0.00, 0.01, 0.03, 0.04}};
  constexpr double dt = 1.0e-4;
  constexpr double stage_equation_relative_tolerance = 1.0e-11;
  constexpr double counterfactual_relative_floor = 1.0e-3;
  bool local = true;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int fixture = 0; fixture < 4; ++fixture)
  {
    const bool physical_boundary = fixture >= 2;
    const bool regularized = (fixture & 1) != 0;
    mfem::ParMesh mesh = physical_boundary
        ? MakeBorrowedMesh() : MakePeriodicBorrowedMesh();
    BoundaryContext boundary{};
    ExactBorrowedEulerFields fields(mesh, 1, initial, source);
    EulerHdgRequest request = Request(
        mesh, 1, boundary, fields, EulerHdgTimeTreatment::rodas3);
    if (physical_boundary)
      request.boundary = {
          &boundary, SmoothAutonomousExteriorForRodasJacobianForTests,
          {nullptr, 0, 0, 0.0}};
    if (regularized)
      request.numerical_policy.stabilization =
          {RegularizationKind::igr, 0.2, 0.4};

    dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
    dg::hdg::runtime::StateCoordinates coordinates{};
    const auto committed_before = core.InitializeBorrowedEuler(request)
        ? core.Snapshot(coordinates)
        : dg::hdg::runtime::cpu::CpuStateSnapshot{};
    std::uint64_t generation = 0;
    const bool prepared = committed_before.interior.data &&
        core.PrepareBorrowedEuler(dt, coordinates, generation) &&
        generation != 0;
    const auto candidate_before = prepared
        ? core.CandidateSnapshot(coordinates)
        : dg::hdg::runtime::cpu::CpuStateSnapshot{};
    std::array<double, 4> absolute_residuals{};
    std::array<double, 4> relative_residuals{};
    std::array<double, 4> counterfactual_relative_residuals{};
    double maximum_zero_mass_entry =
        std::numeric_limits<double>::infinity();
    const bool audited = prepared &&
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditPreparedRodasStageEquations(
                core, absolute_residuals, relative_residuals,
                counterfactual_relative_residuals,
                maximum_zero_mass_entry);
    const auto candidate_after = audited
        ? core.CandidateSnapshot(coordinates)
        : dg::hdg::runtime::cpu::CpuStateSnapshot{};
    const bool exact_passes = audited && std::all_of(
        relative_residuals.begin(), relative_residuals.end(),
        [](const double residual)
        {
          return std::isfinite(residual) && residual >= 0.0 &&
              residual <= stage_equation_relative_tolerance;
        });
    const bool counterfactual_fails = audited && std::all_of(
        counterfactual_relative_residuals.begin(),
        counterfactual_relative_residuals.end(),
        [](const double residual)
        {
          return std::isfinite(residual) &&
              residual > counterfactual_relative_floor;
        });
    const bool candidate_unchanged = audited &&
        candidate_before.state_hash == candidate_after.state_hash &&
        candidate_before.interior_hash == candidate_after.interior_hash &&
        candidate_before.trace_hash == candidate_after.trace_hash;
    const bool aborted = prepared &&
        core.AbortBorrowedEuler(generation, coordinates);
    const auto committed_after = aborted
        ? core.Snapshot(coordinates)
        : dg::hdg::runtime::cpu::CpuStateSnapshot{};
    int global_boundary_calls = 0;
    const bool calls_reduced = MPI_Allreduce(
        &boundary.calls, &global_boundary_calls, 1, MPI_INT, MPI_SUM,
        mesh.GetComm()) == MPI_SUCCESS;
    const bool boundary_exercised = physical_boundary
        ? global_boundary_calls > 0 : global_boundary_calls == 0;
    const bool verified = exact_passes && counterfactual_fails &&
        maximum_zero_mass_entry == 0.0 && candidate_unchanged && aborted &&
        committed_before.state_hash == committed_after.state_hash &&
        calls_reduced && boundary_exercised && coordinates.time == 0.0 &&
        coordinates.step == 0 && coordinates.generation == 0 &&
        coordinates.cursor == 0 && coordinates.dt == 0.0;
    if (!verified && rank == 0)
    {
      std::cerr << "RODAS3 singular-mass stage audit failed fixture="
                << fixture << " prepared=" << prepared
                << " audited=" << audited << " zero_mass="
                << maximum_zero_mass_entry << " boundary_calls="
                << global_boundary_calls << " absolute=";
      for (const double value : absolute_residuals)
        std::cerr << ' ' << value;
      std::cerr << " relative=";
      for (const double value : relative_residuals)
        std::cerr << ' ' << value;
      std::cerr << " counterfactual=";
      for (const double value : counterfactual_relative_residuals)
        std::cerr << ' ' << value;
      std::cerr << '\n';
    }
    local = local && verified;
  }
  return All(local);
}

bool CheckRodas3ScaledInconsistentEntryRollback()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  constexpr std::array<double, 8> initial{{
      1.00, 1.05, 0.10, 0.12, 0.00, 0.01, 2.55, 2.62}};
  constexpr std::array<double, 8> source{{
      0.01, 0.02, 0.00, 0.01, 0.00, 0.00, 0.03, 0.04}};
  constexpr double dt = 1.0e-4;
  BoundaryContext reference_boundary{};
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  EulerHdgRequest reference_request = Request(
      mesh, 2, reference_boundary, fields,
      EulerHdgTimeTreatment::rodas3);

  std::vector<double> coherent_trace;
  double coherent_entry_residual = std::numeric_limits<double>::infinity();
  double coherent_entry_correction = std::numeric_limits<double>::infinity();
  {
    dg::hdg::runtime::cpu::detail::CpuNumericsCore reference;
    dg::hdg::runtime::StateCoordinates coordinates{};
    if (!reference.InitializeBorrowedEuler(reference_request)) return false;
    const auto initial_snapshot = reference.Snapshot(coordinates);
    if (initial_snapshot.trace.size == 0 || !initial_snapshot.trace.data)
      return false;
    coherent_trace.assign(
        initial_snapshot.trace.data,
        initial_snapshot.trace.data + initial_snapshot.trace.size);
    std::uint64_t generation = 0;
    if (!reference.PrepareBorrowedEuler(dt, coordinates, generation) ||
        generation == 0)
    {
      return false;
    }
    const auto& evidence = reference.evidence().last_attempt;
    coherent_entry_residual = evidence.initial_nonlinear_residual;
    coherent_entry_correction =
        evidence.initial_algebraic_correction_norm;
    if (!reference.AbortBorrowedEuler(generation, coordinates)) return false;
  }

  constexpr double inconsistent_scale = 16.0;
  std::vector<double> inconsistent_trace = coherent_trace;
  for (double& value : inconsistent_trace) value *= inconsistent_scale;
  BoundaryContext boundary{};
  EulerHdgRequest request = Request(
      mesh, 2, boundary, fields, EulerHdgTimeTreatment::rodas3);
  request.initial_true_trace = {
      inconsistent_trace.data(), inconsistent_trace.size(), 4};
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  dg::hdg::runtime::StateCoordinates coordinates{};
  if (!core.InitializeBorrowedEuler(request)) return false;
  double inconsistent_entry_residual =
      std::numeric_limits<double>::infinity();
  double inconsistent_entry_correction =
      std::numeric_limits<double>::infinity();
  const bool inconsistent_projection_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          VerifyRodasAlgebraicBlock(
              core, inconsistent_entry_residual,
              inconsistent_entry_correction);
  const auto before = core.Snapshot(coordinates);
  std::vector<double> interior_before(
      before.interior.data, before.interior.data + before.interior.size);
  std::vector<double> trace_before(
      before.trace.data, before.trace.data + before.trace.size);

  std::uint64_t generation = 123;
  const bool first_rejected =
      !core.PrepareBorrowedEuler(dt, coordinates, generation) &&
      generation == 0;
  const auto after = core.Snapshot(coordinates);
  const bool byte_equal = after.interior.size == interior_before.size() &&
      after.trace.size == trace_before.size() &&
      std::equal(interior_before.begin(), interior_before.end(),
                 after.interior.data) &&
      std::equal(trace_before.begin(), trace_before.end(), after.trace.data);
  generation = 456;
  const bool retry_rejected =
      !core.PrepareBorrowedEuler(dt, coordinates, generation) &&
      generation == 0;
  const double residual_allowance = std::max(
      request.numerical_policy.nonlinear_relative_tolerance,
      1024.0 * std::numeric_limits<double>::epsilon());
  constexpr double entry_correction_allowance = 0.25;
  const bool accepted = inconsistent_projection_audited && first_rejected &&
      retry_rejected && byte_equal &&
      std::isfinite(coherent_entry_residual) &&
      coherent_entry_residual <= residual_allowance &&
      std::isfinite(coherent_entry_correction) &&
      coherent_entry_correction <= entry_correction_allowance &&
      std::isfinite(inconsistent_entry_residual) &&
      inconsistent_entry_residual <= residual_allowance &&
      std::isfinite(inconsistent_entry_correction) &&
      inconsistent_entry_correction > entry_correction_allowance &&
      coordinates.time == 0.0 && coordinates.step == 0 &&
      coordinates.generation == 0 && coordinates.cursor == 0 &&
      coordinates.dt == 0.0;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!accepted && rank == 0)
  {
    std::cerr << "RODAS3 inconsistent-entry rollback failed first="
              << first_rejected << " retry=" << retry_rejected
              << " audited=" << inconsistent_projection_audited
              << " byte_equal=" << byte_equal << " coherent_residual="
              << coherent_entry_residual << " coherent_correction="
              << coherent_entry_correction << " inconsistent_residual="
              << inconsistent_entry_residual
              << " inconsistent_correction="
              << inconsistent_entry_correction << " residual_allowance="
              << residual_allowance << " correction_allowance="
              << entry_correction_allowance << '\n';
  }
  return All(accepted);
}

bool CheckConventionalConservationEvidence()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  // A nonzero source makes the oracle check the full conservative balance,
  // not only preservation of homogeneous periodic totals.
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  if (request.numerical_policy.conservation_relative_tolerance != 1.0e-10)
    return false;

  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request) ||
      core.evidence().tuple.formulation.view() !=
          "hdg.conventional.v1" ||
      core.evidence().tuple.stabilization.view() !=
          dg::hdg::runtime::borrowed_euler_llf_stabilization_id)
  {
    return false;
  }
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation)) return false;
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(generation, coordinates, maximum_wave_speed,
                                stable_dt))
  {
    return false;
  }

  double maximum_conservation_error =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::quiet_NaN();
  bool conservation_gate = false;
  const bool conservation_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditLastConservation(
              core, maximum_conservation_error,
              maximum_conservation_relative_error, conservation_gate);
  const auto& attempt = core.evidence().last_attempt;
  return conservation_audited && conservation_gate && attempt.committed &&
      LinearWorkAccountingExact(attempt) &&
      attempt.status == dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate &&
      attempt.work.accepted_newton_updates ==
          static_cast<std::uint64_t>(attempt.nonlinear_updates) &&
      attempt.work.complete_factor_setups == attempt.work.complete_trace_solves &&
      attempt.work.complete_trace_solves == attempt.work.complete_recoveries &&
      attempt.work.complete_trace_solves ==
          attempt.work.accepted_newton_updates + 1 &&
      attempt.final_nonlinear_residual <= attempt.initial_nonlinear_residual &&
      std::isfinite(attempt.maximum_true_relative_residual) &&
      attempt.maximum_true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      std::isfinite(maximum_conservation_error) &&
      maximum_conservation_error >= 0.0 &&
      std::isfinite(maximum_conservation_relative_error) &&
      maximum_conservation_relative_error <=
          request.numerical_policy.conservation_relative_tolerance &&
      coordinates.step == 1 && coordinates.generation == 1 &&
      coordinates.time == 0.001 && std::isfinite(maximum_wave_speed) &&
      maximum_wave_speed > 0.0 && std::isfinite(stable_dt) &&
      stable_dt > 0.0 && boundary.calls == 0;
}

bool CheckIgrExtensionLifecycle()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.stabilization = {RegularizationKind::igr, 0.2, 0.4};
  GenericEulerHdgCpuSession session;
  const auto initialized = session.Initialize(request);
  if (initialized.status != GenericEulerCpuSessionStatus::initialized) return false;
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre,
                                   mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(&mesh, &collection, 4,
                                    mfem::Ordering::byNODES);
  const std::size_t scalar_count =
      static_cast<std::size_t>(space.GetVSize() / 4);
  std::vector<double> before(static_cast<std::size_t>(space.GetVSize()), 0.0);
  std::vector<double> sigma_before(
      scalar_count, std::numeric_limits<double>::quiet_NaN());
  const bool exported_before = session.ExportCommittedFields(
      {before.data(), sigma_before.data(), scalar_count});
  // The auxiliary field is initialized to the neutral IGR state.  It is
  // therefore allowed to be identically zero before the first nonlinear
  // solve, but it must be a finite member of the same committed transaction.
  const bool finite_initial_sigma = std::all_of(
      sigma_before.begin(), sigma_before.end(),
      [](const double value) { return std::isfinite(value); });
  std::vector<double> bad_state(before.size(), 17.0);
  std::vector<double> bad_sigma(sigma_before.size(), 19.0);
  const bool rejected_bad_extent = scalar_count > 0 &&
      !session.ExportCommittedFields(
          {bad_state.data(), bad_sigma.data(), scalar_count - 1}) &&
      std::all_of(bad_state.begin(), bad_state.end(),
                  [](const double value) { return value == 17.0; }) &&
      std::all_of(bad_sigma.begin(), bad_sigma.end(),
                  [](const double value) { return value == 19.0; });
  GenericEulerCpuPreparedStep first;
  const auto prepared = session.Prepare(0.001, first);
  std::vector<double> candidate(before.size(), 0.0);
  std::vector<double> sigma_candidate(scalar_count, 0.0);
  const bool exported_candidate = prepared.status == GenericEulerCpuSessionStatus::prepared &&
      session.ExportPreparedFields(
          first, {candidate.data(), sigma_candidate.data(), scalar_count});
  const bool finite_candidate_sigma = std::all_of(
      sigma_candidate.begin(), sigma_candidate.end(),
      [](const double value) { return std::isfinite(value); });
  const bool changed = candidate != before || sigma_candidate != sigma_before;
  const auto aborted = prepared.status == GenericEulerCpuSessionStatus::prepared
      ? session.Abort(std::move(first)) : decltype(session.Abort(std::move(first))){ };
  std::vector<double> after_abort(before.size(), 0.0);
  std::vector<double> sigma_after_abort(scalar_count, 0.0);
  const bool rollback = session.ExportCommittedFields(
      {after_abort.data(), sigma_after_abort.data(), scalar_count}) &&
      after_abort == before && sigma_after_abort == sigma_before;
  GenericEulerCpuPreparedStep retry;
  const auto retried = session.Prepare(0.001, retry);
  std::vector<double> retry_candidate(before.size(), 0.0);
  std::vector<double> sigma_retry_candidate(scalar_count, 0.0);
  const bool exported_retry =
      retried.status == GenericEulerCpuSessionStatus::prepared &&
      session.ExportPreparedFields(
          retry,
          {retry_candidate.data(), sigma_retry_candidate.data(), scalar_count});
  const auto committed = retried.status == GenericEulerCpuSessionStatus::prepared
      ? session.Commit(std::move(retry)) : decltype(session.Commit(std::move(retry))){ };
  std::vector<double> after_commit(before.size(), 0.0);
  std::vector<double> sigma_after_commit(scalar_count, 0.0);
  const bool committed_identity = session.ExportCommittedFields(
      {after_commit.data(), sigma_after_commit.data(), scalar_count}) &&
      after_commit == retry_candidate &&
      sigma_after_commit == sigma_retry_candidate;
  const auto second = session.Advance(0.001);
  return exported_before && finite_initial_sigma && rejected_bad_extent &&
      exported_candidate && finite_candidate_sigma && changed &&
      aborted.status == GenericEulerCpuSessionStatus::aborted && rollback &&
      exported_retry &&
      committed.status == GenericEulerCpuSessionStatus::committed &&
      committed_identity &&
      second.status == GenericEulerCpuSessionStatus::advanced &&
      second.coordinates.step == 2 && second.coordinates.generation == 2 &&
      boundary.calls == 0;
}

bool CheckPartitionedBackwardEulerIgrLifecycle()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks < 1 || ranks > 3) return true;

  // The three-rank gate used to skip every substantive partitioned test.
  // Put the complete prepare/abort/retry/commit lifecycle on a deliberately
  // imbalanced periodic mesh there so the scalar closure must collectively
  // include both zero-local-element ranks.
  mfem::ParMesh mesh = ranks == 3
      ? MakePeriodicRankZeroOwnedBorrowedMesh()
      : MakePeriodicBorrowedMesh();
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int empty_ranks = 0;
  if (MPI_Allreduce(&local_empty, &empty_ranks, 1, MPI_INT, MPI_SUM,
                    mesh.GetComm()) != MPI_SUCCESS ||
      (ranks == 3 && empty_ranks != 2))
  {
    return false;
  }
  const std::array<double, 8> free_stream{{1.0, 1.0, 0.20, 0.20,
                                           -0.10, -0.10, 2.60, 2.60}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, free_stream, source);
  EulerHdgRequest request = Request(
      mesh, 1, boundary, fields,
      EulerHdgTimeTreatment::partitioned_backward_euler);
  request.numerical_policy.stabilization =
      {RegularizationKind::igr, 0.2, 0.4};
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      dg::hdg::generic::ContractStatus::accepted)
  {
    return false;
  }

  GenericEulerHdgCpuSession capability_session;
  const std::uint32_t assessment_capability =
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::
              prepared_embedded_assessment);
  if (capability_session.Initialize(request).status !=
          GenericEulerCpuSessionStatus::initialized ||
      dg::hdg::generic::HasHdgEulerSessionCapabilities(
          capability_session.capabilities(), assessment_capability))
  {
    return false;
  }

  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request) ||
      core.evidence().tuple.time_treatment.view() !=
          "partitioned-backward-euler.v1")
  {
    return false;
  }

  const std::size_t scalar_count = fields.initial.size() / 4;
  std::vector<double> committed_before(fields.initial.size(), 0.0);
  std::vector<double> sigma_before(scalar_count, 0.0);
  double initial_closure_residual = std::numeric_limits<double>::infinity();
  double initial_closure_tolerance = std::numeric_limits<double>::infinity();
  if (!core.ExportCommittedFields(
          {committed_before.data(), sigma_before.data(), scalar_count}) ||
      committed_before != fields.initial || core.evidence().attempts != 0 ||
      core.evidence().accepted_step_count != 0 ||
      !dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditCommittedClosure(core, initial_closure_residual,
                                initial_closure_tolerance) ||
      !std::isfinite(initial_closure_residual) ||
      !std::isfinite(initial_closure_tolerance) ||
      initial_closure_residual > initial_closure_tolerance ||
      !AdmissibleEulerCoefficients(committed_before) ||
      !std::all_of(sigma_before.begin(), sigma_before.end(),
                   [](const double value) { return std::isfinite(value); }))
  {
    return false;
  }

  // Keep this local one-step comparison inside the asymptotic lagging regime;
  // the separate fixed-final-time check below exercises practical first-order
  // trajectory convergence at the coarser 2e-4 horizon.
  constexpr double dt = 2.5e-5;
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t first_generation = 0;
  if (!core.PrepareBorrowedEuler(dt, coordinates, first_generation) ||
      first_generation == 0 || coordinates.time != 0.0 ||
      coordinates.step != 0 || coordinates.generation != 0)
  {
    return false;
  }
  std::vector<double> first_candidate(fields.initial.size(), 0.0);
  std::vector<double> first_sigma(scalar_count, 0.0);
  const bool first_observation = core.ExportPreparedFields(
      first_generation,
      {first_candidate.data(), first_sigma.data(), scalar_count});
  const bool first_coherent = first_observation &&
      SameEulerCoefficientsWithinRoundoff(
          first_candidate, committed_before) &&
      AdmissibleEulerCoefficients(first_candidate) &&
      std::all_of(first_sigma.begin(), first_sigma.end(),
                  [](const double value) { return std::isfinite(value); });
  if (!first_coherent ||
      !core.AbortBorrowedEuler(first_generation, coordinates))
  {
    return false;
  }

  std::vector<double> committed_after_abort(fields.initial.size(), 0.0);
  std::vector<double> sigma_after_abort(scalar_count, 0.0);
  if (!core.ExportCommittedFields(
          {committed_after_abort.data(), sigma_after_abort.data(),
           scalar_count}) ||
      committed_after_abort != committed_before ||
      sigma_after_abort != sigma_before ||
      !core.evidence().last_attempt.rollback_byte_equal ||
      core.evidence().last_attempt.committed || coordinates.time != 0.0 ||
      coordinates.step != 0 || coordinates.generation != 0)
  {
    return false;
  }

  std::uint64_t retry_generation = 0;
  if (!core.PrepareBorrowedEuler(dt, coordinates, retry_generation) ||
      retry_generation == 0)
  {
    return false;
  }
  std::vector<double> retry_candidate(fields.initial.size(), 0.0);
  std::vector<double> retry_sigma(scalar_count, 0.0);
  if (!core.ExportPreparedFields(
          retry_generation,
          {retry_candidate.data(), retry_sigma.data(), scalar_count}) ||
      retry_candidate != first_candidate || retry_sigma != first_sigma)
  {
    return false;
  }

  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(retry_generation, coordinates,
                                maximum_wave_speed, stable_dt))
  {
    return false;
  }
  std::vector<double> committed_after(fields.initial.size(), 0.0);
  std::vector<double> sigma_after(scalar_count, 0.0);
  if (!core.ExportCommittedFields(
          {committed_after.data(), sigma_after.data(), scalar_count}) ||
      committed_after != retry_candidate || sigma_after != retry_sigma ||
      !AdmissibleEulerCoefficients(committed_after))
  {
    return false;
  }

  double maximum_conservation_error =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::quiet_NaN();
  bool conservation_gate = false;
  const bool conservation_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditLastConservation(
              core, maximum_conservation_error,
              maximum_conservation_relative_error, conservation_gate);
  const auto& evidence = core.evidence();
  const auto& attempt = evidence.last_attempt;
  return conservation_audited && conservation_gate && attempt.committed &&
      attempt.status == dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate && attempt.nonlinear_updates == 0 &&
      attempt.final_line_search_trials == 0 &&
      attempt.work.accepted_newton_updates == 0 &&
      attempt.newton_correction_linear_iterations == 0 &&
      attempt.work.complete_trace_solves == 2 &&
      attempt.work.complete_scalar_closure_solves == 1 &&
      attempt.scalar_closure_linear_solve_attempts >= 1 &&
      attempt.work.complete_condensations ==
          attempt.work.complete_trace_solves &&
      attempt.work.complete_recoveries == attempt.work.complete_trace_solves &&
      attempt.work.complete_assemblies >= attempt.work.complete_trace_solves &&
      attempt.minimum_density > 0.0 && attempt.minimum_pressure > 0.0 &&
      std::isfinite(attempt.maximum_true_relative_residual) &&
      attempt.maximum_true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      std::isfinite(maximum_conservation_error) &&
      maximum_conservation_error >= 0.0 &&
      std::isfinite(maximum_conservation_relative_error) &&
      maximum_conservation_relative_error <=
          request.numerical_policy.conservation_relative_tolerance &&
      evidence.attempts == 2 && evidence.failed_attempts == 1 &&
      evidence.commits == 1 && evidence.accepted_step_count == 1 &&
      coordinates.time == dt && coordinates.dt == dt &&
      coordinates.step == 1 && coordinates.generation == 1 &&
      std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
      std::isfinite(stable_dt) && stable_dt > 0.0 && boundary.calls == 0;
}

bool CheckPartitionedConventionalDegeneracy()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks != 1 && ranks != 2) return true;

  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  std::array<BoundaryContext, 2> boundaries{};
  OneStepEulerState backward_euler;
  OneStepEulerState partitioned;
  constexpr double dt = 1.0e-4;
  const bool backward_euler_ok = RunEulerSteps(
      mesh, fields, boundaries[0], EulerHdgTimeTreatment::backward_euler,
      2, false, dt, backward_euler);
  const bool partitioned_ok = RunEulerSteps(
      mesh, fields, boundaries[1],
      EulerHdgTimeTreatment::partitioned_backward_euler,
      2, false, dt, partitioned);
  return backward_euler_ok && partitioned_ok &&
      backward_euler.interior_u4 == partitioned.interior_u4 &&
      backward_euler.trace_uhat4 == partitioned.trace_uhat4 &&
      boundaries[0].calls > 0 &&
      boundaries[0].calls == boundaries[1].calls;
}

bool CheckPartitionedIgrOrdersAndPhysicalBoundary()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks != 1 && ranks != 2) return true;

  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.02, 0.08, 0.09,
                                       0.00, 0.01, 2.55, 2.58}};
  const std::array<double, 8> source{{0.002, 0.003, 0.0, 0.001,
                                      0.0, 0.0, 0.004, 0.005}};
  for (int order = 1; order <= 3; ++order)
  {
    ExactBorrowedEulerFields fields(mesh, order, initial, source);
    BoundaryContext boundary{};
    OneStepEulerState state;
    if (!RunEulerSteps(
            mesh, fields, boundary,
            EulerHdgTimeTreatment::partitioned_backward_euler,
            order, true, 1.0e-4, state) ||
        boundary.calls == 0)
    {
      return false;
    }
  }
  return true;
}

bool CheckPartitionedSmallDtTrajectory()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks != 1) return true;

  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  // Reuse the established nonuniform Euler/IGR coefficients and static source
  // at p2.  The alternating local coefficients exercise a genuine spatial
  // residual while retaining the same admissible state used by the existing
  // IGR lifecycle and convergence checks.
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  std::array<BoundaryContext, 4> boundaries{};
  OneStepEulerState backward_euler;
  OneStepEulerState partitioned;
  OneStepEulerState backward_euler_half;
  OneStepEulerState partitioned_half;
  constexpr double dt = 1.0e-4;
  constexpr double half_dt = 0.5 * dt;
  const bool backward_euler_ok = RunEulerSteps(
      mesh, fields, boundaries[0], EulerHdgTimeTreatment::backward_euler,
      2, true, dt, backward_euler);
  const bool partitioned_ok = RunEulerSteps(
      mesh, fields, boundaries[1],
      EulerHdgTimeTreatment::partitioned_backward_euler,
      2, true, dt, partitioned);
  const bool backward_euler_half_ok = RunEulerSteps(
      mesh, fields, boundaries[2], EulerHdgTimeTreatment::backward_euler,
      2, true, half_dt, backward_euler_half);
  const bool partitioned_half_ok = RunEulerSteps(
      mesh, fields, boundaries[3],
      EulerHdgTimeTreatment::partitioned_backward_euler,
      2, true, half_dt, partitioned_half);
  if (!backward_euler_ok || !partitioned_ok ||
      !backward_euler_half_ok || !partitioned_half_ok)
  {
    std::cerr << "small-dt IGR run gate failed: be=" << backward_euler_ok
              << " partitioned_be=" << partitioned_ok
              << " be_half=" << backward_euler_half_ok
              << " partitioned_be_half=" << partitioned_half_ok << '\n';
    return false;
  }

  const double interior_difference = MaximumScaledDifference(
      backward_euler.interior_u4, partitioned.interior_u4);
  const double trace_difference = MaximumScaledDifference(
      backward_euler.trace_uhat4, partitioned.trace_uhat4);
  const double half_interior_difference = MaximumScaledDifference(
      backward_euler_half.interior_u4,
      partitioned_half.interior_u4);
  const double half_trace_difference = MaximumScaledDifference(
      backward_euler_half.trace_uhat4,
      partitioned_half.trace_uhat4);

  // The supplied state has a face-projected trace rather than an Euler trace
  // constraint witness.  The partitioned route projects the scalar closure
  // first while monolithic BE solves directly onto the coupled constraint
  // manifold, so this DAE startup comparison is required to contract at first
  // order in both U4 and Uhat4.  The separate fixed-final-time check below
  // establishes the temporal-order claim after startup.
  constexpr double startup_first_order_bound = 1.0e-3;
  constexpr double roundoff_floor =
      2048.0 * std::numeric_limits<double>::epsilon();
  const auto contracts = [roundoff_floor](
      const double coarse, const double fine,
      const double minimum_ratio) noexcept
  {
    if (!std::isfinite(coarse) || !std::isfinite(fine)) return false;
    if (coarse <= 8.0 * roundoff_floor)
      return fine <= 8.0 * roundoff_floor;
    return minimum_ratio * fine <= coarse + roundoff_floor;
  };
  const double interior_bound = startup_first_order_bound * dt;
  const double half_interior_bound = startup_first_order_bound * half_dt;
  const double trace_bound = startup_first_order_bound * dt;
  const double half_trace_bound = startup_first_order_bound * half_dt;
  const bool interior_bounded = interior_difference <= interior_bound;
  const bool trace_bounded = trace_difference <= trace_bound;
  const bool half_interior_bounded =
      half_interior_difference <= half_interior_bound;
  const bool half_trace_bounded = half_trace_difference <= half_trace_bound;
  const bool interior_contracts =
      contracts(interior_difference, half_interior_difference, 1.8);
  const bool trace_contracts =
      contracts(trace_difference, half_trace_difference, 1.8);
  const bool accepted = interior_bounded && trace_bounded &&
      half_interior_bounded && half_trace_bounded && interior_contracts &&
      trace_contracts &&
      std::all_of(boundaries.begin(), boundaries.end(),
                  [](const BoundaryContext& boundary)
                  { return boundary.calls == 0; });
  if (!accepted)
  {
    std::cerr << "small-dt partitioned-BE comparison failed: U4="
              << interior_difference << " U4_half="
              << half_interior_difference << " Uhat4=" << trace_difference
              << " Uhat4_half=" << half_trace_difference
              << " U4_bound=" << interior_bound
              << " U4_half_bound=" << half_interior_bound
              << " Uhat4_bound=" << trace_bound
              << " Uhat4_half_bound=" << half_trace_bound
              << " U4_ratio="
              << interior_difference /
                     std::max(half_interior_difference, roundoff_floor)
              << " Uhat4_ratio="
              << trace_difference /
                     std::max(half_trace_difference, roundoff_floor)
              << " roundoff_floor=" << roundoff_floor
              << " U4_bound_ok=" << interior_bounded
              << " U4_half_bound_ok=" << half_interior_bounded
              << " Uhat4_bound_ok=" << trace_bounded
              << " Uhat4_half_bound_ok=" << half_trace_bounded
              << " U4_contracts=" << interior_contracts
              << " Uhat4_contracts=" << trace_contracts << '\n';
  }
  return accepted;
}

bool CheckPartitionedFixedTimeFirstOrder()
{
  int ranks = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return false;
  if (ranks != 1) return true;

  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  std::array<BoundaryContext, 3> boundaries{};
  OneStepEulerState coarse;
  OneStepEulerState medium;
  OneStepEulerState fine;
  constexpr double final_time = 2.0e-4;
  const bool coarse_ok = RunEulerSteps(
      mesh, fields, boundaries[0],
      EulerHdgTimeTreatment::partitioned_backward_euler,
      2, true, final_time, coarse, 1);
  const bool medium_ok = RunEulerSteps(
      mesh, fields, boundaries[1],
      EulerHdgTimeTreatment::partitioned_backward_euler,
      2, true, 0.5 * final_time, medium, 2);
  const bool fine_ok = RunEulerSteps(
      mesh, fields, boundaries[2],
      EulerHdgTimeTreatment::partitioned_backward_euler,
      2, true, 0.25 * final_time, fine, 4);
  if (!coarse_ok || !medium_ok || !fine_ok)
  {
    std::cerr << "fixed-time partitioned-BE run gate failed: coarse="
              << coarse_ok << " medium=" << medium_ok
              << " fine=" << fine_ok << '\n';
    return false;
  }

  const double u4_coarse = MaximumScaledDifference(
      coarse.interior_u4, medium.interior_u4);
  const double u4_fine = MaximumScaledDifference(
      medium.interior_u4, fine.interior_u4);
  const double uhat_coarse = MaximumScaledDifference(
      coarse.trace_uhat4, medium.trace_uhat4);
  const double uhat_fine = MaximumScaledDifference(
      medium.trace_uhat4, fine.trace_uhat4);
  constexpr double roundoff_floor =
      4096.0 * std::numeric_limits<double>::epsilon();
  const auto first_order = [roundoff_floor](
      const double coarse_difference, const double fine_difference) noexcept
  {
    return std::isfinite(coarse_difference) &&
        std::isfinite(fine_difference) &&
        coarse_difference > 32.0 * roundoff_floor &&
        fine_difference > roundoff_floor &&
        1.7 * fine_difference <= coarse_difference + roundoff_floor;
  };
  const bool accepted = first_order(u4_coarse, u4_fine) &&
      first_order(uhat_coarse, uhat_fine) &&
      std::all_of(boundaries.begin(), boundaries.end(),
                  [](const BoundaryContext& boundary)
                  { return boundary.calls == 0; });
  if (!accepted)
  {
    std::cerr << "fixed-time partitioned-BE order failed: U4_coarse="
              << u4_coarse << " U4_fine=" << u4_fine
              << " U4_ratio="
              << u4_coarse / std::max(u4_fine, roundoff_floor)
              << " Uhat4_coarse=" << uhat_coarse
              << " Uhat4_fine=" << uhat_fine
              << " Uhat4_ratio="
              << uhat_coarse / std::max(uhat_fine, roundoff_floor)
              << " roundoff_floor=" << roundoff_floor << '\n';
  }
  return accepted;
}

bool CheckIgrExtensionConvergenceEvidence()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  // A nonzero static source exercises the conservative source balance rather
  // than merely accepting the homogeneous periodic special case.
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.stabilization =
      {RegularizationKind::igr, 0.2, 0.4};
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return false;
  if (core.evidence().tuple.formulation.view() !=
          "hdg.conventional.v1" ||
      core.evidence().tuple.stabilization.view() !=
          "igr.scalar.traced.alpha-beta.v1")
  {
    return false;
  }
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation)) return false;
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(generation, coordinates, maximum_wave_speed,
                                stable_dt)) return false;
  double maximum_conservation_error =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::quiet_NaN();
  bool conservation_gate = false;
  const bool conservation_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditLastConservation(
              core, maximum_conservation_error,
              maximum_conservation_relative_error, conservation_gate);
  const auto& attempt = core.evidence().last_attempt;
  return conservation_audited && conservation_gate && attempt.committed &&
      LinearWorkAccountingExact(attempt) &&
      attempt.nonlinear_updates > 1 &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate &&
      attempt.work.accepted_newton_updates ==
          static_cast<std::uint64_t>(attempt.nonlinear_updates) &&
      attempt.work.complete_factor_setups == attempt.work.complete_trace_solves &&
      attempt.work.complete_trace_solves == attempt.work.complete_recoveries &&
      attempt.work.complete_trace_solves ==
          attempt.work.accepted_newton_updates + 1 &&
      attempt.final_nonlinear_residual <= attempt.initial_nonlinear_residual &&
      std::isfinite(attempt.maximum_true_relative_residual) &&
      attempt.maximum_true_relative_residual <= 1.0e-10 &&
      std::isfinite(maximum_conservation_error) &&
      std::isfinite(maximum_conservation_relative_error) &&
      maximum_conservation_relative_error <= 1.0e-10 &&
      boundary.calls == 0;
}

bool CheckIgrExtensionNumericalPolicy()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.stabilization =
      {RegularizationKind::igr, 0.2, 0.4};
  // Deliberately distinct policy controls ensure the IGR assembly,
  // Krylov stopping, true-residual acceptance, and source conservation gates
  // do not silently borrow the nonlinear tolerance or a helper default.
  request.numerical_policy.igr_assembly_quadrature_orders = {10, 8};
  request.numerical_policy.maximum_linear_iterations = 31;
  request.numerical_policy.newton_correction_linear_relative_tolerance =
      1.0e-12;
  request.numerical_policy.final_coherent_linear_relative_tolerance =
      1.0e-12;
  request.numerical_policy.
      true_residual_tightening_linear_relative_tolerance = 1.0e-12;
  request.numerical_policy.true_residual_relative_tolerance = 1.0e-9;
  request.numerical_policy.conservation_relative_tolerance = 1.0e-8;
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return false;
  const auto requested_orders =
      request.numerical_policy.igr_assembly_quadrature_orders;
  const bool selected_quadrature =
      core.evidence().assembly_quadrature_order == requested_orders.volume_order;
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation)) return false;
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(generation, coordinates, maximum_wave_speed,
                                stable_dt)) return false;
  double maximum_conservation_error =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::quiet_NaN();
  bool conservation_gate = false;
  const bool conservation_audited =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          AuditLastConservation(
              core, maximum_conservation_error,
              maximum_conservation_relative_error, conservation_gate);
  const auto& attempt = core.evidence().last_attempt;
  return selected_quadrature && conservation_audited && conservation_gate &&
      LinearWorkAccountingExact(attempt) &&
      std::isfinite(maximum_conservation_error) && attempt.committed &&
      attempt.linear_iterations > 0 &&
      attempt.linear_iterations <=
          request.numerical_policy.maximum_linear_iterations *
              static_cast<int>(attempt.work.complete_trace_solves) &&
      attempt.maximum_true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      maximum_conservation_relative_error <=
          request.numerical_policy.conservation_relative_tolerance &&
      maximum_conservation_relative_error >= 0.0 &&
      boundary.calls == 0;
}

bool CheckIgrExtensionZeroLocal()
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks == 1) return true;
  mfem::ParMesh mesh = MakePeriodicRankZeroOwnedBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.1, 0.1,
                                       0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.stabilization = {RegularizationKind::igr, 0.2, 0.4};
  GenericEulerHdgCpuSession session;
  const auto initialized = session.Initialize(request);
  const auto advanced = initialized.status == GenericEulerCpuSessionStatus::initialized
      ? session.Advance(0.001) : GenericEulerCpuSessionResult{};
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int any_empty = 0;
  const int reduction = MPI_Allreduce(&local_empty, &any_empty, 1, MPI_INT,
                                      MPI_MAX, mesh.GetComm());
  return initialized.status == GenericEulerCpuSessionStatus::initialized &&
      advanced.status == GenericEulerCpuSessionStatus::advanced &&
      reduction == MPI_SUCCESS && any_empty == 1 &&
      boundary.calls == 0;
}

bool CheckIgrExtensionInitializationDeterminism()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.stabilization = {RegularizationKind::igr, 0.2, 0.4};
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre,
                                   mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(&mesh, &collection, 4,
                                    mfem::Ordering::byNODES);
  std::vector<double> first(static_cast<std::size_t>(space.GetVSize()), 0.0);
  std::vector<double> second(static_cast<std::size_t>(space.GetVSize()), 0.0);
  bool first_ok = false;
  {
    GenericEulerHdgCpuSession session;
    const auto initialized = session.Initialize(request);
    first_ok = initialized.status == GenericEulerCpuSessionStatus::initialized &&
        session.ExportCommittedInterior({first.data(), first.size()});
  }
  bool second_ok = false;
  {
    GenericEulerHdgCpuSession session;
    const auto initialized = session.Initialize(request);
    second_ok = initialized.status == GenericEulerCpuSessionStatus::initialized &&
        session.ExportCommittedInterior({second.data(), second.size()});
  }
  return first_ok && second_ok && first == second && boundary.calls == 0;
}

bool CheckUnifiedIgrAssemblerJacobianForTests()
{
  constexpr std::array<double, 8> initial{{
      1.10, 1.12, 0.08, 0.09, -0.04, -0.03, 2.90, 2.95}};
  constexpr std::array<double, 8> source{};
  constexpr double relative_tolerance = 5.0e-5;
  bool local = true;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int order = 1; order <= 3; ++order)
    for (int boundary_case = 0; boundary_case < 2; ++boundary_case)
    {
      const bool physical_boundary = boundary_case != 0;
      mfem::ParMesh mesh = physical_boundary
          ? MakeBorrowedMesh() : MakePeriodicBorrowedMesh();
      BoundaryContext boundary{};
      ExactBorrowedEulerFields fields(mesh, order, initial, source);
      EulerHdgRequest request = Request(mesh, order, boundary, fields);
      request.boundary = {
          &boundary, FixedExteriorForAssemblerJacobianForTests,
          {nullptr, 0, 0, 0.0}};
      request.numerical_policy.stabilization =
          {RegularizationKind::igr, 0.2, 0.8};
      dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
      std::array<double, 6> relative_errors{};
      const bool initialized = core.InitializeBorrowedEuler(request);
      const bool verified = initialized &&
          dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
              VerifyAssemblerJacobian(
                  core, relative_tolerance, relative_errors);
      int global_boundary_calls = 0;
      const bool calls_reduced = MPI_Allreduce(
          &boundary.calls, &global_boundary_calls, 1, MPI_INT, MPI_SUM,
          mesh.GetComm()) == MPI_SUCCESS;
      const bool boundary_exercised = physical_boundary
          ? global_boundary_calls > 0 : global_boundary_calls == 0;
      if ((!verified || !calls_reduced || !boundary_exercised) && rank == 0)
      {
        std::cerr << "unified IGR assembler Jacobian check failed order="
                  << order << " physical_boundary=" << physical_boundary
                  << " initialized=" << initialized
                  << " boundary_calls=" << global_boundary_calls
                  << " errors=";
        for (const double error : relative_errors)
          std::cerr << ' ' << error;
        std::cerr << '\n';
      }
      local = local && verified && calls_reduced && boundary_exercised;
    }
  return All(local);
}

bool CheckRodas3PhysicalBoundaryJacobianForTests()
{
  constexpr std::array<double, 8> initial{{
      1.10, 1.12, 0.08, 0.09, -0.04, -0.03, 2.90, 2.95}};
  constexpr std::array<double, 8> source{};
  constexpr double relative_tolerance = 1.0e-4;
  bool local = true;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int regularized_case = 0; regularized_case < 2; ++regularized_case)
    for (int boundary_case = 0; boundary_case < 2; ++boundary_case)
    {
      mfem::ParMesh mesh = MakeBorrowedMesh();
      BoundaryContext boundary{};
      ExactBorrowedEulerFields fields(mesh, 2, initial, source);
      EulerHdgRequest request = Request(
          mesh, 2, boundary, fields, EulerHdgTimeTreatment::rodas3);
      request.boundary = {
          &boundary,
          boundary_case == 0
              ? NonperiodicExterior
              : SmoothAutonomousExteriorForRodasJacobianForTests,
          {nullptr, 0, 0, 0.0}};
      if (regularized_case != 0)
        request.numerical_policy.stabilization =
            {RegularizationKind::igr, 0.2, 0.8};

      dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
      std::array<double, 6> relative_errors{};
      const bool initialized = core.InitializeBorrowedEuler(request);
      const bool verified = initialized &&
          dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
              VerifyAssemblerJacobian(
                  core, relative_tolerance, relative_errors);
      int global_boundary_calls = 0;
      const bool calls_reduced = MPI_Allreduce(
          &boundary.calls, &global_boundary_calls, 1, MPI_INT, MPI_SUM,
          mesh.GetComm()) == MPI_SUCCESS;
      if ((!verified || !calls_reduced || global_boundary_calls <= 0) &&
          rank == 0)
      {
        std::cerr << "RODAS3 physical-boundary Jacobian check failed igr="
                  << regularized_case << " boundary_case=" << boundary_case
                  << " initialized=" << initialized << " boundary_calls="
                  << global_boundary_calls << " errors=";
        for (const double error : relative_errors) std::cerr << ' ' << error;
        std::cerr << '\n';
      }
      local = local && verified && calls_reduced && global_boundary_calls > 0;
    }
  return All(local);
}

bool CheckRodas3AlgebraicBlockSolvabilityForTests()
{
  constexpr std::array<double, 8> initial{{
      1.10, 1.12, 0.08, 0.09, -0.04, -0.03, 2.90, 2.95}};
  constexpr std::array<double, 8> source{};
  bool local = true;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int fixture = 0; fixture < 3; ++fixture)
  {
    const bool physical_boundary = fixture != 0;
    const bool igr_auxiliary_neumann = fixture == 2;
    mfem::ParMesh mesh = physical_boundary
        ? MakeBorrowedMesh() : MakePeriodicBorrowedMesh();
    BoundaryContext boundary{};
    ExactBorrowedEulerFields fields(mesh, 1, initial, source);
    EulerHdgRequest request = Request(
        mesh, 1, boundary, fields, EulerHdgTimeTreatment::rodas3);
    if (igr_auxiliary_neumann)
      request.numerical_policy.stabilization =
          {RegularizationKind::igr, 0.2, 0.8};

    dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
    double algebraic_residual = std::numeric_limits<double>::infinity();
    double correction_norm = std::numeric_limits<double>::infinity();
    const bool initialized = core.InitializeBorrowedEuler(request);
    const bool verified = initialized &&
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            VerifyRodasAlgebraicBlock(
                core, algebraic_residual, correction_norm);
    int global_boundary_calls = 0;
    const bool calls_reduced = MPI_Allreduce(
        &boundary.calls, &global_boundary_calls, 1, MPI_INT, MPI_SUM,
        mesh.GetComm()) == MPI_SUCCESS;
    const bool boundary_exercised = physical_boundary
        ? global_boundary_calls > 0 : global_boundary_calls == 0;
    if ((!verified || !calls_reduced || !boundary_exercised) && rank == 0)
    {
      std::cerr << "RODAS3 algebraic-block solvability check failed fixture="
                << fixture << " initialized=" << initialized
                << " boundary_calls=" << global_boundary_calls
                << " residual=" << algebraic_residual
                << " correction=" << correction_norm << '\n';
    }
    local = local && verified && calls_reduced && boundary_exercised &&
        std::isfinite(algebraic_residual) && std::isfinite(correction_norm) &&
        correction_norm > 0.0;
  }
  return All(local);
}

bool CheckBorrowedLlfNoFloorAndFrozenRefresh()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.10, 0.10,
                                       0.00, 0.00, 2.55, 2.55}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request) ||
      core.evidence().tuple.stabilization.view() !=
          dg::hdg::runtime::borrowed_euler_llf_stabilization_id)
  {
    return false;
  }

  double base_stabilization = std::numeric_limits<double>::infinity();
  double refreshed_stabilization = std::numeric_limits<double>::infinity();
  double frozen_jacobian_error = std::numeric_limits<double>::infinity();
  double refreshed_candidate_difference =
      std::numeric_limits<double>::infinity();
  const bool verified =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          VerifyLlfPolicy(
              core, base_stabilization, refreshed_stabilization,
              frozen_jacobian_error, refreshed_candidate_difference);
  return verified && base_stabilization > 0.0 &&
      base_stabilization < dg::hdg::runtime::cpu::cpu_fixed_tau &&
      std::abs(refreshed_stabilization - base_stabilization) > 1.0e-6 &&
      frozen_jacobian_error <= 1.0e-7 &&
      refreshed_candidate_difference >= 1.0e-4 && boundary.calls == 0;
}

bool CheckZeroUpdateFreeStreamFinalReconstruction()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> free_stream{{1.0, 1.0, 0.20, 0.20,
                                           -0.10, -0.10, 2.60, 2.60}};
  const std::array<double, 8> source{};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, free_stream, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) { return false; }

  std::vector<double> before(fields.initial.size(), 0.0);
  std::vector<double> candidate(fields.initial.size(), 0.0);
  std::vector<double> committed(fields.initial.size(), 0.0);
  // Arm the private exact-zero residual seam after initializing a genuine
  // admissible free stream.  The attempt must validate its final linearized
  // reconstruction without applying any update to these bytes.
  if (!core.ExportCommittedInterior(before.data(), before.size()) ||
      !dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          ArmExactZeroInitialResidual(core))
  {
    return false;
  }

  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation) ||
      generation == 0 ||
      !core.ExportPreparedInterior(
          generation, candidate.data(), candidate.size()) ||
      candidate != before)
  {
    return false;
  }
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(
          generation, coordinates, maximum_wave_speed, stable_dt) ||
      !core.ExportCommittedInterior(committed.data(), committed.size()))
  {
    return false;
  }

  const auto& attempt = core.evidence().last_attempt;
  return committed == before && attempt.committed &&
      LinearWorkAccountingExact(attempt) &&
      attempt.newton_correction_linear_iterations == 0 &&
      attempt.true_residual_tightenings == 0 &&
      attempt.status == dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
      attempt.nonlinear_updates == 0 && attempt.linear_iterations >= 0 &&
      attempt.initial_nonlinear_residual == 0.0 &&
      attempt.final_nonlinear_residual == 0.0 &&
      attempt.committed_interior_before == attempt.committed_interior_after &&
      attempt.committed_trace_before == attempt.committed_trace_after &&
      attempt.work.complete_assemblies == 1 &&
      attempt.work.trial_reassemblies == 0 &&
      attempt.work.accepted_newton_updates == 0 &&
      attempt.work.complete_condensations == 1 &&
      attempt.work.complete_factor_setups == 1 &&
      attempt.work.complete_trace_solves == 1 &&
      attempt.work.complete_recoveries == 1 &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate &&
      coordinates.step == 1 && coordinates.generation == 1 &&
      coordinates.time == 0.001 && std::isfinite(maximum_wave_speed) &&
      std::isfinite(stable_dt) && stable_dt > 0.0 && boundary.calls == 0;
}

bool CheckDynamicTauHighWaveTwoSteps()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  // |u . n| + c exceeds the standalone fixed cap on x-normal faces.  The
  // borrowed path must instead freeze its local face spectral-radius bound.
  const std::array<double, 8> initial{{1.0, 1.0, 5.0, 5.0,
                                        0.0, 0.0, 15.0, 15.0}};
  const std::array<double, 8> source{{0.03, 0.04, 0.01, 0.02,
                                       0.00, 0.00, 0.05, 0.06}};
  BoundaryContext context{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, context, fields);
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
  GenericEulerHdgCpuSession session;
  return session.Initialize(request).status ==
             GenericEulerCpuSessionStatus::initialized &&
      session.Advance(0.001).status == GenericEulerCpuSessionStatus::advanced &&
      session.Advance(0.001).status == GenericEulerCpuSessionStatus::advanced &&
      session.coordinates().step == 2 && session.coordinates().generation == 2 &&
      session.coordinates().time == 0.002 && context.calls == 0 &&
      SameBorrowedMesh(mesh, before, request.communicator);
}

bool CheckClosedBoundaryPicardAssembly()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{0.01, 0.02, 0.0, 0.0,
                                       0.0, 0.0, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
  dg::hdg::runtime::cpu::detail::ResetCpuGenericBoundaryPicardTestObservation();
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.stabilization =
      {RegularizationKind::igr, 0.2, 0.4};
  const bool accepted = ExecuteGenericEulerHdgCpu(request);
  // This executes the closed nonperiodic IGR-augmented solve.  The production
  // hook observes the conventional Euler physical trace rows in the same
  // system: one callback value per face quadrature point, C = 0, and a
  // nonzero trace-mass D contribution.  The auxiliary trace row retains its
  // homogeneous Neumann closure.
  return accepted && boundary.calls > 0 &&
      dg::hdg::runtime::cpu::detail::
          CpuGenericBoundaryPicardAssemblyObservedForTests() &&
      SameBorrowedMesh(mesh, before, mesh.GetComm());
}

bool CheckZeroOwnedElementEngine()
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks == 1) return true;
  mfem::ParMesh mesh = MakeRankZeroOwnedBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  GenericEulerHdgCpuSession session;
  const bool accepted =
      session.Initialize(request).status ==
          GenericEulerCpuSessionStatus::initialized &&
      session.Advance(0.019).status == GenericEulerCpuSessionStatus::advanced &&
      session.Advance(0.041).status == GenericEulerCpuSessionStatus::advanced;
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int any_empty = 0;
  if (MPI_Allreduce(&local_empty, &any_empty, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return accepted && any_empty != 0 &&
      SameBorrowedMesh(mesh, before, mesh.GetComm());
}

bool CheckRodas3CommittedCheckpointRestart()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  constexpr std::array<double, 8> initial{{
      1.00, 1.05, 0.10, 0.12, 0.00, 0.01, 2.55, 2.62}};
  constexpr std::array<double, 8> source{{
      0.01, 0.02, 0.00, 0.01, 0.00, 0.00, 0.03, 0.04}};
  constexpr double dt = 1.0e-4;
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(
      mesh, 1, boundary, fields, EulerHdgTimeTreatment::rodas3);
  request.numerical_policy.stabilization =
      {RegularizationKind::igr, 0.2, 0.4};

  GenericEulerCpuCheckpoint uninterrupted_final;
  GenericEulerCpuSessionResult uninterrupted_result{};
  {
    GenericEulerHdgCpuSession uninterrupted;
    bool ready = uninterrupted.Initialize(request).status ==
        GenericEulerCpuSessionStatus::initialized;
    for (int step = 0; step < 3 && ready; ++step)
    {
      uninterrupted_result = uninterrupted.Advance(dt);
      ready = uninterrupted_result.status ==
          GenericEulerCpuSessionStatus::advanced;
    }
    if (!ready || !uninterrupted.CaptureCheckpoint(uninterrupted_final))
    {
      return false;
    }
  }

  GenericEulerCpuCheckpoint split_checkpoint;
  dg::hdg::generic::HdgEulerCommittedDiagnostics split_diagnostics{};
  GenericEulerCpuSessionCoordinates split_coordinates{};
  {
    GenericEulerHdgCpuSession split;
    if (split.Initialize(request).status !=
            GenericEulerCpuSessionStatus::initialized ||
        split.Advance(dt).status != GenericEulerCpuSessionStatus::advanced)
    {
      return false;
    }

    // A live candidate is not a restart boundary.  Explicitly aborting the
    // candidate restores the same committed bytes, after which capture is
    // admitted and the resumed trajectory must remain exact.
    GenericEulerCpuPreparedStep rejected_candidate;
    GenericEulerCpuCheckpoint inadmissible_checkpoint;
    const bool prepared = split.Prepare(dt, rejected_candidate).status ==
        GenericEulerCpuSessionStatus::prepared;
    const bool pending_capture_rejected = prepared &&
        !split.CaptureCheckpoint(inadmissible_checkpoint) &&
        !inadmissible_checkpoint.valid();
    const bool aborted = prepared &&
        split.Abort(std::move(rejected_candidate)).status ==
            GenericEulerCpuSessionStatus::aborted;
    if (!pending_capture_rejected || !aborted ||
        !split.CaptureCheckpoint(split_checkpoint))
    {
      return false;
    }
    split_coordinates = split.coordinates();
    split_diagnostics = split.diagnostics();
  }

  GenericEulerCpuCheckpoint resume_checkpoint(std::move(split_checkpoint));
  if (split_checkpoint.valid() || !resume_checkpoint.valid() ||
      !SameSessionCoordinates(
          resume_checkpoint.coordinates(), split_coordinates))
  {
    return false;
  }

  // A numerically different solver policy is a fresh compatible mesh but not
  // a compatible restart configuration.  Rejection must leave generation
  // zero untouched.
  {
    EulerHdgRequest mismatched_request = request;
    mismatched_request.numerical_policy.maximum_linear_iterations = 37;
    GenericEulerHdgCpuSession mismatched;
    const bool initialized = mismatched.Initialize(mismatched_request).status ==
        GenericEulerCpuSessionStatus::initialized;
    const auto rejected = initialized
        ? mismatched.Restore(resume_checkpoint)
        : GenericEulerCpuSessionResult{};
    const auto coordinates = mismatched.coordinates();
    if (!initialized ||
        rejected.status != GenericEulerCpuSessionStatus::rejected ||
        coordinates.time != 0.0 || coordinates.step != 0 ||
        coordinates.generation != 0 || coordinates.dt != 0.0)
    {
      return false;
    }
  }

  GenericEulerCpuCheckpoint restarted_final;
  GenericEulerCpuSessionResult restarted_result{};
  {
    GenericEulerHdgCpuSession restarted;
    if (restarted.Initialize(request).status !=
        GenericEulerCpuSessionStatus::initialized)
    {
      return false;
    }
    const auto restored = restarted.Restore(resume_checkpoint);
    const auto restored_diagnostics = restarted.diagnostics();
    const bool diagnostics_exact = restored_diagnostics.available &&
        restored_diagnostics.wave_generation ==
            split_diagnostics.wave_generation &&
        dg::hdg::generic::DoubleBits(
            restored_diagnostics.coordinates.time) ==
            dg::hdg::generic::DoubleBits(
                split_diagnostics.coordinates.time) &&
        dg::hdg::generic::DoubleBits(
            restored_diagnostics.coordinates.dt) ==
            dg::hdg::generic::DoubleBits(
                split_diagnostics.coordinates.dt) &&
        restored_diagnostics.coordinates.step ==
            split_diagnostics.coordinates.step &&
        restored_diagnostics.coordinates.generation ==
            split_diagnostics.coordinates.generation &&
        dg::hdg::generic::DoubleBits(
            restored_diagnostics.maximum_wave_speed) ==
            dg::hdg::generic::DoubleBits(
                split_diagnostics.maximum_wave_speed) &&
        dg::hdg::generic::DoubleBits(
            restored_diagnostics.backend_stable_dt) ==
            dg::hdg::generic::DoubleBits(
                split_diagnostics.backend_stable_dt);
    if (restored.status != GenericEulerCpuSessionStatus::restored ||
        !SameSessionCoordinates(restored.coordinates, split_coordinates) ||
        !diagnostics_exact)
    {
      return false;
    }
    for (int step = 1; step < 3; ++step)
    {
      restarted_result = restarted.Advance(dt);
      if (restarted_result.status != GenericEulerCpuSessionStatus::advanced)
      {
        return false;
      }
    }
    if (!restarted.CaptureCheckpoint(restarted_final)) return false;
  }

  const bool exact_state =
      dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
          SameCheckpointState(uninterrupted_final, restarted_final);
  return All(exact_state && uninterrupted_final.valid() &&
      restarted_final.valid() &&
      SameSessionCoordinates(uninterrupted_result.coordinates,
                             restarted_result.coordinates) &&
      SameSessionCoordinates(uninterrupted_final.coordinates(),
                             restarted_final.coordinates()) &&
      uninterrupted_result.coordinates.step == 3 &&
      uninterrupted_result.coordinates.generation == 3 &&
      dg::hdg::generic::DoubleBits(
          uninterrupted_result.coordinates.time) ==
          dg::hdg::generic::DoubleBits(3.0 * dt));
}

bool CheckSubcommunicatorIsolation()
{
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm subcommunicator = MPI_COMM_NULL;
  if (MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank,
                     &subcommunicator) != MPI_SUCCESS)
  {
    return false;
  }
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  bool local = false;
  {
    mfem::ParMesh mesh = MakeBorrowedMesh(subcommunicator);
    BoundaryContext boundary{};
    ExactBorrowedEulerFields fields(mesh, 1, initial, source);
    EulerHdgRequest request = Request(mesh, 1, boundary, fields);
    const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
    GenericEulerHdgCpuSession session;
    GenericEulerCpuPreparedStep prepared;
    const bool initialized = session.Initialize(request).status ==
        GenericEulerCpuSessionStatus::initialized;
    const bool prepared_step = initialized &&
        session.Prepare(0.013, prepared).status ==
            GenericEulerCpuSessionStatus::prepared;
    const bool even_group = (world_rank % 2) == 0;
    const bool resolved_prepared = prepared_step &&
        (even_group
            ? session.Commit(std::move(prepared)).status ==
                GenericEulerCpuSessionStatus::committed
            : session.Abort(std::move(prepared)).status ==
                GenericEulerCpuSessionStatus::aborted);
    const bool advanced = resolved_prepared &&
        (even_group
            ? session.Advance(0.047).status ==
                GenericEulerCpuSessionStatus::advanced
            : session.Advance(0.013).status ==
                    GenericEulerCpuSessionStatus::advanced &&
                session.Advance(0.047).status ==
                    GenericEulerCpuSessionStatus::advanced);
    local = initialized && prepared_step && resolved_prepared && advanced &&
        boundary.calls > 0 && SameBorrowedMesh(mesh, before, subcommunicator);
    local = All(local, subcommunicator);
  }
  const bool freed = MPI_Comm_free(&subcommunicator) == MPI_SUCCESS;
  return All(local && freed);
}

bool CheckBorrowedSessionDoesNotUseWorldCollectives(
    const EulerHdgTimeTreatment time_treatment =
        EulerHdgTimeTreatment::backward_euler,
    const bool igr = false)
{
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size == 1) { return true; }

  MPI_Comm active_communicator = MPI_COMM_NULL;
  const int active_color = (world_rank % 2) == 0 ? 0 : MPI_UNDEFINED;
  if (MPI_Comm_split(MPI_COMM_WORLD, active_color, world_rank,
                     &active_communicator) != MPI_SUCCESS)
  {
    return false;
  }

  bool local = true;
  if (active_communicator != MPI_COMM_NULL)
  {
    const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                          0.0, 0.0, 2.5, 2.5}};
    const std::array<double, 8> source{{}};
    mfem::ParMesh mesh = MakeBorrowedMesh(active_communicator);
    BoundaryContext boundary{};
    ExactBorrowedEulerFields fields(mesh, 1, initial, source);
    EulerHdgRequest request = Request(
        mesh, 1, boundary, fields, time_treatment);
    if (igr)
      request.numerical_policy.stabilization =
          {RegularizationKind::igr, 0.2, 0.4};
    const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
    GenericEulerHdgCpuSession session;
    GenericEulerCpuPreparedStep prepared;
    local = session.Initialize(request).status ==
            GenericEulerCpuSessionStatus::initialized &&
        session.Prepare(0.013, prepared).status ==
            GenericEulerCpuSessionStatus::prepared &&
        session.Commit(std::move(prepared)).status ==
            GenericEulerCpuSessionStatus::committed &&
        session.Advance(0.047).status ==
            GenericEulerCpuSessionStatus::advanced &&
        boundary.calls > 0 && SameBorrowedMesh(mesh, before, active_communicator);
    if (local && igr)
    {
      const std::size_t scalar_count = fields.initial.size() / 4;
      std::vector<double> committed(fields.initial.size(), 0.0);
      std::vector<double> sigma(scalar_count,
                                std::numeric_limits<double>::quiet_NaN());
      local = session.ExportCommittedFields(
                  {committed.data(), sigma.data(), scalar_count}) &&
          AdmissibleEulerCoefficients(committed) &&
          std::all_of(sigma.begin(), sigma.end(),
                      [](const double value) { return std::isfinite(value); });
    }
    local = All(local, active_communicator);
  }

  const bool freed = active_communicator == MPI_COMM_NULL ||
      MPI_Comm_free(&active_communicator) == MPI_SUCCESS;
  // Passive world ranks join only after the active subgroup has completed all
  // borrowed-session operations, including its unequal number of commits.
  return All(local && freed);
}

bool CheckCollectiveMismatchAndAtomicFailure()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  BoundaryContext boundary{};
  const int order = rank == 0
      ? dg::hdg::generic::maximum_polynomial_order + 1 : 2;
  ExactBorrowedEulerFields fields(mesh, order, initial, source);
  EulerHdgRequest request = Request(mesh, order, boundary, fields);
  const bool rejected = !ExecuteGenericEulerHdgCpu(request);
  const bool restart_rejected = !RestartGenericEulerHdgCpu(request);
  return All(rejected && restart_rejected);
}

bool CheckCollectiveInitialTraceAdmission()
{
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  // This regression specifically exercises the two-rank split where a
  // component-aligned external trace has the wrong local extent on one rank.
  if (ranks != 2) { return true; }

  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);

  mfem::DG_Interface_FECollection trace_collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, 4, mfem::Ordering::byNODES);
  const std::size_t local_true_size = static_cast<std::size_t>(
      trace_space.GetTrueVSize());
  std::vector<double> supplied(local_true_size + (rank == 0 ? 4U : 0U), 0.0);
  request.initial_true_trace = {
      supplied.empty() ? nullptr : supplied.data(), supplied.size(), 4};

  GenericEulerHdgCpuSession session;
  const auto initialization = session.Initialize(request);
  const auto coordinates = session.coordinates();
  const bool rejected = initialization.status ==
          GenericEulerCpuSessionStatus::rejected &&
      coordinates.step == 0 && coordinates.generation == 0 &&
      coordinates.time == 0.0 &&
      session.Advance(0.013).status == GenericEulerCpuSessionStatus::rejected &&
      SameBorrowedMesh(mesh, before, request.communicator);
  return All(rejected);
}

bool CheckSuppliedInitialTraceWithZeroLocalExtent()
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 2) { return true; }

  mfem::ParMesh mesh = MakeRankZeroOwnedBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);

  mfem::DG_Interface_FECollection trace_collection(
      1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, 4, mfem::Ordering::byNODES);
  const int local_true_size = trace_space.GetTrueVSize();
  int has_empty_true_trace = local_true_size == 0 ? 1 : 0;
  if (MPI_Allreduce(MPI_IN_PLACE, &has_empty_true_trace, 1, MPI_INT, MPI_MAX,
                    mesh.GetComm()) != MPI_SUCCESS ||
      has_empty_true_trace == 0)
  {
    return false;
  }

  // An external trace must be the same trace restriction used by the derived
  // compatibility path.  A finite all-zero Uhat is shape-valid, but it is not
  // admissible for this uniform positive Euler state.
  mfem::L2_FECollection interior_collection(
      1, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace interior_space(
      &mesh, &interior_collection, 4, mfem::Ordering::byNODES);
  mfem::ParGridFunction local_interior(&interior_space);
  if (local_interior.Size() != static_cast<int>(fields.initial.size()))
  {
    return false;
  }
  for (int index = 0; index < local_interior.Size(); ++index)
  {
    local_interior[index] = fields.initial[static_cast<std::size_t>(index)];
  }
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::Vector projected_true_trace(local_true_size);
  local_trace = 0.0;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    mfem::FaceElementTransformations* const transformations =
        mesh.GetFaceElementTransformations(face);
    const mfem::FiniteElement* const trace_element =
        trace_space.GetFaceElement(face);
    if (!transformations || !trace_element || transformations->Elem1No < 0 ||
        transformations->Elem1No >= mesh.GetNE())
    {
      return false;
    }
    const mfem::FiniteElement* const interior_element =
        interior_space.GetFE(transformations->Elem1No);
    mfem::Array<int> interior_vdofs;
    mfem::Array<int> trace_vdofs;
    interior_space.GetElementVDofs(transformations->Elem1No, interior_vdofs);
    trace_space.GetFaceVDofs(face, trace_vdofs);
    if (!interior_element ||
        interior_vdofs.Size() != 4 * interior_element->GetDof() ||
        trace_vdofs.Size() != 4 * trace_element->GetDof())
    {
      return false;
    }
    mfem::Vector shape(interior_element->GetDof());
    const mfem::IntegrationRule& nodes = trace_element->GetNodes();
    for (int scalar = 0; scalar < trace_element->GetDof(); ++scalar)
    {
      const mfem::IntegrationPoint& point = nodes.IntPoint(scalar);
      transformations->SetAllIntPoints(&point);
      interior_element->CalcShape(transformations->GetElement1IntPoint(), shape);
      for (int component = 0; component < 4; ++component)
      {
        double restricted = 0.0;
        for (int interior_scalar = 0;
             interior_scalar < interior_element->GetDof(); ++interior_scalar)
        {
          double sign = 1.0;
          const int dof = mfem::FiniteElementSpace::DecodeDof(
              interior_vdofs[ByNodes(interior_scalar,
                                     interior_element->GetDof(), component)],
              sign);
          if (dof < 0 || dof >= local_interior.Size()) return false;
          restricted += shape[interior_scalar] * sign * local_interior[dof];
        }
        double sign = 1.0;
        const int dof = mfem::FiniteElementSpace::DecodeDof(
            trace_vdofs[ByNodes(scalar, trace_element->GetDof(), component)],
            sign);
        if (dof < 0 || dof >= local_trace.Size()) return false;
        local_trace[dof] = sign * restricted;
      }
    }
  }
  local_trace.ParallelProject(projected_true_trace);
  const double* const projected = projected_true_trace.HostRead();
  if (projected == nullptr && projected_true_trace.Size() != 0) return false;
  std::vector<double> supplied;
  if (projected_true_trace.Size() != 0)
  {
    supplied.assign(projected, projected + projected_true_trace.Size());
  }
  request.initial_true_trace = {
      supplied.empty() ? nullptr : supplied.data(), supplied.size(), 4};

  GenericEulerHdgCpuSession session;
  const bool initialized = session.Initialize(request).status ==
      GenericEulerCpuSessionStatus::initialized;
  return All(initialized && SameBorrowedMesh(mesh, before, request.communicator));
}

bool CheckAcceptedRequestRankLocalCallbackFailure()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  BoundaryContext boundary{};
  boundary.fail = rank == 0;
  // The request itself is collectively admitted.  Only the boundary callback
  // fails on rank zero; every rank must return rather than entering a lone
  // neighbor exchange.
  ExactBorrowedEulerFields fields(mesh, 2, initial, source);
  const bool rejected = !ExecuteGenericEulerHdgCpu(
      Request(mesh, 2, boundary, fields));
  return All(rejected);
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  const bool boundary_semantics = CheckBoundaryExteriorSemantics();
  const bool trace_restriction = CheckBorrowedTraceRestrictsInterior();
  const bool orders = CheckOrdersAndDeterminism();
  const bool prepared_lifecycle = CheckPreparedSessionLifecycle();
  const bool committed_export = CheckCommittedExportLifecycle();
  const bool committed_export_zero_owned =
      CheckCommittedExportZeroOwnedRank();
  const bool default_numerical_policy =
      CheckDefaultNumericalPolicyAdaptiveLinearCap();
  const bool periodic = CheckPeriodicAnnotationsRemainTraceCouplings();
  const bool natural_free_stream = CheckNaturalFreeStreamPreservation();
  const bool rodas3_free_stream = CheckRodas3CommonShiftFreeStream();
  const bool rodas3_nontrivial_dae = CheckRodas3NontrivialDaeLifecycle();
  const bool rodas3_igr_initial_closure =
      CheckRodas3IgrInitialScalarClosure();
  const bool rodas3_prepared_assessment =
      CheckRodas3PreparedAssessmentAuthority();
  const bool rodas3_fixed_time_order = CheckRodas3FixedTimeThirdOrder();
  const bool rodas3_physical_conservation =
      CheckRodas3PhysicalBoundaryConservation();
  const bool rodas3_stage_equations =
      CheckRodas3SingularMassStageEquationsForTests();
  const bool rodas3_inconsistent_entry =
      CheckRodas3ScaledInconsistentEntryRollback();
  const bool conventional_conservation =
      CheckConventionalConservationEvidence();
  const bool igr_extension = CheckIgrExtensionLifecycle();
  const bool partitioned_igr =
      CheckPartitionedBackwardEulerIgrLifecycle();
  const bool partitioned_conventional =
      CheckPartitionedConventionalDegeneracy();
  const bool partitioned_orders_boundary =
      CheckPartitionedIgrOrdersAndPhysicalBoundary();
  const bool partitioned_small_dt = CheckPartitionedSmallDtTrajectory();
  const bool partitioned_fixed_time = CheckPartitionedFixedTimeFirstOrder();
  const bool igr_extension_convergence =
      CheckIgrExtensionConvergenceEvidence();
  const bool igr_extension_numerical_policy =
      CheckIgrExtensionNumericalPolicy();
  const bool igr_extension_zero_local = CheckIgrExtensionZeroLocal();
  const bool igr_extension_initialization =
      CheckIgrExtensionInitializationDeterminism();
  const bool igr_assembler_jacobian =
      CheckUnifiedIgrAssemblerJacobianForTests();
  const bool rodas3_boundary_jacobian =
      CheckRodas3PhysicalBoundaryJacobianForTests();
  const bool rodas3_algebraic_block =
      CheckRodas3AlgebraicBlockSolvabilityForTests();
  const bool llf_policy = CheckBorrowedLlfNoFloorAndFrozenRefresh();
  const bool zero_update = CheckZeroUpdateFreeStreamFinalReconstruction();
  const bool dynamic_tau = CheckDynamicTauHighWaveTwoSteps();
  const bool rodas3_checkpoint = CheckRodas3CommittedCheckpointRestart();
  const bool boundary_picard = CheckClosedBoundaryPicardAssembly();
  const bool zero_owned = CheckZeroOwnedElementEngine();
  const bool subcommunicator = CheckSubcommunicatorIsolation();
  const bool passive_world = CheckBorrowedSessionDoesNotUseWorldCollectives();
  const bool partitioned_passive_world =
      CheckBorrowedSessionDoesNotUseWorldCollectives(
          EulerHdgTimeTreatment::partitioned_backward_euler, true);
  const bool mismatch = CheckCollectiveMismatchAndAtomicFailure();
  const bool initial_trace_admission = CheckCollectiveInitialTraceAdmission();
  const bool initial_trace_zero_local =
      CheckSuppliedInitialTraceWithZeroLocalExtent();
  const bool callback_failure = CheckAcceptedRequestRankLocalCallbackFailure();
  const bool local = boundary_semantics && trace_restriction && orders &&
      prepared_lifecycle &&
      committed_export && committed_export_zero_owned &&
      default_numerical_policy &&
      periodic && natural_free_stream && conventional_conservation &&
      rodas3_free_stream && rodas3_nontrivial_dae &&
      rodas3_igr_initial_closure &&
      rodas3_prepared_assessment &&
      rodas3_fixed_time_order && rodas3_physical_conservation &&
      rodas3_stage_equations &&
      rodas3_inconsistent_entry &&
      igr_extension && igr_extension_zero_local &&
      partitioned_igr && partitioned_conventional &&
      partitioned_orders_boundary && partitioned_small_dt &&
      partitioned_fixed_time &&
      igr_extension_convergence &&
      igr_extension_numerical_policy &&
      igr_extension_initialization && igr_assembler_jacobian &&
      rodas3_boundary_jacobian && rodas3_algebraic_block && llf_policy &&
      zero_update && dynamic_tau && rodas3_checkpoint &&
      boundary_picard &&
      zero_owned && subcommunicator && passive_world &&
      partitioned_passive_world && mismatch &&
      initial_trace_admission && initial_trace_zero_local && callback_failure;
  if (!All(local))
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "generic Euler CPU test failed on rank " << rank
              << " boundary_semantics=" << boundary_semantics
              << " trace_restriction=" << trace_restriction
              << " orders=" << orders << " periodic=" << periodic
              << " rodas3_free_stream=" << rodas3_free_stream
              << " rodas3_nontrivial_dae=" << rodas3_nontrivial_dae
              << " rodas3_igr_initial_closure="
              << rodas3_igr_initial_closure
              << " rodas3_prepared_assessment="
              << rodas3_prepared_assessment
              << " rodas3_fixed_time_order=" << rodas3_fixed_time_order
              << " rodas3_physical_conservation="
              << rodas3_physical_conservation
              << " rodas3_stage_equations=" << rodas3_stage_equations
              << " rodas3_inconsistent_entry="
              << rodas3_inconsistent_entry
              << " natural_free_stream=" << natural_free_stream
              << " conventional_conservation=" << conventional_conservation
              << " igr_extension=" << igr_extension
              << " partitioned_igr=" << partitioned_igr
              << " partitioned_conventional=" << partitioned_conventional
              << " partitioned_orders_boundary="
              << partitioned_orders_boundary
              << " partitioned_small_dt=" << partitioned_small_dt
              << " partitioned_fixed_time=" << partitioned_fixed_time
              << " igr_extension_convergence=" << igr_extension_convergence
              << " igr_extension_numerical_policy="
              << igr_extension_numerical_policy
              << " igr_extension_zero_local=" << igr_extension_zero_local
              << " igr_extension_initialization="
              << igr_extension_initialization
              << " igr_assembler_jacobian=" << igr_assembler_jacobian
              << " rodas3_boundary_jacobian=" << rodas3_boundary_jacobian
              << " rodas3_algebraic_block=" << rodas3_algebraic_block
              << " llf_policy=" << llf_policy
              << " zero_update=" << zero_update
              << " prepared_lifecycle=" << prepared_lifecycle
              << " committed_export=" << committed_export
              << " committed_export_zero_owned="
              << committed_export_zero_owned
              << " default_numerical_policy=" << default_numerical_policy
              << " dynamic_tau=" << dynamic_tau
              << " rodas3_checkpoint=" << rodas3_checkpoint
              << " boundary_picard=" << boundary_picard
              << " zero_owned=" << zero_owned
              << " passive_world=" << passive_world
              << " partitioned_passive_world=" << partitioned_passive_world
              << " subcommunicator=" << subcommunicator
              << " mismatch=" << mismatch
              << " initial_trace_admission=" << initial_trace_admission
              << " initial_trace_zero_local=" << initial_trace_zero_local
              << " callback_failure=" << callback_failure << '\n';
    return 1;
  }
  return 0;
}
