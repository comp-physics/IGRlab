#include "numerics/hdg/generic/contract.hpp"
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

mfem::ParMesh MakePeriodicMesh()
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
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakePhysicalMesh()
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
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
    BoundaryContext& boundary, const bool igr)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
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
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
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
  bool conservation_audited = false;
  bool conservation_gate = false;
  double conservation_error = std::numeric_limits<double>::infinity();
  double conservation_relative_error =
      std::numeric_limits<double>::infinity();
  bool closure_audited = false;
  double closure_residual = std::numeric_limits<double>::infinity();
  double closure_tolerance = std::numeric_limits<double>::infinity();
  bool strict_algebraic_rejected = false;
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
  if (!core.ExportCommittedInterior(
          result.interior.data(), result.interior.size()))
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
  return off.evidence.committed && inactive.evidence.committed &&
      off.conservation_audited && off.conservation_gate &&
      inactive.conservation_audited && inactive.conservation_gate &&
      zero_limiter_work(off.evidence) &&
      inactive_limiter_work(inactive.evidence) &&
      SameBytes(off.interior, inactive.interior) &&
      SameBytes(off.trace, inactive.trace) &&
      off.coordinates.time == inactive.coordinates.time &&
      off.coordinates.step == inactive.coordinates.step &&
      off.coordinates.generation == inactive.coordinates.generation &&
      off_boundary.calls == 0 && inactive_boundary.calls == 0;
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
  if (!RunSuccessfulStep(request, fields, true, step, dt, 1.0e-2, 0.30))
    return false;

  const CpuAttemptEvidence& evidence = step.evidence;
  return evidence.committed && evidence.status == CpuAttemptStatus::success &&
      evidence.work.positivity_limiter_calls == 6 &&
      evidence.work.positivity_limiter_active_calls >= 1 &&
      evidence.work.positivity_algebraic_projections ==
          evidence.work.positivity_limiter_active_calls &&
      evidence.work.common_shift_operator_rebuilds >= 1 &&
      evidence.work.common_shift_operator_builds ==
          1 + evidence.work.common_shift_operator_rebuilds &&
      evidence.work.common_shift_operator_reuses == 3 &&
      step.strict_algebraic_rejected &&
      evidence.maximum_positivity_limiter_correction_norm > 0.0 &&
      std::isfinite(evidence.maximum_positivity_algebraic_correction_norm) &&
      evidence.finite_gate && evidence.admissibility_gate &&
      evidence.recovery_gate && evidence.nonlinear_gate &&
      evidence.true_residual_gate && step.conservation_audited &&
      step.conservation_gate && step.closure_audited &&
      std::isfinite(step.closure_residual) &&
      step.closure_residual <= step.closure_tolerance;
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
  const bool active_closure = CheckActiveAdmissibleMeanAndIgrClosure();
  const bool active_stage = CheckActiveCausalStageAndIgrSeeding();
  const bool bad_mean = CheckInadmissibleMeanRejectsWithoutRepair();
  if (!(off_identity && active_closure && active_stage && bad_mean))
  {
    std::cerr << "HDG Zhang--Shu lifecycle checks failed: off_identity="
              << off_identity << " active_closure=" << active_closure
              << " active_stage=" << active_stage
              << " bad_mean=" << bad_mean << '\n';
    return 1;
  }
  std::cout << "HDG Zhang--Shu CPU lifecycle checks passed\n";
  return 0;
}
