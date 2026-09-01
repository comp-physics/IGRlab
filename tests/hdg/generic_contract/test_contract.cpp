#include "numerics/hdg/generic/contract.hpp"

#include <mfem.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerBoundaryPolicy;
using dg::hdg::generic::EulerBoundaryPolicyKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::EulerHdgConservationRelativeError;
using dg::hdg::generic::EulerHdgPositivityLimiterKind;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::EulerHdgZhangShuPointSet;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::generic::ValidateEulerHdgRequest;

constexpr std::uint32_t all_cpu_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cpu) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

constexpr std::uint32_t all_cuda_capabilities =
    (all_cpu_capabilities & ~CapabilityBit(Capability::cpu)) |
    CapabilityBit(Capability::cuda);

bool ExteriorState(const void*, const dg::hdg::generic::BoundaryPointView&,
                   const double* interior, double* exterior) noexcept
{
  for (int component = 0; component < 4; ++component)
  {
    exterior[component] = interior[component];
  }
  return true;
}

mfem::ParMesh MakeSquareMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_SELF, serial);
}

mfem::ParMesh MakePeriodicSquareMesh(const MPI_Comm communicator = MPI_COMM_WORLD)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(communicator, periodic);
}

mfem::ParMesh MakeMixedPeriodicSquareMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(1, mfem::Vector(2));
  translations[0] = 0.0;
  translations[0][0] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakeRankZeroOwnedParallelMesh(const mfem::Element::Type type)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, type, true, 1.0, 1.0);
  // MFEM requires one partition owner per serial element; the 1x1 triangle
  // fixture contains two triangles even though the quadrilateral fixture has one.
  std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()));
  return mfem::ParMesh(MPI_COMM_WORLD, serial, partition.data());
}

EulerHdgRequest ValidRequest(const mfem::ParMesh& mesh,
                             const std::array<double, 4>& initial)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = 2;
  request.backend = {BackendKind::cpu, -1, CudaEngineKind::ordinary};
  request.capabilities = all_cpu_capabilities;
  request.initial_state = {initial.data(), initial.size(), 4};
  request.initial_true_trace = {initial.data(), initial.size(), 4};
  request.source = {nullptr, 0, 4};
  request.boundary = {nullptr, ExteriorState, {nullptr, 0, 0, 0.0}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(request.polynomial_order);
  return request;
}

std::vector<EulerBoundaryPolicy> PhysicalPolicies(const mfem::ParMesh& mesh)
{
  const int attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  std::vector<EulerBoundaryPolicy> policies(
      static_cast<std::size_t>(attributes));
  for (EulerBoundaryPolicy& policy : policies)
  {
    policy.kind = EulerBoundaryPolicyKind::slip_wall;
  }
  return policies;
}

void AttachPolicyTable(EulerHdgRequest& request,
                       const std::vector<EulerBoundaryPolicy>& policies)
{
  request.boundary.policies = {
      policies.empty() ? nullptr : policies.data(), policies.size(), 2, 1.4};
}

bool CheckPODSurface()
{
  const EulerHdgRequest zero_initialized{};
  return std::is_trivial<EulerHdgRequest>::value &&
      std::is_standard_layout<EulerHdgRequest>::value &&
      !std::is_polymorphic<EulerHdgRequest>::value &&
      std::is_trivial<dg::hdg::generic::BackendSelection>::value &&
      zero_initialized.numerical_policy.positivity_limiter.kind ==
          EulerHdgPositivityLimiterKind::none &&
      zero_initialized.time_treatment ==
          EulerHdgTimeTreatment::backward_euler;
}

bool CheckConservationNormalization()
{
  constexpr double error = 0.25;
  constexpr double scale = 0.5;
  constexpr double measure = 1.0;
  constexpr double replication = 8.0;
  const double unit = EulerHdgConservationRelativeError(
      error, scale, measure);
  const double replicated = EulerHdgConservationRelativeError(
      replication * error, replication * scale,
      replication * measure);
  return unit == replicated && unit == 1.0 / 6.0 &&
      std::isinf(EulerHdgConservationRelativeError(error, scale, 0.0)) &&
      std::isinf(EulerHdgConservationRelativeError(-error, scale, measure));
}

bool CheckAcceptAndReject()
{
  mfem::ParMesh mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }

  if (request.time_treatment != EulerHdgTimeTreatment::backward_euler)
  {
    return false;
  }
  request.time_treatment =
      EulerHdgTimeTreatment::partitioned_backward_euler;
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted ||
      std::string_view(dg::hdg::generic::EulerHdgTimeTreatmentName(
        request.time_treatment)) != "rodas3")
  {
    return false;
  }
  request.time_treatment = static_cast<EulerHdgTimeTreatment>(255);
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_time_treatment)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);

  request.polynomial_order = 0;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::unsupported_polynomial_order)
  {
    return false;
  }
  request.polynomial_order = 4;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::unsupported_polynomial_order)
  {
    return false;
  }
  request.polynomial_order = 1;
  request.capabilities &= ~CapabilityBit(Capability::cpu);
  if (ValidateEulerHdgRequest(request) != ContractStatus::missing_capability)
  {
    return false;
  }
  request.capabilities = all_cuda_capabilities;
  request.backend = {BackendKind::cuda, 0, CudaEngineKind::ordinary};
  if (request.backend.cuda_engine != CudaEngineKind::ordinary ||
      ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }
  request.backend.cuda_engine = CudaEngineKind::tensor_core;
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }
  request.backend.cuda_engine = static_cast<CudaEngineKind>(255);
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_backend)
  {
    return false;
  }
  request.backend.cuda_engine = CudaEngineKind::ordinary;
  request.backend.device_ordinal = -1;
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_backend)
  {
    return false;
  }
  request.backend = {BackendKind::cpu, -1, CudaEngineKind::ordinary};
  request.capabilities = all_cpu_capabilities;
  request.backend.cuda_engine = CudaEngineKind::tensor_core;
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_backend)
  {
    return false;
  }
  request.backend.cuda_engine = CudaEngineKind::ordinary;
  request.initial_state = {nullptr, initial.size(), 4};
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_initial_state)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.source = {initial.data(), initial.size(), 3};
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_source)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.initial_true_trace = {initial.data(), initial.size(), 3};
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_initial_trace)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.numerical_policy.conventional_assembly_quadrature_order = 0;
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_numerical_policy)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.numerical_policy.igr_assembly_quadrature_orders.volume_order = 0;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_numerical_policy)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.numerical_policy.igr_assembly_quadrature_orders.face_order = 0;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_numerical_policy)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.boundary.exterior_state = nullptr;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  return true;
}

bool CheckTriangleRejection()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh triangle(MPI_COMM_SELF, serial);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  return ValidateEulerHdgRequest(ValidRequest(triangle, initial)) ==
      ContractStatus::nonquadrilateral_mesh;
}

bool CheckBoundaryPolicyAdmission()
{
  mfem::ParMesh mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  std::vector<EulerBoundaryPolicy> policies = PhysicalPolicies(mesh);
  if (policies.empty()) return false;
  EulerHdgRequest request = ValidRequest(mesh, initial);
  AttachPolicyTable(request, policies);
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }

  policies[0].kind = static_cast<EulerBoundaryPolicyKind>(255);
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  policies[0].kind = EulerBoundaryPolicyKind::slip_wall;
  policies[0].reference_state[0] =
      std::numeric_limits<double>::quiet_NaN();
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  policies[0].reference_state[0] = 0.0;
  request.boundary.policies.gamma = 1.0;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  request.boundary.policies.gamma = 1.4;
  request.boundary.policies.policies = nullptr;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  AttachPolicyTable(request, policies);
  policies.push_back(EulerBoundaryPolicy{});
  AttachPolicyTable(request, policies);
  return ValidateEulerHdgRequest(request) ==
      ContractStatus::invalid_boundary_operator;
}

bool CheckBoundaryPolicyCollectiveMismatch()
{
  int rank = 0;
  int ranks = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS)
  {
    return false;
  }
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const int local_attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  int minimum_attributes = local_attributes;
  int maximum_attributes = local_attributes;
  if (MPI_Allreduce(MPI_IN_PLACE, &minimum_attributes, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(MPI_IN_PLACE, &maximum_attributes, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      minimum_attributes != maximum_attributes || minimum_attributes <= 0)
  {
    return false;
  }
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  std::vector<EulerBoundaryPolicy> policies = PhysicalPolicies(mesh);
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  AttachPolicyTable(request, policies);
  policies[0].pressure_out = rank == 0 ? 0.0 : 1.0;
  const ContractStatus status = ValidateEulerHdgRequest(request);
  int minimum = static_cast<int>(status);
  int maximum = minimum;
  const int minimum_result = MPI_Allreduce(
      MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_result = MPI_Allreduce(
      MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const ContractStatus expected = ranks == 1
      ? ContractStatus::accepted
      : ContractStatus::invalid_boundary_operator;
  return status == expected && minimum_result == MPI_SUCCESS &&
      maximum_result == MPI_SUCCESS && minimum == maximum;
}

bool CheckMixedBoundaryPolicyAdmission()
{
  mfem::ParMesh mesh = MakeMixedPeriodicSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  const int retained_attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  int physical_attribute_count = 0;
  int collective_retained_attributes = 0;
  std::vector<unsigned char> physical_attribute_mask;
  const bool topology_ready =
      dg::hdg::generic::GlobalPhysicalBoundaryAttributeCount(
          mesh, mesh.GetComm(), physical_attribute_count,
          collective_retained_attributes, physical_attribute_mask);
  const std::vector<unsigned char> expected_physical_mask{1, 0, 1, 0};
  if (!topology_ready || retained_attributes != 4 ||
      physical_attribute_count != retained_attributes ||
      collective_retained_attributes != retained_attributes ||
      physical_attribute_mask != expected_physical_mask)
    return false;
  std::vector<EulerBoundaryPolicy> policies(
      static_cast<std::size_t>(retained_attributes));
  for (EulerBoundaryPolicy& policy : policies)
    policy.kind = EulerBoundaryPolicyKind::slip_wall;
  // A one-direction periodic mesh retains annotations for the identified
  // sides while the orthogonal sides remain physical.  The exact retained
  // table must therefore permit both semantic entries.
  policies[1].kind = EulerBoundaryPolicyKind::periodic_topology;
  policies[3].kind = EulerBoundaryPolicyKind::periodic_topology;
  EulerHdgRequest request = ValidRequest(mesh, initial);
  AttachPolicyTable(request, policies);
  const ContractStatus mixed = ValidateEulerHdgRequest(request);

  std::vector<EulerBoundaryPolicy> physical_marked_periodic = policies;
  physical_marked_periodic[0].kind =
      EulerBoundaryPolicyKind::periodic_topology;
  AttachPolicyTable(request, physical_marked_periodic);
  const ContractStatus wrong_physical_kind =
      ValidateEulerHdgRequest(request);

  std::vector<EulerBoundaryPolicy> periodic_marked_physical = policies;
  periodic_marked_physical[1].kind = EulerBoundaryPolicyKind::slip_wall;
  AttachPolicyTable(request, periodic_marked_physical);
  const ContractStatus wrong_periodic_kind =
      ValidateEulerHdgRequest(request);

  policies.pop_back();
  AttachPolicyTable(request, policies);
  const ContractStatus short_table = ValidateEulerHdgRequest(request);
  return mixed == ContractStatus::accepted &&
      wrong_physical_kind == ContractStatus::invalid_boundary_operator &&
      wrong_periodic_kind == ContractStatus::invalid_boundary_operator &&
      short_table == ContractStatus::invalid_boundary_operator;
}

bool CheckIgrSelectionAdmission()
{
  mfem::ParMesh mesh = MakePeriodicSquareMesh();
  mfem::ParMesh physical_mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  const auto admit_igr = [&initial](const mfem::ParMesh& target,
                                    const int order)
  {
    EulerHdgRequest candidate = ValidRequest(target, initial);
    candidate.polynomial_order = order;
    candidate.numerical_policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
    candidate.numerical_policy.stabilization =
        {RegularizationKind::igr, 0.25, 0.5};
    return ValidateEulerHdgRequest(candidate);
  };
  const ContractStatus periodic_p1 = admit_igr(mesh, 1);
  const ContractStatus periodic_p2 = admit_igr(mesh, 2);
  const ContractStatus periodic_p3 = admit_igr(mesh, 3);

  EulerHdgRequest present_empty_periodic = ValidRequest(mesh, initial);
  present_empty_periodic.boundary.policies = {nullptr, 0, 2, 1.4};
  const ContractStatus present_empty_periodic_status =
      ValidateEulerHdgRequest(present_empty_periodic);

  const int retained_attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  std::vector<EulerBoundaryPolicy> retained_periodic_policies(
      static_cast<std::size_t>(retained_attributes));
  for (EulerBoundaryPolicy& policy : retained_periodic_policies)
    policy.kind = EulerBoundaryPolicyKind::periodic_topology;
  EulerHdgRequest retained_periodic = ValidRequest(mesh, initial);
  AttachPolicyTable(retained_periodic, retained_periodic_policies);
  const ContractStatus retained_periodic_status =
      ValidateEulerHdgRequest(retained_periodic);

  std::vector<EulerBoundaryPolicy> short_retained_periodic_policies =
      retained_periodic_policies;
  if (!short_retained_periodic_policies.empty())
    short_retained_periodic_policies.pop_back();
  EulerHdgRequest short_retained_periodic = ValidRequest(mesh, initial);
  AttachPolicyTable(
      short_retained_periodic, short_retained_periodic_policies);
  const ContractStatus short_retained_periodic_status =
      ValidateEulerHdgRequest(short_retained_periodic);

  std::vector<EulerBoundaryPolicy> extra_retained_periodic_policies =
      retained_periodic_policies;
  extra_retained_periodic_policies.push_back(EulerBoundaryPolicy{});
  extra_retained_periodic_policies.back().kind =
      EulerBoundaryPolicyKind::periodic_topology;
  EulerHdgRequest extra_retained_periodic = ValidRequest(mesh, initial);
  AttachPolicyTable(
      extra_retained_periodic, extra_retained_periodic_policies);
  const ContractStatus extra_retained_periodic_status =
      ValidateEulerHdgRequest(extra_retained_periodic);

  if (!retained_periodic_policies.empty())
    retained_periodic_policies.front().kind =
        EulerBoundaryPolicyKind::slip_wall;
  const ContractStatus retained_physical_status =
      ValidateEulerHdgRequest(retained_periodic);
  const ContractStatus physical_p1 = admit_igr(physical_mesh, 1);
  const ContractStatus physical_p2 = admit_igr(physical_mesh, 2);
  const ContractStatus physical_p3 = admit_igr(physical_mesh, 3);

  EulerHdgRequest request = ValidRequest(mesh, initial);
  request.polynomial_order = 1;
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(1);
  request.numerical_policy.stabilization = {RegularizationKind::igr, 0.25, 0.5};
  request.backend = {BackendKind::cuda, 0, CudaEngineKind::ordinary};
  request.capabilities = all_cuda_capabilities;
  const ContractStatus cuda = ValidateEulerHdgRequest(request);
  request.backend = {BackendKind::cpu, -1, CudaEngineKind::ordinary};
  request.capabilities = all_cpu_capabilities;
  int rank = 0, ranks = 0;
  MPI_Comm_rank(mesh.GetComm(), &rank);
  MPI_Comm_size(mesh.GetComm(), &ranks);
  request.numerical_policy.igr_assembly_quadrature_orders.face_order =
      rank == 0 ? 12 : 10;
  const ContractStatus quadrature_mismatch =
      ValidateEulerHdgRequest(request);
  request.numerical_policy.igr_assembly_quadrature_orders.face_order = 12;
  request.numerical_policy.stabilization.alpha = rank == 0 ? 0.25 : 0.5;
  const ContractStatus mismatch = ValidateEulerHdgRequest(request);
  int min_status = static_cast<int>(mismatch), max_status = min_status;
  const int min_result = MPI_Allreduce(MPI_IN_PLACE, &min_status, 1, MPI_INT,
                                       MPI_MIN, mesh.GetComm());
  const int max_result = MPI_Allreduce(MPI_IN_PLACE, &max_status, 1, MPI_INT,
                                       MPI_MAX, mesh.GetComm());
  return periodic_p1 == ContractStatus::accepted &&
      periodic_p2 == ContractStatus::accepted &&
      periodic_p3 == ContractStatus::accepted &&
      present_empty_periodic_status == ContractStatus::accepted &&
      !retained_periodic_policies.empty() &&
      retained_periodic_status == ContractStatus::accepted &&
      short_retained_periodic_status ==
          ContractStatus::invalid_boundary_operator &&
      extra_retained_periodic_status ==
          ContractStatus::invalid_boundary_operator &&
      retained_physical_status == ContractStatus::invalid_boundary_operator &&
      physical_p1 == ContractStatus::accepted &&
      physical_p2 == ContractStatus::accepted &&
      physical_p3 == ContractStatus::accepted &&
      cuda == ContractStatus::accepted &&
      quadrature_mismatch ==
          (ranks == 1 ? ContractStatus::accepted
                      : ContractStatus::invalid_numerical_policy) &&
      mismatch == (ranks == 1 ? ContractStatus::accepted
                              : ContractStatus::invalid_numerical_policy) &&
      min_result == MPI_SUCCESS && max_result == MPI_SUCCESS &&
      min_status == max_status;
}

bool CheckRoleSpecificLinearTolerancePolicy()
{
  dg::hdg::generic::EulerHdgNumericalPolicy policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(2);
  const bool defaults_exact =
      dg::hdg::generic::ValidEulerHdgNumericalPolicy(policy) &&
      dg::hdg::generic::EulerHdgRequestedLinearRelativeTolerance(
          policy,
          dg::hdg::generic::EulerHdgLinearSolveRole::newton_correction) ==
          1.0e-12 &&
      dg::hdg::generic::EulerHdgRequestedLinearRelativeTolerance(
          policy,
          dg::hdg::generic::EulerHdgLinearSolveRole::final_coherent) ==
          1.0e-12;

  policy.newton_correction_linear_relative_tolerance = 1.0e-6;
  policy.final_coherent_linear_relative_tolerance = 1.0e-10;
  policy.true_residual_tightening_linear_relative_tolerance = 1.0e-12;
  const bool split_exact =
      dg::hdg::generic::ValidEulerHdgNumericalPolicy(policy) &&
      dg::hdg::generic::EulerHdgRequestedLinearRelativeTolerance(
          policy,
          dg::hdg::generic::EulerHdgLinearSolveRole::newton_correction) ==
          1.0e-6 &&
      dg::hdg::generic::EulerHdgRequestedLinearRelativeTolerance(
          policy,
          dg::hdg::generic::EulerHdgLinearSolveRole::final_coherent) ==
          1.0e-10;

  dg::hdg::generic::EulerHdgNumericalPolicy invalid = policy;
  invalid.true_residual_tightening_linear_relative_tolerance = 1.0e-5;
  const bool rejects_looser_fallback =
      !dg::hdg::generic::ValidEulerHdgNumericalPolicy(invalid);
  invalid = policy;
  invalid.true_residual_tightening_linear_relative_tolerance = 0.0;
  const bool rejects_zero_fallback =
      !dg::hdg::generic::ValidEulerHdgNumericalPolicy(invalid);
  return defaults_exact && split_exact && rejects_looser_fallback &&
      rejects_zero_fallback;
}

bool CheckPositivityLimiterPolicy()
{
  auto policy = dg::hdg::generic::DefaultEulerHdgNumericalPolicy(2);
  const bool default_off =
      policy.positivity_limiter.kind ==
          EulerHdgPositivityLimiterKind::none &&
      dg::hdg::generic::ValidEulerHdgNumericalPolicy(policy);

  policy.positivity_limiter = {
      EulerHdgPositivityLimiterKind::zhang_shu,
      EulerHdgZhangShuPointSet::consumer,
      5.0e-7,
      1.0e-6,
      12};
  const bool active_valid =
      dg::hdg::generic::ValidEulerHdgNumericalPolicy(policy);

  auto invalid = dg::hdg::generic::DefaultEulerHdgNumericalPolicy(2);
  invalid.positivity_limiter.density_floor = 5.0e-7;
  const bool rejects_noncanonical_off =
      !dg::hdg::generic::ValidEulerHdgNumericalPolicy(invalid);
  invalid = policy;
  invalid.positivity_limiter.point_set =
      static_cast<EulerHdgZhangShuPointSet>(255);
  const bool rejects_unknown_points =
      !dg::hdg::generic::ValidEulerHdgNumericalPolicy(invalid);
  invalid = policy;
  invalid.positivity_limiter.pressure_bisection_iterations = 0;
  const bool rejects_zero_iterations =
      !dg::hdg::generic::ValidEulerHdgNumericalPolicy(invalid);
  invalid = policy;
  invalid.positivity_limiter.pressure_bisection_iterations = 65;
  const bool rejects_excessive_iterations =
      !dg::hdg::generic::ValidEulerHdgNumericalPolicy(invalid);

  mfem::ParMesh mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  request.numerical_policy.positivity_limiter = policy.positivity_limiter;
  const bool rejects_backward_euler =
      ValidateEulerHdgRequest(request) ==
          ContractStatus::invalid_numerical_policy;
  request.time_treatment =
      EulerHdgTimeTreatment::partitioned_backward_euler;
  const bool rejects_partitioned_backward_euler =
      ValidateEulerHdgRequest(request) ==
          ContractStatus::invalid_numerical_policy;
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  const bool admits_rodas3 =
      ValidateEulerHdgRequest(request) == ContractStatus::accepted;

  return default_off && active_valid && rejects_noncanonical_off &&
      rejects_unknown_points && rejects_zero_iterations &&
      rejects_excessive_iterations && rejects_backward_euler &&
      rejects_partitioned_backward_euler && admits_rodas3;
}

bool CheckZeroOwnedElementAdmission()
{
  int ranks = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS || ranks < 2)
  {
    return true;
  }
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  const ContractStatus status = ValidateEulerHdgRequest(request);
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int has_empty_rank = 0;
  int status_code = static_cast<int>(status);
  int minimum_status = status_code;
  int maximum_status = status_code;
  if (MPI_Allreduce(&local_empty, &has_empty_rank, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&status_code, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&status_code, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return has_empty_rank == 1 && status == ContractStatus::accepted &&
      minimum_status == static_cast<int>(ContractStatus::accepted) &&
      maximum_status == static_cast<int>(ContractStatus::accepted);
}

bool CheckCudaEngineCollectiveMismatch()
{
  int rank = 0;
  int ranks = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS)
  {
    return false;
  }
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  request.backend = {
      BackendKind::cuda, 0,
      rank == 0 ? CudaEngineKind::ordinary : CudaEngineKind::tensor_core};
  request.capabilities = all_cuda_capabilities;
  const ContractStatus backend_status = ValidateEulerHdgRequest(request);
  int minimum = static_cast<int>(backend_status);
  int maximum = minimum;
  const int minimum_result = MPI_Allreduce(
      MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_result = MPI_Allreduce(
      MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const ContractStatus expected_backend = ranks == 1
      ? ContractStatus::accepted
      : ContractStatus::invalid_backend;
  request.backend = {BackendKind::cuda, 0, CudaEngineKind::ordinary};
  request.time_treatment = rank == 0
      ? EulerHdgTimeTreatment::backward_euler
      : EulerHdgTimeTreatment::partitioned_backward_euler;
  const ContractStatus time_status = ValidateEulerHdgRequest(request);
  const ContractStatus expected_time = ranks == 1
      ? ContractStatus::accepted
      : ContractStatus::invalid_time_treatment;
  return backend_status == expected_backend && time_status == expected_time &&
      minimum_result == MPI_SUCCESS && maximum_result == MPI_SUCCESS &&
      minimum == maximum;
}

bool CheckAsymmetricNullMeshConvergence()
{
  int rank = 0;
  int ranks = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS)
  {
    return false;
  }
  if (ranks < 2) { return true; }

  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  if (rank == 0) { request.mesh = nullptr; }

  const ContractStatus status = ValidateEulerHdgRequest(request);
  int minimum = static_cast<int>(status);
  int maximum = minimum;
  const int minimum_result = MPI_Allreduce(
      MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_result = MPI_Allreduce(
      MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return status == ContractStatus::null_mesh &&
      minimum_result == MPI_SUCCESS && maximum_result == MPI_SUCCESS &&
      minimum == static_cast<int>(ContractStatus::null_mesh) &&
      maximum == minimum;
}

bool CheckCollectiveFailureConvergence()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  request.polynomial_order = rank == 0 ? 4 : 2;
  const int local_status = static_cast<int>(ValidateEulerHdgRequest(request));
  int minimum_status = local_status;
  int maximum_status = local_status;
  if (MPI_Allreduce(&local_status, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&local_status, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return local_status == static_cast<int>(ContractStatus::unsupported_polynomial_order) &&
      minimum_status == local_status && maximum_status == local_status;
}

bool CheckCollectiveGeometryConvergence()
{
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::TRIANGLE);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  const int local_status =
      static_cast<int>(ValidateEulerHdgRequest(ValidRequest(mesh, initial)));
  int minimum_status = local_status;
  int maximum_status = local_status;
  if (MPI_Allreduce(&local_status, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&local_status, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return local_status == static_cast<int>(ContractStatus::nonquadrilateral_mesh) &&
      minimum_status == local_status && maximum_status == local_status;
}

bool CheckPhysicalPolicyWithZeroOwnedRank()
{
  int ranks = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS || ranks < 2)
  {
    return true;
  }
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  const int local_attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  int global_attributes = 0;
  if (MPI_Allreduce(&local_attributes, &global_attributes, 1, MPI_INT,
                    MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS ||
      global_attributes <= 0)
    return false;
  std::vector<EulerBoundaryPolicy> policies(
      static_cast<std::size_t>(global_attributes));
  for (EulerBoundaryPolicy& policy : policies)
    policy.kind = EulerBoundaryPolicyKind::slip_wall;
  AttachPolicyTable(request, policies);
  const int accepted_status = static_cast<int>(
      ValidateEulerHdgRequest(request));
  policies.front().kind = EulerBoundaryPolicyKind::periodic_topology;
  const int rejected_status = static_cast<int>(
      ValidateEulerHdgRequest(request));
  int minimum_status = rejected_status;
  int maximum_status = rejected_status;
  if (MPI_Allreduce(&rejected_status, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&rejected_status, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return !policies.empty() &&
      accepted_status == static_cast<int>(ContractStatus::accepted) &&
      rejected_status ==
          static_cast<int>(ContractStatus::invalid_boundary_operator) &&
      minimum_status == rejected_status && maximum_status == rejected_status;
}

}  // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  bool ok = CheckPODSurface();
  const bool conservation_normalization = CheckConservationNormalization();
  const bool basic = CheckAcceptAndReject();
  const bool triangle = CheckTriangleRejection();
  const bool boundary = CheckBoundaryPolicyAdmission();
  const bool boundary_mismatch = CheckBoundaryPolicyCollectiveMismatch();
  const bool mixed_boundary = CheckMixedBoundaryPolicyAdmission();
  const bool igr = CheckIgrSelectionAdmission();
  const bool linear_tolerances = CheckRoleSpecificLinearTolerancePolicy();
  const bool positivity_limiter = CheckPositivityLimiterPolicy();
  const bool empty = CheckZeroOwnedElementAdmission();
  const bool cuda_engine_mismatch = CheckCudaEngineCollectiveMismatch();
  const bool physical_empty = CheckPhysicalPolicyWithZeroOwnedRank();
  const bool asymmetric_null = CheckAsymmetricNullMeshConvergence();
  const bool mismatch = CheckCollectiveFailureConvergence();
  const bool geometry = CheckCollectiveGeometryConvergence();
  ok = ok && conservation_normalization && basic && triangle && boundary &&
      boundary_mismatch &&
      mixed_boundary && igr && linear_tolerances && positivity_limiter &&
      empty &&
      cuda_engine_mismatch &&
      physical_empty && asymmetric_null &&
      mismatch && geometry;
  return ok ? 0 : 1;
}
