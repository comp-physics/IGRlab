#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cpu::ExecuteGenericEulerHdgCpu;
using dg::hdg::runtime::cpu::GenericEulerCpuPreparedStep;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionResult;
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
                        const ExactBorrowedEulerFields& fields)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = cpu_capabilities;
  request.initial_state = {fields.initial.empty() ? nullptr : fields.initial.data(),
                           fields.initial.size(), 4};
  request.initial_true_trace = {nullptr, 0, 4};
  request.source = {fields.source.empty() ? nullptr : fields.source.data(),
                    fields.source.size(), 4};
  request.boundary = {&boundary, NonperiodicExterior};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
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
    request.backend = {BackendKind::cpu, -1};
    request.capabilities = cpu_capabilities;
    request.initial_state = initial.empty()
        ? dg::hdg::generic::EulerFieldView{zero_local_initial.data(),
                                           zero_local_initial.size(), 4}
        : dg::hdg::generic::EulerFieldView{initial.data(), initial.size(), 4};
    request.initial_true_trace = {nullptr, 0, 4};
    request.source = {nullptr, 0, 4};
    request.boundary = {&boundary, NonperiodicExterior};
    request.numerical_policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
    if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
        dg::hdg::generic::ContractStatus::accepted)
    {
      return false;
    }
    dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
    if (!core.InitializeBorrowedEuler(request)) return false;
    if (core.evidence().tuple.stabilization.view() !=
        "tau.llf.max-fixed-local.v1")
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
  if (!dg::hdg::generic::HasHdgEulerSessionCapabilities(
          session.capabilities(), required_capabilities) ||
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

bool SameCommittedObservables(
    const dg::hdg::generic::HdgEulerCommittedObservables& left,
    const dg::hdg::generic::HdgEulerCommittedObservables& right)
{
  return left.available == right.available &&
      left.coordinates.time == right.coordinates.time &&
      left.coordinates.dt == right.coordinates.dt &&
      left.coordinates.step == right.coordinates.step &&
      left.coordinates.generation == right.coordinates.generation &&
      left.quadrature_order == right.quadrature_order &&
      left.conserved_integrals == right.conserved_integrals &&
      left.conserved_l2_norms == right.conserved_l2_norms &&
      left.minimum_density == right.minimum_density &&
      left.minimum_pressure == right.minimum_pressure;
}

bool CheckCommittedObservables()
{
  const dg::hdg::generic::HdgEulerCommittedObservables unavailable{};
  if (dg::hdg::generic::IsValidHdgEulerCommittedObservables(unavailable, 0))
  {
    return false;
  }
  GenericEulerHdgCpuSession uninitialized;
  if (uninitialized.ObserveCommitted().available) { return false; }

  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                        0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                       0.00, 0.00, 0.03, 0.04}};
  BoundaryContext boundary{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  if (request.numerical_policy.observable_quadrature_order != 4) { return false; }
  request.numerical_policy.observable_quadrature_order = 9;
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  const auto before_coordinates = session.coordinates();
  const auto before = session.ObserveCommitted();
  if (!before.available || before.quadrature_order != 9 ||
      !dg::hdg::generic::IsValidHdgEulerCommittedObservables(
          before, session.capabilities()) ||
      before.coordinates.time != before_coordinates.time ||
      before.coordinates.dt != before_coordinates.dt ||
      before.coordinates.step != before_coordinates.step ||
      before.coordinates.generation != before_coordinates.generation)
  {
    return false;
  }
  std::vector<double> committed_before(fields.initial.size());
  if (!session.ExportCommittedInterior(
          {committed_before.data(), committed_before.size()}))
  {
    return false;
  }

  GenericEulerCpuPreparedStep prepared;
  if (session.Prepare(0.005, prepared).status !=
          GenericEulerCpuSessionStatus::prepared ||
      !SameCommittedObservables(before, session.ObserveCommitted()) ||
      session.coordinates().generation != before_coordinates.generation)
  {
    return false;
  }
  std::vector<double> committed_during_prepare(committed_before.size());
  if (!session.ExportCommittedInterior(
          {committed_during_prepare.data(), committed_during_prepare.size()}) ||
      committed_during_prepare != committed_before ||
      session.Abort(std::move(prepared)).status !=
          GenericEulerCpuSessionStatus::aborted ||
      !SameCommittedObservables(before, session.ObserveCommitted()))
  {
    return false;
  }

  if (session.Advance(0.007).status != GenericEulerCpuSessionStatus::advanced)
  {
    return false;
  }
  std::vector<double> committed_after_before_observe(committed_before.size());
  if (!session.ExportCommittedInterior(
          {committed_after_before_observe.data(),
           committed_after_before_observe.size()}))
  {
    return false;
  }
  const auto after = session.ObserveCommitted();
  std::vector<double> committed_after(committed_before.size());
  return after.available && after.quadrature_order == 9 &&
      after.coordinates.step == 1 && after.coordinates.generation == 1 &&
      after.coordinates.dt == 0.007 && after.coordinates.time == 0.007 &&
      dg::hdg::generic::IsValidHdgEulerCommittedObservables(
          after, session.capabilities()) &&
      session.ExportCommittedInterior(
          {committed_after.data(), committed_after.size()}) &&
      committed_after == committed_after_before_observe &&
      session.coordinates().step == after.coordinates.step;
}

bool CheckCommittedObservablesZeroOwnedRank()
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
  const auto observables = session.ObserveCommitted();
  return observables.available &&
      dg::hdg::generic::IsValidHdgEulerCommittedObservables(
          observables, session.capabilities());
}

bool SameCommittedResidualAudit(
    const dg::hdg::generic::HdgEulerCommittedResidualAudit& left,
    const dg::hdg::generic::HdgEulerCommittedResidualAudit& right)
{
  return left.available == right.available &&
      left.coordinates.time == right.coordinates.time &&
      left.coordinates.dt == right.coordinates.dt &&
      left.coordinates.step == right.coordinates.step &&
      left.coordinates.generation == right.coordinates.generation &&
      left.residual_generation == right.residual_generation &&
      left.quadrature_order == right.quadrature_order &&
      left.formulation == right.formulation &&
      ((std::isnan(left.norm) && std::isnan(right.norm)) ||
       left.norm == right.norm);
}

bool CheckCommittedResidualAuditLifecycle()
{
  GenericEulerHdgCpuSession uninitialized;
  const auto empty = uninitialized.committed_residual_audit();
  if (empty.available || !std::isnan(empty.norm) ||
      dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(empty, 0))
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
  EulerHdgRequest request = Request(mesh, 1, boundary, fields);
  request.numerical_policy.conventional_assembly_quadrature_order = 4;
  request.numerical_policy.audit_quadrature_order = 9;
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
          GenericEulerCpuSessionStatus::initialized ||
      !dg::hdg::generic::HasHdgEulerSessionCapabilities(
          session.capabilities(),
          dg::hdg::generic::HdgEulerSessionCapabilityBit(
              dg::hdg::generic::HdgEulerSessionCapability::committed_nonlinear_residual_audit)))
  {
    return false;
  }
  const auto before = session.committed_residual_audit();
  GenericEulerCpuPreparedStep prepared;
  if (session.Prepare(0.005, prepared).status !=
          GenericEulerCpuSessionStatus::prepared ||
      !SameCommittedResidualAudit(before, session.committed_residual_audit()) ||
      session.Abort(std::move(prepared)).status !=
          GenericEulerCpuSessionStatus::aborted ||
      !SameCommittedResidualAudit(before, session.committed_residual_audit()))
  {
    return false;
  }
  GenericEulerCpuPreparedStep committed;
  if (session.Prepare(0.007, committed).status !=
          GenericEulerCpuSessionStatus::prepared ||
      session.Commit(std::move(committed)).status !=
          GenericEulerCpuSessionStatus::committed)
  {
    return false;
  }
  const auto first = session.committed_residual_audit();
  if (!dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
          first, session.capabilities()) ||
      first.coordinates.step != 1 || first.coordinates.generation != 1 ||
      first.residual_generation != 1 || first.quadrature_order != 9 ||
      first.formulation !=
          dg::hdg::generic::HdgEulerResidualFormulation::conventional_u4_uhat4)
  {
    return false;
  }
  if (session.Advance(0.009).status != GenericEulerCpuSessionStatus::advanced)
  {
    return false;
  }
  const auto second = session.committed_residual_audit();
  return dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
             second, session.capabilities()) &&
      second.coordinates.step == 2 && second.coordinates.generation == 2 &&
      second.residual_generation == 2 && second.quadrature_order == 9 &&
      second.formulation ==
          dg::hdg::generic::HdgEulerResidualFormulation::conventional_u4_uhat4;
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
      default_policy.linear_relative_tolerance != 1.0e-12 ||
      default_policy.igr_assembly_quadrature_order != 12 ||
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
  request.numerical_policy.linear_relative_tolerance = 1.0e-11;
  request.numerical_policy.true_residual_relative_tolerance = 1.0e-9;
  GenericEulerHdgCpuSession session;
  return session.Initialize(request).status ==
      GenericEulerCpuSessionStatus::initialized &&
      session.Advance(0.001).status ==
      GenericEulerCpuSessionStatus::advanced &&
      context.calls == 0;
}

bool CheckPrivateTracedScalarIgrLifecycle()
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
  const auto audit_before = session.committed_residual_audit();
  if (audit_before.available || !std::isnan(audit_before.norm)) return false;
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre,
                                   mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(&mesh, &collection, 4,
                                    mfem::Ordering::byNODES);
  std::vector<double> before(static_cast<std::size_t>(space.GetVSize()), 0.0);
  const bool exported_before = session.ExportCommittedInterior(
      {before.data(), before.size()});
  GenericEulerCpuPreparedStep first;
  const auto prepared = session.Prepare(0.001, first);
  std::vector<double> candidate(before.size(), 0.0);
  const bool exported_candidate = prepared.status == GenericEulerCpuSessionStatus::prepared &&
      session.ExportPreparedInterior(first, {candidate.data(), candidate.size()});
  const bool audit_hidden_during_prepare =
      SameCommittedResidualAudit(audit_before, session.committed_residual_audit());
  const bool changed = candidate != before;
  const auto aborted = prepared.status == GenericEulerCpuSessionStatus::prepared
      ? session.Abort(std::move(first)) : decltype(session.Abort(std::move(first))){ };
  std::vector<double> after_abort(before.size(), 0.0);
  const bool rollback = session.ExportCommittedInterior({after_abort.data(), after_abort.size()}) &&
      after_abort == before;
  const bool audit_preserved_by_abort =
      SameCommittedResidualAudit(audit_before, session.committed_residual_audit());
  GenericEulerCpuPreparedStep retry;
  const auto retried = session.Prepare(0.001, retry);
  const auto committed = retried.status == GenericEulerCpuSessionStatus::prepared
      ? session.Commit(std::move(retry)) : decltype(session.Commit(std::move(retry))){ };
  const auto audit_after_commit = session.committed_residual_audit();
  const auto second = session.Advance(0.001);
  const auto audit_after_second = session.committed_residual_audit();
  return exported_before && exported_candidate && changed &&
      audit_hidden_during_prepare &&
      aborted.status == GenericEulerCpuSessionStatus::aborted && rollback &&
      audit_preserved_by_abort &&
      committed.status == GenericEulerCpuSessionStatus::committed &&
      dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
          audit_after_commit, session.capabilities()) &&
      audit_after_commit.formulation ==
          dg::hdg::generic::HdgEulerResidualFormulation::traced_scalar_igr_p1_u7_uhat5 &&
      audit_after_commit.quadrature_order !=
          request.numerical_policy.igr_assembly_quadrature_order &&
      second.status == GenericEulerCpuSessionStatus::advanced &&
      dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
          audit_after_second, session.capabilities()) &&
      audit_after_second.coordinates.generation == 2 && boundary.calls == 0;
}

bool CheckPrivateTracedScalarIgrConvergenceEvidence()
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
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation)) return false;
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(generation, coordinates, maximum_wave_speed,
                                stable_dt)) return false;
  const auto& attempt = core.evidence().last_attempt;
  return attempt.committed && attempt.nonlinear_updates > 1 &&
      attempt.finite_gate && attempt.admissibility_gate &&
      attempt.recovery_gate && attempt.nonlinear_gate &&
      attempt.true_residual_gate && attempt.conservation_gate &&
      attempt.work.accepted_newton_updates ==
          static_cast<std::uint64_t>(attempt.nonlinear_updates) &&
      attempt.work.complete_factor_setups == attempt.work.complete_trace_solves &&
      attempt.work.complete_trace_solves == attempt.work.complete_recoveries &&
      attempt.final_nonlinear_residual <= attempt.initial_nonlinear_residual &&
      std::isfinite(attempt.maximum_true_relative_residual) &&
      attempt.maximum_true_relative_residual <= 1.0e-10 &&
      std::isfinite(attempt.maximum_conservation_error) &&
      std::isfinite(attempt.maximum_conservation_relative_error) &&
      attempt.maximum_conservation_relative_error <= 1.0e-10 &&
      boundary.calls == 0;
}

bool SamePrimaryIgrAttemptEvidence(
    const dg::hdg::runtime::cpu::CpuAttemptEvidence& left,
    const dg::hdg::runtime::cpu::CpuAttemptEvidence& right)
{
  return left.status == right.status &&
      left.failure_phase == right.failure_phase &&
      left.nonlinear_updates == right.nonlinear_updates &&
      left.linear_iterations == right.linear_iterations &&
      left.final_line_search_trials == right.final_line_search_trials &&
      left.initial_nonlinear_residual == right.initial_nonlinear_residual &&
      left.final_nonlinear_residual == right.final_nonlinear_residual &&
      left.maximum_true_relative_residual == right.maximum_true_relative_residual &&
      left.maximum_conservation_error == right.maximum_conservation_error &&
      left.maximum_conservation_relative_error ==
          right.maximum_conservation_relative_error &&
      left.minimum_density == right.minimum_density &&
      left.minimum_pressure == right.minimum_pressure &&
      left.maximum_wave_speed == right.maximum_wave_speed &&
      left.committed_interior_before == right.committed_interior_before &&
      left.committed_trace_before == right.committed_trace_before &&
      left.committed_interior_after == right.committed_interior_after &&
      left.committed_trace_after == right.committed_trace_after &&
      left.candidate_generation == right.candidate_generation &&
      left.candidate_generation_invalidated ==
          right.candidate_generation_invalidated &&
      left.work.complete_assemblies == right.work.complete_assemblies &&
      left.work.complete_condensations == right.work.complete_condensations &&
      left.work.complete_factor_setups == right.work.complete_factor_setups &&
      left.work.complete_trace_solves == right.work.complete_trace_solves &&
      left.work.complete_recoveries == right.work.complete_recoveries &&
      left.work.trial_reassemblies == right.work.trial_reassemblies &&
      left.work.accepted_newton_updates ==
          right.work.accepted_newton_updates &&
      left.work.phase_entries == right.work.phase_entries &&
      left.work.phase_completions == right.work.phase_completions &&
      left.finite_gate == right.finite_gate &&
      left.admissibility_gate == right.admissibility_gate &&
      left.recovery_gate == right.recovery_gate &&
      left.nonlinear_gate == right.nonlinear_gate &&
      left.true_residual_gate == right.true_residual_gate &&
      left.conservation_gate == right.conservation_gate;
}

bool SameCommittedState(const std::vector<double>& left_interior,
                        const std::vector<double>& left_trace,
                        const std::vector<double>& right_interior,
                        const std::vector<double>& right_trace)
{
  return left_interior == right_interior && left_trace == right_trace;
}

bool CheckPrivateTracedScalarIgrAuditQuadratureNoninterference()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.05, 0.10, 0.12,
                                       0.00, 0.01, 2.55, 2.62}};
  const std::array<double, 8> source{{0.01, 0.02, 0.00, 0.01,
                                      0.00, 0.00, 0.03, 0.04}};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  struct Run final
  {
    std::vector<double> committed_interior;
    std::vector<double> committed_trace;
    dg::hdg::runtime::StateHash interior_hash{};
    dg::hdg::runtime::StateHash trace_hash{};
    dg::hdg::runtime::cpu::CpuAttemptEvidence attempt{};
    dg::hdg::generic::HdgEulerCommittedResidualAudit audit{};
    bool initialized = false;
    bool prepared = false;
    bool committed_step = false;
    bool complete = false;
  };
  const auto run = [&](const int audit_quadrature_order) -> Run
  {
    BoundaryContext boundary{};
    EulerHdgRequest request = Request(mesh, 1, boundary, fields);
    request.numerical_policy.stabilization = {RegularizationKind::igr, 0.2, 0.4};
    request.numerical_policy.audit_quadrature_order = audit_quadrature_order;
    dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
    dg::hdg::runtime::StateCoordinates coordinates{};
    std::uint64_t generation = 0;
    double maximum_wave_speed = 0.0;
    double stable_dt = 0.0;
    Run result{};
    result.initialized = core.InitializeBorrowedEuler(request);
    result.prepared = result.initialized &&
        core.PrepareBorrowedEuler(0.001, coordinates, generation);
    result.committed_step = result.prepared && core.CommitBorrowedEuler(
        generation, coordinates, maximum_wave_speed, stable_dt);
    if (!result.committed_step)
    {
      return result;
    }
    const dg::hdg::runtime::cpu::CpuStateSnapshot snapshot =
        core.Snapshot(coordinates);
    if ((snapshot.interior.size != 0 && snapshot.interior.data == nullptr) ||
        (snapshot.trace.size != 0 && snapshot.trace.data == nullptr))
    {
      return result;
    }
    if (snapshot.interior.size != 0)
    {
      result.committed_interior.assign(snapshot.interior.data,
                                       snapshot.interior.data +
                                           snapshot.interior.size);
    }
    if (snapshot.trace.size != 0)
    {
      result.committed_trace.assign(snapshot.trace.data,
                                    snapshot.trace.data + snapshot.trace.size);
    }
    result.interior_hash = snapshot.interior_hash;
    result.trace_hash = snapshot.trace_hash;
    result.attempt = core.evidence().last_attempt;
    result.audit = core.CommittedBorrowedEulerResidualAudit();
    result.complete = boundary.calls == 0;
    return result;
  };

  constexpr int first_audit_q = 14;
  constexpr int second_audit_q = 18;
  const Run first = run(first_audit_q);
  const Run second = run(second_audit_q);
  const std::uint32_t audit_capability =
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::committed_nonlinear_residual_audit);
  const bool first_audit_valid =
      dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
          first.audit, audit_capability);
  const bool second_audit_valid =
      dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
          second.audit, audit_capability);
  const bool same_state = SameCommittedState(
      first.committed_interior, first.committed_trace,
      second.committed_interior, second.committed_trace);
  const bool same_state_hashes = first.interior_hash == second.interior_hash &&
      first.trace_hash == second.trace_hash;
  const bool same_evidence =
      SamePrimaryIgrAttemptEvidence(first.attempt, second.attempt);
  const bool audit_metadata =
      first.audit.quadrature_order == first_audit_q &&
      second.audit.quadrature_order == second_audit_q &&
      first.audit.formulation ==
          dg::hdg::generic::HdgEulerResidualFormulation::traced_scalar_igr_p1_u7_uhat5 &&
      second.audit.formulation ==
          dg::hdg::generic::HdgEulerResidualFormulation::traced_scalar_igr_p1_u7_uhat5;
  const bool passed = first.complete && second.complete && same_state &&
      same_state_hashes &&
      same_evidence && first_audit_valid && second_audit_valid && audit_metadata;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "igr audit noninterference rank=" << rank
              << " stages=" << first.initialized << ',' << first.prepared
              << ',' << first.committed_step << '/' << second.initialized << ','
              << second.prepared << ',' << second.committed_step
              << " complete=" << first.complete << ',' << second.complete
              << " state=" << same_state << " state_hashes="
              << same_state_hashes << " evidence=" << same_evidence
              << " audit_valid=" << first_audit_valid << ',' << second_audit_valid
              << " audit_metadata=" << audit_metadata << '\n';
    for (std::size_t index = 0;
         index < std::min(first.committed_interior.size(),
                          second.committed_interior.size());
         ++index)
    {
      if (first.committed_interior[index] != second.committed_interior[index])
      {
        std::cerr << "igr audit state mismatch index=" << index
                  << " field=interior first="
                  << first.committed_interior[index] << " second="
                  << second.committed_interior[index] << '\n';
        break;
      }
    }
    for (std::size_t index = 0;
         index < std::min(first.committed_trace.size(),
                          second.committed_trace.size());
         ++index)
    {
      if (first.committed_trace[index] != second.committed_trace[index])
      {
        std::cerr << "igr audit state mismatch index=" << index
                  << " field=trace first=" << first.committed_trace[index]
                  << " second=" << second.committed_trace[index] << '\n';
        break;
      }
    }
    std::cerr << "igr audit evidence first="
              << first.attempt.nonlinear_updates << ','
              << first.attempt.linear_iterations << ','
              << first.attempt.initial_nonlinear_residual << ','
              << first.attempt.final_nonlinear_residual << ','
              << first.attempt.maximum_true_relative_residual << ','
              << first.attempt.maximum_conservation_error << ','
              << first.attempt.maximum_conservation_relative_error << ','
              << first.attempt.work.complete_assemblies << ','
              << first.attempt.work.complete_condensations << ','
              << first.attempt.work.complete_factor_setups << ','
              << first.attempt.work.complete_trace_solves << ','
              << first.attempt.work.complete_recoveries << ','
              << first.attempt.work.trial_reassemblies << ','
              << first.attempt.work.accepted_newton_updates << ','
              << first.attempt.work.phase_entries << ','
              << first.attempt.work.phase_completions << " second="
              << second.attempt.nonlinear_updates << ','
              << second.attempt.linear_iterations << ','
              << second.attempt.initial_nonlinear_residual << ','
              << second.attempt.final_nonlinear_residual << ','
              << second.attempt.maximum_true_relative_residual << ','
              << second.attempt.maximum_conservation_error << ','
              << second.attempt.maximum_conservation_relative_error << ','
              << second.attempt.work.complete_assemblies << ','
              << second.attempt.work.complete_condensations << ','
              << second.attempt.work.complete_factor_setups << ','
              << second.attempt.work.complete_trace_solves << ','
              << second.attempt.work.complete_recoveries << ','
              << second.attempt.work.trial_reassemblies << ','
              << second.attempt.work.accepted_newton_updates << ','
              << second.attempt.work.phase_entries << ','
              << second.attempt.work.phase_completions << '\n';
    std::cerr << "igr audit records first=" << first.audit.available << ','
              << first.audit.quadrature_order << ','
              << static_cast<int>(first.audit.formulation) << ','
              << first.audit.norm << " second=" << second.audit.available << ','
              << second.audit.quadrature_order << ','
              << static_cast<int>(second.audit.formulation) << ','
              << second.audit.norm << '\n';
  }
  return passed;
}

bool CheckPrivateTracedScalarIgrNumericalPolicy()
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
  request.numerical_policy.igr_assembly_quadrature_order = 10;
  request.numerical_policy.maximum_linear_iterations = 31;
  request.numerical_policy.linear_relative_tolerance = 1.0e-12;
  request.numerical_policy.true_residual_relative_tolerance = 1.0e-9;
  request.numerical_policy.conservation_relative_tolerance = 1.0e-8;
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (!core.InitializeBorrowedEuler(request)) return false;
  const bool selected_quadrature =
      core.evidence().assembly_quadrature_order ==
      request.numerical_policy.igr_assembly_quadrature_order;
  dg::hdg::runtime::StateCoordinates coordinates{};
  std::uint64_t generation = 0;
  if (!core.PrepareBorrowedEuler(0.001, coordinates, generation)) return false;
  double maximum_wave_speed = 0.0;
  double stable_dt = 0.0;
  if (!core.CommitBorrowedEuler(generation, coordinates, maximum_wave_speed,
                                stable_dt)) return false;
  const auto& attempt = core.evidence().last_attempt;
  return selected_quadrature && attempt.committed &&
      attempt.linear_iterations > 0 &&
      attempt.linear_iterations <=
          request.numerical_policy.maximum_linear_iterations *
              static_cast<int>(attempt.work.complete_trace_solves) &&
      attempt.maximum_true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      attempt.maximum_conservation_relative_error <=
          request.numerical_policy.conservation_relative_tolerance &&
      attempt.maximum_conservation_relative_error >= 0.0 &&
      boundary.calls == 0;
}

bool CheckPrivateTracedScalarIgrZeroLocal()
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
  const auto audit = session.committed_residual_audit();
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int any_empty = 0;
  const int reduction = MPI_Allreduce(&local_empty, &any_empty, 1, MPI_INT,
                                      MPI_MAX, mesh.GetComm());
  return initialized.status == GenericEulerCpuSessionStatus::initialized &&
      advanced.status == GenericEulerCpuSessionStatus::advanced &&
      reduction == MPI_SUCCESS && any_empty == 1 &&
      dg::hdg::generic::IsValidHdgEulerCommittedResidualAudit(
          audit, session.capabilities()) &&
      audit.formulation ==
          dg::hdg::generic::HdgEulerResidualFormulation::traced_scalar_igr_p1_u7_uhat5 &&
      boundary.calls == 0;
}

bool CheckPrivateTracedScalarIgrInitializationDeterminism()
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
  const bool accepted = ExecuteGenericEulerHdgCpu(
      Request(mesh, 1, boundary, fields));
  // This executes the closed nonperiodic solve.  The production hook observes
  // the actual assembled physical trace rows: one callback value per face
  // quadrature point, C = 0, and a nonzero trace-mass D contribution.
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

bool CheckBorrowedSessionDoesNotUseWorldCollectives()
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
    EulerHdgRequest request = Request(mesh, 1, boundary, fields);
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
  const int order = rank == 0 ? 4 : 2;
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
  const std::size_t local_true_size = static_cast<std::size_t>(
      trace_space.GetTrueVSize());
  int has_empty_true_trace = local_true_size == 0 ? 1 : 0;
  if (MPI_Allreduce(MPI_IN_PLACE, &has_empty_true_trace, 1, MPI_INT, MPI_MAX,
                    mesh.GetComm()) != MPI_SUCCESS ||
      has_empty_true_trace == 0)
  {
    return false;
  }
  std::vector<double> supplied(local_true_size, 0.0);
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
  const bool committed_observables = CheckCommittedObservables();
  const bool committed_observables_zero_owned =
      CheckCommittedObservablesZeroOwnedRank();
  const bool committed_residual_audit = CheckCommittedResidualAuditLifecycle();
  const bool default_numerical_policy =
      CheckDefaultNumericalPolicyAdaptiveLinearCap();
  const bool periodic = CheckPeriodicAnnotationsRemainTraceCouplings();
  const bool private_igr = CheckPrivateTracedScalarIgrLifecycle();
  const bool private_igr_convergence =
      CheckPrivateTracedScalarIgrConvergenceEvidence();
  const bool private_igr_audit_noninterference =
      CheckPrivateTracedScalarIgrAuditQuadratureNoninterference();
  const bool private_igr_numerical_policy =
      CheckPrivateTracedScalarIgrNumericalPolicy();
  const bool private_igr_zero_local = CheckPrivateTracedScalarIgrZeroLocal();
  const bool private_igr_initialization =
      CheckPrivateTracedScalarIgrInitializationDeterminism();
  const bool dynamic_tau = CheckDynamicTauHighWaveTwoSteps();
  const bool boundary_picard = CheckClosedBoundaryPicardAssembly();
  const bool zero_owned = CheckZeroOwnedElementEngine();
  const bool subcommunicator = CheckSubcommunicatorIsolation();
  const bool passive_world = CheckBorrowedSessionDoesNotUseWorldCollectives();
  const bool mismatch = CheckCollectiveMismatchAndAtomicFailure();
  const bool initial_trace_admission = CheckCollectiveInitialTraceAdmission();
  const bool initial_trace_zero_local =
      CheckSuppliedInitialTraceWithZeroLocalExtent();
  const bool callback_failure = CheckAcceptedRequestRankLocalCallbackFailure();
  const bool local = boundary_semantics && trace_restriction && orders &&
      prepared_lifecycle &&
      committed_observables && committed_observables_zero_owned &&
      committed_residual_audit &&
      default_numerical_policy &&
      periodic && private_igr && private_igr_zero_local &&
      private_igr_convergence &&
      private_igr_audit_noninterference &&
      private_igr_numerical_policy &&
      private_igr_initialization && dynamic_tau && boundary_picard &&
      zero_owned && subcommunicator && passive_world && mismatch &&
      initial_trace_admission && initial_trace_zero_local && callback_failure;
  if (!All(local))
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "generic Euler CPU test failed on rank " << rank
              << " boundary_semantics=" << boundary_semantics
              << " trace_restriction=" << trace_restriction
              << " orders=" << orders << " periodic=" << periodic
              << " private_igr=" << private_igr
              << " private_igr_convergence=" << private_igr_convergence
              << " private_igr_audit_noninterference="
              << private_igr_audit_noninterference
              << " private_igr_numerical_policy="
              << private_igr_numerical_policy
              << " private_igr_zero_local=" << private_igr_zero_local
              << " private_igr_initialization=" << private_igr_initialization
              << " prepared_lifecycle=" << prepared_lifecycle
              << " committed_observables=" << committed_observables
              << " committed_observables_zero_owned="
              << committed_observables_zero_owned
              << " committed_residual_audit=" << committed_residual_audit
              << " default_numerical_policy=" << default_numerical_policy
              << " dynamic_tau=" << dynamic_tau
              << " boundary_picard=" << boundary_picard
              << " zero_owned=" << zero_owned
              << " passive_world=" << passive_world
              << " subcommunicator=" << subcommunicator
              << " mismatch=" << mismatch
              << " initial_trace_admission=" << initial_trace_admission
              << " initial_trace_zero_local=" << initial_trace_zero_local
              << " callback_failure=" << callback_failure << '\n';
    return 1;
  }
  return 0;
}
