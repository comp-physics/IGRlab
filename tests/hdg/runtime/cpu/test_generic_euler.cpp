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

bool CheckPeriodicAnnotationsRemainTraceCouplings()
{
  mfem::ParMesh mesh = MakePeriodicBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                       0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{};
  BoundaryContext context{};
  ExactBorrowedEulerFields fields(mesh, 1, initial, source);
  EulerHdgRequest request = Request(mesh, 1, context, fields);
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
  const bool changed = candidate != before;
  const auto aborted = prepared.status == GenericEulerCpuSessionStatus::prepared
      ? session.Abort(std::move(first)) : decltype(session.Abort(std::move(first))){ };
  std::vector<double> after_abort(before.size(), 0.0);
  const bool rollback = session.ExportCommittedInterior({after_abort.data(), after_abort.size()}) &&
      after_abort == before;
  GenericEulerCpuPreparedStep retry;
  const auto retried = session.Prepare(0.001, retry);
  const auto committed = retried.status == GenericEulerCpuSessionStatus::prepared
      ? session.Commit(std::move(retry)) : decltype(session.Commit(std::move(retry))){ };
  const auto second = session.Advance(0.001);
  return exported_before && exported_candidate && changed &&
      aborted.status == GenericEulerCpuSessionStatus::aborted && rollback &&
      committed.status == GenericEulerCpuSessionStatus::committed &&
      second.status == GenericEulerCpuSessionStatus::advanced && boundary.calls == 0;
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
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int any_empty = 0;
  const int reduction = MPI_Allreduce(&local_empty, &any_empty, 1, MPI_INT,
                                      MPI_MAX, mesh.GetComm());
  return initialized.status == GenericEulerCpuSessionStatus::initialized &&
      advanced.status == GenericEulerCpuSessionStatus::advanced &&
      reduction == MPI_SUCCESS && any_empty == 1 && boundary.calls == 0;
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
  const bool periodic = CheckPeriodicAnnotationsRemainTraceCouplings();
  const bool private_igr = CheckPrivateTracedScalarIgrLifecycle();
  const bool private_igr_convergence =
      CheckPrivateTracedScalarIgrConvergenceEvidence();
  const bool private_igr_zero_local = CheckPrivateTracedScalarIgrZeroLocal();
  const bool private_igr_initialization =
      CheckPrivateTracedScalarIgrInitializationDeterminism();
  const bool dynamic_tau = CheckDynamicTauHighWaveTwoSteps();
  const bool boundary_picard = CheckClosedBoundaryPicardAssembly();
  const bool zero_owned = CheckZeroOwnedElementEngine();
  const bool subcommunicator = CheckSubcommunicatorIsolation();
  const bool passive_world = CheckBorrowedSessionDoesNotUseWorldCollectives();
  const bool mismatch = CheckCollectiveMismatchAndAtomicFailure();
  const bool callback_failure = CheckAcceptedRequestRankLocalCallbackFailure();
  const bool local = boundary_semantics && trace_restriction && orders &&
      prepared_lifecycle &&
      periodic && private_igr && private_igr_zero_local &&
      private_igr_convergence &&
      private_igr_initialization && dynamic_tau && boundary_picard &&
      zero_owned && subcommunicator && passive_world && mismatch &&
      callback_failure;
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
              << " private_igr_zero_local=" << private_igr_zero_local
              << " private_igr_initialization=" << private_igr_initialization
              << " prepared_lifecycle=" << prepared_lifecycle
              << " dynamic_tau=" << dynamic_tau
              << " boundary_picard=" << boundary_picard
              << " zero_owned=" << zero_owned
              << " passive_world=" << passive_world
              << " subcommunicator=" << subcommunicator
              << " mismatch=" << mismatch
              << " callback_failure=" << callback_failure << '\n';
    return 1;
  }
  return 0;
}
