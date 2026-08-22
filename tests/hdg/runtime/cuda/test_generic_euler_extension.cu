#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::EulerBoundaryPolicy;
using dg::hdg::generic::EulerBoundaryPolicyKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::BorrowedCudaRank1OneGpuRuntimeSelection;
using dg::hdg::runtime::BorrowedCudaRank2OneGpuHostStagedRuntimeSelection;
using dg::hdg::runtime::cuda_w8a::CudaConfig;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaPrecommitSnapshot;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::CudaRunEvidence;
using dg::hdg::runtime::cuda_w8a::CudaStepEvidence;
using dg::hdg::runtime::cuda_w8a::PersistentCudaSmoothEulerEngine;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
             MPI_SUCCESS &&
      output == 1;
}

mfem::ParMesh MakePeriodicMesh()
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakePhysicalMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      3, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_WORLD, serial);
}

struct BoundaryContext
{
  std::array<int, 4> calls{};
  std::array<EulerBoundaryPolicy, 4> policies{};
};

void ConfigurePolicies(BoundaryContext& context, const bool physical)
{
  const std::array<double, components> reference{{1.0, 0.05, -0.02, 2.60}};
  const std::array<double, components> unused{};
  if (!physical)
  {
    for (EulerBoundaryPolicy& policy : context.policies)
    {
      policy = {EulerBoundaryPolicyKind::periodic_topology, unused, 0.0};
    }
    return;
  }
  context.policies = {{
      {EulerBoundaryPolicyKind::slip_wall, unused, 0.0},
      {EulerBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
      {EulerBoundaryPolicyKind::far_field, reference, 0.0},
      {EulerBoundaryPolicyKind::supersonic_inlet, reference, 0.0}}};
}

bool PhysicalEulerExterior(
    const void* opaque, const dg::hdg::generic::BoundaryPointView& point,
    const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(
      const_cast<void*>(opaque));
  if (context == nullptr || point.position == nullptr ||
      point.outward_normal == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 4 ||
      interior == nullptr || exterior == nullptr)
  {
    return false;
  }
  ++context->calls[static_cast<std::size_t>(point.boundary_attribute - 1)];
  std::copy_n(interior, components, exterior);
  switch (point.boundary_attribute)
  {
    case 1:
    {
      // Slip-wall exterior state.
      const double normal_momentum =
          interior[1] * point.outward_normal[0] +
          interior[2] * point.outward_normal[1];
      exterior[1] -= 2.0 * normal_momentum * point.outward_normal[0];
      exterior[2] -= 2.0 * normal_momentum * point.outward_normal[1];
      break;
    }
    case 2:
      // Supersonic outlet: the interior state is the exterior state.
      break;
    case 3:
      // Far-field/nonreflecting representative state.
      exterior[0] = 1.0;
      exterior[1] = 0.05;
      exterior[2] = -0.02;
      exterior[3] = 2.60;
      break;
    case 4:
      // Inlet representative state.
      exterior[0] = 1.0;
      exterior[1] = 0.05;
      exterior[2] = -0.02;
      exterior[3] = 2.60;
      break;
  }
  return true;
}

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> initial_true_trace;
  std::vector<double> source;
  std::size_t volume_scalar_count = 0;
};

enum class SourceMode
{
  empty,
  explicit_zero,
  nonzero
};

BorrowedFields MakeBorrowedFields(
    mfem::ParMesh& mesh, const int order, const SourceMode source_mode)
{
  BorrowedFields fields;
  mfem::L2_FECollection volume_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &volume_collection, components, mfem::Ordering::byNODES);
  fields.volume_scalar_count = static_cast<std::size_t>(
      volume_space.GetVSize() / components);
  fields.initial.resize(static_cast<std::size_t>(volume_space.GetVSize()));
  if (source_mode != SourceMode::empty)
  {
    fields.source.resize(fields.initial.size(), 0.0);
  }
  for (std::size_t scalar = 0; scalar < fields.volume_scalar_count; ++scalar)
  {
    fields.initial[scalar] = 1.0;
    fields.initial[fields.volume_scalar_count + scalar] = 0.05;
    fields.initial[2 * fields.volume_scalar_count + scalar] = -0.02;
    fields.initial[3 * fields.volume_scalar_count + scalar] = 2.60;
    if (source_mode == SourceMode::nonzero)
    {
      fields.source[scalar] = 1.0e-3;
      fields.source[fields.volume_scalar_count + scalar] = 2.0e-4;
      fields.source[2 * fields.volume_scalar_count + scalar] = -1.0e-4;
      fields.source[3 * fields.volume_scalar_count + scalar] = 2.5e-3;
    }
  }

  mfem::DG_Interface_FECollection trace_collection(
      order, 2, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::VectorFunctionCoefficient trace_coefficient(
      components, [](const mfem::Vector&, mfem::Vector& state)
      {
        state.SetSize(components);
        state[0] = 1.0;
        state[1] = 0.05;
        state[2] = -0.02;
        state[3] = 2.60;
      });
  local_trace.ProjectCoefficient(trace_coefficient);
  mfem::Vector true_trace;
  local_trace.GetTrueDofs(true_trace);
  if (true_trace.Size() > 0)
  {
    const double* const trace_values = true_trace.HostRead();
    fields.initial_true_trace.assign(
        trace_values, trace_values + true_trace.Size());
  }
  return fields;
}

EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const int order, BorrowedFields& fields,
    BoundaryContext& boundary, const bool physical,
    const RegularizationKind kind)
{
  ConfigurePolicies(boundary, physical);
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cuda, 0};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.empty() ? nullptr : fields.initial.data(),
      fields.initial.size(), components};
  request.initial_true_trace = {
      fields.initial_true_trace.empty() ? nullptr
                                        : fields.initial_true_trace.data(),
      fields.initial_true_trace.size(), components};
  request.source = {fields.source.empty() ? nullptr : fields.source.data(),
                    fields.source.size(), components};
  request.boundary.context = &boundary;
  request.boundary.exterior_state = PhysicalEulerExterior;
  request.boundary.policies = physical
      ? dg::hdg::generic::EulerBoundaryPolicyTableView{
            boundary.policies.data(), boundary.policies.size(), 2, 1.4}
      : dg::hdg::generic::EulerBoundaryPolicyTableView{
            nullptr, 0, 2, 1.4};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  request.numerical_policy.stabilization = kind == RegularizationKind::igr
      ? dg::hdg::generic::RegularizationSelection{
            RegularizationKind::igr, 0.2, 0.8}
      : dg::hdg::generic::RegularizationSelection{
            RegularizationKind::none, 0.0, 0.0};
  return request;
}

template <typename Selection>
bool CheckIgrAlgebraForSelection(const EulerHdgRequest& request,
                                 const int ranks,
                                 const int order)
{
  CudaConfig config{};
  config.order = order;
  config.mesh_n = 0;
  config.requested_device = 0;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = const_cast<mfem::ParMesh*>(request.mesh);
  config.borrowed_communicator = request.communicator;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {
      request.initial_state.data,
      request.initial_state.scalar_count / static_cast<std::size_t>(components),
      request.initial_state.components};
  config.borrowed_initial_true_trace = {
      request.initial_true_trace.data,
      request.initial_true_trace.scalar_count /
          static_cast<std::size_t>(components),
      request.initial_true_trace.components};
  config.borrowed_source = {
      request.source.data,
      request.source.scalar_count / static_cast<std::size_t>(components),
      request.source.components};
  config.borrowed_boundary = request.boundary;
  config.use_generic_numerical_policy = true;
  config.numerical_policy = request.numerical_policy;

  PersistentCudaSmoothEulerEngine<Selection> engine(config);
  CudaRunEvidence initialization{};
  std::array<double, 6> jacobian_errors{};
  constexpr double jacobian_tolerance = 5.0e-5;
  const bool initialized = engine.Initialize(initialization);
  const bool jacobian = initialized &&
      engine.VerifyBorrowedEulerAssemblerJacobianForTests(
          jacobian_tolerance, jacobian_errors);

  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence step{};
  constexpr double dt = 1.0e-5;
  const bool prepared = jacobian &&
      engine.PrepareStep(false, dt, snapshot, step);
  double conservation_error = std::numeric_limits<double>::infinity();
  double conservation_relative_error =
      std::numeric_limits<double>::infinity();
  const bool conservation = prepared &&
      engine.VerifyPreparedIgrConservationForTests(
          conservation_error, conservation_relative_error) &&
      std::isfinite(conservation_error) &&
      std::isfinite(conservation_relative_error) &&
      conservation_relative_error <=
          config.numerical_policy.conservation_relative_tolerance;

  CudaPreparedOutputMetadataV1 output{};
  const bool staged = conservation &&
      engine.StagePreparedOutputMetadata(output, step);
  const bool acknowledged = staged && engine.MarkOutputStaged(output);
  const bool commit_prepared = acknowledged &&
      engine.PrepareCommit(output, step);
  const bool provisional = commit_prepared &&
      engine.CommitPrepared(output, step);
  const bool committed = provisional && engine.FinalizeCommit(output, step);
  const bool passed = committed && output.successful_commit;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA IGR algebra qualification failed order=" << order
                << " initialized=" << initialized
                << " jacobian=" << jacobian
                << " prepared=" << prepared
                << " conservation=" << conservation
                << " conservation_error=" << conservation_error
                << " conservation_relative_error="
                << conservation_relative_error << " jacobian_errors=";
      for (const double error : jacobian_errors) std::cerr << ' ' << error;
      std::cerr << '\n';
    }
  }
  return All(passed, request.communicator);
}

bool CheckIgrAlgebra(const int order, const int ranks)
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty);
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  return ranks == 1
      ? CheckIgrAlgebraForSelection<
            BorrowedCudaRank1OneGpuRuntimeSelection>(request, ranks, order)
      : CheckIgrAlgebraForSelection<
            BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>(
                request, ranks, order);
}

bool SameValues(const std::vector<double>& left,
                const std::vector<double>& right)
{
  return left == right;
}

bool NearValues(const std::vector<double>& left,
                const std::vector<double>& right)
{
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
  {
    const double scale = std::max({1.0, std::abs(left[i]),
                                   std::abs(right[i])});
    if (!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
        std::abs(left[i] - right[i]) > 1.0e-12 * scale)
    {
      return false;
    }
  }
  return true;
}

bool DifferentValues(const std::vector<double>& left,
                     const std::vector<double>& right)
{
  return left.size() == right.size() &&
      !std::equal(left.begin(), left.end(), right.begin());
}

bool BoundaryCoverage(const BoundaryContext& context,
                      const bool physical,
                      const MPI_Comm communicator)
{
  // CUDA consumes the copied POD table directly; it is not required to call
  // the host callback for physical faces.  Periodic topology, however, must
  // never be reclassified as a physical callback closure.
  if (physical) return true;
  std::array<int, 4> global{};
  if (MPI_Allreduce(context.calls.data(), global.data(),
                    static_cast<int>(global.size()), MPI_INT, MPI_SUM,
                    communicator) != MPI_SUCCESS)
  {
    return false;
  }
  return std::all_of(global.begin(), global.end(),
                     [](const int calls) { return calls == 0; });
}

bool RejectedBeforeMfemDevice(const EulerHdgRequest& request)
{
  const bool unconfigured_before = !mfem::Device::IsConfigured();
  GenericEulerHdgCudaSession session;
  const auto result = session.Initialize(request);
  const auto coordinates = session.coordinates();
  return unconfigured_before &&
      result.status == HdgEulerSessionStatus::rejected &&
      coordinates.step == 0 && coordinates.generation == 0 &&
      coordinates.time == 0.0 && !mfem::Device::IsConfigured();
}

bool CheckContractAdmission(const int order)
{
  mfem::ParMesh mesh = MakePhysicalMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero);
  BoundaryContext boundary{};
  EulerHdgRequest conventional = MakeRequest(
      mesh, order, fields, boundary, true, RegularizationKind::none);
  EulerHdgRequest igr = MakeRequest(
      mesh, order, fields, boundary, true, RegularizationKind::igr);
  EulerHdgRequest invalid_igr = igr;
  invalid_igr.numerical_policy.stabilization.alpha = 0.0;
  EulerHdgRequest null_table = conventional;
  null_table.boundary.policies.policies = nullptr;
  EulerHdgRequest short_table = conventional;
  short_table.boundary.policies.count = boundary.policies.size() - 1;
  EulerHdgRequest wrong_dimension = conventional;
  wrong_dimension.boundary.policies.space_dimension = 3;
  EulerHdgRequest nonfinite_gamma = conventional;
  nonfinite_gamma.boundary.policies.gamma =
      std::numeric_limits<double>::quiet_NaN();

  std::array<EulerBoundaryPolicy, 4> invalid_kind_policies = boundary.policies;
  invalid_kind_policies[0].kind = static_cast<EulerBoundaryPolicyKind>(255);
  EulerHdgRequest invalid_kind = conventional;
  invalid_kind.boundary.policies.policies = invalid_kind_policies.data();

  int ranks = 1;
  int rank = 0;
  MPI_Comm_size(mesh.GetComm(), &ranks);
  MPI_Comm_rank(mesh.GetComm(), &rank);
  std::array<EulerBoundaryPolicy, 4> mismatched_policies = boundary.policies;
  if (ranks == 2 && rank == 0)
  {
    mismatched_policies[0].kind =
        EulerBoundaryPolicyKind::supersonic_outlet;
  }
  EulerHdgRequest mismatched = conventional;
  mismatched.boundary.policies.policies = mismatched_policies.data();

  const bool valid_contracts =
      dg::hdg::generic::ValidateEulerHdgRequest(conventional) ==
          ContractStatus::accepted &&
      dg::hdg::generic::ValidateEulerHdgRequest(igr) ==
          ContractStatus::accepted;
  const bool invalid_igr_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(invalid_igr) !=
          ContractStatus::accepted;
  const bool null_table_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(null_table) !=
          ContractStatus::accepted;
  const bool short_table_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(short_table) !=
          ContractStatus::accepted;
  const bool wrong_dimension_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(wrong_dimension) !=
          ContractStatus::accepted;
  const bool nonfinite_gamma_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(nonfinite_gamma) !=
          ContractStatus::accepted;
  const bool invalid_kind_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(invalid_kind) !=
          ContractStatus::accepted;
  const bool mismatched_contract = ranks != 2 ||
      dg::hdg::generic::ValidateEulerHdgRequest(mismatched) !=
          ContractStatus::accepted;

  // The public session repeats the generic contract before any device or
  // engine allocation.  Each malformed request owns a fresh one-shot facade.
  const bool invalid_igr_session = RejectedBeforeMfemDevice(invalid_igr);
  const bool null_table_session = RejectedBeforeMfemDevice(null_table);
  const bool short_table_session = RejectedBeforeMfemDevice(short_table);
  const bool wrong_dimension_session =
      RejectedBeforeMfemDevice(wrong_dimension);
  const bool nonfinite_gamma_session =
      RejectedBeforeMfemDevice(nonfinite_gamma);
  const bool invalid_kind_session = RejectedBeforeMfemDevice(invalid_kind);
  const bool mismatched_session = ranks != 2 ||
      RejectedBeforeMfemDevice(mismatched);
  return All(valid_contracts && invalid_igr_contract &&
                 null_table_contract && short_table_contract &&
                 wrong_dimension_contract && nonfinite_gamma_contract &&
                 invalid_kind_contract && mismatched_contract &&
                 invalid_igr_session && null_table_session &&
                 short_table_session && wrong_dimension_session &&
                 nonfinite_gamma_session && invalid_kind_session &&
                 mismatched_session,
             mesh.GetComm());
}

bool CheckLifecycle(const int order, const bool physical,
                    const RegularizationKind kind,
                    const SourceMode source_mode,
                    std::vector<double>& committed_result)
{
  mfem::ParMesh mesh = physical ? MakePhysicalMesh() : MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(mesh, order, source_mode);
  const std::vector<double> expected_initial = fields.initial;
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, physical, kind);
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
  {
    return false;
  }

  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(request);
  std::vector<double> committed_initial(expected_initial.size(), -1.0);
  const bool initialized_everywhere = All(
      initialized.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  const bool initial_export = initialized_everywhere &&
      session.ExportCommittedInterior(CudaMutableHostL2Output{
          committed_initial.data(), fields.volume_scalar_count, components});
  if (!All(initial_export && SameValues(committed_initial, expected_initial),
           request.communicator))
  {
    return false;
  }

  // Every borrowed numerical view must have been copied by Initialize.
  const double poisoned = std::numeric_limits<double>::quiet_NaN();
  std::fill(fields.initial.begin(), fields.initial.end(), poisoned);
  std::fill(fields.initial_true_trace.begin(),
            fields.initial_true_trace.end(), poisoned);
  std::fill(fields.source.begin(), fields.source.end(), poisoned);

  constexpr double dt = 1.0e-5;
  GenericEulerCudaPreparedStep first_authority;
  GenericEulerHdgCudaSession::prepared_output_type first_output{};
  const auto before = session.coordinates();
  const auto first_prepared = session.Prepare(
      dt, first_authority, first_output);
  std::vector<double> first_candidate(expected_initial.size(), -2.0);
  const bool first_prepared_everywhere = All(
      first_prepared.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  const bool first_export = first_prepared_everywhere &&
      session.ExportPreparedInterior(
          first_authority, first_output,
          CudaMutableHostL2Output{first_candidate.data(),
                                 fields.volume_scalar_count, components});
  const bool expected_update = source_mode != SourceMode::nonzero ||
      DifferentValues(first_candidate, committed_initial);
  if (!All(first_export && expected_update,
           request.communicator))
  {
    return false;
  }
  const auto aborted = session.Abort(
      std::move(first_authority), first_output);
  std::vector<double> after_abort(expected_initial.size(), -3.0);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted,
      request.communicator);
  const bool abort_export = aborted_everywhere &&
      session.ExportCommittedInterior(CudaMutableHostL2Output{
          after_abort.data(), fields.volume_scalar_count, components});
  if (!All(abort_export && SameValues(after_abort, committed_initial) &&
               session.coordinates().step == before.step &&
               session.coordinates().generation == before.generation &&
               session.coordinates().time == before.time,
           request.communicator))
  {
    return false;
  }

  GenericEulerCudaPreparedStep retry_authority;
  GenericEulerHdgCudaSession::prepared_output_type retry_output{};
  const auto retry_prepared = session.Prepare(
      dt, retry_authority, retry_output);
  std::vector<double> retry_candidate(expected_initial.size(), -4.0);
  const bool retry_prepared_everywhere = All(
      retry_prepared.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  const bool retry_export = retry_prepared_everywhere &&
      session.ExportPreparedInterior(
          retry_authority, retry_output,
          CudaMutableHostL2Output{retry_candidate.data(),
                                 fields.volume_scalar_count, components});
  if (!All(retry_export && NearValues(first_candidate, retry_candidate),
           request.communicator))
  {
    return false;
  }
  const bool acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_output);
  if (!All(acknowledged, request.communicator)) return false;
  const auto committed = session.Commit(
      std::move(retry_authority), retry_output);
  std::vector<double> committed_final(expected_initial.size(), -5.0);
  const bool committed_everywhere = All(
      committed.status == HdgEulerSessionStatus::committed,
      request.communicator);
  const bool final_export = committed_everywhere &&
      session.ExportCommittedInterior(CudaMutableHostL2Output{
          committed_final.data(), fields.volume_scalar_count, components});
  const auto after = session.coordinates();
  const bool boundary_coverage = BoundaryCoverage(
      boundary, physical, request.communicator);
  const bool passed = All(
      final_export && SameValues(committed_final, retry_candidate) &&
          after.step == before.step + 1 &&
          after.generation == before.generation + 1 &&
          after.time == before.time + dt &&
          retry_output.successful_commit && boundary_coverage,
      request.communicator);
  if (passed) committed_result = std::move(committed_final);
  return passed;
}

int ParseOrder(const int argc, char* argv[])
{
  if (argc != 3 || std::string(argv[1]) != "--order") return -1;
  try
  {
    const int order = std::stoi(argv[2]);
    return order >= dg::hdg::generic::minimum_polynomial_order &&
            order <= dg::hdg::generic::maximum_polynomial_order
        ? order
        : -1;
  }
  catch (...) { return -1; }
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int order = ParseOrder(argc, argv);
  const bool admission = order > 0 && CheckContractAdmission(order);
  mfem::Device device("cuda");
  (void)device;
  std::vector<double> conventional_periodic_state;
  std::vector<double> conventional_physical_state;
  std::vector<double> igr_periodic_state;
  std::vector<double> igr_physical_state;
  std::vector<double> conventional_empty_source_state;
  std::vector<double> conventional_zero_source_state;
  std::vector<double> igr_empty_source_state;
  std::vector<double> igr_zero_source_state;
  const bool igr_algebra = order > 0 && CheckIgrAlgebra(order, ranks);
  const bool conventional_periodic = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::nonzero,
      conventional_periodic_state);
  const bool conventional_physical = order > 0 && CheckLifecycle(
      order, true, RegularizationKind::none, SourceMode::nonzero,
      conventional_physical_state);
  const bool igr_periodic = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::nonzero,
      igr_periodic_state);
  const bool igr_physical = order > 0 && CheckLifecycle(
      order, true, RegularizationKind::igr, SourceMode::nonzero,
      igr_physical_state);
  const bool conventional_empty_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::empty,
      conventional_empty_source_state);
  const bool conventional_zero_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::explicit_zero,
      conventional_zero_source_state);
  const bool igr_empty_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::empty,
      igr_empty_source_state);
  const bool igr_zero_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::explicit_zero,
      igr_zero_source_state);
  const bool source_equivalence = conventional_empty_source &&
      conventional_zero_source && igr_empty_source && igr_zero_source &&
      NearValues(conventional_empty_source_state,
                 conventional_zero_source_state) &&
      NearValues(igr_empty_source_state, igr_zero_source_state);
  const bool boundary_effect = conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical &&
      DifferentValues(conventional_periodic_state,
                      conventional_physical_state) &&
      DifferentValues(igr_periodic_state, igr_physical_state);
  const bool passed = (ranks == 1 || ranks == 2) && All(
      admission && igr_algebra &&
      conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical && boundary_effect && source_equivalence);
  if (!passed)
  {
    std::cerr << "generic CUDA Euler-extension qualification failed rank="
              << rank << " ranks=" << ranks << " order=" << order
              << " admission=" << admission
              << " igr_algebra=" << igr_algebra
              << " conventional_periodic=" << conventional_periodic
              << " conventional_physical=" << conventional_physical
              << " igr_periodic=" << igr_periodic
              << " igr_physical=" << igr_physical
              << " boundary_effect=" << boundary_effect
              << " source_equivalence=" << source_equivalence << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
