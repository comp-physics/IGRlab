#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/hdg/state_io.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

// Test-only friend access keeps engine-selection evidence attached to the
// existing subcommunicator lifecycle instead of adding a production API.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool VerifyPreparedCudaEngine(
      Engine& engine, const generic::CudaEngineKind expected) noexcept
  {
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging)
      return false;
    const auto& evidence = engine.pending_.evidence;
    const auto& tensor = evidence.tensor_core;
    return tensor.selected_engine == expected &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner ==
            CudaTracePreconditionerForEngine(expected) &&
        tensor.fallback_count == 0 && evidence.no_fallback &&
        (expected == generic::CudaEngineKind::ordinary
             ? tensor.condensation_launches == 0 && tensor.mma_tiles == 0
             : tensor.hardware_supported &&
                   tensor.condensation_launches > 0 && tensor.mma_tiles > 0);
  }

  static bool VerifyPreparedCudaEngine(
      cuda::GenericEulerHdgCudaSession& session,
      const generic::CudaEngineKind expected) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyPreparedCudaEngine(*session.borrowed_rank1_, expected);
    if (session.borrowed_rank2_)
      return VerifyPreparedCudaEngine(*session.borrowed_rank2_, expected);
    return false;
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

bool All(const MPI_Comm communicator, const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      output == 1;
}

bool ExteriorState(const void*,
                   const dg::hdg::generic::BoundaryPointView&,
                   const double* interior, double* exterior) noexcept
{
  if (interior == nullptr || exterior == nullptr) return false;
  for (int component = 0; component < components; ++component)
    exterior[component] = interior[component];
  return true;
}

mfem::Mesh PeriodicMesh()
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  return mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
}

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> initial_true_trace;
};

BorrowedFields MakeFields(mfem::ParMesh& mesh)
{
  constexpr int order = 1;
  BorrowedFields fields;
  mfem::L2_FECollection volume_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &volume_collection, components, mfem::Ordering::byNODES);
  const std::size_t volume_scalars = static_cast<std::size_t>(
      volume_space.GetVSize() / components);
  fields.initial.resize(static_cast<std::size_t>(volume_space.GetVSize()));
  for (std::size_t scalar = 0; scalar < volume_scalars; ++scalar)
  {
    fields.initial[scalar] = 1.0;
    fields.initial[volume_scalars + scalar] = 0.05;
    fields.initial[2 * volume_scalars + scalar] = -0.02;
    fields.initial[3 * volume_scalars + scalar] = 2.60;
  }

  mfem::DG_Interface_FECollection trace_collection(
      order, 2, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::VectorFunctionCoefficient coefficient(
      components, [](const mfem::Vector&, mfem::Vector& state)
      {
        state.SetSize(components);
        state[0] = 1.0;
        state[1] = 0.05;
        state[2] = -0.02;
        state[3] = 2.60;
      });
  mfem::Vector true_trace;
  const auto policy = dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  if (!dg_sim::hdg::ProjectEulerTrueTraceCoefficientL2(
          trace_space, local_trace, true_trace, coefficient,
          policy.true_trace_projection_quadrature_order))
  {
    return {};
  }
  if (true_trace.Size() > 0)
  {
    const double* const values = true_trace.HostRead();
    fields.initial_true_trace.assign(values, values + true_trace.Size());
  }
  return fields;
}

EulerHdgRequest MakeRequest(const mfem::ParMesh& mesh,
                            BorrowedFields& fields,
                            const CudaEngineKind engine)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = 1;
  request.backend = {BackendKind::cuda, 0, engine};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.empty() ? nullptr : fields.initial.data(),
      fields.initial.size(), components};
  request.initial_true_trace = {
      fields.initial_true_trace.empty() ? nullptr
                                        : fields.initial_true_trace.data(),
      fields.initial_true_trace.size(), components};
  request.source = {nullptr, 0, components};
  request.boundary = {
      nullptr, ExteriorState, {nullptr, 0, 2, 1.4}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(1);
  request.numerical_policy.stabilization = {
      RegularizationKind::igr, 0.2, 0.8};
  return request;
}

bool CheckActiveSubcommunicator(const MPI_Comm communicator,
                                const int active_ranks,
                                const CudaEngineKind engine)
{
  int rank = -1;
  int ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      ranks != active_ranks)
    return false;

  mfem::Mesh serial = PeriodicMesh();
  mfem::ParMesh mesh(communicator, serial);
  BorrowedFields fields = MakeFields(mesh);
  if (!All(communicator, HYPRE_GetError() == 0)) return false;
  EulerHdgRequest request = MakeRequest(mesh, fields, engine);

  if (ranks == 2)
  {
    GenericEulerHdgCudaSession rejected_session;
    EulerHdgRequest asymmetric = request;
    if (rank == 0) asymmetric.mesh = nullptr;
    const bool rejected = rejected_session.Initialize(asymmetric).status ==
        HdgEulerSessionStatus::rejected;
    if (!All(communicator, rejected)) return false;
  }

  GenericEulerHdgCudaSession session;
  const bool initialized = session.Initialize(request).status ==
      HdgEulerSessionStatus::initialized;
  if (!All(communicator, initialized)) return false;

  GenericEulerCudaPreparedStep authority;
  CudaPreparedOutputMetadataV1 output{};
  const bool prepared = session.Prepare(0.025, authority, output).status ==
      HdgEulerSessionStatus::prepared;
  if (!All(communicator, prepared)) return false;
  const bool engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(session, engine);
  if (!All(communicator, engine_evidence)) return false;
  const bool acknowledged = session.AcknowledgePreparedOutput(authority, output);
  if (!All(communicator, acknowledged)) return false;
  const bool committed = session.Commit(std::move(authority), output).status ==
      HdgEulerSessionStatus::committed;
  return All(communicator, committed && session.coordinates().step == 1);
}

struct Options
{
  int active_ranks = 0;
  CudaEngineKind engine = CudaEngineKind::ordinary;
};

bool ParseOptions(const int argc, char* argv[], Options& options) noexcept
{
  for (int index = 1; index < argc; ++index)
  {
    const std::string_view argument(argv[index]);
    if (argument == "--active-ranks")
    {
      if (++index >= argc) return false;
      options.active_ranks = std::atoi(argv[index]);
      continue;
    }
    if (argument == "--engine")
    {
      if (++index >= argc) return false;
      const std::string_view engine(argv[index]);
      if (engine == "ordinary")
        options.engine = CudaEngineKind::ordinary;
      else if (engine == "tensor")
        options.engine = CudaEngineKind::tensor_core;
      else
        return false;
      continue;
    }
    return false;
  }
  return options.active_ranks == 1 || options.active_ranks == 2;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  Options options{};
  const bool options_ready = ParseOptions(argc, argv, options);
  const int active_ranks = options.active_ranks;

  int world_rank = -1;
  int world_ranks = 0;
  const bool world_ready =
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS &&
      MPI_Comm_size(MPI_COMM_WORLD, &world_ranks) == MPI_SUCCESS;
  bool local = options_ready && world_ready &&
      world_ranks == active_ranks + 1;

  MPI_Comm active = MPI_COMM_NULL;
  if (local)
  {
    const int color = world_rank < active_ranks ? 0 : MPI_UNDEFINED;
    local = MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active) ==
        MPI_SUCCESS;
  }
  if (local && active != MPI_COMM_NULL)
  {
    try
    {
      local = CheckActiveSubcommunicator(
          active, active_ranks, options.engine);
    }
    catch (...) { local = false; }
  }
  if (active != MPI_COMM_NULL)
    local = MPI_Comm_free(&active) == MPI_SUCCESS && local;

  const int input = local ? 1 : 0;
  int output = 0;
  const bool reduced = MPI_Allreduce(
      &input, &output, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS;
  return reduced && output == 1 ? 0 : 1;
}
