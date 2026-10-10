#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

#include <mfem.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;

struct ModelTag { inline static constexpr std::string_view id = "euler.compressible.2d"; };
struct FormulationTag { inline static constexpr std::string_view id = "hdg.conventional"; };
struct TraceTag { inline static constexpr std::string_view id = "mfem.dg_interface.trace"; };
struct StabilizationTag { inline static constexpr std::string_view id = "tau.fixed"; };
struct TimeTag { inline static constexpr std::string_view id = "backward_euler"; };
struct RankOneTag { inline static constexpr std::string_view id = "cuda.rank1.one_gpu"; };
struct RankTwoTag { inline static constexpr std::string_view id = "cuda.rank2.one_gpu.host_staged"; };
struct PrecisionTag { inline static constexpr std::string_view id = "fp64"; };

template <typename Backend>
using Selection = StaticRunSelection<ModelTag, FormulationTag, TraceTag,
    StabilizationTag, TimeTag, Backend, PrecisionTag>;
template <typename Backend>
using Engine = PersistentCudaSmoothEulerEngine<Selection<Backend>,
    UmpireDeviceAllocatorPolicy, CudaTransportPolicy, CudaMpiCollectivePolicy>;

bool All(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

mfem::Mesh PeriodicSquareMesh(const int n)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  return mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
}

mfem::Mesh NonperiodicSquareMesh(const int n)
{
  return mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
}

CudaConfig BorrowedConfig(mfem::ParMesh& mesh, const int ranks,
                          const int order, const int n) noexcept
{
  CudaConfig config{};
  config.order = order;
  config.mesh_n = n;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = &mesh;
  config.borrowed_communicator = MPI_COMM_WORLD;
  return config;
}

template <typename Backend>
bool CheckBorrowedTopology(const int ranks, const int order, const int n)
{
  mfem::Mesh serial = PeriodicSquareMesh(n);
  // This is intentionally MFEM's default ParMesh partitioning.  The CUDA
  // engine receives only the completed external ParMesh and must not supply a
  // partition vector of its own.
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  CudaConfig config = BorrowedConfig(mesh, ranks, order, n);
  auto engine = std::make_unique<Engine<Backend>>(config);
  CudaRunEvidence evidence{};
  const bool initialized = engine->Initialize(evidence);
  const bool borrowed = initialized && evidence.borrowed_topology &&
      config.borrowed_mesh == &mesh && mesh.GetNE() > 0 &&
      evidence.status == CudaRunStatus::ready &&
      evidence.ranks == ranks && evidence.mfem_device_valid &&
      evidence.hypre_cuda_umpire_device && evidence.same_physical_gpu &&
      evidence.no_host_solver_fallback &&
      (ranks == 1 || evidence.host_staged_rank2);
  return All(borrowed);
}

template <typename Backend>
bool RejectedBeforeDevice(const CudaConfig& config) noexcept
{
  auto engine = std::make_unique<Engine<Backend>>(config);
  CudaRunEvidence evidence{};
  return !engine->Initialize(evidence) && evidence.device == -1 &&
      evidence.status == CudaRunStatus::setup_failure;
}

template <typename Backend>
bool CheckNegatives(const int ranks)
{
  mfem::Mesh valid_serial = PeriodicSquareMesh(functional_elements_per_direction);
  mfem::ParMesh valid(MPI_COMM_WORLD, valid_serial);
  CudaConfig unsupported_order = BorrowedConfig(
      valid, ranks, dg::hdg::generic::maximum_polynomial_order + 1,
      functional_elements_per_direction);
  CudaConfig rank3 = BorrowedConfig(valid, 3, functional_polynomial_order,
                                    functional_elements_per_direction);
  CudaConfig null_communicator = BorrowedConfig(
      valid, ranks, functional_polynomial_order,
      functional_elements_per_direction);
  null_communicator.borrowed_communicator = MPI_COMM_NULL;
  bool mismatched_communicator_rejected = true;
  MPI_Comm mismatched_communicator = MPI_COMM_NULL;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (ranks == 2 &&
      MPI_Comm_split(MPI_COMM_WORLD, rank /* one-rank subcommunicators */, 0,
                     &mismatched_communicator) == MPI_SUCCESS)
  {
    CudaConfig mismatched = BorrowedConfig(
        valid, ranks, functional_polynomial_order,
        functional_elements_per_direction);
    mismatched.borrowed_communicator = mismatched_communicator;
    mismatched_communicator_rejected =
        RejectedBeforeDevice<Backend>(mismatched);
    MPI_Comm_free(&mismatched_communicator);
  }

  mfem::Mesh oversized_serial = PeriodicSquareMesh(7);
  mfem::ParMesh oversized(MPI_COMM_WORLD, oversized_serial);
  CudaConfig capacity = BorrowedConfig(oversized, ranks,
                                       functional_polynomial_order,
                                       functional_elements_per_direction);

  // A bare borrowed topology has no copied physical-boundary policy table.
  // Admission must reject this true-boundary mesh before cudaSetDevice rather
  // than treating physical faces as periodic trace couplings.
  mfem::Mesh boundary_serial =
      NonperiodicSquareMesh(functional_elements_per_direction);
  mfem::ParMesh boundary_trace(MPI_COMM_WORLD, boundary_serial);
  CudaConfig boundary_capacity = BorrowedConfig(
      boundary_trace, ranks, functional_polynomial_order,
      functional_elements_per_direction);

  mfem::Mesh triangles = mfem::Mesh::MakeCartesian2D(
      smoke_elements_per_direction, smoke_elements_per_direction,
      mfem::Element::TRIANGLE, false, 1.0, 1.0, false);
  mfem::ParMesh nonquad(MPI_COMM_WORLD, triangles);
  CudaConfig geometry = BorrowedConfig(nonquad, ranks, smoke_polynomial_order,
                                       smoke_elements_per_direction);
  // Each rejection enters collective preflight.  Invoke every case in a fixed
  // rank-independent order before combining the local results.
  const bool unsupported_order_rejected =
      RejectedBeforeDevice<Backend>(unsupported_order);
  const bool rank3_rejected = RejectedBeforeDevice<Backend>(rank3);
  const bool null_communicator_rejected =
      RejectedBeforeDevice<Backend>(null_communicator);
  const bool capacity_rejected = RejectedBeforeDevice<Backend>(capacity);
  const bool boundary_capacity_rejected =
      RejectedBeforeDevice<Backend>(boundary_capacity);
  const bool geometry_rejected = RejectedBeforeDevice<Backend>(geometry);
  return All(unsupported_order_rejected && rank3_rejected &&
             null_communicator_rejected &&
             mismatched_communicator_rejected && capacity_rejected &&
             boundary_capacity_rejected && geometry_rejected);
}
}  // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool known_mode = mode.empty() || mode == "--negatives";
  bool pass = false;
  if (known_mode && ranks == 1)
  {
    if (mode == "--negatives")
      pass = CheckNegatives<RankOneTag>(ranks);
    else
    {
      const bool smoke = CheckBorrowedTopology<RankOneTag>(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      const bool functional = CheckBorrowedTopology<RankOneTag>(
          ranks, functional_polynomial_order,
          functional_elements_per_direction);
      pass = smoke && functional;
    }
  }
  else if (known_mode && ranks == 2)
  {
    if (mode == "--negatives")
      pass = CheckNegatives<RankTwoTag>(ranks);
    else
    {
      const bool smoke = CheckBorrowedTopology<RankTwoTag>(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      const bool functional = CheckBorrowedTopology<RankTwoTag>(
          ranks, functional_polynomial_order,
          functional_elements_per_direction);
      pass = smoke && functional;
    }
  }
  if (!pass && rank == 0)
    std::cerr << "CUDA generic external-topology audit failed\n";
  return pass ? 0 : 1;
}
