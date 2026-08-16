#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private_numerics.cuh"

#include <mfem.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <limits>
#include <string_view>
#include <vector>

namespace
{
using Topology = dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics::RetainedTopology;

bool All(const bool local) noexcept
{
  int in = local ? 1 : 0, out = 0;
  return MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS && out == 1;
}

int Decode(const int value) noexcept
{ return value == std::numeric_limits<int>::min() ? -1 : (value < 0 ? -1 - value : value); }

std::unique_ptr<mfem::ParMesh> PeriodicThreeByThree(const bool rank_zero_only)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::QUADRILATERAL,
      false, 1.0, 1.0, true);
  std::vector<mfem::Vector> translation(2, mfem::Vector(2));
  translation[0] = 0.0; translation[1] = 0.0;
  translation[0][0] = 1.0; translation[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(serial,
      serial.CreatePeriodicVertexMapping(translation));
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  std::vector<int> partition(static_cast<std::size_t>(periodic.GetNE()));
  for (int element = 0; element < periodic.GetNE(); ++element)
    partition[static_cast<std::size_t>(element)] = rank_zero_only ? 0 : element % ranks;
  return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, periodic, partition.data());
}

bool ComponentMajorMaps(const Topology& topology)
{
  const auto* interior = topology.InteriorSpace();
  const auto* trace = topology.TraceSpace();
  bool local = topology.IsReady() && interior && trace && interior->GetVDim() == 7 &&
      trace->GetVDim() == 5 && interior->GetOrdering() == mfem::Ordering::byNODES &&
      trace->GetOrdering() == mfem::Ordering::byNODES &&
      topology.SignedElementInteriorVDofs().size() ==
          static_cast<std::size_t>(topology.LocalElementCount()) * 28 &&
      topology.SignedElementTraceVDofs().size() ==
          static_cast<std::size_t>(topology.LocalElementCount()) * 40;
  for (int element = 0; local && element < topology.LocalElementCount(); ++element)
  {
    mfem::Array<int> actual_interior, actual_trace;
    interior->GetElementVDofs(element, actual_interior);
    trace->GetElementVDofs(element, actual_trace);
    local = actual_interior.Size() == 28 && actual_trace.Size() == 40;
    for (int component = 0; local && component < 7; ++component)
      for (int dof = 0; local && dof < 4; ++dof)
      {
        const int slot = component * 4 + dof;
        const int value = topology.SignedElementInteriorVDofs()[static_cast<std::size_t>(element) * 28 + slot];
        const int base = topology.SignedElementInteriorVDofs()[static_cast<std::size_t>(element) * 28 + dof];
        local = value == actual_interior[slot] && Decode(value) >= 0 &&
            Decode(value) == Decode(base) + component * (interior->GetVSize() / 7) &&
            (value < 0) == (base < 0);
      }
    for (int component = 0; local && component < 5; ++component)
      for (int face_dof = 0; local && face_dof < 8; ++face_dof)
      {
        const int slot = component * 8 + face_dof;
        const int value = topology.SignedElementTraceVDofs()[static_cast<std::size_t>(element) * 40 + slot];
        const int base = topology.SignedElementTraceVDofs()[static_cast<std::size_t>(element) * 40 + face_dof];
        local = value == actual_trace[slot] && Decode(value) >= 0 &&
            Decode(value) == Decode(base) + component * (trace->GetVSize() / 5) &&
            (value < 0) == (base < 0);
      }
  }
  return All(local);
}

bool Fixture(const bool rank_zero_only)
{
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  std::unique_ptr<mfem::ParMesh> mesh;
  bool mesh_ready = false;
  try { mesh = PeriodicThreeByThree(rank_zero_only); mesh_ready = mesh != nullptr; }
  catch (...) {}
  if (!All(mesh_ready)) return false;
  Topology topology;
  const bool setup = topology.Setup(*mesh);
  bool local = setup && ComponentMajorMaps(topology) && topology.IdentityPrimed() &&
      topology.GlobalTrueTraceSize() == 180 && std::isfinite(topology.PrimeResidual()) &&
      topology.PrimeResidual() <= 1.0e-12 && topology.PrimeMaximumSolutionError() <= 1.0e-12 &&
      topology.PrimeUsedDeviceIlu0();
  const auto& arena = topology.Arena();
  local = local && arena.integer_physical_bytes > 0 && arena.double_physical_bytes > 0 &&
      arena.interior_map_offset % static_cast<std::int64_t>(alignof(int)) == 0 &&
      arena.trace_map_offset % static_cast<std::int64_t>(alignof(int)) == 0 &&
      arena.vertices_offset % static_cast<std::int64_t>(alignof(double)) == 0;
  long long local_true = topology.LocalTrueTraceSize(), global_true = 0;
  const bool true_sum = MPI_Allreduce(&local_true, &global_true, 1, MPI_LONG_LONG,
                                      MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
  int local_elements = topology.LocalElementCount(), global_elements = 0;
  const bool element_sum = MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT,
                                         MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
  local = local && true_sum && element_sum && global_true == 180 && global_elements == 9;
  if (ranks == 1)
    local = local && topology.LocalElementCount() == 9 && topology.LocalTraceVSize() == 180 &&
        topology.LocalTrueTraceSize() == 180 && topology.InteriorSpace()->GetVSize() == 252;
  else if (rank_zero_only)
    local = local && (rank == 0 ? topology.LocalElementCount() == 9 &&
        topology.LocalTraceVSize() == 180 && topology.LocalTrueTraceSize() == 180 &&
        topology.InteriorSpace()->GetVSize() == 252 : topology.LocalElementCount() == 0 &&
        topology.LocalTraceVSize() == 0 && topology.LocalTrueTraceSize() == 0 &&
        topology.InteriorSpace()->GetVSize() == 0 && arena.integer_logical_bytes == 0 &&
        arena.double_logical_bytes > 0 && topology.SignedElementInteriorVDofs().empty() &&
        topology.SignedElementTraceVDofs().empty());
  else
    local = local && (rank == 0 ? topology.LocalElementCount() == 5 &&
        topology.LocalTraceVSize() == 160 && topology.LocalTrueTraceSize() > 0 &&
        topology.InteriorSpace()->GetVSize() == 140 : topology.LocalElementCount() == 4 &&
        topology.LocalTraceVSize() == 140 && topology.LocalTrueTraceSize() > 0 &&
        topology.InteriorSpace()->GetVSize() == 112);
  const bool second = topology.Setup(*mesh);
  int empty = topology.LocalElementCount() == 0 ? 1 : 0, empty_total = 0;
  const bool empty_count = MPI_Allreduce(&empty, &empty_total, 1, MPI_INT, MPI_SUM,
                                         MPI_COMM_WORLD) == MPI_SUCCESS;
  local = local && (!rank_zero_only || (empty_count && empty_total == 1));
  return All(local && !second && topology.IsReady() && !topology.IsPoisoned());
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  bool pass = (mode.empty() || mode == "--rank0-only") && (ranks == 1 || ranks == 2);
  bool cuda_hypre_ready = false;
  std::unique_ptr<mfem::Device> device;
  try
  {
    device = std::make_unique<mfem::Device>("cuda");
    mfem::Hypre::Init();
    cuda_hypre_ready = mfem::Device::IsConfigured() && mfem::HypreUsingGPU();
  }
  catch (...) { cuda_hypre_ready = false; }
  pass = All(cuda_hypre_ready) && pass;
  if (pass) pass = Fixture(mode == "--rank0-only");
  pass = All(pass);
  if (!pass && rank == 0) std::cerr << "traced-scalar IGR CUDA topology/prime failed\n";
  return pass ? 0 : 1;
}
