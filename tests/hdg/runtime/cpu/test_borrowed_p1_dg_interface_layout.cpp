#include "numerics/hdg/runtime/cpu/borrowed_p1_dg_interface_layout.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <cmath>
#include <limits>
#include <vector>

namespace
{

using Layout = dg::hdg::runtime::cpu::BorrowedP1DGInterfaceLayout;

int DecodeSignedVDof(const int signed_vdof)
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

int SignedVDofSign(const int signed_vdof)
{
  return signed_vdof < 0 ? -1 : 1;
}

bool AllRanks(const MPI_Comm communicator, const bool local)
{
  const int value = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

mfem::ParMesh MakeMesh(const MPI_Comm communicator)
{
  int ranks = 1;
  MPI_Comm_size(communicator, &ranks);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      6, 2, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()));
  for (int element = 0; element < serial.GetNE(); ++element)
  {
    partition[static_cast<std::size_t>(element)] = element % ranks;
  }
  return mfem::ParMesh(communicator, serial, partition.data());
}

mfem::ParMesh MakeRankZeroOnlyMesh(const MPI_Comm communicator)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  const std::vector<int> partition(1, 0);
  return mfem::ParMesh(communicator, serial, partition.data());
}

double TrueValue(const long long global_true_dof)
{
  return 0.25 + 17.0 * static_cast<double>(global_true_dof);
}

bool FillOwnedTrueValues(const mfem::ParFiniteElementSpace& space,
                         mfem::Vector& values)
{
  if (values.Size() != space.GetTrueVSize()) { return false; }
  for (int local = 0; local < space.GetVSize(); ++local)
  {
    const int local_true = space.GetLocalTDofNumber(local);
    if (local_true < 0) { continue; }
    if (local_true >= values.Size()) { return false; }
    const long long global = static_cast<long long>(
        space.GetGlobalTDofNumber(local));
    if (global < 0) { return false; }
    values[local_true] = TrueValue(global);
  }
  return true;
}

bool CheckAnalyticDistributionAndTraceGather(const MPI_Comm communicator)
{
  mfem::ParMesh mesh = MakeMesh(communicator);
  Layout layout(mesh);
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  bool local = layout.IsUsable() && trace_space != nullptr &&
      layout.TraceTrueVSize() == trace_space->GetTrueVSize() &&
      layout.TraceLocalVSize() == trace_space->GetVSize() &&
      layout.LocalElementCount() == mesh.GetNE();
  if (!AllRanks(communicator, local)) { return false; }

  mfem::Vector true_values(layout.TraceTrueVSize());
  true_values = -1.0;
  local = FillOwnedTrueValues(*trace_space, true_values);
  mfem::Vector distributed(layout.TraceLocalVSize());
  distributed = -3.0;
  local = layout.DistributeTrace(true_values, distributed) && local;
  for (int local_dof = 0; local_dof < distributed.Size(); ++local_dof)
  {
    const long long global = static_cast<long long>(
        trace_space->GetGlobalTDofNumber(local_dof));
    local = global >= 0 &&
        std::abs(distributed[local_dof] - TrueValue(global)) < 1.0e-12 &&
        local;
  }
  if (!AllRanks(communicator, local)) { return false; }

  for (int local_dof = 0; local_dof < distributed.Size(); ++local_dof)
  {
    distributed[local_dof] = 1000.0 + static_cast<double>(local_dof);
  }
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    mfem::Array<int> expected_vdofs;
    trace_space->GetElementVDofs(element, expected_vdofs);
    mfem::Vector gathered(Layout::trace_values_per_element);
    local = expected_vdofs.Size() == Layout::trace_values_per_element &&
        layout.GatherTraceElement(element, distributed, gathered) && local;
    const int* const expected = expected_vdofs.HostRead();
    if (expected == nullptr) { local = false; }
    for (int index = 0; expected != nullptr && index < expected_vdofs.Size();
         ++index)
    {
      const int vdof = DecodeSignedVDof(expected[index]);
      local = vdof >= 0 && vdof < distributed.Size() &&
          gathered[index] == SignedVDofSign(expected[index]) *
              distributed[vdof] && local;
    }
  }
  return AllRanks(communicator, local);
}

bool CheckInteriorRoundTripAndNegatives(const MPI_Comm communicator)
{
  mfem::ParMesh mesh = MakeMesh(communicator);
  Layout layout(mesh);
  const mfem::ParFiniteElementSpace* const interior_space =
      layout.InteriorSpace();
  bool local = layout.IsUsable() && interior_space != nullptr &&
      layout.InteriorLocalVSize() == interior_space->GetVSize();
  if (!AllRanks(communicator, local)) { return false; }

  mfem::Vector interior(layout.InteriorLocalVSize());
  mfem::Vector recovered(layout.InteriorLocalVSize());
  for (int dof = 0; dof < interior.Size(); ++dof)
  {
    interior[dof] = -20.0 + 0.5 * static_cast<double>(dof);
  }
  recovered = std::numeric_limits<double>::quiet_NaN();
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    mfem::Vector element_values(Layout::interior_values_per_element);
    local = layout.GatherInteriorElement(element, interior, element_values) &&
        layout.ScatterInteriorElement(element, element_values, recovered) &&
        local;
  }
  for (int dof = 0; dof < interior.Size(); ++dof)
  {
    local = recovered[dof] == interior[dof] && local;
  }

  mfem::Vector trace_wrong_size(layout.TraceLocalVSize() + 1);
  mfem::Vector trace_element(Layout::trace_values_per_element);
  mfem::Vector interior_wrong_size(layout.InteriorLocalVSize() + 1);
  mfem::Vector interior_element(Layout::interior_values_per_element);
  const int invalid_element = mesh.GetNE();
  local = !layout.GatherTraceElement(0, trace_wrong_size, trace_element) &&
      !layout.GatherTraceElement(invalid_element, trace_wrong_size,
                                 trace_element) &&
      !layout.GatherInteriorElement(0, interior_wrong_size, interior_element) &&
      !layout.ScatterInteriorElement(invalid_element, interior_element,
                                     recovered) && local;
  return AllRanks(communicator, local);
}

bool CheckZeroLocalParticipation(const MPI_Comm communicator)
{
  int ranks = 1;
  MPI_Comm_size(communicator, &ranks);
  mfem::ParMesh mesh = MakeRankZeroOnlyMesh(communicator);
  Layout layout(mesh);
  bool local = layout.IsUsable();
  if (!AllRanks(communicator, local)) { return false; }
  if (mesh.GetNE() == 0)
  {
    local = layout.LocalElementCount() == 0 && layout.TraceTrueVSize() == 0 &&
        layout.TraceLocalVSize() == 0 && layout.InteriorLocalVSize() == 0 &&
        local;
  }
  mfem::Vector owned(layout.TraceTrueVSize());
  mfem::Vector distributed(layout.TraceLocalVSize());
  owned = 1.0;
  distributed = -1.0;
  local = layout.DistributeTrace(owned, distributed) && local;
  for (int dof = 0; dof < distributed.Size(); ++dof)
  {
    local = distributed[dof] == 1.0 && local;
  }
  return (ranks == 1 || mesh.GetNE() == 0 || mesh.GetNE() == 1) &&
      AllRanks(communicator, local);
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session session(argc, argv);
  const MPI_Comm communicator = MPI_COMM_WORLD;
  const bool local = CheckAnalyticDistributionAndTraceGather(communicator) &&
      CheckInteriorRoundTripAndNegatives(communicator) &&
      CheckZeroLocalParticipation(communicator);
  return AllRanks(communicator, local) ? 0 : 1;
}
