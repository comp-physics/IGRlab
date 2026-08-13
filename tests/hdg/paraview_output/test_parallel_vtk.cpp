#include "sim/hdg/parallel_vtk.hpp"

#include <mpi.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

bool Contains(const std::string& text, const std::string& needle)
{
  return text.find(needle) != std::string::npos;
}

std::size_t CountOccurrences(const std::string& text,
                             const std::string& needle)
{
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string::npos)
  {
    ++count;
    offset += needle.size();
  }
  return count;
}

std::string ReadFile(const std::filesystem::path& path)
{
  std::ifstream input(path);
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

bool AllRanks(MPI_Comm communicator, bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator);
  return output == 1;
}

bool ParseOutputArgument(int argc, char* argv[], std::filesystem::path& output)
{
  if (argc != 3 || std::string(argv[1]) != "--output") { return false; }
  output = argv[2];
  return !output.empty();
}

bool Run(const std::filesystem::path& output)
{
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  if (rank == 0) { std::filesystem::remove_all(output); }
  MPI_Barrier(MPI_COMM_WORLD);

  // A publication-only adapter must not import or retain a PVD history from
  // an earlier writer.  Seed an incompatible historical record before every
  // fresh collection save and verify that MFEM replaces it.
  if (rank == 0)
  {
    const std::filesystem::path collection_root = output / "hdg_physical";
    std::filesystem::create_directories(collection_root);
    std::ofstream stale_pvd(collection_root / "hdg_physical.pvd");
    stale_pvd << "stale-history-cycle-000002\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const std::filesystem::path stale_cycle =
      output / "hdg_physical" / "Cycle000003";
  std::filesystem::create_directories(stale_cycle);
  std::ostringstream rank_name;
  rank_name << "proc" << std::setw(6) << std::setfill('0') << rank << ".vtu";
  {
    std::ofstream stale_local(stale_cycle / rank_name.str());
    stale_local << "stale-local-output</VTKFile>\n";
  }
  if (rank == 0)
  {
    std::ofstream stale_pvtu(stale_cycle / "data.pvtu");
    stale_pvtu << "stale-parallel-output</VTKFile>\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, false, 2.0, 1.0, false);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace scalar_space(&mesh, &collection);
  mfem::ParFiniteElementSpace momentum_space(
      &mesh, &collection, 2, mfem::Ordering::byNODES);
  mfem::ParGridFunction density(&scalar_space);
  mfem::ParGridFunction momentum(&momentum_space);
  density = static_cast<double>(10 + rank);
  double* const momentum_values = momentum.HostWrite();
  if (momentum_values == nullptr) { return false; }
  for (int i = 0; i < momentum.Size(); ++i)
  {
    momentum_values[i] = 100.0 * static_cast<double>(rank + 1) + i;
  }

  dg_sim::hdg::HDGParaViewRequest request{
      mesh,
      {{"density", density}, {"momentum", momentum}},
      {output.string(), "hdg_physical", mfem::VTKFormat::ASCII, 1, false}};
  dg_sim::hdg::HDGParaViewSeries publisher(request);
  publisher.Write({3, 20, 1.25});
  publisher.Write({4, 25, 1.5});
  MPI_Barrier(MPI_COMM_WORLD);

  const std::filesystem::path first_cycle =
      output / "hdg_physical" / "Cycle000003";
  const std::filesystem::path second_cycle =
      output / "hdg_physical" / "Cycle000004";
  const std::filesystem::path first_local_vtu = first_cycle /
      (std::string("proc") +
       (rank == 0 ? "000000" : "000001") + ".vtu");
  const std::filesystem::path second_local_vtu = second_cycle /
      (std::string("proc") +
       (rank == 0 ? "000000" : "000001") + ".vtu");
  const std::string local_text = ReadFile(first_local_vtu);
  const std::string local_cells =
      "NumberOfCells=\"" + std::to_string(mesh.GetNE()) + "\"";
  bool passed = std::filesystem::exists(first_local_vtu) &&
      std::filesystem::exists(second_local_vtu) &&
      Contains(local_text, "Name=\"density\"") &&
      Contains(local_text, "Name=\"momentum\"") &&
      Contains(local_text, "NumberOfComponents=\"2\"") &&
      Contains(local_text, local_cells);
  passed = AllRanks(MPI_COMM_WORLD, passed);

  if (rank == 0)
  {
    const std::string pvd = ReadFile(output / "hdg_physical" / "hdg_physical.pvd");
    const std::string first_pvtu = ReadFile(first_cycle / "data.pvtu");
    const std::string second_pvtu = ReadFile(second_cycle / "data.pvtu");
    passed = passed && Contains(pvd, "timestep=\"1.25\"") &&
        Contains(pvd, "timestep=\"1.5\"") &&
        Contains(pvd, "Cycle000003/data.pvtu") &&
        Contains(pvd, "Cycle000004/data.pvtu") &&
        CountOccurrences(pvd, "Cycle000003/data.pvtu") == 1 &&
        CountOccurrences(pvd, "Cycle000004/data.pvtu") == 1 &&
        !Contains(pvd, "stale-history-cycle-000002") &&
        Contains(first_pvtu, "proc000000.vtu") &&
        Contains(second_pvtu, "proc000000.vtu");
    if (ranks == 2)
    {
      passed = passed && Contains(first_pvtu, "proc000001.vtu") &&
          Contains(second_pvtu, "proc000001.vtu");
    }
  }
  return AllRanks(MPI_COMM_WORLD, passed);
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  std::filesystem::path output;
  const bool valid_arguments = ParseOutputArgument(argc, argv, output);
  const bool passed = valid_arguments && Run(output);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!passed && rank == 0)
  {
    std::cerr << "HDG ParaView output adapter test failed\n";
  }
  MPI_Finalize();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
