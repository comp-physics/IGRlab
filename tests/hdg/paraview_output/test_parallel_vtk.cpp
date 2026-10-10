#include "sim/hdg/euler_output_fields.hpp"
#include "sim/hdg/parallel_vtk.hpp"

#include <mpi.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

bool Contains(const std::string& text, const std::string& needle)
{
  return text.find(needle) != std::string::npos;
}

mfem::ParGridFunction* FindField(
    std::vector<dg_sim::hdg::HDGInteriorFieldView>& fields,
    const std::string& name)
{
  for (auto& field : fields)
  {
    if (field.name == name) { return &field.values; }
  }
  return nullptr;
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
      output / "hdg_physical" / "Cycle000000";
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

  // Keep one globally owned cell.  In the rank-two gate this deliberately
  // leaves rank one empty and exercises the standard Euler field views and
  // writer without assuming that a zero-size MFEM Vector has non-null data.
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  const std::vector<int> partition(
      static_cast<std::size_t>(serial_mesh.GetNE()), 0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh, partition.data());
  mfem::L2_FECollection collection(1, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace conservative_space(
      &mesh, &collection, 4, mfem::Ordering::byNODES);
  mfem::ParGridFunction conservative(&conservative_space);
  const int scalar_count = conservative.Size() / 4;
  double* const conservative_values = conservative.HostWrite();
  if (scalar_count != 0 && conservative_values == nullptr) { return false; }
  for (int i = 0; i < scalar_count; ++i)
  {
    conservative_values[i] = 1.0 + 0.01 * rank;
    conservative_values[scalar_count + i] = 0.1;
    conservative_values[2 * scalar_count + i] = -0.05;
    conservative_values[3 * scalar_count + i] = 2.5;
  }
  dg_sim::hdg::HdgEulerParaViewFields output_fields(
      mesh, collection, conservative, /*igr_active=*/true,
      /*enable_zhang_shu_diagnostics=*/true,
      /*gamma=*/1.4, /*density_floor=*/1.0e-16);
  dg::hdg::generic::HdgEulerHostInteriorFields destination{};
  if (!output_fields.FreshHostOutput(destination) ||
      destination.scalar_count != static_cast<std::size_t>(scalar_count) ||
      destination.zhang_shu_element_count !=
          static_cast<std::size_t>(mesh.GetNE()) ||
      (scalar_count != 0 && (destination.conservative_coefficients == nullptr ||
                            destination.sigma_coefficients == nullptr)) ||
      (mesh.GetNE() != 0 &&
       (destination.zhang_shu_theta_rho == nullptr ||
        destination.zhang_shu_theta_p == nullptr)))
  {
    return false;
  }
  for (int i = 0; i < scalar_count; ++i)
  {
    destination.sigma_coefficients[i] = 0.25;
  }
  for (int element = 0; element < mesh.GetNE(); ++element)
  {
    destination.zhang_shu_theta_rho[element] = 0.25;
    destination.zhang_shu_theta_p[element] = 0.75;
  }
  if (mesh.GetNE() != 0)
  {
    destination.zhang_shu_theta_p[0] = 1.25;
  }
  const bool invalid_theta_rejected = !output_fields.UpdateDerived();
  if (!AllRanks(MPI_COMM_WORLD,
                invalid_theta_rejected == (mesh.GetNE() != 0)))
  {
    return false;
  }
  if (mesh.GetNE() != 0)
  {
    destination.zhang_shu_theta_p[0] = 0.75;
  }
  if (!output_fields.UpdateDerived()) { return false; }

  auto fields = output_fields.Fields();
  mfem::ParGridFunction* const zhang_shu_active =
      FindField(fields, "zhang_shu_ever_active");
  mfem::ParGridFunction* const zhang_shu_theta_rho =
      FindField(fields, "zhang_shu_theta_rho_min_cumulative");
  mfem::ParGridFunction* const zhang_shu_theta_p =
      FindField(fields, "zhang_shu_theta_p_min_cumulative");
  if (!zhang_shu_active || !zhang_shu_theta_rho || !zhang_shu_theta_p ||
      zhang_shu_active->Size() != mesh.GetNE() ||
      zhang_shu_theta_rho->Size() != mesh.GetNE() ||
      zhang_shu_theta_p->Size() != mesh.GetNE())
  {
    return false;
  }
  if (mesh.GetNE() != 0)
  {
    const double* const active = zhang_shu_active->HostRead();
    const double* const theta_rho = zhang_shu_theta_rho->HostRead();
    const double* const theta_p = zhang_shu_theta_p->HostRead();
    if (!active || !theta_rho || !theta_p || active[0] != 1.0 ||
        theta_rho[0] != 0.25 || theta_p[0] != 0.75)
    {
      return false;
    }
  }

  dg_sim::hdg::HdgEulerParaViewFields disabled_output_fields(
      mesh, collection, conservative, /*igr_active=*/false,
      /*enable_zhang_shu_diagnostics=*/false,
      /*gamma=*/1.4, /*density_floor=*/1.0e-16);
  dg::hdg::generic::HdgEulerHostInteriorFields disabled_destination{};
  auto disabled_fields = disabled_output_fields.Fields();
  if (!disabled_output_fields.FreshHostOutput(disabled_destination) ||
      disabled_destination.zhang_shu_theta_rho != nullptr ||
      disabled_destination.zhang_shu_theta_p != nullptr ||
      disabled_destination.zhang_shu_element_count != 0 ||
      FindField(disabled_fields, "zhang_shu_ever_active") ||
      FindField(disabled_fields, "zhang_shu_theta_rho_min_cumulative") ||
      FindField(disabled_fields, "zhang_shu_theta_p_min_cumulative"))
  {
    return false;
  }

  dg_sim::hdg::HDGParaViewRequest request{
      mesh,
      fields,
      {output.string(), "hdg_physical", mfem::VTKFormat::ASCII, 1, false}};
  {
    dg_sim::hdg::HDGParaViewSeries publisher(request);
    publisher.Write({0, 20, 1.25});
    publisher.Write({1, 25, 1.5});
  }
  request.options.restart =
      dg_sim::hdg::HDGParaViewOptions::RestartSeed{1, 1.5};
  bool stale_cycle_rejected = false;
  bool nonmonotone_time_rejected = false;
  {
    dg_sim::hdg::HDGParaViewSeries resumed(request);
    resumed.Write({2, 30, 1.75});
    resumed.Write({3, 35, 2.0});
    try { resumed.Write({3, 40, 2.25}); }
    catch (const std::invalid_argument&) { stale_cycle_rejected = true; }
    try { resumed.Write({4, 40, 2.0}); }
    catch (const std::invalid_argument&) { nonmonotone_time_rejected = true; }
  }
  bool stale_seed_rejected = false;
  request.options.restart =
      dg_sim::hdg::HDGParaViewOptions::RestartSeed{2, 1.75};
  try
  {
    static_cast<void>(dg_sim::hdg::HDGParaViewSeries(request));
  }
  catch (const std::invalid_argument&) { stale_seed_rejected = true; }
  MPI_Barrier(MPI_COMM_WORLD);

  const std::filesystem::path first_cycle =
      output / "hdg_physical" / "Cycle000000";
  const std::filesystem::path second_cycle =
      output / "hdg_physical" / "Cycle000001";
  const std::filesystem::path resumed_first_cycle =
      output / "hdg_physical" / "Cycle000002";
  const std::filesystem::path resumed_second_cycle =
      output / "hdg_physical" / "Cycle000003";
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
      std::filesystem::exists(resumed_first_cycle / rank_name.str()) &&
      std::filesystem::exists(resumed_second_cycle / rank_name.str()) &&
      stale_cycle_rejected && nonmonotone_time_rejected &&
      stale_seed_rejected &&
      Contains(local_text, "Name=\"density\"") &&
      Contains(local_text, "Name=\"momentum\"") &&
      Contains(local_text, "Name=\"energy\"") &&
      Contains(local_text, "Name=\"velocity\"") &&
      Contains(local_text, "Name=\"sigma\"") &&
      Contains(local_text, "Name=\"pressure\"") &&
      Contains(local_text, "Name=\"zhang_shu_ever_active\"") &&
      Contains(local_text,
               "Name=\"zhang_shu_theta_rho_min_cumulative\"") &&
      Contains(local_text,
               "Name=\"zhang_shu_theta_p_min_cumulative\"") &&
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
        Contains(pvd, "Cycle000000/data.pvtu") &&
        Contains(pvd, "Cycle000001/data.pvtu") &&
        Contains(pvd, "Cycle000002/data.pvtu") &&
        Contains(pvd, "Cycle000003/data.pvtu") &&
        CountOccurrences(pvd, "Cycle000000/data.pvtu") == 1 &&
        CountOccurrences(pvd, "Cycle000001/data.pvtu") == 1 &&
        CountOccurrences(pvd, "Cycle000002/data.pvtu") == 1 &&
        CountOccurrences(pvd, "Cycle000003/data.pvtu") == 1 &&
        !Contains(pvd, "stale-history-cycle-000002") &&
        Contains(first_pvtu, "Name=\"zhang_shu_ever_active\"") &&
        Contains(first_pvtu,
                 "Name=\"zhang_shu_theta_rho_min_cumulative\"") &&
        Contains(first_pvtu,
                 "Name=\"zhang_shu_theta_p_min_cumulative\"") &&
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
