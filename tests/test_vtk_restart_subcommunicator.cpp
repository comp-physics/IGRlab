#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/io/run_metadata.hpp"
#include "sim/io/save_state.hpp"
#include "sim/io/vtk_restart.hpp"
#include "sim/solver/solver.hpp"
#include "sim/state/state_views.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

namespace
{

bool RunSplitRestart(const std::filesystem::path& output_root,
                     const std::string& mode)
{
  const int world_rank = mfem::Mpi::WorldRank();
  const int world_size = mfem::Mpi::WorldSize();
  if (world_size != 4) { return false; }

  if (world_rank == 0) { std::filesystem::remove_all(output_root); }
  MPI_Barrier(MPI_COMM_WORLD);

  const int color = world_rank / 2;
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);

  int comm_rank = -1;
  int comm_size = 0;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  int communicator_relation = MPI_IDENT;
  MPI_Comm_compare(comm, MPI_COMM_WORLD, &communicator_relation);
  bool local_ok =
    comm_size == 2 && communicator_relation == MPI_UNEQUAL;

  {
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
      2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
    auto mesh = std::make_unique<mfem::ParMesh>(comm, serial_mesh);
    const std::filesystem::path group_root =
      output_root / ("group" + std::to_string(color));

    dg_sim::RunConfig run_cfg;
    run_cfg.case_config.case_name = "vtk_restart_subcommunicator";
    run_cfg.case_config.gamma = 1.4;
    run_cfg.case_config.limiter_pipeline.zhang_shu = false;
    run_cfg.case_config.vtk_config.format = mfem::VTKFormat::BINARY;
    run_cfg.case_config.vtk_config.high_order_output = true;
    run_cfg.case_config.vtk_config.prefix_path =
      (group_root / "metadata-prefix").string();
    run_cfg.opts.order = 1;
    run_cfg.opts.ser_ref_levels = 0;
    run_cfg.opts.par_ref_levels = 0;
    run_cfg.opts.dt_save = 0.25;
    run_cfg.opts.vtk_save = true;
    run_cfg.opts.zhang_shu_points = "consumer";
    run_cfg.igr_opts.space = "L2";
    run_cfg.igr_opts.grad = "brokenL2";
    run_cfg.igr_opts.c_alpha_user = 1.0;
    run_cfg.regularization = dg_sim::Regularization::None;

    dg_sim::ExampleContext ctx(
      run_cfg, std::move(mesh), "L2", false, false);
    mfem::ParGridFunction state(&ctx.spaces.vfes);
    mfem::ParGridFunction static_source(&ctx.spaces.vfes);
    const int scalar_dofs = ctx.spaces.fes.GetNDofs();
    const double density = 1.0 + 0.1 * color + 0.01 * comm_rank;
    const double momentum_x = 0.1 + 0.02 * comm_rank;
    const double momentum_y = -0.05 - 0.01 * color;
    const double energy = 3.0 + 0.1 * color + 0.01 * comm_rank;
    double* values = state.HostWrite();
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      values[dof] = density;
      values[scalar_dofs + dof] = momentum_x;
      values[2 * scalar_dofs + dof] = momentum_y;
      values[3 * scalar_dofs + dof] = energy;
    }
    double* source_values = static_source.HostWrite();
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      source_values[dof] = 1.0e-3 * (1.0 + color);
      source_values[scalar_dofs + dof] = 0.0;
      source_values[2 * scalar_dofs + dof] = -2.0e-4 * (1.0 + comm_rank);
      source_values[3 * scalar_dofs + dof] = 3.0e-4;
    }
    mfem::ParGridFunction identical_source(&ctx.spaces.vfes);
    identical_source = static_source;
    mfem::ParGridFunction changed_source(&ctx.spaces.vfes);
    changed_source = static_source;
    changed_source.HostReadWrite()[0] += 1.0e-12;
    const std::uint64_t source_fingerprint =
      dg_sim::ComputeVTKRestartFieldFingerprint(static_source, ctx);
    local_ok = local_ok &&
               source_fingerprint ==
                 dg_sim::ComputeVTKRestartFieldFingerprint(identical_source,
                                                           ctx) &&
               source_fingerprint !=
                 dg_sim::ComputeVTKRestartFieldFingerprint(changed_source,
                                                           ctx);
    const std::string output_dataset_name =
      dg_sim::detail::MakeEulerOutputDatasetName(run_cfg.case_config,
                                                 run_cfg,
                                                 1.0,
                                                 nullptr,
                                                 ctx,
                                                 &static_source);
    std::ostringstream source_tag;
    source_tag << "_src_" << std::hex << std::setfill('0') << std::setw(16)
               << source_fingerprint;
    const std::string communicator_tag =
      "_commroot" + std::to_string(2 * color);
    local_ok = local_ok &&
               output_dataset_name.find(source_tag.str()) !=
                 std::string::npos &&
               output_dataset_name.size() >= communicator_tag.size() &&
               output_dataset_name.compare(output_dataset_name.size() -
                                              communicator_tag.size(),
                                            communicator_tag.size(),
                                            communicator_tag) == 0;
    dg_sim::EulerStateViews views(
      ctx.spaces.fes, ctx.spaces.dfes, state, ctx.dim);

    const std::filesystem::path glvis_root = group_root / "glvis";
    dg_sim::SaveInitialEulerState(
      ctx, state, 16, "euler-mesh", "euler", glvis_root.string());
    dg_sim::SaveFinalEulerState(
      ctx, state, 16, "euler-mesh-final", "euler", glvis_root.string());
    MPI_Barrier(comm);
    for (int rank = 0; rank < comm_size; ++rank)
    {
      std::ostringstream suffix;
      suffix << '.' << std::setfill('0') << std::setw(6) << rank;
      local_ok = local_ok &&
                 std::filesystem::is_regular_file(
                   glvis_root / ("euler-mesh" + suffix.str())) &&
                 std::filesystem::is_regular_file(
                   glvis_root / ("euler-mesh-final" + suffix.str()));
      for (int equation = 0; equation < ctx.num_equations; ++equation)
      {
        local_ok = local_ok &&
                   std::filesystem::is_regular_file(
                     glvis_root /
                     ("euler-" + std::to_string(equation) + "-init" +
                      suffix.str())) &&
                   std::filesystem::is_regular_file(
                     glvis_root /
                     ("euler-" + std::to_string(equation) + "-final" +
                      suffix.str()));
      }
    }
    for (int rank = comm_size; rank < world_size; ++rank)
    {
      std::ostringstream suffix;
      suffix << '.' << std::setfill('0') << std::setw(6) << rank;
      local_ok = local_ok &&
                 !std::filesystem::exists(
                   glvis_root / ("euler-mesh" + suffix.str())) &&
                 !std::filesystem::exists(
                   glvis_root / ("euler-mesh-final" + suffix.str()));
    }

    const std::string metadata_dataset = output_dataset_name;
    dg_sim::WriteRunMetadataSidecar(run_cfg.case_config,
                                    run_cfg,
                                    nullptr,
                                    nullptr,
                                    1.0,
                                    0,
                                    nullptr,
                                    true,
                                    metadata_dataset,
                                    nullptr,
                                    comm,
                                    &static_source,
                                    &ctx);
    MPI_Barrier(comm);
    if (comm_rank == 0)
    {
      const std::filesystem::path metadata_path =
        std::filesystem::path(
          dg_sim::MakeCaseVTKConfig(run_cfg.case_config).prefix_path) /
        metadata_dataset / "run_metadata.json";
      std::ifstream metadata_input(metadata_path);
      const std::string metadata(
        (std::istreambuf_iterator<char>(metadata_input)),
        std::istreambuf_iterator<char>());
      local_ok = local_ok && metadata_input.is_open() &&
                 metadata.find("\"mpi_ranks\": 2") != std::string::npos &&
                 metadata.find("\"static_source_present\": true") !=
                   std::string::npos &&
                 metadata.find("\"static_source_fingerprint\": \"" +
                                 source_tag.str().substr(5) + "\"") !=
                   std::string::npos;
    }

    const std::string dataset_name = output_dataset_name;
    mfem::ParaViewDataCollection collection(dataset_name, &ctx.Mesh());
    collection.SetPrefixPath(group_root.string());
    collection.SetDataFormat(mfem::VTKFormat::BINARY);
    collection.SetHighOrderOutput(true);
    collection.SetLevelsOfDetail(run_cfg.opts.order);
    collection.RegisterField("density", &views.density);
    collection.RegisterField("momentum", &views.momentum);
    collection.RegisterField("energy", &views.energy);
    collection.SetCycle(3);
    collection.SetTime(0.25);
    collection.Save();

    const std::filesystem::path dataset_dir = group_root / dataset_name;
    dg_sim::WriteVTKRestartManifest(dataset_dir.string(),
                                    dataset_name,
                                    3,
                                    0.25,
                                    17,
                                    run_cfg,
                                    ctx,
                                    false,
                                    collection.GetCompressionLevel(),
                                    &static_source);
    if (mode == "schema2-source")
    {
      if (comm_rank == 0)
      {
        const std::filesystem::path manifest_path =
          dataset_dir / "Cycle000003" / "restart.json";
        std::ifstream input(manifest_path);
        const bool input_open = static_cast<bool>(input);
        std::string document((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
        const std::string key = "\"schema_version\"";
        const std::size_t key_position = document.find(key);
        const std::size_t colon_position = document.find(':', key_position);
        const std::size_t value_position =
          document.find_first_of("0123456789", colon_position);
        local_ok = local_ok && input_open &&
                   key_position != std::string::npos &&
                   colon_position != std::string::npos &&
                   value_position != std::string::npos;
        if (local_ok)
        {
          document.replace(value_position, 1, "2");
          input.close();
          std::ofstream output(manifest_path, std::ios::trunc);
          output << document;
          local_ok = output.good();
        }
      }
      MPI_Barrier(comm);
    }

    dg_sim::VTKRestartRequest request;
    request.path = (dataset_dir / (dataset_name + ".pvd")).string();
    request.cycle = 3;
    mfem::ParGridFunction restored(&ctx.spaces.vfes);
    const mfem::ParGridFunction* restart_source =
      mode == "changed-source" ? &changed_source : &static_source;
    const dg_sim::VTKRestartInfo info =
      dg_sim::LoadEulerVTKRestart(
        request, run_cfg, ctx, restored, false, restart_source);

    const double* expected = state.HostRead();
    const double* actual = restored.HostRead();
    double local_max_abs = 0.0;
    for (int dof = 0; dof < state.Size(); ++dof)
    {
      local_max_abs =
        std::max(local_max_abs, std::abs(actual[dof] - expected[dof]));
    }
    double group_max_abs = 0.0;
    MPI_Allreduce(
      &local_max_abs, &group_max_abs, 1, MPI_DOUBLE, MPI_MAX, comm);
    local_ok = local_ok && info.cycle == 3 && info.step == 17 &&
               info.time == 0.25 && !info.legacy && group_max_abs <= 1.0e-12;
  }

  MPI_Comm_free(&comm);
  const int local_pass = local_ok ? 1 : 0;
  int global_pass = 0;
  MPI_Allreduce(
    &local_pass, &global_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global_pass == 1;
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  if (argc != 2 && argc != 3)
  {
    if (mfem::Mpi::Root())
    {
      std::cerr << "usage: test_vtk_restart_subcommunicator OUTPUT_ROOT "
                   "[changed-source|schema2-source]\n";
    }
    return 2;
  }
  const std::string mode = argc == 3 ? argv[2] : "normal";
  if (mode != "normal" && mode != "changed-source" &&
      mode != "schema2-source")
  {
    if (mfem::Mpi::Root()) { std::cerr << "unknown test mode\n"; }
    return 2;
  }

  const bool passed =
    RunSplitRestart(std::filesystem::absolute(argv[1]), mode);
  if (mfem::Mpi::Root())
  {
    std::cout << "vtk-restart-subcommunicator: "
              << (passed ? "PASS" : "FAIL") << '\n';
  }
  return passed ? 0 : 1;
}
