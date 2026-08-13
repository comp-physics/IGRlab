#include "mfem.hpp"

#include "numerics/hdg/runtime/app/application_contract.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/solver.hpp"

#include <mpi.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace
{

using dg::hdg::runtime::app::ApplicationExit;
using dg::hdg::runtime::app::ApplicationOptionsView;
using dg::hdg::runtime::app::ApplicationRunResult;
using dg::hdg::runtime::CpuSerialBackendTag;
using dg::hdg::runtime::app::ExitForResult;
using dg::hdg::runtime::app::HdgMethodRequestV1;
using dg::hdg::runtime::app::MpiApplicationContextV1;
using dg::hdg::runtime::app::OptionalCheckpointStepV1;
using dg::hdg::runtime::app::OutputDirectoryPathV1;
using dg::hdg::runtime::app::CheckpointArtifactPathV1;
using dg::hdg::runtime::app::RunHdgMethodV1;
using dg::hdg::runtime::RunStatus;
using dg::hdg::runtime::app::ValidateApplicationOptions;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_checkpoint_id;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_checkpoint_step;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_final_time;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_id;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_mesh_n;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_output_steps;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_polynomial_order;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_q_a;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_q_e;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_q_e4;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_q_o;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_seed;
using dg::hdg::runtime::app::smooth_periodic_euler_v1_functional_steps;

struct AdapterOptions
{
  bool help = false;
  std::string output_directory;
};

void PrintHelp(std::ostream& output)
{
  output << "Usage: hdg_smooth_periodic --discretization hdg "
            "--hdg-output-dir ABSOLUTE_PATH\n\n"
            "Runs the admitted CPU-serial conventional HDG smooth-periodic "
            "functional tuple (steps 0, 10, and 20 are published as VTK).\n";
}

bool ParseOptionValue(const int argc, char* argv[], int& index,
                      const std::string_view name, std::string_view& value,
                      std::string& diagnostic)
{
  const std::string_view argument{argv[index]};
  const std::string equals_prefix = std::string{name} + "=";
  if (argument.rfind(equals_prefix, 0) == 0)
  {
    value = argument.substr(equals_prefix.size());
    if (!value.empty()) return true;
  }
  else if (argument == name && index + 1 < argc && argv[index + 1] != nullptr)
  {
    value = argv[++index];
    if (!value.empty()) return true;
  }
  diagnostic = "missing value for " + std::string{name};
  return false;
}

bool ParseAdapterOptions(const int argc, char* argv[], AdapterOptions& options,
                         std::string& diagnostic)
{
  if (argc < 1 || argv == nullptr || argv[0] == nullptr)
  {
    diagnostic = "missing argv";
    return false;
  }
  bool discretization_seen = false;
  bool output_seen = false;
  for (int index = 1; index < argc; ++index)
  {
    if (argv[index] == nullptr)
    {
      diagnostic = "null argv entry";
      return false;
    }
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h")
    {
      options.help = true;
      continue;
    }
    std::string_view value;
    if (argument == "--discretization" ||
        argument.rfind("--discretization=", 0) == 0)
    {
      if (discretization_seen)
      {
        diagnostic = "duplicate --discretization option";
        return false;
      }
      if (!ParseOptionValue(argc, argv, index, "--discretization", value,
                            diagnostic))
        return false;
      if (value != "hdg")
      {
        diagnostic = "--discretization must be hdg";
        return false;
      }
      discretization_seen = true;
      continue;
    }
    if (argument == "--hdg-output-dir" ||
        argument.rfind("--hdg-output-dir=", 0) == 0)
    {
      if (output_seen)
      {
        diagnostic = "duplicate --hdg-output-dir option";
        return false;
      }
      if (!ParseOptionValue(argc, argv, index, "--hdg-output-dir", value,
                            diagnostic))
        return false;
      const std::filesystem::path path{std::string{value}};
      if (!path.is_absolute() || path.lexically_normal() != path)
      {
        diagnostic = "--hdg-output-dir must be a normalized absolute path";
        return false;
      }
      options.output_directory = std::string{value};
      output_seen = true;
      continue;
    }
    if (argument == "--dg-backend" || argument.rfind("--dg-backend=", 0) == 0 ||
        argument == "-dg")
    {
      diagnostic = "--dg-backend is a DG/DGSEM-only legacy selector";
      return false;
    }
    if (argument == "--device" || argument.rfind("--device=", 0) == 0 ||
        argument == "-dev")
    {
      diagnostic = "device selection is unsupported by CPU-serial HDG";
      return false;
    }
    diagnostic = "unsupported HDG adapter option: " + std::string{argument};
    return false;
  }
  if (options.help) return true;
  if (!discretization_seen || !output_seen)
  {
    diagnostic = "--discretization hdg and --hdg-output-dir are required";
    return false;
  }
  return true;
}

ApplicationOptionsView MakeFunctionalRequest(
    const std::string_view output_directory,
    const std::string_view checkpoint_path) noexcept
{
  ApplicationOptionsView options{};
  options.backend_id = CpuSerialBackendTag::id;
  options.scenario_id = smooth_periodic_euler_v1_functional_id;
  options.polynomial_order = smooth_periodic_euler_v1_functional_polynomial_order;
  options.mesh_n = smooth_periodic_euler_v1_functional_mesh_n;
  options.q_a = smooth_periodic_euler_v1_functional_q_a;
  options.q_e = smooth_periodic_euler_v1_functional_q_e;
  options.q_e4 = smooth_periodic_euler_v1_functional_q_e4;
  options.q_o = smooth_periodic_euler_v1_functional_q_o;
  options.accepted_steps = smooth_periodic_euler_v1_functional_steps;
  options.final_time = smooth_periodic_euler_v1_functional_final_time;
  options.output_steps = smooth_periodic_euler_v1_functional_output_steps;
  options.checkpoint_id = smooth_periodic_euler_v1_functional_checkpoint_id;
  options.checkpoint_step = OptionalCheckpointStepV1::Present(
      smooth_periodic_euler_v1_functional_checkpoint_step);
  options.seed = smooth_periodic_euler_v1_functional_seed;
  options.output_directory = OutputDirectoryPathV1::Present(output_directory);
  options.checkpoint_out = CheckpointArtifactPathV1::Present(checkpoint_path);
  return options;
}

int FailBeforeRuntime(const std::string_view diagnostic)
{
  std::cerr << "hdg_smooth_periodic: " << diagnostic << '\n';
  return static_cast<int>(ApplicationExit::configuration_error);
}

int FinishAfterMpi(const int exit_code)
{
  (void)MPI_Finalize();
  return exit_code;
}

int FinishAfterHypre(const int exit_code)
{
  mfem::Hypre::Finalize();
  return FinishAfterMpi(exit_code);
}

} // namespace

int main(int argc, char* argv[])
{
  AdapterOptions adapter_options;
  std::string diagnostic;
  if (!ParseAdapterOptions(argc, argv, adapter_options, diagnostic))
    return FailBeforeRuntime(diagnostic);
  if (adapter_options.help)
  {
    PrintHelp(std::cout);
    return static_cast<int>(ApplicationExit::success);
  }

  mfem::Mpi::Init(argc, argv);
  MpiApplicationContextV1 context;
  if (context.ranks() != 1)
  {
    if (context.rank() == 0)
      std::cerr << "hdg_smooth_periodic: CPU-serial HDG requires exactly one MPI rank\n";
    return FinishAfterMpi(
        static_cast<int>(ApplicationExit::unsupported_discretization));
  }
  mfem::Hypre::Init();

  const std::string checkpoint_path = adapter_options.output_directory +
      ".checkpoint";
  const ApplicationOptionsView options = MakeFunctionalRequest(
      adapter_options.output_directory, checkpoint_path);
  const auto validation = ValidateApplicationOptions(options);
  if (!validation.accepted())
  {
    std::cerr << "hdg_smooth_periodic: internal HDG tuple rejected\n";
    return FinishAfterHypre(
        static_cast<int>(ApplicationExit::configuration_error));
  }
  const HdgMethodRequestV1 request{&options, validation.config(), &context, nullptr};
  const auto result = RunHdgMethodV1(request);
  const ApplicationRunResult& run = result.run;
  if (context.rank() == 0)
  {
    std::ostream& stream = run.status == RunStatus::committed ? std::cout : std::cerr;
    stream << "hdg_smooth_periodic: status="
           << dg::hdg::runtime::app::ToString(run.status)
           << " step=" << run.coordinates.step
           << " time=" << run.coordinates.time << '\n';
  }
  return FinishAfterHypre(static_cast<int>(ExitForResult(run)));
}
