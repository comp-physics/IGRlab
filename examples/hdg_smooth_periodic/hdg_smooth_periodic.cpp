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
using dg::hdg::runtime::CpuRank2BackendTag;
using dg::hdg::runtime::CpuSerialBackendTag;
using dg::hdg::runtime::CudaRank1OneGpuBackendTag;
using dg::hdg::runtime::CudaRank2OneGpuHostStagedBackendTag;
using dg::hdg::runtime::app::CudaApplicationContextV1;
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
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
  bool cuda_device_seen = false;
#endif
};

void PrintHelp(std::ostream& output)
{
  output << "Usage: hdg_smooth_periodic --discretization hdg "
            "--hdg-output-dir ABSOLUTE_PATH"
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
            " --device cuda"
#endif
            "\n\n"
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
            "Runs the admitted conventional HDG smooth-periodic "
#else
            "Runs the admitted CPU conventional HDG smooth-periodic "
#endif
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
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
      if (options.cuda_device_seen)
      {
        diagnostic = "duplicate --device option";
        return false;
      }
      const std::string_view option_name =
          argument == "-dev" ? "-dev" : "--device";
      if (!ParseOptionValue(argc, argv, index, option_name, value, diagnostic))
        return false;
      if (value != "cuda")
      {
        diagnostic = "CUDA HDG requires --device cuda";
        return false;
      }
      options.cuda_device_seen = true;
      continue;
#else
      diagnostic = "device selection is unsupported by CPU HDG";
      return false;
#endif
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
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
  if (!options.cuda_device_seen)
  {
    diagnostic = "CUDA HDG requires --device cuda";
    return false;
  }
#endif
  return true;
}

ApplicationOptionsView MakeFunctionalRequest(
    const std::string_view backend_id,
    const int device_id,
    const std::string_view output_directory,
    const std::string_view checkpoint_path) noexcept
{
  ApplicationOptionsView options{};
  options.backend_id = backend_id;
  options.device_id = device_id;
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

#if !DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
int FinishAfterHypre(const int exit_code)
{
  mfem::Hypre::Finalize();
  return FinishAfterMpi(exit_code);
}
#endif

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
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
  CudaApplicationContextV1 context;
#else
  MpiApplicationContextV1 context;
#endif
  std::string_view backend_id;
  if (context.ranks() == 1)
  {
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
    backend_id = CudaRank1OneGpuBackendTag::id;
#else
    backend_id = CpuSerialBackendTag::id;
#endif
  }
  else if (context.ranks() == 2)
  {
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
    backend_id = CudaRank2OneGpuHostStagedBackendTag::id;
#else
    backend_id = CpuRank2BackendTag::id;
#endif
  }
  else
  {
    if (context.rank() == 0)
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
      std::cerr << "hdg_smooth_periodic: CUDA HDG requires exactly one or two MPI ranks\n";
#else
      std::cerr << "hdg_smooth_periodic: CPU HDG requires exactly one or two MPI ranks\n";
#endif
    return FinishAfterMpi(
        static_cast<int>(ApplicationExit::unsupported_discretization));
  }
#if !DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
  mfem::Hypre::Init();
#endif

  const std::string checkpoint_path = adapter_options.output_directory +
      ".checkpoint";
  const ApplicationOptionsView options = MakeFunctionalRequest(
      backend_id,
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
      0,
#else
      -1,
#endif
      adapter_options.output_directory, checkpoint_path);
  const auto validation = ValidateApplicationOptions(options);
  if (!validation.accepted())
  {
    std::cerr << "hdg_smooth_periodic: internal HDG tuple rejected\n";
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
    return FinishAfterMpi(
        static_cast<int>(ApplicationExit::configuration_error));
#else
    return FinishAfterHypre(
        static_cast<int>(ApplicationExit::configuration_error));
#endif
  }
  const HdgMethodRequestV1 request{
      &options, validation.config(),
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
      nullptr, &context};
#else
      &context, nullptr};
#endif
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
#if DG_IGR_HDG_SMOOTH_PERIODIC_HAS_CUDA
  return FinishAfterMpi(static_cast<int>(ExitForResult(run)));
#else
  return FinishAfterHypre(static_cast<int>(ExitForResult(run)));
#endif
}
