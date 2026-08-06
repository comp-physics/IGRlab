#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/solver.hpp"

#include <mpi.h>

#include <iostream>

int main(int argc, char** argv)
{
  using namespace dg::hdg::runtime::app;
  const ParsedCommandLine parsed = ParseCommandLine(
      argc, const_cast<const char* const*>(argv));
  if (parsed.status == CommandLineStatus::help)
  {
    PrintCommandLineHelp(std::cout);
    return static_cast<int>(ApplicationExit::success);
  }
  if (parsed.status != CommandLineStatus::run)
  {
    std::cerr << "dg_igr_hdg_runtime: " << parsed.diagnostic << '\n';
    return static_cast<int>(parsed.status == CommandLineStatus::rejected
        ? ExitForRejection(parsed.rejection)
        : ApplicationExit::configuration_error);
  }

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
  {
    std::cerr << "dg_igr_hdg_runtime: MPI initialization failed\n";
    return static_cast<int>(ApplicationExit::runtime_failure);
  }
  MpiApplicationContextV1 cpu_context;
  CudaApplicationContextV1 cuda_context;
  const ApplicationRunResult result = DispatchApplication(
      parsed.options, cpu_context, &cuda_context);
  if (cpu_context.rank() == 0)
  {
    std::ostream& stream = result.status == dg::hdg::runtime::RunStatus::committed
        ? std::cout : std::cerr;
    stream << "dg_igr_hdg_runtime: status=" << ToString(result.status)
           << " reason=" << ToString(result.reason)
           << " step=" << result.coordinates.step
           << " time=" << result.coordinates.time
           << " retries=" << result.retries;
    const std::string& diagnostic = cpu_context.diagnostic().empty()
        ? cuda_context.diagnostic() : cpu_context.diagnostic();
    if (!diagnostic.empty()) { stream << " diagnostic=" << diagnostic; }
    stream << '\n';
  }
  const int exit = static_cast<int>(ExitForResult(result));
  (void)MPI_Finalize();
  return exit;
}
