#include "numerics/hdg/runtime/app/command_line.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "CLI test must compile as strict C++17");
static_assert(!std::is_polymorphic<ParsedCommandLine>::value &&
                  !std::is_polymorphic<ApplicationOptionsView>::value,
              "CLI and configuration use static values");

bool Check(const bool condition, const char* name)
{
  if (!condition) std::cerr << "FAILED: " << name << '\n';
  return condition;
}

std::vector<std::string> SmokeArguments()
{
  return {"dg_igr_hdg_runtime",
      "--card", "exp.hdg.euler.conventional.smooth_persistent.w8a.v1",
      "--case", "conventional-smooth-persistent-v1",
      "--space", "l2", "--mesh-family", "periodic-cartesian-quad",
      "--mesh-n", "3", "--order", "1", "--dt", "0.05",
      "--steps", "3", "--final-time", "0.15", "--backend", "cpu",
      "--device-id", "none", "--output-dir", "w8a-test-output",
      "--output-steps", "0,1,3", "--evidence-dir", "none",
      "--validation-capture", "none", "--checkpoint-out",
      "w8a-test-checkpoint", "--checkpoint-step", "1",
      "--stop-after-checkpoint", "false", "--restart-from", "none",
      "--max-retries", "0", "--inject-failure", "off", "--seed",
      "0x573841534d4f4b31"};
}

ParsedCommandLine Parse(std::vector<std::string> values)
{
  std::vector<const char*> argv;
  for (const std::string& value : values) argv.push_back(value.c_str());
  return ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}

bool DirectAndNegativeControls()
{
  bool passed = true;
  const ParsedCommandLine valid = Parse(SmokeArguments());
  passed = Check(valid.status == CommandLineStatus::run &&
                     valid.options.card_id == smooth_periodic_euler_v1_card_id &&
                     valid.options.case_id == smooth_periodic_euler_v1_case_id &&
                     valid.options.backend_id == CpuSerialBackendTag::id &&
                     valid.options.device_id == -1 &&
                     valid.options.output_steps ==
                         smooth_periodic_euler_v1_smoke_output_steps,
                 "complete_direct_smoke") && passed;

  auto h1 = SmokeArguments();
  h1[6] = "h1";
  const ParsedCommandLine unsupported = Parse(h1);
  passed = Check(unsupported.status == CommandLineStatus::rejected &&
                     unsupported.rejection == ApplicationRejection::unsupported_h1 &&
                     ExitForRejection(unsupported.rejection) ==
                         ApplicationExit::unsupported_discretization,
                 "h1_exact_rejection") && passed;

  auto invalid_pair = SmokeArguments();
  invalid_pair[10] = "6";
  const ParsedCommandLine invalid = Parse(invalid_pair);
  passed = Check(invalid.status == CommandLineStatus::rejected &&
                     invalid.rejection ==
                         ApplicationRejection::invalid_polynomial_mesh_pair &&
                     ExitForRejection(invalid.rejection) ==
                         ApplicationExit::configuration_error,
                 "unregistered_polynomial_mesh_pair") && passed;

  auto duplicate = SmokeArguments();
  duplicate[1] = "--case";
  passed = Check(Parse(duplicate).status == CommandLineStatus::usage_error,
                 "duplicate_missing_census") && passed;
  return passed;
}

bool HelpAndConfig()
{
  bool passed = true;
  const char* help_argv[] = {"dg_igr_hdg_runtime", "--help"};
  const ParsedCommandLine help = ParseCommandLine(2, help_argv);
  std::ostringstream text;
  PrintCommandLineHelp(text);
  passed = Check(help.status == CommandLineStatus::help &&
                     text.str().find("--config") != std::string::npos &&
                     text.str().find("--validation-capture") != std::string::npos &&
                     text.str().find("AfterRecoveryBeforeCommit") !=
                         std::string::npos,
                 "help_complete") && passed;

  const std::filesystem::path path = std::filesystem::path{"/tmp"} /
      ("dg-igr-hdg-cli-" + std::to_string(static_cast<long long>(::getpid())) +
       ".cfg");
  {
    std::ofstream output(path);
    const auto args = SmokeArguments();
    for (std::size_t index = 1; index < args.size(); index += 2)
      output << args[index].substr(2) << '=' << args[index + 1] << '\n';
  }
  const std::string path_text = path.string();
  const char* config_argv[] = {"dg_igr_hdg_runtime", "--config",
                               path_text.c_str()};
  const ParsedCommandLine config = ParseCommandLine(3, config_argv);
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  passed = Check(config.status == CommandLineStatus::run &&
                     config.options.seed ==
                         smooth_periodic_euler_v1_smoke_seed,
                 "dependency_free_config") && passed;
  return passed;
}
} // namespace

int main()
{
  const bool passed = DirectAndNegativeControls() && HelpAndConfig();
  if (passed) std::cout << "HDG runtime CLI contract PASS\n";
  return passed ? 0 : 1;
}
