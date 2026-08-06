#pragma once

#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::app
{

enum class CommandLineStatus : unsigned char
{
  run,
  help,
  usage_error,
  rejected
};

enum class ApplicationExit : int
{
  success = 0,
  configuration_error = 64,
  unsupported_discretization = 65,
  restart_identity_mismatch = 66,
  runtime_failure = 70,
  scientific_gate_failure = 71,
  injected_failure = 72
};

enum class RequestedBackend : unsigned char
{
  cpu,
  cuda
};

struct ParsedCommandLine
{
  CommandLineStatus status = CommandLineStatus::usage_error;
  ApplicationOptionsView options{};
  ApplicationRejection rejection = ApplicationRejection::none;
  std::string_view diagnostic{};
  RequestedBackend requested_backend = RequestedBackend::cpu;
  std::shared_ptr<std::vector<std::pair<std::string, std::string>>>
      owned_settings{};
};

ParsedCommandLine ParseCommandLine(int argc, const char* const* argv) noexcept;
void PrintCommandLineHelp(std::ostream& output);
std::string_view ToString(ApplicationRejection rejection) noexcept;
std::string_view ToString(RunStatus status) noexcept;
std::string_view ToString(FailureReason reason) noexcept;
ApplicationExit ExitForRejection(ApplicationRejection rejection) noexcept;
ApplicationExit ExitForResult(const ApplicationRunResult& result) noexcept;

} // namespace dg::hdg::runtime::app
