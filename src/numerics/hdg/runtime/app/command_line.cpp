#include "numerics/hdg/runtime/app/command_line.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace dg::hdg::runtime::app
{
namespace
{

using Setting = std::pair<std::string, std::string>;
using Settings = std::vector<Setting>;

constexpr std::array<std::string_view, 22> required_keys{{
    "card", "case", "space", "mesh-family", "mesh-n", "order", "dt",
    "steps", "final-time", "backend", "device-id", "output-dir",
    "output-steps", "evidence-dir", "validation-capture",
    "checkpoint-out", "checkpoint-step", "stop-after-checkpoint",
    "restart-from", "max-retries", "inject-failure", "seed"}};

bool KnownKey(const std::string_view key) noexcept
{
  return std::find(required_keys.begin(), required_keys.end(), key) !=
      required_keys.end();
}

bool AddSetting(Settings& settings, const std::string_view key,
                const std::string_view value) noexcept
{
  if (!KnownKey(key) || value.empty()) { return false; }
  for (const auto& setting : settings)
  {
    if (setting.first == key) { return false; }
  }
  try
  {
    settings.emplace_back(std::string{key}, std::string{value});
    return true;
  }
  catch (...)
  {
    return false;
  }
}

const std::string_view* Find(const Settings& settings,
                             const std::string_view key,
                             std::string_view& value) noexcept
{
  for (const auto& setting : settings)
  {
    if (setting.first == key)
    {
      value = setting.second;
      return &value;
    }
  }
  return nullptr;
}

bool ParseUnsigned(const std::string_view text, std::uint64_t& value) noexcept
{
  std::size_t index = 0;
  unsigned base = 10;
  if (text.size() > 2 && text[0] == '0' &&
      (text[1] == 'x' || text[1] == 'X'))
  {
    index = 2;
    base = 16;
  }
  if (index == text.size()) { return false; }
  std::uint64_t parsed = 0;
  for (; index < text.size(); ++index)
  {
    const char character = text[index];
    unsigned digit = 0;
    if (character >= '0' && character <= '9')
      digit = static_cast<unsigned>(character - '0');
    else if (character >= 'a' && character <= 'f')
      digit = 10U + static_cast<unsigned>(character - 'a');
    else if (character >= 'A' && character <= 'F')
      digit = 10U + static_cast<unsigned>(character - 'A');
    else
      return false;
    if (digit >= base || parsed >
        (std::numeric_limits<std::uint64_t>::max() - digit) / base)
      return false;
    parsed = parsed * base + digit;
  }
  value = parsed;
  return true;
}

bool ParsePositiveInt(const std::string_view text, int& value) noexcept
{
  std::uint64_t parsed = 0;
  if (!ParseUnsigned(text, parsed) || parsed == 0 ||
      parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    return false;
  value = static_cast<int>(parsed);
  return true;
}

bool ParsePositiveDouble(const std::string_view text, double& value) noexcept
{
  if (text.empty() || text.size() > 64) { return false; }
  std::array<char, 65> buffer{};
  std::copy(text.begin(), text.end(), buffer.begin());
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(buffer.data(), &end);
  if (errno != 0 || end != buffer.data() + text.size() ||
      !std::isfinite(parsed) || parsed <= 0.0)
    return false;
  value = parsed;
  return true;
}

bool ParseBoolean(const std::string_view text, bool& value) noexcept
{
  if (text == "true") { value = true; return true; }
  if (text == "false") { value = false; return true; }
  return false;
}

bool ParseOutputSteps(const std::string_view text,
                      std::array<std::uint64_t, 3>& values) noexcept
{
  std::size_t begin = 0;
  for (std::size_t index = 0; index < values.size(); ++index)
  {
    const std::size_t comma = text.find(',', begin);
    if ((index + 1 < values.size() && comma == std::string_view::npos) ||
        (index + 1 == values.size() && comma != std::string_view::npos))
      return false;
    const std::size_t end = comma == std::string_view::npos
        ? text.size() : comma;
    if (!ParseUnsigned(text.substr(begin, end - begin), values[index]))
      return false;
    begin = end + 1;
  }
  return true;
}

bool ValidArtifactPath(const std::string_view value) noexcept
{
  if (value.empty() || value.size() > 512) { return false; }
  const bool absolute = value.front() == '/';
  std::size_t begin = absolute ? 1 : 0;
  if (begin == value.size() || value.back() == '/') { return false; }
  while (begin < value.size())
  {
    const std::size_t slash = value.find('/', begin);
    const std::size_t end = slash == std::string_view::npos
        ? value.size() : slash;
    const std::string_view component = value.substr(begin, end - begin);
    if (component.empty() || component == "." || component == "..")
      return false;
    begin = end + 1;
  }
  return true;
}

bool ParseFailurePhase(const std::string_view text,
                       FailurePhase& phase) noexcept
{
#define DG_IGR_PHASE(name) if (text == #name) { phase = FailurePhase::name; return true; }
  DG_IGR_PHASE(Assembly)
  DG_IGR_PHASE(Condensation)
  DG_IGR_PHASE(TraceSolve)
  DG_IGR_PHASE(Recovery)
  DG_IGR_PHASE(Diagnostics)
  DG_IGR_PHASE(CheckpointStaging)
  DG_IGR_PHASE(OutputStaging)
  DG_IGR_PHASE(AfterRecoveryBeforeCommit)
#undef DG_IGR_PHASE
  return false;
}

bool ParseInjection(const std::string_view text,
                    FailureInjectionControlV1& injection) noexcept
{
  if (text == "off")
  {
    injection = {};
    return true;
  }
  const std::size_t first = text.find(':');
  const std::size_t second = first == std::string_view::npos
      ? std::string_view::npos : text.find(':', first + 1);
  if (first == std::string_view::npos || second == std::string_view::npos ||
      text.find(':', second + 1) != std::string_view::npos)
    return false;
  std::uint64_t target = 0;
  std::uint64_t rank = 0;
  FailurePhase phase = FailurePhase::Setup;
  if (!ParseFailurePhase(text.substr(0, first), phase) ||
      !ParseUnsigned(text.substr(first + 1, second - first - 1), target) ||
      target == 0 || !ParseUnsigned(text.substr(second + 1), rank) ||
      rank > std::numeric_limits<std::uint32_t>::max())
    return false;
  injection = {FailureInjectionModeV1::prescribed, phase, target,
               static_cast<std::uint32_t>(rank)};
  return true;
}

void SelectScenario(ApplicationOptionsView& options,
                    const ApplicationScenarioV1 scenario) noexcept
{
  const bool functional = scenario == ApplicationScenarioV1::functional;
  options.scenario_id = functional ? smooth_periodic_euler_v1_functional_id
                                   : smooth_periodic_euler_v1_smoke_id;
  options.q_a = functional ? smooth_periodic_euler_v1_functional_q_a
                           : smooth_periodic_euler_v1_smoke_q_a;
  options.q_e = functional ? smooth_periodic_euler_v1_functional_q_e
                           : smooth_periodic_euler_v1_smoke_q_e;
  options.q_e4 = functional ? smooth_periodic_euler_v1_functional_q_e4
                            : smooth_periodic_euler_v1_smoke_q_e4;
  options.q_o = functional ? smooth_periodic_euler_v1_functional_q_o
                           : smooth_periodic_euler_v1_smoke_q_o;
  options.checkpoint_id = functional
      ? smooth_periodic_euler_v1_functional_checkpoint_id
      : smooth_periodic_euler_v1_smoke_checkpoint_id;
}

bool LoadConfigFile(const std::string_view path, Settings& settings) noexcept
{
  try
  {
    std::ifstream input{std::string{path}, std::ios::binary};
    if (!input) { return false; }
    std::string line;
    std::size_t bytes = 0;
    while (std::getline(input, line))
    {
      bytes += line.size() + 1;
      if (bytes > 64U * 1024U || line.find('\0') != std::string::npos)
        return false;
      if (!line.empty() && line.back() == '\r') { return false; }
      if (line.empty() || line.front() == '#') { continue; }
      const std::size_t equal = line.find('=');
      if (equal == std::string::npos || equal == 0 || equal + 1 >= line.size() ||
          line.find('=', equal + 1) != std::string::npos)
        return false;
      const std::string_view key{line.data(), equal};
      const std::string_view value{line.data() + equal + 1,
                                   line.size() - equal - 1};
      if (key.front() == ' ' || key.back() == ' ' || key.front() == '\t' ||
          key.back() == '\t' || value.front() == ' ' || value.back() == ' ' ||
          value.front() == '\t' || value.back() == '\t' ||
          !AddSetting(settings, key, value))
        return false;
    }
    return input.eof() && settings.size() == required_keys.size();
  }
  catch (...)
  {
    return false;
  }
}

bool CollectSettings(const int argc, const char* const* argv,
                     Settings& settings) noexcept
{
  if (argc == 3 && argv[1] != nullptr && argv[2] != nullptr &&
      std::string_view{argv[1]} == "--config")
    return LoadConfigFile(argv[2], settings);
  if (argc != 1 + 2 * static_cast<int>(required_keys.size())) { return false; }
  for (int index = 1; index < argc; index += 2)
  {
    if (argv[index] == nullptr || argv[index + 1] == nullptr) { return false; }
    const std::string_view flag{argv[index]};
    if (flag.size() <= 2 || flag.substr(0, 2) != "--" ||
        flag == "--config" || !AddSetting(settings, flag.substr(2),
                                           argv[index + 1]))
      return false;
  }
  return settings.size() == required_keys.size();
}

} // namespace

ParsedCommandLine ParseCommandLine(const int argc,
                                   const char* const* argv) noexcept
{
  ParsedCommandLine parsed{};
  if (argc < 1 || argv == nullptr || argv[0] == nullptr)
  {
    parsed.diagnostic = "missing argv";
    return parsed;
  }
  if (argc == 2 && argv[1] != nullptr &&
      (std::string_view{argv[1]} == "--help" ||
       std::string_view{argv[1]} == "-h"))
  {
    parsed.status = CommandLineStatus::help;
    parsed.diagnostic = "HelpOnly";
    return parsed;
  }

  auto settings = std::make_shared<Settings>();
  if (!CollectSettings(argc, argv, *settings))
  {
    parsed.diagnostic = "ConfigurationError: expected every registered option exactly once, or --config FILE";
    return parsed;
  }
  parsed.owned_settings = settings;

  ApplicationOptionsView options{};
  std::string_view card, case_id, space, mesh_family, mesh_n, order, dt,
      steps, final_time, backend, device, output_dir, output_steps,
      evidence_dir, capture, checkpoint_out, checkpoint_step, stop,
      restart_from, retries, injection, seed;
#define DG_IGR_FIND(name, key) if (!Find(*settings, key, name)) { parsed.diagnostic = "ConfigurationError: missing required value"; return parsed; }
  DG_IGR_FIND(card, "card")
  DG_IGR_FIND(case_id, "case")
  DG_IGR_FIND(space, "space")
  DG_IGR_FIND(mesh_family, "mesh-family")
  DG_IGR_FIND(mesh_n, "mesh-n")
  DG_IGR_FIND(order, "order")
  DG_IGR_FIND(dt, "dt")
  DG_IGR_FIND(steps, "steps")
  DG_IGR_FIND(final_time, "final-time")
  DG_IGR_FIND(backend, "backend")
  DG_IGR_FIND(device, "device-id")
  DG_IGR_FIND(output_dir, "output-dir")
  DG_IGR_FIND(output_steps, "output-steps")
  DG_IGR_FIND(evidence_dir, "evidence-dir")
  DG_IGR_FIND(capture, "validation-capture")
  DG_IGR_FIND(checkpoint_out, "checkpoint-out")
  DG_IGR_FIND(checkpoint_step, "checkpoint-step")
  DG_IGR_FIND(stop, "stop-after-checkpoint")
  DG_IGR_FIND(restart_from, "restart-from")
  DG_IGR_FIND(retries, "max-retries")
  DG_IGR_FIND(injection, "inject-failure")
  DG_IGR_FIND(seed, "seed")
#undef DG_IGR_FIND

  options.card_id = card;
  options.case_id = case_id;
  options.mesh_family_id = mesh_family;
  if (space == "l2") options.interior_space = InteriorSpace::l2;
  else if (space == "h1") options.interior_space = InteriorSpace::h1;
  else { parsed.diagnostic = "ConfigurationError: --space must be l2 or h1"; return parsed; }
  if (options.card_id != smooth_periodic_euler_v1_card_id)
  {
    parsed.status = CommandLineStatus::rejected;
    parsed.rejection = ApplicationRejection::unknown_card;
    parsed.diagnostic = ToString(parsed.rejection);
    parsed.options = options;
    return parsed;
  }
  if (options.case_id != smooth_periodic_euler_v1_case_id)
  {
    parsed.status = CommandLineStatus::rejected;
    parsed.rejection = ApplicationRejection::unknown_case;
    parsed.diagnostic = ToString(parsed.rejection);
    parsed.options = options;
    return parsed;
  }
  if (options.interior_space == InteriorSpace::h1)
  {
    parsed.status = CommandLineStatus::rejected;
    parsed.rejection = ApplicationRejection::unsupported_h1;
    parsed.diagnostic = ToString(parsed.rejection);
    parsed.options = options;
    return parsed;
  }
  int mesh = 0;
  if (!ParsePositiveInt(mesh_n, mesh) ||
      !ParsePositiveInt(order, options.polynomial_order) ||
      !ParsePositiveDouble(dt, options.dt) ||
      !ParseUnsigned(steps, options.accepted_steps) ||
      options.accepted_steps == 0 ||
      !ParsePositiveDouble(final_time, options.final_time) ||
      !ParseOutputSteps(output_steps, options.output_steps))
  {
    parsed.diagnostic = "ConfigurationError: malformed numerical value";
    return parsed;
  }
  options.mesh_n = static_cast<std::uint32_t>(mesh);
  if (options.accepted_steps == smooth_periodic_euler_v1_smoke_steps)
    SelectScenario(options, ApplicationScenarioV1::smoke);
  else if (options.accepted_steps == smooth_periodic_euler_v1_functional_steps)
    SelectScenario(options, ApplicationScenarioV1::functional);
  else
  {
    parsed.diagnostic = "ConfigurationError: unregistered step count";
    return parsed;
  }

  if (backend == "cpu")
  {
    parsed.requested_backend = RequestedBackend::cpu;
    options.backend_id = CpuSerialBackendTag::id;
    if (device != "none")
    {
      parsed.diagnostic = "ConfigurationError: CPU requires device-id none";
      return parsed;
    }
    options.device_id = -1;
  }
  else if (backend == "cuda")
  {
    parsed.requested_backend = RequestedBackend::cuda;
    options.backend_id = CudaRank1OneGpuBackendTag::id;
    if (device != "0")
    {
      parsed.diagnostic = "ConfigurationError: CUDA requires device-id 0";
      return parsed;
    }
    options.device_id = 0;
  }
  else
  {
    parsed.diagnostic = "ConfigurationError: --backend must be cpu or cuda";
    return parsed;
  }

  if (!ValidArtifactPath(output_dir))
  {
    parsed.diagnostic = "ConfigurationError: invalid output path";
    return parsed;
  }
  options.output_directory = OutputDirectoryPathV1::Present(output_dir);
  if (evidence_dir == "none") options.evidence_directory = {};
  else if (ValidArtifactPath(evidence_dir))
    options.evidence_directory = EvidenceDirectoryPathV1::Present(evidence_dir);
  else { parsed.diagnostic = "ConfigurationError: invalid evidence path"; return parsed; }
  if (capture == "none") options.validation_capture = ValidationCaptureModeV1::none;
  else if (capture == "persistent-v1")
    options.validation_capture = ValidationCaptureModeV1::persistent_v1;
  else { parsed.diagnostic = "ConfigurationError: invalid validation capture"; return parsed; }

  if (checkpoint_out == "none") options.checkpoint_out = {};
  else if (ValidArtifactPath(checkpoint_out))
    options.checkpoint_out = CheckpointArtifactPathV1::Present(checkpoint_out);
  else { parsed.diagnostic = "ConfigurationError: invalid checkpoint output path"; return parsed; }
  if (checkpoint_step == "none") options.checkpoint_step = {};
  else
  {
    std::uint64_t value = 0;
    if (!ParseUnsigned(checkpoint_step, value) || value == 0)
    { parsed.diagnostic = "ConfigurationError: invalid checkpoint step"; return parsed; }
    options.checkpoint_step = OptionalCheckpointStepV1::Present(value);
  }
  if (restart_from == "none") options.restart_from = {};
  else if (ValidArtifactPath(restart_from))
    options.restart_from = CheckpointArtifactPathV1::Present(restart_from);
  else { parsed.diagnostic = "ConfigurationError: invalid restart path"; return parsed; }

  bool stop_value = false;
  std::uint64_t retry_value = 0;
  if (!ParseBoolean(stop, stop_value) || !ParseUnsigned(retries, retry_value) ||
      retry_value > std::numeric_limits<std::uint32_t>::max() ||
      !ParseInjection(injection, options.injection) ||
      !ParseUnsigned(seed, options.seed))
  {
    parsed.diagnostic = "ConfigurationError: malformed run control";
    return parsed;
  }
  options.stop_after_checkpoint = stop_value ? StopAfterCheckpointV1::yes
                                              : StopAfterCheckpointV1::no;
  options.retry_limit = static_cast<std::uint32_t>(retry_value);
  if (options.restart_from.present)
    options.run_role = ApplicationRunRoleV1::split_reader;
  else if (stop_value)
    options.run_role = ApplicationRunRoleV1::split_writer;
  else if (options.injection.mode == FailureInjectionModeV1::prescribed)
    options.run_role = ApplicationRunRoleV1::focused_terminal_injection;
  else if (options.seed == smooth_periodic_euler_v1_alternate_functional_seed)
    options.run_role = ApplicationRunRoleV1::alternate_seed_reference;
  else
    options.run_role = ApplicationRunRoleV1::uninterrupted_reference;

  const ApplicationValidation validation = ValidateApplicationOptions(options);
  parsed.options = options;
  if (!validation.accepted())
  {
    parsed.status = CommandLineStatus::rejected;
    parsed.rejection = validation.rejection();
    parsed.diagnostic = ToString(validation.rejection());
    return parsed;
  }
  parsed.status = CommandLineStatus::run;
  parsed.diagnostic = "Success";
  return parsed;
}

void PrintCommandLineHelp(std::ostream& output)
{
  output <<
      "Usage: dg_igr_hdg_runtime --config FILE\n"
      "   or: dg_igr_hdg_runtime [every value option below exactly once]\n\n"
      "  --card exp.hdg.euler.conventional.smooth_persistent.w8a.v1\n"
      "  --case conventional-smooth-persistent-v1\n"
      "  --space l2|h1                    (h1 reports UnsupportedDiscretization)\n"
      "  --mesh-family periodic-cartesian-quad --mesh-n 3|6|12\n"
      "  --order 1|2 --dt 0.05 --steps 3|20 --final-time 0.15|1.0\n"
      "  --backend cpu|cuda --device-id none|0\n"
      "  --output-dir PATH --output-steps 0,1,3|0,10,20\n"
      "  --evidence-dir PATH|none --validation-capture persistent-v1|none\n"
      "  --checkpoint-out PATH|none --checkpoint-step 1|10|none\n"
      "  --stop-after-checkpoint true|false --restart-from PATH|none\n"
      "  --max-retries 0|1\n"
      "  --inject-failure off|PHASE:TARGET_STEP:RANK\n"
      "  --seed UINT64\n"
      "  -h, --help\n\n"
      "Config files use the same names without leading '--', one key=value per line; "
      "duplicates, unknown keys, and surrounding whitespace are rejected.\n"
      "Injection PHASE is Assembly, Condensation, TraceSolve, Recovery, Diagnostics, "
      "CheckpointStaging, OutputStaging, or AfterRecoveryBeforeCommit.\n";
}

std::string_view ToString(const ApplicationRejection value) noexcept
{
  switch (value)
  {
#define DG_IGR_APP_REJECTION(name) case ApplicationRejection::name: return #name
    DG_IGR_APP_REJECTION(none);
    DG_IGR_APP_REJECTION(unknown_card);
    DG_IGR_APP_REJECTION(unknown_case);
    DG_IGR_APP_REJECTION(unknown_scenario);
    DG_IGR_APP_REJECTION(unknown_tuple);
    DG_IGR_APP_REJECTION(unsupported_h1);
    DG_IGR_APP_REJECTION(invalid_interior_space);
    DG_IGR_APP_REJECTION(unknown_backend);
    DG_IGR_APP_REJECTION(invalid_device);
    DG_IGR_APP_REJECTION(unbuilt_backend);
    DG_IGR_APP_REJECTION(invalid_scientific_identity);
    DG_IGR_APP_REJECTION(invalid_config_hash);
    DG_IGR_APP_REJECTION(invalid_polynomial_mesh_pair);
    DG_IGR_APP_REJECTION(invalid_quadrature);
    DG_IGR_APP_REJECTION(invalid_dt);
    DG_IGR_APP_REJECTION(invalid_step_count);
    DG_IGR_APP_REJECTION(invalid_final_time);
    DG_IGR_APP_REJECTION(invalid_output_schedule);
    DG_IGR_APP_REJECTION(invalid_checkpoint_schedule);
    DG_IGR_APP_REJECTION(invalid_checkpoint_identity);
    DG_IGR_APP_REJECTION(invalid_restart_mode);
    DG_IGR_APP_REJECTION(invalid_restart_trajectory);
    DG_IGR_APP_REJECTION(invalid_run_role);
    DG_IGR_APP_REJECTION(invalid_path_identity);
    DG_IGR_APP_REJECTION(invalid_validation_capture);
    DG_IGR_APP_REJECTION(invalid_checkpoint_control);
    DG_IGR_APP_REJECTION(invalid_stop_control);
    DG_IGR_APP_REJECTION(invalid_injection_control);
    DG_IGR_APP_REJECTION(invalid_failure_injection);
    DG_IGR_APP_REJECTION(invalid_failure_phase);
    DG_IGR_APP_REJECTION(invalid_rollback_obligation);
    DG_IGR_APP_REJECTION(invalid_source_rank_rule);
    DG_IGR_APP_REJECTION(invalid_retry_limit);
    DG_IGR_APP_REJECTION(invalid_seed);
#undef DG_IGR_APP_REJECTION
  }
  return "unknown_rejection";
}

std::string_view ToString(const RunStatus value) noexcept
{
  switch (value)
  {
    case RunStatus::ready: return "ready";
    case RunStatus::attempt_active: return "attempt_active";
    case RunStatus::retry_available: return "retry_available";
    case RunStatus::committed: return "Success";
    case RunStatus::terminal_failure: return "RuntimeFailure";
    case RunStatus::unsupported_h1: return "UnsupportedDiscretization";
    case RunStatus::invalid_request: return "ConfigurationError";
  }
  return "unknown_status";
}

std::string_view ToString(const FailureReason value) noexcept
{
  switch (value)
  {
    case FailureReason::none: return "none";
    case FailureReason::invalid_resource_descriptor: return "invalid_resource_descriptor";
    case FailureReason::unsupported_h1: return "unsupported_h1";
    case FailureReason::invalid_interior_space: return "invalid_interior_space";
    case FailureReason::invalid_dt: return "invalid_dt";
    case FailureReason::incarnation_exhausted: return "incarnation_exhausted";
    case FailureReason::coordinate_overflow: return "coordinate_overflow";
    case FailureReason::candidate_preparation: return "candidate_preparation";
    case FailureReason::shared_backing_store: return "shared_backing_store";
    case FailureReason::publication_precondition: return "publication_precondition";
    case FailureReason::stale_attempt: return "stale_attempt";
    case FailureReason::collective_protocol: return "collective_protocol";
    case FailureReason::restart_incompatible: return "restart_incompatible";
    case FailureReason::missing_product: return "missing_product";
    case FailureReason::invalid_product: return "invalid_product";
    case FailureReason::stale_product: return "stale_product";
    case FailureReason::finite_gate: return "finite_gate";
    case FailureReason::admissibility_gate: return "admissibility_gate";
    case FailureReason::recovery_gate: return "recovery_gate";
    case FailureReason::true_residual_gate: return "true_residual_gate";
    case FailureReason::telemetry_gate: return "telemetry_gate";
    case FailureReason::success_gate: return "success_gate";
    case FailureReason::backend_failure: return "backend_failure";
    case FailureReason::retry_exhausted: return "retry_exhausted";
    case FailureReason::count: return "count";
  }
  return "unknown_failure";
}

ApplicationExit ExitForRejection(const ApplicationRejection rejection) noexcept
{
  return rejection == ApplicationRejection::unsupported_h1
      ? ApplicationExit::unsupported_discretization
      : ApplicationExit::configuration_error;
}

ApplicationExit ExitForResult(const ApplicationRunResult& result) noexcept
{
  if (result.status == RunStatus::committed &&
      result.reason == FailureReason::none &&
      result.failure_class == ApplicationRunResult::FailureClass::none)
    return ApplicationExit::success;
  if (result.failure_class == ApplicationRunResult::FailureClass::checkpoint_data ||
      result.reason == FailureReason::restart_incompatible)
    return ApplicationExit::restart_identity_mismatch;
  if (result.failure_class == ApplicationRunResult::FailureClass::scientific)
    return ApplicationExit::scientific_gate_failure;
  if (result.failure_class == ApplicationRunResult::FailureClass::injected)
    return ApplicationExit::injected_failure;
  if (result.status == RunStatus::unsupported_h1)
    return ApplicationExit::unsupported_discretization;
  if (result.status == RunStatus::invalid_request ||
      result.failure_class == ApplicationRunResult::FailureClass::configuration)
    return ApplicationExit::configuration_error;
  return ApplicationExit::runtime_failure;
}

} // namespace dg::hdg::runtime::app
