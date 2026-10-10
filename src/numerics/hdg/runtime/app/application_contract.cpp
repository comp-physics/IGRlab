#include "numerics/hdg/runtime/app/application_contract.hpp"

#include "../cpu/hdg_cpu_serial_lifecycle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#ifndef DG_IGR_HDG_APP_HAS_CPU_SERIAL
#error "build-owned CPU serial availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_CPU_RANK2
#error "build-owned CPU rank-2 availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_CUDA_RANK1
#error "build-owned CUDA rank-1 availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_CUDA_RANK2
#error "build-owned CUDA rank-2 availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
#error "build-owned scalar-IGR CPU availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1
#error "build-owned scalar-IGR CUDA rank-1 availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2
#error "build-owned scalar-IGR CUDA rank-2 availability definition is required"
#endif

namespace dg::hdg::runtime::app
{
namespace
{

class CanonicalHashWriter final
{
public:
  bool U8(const std::uint8_t value) noexcept
  {
    return hash_.Update(&value, sizeof(value));
  }

  template <typename Unsigned>
  bool UnsignedLittleEndian(const Unsigned value) noexcept
  {
    static_assert(std::is_unsigned<Unsigned>::value,
                  "canonical application hashing requires unsigned fields");
    std::array<std::uint8_t, sizeof(Unsigned)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
      bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
    }
    return hash_.Update(bytes.data(), bytes.size());
  }

  bool Text(const std::string_view value) noexcept
  {
    return value.size() <= std::numeric_limits<std::uint32_t>::max() &&
        UnsignedLittleEndian(static_cast<std::uint32_t>(value.size())) &&
        hash_.Update(value);
  }

  bool Hash(const StateHash& value) noexcept
  {
    return hash_.Update(value.bytes.data(), value.bytes.size());
  }

  bool Finalize(StateHash& value) noexcept
  {
    Sha256Digest digest{};
    if (!hash_.Finalize(digest)) { return false; }
    value.bytes = digest;
    return true;
  }

private:
  Sha256 hash_{};
};

struct ApplicationConfigHashes
{
  StateHash scientific{};
  StateHash run{};
};

template <typename Path>
bool CanonicalPathViewValid(const Path& path) noexcept
{
  const StableId<512> copied = StableId<512>::Copy(path.canonical);
  return path.present
      ? copied.valid() && !copied.empty()
      : path.canonical.empty();
}

template <typename Path>
bool HashPath(CanonicalHashWriter& writer,
              const std::string_view domain,
              const Path& path) noexcept
{
  return writer.Text(domain) && writer.U8(path.present ? 1U : 0U) &&
      (!path.present || writer.Text(path.canonical));
}

bool HashTuple(CanonicalHashWriter& writer,
               const RuntimeTupleDescriptor& tuple) noexcept
{
  return writer.Text(tuple.model.view()) &&
      writer.Text(tuple.formulation.view()) &&
      writer.Text(tuple.trace.view()) &&
      writer.Text(tuple.stabilization.view()) &&
      writer.Text(tuple.time_treatment.view()) &&
      writer.Text(tuple.backend.view()) && writer.Text(tuple.precision.view());
}

bool ComputeApplicationConfigHashes(
    const ApplicationOptionsView& options,
    const ApplicationScenarioV1 scenario,
    const ApplicationBackend backend,
    const RuntimeTupleDescriptor& tuple,
    const DeferredRuntimePredicatesV1& deferred,
    ApplicationConfigHashes& hashes) noexcept
{
  const bool w8c_case = cpu::w8c::IsCaseId(options.case_id);
  const bool smooth_transition =
      options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id;
  CanonicalHashWriter scientific;
  if (!scientific.Text(
          smooth_transition ? "DGIGR-HDG-W8C-SMOOTH-SCIENTIFIC-CONFIG-V2" :
          w8c_case ? "DGIGR-HDG-W8C-SCIENTIFIC-CONFIG-V1" :
          "DGIGR-HDG-W8A-SCIENTIFIC-CONFIG-V1") ||
      !scientific.Text(options.card_id) || !scientific.Text(options.case_id) ||
      !scientific.U8(static_cast<std::uint8_t>(scenario)) ||
      !scientific.U8(static_cast<std::uint8_t>(options.interior_space)) ||
      !scientific.Text(options.mesh_family_id) ||
      !scientific.UnsignedLittleEndian(
          static_cast<std::uint32_t>(options.polynomial_order)) ||
      !scientific.UnsignedLittleEndian(options.mesh_n) ||
      !scientific.UnsignedLittleEndian(
          ::dg::hdg::runtime::detail::DoubleBits(options.dt)) ||
      !scientific.UnsignedLittleEndian(options.accepted_steps) ||
      !scientific.UnsignedLittleEndian(
          ::dg::hdg::runtime::detail::DoubleBits(options.final_time)) ||
      !scientific.U8(static_cast<std::uint8_t>(backend)) ||
      !scientific.U8(static_cast<std::uint8_t>(options.device_id + 1)) ||
      !scientific.UnsignedLittleEndian(deferred.required_mpi_rank_count) ||
      !scientific.U8(static_cast<std::uint8_t>(deferred.gpu_rule)) ||
      !scientific.UnsignedLittleEndian(
          ::dg::hdg::runtime::detail::DoubleBits(options.gamma)) ||
      !scientific.UnsignedLittleEndian(
          ::dg::hdg::runtime::detail::DoubleBits(options.tau)) ||
      !scientific.UnsignedLittleEndian(
          static_cast<std::uint32_t>(options.q_a)) ||
      !scientific.UnsignedLittleEndian(
          static_cast<std::uint32_t>(options.q_e)) ||
      !scientific.UnsignedLittleEndian(
          static_cast<std::uint32_t>(options.q_e4)) ||
      !scientific.UnsignedLittleEndian(
          static_cast<std::uint32_t>(options.q_o)) ||
      !scientific.Text(options.source_id) ||
      !scientific.Text(options.flux_id) ||
      !scientific.Text(options.time_rule_id) ||
      !scientific.Text(options.layout_id) || !HashTuple(scientific, tuple) ||
      !scientific.Finalize(hashes.scientific))
  {
    return false;
  }

  CanonicalHashWriter run;
  if (!run.Text(
          smooth_transition ? "DGIGR-HDG-W8C-SMOOTH-RUN-CONFIG-V2" :
          w8c_case ? "DGIGR-HDG-W8C-RUN-CONFIG-V1" :
          "DGIGR-HDG-W8A-RUN-CONFIG-V1") ||
      !run.Hash(hashes.scientific) ||
      !run.U8(static_cast<std::uint8_t>(options.run_role)) ||
      !HashPath(run, "output-directory", options.output_directory) ||
      !HashPath(run, "evidence-directory", options.evidence_directory) ||
      !run.U8(static_cast<std::uint8_t>(options.validation_capture)) ||
      !HashPath(run, "checkpoint-artifact-out", options.checkpoint_out) ||
      !run.U8(options.checkpoint_step.present ? 1U : 0U) ||
      (options.checkpoint_step.present &&
       !run.UnsignedLittleEndian(options.checkpoint_step.value)) ||
      !HashPath(run, "checkpoint-artifact-restart", options.restart_from) ||
      !run.U8(static_cast<std::uint8_t>(options.stop_after_checkpoint)) ||
      !run.U8(static_cast<std::uint8_t>(options.injection.mode)) ||
      !run.U8(static_cast<std::uint8_t>(options.injection.phase)) ||
      !run.UnsignedLittleEndian(options.injection.target_step) ||
      !run.UnsignedLittleEndian(options.injection.source_rank) ||
      !run.U8(static_cast<std::uint8_t>(options.restart_mode)) ||
      !run.U8(static_cast<std::uint8_t>(options.restart_trajectory)) ||
      !run.U8(static_cast<std::uint8_t>(options.rollback_obligation)) ||
      !run.U8(static_cast<std::uint8_t>(options.source_rank_rule)) ||
      !run.UnsignedLittleEndian(options.retry_limit) ||
      !run.UnsignedLittleEndian(options.seed) ||
      !run.Text(options.checkpoint_id))
  {
    return false;
  }
  for (const std::uint64_t step : options.output_steps)
  {
    if (!run.UnsignedLittleEndian(step)) { return false; }
  }
  return run.Finalize(hashes.run);
}

bool ExactFrozenTuple(const ApplicationOptionsView& options) noexcept
{
  return options.card_id == smooth_periodic_euler_v1_card_id &&
      options.model_id == EulerCompressible2dTag::id &&
      options.formulation_id == ConventionalHdgTag::id &&
      options.trace_id == MfemDgInterfaceTraceTag::id &&
      options.stabilization_id == FixedScalarTauTag::id &&
      options.time_treatment_id == BackwardEulerRuntimeTag::id &&
      options.precision_id == Fp64RuntimeTag::id;
}

bool ExactScientificIdentities(const ApplicationOptionsView& options) noexcept
{
  const bool exact_jump =
      options.case_id == cpu::w8c::ExactJumpV1Case::case_id;
  const bool smooth_transition =
      options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id;
  return options.mesh_family_id ==
          smooth_periodic_euler_v1_mesh_family_id &&
      options.source_id ==
          (smooth_transition
               ? cpu::w8c::SmoothTransitionV2Case::conventional_source_id
               : (exact_jump
                      ? cpu::w8c::ExactJumpV1Case::conventional_source_id
                      : smooth_periodic_euler_v1_source_id)) &&
      options.flux_id == smooth_periodic_euler_v1_flux_id &&
      options.time_rule_id == smooth_periodic_euler_v1_time_rule_id &&
      options.layout_id == smooth_periodic_euler_v1_layout_id &&
      options.gamma == smooth_periodic_euler_v1_gamma &&
      options.tau == smooth_periodic_euler_v1_tau;
}

bool SmoothTransitionScenarioId(const std::string_view scenario) noexcept
{
  return scenario == cpu::w8c::SmoothTransitionV2Case::scenario_id ||
      scenario == cpu::w8c::SmoothTransitionN16Dt11Case::scenario_id ||
      scenario == cpu::w8c::SmoothTransitionN16Dt12Case::scenario_id ||
      scenario == cpu::w8c::SmoothTransitionN32Dt12Case::scenario_id;
}

RuntimeTupleDescriptor TupleForBackend(const ApplicationBackend backend) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return CpuSerialRuntimeSelection::Descriptor();
    case ApplicationBackend::cpu_rank2:
      return CpuRank2RuntimeSelection::Descriptor();
    case ApplicationBackend::cuda_rank1_one_gpu:
      return CudaRank1OneGpuRuntimeSelection::Descriptor();
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return CudaRank2OneGpuHostStagedRuntimeSelection::Descriptor();
    case ApplicationBackend::unknown: return {};
  }
  return {};
}

DeferredRuntimePredicatesV1 DeferredForBackend(
    const ApplicationBackend backend) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return {1, DeferredGpuRuleV1::no_cuda_device,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              0};
    case ApplicationBackend::cpu_rank2:
      return {2, DeferredGpuRuleV1::no_cuda_device,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              1};
    case ApplicationBackend::cuda_rank1_one_gpu:
      return {1, DeferredGpuRuleV1::one_physical_gpu,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              0};
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return {2, DeferredGpuRuleV1::one_shared_physical_gpu,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              1};
    case ApplicationBackend::unknown: return {};
  }
  return {};
}

bool ExactEndTime(const ApplicationOptionsView& options) noexcept
{
  if (!std::isfinite(options.dt) || !std::isfinite(options.final_time))
  {
    return false;
  }
  const double reconstructed =
      static_cast<double>(options.accepted_steps) * options.dt;
  const double scale = std::max(1.0, std::abs(options.final_time));
  return std::abs(options.final_time - reconstructed) <=
      8.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool CanonicalOffInjection(
    const FailureInjectionControlV1& injection) noexcept
{
  return injection.mode == FailureInjectionModeV1::off &&
      injection.phase == FailurePhase::Setup && injection.target_step == 0 &&
      injection.source_rank == 0;
}

bool FocusedInjectionPhase(const FailurePhase phase) noexcept
{
  switch (phase)
  {
    case FailurePhase::Assembly:
    case FailurePhase::Condensation:
    case FailurePhase::TraceSolve:
    case FailurePhase::Recovery:
    case FailurePhase::Diagnostics:
    case FailurePhase::CheckpointStaging:
    case FailurePhase::OutputStaging:
    case FailurePhase::AfterRecoveryBeforeCommit: return true;
    case FailurePhase::Setup:
    case FailurePhase::Commit:
    case FailurePhase::RollbackAudit:
    case FailurePhase::TerminalEvidence: return false;
  }
  return false;
}

std::uint64_t ScenarioCheckpointStep(
    const ApplicationOptionsView& options,
    const ApplicationScenarioV1 scenario) noexcept
{
  if (options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id)
  {
    if (options.scenario_id == cpu::w8c::SmoothTransitionN16Dt11Case::scenario_id)
      return cpu::w8c::SmoothTransitionN16Dt11Case::checkpoint_step;
    if (options.scenario_id == cpu::w8c::SmoothTransitionN16Dt12Case::scenario_id ||
        options.scenario_id == cpu::w8c::SmoothTransitionN32Dt12Case::scenario_id)
      return cpu::w8c::SmoothTransitionN16Dt12Case::checkpoint_step;
    return cpu::w8c::SmoothTransitionV2Case::checkpoint_step;
  }
  if (options.case_id == cpu::w8c::ExactJumpV1Case::case_id)
    return cpu::w8c::ExactJumpV1Case::checkpoint_step;
  return scenario == ApplicationScenarioV1::smoke
      ? smooth_periodic_euler_v1_smoke_checkpoint_step
      : smooth_periodic_euler_v1_functional_checkpoint_step;
}

std::uint64_t ScenarioFailureTarget(
    const ApplicationOptionsView& options,
    const ApplicationScenarioV1 scenario) noexcept
{
  if (options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id)
    return cpu::w8c::SmoothTransitionV2Case::failure_target_step;
  if (options.case_id == cpu::w8c::ExactJumpV1Case::case_id)
    return cpu::w8c::ExactJumpV1Case::failure_target_step;
  return scenario == ApplicationScenarioV1::smoke
      ? smooth_periodic_euler_v1_smoke_failure_target_step
      : smooth_periodic_euler_v1_functional_failure_target_step;
}

std::uint64_t ScenarioSeed(const ApplicationOptionsView& options,
                           const ApplicationScenarioV1 scenario) noexcept
{
  if (options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id)
    return cpu::w8c::SmoothTransitionV2Case::seed;
  if (options.case_id == cpu::w8c::ExactJumpV1Case::case_id)
    return cpu::w8c::ExactJumpV1Case::seed;
  return scenario == ApplicationScenarioV1::smoke
      ? smooth_periodic_euler_v1_smoke_seed
      : smooth_periodic_euler_v1_functional_seed;
}

ApplicationRejection ValidateRoleControls(
    const ApplicationOptionsView& options,
    const ApplicationScenarioV1 scenario,
    const ApplicationBackend backend,
    const DeferredRuntimePredicatesV1& deferred) noexcept
{
  if (!CanonicalPathViewValid(options.output_directory) ||
      !CanonicalPathViewValid(options.evidence_directory) ||
      !CanonicalPathViewValid(options.checkpoint_out) ||
      !CanonicalPathViewValid(options.restart_from) ||
      !options.output_directory.present)
  {
    return ApplicationRejection::invalid_path_identity;
  }
  if (options.validation_capture != ValidationCaptureModeV1::none ||
      options.evidence_directory.present)
  {
    return ApplicationRejection::invalid_validation_capture;
  }
  if (!options.checkpoint_step.present && options.checkpoint_step.value != 0)
  {
    return ApplicationRejection::invalid_checkpoint_control;
  }
  if (options.stop_after_checkpoint != StopAfterCheckpointV1::no &&
      options.stop_after_checkpoint != StopAfterCheckpointV1::yes)
  {
    return ApplicationRejection::invalid_stop_control;
  }
  if (options.injection.mode != FailureInjectionModeV1::off &&
      options.injection.mode != FailureInjectionModeV1::prescribed)
  {
    return ApplicationRejection::invalid_injection_control;
  }

  const bool w8c_case = cpu::w8c::IsCaseId(options.case_id);
  const std::uint64_t checkpoint_step =
      ScenarioCheckpointStep(options, scenario);
  const std::uint64_t seed = ScenarioSeed(options, scenario);
  const bool writes_checkpoint = options.checkpoint_out.present &&
      options.checkpoint_step.present &&
      options.checkpoint_step.value == checkpoint_step &&
      !options.restart_from.present;
  const bool reads_checkpoint = !options.checkpoint_out.present &&
      !options.checkpoint_step.present && options.restart_from.present;
  const bool off = CanonicalOffInjection(options.injection);
  const bool prescribed_principal =
      options.injection.mode == FailureInjectionModeV1::prescribed &&
      options.injection.phase == smooth_periodic_euler_v1_failure_phase &&
      options.injection.target_step ==
          ScenarioFailureTarget(options, scenario) &&
      options.injection.source_rank == deferred.required_injected_source_rank;
  if (w8c_case)
  {
    const bool focused =
        options.injection.mode == FailureInjectionModeV1::prescribed &&
        options.injection.phase == FailurePhase::AfterRecoveryBeforeCommit &&
        options.injection.target_step == cpu::w8c::failure_target_step &&
        options.injection.source_rank == 0 &&
        options.retry_limit == cpu::w8c::retry_limit;
    const bool ordinary = off && options.retry_limit == 0;
    switch (options.run_role)
    {
      case ApplicationRunRoleV1::uninterrupted_reference:
        return writes_checkpoint &&
                options.stop_after_checkpoint == StopAfterCheckpointV1::no &&
                ordinary &&
                options.seed == seed
            ? ApplicationRejection::none
            : ApplicationRejection::invalid_run_role;
      case ApplicationRunRoleV1::split_writer:
        return writes_checkpoint &&
                options.stop_after_checkpoint == StopAfterCheckpointV1::yes &&
                ordinary &&
                options.seed == seed
            ? ApplicationRejection::none
            : ApplicationRejection::invalid_run_role;
      case ApplicationRunRoleV1::split_reader:
        return reads_checkpoint &&
                options.stop_after_checkpoint == StopAfterCheckpointV1::no &&
                ordinary && options.seed == seed
            ? ApplicationRejection::none
            : ApplicationRejection::invalid_run_role;
      case ApplicationRunRoleV1::focused_terminal_injection:
        return writes_checkpoint &&
                options.stop_after_checkpoint == StopAfterCheckpointV1::no &&
                focused && options.seed == seed
            ? ApplicationRejection::none
            : ApplicationRejection::invalid_run_role;
      case ApplicationRunRoleV1::alternate_seed_reference:
        return ApplicationRejection::invalid_run_role;
    }
  }

  switch (options.run_role)
  {
    case ApplicationRunRoleV1::uninterrupted_reference:
      if (!writes_checkpoint ||
          options.stop_after_checkpoint != StopAfterCheckpointV1::no ||
          !off || options.retry_limit != 0 ||
          options.seed != seed)
      {
        return ApplicationRejection::invalid_run_role;
      }
      return ApplicationRejection::none;
    case ApplicationRunRoleV1::split_writer:
      if (!writes_checkpoint ||
          options.stop_after_checkpoint != StopAfterCheckpointV1::yes ||
          !off || options.retry_limit != 0 ||
          options.seed != seed)
      {
        return ApplicationRejection::invalid_run_role;
      }
      return ApplicationRejection::none;
    case ApplicationRunRoleV1::split_reader:
      if (!reads_checkpoint ||
          options.stop_after_checkpoint != StopAfterCheckpointV1::no ||
          !prescribed_principal ||
          options.retry_limit != smooth_periodic_euler_v1_retry_limit ||
          options.seed != seed)
      {
        return ApplicationRejection::invalid_run_role;
      }
      return ApplicationRejection::none;
    case ApplicationRunRoleV1::focused_terminal_injection:
    {
      const bool rank_valid = deferred.required_mpi_rank_count == 1
          ? options.injection.source_rank == 0
          : options.injection.source_rank < deferred.required_mpi_rank_count;
      const bool staging =
          options.injection.phase == FailurePhase::CheckpointStaging ||
          options.injection.phase == FailurePhase::OutputStaging;
      const std::uint64_t target = staging ? checkpoint_step : 2;
      if (scenario != ApplicationScenarioV1::smoke || !writes_checkpoint ||
          options.stop_after_checkpoint != StopAfterCheckpointV1::no ||
          options.injection.mode != FailureInjectionModeV1::prescribed ||
          !FocusedInjectionPhase(options.injection.phase) ||
          options.injection.target_step != target || !rank_valid ||
          options.retry_limit != smooth_periodic_euler_v1_retry_limit ||
          options.seed != seed)
      {
        return ApplicationRejection::invalid_run_role;
      }
      return ApplicationRejection::none;
    }
    case ApplicationRunRoleV1::alternate_seed_reference:
      if (scenario != ApplicationScenarioV1::functional ||
          backend != ApplicationBackend::cpu_serial || options.mesh_n != 6 ||
          !writes_checkpoint ||
          options.stop_after_checkpoint != StopAfterCheckpointV1::no ||
          !off || options.retry_limit != 0 ||
          options.seed != smooth_periodic_euler_v1_alternate_functional_seed)
      {
        return ApplicationRejection::invalid_run_role;
      }
      return ApplicationRejection::none;
  }
  return ApplicationRejection::invalid_run_role;
}

ApplicationRejection ValidateSmoke(
    const ApplicationOptionsView& options) noexcept
{
  if (options.polynomial_order !=
          smooth_periodic_euler_v1_smoke_polynomial_order ||
      options.mesh_n != smooth_periodic_euler_v1_smoke_mesh_n)
  {
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  }
  if (options.q_a != smooth_periodic_euler_v1_smoke_q_a ||
      options.q_e != smooth_periodic_euler_v1_smoke_q_e ||
      options.q_e4 != smooth_periodic_euler_v1_smoke_q_e4 ||
      options.q_o != smooth_periodic_euler_v1_smoke_q_o)
  {
    return ApplicationRejection::invalid_quadrature;
  }
  if (options.accepted_steps != smooth_periodic_euler_v1_smoke_steps)
  {
    return ApplicationRejection::invalid_step_count;
  }
  if (options.final_time != smooth_periodic_euler_v1_smoke_final_time ||
      !ExactEndTime(options))
  {
    return ApplicationRejection::invalid_final_time;
  }
  if (options.output_steps != smooth_periodic_euler_v1_smoke_output_steps)
  {
    return ApplicationRejection::invalid_output_schedule;
  }
  if (options.checkpoint_id !=
      smooth_periodic_euler_v1_smoke_checkpoint_id)
  {
    return ApplicationRejection::invalid_checkpoint_identity;
  }
  return ApplicationRejection::none;
}

ApplicationRejection ValidateFunctional(
    const ApplicationOptionsView& options,
    const ApplicationBackend backend) noexcept
{
  if (options.polynomial_order !=
      smooth_periodic_euler_v1_functional_polynomial_order)
  {
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  }
  const bool serial_ladder = backend == ApplicationBackend::cpu_serial &&
      (options.mesh_n == 3 || options.mesh_n == 6 || options.mesh_n == 12);
  const bool principal_other_backend =
      backend != ApplicationBackend::cpu_serial &&
      options.mesh_n == smooth_periodic_euler_v1_functional_mesh_n;
  if (!serial_ladder && !principal_other_backend)
  {
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  }
  if (options.q_a != smooth_periodic_euler_v1_functional_q_a ||
      options.q_e != smooth_periodic_euler_v1_functional_q_e ||
      options.q_e4 != smooth_periodic_euler_v1_functional_q_e4 ||
      options.q_o != smooth_periodic_euler_v1_functional_q_o)
  {
    return ApplicationRejection::invalid_quadrature;
  }
  if (options.accepted_steps != smooth_periodic_euler_v1_functional_steps)
  {
    return ApplicationRejection::invalid_step_count;
  }
  if (options.final_time != smooth_periodic_euler_v1_functional_final_time ||
      !ExactEndTime(options))
  {
    return ApplicationRejection::invalid_final_time;
  }
  if (options.output_steps !=
      smooth_periodic_euler_v1_functional_output_steps)
  {
    return ApplicationRejection::invalid_output_schedule;
  }
  if (options.checkpoint_id !=
      smooth_periodic_euler_v1_functional_checkpoint_id)
  {
    return ApplicationRejection::invalid_checkpoint_identity;
  }
  return ApplicationRejection::none;
}

template <typename Case>
ApplicationRejection ValidateW8c(
    const ApplicationOptionsView& options,
    const ApplicationBackend backend) noexcept
{
  if (backend != ApplicationBackend::cpu_serial)
    return ApplicationRejection::unknown_backend;
  if (options.case_id != Case::case_id || options.scenario_id != Case::scenario_id)
    return ApplicationRejection::unknown_scenario;
  if (options.polynomial_order != Case::polynomial_order ||
      options.mesh_n != Case::mesh_n)
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  if (options.q_a != Case::conventional_q_a ||
      options.q_e != 0 || options.q_e4 != 0 ||
      options.q_o != Case::observable_q_o)
    return ApplicationRejection::invalid_quadrature;
  if (options.dt != Case::dt)
    return ApplicationRejection::invalid_dt;
  if (options.accepted_steps != Case::accepted_steps)
    return ApplicationRejection::invalid_step_count;
  if (options.final_time != Case::final_time || !ExactEndTime(options))
    return ApplicationRejection::invalid_final_time;
  const std::array<std::uint64_t, 3> control_outputs{{
      0, Case::checkpoint_step, Case::accepted_steps}};
  if (options.output_steps != control_outputs)
    return ApplicationRejection::invalid_output_schedule;
  return options.checkpoint_id == Case::conventional_checkpoint_id
      ? ApplicationRejection::none
      : ApplicationRejection::invalid_checkpoint_identity;
}

} // namespace

HdgRunConfigV1::HdgRunConfigV1(
    const ApplicationOptionsView& options,
    const ApplicationScenarioV1 scenario,
    const ApplicationBackend backend,
    const RuntimeTupleDescriptor tuple,
    const DeferredRuntimePredicatesV1 deferred) noexcept
    : scenario_(scenario),
      backend_(backend),
      device_id_(options.device_id),
      tuple_(tuple),
      card_id_(StableIdentity::Copy(options.card_id)),
      case_id_(StableIdentity::Copy(options.case_id)),
      scenario_id_(StableIdentity::Copy(options.scenario_id)),
      checkpoint_id_(StableIdentity::Copy(options.checkpoint_id)),
      mesh_family_id_(StableIdentity::Copy(options.mesh_family_id)),
      source_id_(StableIdentity::Copy(options.source_id)),
      flux_id_(StableIdentity::Copy(options.flux_id)),
      time_rule_id_(StableIdentity::Copy(options.time_rule_id)),
      layout_id_(StableIdentity::Copy(options.layout_id)),
      interior_space_(options.interior_space),
      polynomial_order_(options.polynomial_order),
      mesh_n_(options.mesh_n),
      q_a_(options.q_a),
      q_e_(options.q_e),
      q_e4_(options.q_e4),
      q_o_(options.q_o),
      gamma_(options.gamma),
      tau_(options.tau),
      dt_(options.dt),
      accepted_steps_(options.accepted_steps),
      final_time_(options.final_time),
      output_steps_(options.output_steps),
      scenario_checkpoint_step_(ScenarioCheckpointStep(options, scenario)),
      run_role_(options.run_role),
      output_directory_(options.output_directory.present,
                        options.output_directory.canonical),
      evidence_directory_(options.evidence_directory.present,
                          options.evidence_directory.canonical),
      validation_capture_(options.validation_capture),
      checkpoint_out_(options.checkpoint_out.present,
                      options.checkpoint_out.canonical),
      checkpoint_step_(options.checkpoint_step),
      restart_from_(options.restart_from.present,
                    options.restart_from.canonical),
      stop_after_checkpoint_(options.stop_after_checkpoint),
      injection_(options.injection),
      restart_mode_(options.restart_mode),
      restart_trajectory_(options.restart_trajectory),
      rollback_obligation_(options.rollback_obligation),
      source_rank_rule_(options.source_rank_rule),
      retry_limit_(options.retry_limit),
      seed_(options.seed),
      deferred_runtime_(deferred)
{
  ApplicationConfigHashes hashes{};
  if (ComputeApplicationConfigHashes(options, scenario, backend, tuple,
                                     deferred, hashes))
  {
    scientific_config_sha256_ = hashes.scientific;
    run_config_sha256_ = hashes.run;
  }
}

ApplicationScenarioV1 ParseApplicationScenario(
    const std::string_view id) noexcept
{
  if (id == smooth_periodic_euler_v1_smoke_id)
  {
    return ApplicationScenarioV1::smoke;
  }
  if (id == smooth_periodic_euler_v1_functional_id)
  {
    return ApplicationScenarioV1::functional;
  }
  return ApplicationScenarioV1::unknown;
}

ApplicationBackend ParseApplicationBackend(const std::string_view id) noexcept
{
  if (id == CpuSerialBackendTag::id)
  {
    return ApplicationBackend::cpu_serial;
  }
  if (id == CpuRank2BackendTag::id)
  {
    return ApplicationBackend::cpu_rank2;
  }
  if (id == CudaRank1OneGpuBackendTag::id)
  {
    return ApplicationBackend::cuda_rank1_one_gpu;
  }
  if (id == CudaRank2OneGpuHostStagedBackendTag::id)
  {
    return ApplicationBackend::cuda_rank2_one_gpu_host_staged;
  }
  return ApplicationBackend::unknown;
}

bool CompiledBackendAvailable(const ApplicationBackend backend) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return DG_IGR_HDG_APP_HAS_CPU_SERIAL == 1;
    case ApplicationBackend::cpu_rank2:
      return DG_IGR_HDG_APP_HAS_CPU_RANK2 == 1;
    case ApplicationBackend::cuda_rank1_one_gpu:
      return DG_IGR_HDG_APP_HAS_CUDA_RANK1 == 1;
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return DG_IGR_HDG_APP_HAS_CUDA_RANK2 == 1;
    case ApplicationBackend::unknown: return false;
  }
  return false;
}

ApplicationValidation ValidateApplicationOptions(
    const ApplicationOptionsView& options) noexcept
{
  const auto reject = [](const ApplicationRejection reason) noexcept {
    return ApplicationValidation(reason);
  };
  const bool w8c_case = cpu::w8c::IsCaseId(options.case_id);
  if (options.card_id != smooth_periodic_euler_v1_card_id)
  {
    return reject(ApplicationRejection::unknown_card);
  }
  if (!w8c_case && options.case_id != smooth_periodic_euler_v1_case_id)
  {
    return reject(ApplicationRejection::unknown_case);
  }
  const ApplicationScenarioV1 scenario = w8c_case
      ? ApplicationScenarioV1::smoke
      : ParseApplicationScenario(options.scenario_id);
  if ((!w8c_case && scenario == ApplicationScenarioV1::unknown) ||
      (options.case_id == cpu::w8c::ExactJumpV1Case::case_id &&
       options.scenario_id != cpu::w8c::ExactJumpV1Case::scenario_id) ||
      (options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id &&
       !SmoothTransitionScenarioId(options.scenario_id)))
  {
    return reject(ApplicationRejection::unknown_scenario);
  }
  const bool scenario_belongs_to_card =
      scenario == ApplicationScenarioV1::smoke ||
      scenario == ApplicationScenarioV1::functional;
  if (!scenario_belongs_to_card)
  {
    return reject(ApplicationRejection::unknown_scenario);
  }
  if (options.interior_space == InteriorSpace::h1)
  {
    return reject(ApplicationRejection::unsupported_h1);
  }
  if (options.interior_space != InteriorSpace::l2)
  {
    return reject(ApplicationRejection::invalid_interior_space);
  }
  if (!ExactFrozenTuple(options))
  {
    return reject(ApplicationRejection::unknown_tuple);
  }
  const ApplicationBackend backend =
      ParseApplicationBackend(options.backend_id);
  if (backend == ApplicationBackend::unknown)
  {
    return reject(ApplicationRejection::unknown_backend);
  }
  const bool cpu_backend = backend == ApplicationBackend::cpu_serial ||
      backend == ApplicationBackend::cpu_rank2;
  if ((cpu_backend && options.device_id != -1) ||
      (!cpu_backend && options.device_id != 0))
  {
    return reject(ApplicationRejection::invalid_device);
  }
  if (!CompiledBackendAvailable(backend))
  {
    return reject(ApplicationRejection::unbuilt_backend);
  }
  if (!ExactScientificIdentities(options))
  {
    return reject(ApplicationRejection::invalid_scientific_identity);
  }
  if (!w8c_case && options.dt != smooth_periodic_euler_v1_dt)
  {
    return reject(ApplicationRejection::invalid_dt);
  }
  const ApplicationRejection scenario_rejection =
      options.case_id == cpu::w8c::SmoothTransitionV2Case::case_id
      ? (options.scenario_id == cpu::w8c::SmoothTransitionN16Dt11Case::scenario_id
             ? ValidateW8c<cpu::w8c::SmoothTransitionN16Dt11Case>(options, backend)
             : options.scenario_id == cpu::w8c::SmoothTransitionN16Dt12Case::scenario_id
             ? ValidateW8c<cpu::w8c::SmoothTransitionN16Dt12Case>(options, backend)
             : options.scenario_id == cpu::w8c::SmoothTransitionN32Dt12Case::scenario_id
             ? ValidateW8c<cpu::w8c::SmoothTransitionN32Dt12Case>(options, backend)
             : ValidateW8c<cpu::w8c::SmoothTransitionV2Case>(options, backend))
      : options.case_id == cpu::w8c::ExactJumpV1Case::case_id
      ? ValidateW8c<cpu::w8c::ExactJumpV1Case>(options, backend)
      : (scenario == ApplicationScenarioV1::smoke
             ? ValidateSmoke(options)
             : ValidateFunctional(options, backend));
  if (scenario_rejection != ApplicationRejection::none)
  {
    return reject(scenario_rejection);
  }
  if (options.restart_mode != RestartMode::same_backend_rank_partition)
  {
    return reject(ApplicationRejection::invalid_restart_mode);
  }
  if (options.restart_trajectory !=
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple)
  {
    return reject(ApplicationRejection::invalid_restart_trajectory);
  }
  if (options.rollback_obligation !=
      RollbackObligationV1::collective_exact_complete_state)
  {
    return reject(ApplicationRejection::invalid_rollback_obligation);
  }
  if (options.source_rank_rule != DeferredSourceRankRuleV1::
          rank_zero_for_rank1_rank_one_for_rank2)
  {
    return reject(ApplicationRejection::invalid_source_rank_rule);
  }
  const RuntimeTupleDescriptor tuple = TupleForBackend(backend);
  if (!tuple.Valid())
  {
    return reject(ApplicationRejection::unknown_tuple);
  }
  const DeferredRuntimePredicatesV1 deferred = DeferredForBackend(backend);
  const ApplicationRejection role_rejection =
      ValidateRoleControls(options, scenario, backend, deferred);
  if (role_rejection != ApplicationRejection::none)
  {
    return reject(role_rejection);
  }
  ApplicationConfigHashes hashes{};
  if (!ComputeApplicationConfigHashes(options, scenario, backend, tuple,
                                      deferred, hashes) ||
      hashes.scientific == StateHash{} || hashes.run == StateHash{})
  {
    return reject(ApplicationRejection::invalid_config_hash);
  }
  return ApplicationValidation(
      HdgRunConfigV1(options, scenario, backend, tuple, deferred));
}

namespace
{

HdgFormulationV1 FormulationFor(const std::string_view card) noexcept
{
  if (card == smooth_periodic_euler_v1_card_id)
    return HdgFormulationV1::conventional;
  return HdgFormulationV1::unknown;
}

HdgPeriodicCaseV1 PeriodicCaseFor(const std::string_view case_id) noexcept
{
  if (case_id == cpu::w8c::ExactJumpV1Case::case_id)
    return HdgPeriodicCaseV1::exact_jump;
  if (case_id == cpu::w8c::SmoothTransitionV2Case::case_id)
    return HdgPeriodicCaseV1::smooth_transition;
  if (case_id == smooth_periodic_euler_v1_case_id)
    return HdgPeriodicCaseV1::smooth;
  return HdgPeriodicCaseV1::unknown;
}

struct ClosedCapabilityRowV1
{
  HdgFormulationV1 formulation;
  ApplicationBackend backend;
  HdgPeriodicCaseV1 periodic_case;
};

/*
 * This is deliberately a closed table.  Tuple-specific validation remains in
 * the existing W8A/W8C validators below it, which keeps the established
 * error taxonomy and does not accidentally advertise a new numerical cell.
 */
constexpr std::array<ClosedCapabilityRowV1, 6> closed_capability_rows_v1{{
    {HdgFormulationV1::conventional, ApplicationBackend::cpu_serial,
     HdgPeriodicCaseV1::smooth},
    {HdgFormulationV1::conventional, ApplicationBackend::cpu_rank2,
     HdgPeriodicCaseV1::smooth},
    {HdgFormulationV1::conventional,
     ApplicationBackend::cuda_rank1_one_gpu, HdgPeriodicCaseV1::smooth},
    {HdgFormulationV1::conventional,
     ApplicationBackend::cuda_rank2_one_gpu_host_staged,
     HdgPeriodicCaseV1::smooth},
    {HdgFormulationV1::conventional, ApplicationBackend::cpu_serial,
     HdgPeriodicCaseV1::exact_jump},
    {HdgFormulationV1::conventional, ApplicationBackend::cpu_serial,
     HdgPeriodicCaseV1::smooth_transition},
}};

bool InClosedCapabilityTable(const HdgFormulationV1 formulation,
                             const ApplicationBackend backend,
                             const HdgPeriodicCaseV1 periodic_case) noexcept
{
  return std::any_of(closed_capability_rows_v1.begin(),
                     closed_capability_rows_v1.end(),
                     [=](const ClosedCapabilityRowV1& row) {
                       return row.formulation == formulation &&
                           row.backend == backend &&
                           row.periodic_case == periodic_case;
                     });
}

std::string_view CapabilityReasonText(const ApplicationRejection reason) noexcept
{
  switch (reason)
  {
    case ApplicationRejection::none: return "supported frozen HDG capability";
    case ApplicationRejection::unknown_backend:
      return "formulation/case/backend topology is not an admitted HDG cell";
    case ApplicationRejection::unbuilt_backend:
      return "the selected HDG capability is not built in this artifact";
    case ApplicationRejection::invalid_polynomial_mesh_pair:
      return "mesh/order tuple is outside the closed HDG capability table";
    case ApplicationRejection::invalid_dt:
    case ApplicationRejection::invalid_step_count:
    case ApplicationRejection::invalid_final_time:
      return "time tuple is outside the closed HDG capability table";
    case ApplicationRejection::invalid_output_schedule:
      return "output policy is not the frozen policy for this HDG capability";
    default: return "request does not satisfy the closed HDG contract";
  }
}

} // namespace

HdgOutputPolicyV1 OutputPolicyForV1(const HdgRunConfigV1& config) noexcept
{
  return {config.output_steps(),
          static_cast<std::uint8_t>(config.output_steps().size())};
}

HdgCheckpointIdentityV1 CheckpointIdentityForV1(
    const HdgRunConfigV1& config) noexcept
{
  /* scientific_config_sha256 deliberately excludes output path/cadence and
   * run role; codec v2 continues to own artifact layout validation. */
  return {config.scientific_config_sha256(), config.tuple(), config.mesh_n(),
          config.polynomial_order(),
          config.deferred_runtime().required_mpi_rank_count};
}

std::optional<PeriodicQuadCaseDescriptorV1> PeriodicQuadCaseProviderV1::Describe(
    const HdgRunConfigV1& config) noexcept
{
  const HdgPeriodicCaseV1 periodic_case = PeriodicCaseFor(config.case_id().view());
  if (periodic_case == HdgPeriodicCaseV1::unknown ||
      config.mesh_family_id().view() != smooth_periodic_euler_v1_mesh_family_id)
  {
    return std::nullopt;
  }
  const HdgInitialConditionV1 initial =
      periodic_case == HdgPeriodicCaseV1::exact_jump
      ? HdgInitialConditionV1::exact_jump
      : periodic_case == HdgPeriodicCaseV1::smooth_transition
          ? HdgInitialConditionV1::smooth_transition
          : HdgInitialConditionV1::smooth_analytic;
  return PeriodicQuadCaseDescriptorV1{
      2, smooth_periodic_euler_v1_mesh_family_id, config.tuple().model.view(),
      initial, "bc.periodic.v1", config.source_id().view(),
      config.flux_id().view(), config.layout_id().view()};
}

HdgMethodCapabilitiesV1 QueryHdgMethodCapabilitiesV1(
    const ApplicationOptionsView& options) noexcept
{
  const HdgFormulationV1 formulation = FormulationFor(options.card_id);
  const ApplicationBackend backend = ParseApplicationBackend(options.backend_id);
  const HdgPeriodicCaseV1 periodic_case = PeriodicCaseFor(options.case_id);
  HdgMethodCapabilitiesV1 result{formulation, backend, periodic_case, false,
                                 ApplicationRejection::unknown_tuple, {}};
  if (formulation == HdgFormulationV1::unknown)
  {
    const ApplicationValidation validation = ValidateApplicationOptions(options);
    result.reason = validation.rejection();
    result.reason_text = CapabilityReasonText(result.reason);
    return result;
  }
  if (!InClosedCapabilityTable(formulation, backend, periodic_case))
  {
    result.reason = ApplicationRejection::unknown_backend;
    result.reason_text = CapabilityReasonText(result.reason);
    return result;
  }
  const ApplicationValidation validation = ValidateApplicationOptions(options);
  result.supported = validation.accepted();
  result.reason = validation.rejection();
  result.reason_text = CapabilityReasonText(result.reason);
  return result;
}

HdgMethodCapabilitiesV1 CapabilitiesForV1(
    const HdgRunConfigV1& config) noexcept
{
  const HdgFormulationV1 formulation = FormulationFor(config.card_id().view());
  const HdgPeriodicCaseV1 periodic_case = PeriodicCaseFor(config.case_id().view());
  const bool supported = InClosedCapabilityTable(formulation, config.backend(),
                                                  periodic_case);
  const ApplicationRejection reason = supported ? ApplicationRejection::none
                                                 : ApplicationRejection::unknown_backend;
  return {formulation, config.backend(), periodic_case, supported, reason,
          CapabilityReasonText(reason)};
}

bool DeriveSplitWriterRunConfigHashV1(
    const ValidatedApplicationConfig& split_reader,
    StateHash& writer_run_config_sha256) noexcept
{
  if (split_reader.run_role() != ApplicationRunRoleV1::split_reader ||
      !split_reader.restart_from().present())
  {
    return false;
  }

  ApplicationOptionsView writer{};
  writer.card_id = split_reader.card_id().view();
  writer.case_id = split_reader.case_id().view();
  writer.scenario_id = split_reader.scenario_id().view();
  writer.model_id = split_reader.tuple().model.view();
  writer.formulation_id = split_reader.tuple().formulation.view();
  writer.trace_id = split_reader.tuple().trace.view();
  writer.stabilization_id = split_reader.tuple().stabilization.view();
  writer.time_treatment_id = split_reader.tuple().time_treatment.view();
  writer.backend_id = split_reader.tuple().backend.view();
  writer.device_id = split_reader.device_id();
  writer.precision_id = split_reader.tuple().precision.view();
  writer.mesh_family_id = split_reader.mesh_family_id().view();
  writer.source_id = split_reader.source_id().view();
  writer.flux_id = split_reader.flux_id().view();
  writer.time_rule_id = split_reader.time_rule_id().view();
  writer.layout_id = split_reader.layout_id().view();
  writer.interior_space = split_reader.interior_space();
  writer.polynomial_order = split_reader.polynomial_order();
  writer.mesh_n = split_reader.mesh_n();
  writer.q_a = split_reader.q_a();
  writer.q_e = split_reader.q_e();
  writer.q_e4 = split_reader.q_e4();
  writer.q_o = split_reader.q_o();
  writer.gamma = split_reader.gamma();
  writer.tau = split_reader.tau();
  writer.dt = split_reader.dt();
  writer.accepted_steps = split_reader.accepted_steps();
  writer.final_time = split_reader.final_time();
  writer.output_steps = split_reader.output_steps();
  writer.checkpoint_id = split_reader.checkpoint_id().view();
  writer.run_role = ApplicationRunRoleV1::split_writer;
  writer.output_directory = OutputDirectoryPathV1::Present(
      split_reader.output_directory().identity().view());
  writer.evidence_directory = split_reader.evidence_directory().present()
      ? EvidenceDirectoryPathV1::Present(
            split_reader.evidence_directory().identity().view())
      : EvidenceDirectoryPathV1{};
  writer.validation_capture = split_reader.validation_capture();
  writer.checkpoint_out = CheckpointArtifactPathV1::Present(
      split_reader.restart_from().identity().view());
  writer.checkpoint_step = OptionalCheckpointStepV1::Present(
      split_reader.scenario_checkpoint_step());
  writer.restart_from = {};
  writer.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  writer.injection = {};
  writer.restart_mode = split_reader.restart_mode();
  writer.restart_trajectory = split_reader.restart_trajectory();
  writer.rollback_obligation = split_reader.rollback_obligation();
  writer.source_rank_rule = split_reader.source_rank_rule();
  writer.retry_limit = 0;
  writer.seed = split_reader.seed();

  ApplicationConfigHashes hashes{};
  if (!ComputeApplicationConfigHashes(
          writer, split_reader.scenario(), split_reader.backend(),
          split_reader.tuple(), split_reader.deferred_runtime(), hashes) ||
      hashes.scientific != split_reader.scientific_config_sha256())
  {
    return false;
  }
  writer_run_config_sha256 = hashes.run;
  return true;
}

} // namespace dg::hdg::runtime::app
