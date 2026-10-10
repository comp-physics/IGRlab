#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/io/run_metadata.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/solver/solver.hpp"
#include "sim/solver/solver_helpers.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/time/time_defaults.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"

#include "test_utils.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace dg_igr_tests
{

int RunTimeDefaultsTests(TestContext& ctx)
{
  constexpr double gamma = 1.4;

  // --- Zhang-Shu point policy defaults and runtime resolution ---
  {
    dg_sim::CommonOptions opts;
    dg_sim::CaseConfig case_cfg;
    dg_sim::IGROptions igr_opts;
    ctx.Check(opts.positivity_limiter == "case",
              "positivity limiter defaults preserve the case policy");
    ctx.Check(opts.zhang_shu_points == "consumer",
              "Zhang-Shu defaults: consumer point set selected");
    ctx.Check(!opts.zhang_shu_vtk_points,
              "Zhang-Shu defaults: refined VTK points are disabled");
    ctx.Check(opts.hdg_igr_alpha == 1.0 && opts.hdg_igr_beta == 1.0,
              "HDG IGR defaults: alpha and beta are one");
    ctx.Check(opts.hdg_cuda_engine == "ordinary",
              "HDG CUDA engine defaults to ordinary");
    ctx.Check(opts.time_atol == 1.0e-8 && opts.time_rtol == 1.0e-5,
              "adaptive time tolerances retain their typed opt-in defaults");
    ctx.CheckNear(opts.cfl,
                  0.3,
                  0.0,
                  "explicit CFL defaults to an SSP-normalized safety factor");
    ctx.Check(dg_sim::ParseZhangShuPointSet("consumer") ==
                mfem::ZhangShuPointSet::kConsumer,
              "Zhang-Shu options: parse consumer point set");
    ctx.Check(dg_sim::ParseZhangShuPointSet("quadrature") ==
                mfem::ZhangShuPointSet::kQuadrature,
              "Zhang-Shu options: parse quadrature point set");
    ctx.Check(dg_sim::ParsePositivityLimiterSelection("case") ==
                  dg_sim::PositivityLimiterSelection::Case &&
                dg_sim::ParsePositivityLimiterSelection("none") ==
                  dg_sim::PositivityLimiterSelection::None &&
                dg_sim::ParsePositivityLimiterSelection("zhang-shu") ==
                  dg_sim::PositivityLimiterSelection::ZhangShu,
              "positivity limiter options have typed case/none/Zhang-Shu selectors");

    case_cfg.limiter_pipeline.zhang_shu = true;
    opts.positivity_limiter = "none";
    const dg_sim::RunConfig limiter_off =
      dg_sim::MakeRunConfig(case_cfg, opts, igr_opts);
    ctx.Check(!limiter_off.case_config.limiter_pipeline.zhang_shu &&
                limiter_off.opts.positivity_limiter == "none",
              "positivity limiter none overrides an enabled case default");

    case_cfg.limiter_pipeline.zhang_shu = false;
    opts.positivity_limiter = "zhang_shu";
    const dg_sim::RunConfig limiter_on =
      dg_sim::MakeRunConfig(case_cfg, opts, igr_opts);
    ctx.Check(limiter_on.case_config.limiter_pipeline.zhang_shu &&
                limiter_on.opts.positivity_limiter == "zhang-shu",
              "positivity limiter Zhang-Shu override is enabled and canonicalized");

    opts.positivity_limiter = "case";
    opts.zhang_shu_points = "quadrature";
    opts.zhang_shu_vtk_points = true;
    const dg_sim::RunConfig run_cfg =
      dg_sim::MakeRunConfig(case_cfg, opts, igr_opts);
    ctx.Check(run_cfg.case_config.limiter_pipeline.zhang_shu_points ==
                mfem::ZhangShuPointSet::kQuadrature &&
                run_cfg.case_config.limiter_pipeline.zhang_shu_vtk_points &&
                run_cfg.opts.zhang_shu_vtk_points,
              "Zhang-Shu options: CLI policy reaches limiter pipeline");
    ctx.Check(std::string(dg_sim::ZhangShuPointSetToString(
                mfem::ZhangShuPointSet::kConsumer)) == "consumer",
              "Zhang-Shu options: consumer policy has stable metadata name");
  }

  // --- General discretization selection and legacy backend adapter ---
  {
    const dg_sim::DiscretizationResolution default_selection =
      dg_sim::ResolveDiscretization("dgsem", "", "dgsem", false, false);
    ctx.Check(default_selection.IsResolved() &&
                default_selection.discretization ==
                  dg_sim::Discretization::DGSEM &&
                default_selection.provenance ==
                  dg_sim::DiscretizationProvenance::CaseDefault &&
                default_selection.dg_backend == dg_sim::DGBackend::DGSEM,
              "discretization: no CLI selector retains the per-case default");

    const dg_sim::DiscretizationResolution discretization_selection =
      dg_sim::ResolveDiscretization("dg", "DGSEM", "dg", true, false);
    ctx.Check(discretization_selection.IsResolved() &&
                discretization_selection.discretization ==
                  dg_sim::Discretization::DGSEM &&
                discretization_selection.provenance ==
                  dg_sim::DiscretizationProvenance::DiscretizationOption &&
                discretization_selection.dg_backend == dg_sim::DGBackend::DGSEM,
              "discretization: general selector overrides the case default");

    const dg_sim::DiscretizationResolution backend_selection =
      dg_sim::ResolveDiscretization("dgsem", "", "dg", false, true);
    ctx.Check(backend_selection.IsResolved() &&
                backend_selection.discretization ==
                  dg_sim::Discretization::DG &&
                backend_selection.provenance ==
                  dg_sim::DiscretizationProvenance::DGBackendOption &&
                backend_selection.dg_backend == dg_sim::DGBackend::DG,
              "discretization: legacy backend selector remains compatible");

    const dg_sim::DiscretizationResolution both_selection =
      dg_sim::ResolveDiscretization("dg", "DGSEM", "dgsem", true, true);
    ctx.Check(both_selection.IsResolved() &&
                both_selection.provenance ==
                  dg_sim::DiscretizationProvenance::BothOptions &&
                both_selection.dg_backend == dg_sim::DGBackend::DGSEM,
              "discretization: normalized equal selectors are accepted");

    const dg_sim::DiscretizationResolution conflict_selection =
      dg_sim::ResolveDiscretization("dg", "dg", "dgsem", true, true);
    ctx.Check(conflict_selection.status ==
                dg_sim::DiscretizationResolutionStatus::Conflict,
              "discretization: conflicting selectors are rejected");

    const dg_sim::DiscretizationResolution legacy_hdg =
      dg_sim::ResolveDiscretization("dg", "", "hdg", false, true);
    ctx.Check(legacy_hdg.status ==
                dg_sim::DiscretizationResolutionStatus::InvalidDGBackend,
              "discretization: legacy backend rejects hdg");

    const dg_sim::DiscretizationResolution hdg_selection =
      dg_sim::ResolveDiscretization("dg", "hdg", "dg", true, false);
    ctx.Check(hdg_selection.IsResolved() &&
                hdg_selection.discretization == dg_sim::Discretization::HDG &&
                !hdg_selection.dg_backend,
              "discretization: hdg resolves without a legacy DG backend");

    ctx.Check(!dg_sim::DGBackendForDiscretization(
                dg_sim::Discretization::HDG),
              "discretization: HDG has no legacy backend adapter");

    char program_name[] = "test";
    char discretization_flag[] = "--discretization";
    char discretization_value[] = "dg";
    char legacy_flag[] = "-dg";
    char legacy_value[] = "dg";
    char* argv[] = {program_name,
                    discretization_flag,
                    discretization_value,
                    legacy_flag,
                    legacy_value};
    ctx.Check(dg_sim::HasDiscretizationOptionOverride(5, argv) &&
                dg_sim::HasDGBackendOptionOverride(5, argv),
              "discretization: explicit selector provenance uses exact argv flags");

    dg_sim::CommonOptions opts;
    dg_sim::CaseConfig case_cfg;
    dg_sim::IGROptions igr_opts;
    opts.discretization = "dgsem";
    const dg_sim::RunConfig run_cfg = dg_sim::MakeRunConfig(
      case_cfg,
      opts,
      igr_opts,
      dg_sim::DiscretizationProvenance::DiscretizationOption);
    ctx.Check(run_cfg.discretization == dg_sim::Discretization::DGSEM &&
                run_cfg.discretization_provenance ==
                  dg_sim::DiscretizationProvenance::DiscretizationOption &&
                run_cfg.dg_backend == dg_sim::DGBackend::DGSEM,
              "discretization: run config records selector and compatibility backend");
  }

  // --- Orthogonal solver request selection and compatibility contract ---
  {
    ctx.Check(dg_sim::TryParseRegularization("IGR") ==
                dg_sim::Regularization::IGR &&
                std::string(dg_sim::RegularizationToString(
                  dg_sim::Regularization::None)) == "none",
              "request: regularization parser and metadata name are stable");
    ctx.Check(!dg_sim::TryParseRegularization("invalid"),
              "request: invalid regularization is not parsed");
    ctx.Check(dg_sim::TryParseTimeIntegrator("BACKWARD-EULER") ==
                dg_sim::TimeIntegrator::BackwardEuler &&
                dg_sim::TryParseTimeIntegrator(
                  "PARTITIONED-BACKWARD-EULER") ==
                  dg_sim::TimeIntegrator::PartitionedBackwardEuler &&
                dg_sim::TryParseTimeIntegrator("RODAS3") ==
                  dg_sim::TimeIntegrator::Rodas3 &&
                dg_sim::TryParseTimeIntegrator("SSPRK104") ==
                  dg_sim::TimeIntegrator::SSPRK104 &&
                std::string(dg_sim::TimeIntegratorToString(
                  dg_sim::TimeIntegrator::RK3SSP)) == "rk3ssp" &&
                std::string(dg_sim::TimeIntegratorToString(
                  dg_sim::TimeIntegrator::SSPRK104)) == "ssprk104" &&
                std::string(dg_sim::TimeIntegratorToString(
                  dg_sim::TimeIntegrator::PartitionedBackwardEuler)) ==
                  "partitioned-backward-euler" &&
                std::string(dg_sim::TimeIntegratorToString(
                  dg_sim::TimeIntegrator::Rodas3)) == "rodas3",
              "request: time-integrator parser and metadata name are stable");
    ctx.Check(dg_sim::SSPRKMethodForTimeIntegrator(
                dg_sim::TimeIntegrator::RK3SSP) ==
                  mfem::SSPRKMethod::SSPRK33 &&
                dg_sim::SSPRKMethodForTimeIntegrator(
                  dg_sim::TimeIntegrator::SSPRK104) ==
                  mfem::SSPRKMethod::SSPRK104,
              "request: typed explicit selectors map to SSPRK methods");
    ctx.Check(
      dg_sim::SSPCoefficientForTimeIntegrator(
        dg_sim::TimeIntegrator::RK3SSP) == 1.0 &&
        dg_sim::SSPCoefficientForTimeIntegrator(
          dg_sim::TimeIntegrator::SSPRK104) == 6.0 &&
        !dg_sim::SSPCoefficientForTimeIntegrator(
          dg_sim::TimeIntegrator::Rodas3),
      "request: SSP coefficients are authoritative only for explicit SSPRK");
    ctx.Check(!dg_sim::TryParseTimeIntegrator("invalid"),
              "request: invalid time integrator is not parsed");
    ctx.Check(dg_sim::TryParseDeviceBackend("CUDA") ==
                dg_sim::DeviceBackend::CUDA &&
                std::string(dg_sim::DeviceBackendToString(
                  dg_sim::DeviceBackend::CPU)) == "cpu",
              "request: device parser and metadata name are stable");
    ctx.Check(!dg_sim::TryParseDeviceBackend("invalid"),
              "request: invalid device backend is not parsed");
    ctx.Check(dg_sim::TryParseHdgCudaEngine("TENSOR") ==
                dg_sim::HdgCudaEngine::TensorCore &&
                std::string(dg_sim::HdgCudaEngineToString(
                  dg_sim::HdgCudaEngine::Ordinary)) == "ordinary",
              "request: HDG CUDA engine parser and metadata name are stable");
    ctx.Check(!dg_sim::TryParseHdgCudaEngine("invalid"),
              "request: invalid HDG CUDA engine is not parsed");
    ctx.Check(std::string(dg_sim::SolverRequestResolutionStatusToString(
                dg_sim::SolverRequestResolutionStatus::UnsupportedTimeIntegrator)) ==
                "unsupported-time-integrator",
              "request: resolution status has a stable metadata name");
    ctx.Check(std::string(dg_sim::SolverRequestResolutionStatusToString(
                dg_sim::SolverRequestResolutionStatus::InvalidDeviceBackend)) ==
                "invalid-device-backend",
              "request: device failure status has a stable metadata name");
    ctx.Check(std::string(dg_sim::SolverRequestResolutionStatusToString(
                dg_sim::SolverRequestResolutionStatus::InvalidHdgCudaEngine)) ==
                "invalid-hdg-cuda-engine" &&
                std::string(dg_sim::SolverRequestResolutionStatusToString(
                  dg_sim::SolverRequestResolutionStatus::
                    UnsupportedHdgCudaEngine)) ==
                  "unsupported-hdg-cuda-engine",
              "request: HDG CUDA engine statuses have stable metadata names");
    const dg_sim::SolverRequestResolution fail_safe_request;
    ctx.Check(fail_safe_request.regularization == dg_sim::Regularization::None &&
                fail_safe_request.device_backend == dg_sim::DeviceBackend::CPU &&
                fail_safe_request.hdg_cuda_engine ==
                  dg_sim::HdgCudaEngine::Ordinary,
              "request: default-constructed request is fail-safe");

    const dg_sim::SolverRequestResolution dg_default =
      dg_sim::ResolveSolverRequest("dg", true, "", "dg", "", "",
                                   false, false, false, false);
    ctx.Check(dg_default.IsResolved() &&
                dg_default.regularization == dg_sim::Regularization::IGR &&
                dg_default.time_integrator == dg_sim::TimeIntegrator::RK3SSP,
              "request: omitted DG selectors retain CaseSpec IGR and RK3SSP defaults");

    const dg_sim::SolverRequestResolution dgsem_default =
      dg_sim::ResolveSolverRequest("dgsem", false, "", "dgsem", "", "",
                                   false, false, false, false);
    ctx.Check(dgsem_default.IsResolved() &&
                dgsem_default.discretization == dg_sim::Discretization::DGSEM &&
                dgsem_default.regularization == dg_sim::Regularization::None &&
                dgsem_default.time_integrator == dg_sim::TimeIntegrator::RK3SSP,
              "request: omitted DGSEM selectors retain CaseSpec "
              "regularization and RK3SSP");

    const dg_sim::SolverRequestResolution dg_ssprk104 =
      dg_sim::ResolveSolverRequest("dg", true, "", "dg", "", "ssprk104",
                                   false, false, false, true);
    ctx.Check(dg_ssprk104.IsResolved() &&
                dg_ssprk104.time_integrator ==
                  dg_sim::TimeIntegrator::SSPRK104,
              "request: DG admits typed SSPRK(10,4) opt-in");

    const dg_sim::SolverRequestResolution dgsem_ssprk104 =
      dg_sim::ResolveSolverRequest("dgsem", true, "", "dgsem", "",
                                   "ssprk104", false, false, false, true);
    ctx.Check(dgsem_ssprk104.IsResolved() &&
                dgsem_ssprk104.time_integrator ==
                  dg_sim::TimeIntegrator::SSPRK104,
              "request: DGSEM admits typed SSPRK(10,4) opt-in");

    const dg_sim::SolverRequestResolution legacy_override =
      dg_sim::ResolveSolverRequest("dg", true, "", "dg", "none", "rk3ssp",
                                   false, false, true, true);
    ctx.Check(legacy_override.IsResolved() &&
                legacy_override.regularization == dg_sim::Regularization::None,
              "request: explicit regularization overrides the CaseSpec default");

    const dg_sim::SolverRequestResolution cuda_request =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "", "",
                                   false, false, false, false, "cuda");
    ctx.Check(cuda_request.IsResolved() &&
                cuda_request.device_backend == dg_sim::DeviceBackend::CUDA &&
                cuda_request.hdg_cuda_engine ==
                  dg_sim::HdgCudaEngine::Ordinary,
              "request: valid device reaches the resolved request");

    const dg_sim::SolverRequestResolution tensor_hdg_cuda =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "", "",
                                   true, false, false, false, "cuda", "tensor");
    ctx.Check(tensor_hdg_cuda.IsResolved() &&
                tensor_hdg_cuda.hdg_cuda_engine ==
                  dg_sim::HdgCudaEngine::TensorCore,
              "request: tensor CUDA is an explicit HDG engine peer");

    const dg_sim::SolverRequestResolution invalid_hdg_cuda_engine =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "", "",
                                   true, false, false, false, "cuda", "invalid");
    ctx.Check(invalid_hdg_cuda_engine.status ==
                dg_sim::SolverRequestResolutionStatus::InvalidHdgCudaEngine,
              "request: invalid HDG CUDA engine is rejected");

    const dg_sim::SolverRequestResolution tensor_hdg_cpu =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "", "",
                                   true, false, false, false, "cpu", "tensor");
    ctx.Check(tensor_hdg_cpu.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedHdgCudaEngine,
              "request: tensor engine is rejected for CPU HDG");

    const dg_sim::SolverRequestResolution tensor_dg_cuda =
      dg_sim::ResolveSolverRequest("dg", false, "dg", "dg", "", "",
                                   true, false, false, false, "cuda", "tensor");
    ctx.Check(tensor_dg_cuda.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedHdgCudaEngine,
              "request: tensor engine is rejected outside HDG");

    const dg_sim::SolverRequestResolution invalid_device =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "", "",
                                   false, false, false, false, "invalid");
    ctx.Check(invalid_device.status ==
                dg_sim::SolverRequestResolutionStatus::InvalidDeviceBackend,
              "request: invalid device backend is rejected");

    const dg_sim::SolverRequestResolution compatible_backends =
      dg_sim::ResolveSolverRequest("dg", false, "dgsem", "dgsem", "", "",
                                   true, true, false, false);
    ctx.Check(compatible_backends.IsResolved() &&
                compatible_backends.discretization_provenance ==
                  dg_sim::DiscretizationProvenance::BothOptions &&
                compatible_backends.dg_backend == dg_sim::DGBackend::DGSEM,
              "request: matching discretization and legacy backend remain compatible");

    const dg_sim::SolverRequestResolution conflicting_backends =
      dg_sim::ResolveSolverRequest("dg", false, "dg", "dgsem", "", "",
                                   true, true, false, false);
    ctx.Check(conflicting_backends.status ==
                dg_sim::SolverRequestResolutionStatus::Conflict,
              "request: conflicting discretization and legacy backend are rejected");

    const dg_sim::SolverRequestResolution hdg_default =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "", "",
                                   true, false, false, false);
    ctx.Check(hdg_default.IsResolved() &&
                hdg_default.discretization == dg_sim::Discretization::HDG &&
                hdg_default.regularization == dg_sim::Regularization::None &&
                hdg_default.time_integrator ==
                  dg_sim::TimeIntegrator::BackwardEuler,
              "request: omitted HDG integrator defaults to backward-euler");

    const dg_sim::SolverRequestResolution legacy_be =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "", "backward-euler",
                                   false, false, false, true);
    ctx.Check(legacy_be.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedTimeIntegrator,
              "request: DG/DGSEM backward-euler is rejected");

    const dg_sim::SolverRequestResolution dg_partitioned_be =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "",
                                   "partitioned-backward-euler",
                                   false, false, false, true);
    ctx.Check(dg_partitioned_be.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedTimeIntegrator,
              "request: DG/DGSEM partitioned backward-euler is rejected");

    const dg_sim::SolverRequestResolution dg_rodas3 =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "", "rodas3",
                                   false, false, false, true);
    ctx.Check(dg_rodas3.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedTimeIntegrator,
              "request: DG/DGSEM RODAS3 is rejected");

    const dg_sim::SolverRequestResolution invalid_regularization =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "invalid", "",
                                   false, false, true, false);
    ctx.Check(invalid_regularization.status ==
                dg_sim::SolverRequestResolutionStatus::InvalidRegularization,
              "request: invalid regularization is rejected");

    const dg_sim::SolverRequestResolution invalid_time_integrator =
      dg_sim::ResolveSolverRequest("dg", false, "", "dg", "", "invalid",
                                   false, false, false, true);
    ctx.Check(invalid_time_integrator.status ==
                dg_sim::SolverRequestResolutionStatus::InvalidTimeIntegrator,
              "request: invalid time integrator is rejected");

    const dg_sim::SolverRequestResolution hdg_rk =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "", "rk3ssp",
                                   true, false, false, true);
    ctx.Check(hdg_rk.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedTimeIntegrator,
              "request: HDG RK3SSP is rejected");

    const dg_sim::SolverRequestResolution hdg_ssprk104 =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "",
                                   "ssprk104", true, false, false, true);
    ctx.Check(hdg_ssprk104.status ==
                dg_sim::SolverRequestResolutionStatus::UnsupportedTimeIntegrator,
              "request: HDG SSPRK(10,4) is rejected");

    const dg_sim::SolverRequestResolution hdg_partitioned_cpu =
      dg_sim::ResolveSolverRequest(
        "dg", false, "hdg", "dg", "none", "partitioned-backward-euler",
        true, false, true, true, "cpu");
    ctx.Check(hdg_partitioned_cpu.IsResolved() &&
                hdg_partitioned_cpu.regularization ==
                  dg_sim::Regularization::None &&
                hdg_partitioned_cpu.time_integrator ==
                  dg_sim::TimeIntegrator::PartitionedBackwardEuler &&
                hdg_partitioned_cpu.device_backend ==
                  dg_sim::DeviceBackend::CPU,
              "request: conventional HDG CPU admits partitioned backward-euler");

    const dg_sim::SolverRequestResolution hdg_partitioned_cuda =
      dg_sim::ResolveSolverRequest(
        "dg", false, "hdg", "dg", "igr", "partitioned-backward-euler",
        true, false, true, true, "cuda");
    ctx.Check(hdg_partitioned_cuda.IsResolved() &&
                hdg_partitioned_cuda.regularization ==
                  dg_sim::Regularization::IGR &&
                hdg_partitioned_cuda.time_integrator ==
                  dg_sim::TimeIntegrator::PartitionedBackwardEuler &&
                hdg_partitioned_cuda.device_backend ==
                  dg_sim::DeviceBackend::CUDA,
              "request: IGR HDG CUDA admits partitioned backward-euler");

    const dg_sim::SolverRequestResolution hdg_rodas3_cpu =
      dg_sim::ResolveSolverRequest(
        "dg", false, "hdg", "dg", "none", "rodas3",
        true, false, true, true, "cpu");
    ctx.Check(hdg_rodas3_cpu.IsResolved() &&
                hdg_rodas3_cpu.regularization ==
                  dg_sim::Regularization::None &&
                hdg_rodas3_cpu.time_integrator ==
                  dg_sim::TimeIntegrator::Rodas3 &&
                hdg_rodas3_cpu.device_backend ==
                  dg_sim::DeviceBackend::CPU,
              "request: conventional HDG CPU admits typed RODAS3 opt-in");

    const dg_sim::SolverRequestResolution hdg_rodas3_cuda =
      dg_sim::ResolveSolverRequest(
        "dg", false, "hdg", "dg", "igr", "rodas3",
        true, false, true, true, "cuda");
    ctx.Check(hdg_rodas3_cuda.IsResolved() &&
                hdg_rodas3_cuda.regularization ==
                  dg_sim::Regularization::IGR &&
                hdg_rodas3_cuda.time_integrator ==
                  dg_sim::TimeIntegrator::Rodas3 &&
                hdg_rodas3_cuda.device_backend ==
                  dg_sim::DeviceBackend::CUDA,
              "request: IGR HDG CUDA admits typed RODAS3 opt-in");

    const dg_sim::SolverRequestResolution hdg_igr_cpu =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "igr", "",
                                   true, false, true, false);
    ctx.Check(hdg_igr_cpu.IsResolved() &&
                hdg_igr_cpu.regularization == dg_sim::Regularization::IGR &&
                hdg_igr_cpu.time_integrator ==
                  dg_sim::TimeIntegrator::BackwardEuler &&
                hdg_igr_cpu.device_backend == dg_sim::DeviceBackend::CPU,
              "request: HDG CPU IGR resolves to backward-euler");

    const dg_sim::SolverRequestResolution hdg_igr_cuda =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "igr", "",
                                   true, false, true, false, "cuda");
    ctx.Check(hdg_igr_cuda.IsResolved() &&
                hdg_igr_cuda.regularization == dg_sim::Regularization::IGR &&
                hdg_igr_cuda.time_integrator ==
                  dg_sim::TimeIntegrator::BackwardEuler &&
                hdg_igr_cuda.device_backend == dg_sim::DeviceBackend::CUDA,
              "request: HDG CUDA IGR resolves to backward-euler");

    const dg_sim::SolverRequestResolution hdg_cuda_none =
      dg_sim::ResolveSolverRequest("dg", false, "hdg", "dg", "none", "",
                                   true, false, true, false, "cuda");
    ctx.Check(hdg_cuda_none.IsResolved() &&
                hdg_cuda_none.regularization == dg_sim::Regularization::None &&
                hdg_cuda_none.device_backend == dg_sim::DeviceBackend::CUDA,
              "request: HDG CUDA none remains supported");

    const dg_sim::SolverRequestResolution dgsem_cuda_igr =
      dg_sim::ResolveSolverRequest("dgsem", true, "", "dgsem", "", "",
                                   false, false, false, false, "cuda");
    ctx.Check(dgsem_cuda_igr.IsResolved() &&
                dgsem_cuda_igr.discretization == dg_sim::Discretization::DGSEM &&
                dgsem_cuda_igr.regularization == dg_sim::Regularization::IGR &&
                dgsem_cuda_igr.device_backend == dg_sim::DeviceBackend::CUDA,
              "request: DGSEM CUDA IGR remains supported");

    char program_name[] = "test";
    char regularization_flag[] = "--regularization";
    char regularization_value[] = "none";
    char integrator_flag[] = "-ti";
    char integrator_value[] = "rk3ssp";
    char* argv[] = {program_name,
                    regularization_flag,
                    regularization_value,
                    integrator_flag,
                    integrator_value};
    ctx.Check(dg_sim::HasRegularizationOptionOverride(5, argv) &&
                dg_sim::HasTimeIntegratorOptionOverride(5, argv),
              "request: selector provenance uses exact argv flags");
  }

  // --- IGR derived runtime defaults ---
  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cuda";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(igr_opts.exact_dg_face_operator,
              "IGR defaults: CUDA L2/BrokenL2 DG enables exact face");
    ctx.Check(igr_opts.linear_solver == "minres",
              "IGR defaults: CUDA L2/BrokenL2 DG selects MINRES");
    ctx.Check(igr_opts.preconditioner == "chebyshev",
              "IGR defaults: CUDA L2/BrokenL2 DG selects Chebyshev");
    ctx.Check(igr_opts.nonconvergence_policy == "accept",
              "IGR defaults: CUDA L2/BrokenL2 DG retains capped iterate");
    ctx.Check(!igr_opts.sigma_aware_cfl,
              "IGR defaults: sigma-aware CFL is disabled");
    ctx.Check(!igr_opts.sigma_aware_rusanov,
              "IGR defaults: sigma-aware Rusanov is disabled");
    ctx.Check(!igr_opts.igr_invariant_domain_flux,
              "IGR defaults: invariant-domain flux is disabled");
    ctx.CheckNear(igr_opts.stage_rtol_scale,
                  1.0,
                  0.0,
                  "IGR defaults: CUDA L2/BrokenL2 DG pins stage rtol scale");
    ctx.CheckNear(igr_opts.lin_rtol,
                  1.0e-2,
                  0.0,
                  "IGR defaults: CUDA L2/BrokenL2 DG uses production rtol");
    ctx.CheckNear(igr_opts.lin_atol,
                  0.0,
                  0.0,
                  "IGR defaults: CUDA L2/BrokenL2 DG uses zero atol");
    ctx.Check(igr_opts.lin_maxit == 500,
              "IGR defaults: CUDA L2/BrokenL2 DG uses production iteration cap");
    dg_igr::IGRConfig igr_config;
    ctx.Check(igr_config.nonconvergence_policy ==
                dg_igr::IGRNonconvergencePolicy::Accept,
              "IGR defaults: linear nonconvergence retains capped iterate");
    ctx.Check(dg_sim::ParseIGRNonconvergencePolicy("error") ==
                dg_igr::IGRNonconvergencePolicy::Error,
              "IGR options: explicit strict error policy parses");
    igr_opts.nonconvergence_policy = "error";
    dg_sim::ApplyIGRRuntimeOptionOverrides(igr_opts, igr_config);
    ctx.Check(igr_config.nonconvergence_policy ==
                dg_igr::IGRNonconvergencePolicy::Error,
              "IGR options: strict nonconvergence policy reaches solver config");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cuda";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/true);
    ctx.Check(!igr_opts.exact_dg_face_operator,
              "IGR defaults: explicit exact-face override is preserved");
    ctx.Check(igr_opts.lin_maxit < 0,
              "IGR defaults: exact-face opt-out keeps the case iteration cap");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cuda";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    igr_opts.lin_maxit = 777;
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(igr_opts.lin_maxit == 777,
              "IGR defaults: explicit iteration cap is preserved");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cpu";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(!igr_opts.exact_dg_face_operator,
              "IGR defaults: CPU L2/BrokenL2 keeps exact face opt-in");
    ctx.Check(igr_opts.linear_solver == "gmres",
              "IGR defaults: CPU L2 DG selects GMRES");
    ctx.Check(igr_opts.preconditioner == "boomeramg",
              "IGR defaults: CPU L2 DG selects BoomerAMG");
    ctx.CheckNear(igr_opts.lin_rtol,
                  1.0e-2,
                  0.0,
                  "IGR defaults: CPU L2 DG uses production rtol");
    ctx.CheckNear(igr_opts.lin_atol,
                  0.0,
                  0.0,
                  "IGR defaults: CPU L2 DG uses zero atol");
    ctx.Check(igr_opts.lin_maxit == 500,
              "IGR defaults: CPU L2 DG uses production iteration cap");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cpu";
    opts.dg_backend = "dg";
    opts.positivity_limiter = "zhang-shu";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    igr_opts.linear_solver = "cg";
    igr_opts.preconditioner = "none";
    igr_opts.lin_rtol = 3.0e-3;
    igr_opts.lin_maxit = 777;
    igr_opts.sigma_aware_cfl = true;
    igr_opts.igr_invariant_domain_flux = true;
    opts.zhang_shu_vtk_points = true;
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(igr_opts.linear_solver == "cg",
              "IGR defaults: explicit CPU L2 solver is preserved");
    ctx.Check(igr_opts.preconditioner == "none",
              "IGR defaults: explicit CPU L2 preconditioner is preserved");
    ctx.CheckNear(igr_opts.lin_rtol,
                  3.0e-3,
                  0.0,
                  "IGR defaults: explicit CPU L2 tolerance is preserved");
    ctx.Check(igr_opts.lin_maxit == 777,
              "IGR defaults: explicit CPU L2 iteration cap is preserved");
    ctx.Check(igr_opts.sigma_aware_cfl,
              "IGR defaults: explicit sigma-aware CFL opt-in is preserved");
    ctx.Check(igr_opts.igr_invariant_domain_flux,
              "IGR defaults: explicit invariant-domain flux is preserved");
    const dg_sim::RunConfig exact_flux_run =
      dg_sim::MakeRunConfig(dg_sim::CaseConfig{}, opts, igr_opts);
    ctx.Check(exact_flux_run.igr_opts.igr_invariant_domain_flux &&
                exact_flux_run.case_config.limiter_pipeline.zhang_shu &&
                exact_flux_run.case_config.limiter_pipeline
                  .zhang_shu_vtk_points,
              "IGR options: invariant-domain DG accepts the VTK-point "
              "limiter policy");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cpu";
    opts.dg_backend = "dg";
    igr_opts.space = "H1";
    igr_opts.grad = "H1";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(igr_opts.linear_solver == "auto",
              "IGR defaults: CPU H1 keeps its CG-resolving auto policy");
    ctx.Check(igr_opts.preconditioner == "auto",
              "IGR defaults: CPU H1 keeps its BoomerAMG-resolving auto policy");
  }

  // --- Exact invariant-domain flux has paired typed CLI controls ---
  {
    char program_name[] = "test";
    char enable_name[] = "--igr-invariant-domain-flux";
    char* enable_argv[] = {program_name, enable_name};
    dg_sim::IGROptions enabled;
    mfem::OptionsParser enable_parser(2, enable_argv);
    dg_sim::AddIGROptions(enable_parser, enabled);
    enable_parser.Parse();
    ctx.Check(enable_parser.Good() && enabled.igr_invariant_domain_flux,
              "IGR options: invariant-domain flux enable flag parses");

    char disable_name[] = "--no-igr-invariant-domain-flux";
    char* disable_argv[] = {program_name, disable_name};
    dg_sim::IGROptions disabled;
    disabled.igr_invariant_domain_flux = true;
    mfem::OptionsParser disable_parser(2, disable_argv);
    dg_sim::AddIGROptions(disable_parser, disabled);
    disable_parser.Parse();
    ctx.Check(disable_parser.Good() && !disabled.igr_invariant_domain_flux,
              "IGR options: invariant-domain flux disable flag parses");
  }

  // --- Extrapolated endpoint cadence has a stable typed spelling ---
  {
    const auto cadence =
      dg_sim::ParseIGRSigmaUpdateCadence("EXTRAPOLATED-EVERY-STEP");
    ctx.Check(
      cadence == dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep &&
        std::string(dg_sim::IGRSigmaUpdateCadenceToString(cadence)) ==
          "extrapolated-every-step" &&
        dg_sim::ParseIGRSigmaUpdateCadence("extrapolated_every_step") == cadence,
      "IGR options: extrapolated-every-step cadence parses and round-trips");
  }

  // --- Modal damping is opt-in and retains stage-consistent IGR coupling ---
  {
    dg_sim::CommonOptions opts;
    dg_sim::CaseConfig case_cfg;
    dg_sim::IGROptions igr_opts;
    auto validate = [&](dg_sim::Discretization discretization,
                        dg_sim::Regularization regularization)
    {
      return dg_sim::ValidateModalDampingSelection(
        discretization, regularization, case_cfg, opts, igr_opts);
    };
    ctx.Check(opts.modal_damping_rate == 0.0 &&
                validate(dg_sim::Discretization::HDG,
                         dg_sim::Regularization::IGR).empty(),
              "modal damping: default zero preserves all backends");

    char program_name[] = "test";
    char rate_name[] = "--modal-damping-rate";
    char rate_value[] = "80000";
    char* rate_argv[] = {program_name, rate_name, rate_value};
    mfem::OptionsParser parser(3, rate_argv);
    dg_sim::AddCommonOptions(parser, opts, "Test mesh.");
    parser.Parse();
    ctx.Check(parser.Good() && opts.modal_damping_rate == 80000.0,
              "modal damping: M16 physical rate parses");

    case_cfg.igr_config.sigma_cadence =
      dg_igr::IGRSigmaUpdateCadence::EveryStage;
    ctx.Check(validate(dg_sim::Discretization::DG,
                       dg_sim::Regularization::IGR).empty() &&
                validate(dg_sim::Discretization::DGSEM,
                         dg_sim::Regularization::IGR).empty(),
              "modal damping: DG and DGSEM admit stage-consistent IGR");
    ctx.Check(!validate(dg_sim::Discretization::HDG,
                        dg_sim::Regularization::None).empty(),
              "modal damping: positive HDG request is rejected before dispatch");

    case_cfg.igr_config.sigma_cadence =
      dg_igr::IGRSigmaUpdateCadence::EveryStep;
    ctx.Check(!validate(dg_sim::Discretization::DG,
                        dg_sim::Regularization::IGR).empty(),
              "modal damping: inherited frozen IGR cadence is rejected");
    igr_opts.sigma_cadence = "every-stage";
    ctx.Check(validate(dg_sim::Discretization::DG,
                       dg_sim::Regularization::IGR).empty(),
              "modal damping: explicit every-stage overrides the case cadence");
    igr_opts.sigma_cadence = "extrapolated-every-step";
    ctx.Check(!validate(dg_sim::Discretization::DGSEM,
                        dg_sim::Regularization::IGR).empty() &&
                validate(dg_sim::Discretization::DGSEM,
                         dg_sim::Regularization::None).empty(),
              "modal damping: cadence restriction applies to active IGR");
    for (const mfem::real_t rate :
         {mfem::real_t(-1.0), std::numeric_limits<mfem::real_t>::infinity(),
          std::numeric_limits<mfem::real_t>::quiet_NaN()})
    {
      opts.modal_damping_rate = rate;
      ctx.Check(!validate(dg_sim::Discretization::DG,
                          dg_sim::Regularization::None).empty(),
                "modal damping: negative and nonfinite rates are rejected");
    }
  }

  // --- Explicit SSPRK CFL is a safety fraction, not a bare Courant number ---
  {
    dg_sim::RunConfig cfl_run;
    cfl_run.time_integrator = dg_sim::TimeIntegrator::SSPRK104;
    cfl_run.opts.cfl = 1.0;
    ctx.Check(dg_sim::ValidateExplicitSSPRKCFLSelection(cfl_run).empty(),
              "explicit CFL: unit SSP safety factor is admitted");

    cfl_run.opts.cfl = 1.01;
    ctx.Check(!dg_sim::ValidateExplicitSSPRKCFLSelection(cfl_run).empty(),
              "explicit CFL: factors above the SSP bound fail closed");

    cfl_run.opts.cfl = -0.1;
    ctx.Check(!dg_sim::ValidateExplicitSSPRKCFLSelection(cfl_run).empty(),
              "explicit CFL: negative safety factors are rejected");

    cfl_run.opts.cfl = 0.0;
    cfl_run.opts.dt = 1.0e-3;
    ctx.Check(dg_sim::ValidateExplicitSSPRKCFLSelection(cfl_run).empty(),
              "explicit CFL: zero selects a valid fixed time step");
    cfl_run.opts.dt = -1.0e-3;
    ctx.Check(!dg_sim::ValidateExplicitSSPRKCFLSelection(cfl_run).empty(),
              "explicit CFL: disabled control requires a positive fixed step");

    cfl_run.time_integrator = dg_sim::TimeIntegrator::Rodas3;
    cfl_run.opts.cfl = 6.0;
    ctx.Check(dg_sim::ValidateExplicitSSPRKCFLSelection(cfl_run).empty(),
              "explicit CFL: non-SSPRK backends retain their own semantics");
  }

  // --- Extrapolated IGR remains a bounded, production-qualified tuple ---
  {
    dg_sim::CaseConfig case_cfg;
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.discretization = "dgsem";
    opts.dg_backend = "dgsem";
    opts.positivity_limiter = "zhang-shu";
    opts.zhang_shu_points = "consumer";
    opts.dgsem_stab = "none";
    opts.dgsem_volume_flux = "central";
    opts.dgsem_volume_dissipation = "llf";
    opts.dgsem_surface_flux = "rusanov";
    igr_opts.alpha_length_policy = "local";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    igr_opts.ip_kind = "sipg";
    igr_opts.solve_mode = "krylov";
    igr_opts.linear_solver = "cg";
    igr_opts.preconditioner = "chebyshev";
    igr_opts.sigma_cadence = "extrapolated-every-step";
    igr_opts.exact_dg_face_operator = true;
    igr_opts.nonconvergence_policy = "error";
    const dg_sim::RunConfig qualified = dg_sim::MakeRunConfig(
      case_cfg,
      opts,
      igr_opts,
      dg_sim::DiscretizationProvenance::DiscretizationOption,
      dg_sim::Regularization::IGR,
      dg_sim::TimeIntegrator::SSPRK104);
    ctx.Check(
      dg_sim::ValidateExtrapolatedEveryStepIGRSelection(qualified).empty(),
      "extrapolated IGR selection: qualified DGSEM tuple is admitted");

    dg_sim::RunConfig extrapolated_with_legacy_entry_limit = qualified;
    extrapolated_with_legacy_entry_limit.case_config.rk3_config.limit_initial =
      true;
    ctx.Check(
      !dg_sim::ResolveRK3StepperConfig(extrapolated_with_legacy_entry_limit)
         .limit_initial,
      "extrapolated IGR selection: redundant RK-entry limiting is disabled");

    dg_sim::RunConfig wrong_transport = qualified;
    wrong_transport.dgsem_volume_dissipation =
      dg_sim::DGSEMVolumeDissipation::None;
    ctx.Check(
      !dg_sim::ValidateExtrapolatedEveryStepIGRSelection(wrong_transport)
         .empty(),
      "extrapolated IGR selection: unqualified transport is rejected");

    dg_sim::RunConfig extra_limiter = qualified;
    extra_limiter.case_config.limiter_pipeline.krivodonova = true;
    ctx.Check(
      !dg_sim::ValidateExtrapolatedEveryStepIGRSelection(extra_limiter)
         .empty(),
      "extrapolated IGR selection: unqualified extra limiter is rejected");

    dg_sim::RunConfig wrong_closure = qualified;
    wrong_closure.igr_opts.grad = "H1";
    ctx.Check(
      !dg_sim::ValidateExtrapolatedEveryStepIGRSelection(wrong_closure)
         .empty(),
      "extrapolated IGR selection: conforming closure is rejected");

    dg_sim::RunConfig uncertified_solver = qualified;
    uncertified_solver.igr_opts.solve_mode = "smoother";
    ctx.Check(
      !dg_sim::ValidateExtrapolatedEveryStepIGRSelection(uncertified_solver)
         .empty(),
      "extrapolated IGR selection: a closure without independent FP64 "
      "residual certification is rejected");

    dg_sim::RunConfig every_stage = qualified;
    every_stage.igr_opts.sigma_cadence = "every-stage";
    every_stage.case_config.rk3_config.limit_initial = true;
    every_stage.dgsem_volume_dissipation =
      dg_sim::DGSEMVolumeDissipation::None;
    ctx.Check(
      dg_sim::ValidateExtrapolatedEveryStepIGRSelection(every_stage).empty(),
      "extrapolated IGR selection: other cadences retain their general surface");
    ctx.Check(dg_sim::ResolveRK3StepperConfig(every_stage).limit_initial,
              "every-stage IGR selection: RK-entry limiting is preserved");

    dg_sim::IGROptions hdg_igr_opts = igr_opts;
    ctx.Check(
      !dg_sim::ValidateHdgIGRSigmaCadence(
         dg_sim::Discretization::HDG,
         dg_sim::Regularization::IGR,
         case_cfg,
         hdg_igr_opts).empty(),
      "HDG IGR selection: extrapolated cadence is rejected before dispatch");
    hdg_igr_opts.sigma_cadence = "every-stage";
    ctx.Check(
      dg_sim::ValidateHdgIGRSigmaCadence(
        dg_sim::Discretization::HDG,
        dg_sim::Regularization::IGR,
        case_cfg,
        hdg_igr_opts).empty(),
      "HDG IGR selection: stage-consistent cadence is admitted");
    hdg_igr_opts.sigma_cadence = "every-step";
    ctx.Check(
      !dg_sim::ValidateHdgIGRSigmaCadence(
         dg_sim::Discretization::HDG,
         dg_sim::Regularization::None,
         case_cfg,
         hdg_igr_opts).empty(),
      "conventional HDG selection: explicit unsupported cadence is not ignored");
    hdg_igr_opts.sigma_cadence.clear();
    ctx.Check(
      dg_sim::ValidateHdgIGRSigmaCadence(
        dg_sim::Discretization::HDG,
        dg_sim::Regularization::None,
        case_cfg,
        hdg_igr_opts).empty(),
      "conventional HDG selection: absent IGR cadence remains irrelevant");
  }

  // --- Refined VTK limiter points have typed CLI and legacy-env controls ---
  {
    constexpr const char* vtk_points_env =
      "DG_IGR_LIMITER_ENFORCE_VTK_POINTS";
    const char* original_env_text = std::getenv(vtk_points_env);
    const std::optional<std::string> original_env = original_env_text
      ? std::optional<std::string>(original_env_text)
      : std::nullopt;
    unsetenv(vtk_points_env);

    char program_name[] = "test";
    char enable_name[] = "--zhang-shu-vtk-points";
    char* enable_argv[] = {program_name, enable_name};
    dg_sim::CommonOptions enabled;
    mfem::OptionsParser enable_parser(2, enable_argv);
    dg_sim::AddCommonOptions(enable_parser, enabled, "test mesh");
    enable_parser.Parse();
    ctx.Check(enable_parser.Good() && enabled.zhang_shu_vtk_points,
              "Zhang-Shu VTK points: typed enable flag parses");

    setenv(vtk_points_env, "1", 1);
    char disable_name[] = "--no-zhang-shu-vtk-points";
    char* disable_argv[] = {program_name, disable_name};
    dg_sim::CommonOptions disabled;
    mfem::OptionsParser disable_parser(2, disable_argv);
    dg_sim::AddCommonOptions(disable_parser, disabled, "test mesh");
    disable_parser.Parse();
    ctx.Check(disable_parser.Good() && !disabled.zhang_shu_vtk_points,
              "Zhang-Shu VTK points: typed disable overrides legacy env");

    char* env_argv[] = {program_name};
    dg_sim::CommonOptions env_enabled;
    mfem::OptionsParser env_parser(1, env_argv);
    dg_sim::AddCommonOptions(env_parser, env_enabled, "test mesh");
    env_parser.Parse();
    ctx.Check(env_parser.Good() && env_enabled.zhang_shu_vtk_points,
              "Zhang-Shu VTK points: legacy env remains a default fallback");

    if (original_env)
    {
      setenv(vtk_points_env, original_env->c_str(), 1);
    }
    else
    {
      unsetenv(vtk_points_env);
    }
  }

  // --- Callback config wiring ---
  {
    dg_sim::DefaultCallbackConfig cfg;
    cfg.enable_alive = true;
    cfg.enable_analysis = true;
    cfg.enable_vtk = false;
    cfg.enable_summary = true;
    cfg.compute_minmax = true;
    const dg_sim::TimeLoopCallbacks callbacks =
      dg_sim::MakeDefaultTimeLoopCallbacks(cfg);
    ctx.Check(static_cast<bool>(callbacks.on_alive),
              "default callbacks: on_alive wired");
    ctx.Check(static_cast<bool>(callbacks.on_analysis),
              "default callbacks: on_analysis wired");
    ctx.Check(!callbacks.on_vtk_save,
              "default callbacks: on_vtk_save disabled by config");
    ctx.Check(static_cast<bool>(callbacks.on_summary),
              "default callbacks: on_summary wired");
    ctx.Check(callbacks.wants_minmax,
              "default callbacks: wants_minmax follows analysis+compute flag");
  }

  {
    dg_sim::DefaultCallbackConfig cfg;
    cfg.enable_alive = false;
    cfg.enable_analysis = false;
    cfg.enable_vtk = true;
    cfg.enable_summary = false;
    cfg.compute_minmax = true;
    const dg_sim::TimeLoopCallbacks callbacks =
      dg_sim::MakeDefaultTimeLoopCallbacks(cfg);
    ctx.Check(!callbacks.on_alive,
              "default callbacks: on_alive disabled by config");
    ctx.Check(!callbacks.on_analysis,
              "default callbacks: on_analysis disabled by config");
    ctx.Check(static_cast<bool>(callbacks.on_vtk_save),
              "default callbacks: on_vtk_save wired");
    ctx.Check(!callbacks.on_summary,
              "default callbacks: on_summary disabled by config");
    ctx.Check(!callbacks.wants_minmax,
              "default callbacks: wants_minmax false when analysis disabled");
  }

  // --- Hook dispatch and payload fields ---
  // Keep one identical global fixture for the one- and two-rank parity runs,
  // while giving every rank in the supported two-rank test local work.
  mfem::Mesh serial_mesh =
    mfem::Mesh::MakeCartesian2D(2,
                                1,
                                mfem::Element::QUADRILATERAL,
                                /*gen_edges=*/true,
                                /*sx=*/1.0,
                                /*sy=*/1.0);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();

  dg_sim::RunConfig run_cfg;
  run_cfg.opts.order = 1;
  run_cfg.opts.glvis = false;
  run_cfg.opts.vtk_save = false;
  run_cfg.opts.dt_min = 5.0e-13;
  run_cfg.dg_backend = dg_sim::DGBackend::DG;

  {
    dg_sim::RunConfig loop_config_run = run_cfg;
    loop_config_run.opts.cfl = 0.75;
    const dg_sim::TimeLoopConfig loop_cfg =
      dg_sim::MakeTimeLoopConfig(loop_config_run);
    ctx.CheckNear(loop_cfg.dt_min,
                  run_cfg.opts.dt_min,
                  1.0e-16,
                  "time loop config: dt_min propagated from runtime options");
    ctx.CheckNear(loop_cfg.cfl,
                  0.75,
                  0.0,
                  "time loop config: SSP safety factor is propagated unscaled");
  }

  dg_sim::ExampleContext example_ctx(run_cfg,
                                     std::move(pmesh),
                                     /*sigma_space=*/"H1",
                                     /*need_sigma=*/false,
                                     /*need_h1=*/false);
  dg_sim::EulerStateBundle bundle(example_ctx, gamma, /*rho_floor=*/1.0e-12);

  const mfem::Vector state =
    EulerConsFromPrim2D(/*rho=*/1.2, /*u=*/0.3, /*v=*/-0.1, /*p=*/0.9, gamma);
  FillConstantState(state, bundle.sol);

  // --- initial Zhang-Shu admission precedes all consumers and is strict ---
  {
    dg_sim::RunConfig limiter_run = run_cfg;
    limiter_run.case_config.gamma = gamma;
    limiter_run.case_config.int_order_offset = 1;
    limiter_run.case_config.rk3_config.gamma = gamma;
    limiter_run.case_config.rk3_config.rho_min = 1.0e-12;
    limiter_run.case_config.rk3_config.p_min = 1.0e-12;
    limiter_run.case_config.rk3_config.quad_order = 0;
    limiter_run.case_config.rk3_config.p_bisect_iters = 20;
    limiter_run.regularization = dg_sim::Regularization::None;
    limiter_run.case_config.limiter_pipeline.zhang_shu = false;
    limiter_run.case_config.limiter_pipeline.zhang_shu_points =
      mfem::ZhangShuPointSet::kConsumer;

    FillConstantState(state, bundle.sol);
    const int scalar_dofs = example_ctx.spaces.vfes.GetNDofs();
    int fixture_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &fixture_rank);
    if (fixture_rank == 0 && scalar_dofs > 0)
    {
      // The element mean remains admissible, but one Euler consumer node has
      // negative pressure and therefore requires a mean-preserving blend.
      bundle.sol.HostReadWrite()[3 * scalar_dofs] = -0.5;
    }
    const mfem::Vector candidate(bundle.sol);
    const bool disabled_applied =
      dg_sim::detail::ApplyInitialEulerZhangShuTransaction(limiter_run,
                                                           bundle);
    mfem::Vector disabled_error(bundle.sol);
    disabled_error -= candidate;
    ctx.Check(!disabled_applied && disabled_error.Normlinf() == 0.0,
              "initial admission: disabled Zhang-Shu policy is an exact no-op");

    limiter_run.regularization = dg_sim::Regularization::IGR;
    limiter_run.case_config.limiter_pipeline.zhang_shu = true;
    const int local_order = example_ctx.spaces.vfes.GetFE(0)->GetOrder();
    const int volume_qord = 2 * local_order + 2;
    const std::vector<int> face_qords{
      2 * local_order + limiter_run.case_config.int_order_offset};
    const auto pre_dof_scan = mfem::dg_igr::ScanEulerAdmissibility(
      example_ctx.spaces.vfes,
      bundle.sol,
      /*dim=*/2,
      gamma,
      limiter_run.case_config.rk3_config.rho_min,
      limiter_run.case_config.rk3_config.p_min,
      volume_qord,
      face_qords,
      mfem::dg_igr::EulerAdmissibilityPointSet::Dof);
    const auto integral_before =
      dg_sim::detail::IntegrateEulerConservedState(bundle.sol);
    mfem::ZhangShuLimiterElementDiagnostics diagnostics;
    const bool enabled_applied =
      dg_sim::detail::ApplyInitialEulerZhangShuTransaction(limiter_run,
                                                           bundle,
                                                           &diagnostics);
    const auto post_dof_scan = mfem::dg_igr::ScanEulerAdmissibility(
      example_ctx.spaces.vfes,
      bundle.sol,
      /*dim=*/2,
      gamma,
      limiter_run.case_config.rk3_config.rho_min,
      limiter_run.case_config.rk3_config.p_min,
      volume_qord,
      face_qords,
      mfem::dg_igr::EulerAdmissibilityPointSet::Dof);
    const auto post_limiter_scan = mfem::dg_igr::ScanEulerAdmissibility(
      example_ctx.spaces.vfes,
      bundle.sol,
      /*dim=*/2,
      gamma,
      limiter_run.case_config.rk3_config.rho_min,
      limiter_run.case_config.rk3_config.p_min,
      volume_qord,
      face_qords,
      mfem::dg_igr::EulerAdmissibilityPointSet::Limiter);
    const auto integral_after =
      dg_sim::detail::IntegrateEulerConservedState(bundle.sol);
    double local_active = 0.0;
    const double* active = diagnostics.active.HostRead();
    for (int i = 0; i < diagnostics.active.Size(); ++i)
    {
      local_active += active[i];
    }
    double global_active = local_active;
    MPI_Allreduce(MPI_IN_PLACE,
                  &global_active,
                  1,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_SUM,
                  example_ctx.spaces.vfes.GetComm());
    ctx.Check(pre_dof_scan.global_bad_pressure > 0 && enabled_applied &&
                post_dof_scan.global_bad_rho == 0 &&
                post_dof_scan.global_bad_pressure == 0 &&
                post_limiter_scan.global_bad_rho == 0 &&
                post_limiter_scan.global_bad_pressure == 0 &&
                global_active > 0.0,
              "initial admission: consumer-point violation is limited before "
              "use");
    for (int component = 0; component < state.Size(); ++component)
    {
      ctx.CheckNear(integral_after.components(component),
                    integral_before.components(component),
                    2.0e-12,
                    "initial admission: limiter preserves component " +
                      std::to_string(component) + " integral");
    }

    mfem::Vector inadmissible_mean(4);
    inadmissible_mean(0) = -1.0;
    inadmissible_mean(1) = 0.0;
    inadmissible_mean(2) = 0.0;
    inadmissible_mean(3) = 2.5;
    FillConstantState(inadmissible_mean, bundle.sol);
    const mfem::Vector invalid_before(bundle.sol);
    bool caught_invalid_mean = false;
    int reported_rejected_elements = 0;
    try
    {
      dg_sim::detail::ApplyInitialEulerZhangShuTransaction(limiter_run,
                                                           bundle);
    }
    catch (const mfem::RecoverableLimiterFailure& failure)
    {
      caught_invalid_mean = true;
      reported_rejected_elements = failure.GlobalRejectedElementCount();
    }
    int caught_min = caught_invalid_mean ? 1 : 0;
    int caught_max = caught_min;
    MPI_Allreduce(MPI_IN_PLACE,
                  &caught_min,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  example_ctx.spaces.vfes.GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &caught_max,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  example_ctx.spaces.vfes.GetComm());
    mfem::Vector invalid_error(bundle.sol);
    invalid_error -= invalid_before;
    double global_invalid_error = invalid_error.Normlinf();
    MPI_Allreduce(MPI_IN_PLACE,
                  &global_invalid_error,
                  1,
                  mfem::MPITypeMap<double>::mpi_type,
                  MPI_MAX,
                  example_ctx.spaces.vfes.GetComm());
    ctx.Check(caught_min == 1 && caught_max == 1 &&
                reported_rejected_elements > 0 && global_invalid_error == 0.0,
              "initial admission: invalid mean rejects collectively without "
              "mutation");
    FillConstantState(state, bundle.sol);
  }

  // A proper subcommunicator must make its own admission decision: an
  // inadmissible group cannot reject an independent admissible group.
  {
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const int color = world_size > 1 ? world_rank % 2 : 0;
    MPI_Comm subcommunicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subcommunicator);
    {
      int subcommunicator_size = 1;
      MPI_Comm_size(subcommunicator, &subcommunicator_size);
      mfem::Mesh submesh = mfem::Mesh::MakeCartesian2D(
        std::max(1, subcommunicator_size),
        1,
        mfem::Element::QUADRILATERAL,
        /*gen_edges=*/true,
        /*sx=*/1.0,
        /*sy=*/1.0);
      auto submesh_parallel =
        std::make_unique<mfem::ParMesh>(subcommunicator, submesh);
      submesh.Clear();

      dg_sim::RunConfig sub_run;
      sub_run.opts.order = 1;
      sub_run.dg_backend = dg_sim::DGBackend::DG;
      sub_run.case_config.gamma = gamma;
      sub_run.case_config.int_order_offset = 1;
      sub_run.case_config.rk3_config.gamma = gamma;
      sub_run.case_config.rk3_config.rho_min = 1.0e-12;
      sub_run.case_config.rk3_config.p_min = 1.0e-12;
      sub_run.case_config.limiter_pipeline.zhang_shu = true;
      sub_run.case_config.limiter_pipeline.zhang_shu_points =
        mfem::ZhangShuPointSet::kConsumer;
      dg_sim::ExampleContext sub_context(sub_run,
                                         std::move(submesh_parallel),
                                         /*sigma_space=*/"H1",
                                         /*need_sigma=*/false,
                                         /*need_h1=*/false);
      dg_sim::EulerStateBundle sub_bundle(
        sub_context, gamma, /*rho_floor=*/1.0e-12);
      mfem::Vector sub_state = state;
      if (color == 0)
      {
        sub_state(0) = -1.0;
        sub_state(1) = 0.0;
        sub_state(2) = 0.0;
        sub_state(3) = 2.5;
      }
      FillConstantState(sub_state, sub_bundle.sol);

      bool subcommunicator_rejected = false;
      try
      {
        dg_sim::detail::ApplyInitialEulerZhangShuTransaction(sub_run,
                                                             sub_bundle);
      }
      catch (const mfem::RecoverableLimiterFailure&)
      {
        subcommunicator_rejected = true;
      }
      ctx.Check(subcommunicator_rejected == (color == 0),
                "initial admission: decisions are isolated by state "
                "subcommunicator");
    }
    MPI_Comm_free(&subcommunicator);
  }

  // --- Explicit SSPRK selections have collision-free output identities ---
  {
    dg_sim::CaseConfig dataset_case;
    dataset_case.case_name = "ssprk-dataset";
    dg_sim::RunConfig dataset_run;
    dataset_run.case_config = dataset_case;
    dataset_run.opts.order = 3;
    dataset_run.opts.ser_ref_levels = 1;
    dataset_run.opts.par_ref_levels = 2;
    dataset_run.igr_opts.space = "L2";
    dataset_run.igr_opts.grad = "BrokenL2";
    dataset_run.igr_opts.c_alpha_user = 0.25;

    dataset_run.time_integrator = dg_sim::TimeIntegrator::RK3SSP;
    const std::string rk3_name =
      dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run);
    dataset_run.time_integrator = dg_sim::TimeIntegrator::SSPRK104;
    const std::string ssprk104_name =
      dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run);

    ctx.Check(ssprk104_name == rk3_name + "_ti_ssprk104",
              "VTK dataset: SSPRK(10,4) has a typed suffix");
    ctx.Check(rk3_name ==
                "ssprk-dataset_p_3_ref_3_is_L2_ig_BrokenL2_ca_0.25",
              "VTK dataset: established RK3SSP name remains unchanged");
    dataset_run.dg_backend = dg_sim::DGBackend::DGSEM;
    dataset_run.dgsem_volume_dissipation =
      dg_sim::DGSEMVolumeDissipation::LLF;
    const std::string dgsem_llf_name =
      dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run);
    ctx.Check(dgsem_llf_name == ssprk104_name + "_dvd_llf",
              "VTK dataset: DGSEM LLF volume dissipation has a typed suffix");
    dataset_run.case_config.igr_config.sigma_cadence =
      dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep;
    dataset_run.igr_opts.alpha_length_policy = "local";
    dataset_run.igr_opts.exact_dg_face_operator = true;
    const std::string shock_candidate_name =
      dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run);
    ctx.Check(shock_candidate_name ==
                dgsem_llf_name +
                  "_iscad_extrapolated-every-step_ial_local_iexface",
              "VTK dataset: effective extrapolated cadence, local alpha, and "
              "exact faces have typed suffixes");
    dataset_run.dg_backend = dg_sim::DGBackend::DG;
    dataset_run.dgsem_volume_dissipation =
      dg_sim::DGSEMVolumeDissipation::None;
    dataset_run.case_config.igr_config.sigma_cadence =
      dg_igr::IGRSigmaUpdateCadence::EveryStage;
    dataset_run.igr_opts.alpha_length_policy.clear();
    dataset_run.igr_opts.exact_dg_face_operator = false;
    dataset_run.igr_opts.igr_invariant_domain_flux = true;
    const std::string invariant_flux_name =
      dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run);
    ctx.Check(invariant_flux_name == ssprk104_name + "_iidflux",
              "VTK dataset: exact IGR-aware face flux has a typed suffix");
    dataset_run.opts.zhang_shu_vtk_points = true;
    const std::string vtk_point_name =
      dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run);
    ctx.Check(vtk_point_name == invariant_flux_name + "_zsvtk",
              "VTK dataset: refined limiter points have a typed suffix");
    dataset_run.opts.modal_damping_rate = 80000.0;
    ctx.Check(dg_sim::MakeCaseVTKDatasetName(dataset_case, dataset_run) ==
                vtk_point_name + "_mdr_80000",
              "VTK dataset: M16 damping has a distinct physical-rate suffix");
  }

  // --- Run metadata records the effective explicit-method identity ---
  {
    constexpr const char* prefix_env_name = "DG_IGR_VTK_PREFIX_PATH";
    const char* original_prefix_text = std::getenv(prefix_env_name);
    const std::optional<std::string> original_prefix = original_prefix_text
      ? std::optional<std::string>(original_prefix_text)
      : std::nullopt;

    long long output_token = 0;
    if (mfem::Mpi::Root())
    {
      output_token = std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count();
    }
    MPI_Bcast(&output_token,
              1,
              mfem::MPITypeMap<long long>::mpi_type,
              0,
              MPI_COMM_WORLD);
    const std::filesystem::path output_root =
      std::filesystem::path("/tmp") /
      ("dg_igr_ssprk_metadata_" + std::to_string(output_token));
    setenv(prefix_env_name, output_root.c_str(), 1);

    dg_sim::CaseConfig metadata_case;
    metadata_case.case_name = "ssprk-metadata";
    metadata_case.igr_config.sigma_cadence =
      dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep;
    metadata_case.igr_config.alpha_length_scale_policy =
      dg_igr::IGRAlphaLengthScalePolicy::Local;
    dg_sim::RunConfig metadata_run;
    metadata_run.case_config = metadata_case;
    metadata_run.opts.vtk_save = true;
    metadata_run.opts.order = 1;
    metadata_run.opts.cfl = 0.5;
    metadata_run.dg_backend = dg_sim::DGBackend::DGSEM;
    metadata_run.dgsem_volume_dissipation =
      dg_sim::DGSEMVolumeDissipation::LLF;
    metadata_run.igr_opts.space = "L2";
    metadata_run.igr_opts.grad = "BrokenL2";
    metadata_run.igr_opts.exact_dg_face_operator = true;
    metadata_run.igr_opts.igr_invariant_domain_flux = true;
    metadata_run.opts.zhang_shu_vtk_points = true;
    const std::vector<dg_sim::TimeIntegrator> methods = {
      dg_sim::TimeIntegrator::SSPRK104};
    dg_sim::RunSummary metadata_summary;
    metadata_summary.accepted_attempts = 4;
    metadata_summary.rejected_attempts = 1;
    metadata_summary.rejected_limiter_elements = 3;
    metadata_summary.rhs_evaluations = 17;
    metadata_summary.minimum_attempted_dt = 0.025;
    metadata_summary.minimum_accepted_dt = 0.05;
    metadata_summary.maximum_attempted_dt = 0.2;
    metadata_summary.maximum_accepted_dt = 0.1;
    metadata_summary.maximum_accepted_step_growth_ratio = 1.75;
    metadata_summary.success = true;
    metadata_summary.termination_reason = "target-reached";
    std::vector<std::string> dataset_names;
    char program_name[] = "test";
    char* metadata_argv[] = {program_name};
    for (const dg_sim::TimeIntegrator method : methods)
    {
      metadata_run.time_integrator = method;
      dataset_names.push_back(
        dg_sim::MakeCaseVTKDatasetName(metadata_case, metadata_run, 0.25));
      dg_sim::WriteRunMetadataSidecar(metadata_case,
                                      metadata_run,
                                      &metadata_summary,
                                      /*igr=*/nullptr,
                                      /*c_alpha_value=*/0.25,
                                      /*argc=*/1,
                                      metadata_argv,
                                      /*success=*/true);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int metadata_fields_ok[1] = {1};
    if (mfem::Mpi::Root())
    {
      for (std::size_t i = 0; i < methods.size(); ++i)
      {
        const std::filesystem::path metadata_path =
          output_root / dataset_names[i] / "run_metadata.json";
        std::ifstream input(metadata_path);
        const bool opened = input.is_open();
        const std::string contents((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
        const std::string expected =
          "\"time_integrator\": \"" +
          std::string(dg_sim::TimeIntegratorToString(methods[i])) + "\"";
        metadata_fields_ok[i] = static_cast<int>(
          opened && contents.find(expected) != std::string::npos &&
          contents.find("\"schema_version\": 3") != std::string::npos &&
          contents.find("\"cfl\": 0.5") != std::string::npos &&
          contents.find(
            "\"cfl_field_semantics\": "
            "\"integrator-normalized-safety-factor\"") !=
            std::string::npos &&
          contents.find("\"cfl_control_enabled\": true") !=
            std::string::npos &&
          contents.find("\"cfl_safety_factor\": 0.5") !=
            std::string::npos &&
          contents.find("\"ssp_coefficient\": 6") !=
            std::string::npos &&
          contents.find("\"ssp_scaled_cfl_factor\": 3") !=
            std::string::npos &&
          contents.find("\"nominal_global_courant\": 1") !=
            std::string::npos &&
          contents.find("\"cfl_diagnostics_observed\": false") !=
            std::string::npos &&
          contents.find("\"cfl_diagnostics_source\": \"requested-config\"") !=
            std::string::npos &&
          contents.find(
            "\"cfl_convention\": \"explicit-ssprk-safety-factor\"") !=
            std::string::npos &&
          contents.find("\"dgsem_stabilization\": \"none\"") !=
            std::string::npos &&
          contents.find("\"dgsem_volume_flux\": \"central\"") !=
            std::string::npos &&
          contents.find("\"dgsem_volume_dissipation\": \"llf\"") !=
            std::string::npos &&
          contents.find("\"dgsem_surface_flux\": \"rusanov\"") !=
            std::string::npos &&
          contents.find("\"dgsem_entropy_jacobian\": \"fd\"") !=
            std::string::npos &&
          contents.find("\"modal_damping_rate\": 0") !=
            std::string::npos &&
          contents.find("\"modal_damping_profile\": \"off\"") !=
            std::string::npos &&
          contents.find(
            "\"igr_sigma_cadence\": \"extrapolated-every-step\"") !=
            std::string::npos &&
          contents.find("\"igr_sigma_update_interval\": 1") !=
            std::string::npos &&
          contents.find("\"igr_alpha_length_policy\": \"local\"") !=
            std::string::npos &&
          contents.find("\"igr_exact_dg_face_operator\": true") !=
            std::string::npos &&
          contents.find("\"accepted_attempts\": 4") != std::string::npos &&
          contents.find("\"rejected_attempts\": 1") != std::string::npos &&
          contents.find("\"rejected_limiter_elements\": 3") !=
            std::string::npos &&
          contents.find("\"rhs_evaluations\": 17") != std::string::npos &&
          contents.find("\"maximum_attempted_dt\": 0.2") !=
            std::string::npos &&
          contents.find("\"maximum_accepted_dt\": 0.1") !=
            std::string::npos &&
          contents.find(
            "\"maximum_accepted_step_growth_ratio\": 1.75") !=
            std::string::npos &&
          contents.find("\"igr_invariant_domain_flux\": true") !=
            std::string::npos &&
          contents.find("\"zhang_shu_vtk_points\": true") !=
            std::string::npos &&
          contents.find("\"success\": true") != std::string::npos &&
          contents.find("\"termination_reason\": \"target-reached\"") !=
            std::string::npos);
      }
    }
    MPI_Bcast(metadata_fields_ok,
              1,
              mfem::MPITypeMap<int>::mpi_type,
              0,
              MPI_COMM_WORLD);
    ctx.Check(metadata_fields_ok[0] == 1,
              "run metadata: effective transport/cadence identity and retry "
              "aggregates persist");

    MPI_Barrier(MPI_COMM_WORLD);
    if (mfem::Mpi::Root())
    {
      std::filesystem::remove_all(output_root);
    }
    if (original_prefix)
    {
      setenv(prefix_env_name, original_prefix->c_str(), 1);
    }
    else
    {
      unsetenv(prefix_env_name);
    }
  }

  // --- VTK pressure can bypass the projected pressure polynomial ---
  {
    constexpr const char* prefix_env_name = "DG_IGR_VTK_PREFIX_PATH";
    const char* original_prefix_text = std::getenv(prefix_env_name);
    const std::optional<std::string> original_prefix = original_prefix_text
      ? std::optional<std::string>(original_prefix_text)
      : std::nullopt;
    dg_sim::CaseConfig vtk_case;
    vtk_case.case_name = "vtk-prefix-case";
    vtk_case.vtk_config.prefix_path = "/default-vtk-prefix";
    setenv(prefix_env_name, "/tmp/dg-igr-vtk-override", 1);
    ctx.Check(dg_sim::MakeCaseVTKConfig(vtk_case).prefix_path ==
                "/tmp/dg-igr-vtk-override",
              "VTK output: environment override isolates prefix path");
    if (original_prefix)
    {
      setenv(prefix_env_name, original_prefix->c_str(), 1);
    }
    else
    {
      unsetenv(prefix_env_name);
    }

    constexpr const char* env_name = "DG_IGR_VTK_PRESSURE_COEFFICIENT";
    const char* original_text = std::getenv(env_name);
    const std::optional<std::string> original =
      original_text ? std::optional<std::string>(original_text) : std::nullopt;

    setenv(env_name, "0", 1);
    mfem::ParaViewDataCollection projected("vtk-pressure-projected",
                                            &example_ctx.Mesh());
    dg_sim::RegisterEulerVTKFields(projected, bundle.field_list);
    ctx.Check(projected.HasField("pressure"),
              "VTK pressure: default registers projected grid function");
    ctx.Check(projected.GetCoeffFieldMap().count("pressure") == 0,
              "VTK pressure: default does not register direct coefficient");

    setenv(env_name, "1", 1);
    mfem::ParaViewDataCollection direct("vtk-pressure-direct",
                                         &example_ctx.Mesh());
    dg_sim::RegisterEulerVTKFields(direct, bundle.field_list);
    ctx.Check(!direct.HasField("pressure"),
              "VTK pressure: direct mode omits projected grid function");
    ctx.Check(direct.GetCoeffFieldMap().count("pressure") == 1,
              "VTK pressure: direct mode registers pressure coefficient");

    long long output_token = 0;
    if (mfem::Mpi::Root())
    {
      output_token = std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count();
    }
    MPI_Bcast(&output_token,
              1,
              mfem::MPITypeMap<long long>::mpi_type,
              0,
              MPI_COMM_WORLD);
    const std::filesystem::path output_root =
      std::filesystem::path("/tmp") /
      ("dg_igr_vtk_pressure_direct_" + std::to_string(output_token));
    direct.SetPrefixPath(output_root.string());
    direct.SetDataFormat(mfem::VTKFormat::ASCII);
    direct.SetLevelsOfDetail(1);
    direct.SetHighOrderOutput(false);
    direct.SetCycle(0);
    direct.SetTime(0.0);
    mfem::Vector state_before_vtk(bundle.sol);
    direct.Save();
    MPI_Barrier(MPI_COMM_WORLD);
    mfem::Vector state_after_vtk(bundle.sol);
    state_after_vtk -= state_before_vtk;
    ctx.CheckNear(state_after_vtk.Normlinf(),
                  0.0,
                  0.0,
                  "VTK pressure: direct save does not mutate RK state");

    bool found_pressure = false;
    int pressure_values = 0;
    double min_saved_pressure = std::numeric_limits<double>::infinity();
    double max_saved_pressure = -std::numeric_limits<double>::infinity();
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(output_root))
    {
      if (!entry.is_regular_file() || entry.path().extension() != ".vtu")
      {
        continue;
      }
      std::ifstream input(entry.path());
      const std::string contents((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
      const std::size_t name = contents.find("Name=\"pressure\"");
      if (name == std::string::npos)
      {
        continue;
      }
      const std::size_t begin = contents.find('>', name);
      const std::size_t end = contents.find("</DataArray>", begin);
      if (begin == std::string::npos || end == std::string::npos)
      {
        continue;
      }
      found_pressure = true;
      std::istringstream values(contents.substr(begin + 1, end - begin - 1));
      double value = 0.0;
      while (values >> value)
      {
        ++pressure_values;
        min_saved_pressure = std::min(min_saved_pressure, value);
        max_saved_pressure = std::max(max_saved_pressure, value);
      }
    }
    ctx.Check(found_pressure && pressure_values > 0,
              "VTK pressure: direct coefficient is present in saved VTU");
    ctx.CheckNear(min_saved_pressure,
                  0.9,
                  1.0e-12,
                  "VTK pressure: direct saved minimum equals p(U)");
    ctx.CheckNear(max_saved_pressure,
                  0.9,
                  1.0e-12,
                  "VTK pressure: direct saved maximum equals p(U)");
    MPI_Barrier(MPI_COMM_WORLD);
    if (mfem::Mpi::Root())
    {
      std::filesystem::remove_all(output_root);
    }

    if (original)
    {
      setenv(env_name, original->c_str(), 1);
    }
    else
    {
      unsetenv(env_name);
    }
  }

  struct TimeLoopDummyOperator
  {
    int abort_calls = 0;
    mutable int proposed_step_limit_calls = 0;
    mfem::real_t last_abort_time = -1.0;
    std::uint64_t last_abort_generation = 0;
    mfem::real_t last_abort_retry_dt = -1.0;
    mfem::real_t proposed_step_limit =
      std::numeric_limits<mfem::real_t>::infinity();

    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const { return 1.0; }
    mfem::real_t LimitProposedRKStepSize(mfem::real_t proposed_dt) const
    {
      ++proposed_step_limit_calls;
      return std::min(proposed_dt, proposed_step_limit);
    }
    void AbortRKStep(const mfem::Vector&,
                     mfem::real_t restored_time,
                     std::uint64_t restored_generation,
                     mfem::real_t retry_dt)
    {
      ++abort_calls;
      last_abort_time = restored_time;
      last_abort_generation = restored_generation;
      last_abort_retry_dt = retry_dt;
    }
  };
  struct TimeLoopInitialQSigmaOperator
  {
    struct Admission
    {
      bool enabled = true;
      std::uint64_t sample_count = 8;
      std::uint64_t nonpositive_count = 3;
      std::uint64_t nonfinite_count = 2;
    };

    mutable int admission_calls = 0;
    mutable int rhs_calls = 0;
    mutable int speed_calls = 0;

    Admission InspectAcceptedStateForTransport(const mfem::Vector&,
                                                mfem::real_t) const
    {
      ++admission_calls;
      return {};
    }
    void Mult(const mfem::Vector&, mfem::Vector& y) const
    {
      ++rhs_calls;
      y = 0.0;
    }
    mfem::real_t GetMaxCharSpeed() const
    {
      ++speed_calls;
      return 1.0;
    }
  };
  struct TimeLoopCountingStepper
  {
    int calls = 0;
    mfem::real_t GetSSPCoefficient() const { return 1.0; }
    void Step(mfem::Vector&, mfem::real_t& t, mfem::real_t& dt)
    {
      ++calls;
      t += dt;
    }
  };
  struct TimeLoopGrowthBoundOperator
  {
    mfem::real_t previous_accepted_dt =
      std::numeric_limits<mfem::real_t>::quiet_NaN();

    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const { return 1.0; }
    mfem::real_t LimitProposedRKStepSize(mfem::real_t proposed_dt) const
    {
      return std::isfinite(previous_accepted_dt)
               ? std::min(proposed_dt, 2.0 * previous_accepted_dt)
               : proposed_dt;
    }
  };
  struct TimeLoopGrowthBoundStepper
  {
    explicit TimeLoopGrowthBoundStepper(TimeLoopGrowthBoundOperator& op)
      : op(op)
    {
    }

    TimeLoopGrowthBoundOperator& op;
    std::vector<mfem::real_t> accepted_dts;

    void Step(mfem::Vector&, mfem::real_t& t, mfem::real_t& dt)
    {
      accepted_dts.push_back(dt);
      op.previous_accepted_dt = dt;
      t += dt;
    }
  };
  struct TimeLoopVariableSpeedOperator
  {
    mutable int speed_calls = 0;
    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const
    {
      ++speed_calls;
      return (speed_calls == 1) ? 1.0 : 200.0;
    }
  };
  struct TimeLoopOverflowDtOperator
  {
    mutable int speed_calls = 0;
    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const
    {
      ++speed_calls;
      return (speed_calls == 1)
               ? 1.0
               : std::numeric_limits<mfem::real_t>::denorm_min();
    }
  };
  struct TimeLoopRetryStepper
  {
    explicit TimeLoopRetryStepper(int failures)
      : failures_remaining(failures)
    {
    }

    int calls = 0;
    int failures_remaining = 0; // negative means reject every attempt
    std::uint64_t generation = 7;
    std::vector<mfem::real_t> attempted_dts;

    std::uint64_t GetStateGeneration() const { return generation; }
    void RestoreStateGeneration(std::uint64_t value) { generation = value; }

    void Step(mfem::Vector& state, mfem::real_t& t, mfem::real_t& dt)
    {
      ++calls;
      attempted_dts.push_back(dt);
      if (state.Size() > 0)
      {
        state.HostReadWrite()[0] += dt;
      }
      t += dt;
      ++generation;
      if (failures_remaining != 0)
      {
        if (failures_remaining > 0)
        {
          --failures_remaining;
        }
        throw mfem::RecoverableLimiterFailure(2);
      }
    }
  };

  struct TimeLoopQSigmaRetryStepper
  {
    int calls = 0;
    bool reject_next_attempt = true;
    std::uint64_t generation = 19;
    std::vector<mfem::real_t> attempted_dts;

    bool RequiresStepTransaction() const { return true; }
    std::uint64_t GetStateGeneration() const { return generation; }
    void RestoreStateGeneration(std::uint64_t value) { generation = value; }

    void Step(mfem::Vector& state, mfem::real_t& t, mfem::real_t& dt)
    {
      ++calls;
      attempted_dts.push_back(dt);
      if (state.Size() > 0)
      {
        state.HostReadWrite()[0] += dt;
      }
      t += dt;
      ++generation;
      if (reject_next_attempt)
      {
        reject_next_attempt = false;
        throw mfem::RecoverableRKStageAdmissionFailure(
          mfem::RKStageAdmissionFailureReason::
            QSigmaNonpositiveAndNonfinite,
          /*global_nonpositive_q_sigma_samples=*/3,
          /*global_nonfinite_q_sigma_samples=*/2,
          /*stage=*/6);
      }
    }
  };

  struct TimeLoopRejectAfterAcceptedStepper
  {
    explicit TimeLoopRejectAfterAcceptedStepper(int accepted_before_reject)
      : accepted_before_reject(accepted_before_reject)
    {
    }

    int calls = 0;
    int accepted_before_reject = 0;
    std::uint64_t generation = 11;

    std::uint64_t GetStateGeneration() const { return generation; }
    void RestoreStateGeneration(std::uint64_t value) { generation = value; }

    void Step(mfem::Vector&, mfem::real_t& t, mfem::real_t& dt)
    {
      ++calls;
      t += dt;
      ++generation;
      if (calls > accepted_before_reject)
      {
        throw mfem::RecoverableLimiterFailure(1);
      }
    }
  };

  // --- CFL mode requires a typed SSP scale; fixed generic stepping does not ---
  {
    TimeLoopRetryStepper generic_stepper(/*failures=*/0);
    ctx.Check(
      !dg_sim::time_loop_detail::
        RKSSPCoefficient<TimeLoopRetryStepper>::available,
      "time loop: a generic stepper does not synthesize an SSP coefficient");

    dg_sim::TimeLoopConfig fixed_cfg;
    fixed_cfg.dt = 0.1;
    fixed_cfg.cfl = 0.0;
    dg_sim::TimeLoopResult fixed_diagnostics;
    dg_sim::PopulateExplicitCFLDiagnostics(
      fixed_diagnostics, fixed_cfg, generic_stepper);
    ctx.Check(!fixed_diagnostics.cfl_control_enabled &&
                fixed_diagnostics.cfl_safety_factor == 0.0 &&
                std::isnan(fixed_diagnostics.ssp_coefficient),
              "time loop: fixed generic stepping needs no claimed SSP scale");

#ifdef MFEM_USE_EXCEPTIONS
    const mfem::ErrorAction previous_action = mfem::get_error_action();
    mfem::set_error_action(mfem::MFEM_ERROR_THROW);
    bool missing_ssp_rejected = false;
    try
    {
      dg_sim::TimeLoopConfig cfl_cfg = fixed_cfg;
      cfl_cfg.cfl = 0.5;
      dg_sim::TimeLoopResult diagnostics;
      dg_sim::PopulateExplicitCFLDiagnostics(
        diagnostics, cfl_cfg, generic_stepper);
    }
    catch (const mfem::ErrorException&)
    {
      missing_ssp_rejected = true;
    }

    bool invalid_fixed_dt_rejected = false;
    try
    {
      dg_sim::TimeLoopConfig invalid_fixed_cfg = fixed_cfg;
      invalid_fixed_cfg.dt = 0.0;
      dg_sim::TimeLoopResult diagnostics;
      dg_sim::PopulateExplicitCFLDiagnostics(
        diagnostics, invalid_fixed_cfg, generic_stepper);
    }
    catch (const mfem::ErrorException&)
    {
      invalid_fixed_dt_rejected = true;
    }
    mfem::set_error_action(previous_action);
    ctx.Check(missing_ssp_rejected && invalid_fixed_dt_rejected,
              "time loop: invalid CFL contracts fail before operator work");
#endif
  }

  // --- an invalid exact initial/restart anchor fails before the CFL RHS ---
  {
    TimeLoopInitialQSigmaOperator op;
    TimeLoopCountingStepper stepper;
    const mfem::Vector state_before(bundle.sol);
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.6;
    cfg.dt = 0.1;
    cfg.dt_min = 0.01;
    cfg.cfl = 0.5;
    cfg.order = 1;
    cfg.vtk_save = false;
    cfg.initial_time = 0.5;
    cfg.initial_step = 9;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(
        op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    mfem::Vector state_error(bundle.sol);
    state_error -= state_before;
    ctx.Check(!result.success && result.steps == 9 &&
                result.accepted_attempts == 0 &&
                result.rejected_attempts == 1 &&
                result.rejected_q_sigma_attempts == 1 &&
                result.rejected_q_sigma_nonpositive_samples == 3 &&
                result.rejected_q_sigma_nonfinite_samples == 2 &&
                result.last_q_sigma_rejection_reason ==
                  "q-sigma-nonpositive-and-nonfinite" &&
                result.last_q_sigma_rejection_stage == -1 &&
                result.rhs_evaluations == 0 &&
                result.characteristic_speed_probes == 0 &&
                result.termination_reason ==
                  "initial-q-sigma-admission-failed",
              "initial q_sigma admission reports one bounded accepted-state "
              "failure");
    ctx.Check(op.admission_calls == 1 && op.rhs_calls == 0 &&
                op.speed_calls == 0 && stepper.calls == 0 &&
                state_error.Normlinf() == 0.0,
              "initial q_sigma admission precedes the restart CFL probe and "
              "leaves state untouched");
    ctx.CheckNear(result.final_time,
                  cfg.initial_time,
                  0.0,
                  "initial q_sigma admission remains at the restart horizon");
    const dg_sim::RunSummary summary =
      dg_sim::BuildRunSummary(example_ctx,
                              result,
                              /*elapsed_seconds=*/0.0,
                              /*num_ranks=*/1);
    ctx.Check(summary.cfl_diagnostics_observed &&
                summary.cfl_control_enabled &&
                summary.cfl_safety_factor == 0.5 &&
                summary.ssp_coefficient == 1.0 &&
                summary.ssp_scaled_cfl_factor == 0.5 &&
                summary.nominal_global_courant == 0.5 / 3.0,
              "run summary: early failure preserves observed CFL factors");
    std::ostringstream structured_summary;
    dg_sim::EmitExplicitTimeRunSummary(result, structured_summary);
    const std::string structured_text = structured_summary.str();
    ctx.Check(structured_text.find("event=run-summary") != std::string::npos &&
                structured_text.find("cfl_control_enabled=1") !=
                  std::string::npos &&
                structured_text.find("cfl_safety_factor=0.5") !=
                  std::string::npos &&
                structured_text.find("ssp_coefficient=1") !=
                  std::string::npos &&
                structured_text.find("termination_reason="
                                     "initial-q-sigma-admission-failed") !=
                  std::string::npos,
              "time loop: canonical early summary exposes the CFL contract");
  }

  // --- a solver-level successful preflight is not repeated in the loop ---
  {
    TimeLoopInitialQSigmaOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.1;
    cfg.dt = 0.1;
    cfg.dt_min = 0.01;
    cfg.cfl = 0.0;
    cfg.order = 1;
    cfg.vtk_save = false;
    cfg.initial_q_sigma_admitted = true;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(
        op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(result.success && result.accepted_attempts == 1 &&
                op.admission_calls == 0 && stepper.calls == 1,
              "time loop trusts one explicitly completed solver-level "
              "accepted-state q_sigma preflight");
  }

  // --- limiter-stage rejection restores the transaction and retries at 1/2 dt ---
  {
    TimeLoopDummyOperator op;
    TimeLoopRetryStepper stepper(/*failures=*/1);
    const mfem::Vector state_before(bundle.sol);
    const mfem::real_t first_value_before =
      bundle.sol.Size() > 0 ? bundle.sol.HostRead()[0] : 0.0;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.1;
    cfg.dt = 0.1;
    cfg.dt_min = 0.01;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    int rejection_hooks = 0;
    bool hook_observed_restored_transaction = true;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_rejected_attempt =
      [&](int step,
          mfem::real_t t,
          mfem::real_t attempted_dt,
          mfem::real_t retry_dt)
    {
      ++rejection_hooks;
      hook_observed_restored_transaction =
        hook_observed_restored_transaction && step == 0 && t == 0.0 &&
        (bundle.sol.Size() == 0 ||
         bundle.sol.HostRead()[0] == first_value_before) &&
        stepper.GetStateGeneration() == 7 && attempted_dt == 0.1 &&
        retry_dt == 0.05;
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(
        op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.steps == 2 && result.accepted_attempts == 2,
              "time loop retry: only accepted attempts advance step counters");
    ctx.Check(result.rejected_attempts == 1 &&
                result.rejected_limiter_elements == 2,
              "time loop retry: rejected limiter work is aggregated");
    const dg_sim::RunSummary retry_summary = dg_sim::BuildRunSummary(
      example_ctx,
      result,
      /*elapsed_seconds=*/1.0,
      mfem::Mpi::WorldSize());
    ctx.Check(retry_summary.accepted_attempts == 2 &&
                retry_summary.rejected_attempts == 1 &&
                retry_summary.rejected_limiter_elements == 2 &&
                retry_summary.minimum_attempted_dt == 0.05 &&
                retry_summary.minimum_accepted_dt == 0.05 &&
                retry_summary.maximum_attempted_dt == 0.1 &&
                retry_summary.maximum_accepted_dt == 0.05 &&
                !retry_summary.dt_min_reached && retry_summary.success &&
                retry_summary.termination_reason == "target-reached",
              "time loop retry: run summary preserves bounded aggregates");
    ctx.CheckNear(result.minimum_attempted_dt,
                  0.05,
                  0.0,
                  "time loop retry: minimum attempted dt is aggregated");
    ctx.CheckNear(result.minimum_accepted_dt,
                  0.05,
                  0.0,
                  "time loop retry: minimum accepted dt is aggregated");
    ctx.CheckNear(result.maximum_attempted_dt,
                  0.1,
                  0.0,
                  "time loop retry: maximum attempted dt is aggregated");
    ctx.CheckNear(result.maximum_accepted_dt,
                  0.05,
                  0.0,
                  "time loop retry: maximum accepted dt is aggregated");
    ctx.Check(stepper.calls == 3 && stepper.attempted_dts.size() == 3 &&
                stepper.attempted_dts[0] == 0.1 &&
                stepper.attempted_dts[1] == 0.05 &&
                stepper.attempted_dts[2] == 0.05,
              "time loop retry: fixed nominal dt remains at the reduced value");
    ctx.Check(rejection_hooks == 1 && hook_observed_restored_transaction,
              "time loop retry: observation hook sees restored transaction");
    ctx.Check(op.abort_calls == 1 && op.last_abort_time == 0.0 &&
                op.last_abort_generation == 7 &&
                op.last_abort_retry_dt == 0.05,
              "time loop retry: operator abort sees restored transaction");
    ctx.CheckNear(result.final_time,
                  0.1,
                  1.0e-14,
                  "time loop retry: accepted attempts reach the horizon");
    if (bundle.sol.Size() > 0)
    {
      ctx.CheckNear(bundle.sol.HostRead()[0],
                    first_value_before + 0.1,
                    1.0e-14,
                    "time loop retry: rejected state mutation is not committed");
    }
    ctx.Check(stepper.GetStateGeneration() == 9,
              "time loop retry: rejected generation is rolled back");
    bundle.sol = state_before;
  }

  // --- an RHS q_sigma failure uses the same exact rollback/retry path ---
  {
    TimeLoopDummyOperator op;
    TimeLoopQSigmaRetryStepper stepper;
    const mfem::Vector state_before(bundle.sol);
    const mfem::real_t first_value_before =
      bundle.sol.Size() > 0 ? bundle.sol.HostRead()[0] : 0.0;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.1;
    cfg.dt = 0.1;
    cfg.dt_min = 0.01;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(
        op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(result.success && result.steps == 2 &&
                result.accepted_attempts == 2 &&
                result.rejected_attempts == 1,
              "q_sigma admission retry: only accepted attempts commit");
    ctx.Check(result.rejected_limiter_elements == 0 &&
                result.rejected_q_sigma_attempts == 1 &&
                result.rejected_q_sigma_nonpositive_samples == 3 &&
                result.rejected_q_sigma_nonfinite_samples == 2 &&
                result.last_q_sigma_rejection_reason ==
                  "q-sigma-nonpositive-and-nonfinite" &&
                result.last_q_sigma_rejection_stage == 6,
              "q_sigma admission retry: bounded reason and sample counts are "
              "reported without masquerading as limiter work");
    ctx.Check(stepper.calls == 3 && stepper.attempted_dts.size() == 3 &&
                stepper.attempted_dts[0] == 0.1 &&
                stepper.attempted_dts[1] == 0.05 &&
                stepper.attempted_dts[2] == 0.05 &&
                stepper.GetStateGeneration() == 21,
              "q_sigma admission retry: failed generation is rolled back and "
              "the timestep is halved");
    ctx.Check(op.abort_calls == 1 && op.last_abort_time == 0.0 &&
                op.last_abort_generation == 19 &&
                op.last_abort_retry_dt == 0.05,
              "q_sigma admission retry: operator abort sees the accepted "
              "transaction and retry timestep");
    if (bundle.sol.Size() > 0)
    {
      ctx.CheckNear(bundle.sol.HostRead()[0],
                    first_value_before + 0.1,
                    1.0e-14,
                    "q_sigma admission retry: failed state mutation is not "
                    "committed");
    }
    bundle.sol = state_before;
  }

  // --- retry-cap exhaustion is distinct from a configured dt floor ---
  {
    TimeLoopDummyOperator op;
    TimeLoopRetryStepper stepper(/*failures=*/-1);
    const mfem::Vector state_before(bundle.sol);
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.04;
    cfg.dt_min = 0.0;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(
        op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(!result.success && !result.dt_min_reached &&
                result.termination_reason == "limiter-retry-limit",
              "time loop retry: retry-cap exhaustion has its own failure reason");
    ctx.Check(result.rejected_attempts == 64 && stepper.calls == 64 &&
                result.accepted_attempts == 0 && result.steps == 0,
              "time loop retry: retry cap commits no stage or time step");
    mfem::Vector rollback_error(bundle.sol);
    rollback_error -= state_before;
    ctx.Check(rollback_error.Normlinf() == 0.0,
              "time loop retry: retry-cap exhaustion preserves state");
    bundle.sol = state_before;
  }

  // --- the application wrapper propagates retry exhaustion as a failed run ---
  {
    struct RejectingRunOperator : public mfem::TimeDependentOperator
    {
      explicit RejectingRunOperator(int size)
        : mfem::TimeDependentOperator(size)
      {
      }
      void Mult(const mfem::Vector&, mfem::Vector& derivative) const override
      {
        derivative = 0.0;
      }
      mfem::real_t GetMaxCharSpeed() const { return 1.0; }
    } op(bundle.sol.Size());

    const mfem::Vector state_before(bundle.sol);
    mfem::RK3SSPWithLimiter stepper(example_ctx.spaces.vfes);
    stepper.AddAlwaysCallback(
      [](mfem::ParFiniteElementSpace&,
         mfem::ParGridFunction&,
         double)
      { throw mfem::RecoverableLimiterFailure(1); });

    dg_sim::CaseConfig failure_case;
    failure_case.case_name = "retry-limit-integration";
    dg_sim::RunConfig failure_run = run_cfg;
    failure_run.case_config = failure_case;
    failure_run.opts.t_final = 0.1;
    failure_run.opts.dt = 0.1;
    failure_run.opts.dt_min = 0.0;
    failure_run.opts.cfl = 0.0;
    failure_run.opts.vtk_save = false;
    failure_run.opts.glvis = false;
    failure_run.opts.vis_steps = 0;
    failure_run.opts.analysis_steps = 0;
    dg_sim::TimeLoopCallbacks callbacks;
    dg_sim::RunSummary summary;
    bool glvis_enabled = false;
    const bool ok = dg_sim::RunEulerCase(example_ctx,
                                         failure_case,
                                         failure_run,
                                         bundle,
                                         stepper,
                                         op,
                                         glvis_enabled,
                                         callbacks,
                                         /*precision=*/8,
                                         &summary);
    int application_failure_ok = 1;
    if (mfem::Mpi::Root())
    {
      application_failure_ok = static_cast<int>(
        !ok && !summary.success && !summary.dt_min_reached &&
        summary.termination_reason == "limiter-retry-limit" &&
        summary.rhs_evaluations == 64 && summary.accepted_attempts == 0 &&
        summary.rejected_attempts == 64 &&
        summary.cfl_diagnostics_observed &&
        !summary.cfl_control_enabled && summary.cfl_safety_factor == 0.0 &&
        summary.ssp_coefficient == 1.0);
    }
    MPI_Bcast(&application_failure_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ctx.Check(application_failure_ok == 1,
              "RunEulerCase: retry exhaustion propagates as a typed failure");
    mfem::Vector rollback_error(bundle.sol);
    rollback_error -= state_before;
    ctx.Check(rollback_error.Normlinf() == 0.0,
              "RunEulerCase: retry exhaustion commits no conservative state");

    const char* prefix_env_name = "DG_IGR_VTK_PREFIX_PATH";
    const char* old_prefix = std::getenv(prefix_env_name);
    const std::optional<std::string> original_prefix =
      old_prefix ? std::optional<std::string>(old_prefix) : std::nullopt;
    long long output_token = 0;
    if (mfem::Mpi::Root())
    {
      output_token = std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count();
    }
    MPI_Bcast(&output_token,
              1,
              mfem::MPITypeMap<long long>::mpi_type,
              0,
              MPI_COMM_WORLD);
    const std::filesystem::path output_root =
      std::filesystem::path("/tmp") /
      ("dg_igr_retry_limit_metadata_" + std::to_string(output_token));
    setenv(prefix_env_name, output_root.c_str(), 1);
    char program_name[] = "test";
    char* metadata_argv[] = {program_name};
    dg_sim::RunConfig metadata_run = failure_run;
    metadata_run.opts.vtk_save = true;
    dg_sim::WriteRunMetadataSidecar(failure_case,
                                    metadata_run,
                                    &summary,
                                    /*igr=*/nullptr,
                                    /*c_alpha_value=*/0.0,
                                    /*argc=*/1,
                                    metadata_argv,
                                    /*success=*/ok,
                                    "retry-limit-integration");
    MPI_Barrier(MPI_COMM_WORLD);
    int metadata_ok = 1;
    if (mfem::Mpi::Root())
    {
      std::ifstream input(output_root / "retry-limit-integration" /
                          "run_metadata.json");
      const std::string contents((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
      metadata_ok = static_cast<int>(
        input.is_open() &&
        contents.find("\"success\": false") != std::string::npos &&
        contents.find("\"cfl_diagnostics_observed\": true") !=
          std::string::npos &&
        contents.find("\"cfl_diagnostics_source\": \"run-summary\"") !=
          std::string::npos &&
        contents.find("\"ssp_scaled_cfl_factor\": null") !=
          std::string::npos &&
        contents.find("\"termination_reason\": \"limiter-retry-limit\"") !=
          std::string::npos);
      std::filesystem::remove_all(output_root);
    }
    MPI_Bcast(&metadata_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ctx.Check(metadata_ok == 1,
              "RunEulerCase: retry-limit failure persists in run metadata");
    if (original_prefix)
    {
      setenv(prefix_env_name, original_prefix->c_str(), 1);
    }
    else
    {
      unsetenv(prefix_env_name);
    }
    bundle.sol = state_before;
  }

  // --- a limiter rejection at dt_min restores state and terminates cleanly ---
  {
    TimeLoopDummyOperator op;
    TimeLoopRetryStepper stepper(/*failures=*/-1);
    const mfem::Vector state_before(bundle.sol);
    const mfem::real_t first_value_before =
      bundle.sol.Size() > 0 ? bundle.sol.HostRead()[0] : 0.0;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.04;
    cfg.dt_min = 0.02;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    int rejection_hooks = 0;
    mfem::real_t last_retry_dt = -1.0;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_rejected_attempt =
      [&](int,
          mfem::real_t,
          mfem::real_t,
          mfem::real_t retry_dt)
    {
      ++rejection_hooks;
      last_retry_dt = retry_dt;
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(
        op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.dt_min_reached && result.steps == 0 &&
                result.accepted_attempts == 0 && !result.success &&
                result.termination_reason == "limiter-dt-min",
              "time loop retry: rejection at dt_min commits no step");
    ctx.Check(result.rejected_attempts == 2 && stepper.calls == 2 &&
                stepper.attempted_dts[0] == 0.04 &&
                stepper.attempted_dts[1] == 0.02,
              "time loop retry: dt_min is attempted once, then terminates");
    ctx.Check(rejection_hooks == 2 && last_retry_dt == 0.0,
              "time loop retry: terminal rejection hook is notified");
    ctx.Check(op.abort_calls == 2 && op.last_abort_time == 0.0 &&
                op.last_abort_generation == 7 &&
                op.last_abort_retry_dt == 0.0,
              "time loop retry: terminal operator abort sees no retry dt");
    ctx.CheckNear(result.final_time,
                  0.0,
                  0.0,
                  "time loop retry: terminal rejection restores time");
    ctx.Check((bundle.sol.Size() == 0 ||
               bundle.sol.HostRead()[0] == first_value_before) &&
                stepper.GetStateGeneration() == 7,
              "time loop retry: terminal rejection restores state/provenance");
    bundle.sol = state_before;
  }

  // --- a first-attempt terminal rejection does not duplicate an initial save ---
  {
    TimeLoopDummyOperator op;
    TimeLoopRetryStepper stepper(/*failures=*/-1);
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.04;
    cfg.dt_min = 0.04;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;
    cfg.initial_state_saved = true;
    cfg.save_index_offset = 1;

    int save_calls = 0;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int, mfem::real_t, int, bool) { ++save_calls; };

    const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
      op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(!result.success && result.steps == 0 &&
                result.termination_reason == "limiter-dt-min",
              "time loop failure output: first attempt rejects at initial time");
    ctx.Check(save_calls == 0,
              "time loop failure output: initial save time is not duplicated");
  }

  // --- terminal rejection after a save boundary does not duplicate its time ---
  {
    TimeLoopDummyOperator op;
    TimeLoopRejectAfterAcceptedStepper stepper(/*accepted_before_reject=*/1);
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.3;
    cfg.dt = 0.1;
    cfg.dt_min = 0.1;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;
    cfg.initial_state_saved = true;
    cfg.save_index_offset = 1;

    std::vector<mfem::real_t> save_times;
    std::vector<int> save_cycles;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int,
                        mfem::real_t time,
                        int cycle,
                        bool)
    {
      save_times.push_back(time);
      save_cycles.push_back(cycle);
    };

    const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
      op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(!result.success && result.steps == 1 &&
                result.termination_reason == "limiter-dt-min",
              "time loop failure output: rejection preserves saved boundary");
    ctx.Check(save_times.size() == 1 && save_cycles == std::vector<int>({1}),
              "time loop failure output: saved cycle is not duplicated");
    if (save_times.size() == 1)
    {
      ctx.CheckNear(save_times[0],
                    0.1,
                    1.0e-14,
                    "time loop failure output: retained save time is exact");
    }
  }

  // --- dt_min guard uses nominal dt (before stepping) ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.04;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(result.dt_min_reached,
              "time loop: dt_min guard uses nominal dt before stepping");
    ctx.Check(!result.success &&
                result.termination_reason == "initial-dt-min",
              "time loop: initial dt_min guard records a typed reason");
    ctx.Check(stepper.calls == 0,
              "time loop: dt_min guard aborts before first step");
    ctx.Check(result.steps == 0,
              "time loop: dt_min abort leaves completed step count at zero");
  }

  // --- a restarted initial-dt failure emits its finite committed state ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.initial_time = 0.25;
    cfg.initial_step = 7;
    cfg.t_final = 0.5;
    cfg.dt = 0.01;
    cfg.dt_min = 0.02;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;
    cfg.save_index_offset = 4;
    cfg.initial_state_saved = false;

    int save_calls = 0;
    int saved_step = -1;
    int saved_cycle = -1;
    mfem::real_t saved_time = -1.0;
    bool saved_final = false;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step,
                        mfem::real_t time,
                        int cycle,
                        bool is_final)
    {
      ++save_calls;
      saved_step = step;
      saved_cycle = cycle;
      saved_time = time;
      saved_final = is_final;
    };

    const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
      op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(!result.success && result.termination_reason == "initial-dt-min" &&
                result.steps == 7 && stepper.calls == 0,
              "time loop failure output: restarted dt floor is typed");
    ctx.Check(save_calls == 1 && saved_step == 7 && saved_cycle == 4 &&
                saved_final,
              "time loop failure output: restarted committed state is saved");
    ctx.CheckNear(saved_time,
                  0.25,
                  0.0,
                  "time loop failure output: restart save retains checkpoint time");
  }

  // --- non-finite post-accept CFL dt emits the finite committed state ---
  {
    TimeLoopOverflowDtOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 1.0;
    cfg.dt = 0.0;
    cfg.dt_min = 0.0;
    cfg.cfl = 0.3;
    cfg.order = 1;
    cfg.vtk_save = true;
    cfg.dt_save = 100.0;

    int final_save_calls = 0;
    int final_save_step = -1;
    mfem::real_t final_save_time = -1.0;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step,
                        mfem::real_t time,
                        int,
                        bool is_final)
    {
      if (is_final)
      {
        ++final_save_calls;
        final_save_step = step;
        final_save_time = time;
      }
    };

    const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
      op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(!result.success &&
                result.termination_reason == "nonfinite-cfl-dt" &&
                result.steps == 1 && stepper.calls == 1,
              "time loop failure output: overflowed CFL update is typed");
    ctx.Check(final_save_calls == 1 && final_save_step == result.steps,
              "time loop failure output: post-accept state is saved once");
    ctx.CheckNear(final_save_time,
                  result.final_time,
                  0.0,
                  "time loop failure output: saved time is committed time");
  }

  // --- clipped/snap dt_real below dt_min is allowed (bookkeeping policy) ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.41;
    cfg.dt = 0.2;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(!result.dt_min_reached,
              "time loop: clipped final dt_real does not trigger dt_min guard");
    ctx.Check(stepper.calls == 3,
              "time loop: clipped final dt_real still advances final step");
    ctx.Check(result.steps == 3,
              "time loop: step counter includes clipped final step");
    ctx.CheckNear(result.final_time,
                  0.41,
                  1.0e-12,
                  "time loop: clipped final dt_real reaches target final time");
    ctx.Check(result.success && result.termination_reason == "target-reached",
              "time loop: clipped final step records successful termination");
  }

  // --- an operator bound respects dt_min before any new attempt ---
  for (const int allowed_steps : {0, 1})
  {
    TimeLoopDummyOperator op;
    op.proposed_step_limit = allowed_steps == 0 ? 0.04 : 0.06;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.initial_time = 0.25;
    cfg.initial_step = 7;
    cfg.t_final = 0.5;
    cfg.dt = 0.2;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 1.0;
    cfg.analysis_steps = 1;
    cfg.save_index_offset = 4;

    int final_save_calls = 0;
    int saved_step = -1;
    int saved_cycle = -1;
    mfem::real_t saved_time = -1.0;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_analysis = [&](int, mfem::real_t, mfem::real_t, bool)
    {
      op.proposed_step_limit = 0.04;
    };
    hooks.on_save = [&](int step, mfem::real_t time, int cycle, bool is_final)
    {
      if (is_final)
      {
        ++final_save_calls;
        saved_step = step;
        saved_cycle = cycle;
        saved_time = time;
      }
    };
    const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
      op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(!result.success && result.dt_min_reached &&
                result.termination_reason == "operator-dt-min" &&
                stepper.calls == allowed_steps &&
                result.accepted_attempts == allowed_steps &&
                result.segment_steps == allowed_steps &&
                result.steps == cfg.initial_step + allowed_steps,
              "time loop: operator dt floor preserves accepted step counts");
    ctx.CheckNear(result.final_time,
                  cfg.initial_time + allowed_steps * 0.06,
                  1.0e-14,
                  "time loop: operator dt floor retains committed time");
    ctx.Check(final_save_calls == 1 && saved_step == result.steps &&
                saved_cycle == cfg.save_index_offset &&
                saved_time == result.final_time,
              "time loop: operator dt floor saves the committed state once");
  }

  // --- an operator can bound nominal growth before horizon/save clipping ---
  {
    TimeLoopDummyOperator op;
    op.proposed_step_limit = 0.06;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.14;
    cfg.dt = 0.2;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(result.success && !result.dt_min_reached && result.steps == 3 &&
                stepper.calls == 3 &&
                op.proposed_step_limit_calls == 3,
              "time loop: operator step-growth limit applies before every "
              "attempt and preserves final-horizon clipping");
    ctx.CheckNear(result.minimum_accepted_dt,
                  0.02,
                  1.0e-12,
                  "time loop: operator growth limit leaves the final clip");
    ctx.CheckNear(result.maximum_accepted_dt,
                  0.06,
                  1.0e-12,
                  "time loop: operator growth limit bounds nominal steps");
  }

  // --- short save/final clips are allowed after a valid operator bound ---
  {
    TimeLoopDummyOperator op;
    op.proposed_step_limit = 0.06;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.145;
    cfg.dt = 0.2;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.07;
    std::vector<mfem::real_t> save_times;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int, mfem::real_t time, int, bool)
    {
      save_times.push_back(time);
    };
    const dg_sim::TimeLoopResult result = dg_sim::RunTimeLoop(
      op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.success && !result.dt_min_reached &&
                result.termination_reason == "target-reached" &&
                result.steps == 5 && stepper.calls == 5 &&
                save_times.size() == 3,
              "time loop: operator dt floor permits short save and final clips");
    ctx.CheckNear(result.minimum_accepted_dt,
                  0.005,
                  1.0e-12,
                  "time loop: final bookkeeping clip can remain below dt_min");
    if (save_times.size() == 3)
    {
      ctx.CheckNear(save_times[0], 0.07, 1.0e-12,
                    "time loop: first short save clip reaches its boundary");
      ctx.CheckNear(save_times[1], 0.14, 1.0e-12,
                    "time loop: second short save clip reaches its boundary");
      ctx.CheckNear(save_times[2], cfg.t_final, 1.0e-12,
                    "time loop: final clip retains its final save");
    }
  }

  // --- a tiny save clip cannot amplify the next accepted step above 2x ---
  {
    TimeLoopGrowthBoundOperator op;
    TimeLoopGrowthBoundStepper stepper(op);
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.18;
    cfg.dt = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.07;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    bool ratios_bounded = stepper.accepted_dts.size() == 5;
    for (std::size_t i = 1; i < stepper.accepted_dts.size(); ++i)
    {
      ratios_bounded = ratios_bounded &&
                       stepper.accepted_dts[i] <=
                         2.0 * stepper.accepted_dts[i - 1] + 1.0e-14;
    }
    ctx.Check(result.success && ratios_bounded,
              "time loop: output-clipped accepted step limits every later "
              "growth ratio to the operator policy");
    ctx.CheckNear(result.maximum_accepted_step_growth_ratio,
                  2.0,
                  1.0e-12,
                  "time loop: bounded maximum accepted growth is measured");
    ctx.CheckNear(result.final_time,
                  0.18,
                  1.0e-12,
                  "time loop: growth limiting preserves the physical horizon");
  }

  // --- dt_min after CFL update records current step and emits final save ---
  {
    TimeLoopVariableSpeedOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 1.0;
    cfg.dt = 0.0;
    cfg.dt_min = 0.05;
    // The two-element fixture has half the original characteristic length;
    // retain the intended first accepted step of 0.1 before the speed jump.
    cfg.cfl = 0.6;
    cfg.order = 1;
    cfg.vtk_save = true;
    cfg.dt_save = 100.0; // avoid save-boundary snapping in this test

    int final_save_calls = 0;
    int final_save_step = -1;
    mfem::real_t final_save_time = -1.0;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step, mfem::real_t t, int /*save_idx*/, bool is_final)
    {
      if (is_final)
      {
        ++final_save_calls;
        final_save_step = step;
        final_save_time = t;
      }
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.dt_min_reached,
              "time loop: dt_min set when CFL update drops nominal dt below threshold");
    ctx.Check(!result.success && result.termination_reason == "cfl-dt-min" &&
                result.rhs_evaluations == 1,
              "time loop: CFL floor reason and initial probe work are recorded");
    ctx.Check(stepper.calls == 1,
              "time loop: dt_min after CFL update occurs after one completed step");
    ctx.Check(result.steps == 1,
              "time loop: dt_min after CFL update preserves completed step count");
    ctx.CheckNear(result.final_time,
                  0.1,
                  1.0e-12,
                  "time loop: dt_min after CFL update preserves advanced time");
    ctx.Check(final_save_calls == 1,
              "time loop: dt_min after CFL update emits one final save callback");
    ctx.Check(final_save_step == result.steps,
              "time loop: final save step matches reported completed steps");
    ctx.CheckNear(final_save_time,
                  result.final_time,
                  1.0e-12,
                  "time loop: final save time matches reported final time");
  }

  // --- zero-step guard: do not call Step when t_final is already reached ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.0;
    cfg.dt = 0.1;
    cfg.dt_min = 0.0;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(stepper.calls == 0,
              "time loop: zero dt_real does not invoke stepper");
    ctx.Check(result.steps == 0,
              "time loop: zero dt_real exits before any step is counted");
    ctx.Check(!result.dt_min_reached,
              "time loop: zero-step exit is distinct from dt_min guard");
    ctx.Check(result.success && result.termination_reason == "target-reached",
              "time loop: zero-step horizon is a successful termination");
    ctx.CheckNear(result.final_time,
                  0.0,
                  1.0e-14,
                  "time loop: zero-step exit keeps initial final_time");
  }

  // --- save-boundary snapping advances even when on_save hook is missing ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.5;
    cfg.dt = 0.2;
    cfg.dt_min = 0.0;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(stepper.calls == 5,
              "time loop: save snapping without hook still advances all steps");
    ctx.Check(result.steps == 5,
              "time loop: step counter reaches final time without save hook");
    ctx.CheckNear(result.final_time,
                  0.5,
                  1.0e-12,
                  "time loop: save snapping without hook reaches t_final");
  }

  // --- final time on save boundary emits one final save, not a duplicate ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.1;
    cfg.dt_min = 1.0e-12;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;

    int save_count = 0;
    int final_save_count = 0;
    int last_save_step = -1;
    int last_save_idx = -1;
    mfem::real_t last_save_time = -1.0;
    bool last_save_final = false;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step,
                        mfem::real_t t,
                        int save_idx,
                        bool is_final)
    {
      ++save_count;
      if (is_final)
      {
        ++final_save_count;
      }
      last_save_step = step;
      last_save_idx = save_idx;
      last_save_time = t;
      last_save_final = is_final;
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.steps == 2,
              "time loop: final save-boundary case completes two steps");
    ctx.Check(save_count == 2,
              "time loop: final save boundary avoids duplicate final save");
    ctx.Check(final_save_count == 1,
              "time loop: final save boundary marks exactly one save final");
    ctx.Check(last_save_step == result.steps && last_save_idx == 1 &&
                last_save_final,
              "time loop: final boundary save carries final step/index flag");
    ctx.CheckNear(last_save_time,
                  result.final_time,
                  1.0e-12,
                  "time loop: final boundary save time matches final time");
  }

  // --- restart semantics preserve cumulative steps and save numbering ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.initial_time = 0.2;
    cfg.initial_step = 7;
    cfg.t_final = 0.4;
    cfg.dt = 0.1;
    cfg.dt_min = 1.0e-12;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;
    cfg.next_save_time = 0.3;
    cfg.save_index_offset = 3;

    std::vector<int> save_steps;
    std::vector<int> save_indices;
    std::vector<mfem::real_t> save_times;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step,
                        mfem::real_t t,
                        int save_idx,
                        bool /*is_final*/)
    {
      save_steps.push_back(step);
      save_indices.push_back(save_idx);
      save_times.push_back(t);
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.steps == 9,
              "time loop restart: cumulative step count advances from saved step");
    ctx.Check(result.segment_steps == 2,
              "time loop restart: segment step count excludes saved history");
    ctx.Check(stepper.calls == 2,
              "time loop restart: only the resumed segment is advanced");
    ctx.Check(save_steps == std::vector<int>({8, 9}),
              "time loop restart: save callbacks use cumulative steps");
    ctx.Check(save_indices == std::vector<int>({3, 4}),
              "time loop restart: save indices continue monotonically");
    ctx.Check(save_times.size() == 2,
              "time loop restart: no duplicate imported checkpoint is saved");
    if (save_times.size() == 2)
    {
      ctx.CheckNear(save_times[0],
                    0.3,
                    1.0e-12,
                    "time loop restart: first save uses the next boundary");
      ctx.CheckNear(save_times[1],
                    0.4,
                    1.0e-12,
                    "time loop restart: final save reaches absolute tf");
    }
  }

  int alive_count = 0;
  int save_count = 0;
  int analysis_count = 0;
  dg_sim::TimeLoopInfo last_alive;
  dg_sim::VTKSaveInfo last_save;
  dg_sim::AnalysisInfo last_analysis;

  dg_sim::TimeLoopCallbacks callbacks;
  callbacks.wants_minmax = true;
  callbacks.on_alive = [&](const dg_sim::TimeLoopInfo& info)
  {
    ++alive_count;
    last_alive = info;
  };
  callbacks.on_vtk_save = [&](const dg_sim::VTKSaveInfo& info)
  {
    ++save_count;
    last_save = info;
  };
  callbacks.on_analysis = [&](const dg_sim::AnalysisInfo& info)
  {
    ++analysis_count;
    last_analysis = info;
  };

  dg_sim::TimeLoopIO io;
  bool glvis_enabled = false;
  io.glvis_enabled = &glvis_enabled;
  int sigma_prepare_count = 0;
  int sigma_prepare_step = -1;
  mfem::real_t sigma_prepare_time = -1.0;
  bool sigma_prepare_final = true;
  bool sigma_output_allowed = true;
  io.prepare_sigma = [&](int step, mfem::real_t t, bool is_final)
  {
    ++sigma_prepare_count;
    sigma_prepare_step = step;
    sigma_prepare_time = t;
    sigma_prepare_final = is_final;
    return dg_sim::SigmaOutputPreparation{0.0, sigma_output_allowed};
  };

  dg_sim::TimeLoopHooks hooks =
    dg_sim::MakeEulerTimeLoopHooks(example_ctx,
                                   bundle,
                                   mfem::Mpi::WorldSize(),
                                   mfem::Mpi::WorldRank(),
                                   io,
                                   callbacks);

  ctx.Check(static_cast<bool>(hooks.on_visualize),
            "time hooks: on_visualize is wired");
  ctx.Check(static_cast<bool>(hooks.on_save), "time hooks: on_save is wired");
  ctx.Check(static_cast<bool>(hooks.on_analysis),
            "time hooks: on_analysis is wired");
  ctx.Check(!hooks.on_rejected_attempt,
            "time hooks: rejected-step algebraic recovery is operator-owned");

  hooks.on_visualize(/*step=*/5, /*t=*/0.2, /*dt=*/0.01, /*done=*/false);
  hooks.on_save(/*step=*/5, /*t=*/0.2, /*save_idx=*/3, /*is_final=*/false);
  hooks.on_analysis(/*step=*/5, /*t=*/0.2, /*dt=*/0.01, /*done=*/true);

  ctx.Check(alive_count == 1, "time hooks: on_alive dispatched once");
  ctx.Check(save_count == 1, "time hooks: on_vtk_save dispatched once");
  ctx.Check(analysis_count == 1, "time hooks: on_analysis dispatched once");
  ctx.Check(sigma_prepare_count == 1 && sigma_prepare_step == 5 &&
              sigma_prepare_time == 0.2 && !sigma_prepare_final,
            "time hooks: save preparation remains independently wired");

  sigma_output_allowed = false;
  hooks.on_save(/*step=*/6, /*t=*/0.3, /*save_idx=*/4, /*is_final=*/true);
  ctx.Check(save_count == 1 && sigma_prepare_count == 2,
            "time hooks: denied algebraic preparation suppresses final output");

  ctx.Check(last_alive.step == 5 && !last_alive.is_final,
            "time hooks: alive payload carries step/final flag");
  ctx.Check(last_save.cycle == 3 && !last_save.is_initial &&
              !last_save.is_final,
            "time hooks: save payload carries cycle/flags");
  ctx.Check(last_analysis.base.step == 5 && last_analysis.base.is_final,
            "time hooks: analysis payload carries step/final flag");
  ctx.Check(last_analysis.has_rho && last_analysis.has_p,
            "time hooks: analysis min/max payload enabled");
  ctx.CheckNear(last_analysis.rho_min,
                1.2,
                1.0e-10,
                "time hooks: rho_min for constant state");
  ctx.CheckNear(last_analysis.rho_max,
                1.2,
                1.0e-10,
                "time hooks: rho_max for constant state");
  ctx.CheckNear(last_analysis.p_min,
                0.9,
                1.0e-10,
                "time hooks: p_min for constant state");
  ctx.CheckNear(last_analysis.p_max,
                0.9,
                1.0e-10,
                "time hooks: p_max for constant state");

  return ctx.failures;
}

} // namespace dg_igr_tests
