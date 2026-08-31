#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/solver/solver_helpers.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/time/time_defaults.hpp"

#include "test_utils.hpp"

#include <chrono>
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
    ctx.Check(opts.zhang_shu_points == "consumer",
              "Zhang-Shu defaults: consumer point set selected");
    ctx.Check(opts.hdg_igr_alpha == 1.0 && opts.hdg_igr_beta == 1.0,
              "HDG IGR defaults: alpha and beta are one");
    ctx.Check(opts.hdg_cuda_engine == "ordinary",
              "HDG CUDA engine defaults to ordinary");
    ctx.Check(dg_sim::ParseZhangShuPointSet("consumer") ==
                mfem::ZhangShuPointSet::kConsumer,
              "Zhang-Shu options: parse consumer point set");
    ctx.Check(dg_sim::ParseZhangShuPointSet("quadrature") ==
                mfem::ZhangShuPointSet::kQuadrature,
              "Zhang-Shu options: parse quadrature point set");

    opts.zhang_shu_points = "quadrature";
    const dg_sim::RunConfig run_cfg =
      dg_sim::MakeRunConfig(case_cfg, opts, igr_opts);
    ctx.Check(run_cfg.case_config.limiter_pipeline.zhang_shu_points ==
                mfem::ZhangShuPointSet::kQuadrature,
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
                std::string(dg_sim::TimeIntegratorToString(
                  dg_sim::TimeIntegrator::RK3SSP)) == "rk3ssp" &&
                std::string(dg_sim::TimeIntegratorToString(
                  dg_sim::TimeIntegrator::PartitionedBackwardEuler)) ==
                  "partitioned-backward-euler",
              "request: time-integrator parser and metadata name are stable");
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
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    igr_opts.linear_solver = "cg";
    igr_opts.preconditioner = "none";
    igr_opts.lin_rtol = 3.0e-3;
    igr_opts.lin_maxit = 777;
    igr_opts.sigma_aware_cfl = true;
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
  mfem::Mesh serial_mesh =
    mfem::Mesh::MakeCartesian2D(1,
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
    const dg_sim::TimeLoopConfig loop_cfg = dg_sim::MakeTimeLoopConfig(run_cfg);
    ctx.CheckNear(loop_cfg.dt_min,
                  run_cfg.opts.dt_min,
                  1.0e-16,
                  "time loop config: dt_min propagated from runtime options");
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
    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const { return 1.0; }
  };
  struct TimeLoopCountingStepper
  {
    int calls = 0;
    void Step(mfem::Vector&, mfem::real_t& t, mfem::real_t& dt)
    {
      ++calls;
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
    ctx.Check(stepper.calls == 0,
              "time loop: dt_min guard aborts before first step");
    ctx.Check(result.steps == 0,
              "time loop: dt_min abort leaves completed step count at zero");
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
  }

  // --- dt_min after CFL update records current step and emits final save ---
  {
    TimeLoopVariableSpeedOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 1.0;
    cfg.dt = 0.0;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.3;
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

  hooks.on_visualize(/*step=*/5, /*t=*/0.2, /*dt=*/0.01, /*done=*/false);
  hooks.on_save(/*step=*/5, /*t=*/0.2, /*save_idx=*/3, /*is_final=*/false);
  hooks.on_analysis(/*step=*/5, /*t=*/0.2, /*dt=*/0.01, /*done=*/true);

  ctx.Check(alive_count == 1, "time hooks: on_alive dispatched once");
  ctx.Check(save_count == 1, "time hooks: on_vtk_save dispatched once");
  ctx.Check(analysis_count == 1, "time hooks: on_analysis dispatched once");

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
