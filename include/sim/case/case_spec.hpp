#pragma once

/**
 * @file sim/case/case_spec.hpp
 * @brief Case specification types for the sim-layer solver.
 */

#include <functional>
#include <limits>
#include <memory>
#include <string>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/hdg/main_backend.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/time/time_callbacks.hpp"

namespace dg_sim
{

/// @brief GLVis runtime configuration.
struct GLVisConfig
{
  bool enabled = false;
  int precision = 8;
};

/// @brief Optional diagnostics fields computed for GLVis/VTK outputs.
struct DiagnosticsConfig
{
  bool h1_diagnostics =
    false; ///< H1-projected diagnostics (velocity_h1, speed_h1, div_h1, grad_frob_h1).
  bool l2_diagnostics =
    false; ///< Broken/L2 diagnostics from conservative-state gradients (div_l2, grad_frob_l2).
  bool mpi_rank_field =
    false; ///< Per-element MPI rank field for VTK debugging.
  bool zhang_shu_limiter =
    false; ///< Per-element Zhang-Shu limiter activity/theta fields for VTK debugging.
};

/// @brief VTK configuration knobs for example outputs.
struct VTKConfig
{
  bool enabled = true;
  bool print_dataset_name = false;
  double c_alpha_value = std::numeric_limits<double>::quiet_NaN();
};

/// @brief Placeholder options type for cases with no extra CLI options.
struct EmptyCaseOptions
{
};

/// @brief Case specification: metadata, callbacks, and configuration hooks.
template <typename CaseOptions, typename NumericalFlux> struct CaseSpec
{
  using Options = CaseOptions;
  using Flux = NumericalFlux;

  std::string case_name;
  std::string default_mesh_path;
  const char* case_file = nullptr;
  double gamma = mfem::dg_igr::EulerDefaults::kGamma;
  DGBackend dg_backend_default = DGBackend::DG;
  /// Default regularization when --regularization is omitted.
  bool use_igr = true;
  /// Optional sigma-space allocation when IGR coupling is disabled.
  /// The resolved regularization selector independently requests IGR storage.
  bool need_sigma = false;
  bool need_h1 = false;

  CaseOptions options;
  IGROptions igr_defaults;
  GLVisConfig glvis;
  DiagnosticsConfig diagnostics;
  VTKConfig vtk;
  DefaultCallbackConfig callback_defaults;
  std::function<TimeLoopCallbacks(const ExampleContext&, EulerStateBundle&)>
    make_callbacks;

  std::function<void(CaseConfig&)> configure_case;
  std::function<void(mfem::OptionsParser&, CaseOptions&)> add_options;
  std::function<std::unique_ptr<mfem::VectorCoefficient>(const CaseOptions&,
                                                         const ExampleContext&,
                                                         double gamma)>
    make_ic;
  std::function<BoundaryMarkerMap(ExampleContext&)> build_markers;
  std::function<void(const CaseOptions&,
                     const ExampleContext&,
                     EulerBoundaryWiring<NumericalFlux>&,
                     const BoundaryMarkerMap&)>
    apply_bc;
  /**
   * Optional physics-only HDG factory.  The generic HDG main adapter owns
   * mesh construction, field projection, boundary lowering, and time-driver
   * wiring; this callback supplies only case physics after that mesh exists.
   */
  std::function<hdg::HdgEulerCaseData(const CaseOptions&,
                                      const CaseConfig&,
                                      const CommonOptions&,
                                      const CaseContext&)>
    make_hdg_case_data;
  std::function<void(const CaseOptions&, ExampleContext&, EulerStateBundle&)>
    after_init;
  std::function<int(const CaseOptions&,
                    const ExampleContext&,
                    const EulerStateBundle&,
                    mfem::VectorCoefficient&)>
    postprocess;
};

/// @brief Resolved runtime state for a case (primarily for tooling).
template <typename CaseOptions, typename NumericalFlux> struct RunSpec
{
  CaseOptions options;
  CommonOptions common_opts;
  IGROptions igr_opts;
  RunConfig run_cfg;
  ExampleContext* ctx = nullptr;
  EulerStateBundle* state = nullptr;
  const mfem::VectorCoefficient* ic = nullptr;
};

} // namespace dg_sim
