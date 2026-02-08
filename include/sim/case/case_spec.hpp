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
#include "sim/state/euler_state_bundle.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/time/time_callbacks.hpp"

namespace dg_sim
{

/// @brief Visualization configuration for GLVis/H1 diagnostics.
struct VisualizationConfig
{
  bool enabled = false;
  bool h1_diagnostics = false;
  int precision = 8;
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
{};

/// @brief Case specification: metadata, callbacks, and configuration hooks.
template <typename CaseOptions, typename NumericalFlux>
struct CaseSpec
{
  using Options = CaseOptions;
  using Flux = NumericalFlux;

  std::string case_name;
  std::string default_mesh_path;
  const char* case_file = nullptr;
  double gamma = mfem::dg_igr::EulerDefaults::kGamma;
  DGBackend dg_backend_default = DGBackend::DG;
  bool use_igr = true;
  bool need_sigma = true;
  bool need_h1 = false;

  CaseOptions options;
  IGROptions igr_defaults;
  VisualizationConfig vis;
  VTKConfig vtk;
  DefaultCallbackConfig callback_defaults;
  std::function<TimeLoopCallbacks(const ExampleContext&, EulerStateBundle&)> make_callbacks;

  std::function<void(CaseConfig&)> configure_case;
  std::function<void(mfem::OptionsParser&, CaseOptions&)> add_options;
  std::function<std::unique_ptr<mfem::VectorCoefficient>(
      const CaseOptions&, const ExampleContext&, double gamma)> make_ic;
  std::function<BoundaryMarkerMap(ExampleContext&)> build_markers;
  std::function<void(const CaseOptions&,
                     const ExampleContext&,
                     EulerBoundaryWiring<NumericalFlux>&,
                     const BoundaryMarkerMap&)> apply_bc;
  std::function<void(const CaseOptions&, ExampleContext&, EulerStateBundle&)> after_init;
  std::function<int(const CaseOptions&,
                    const ExampleContext&,
                    const EulerStateBundle&,
                    mfem::VectorCoefficient&)> postprocess;
};

/// @brief Resolved runtime state for a case (primarily for tooling).
template <typename CaseOptions, typename NumericalFlux>
struct RunSpec
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
