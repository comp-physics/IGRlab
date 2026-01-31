#pragma once

/**
 * @file sim/igr_sim.hpp
 * @brief Umbrella include for the sim-layer (one-include rule for examples).
 */

#include "mfem.hpp"

#include "sim/case/case_spec.hpp"
#include "sim/config/case_config.hpp"
#include "sim/config/common_options.hpp"
#include "sim/context/example_context.hpp"
#include "sim/context/mesh_build.hpp"
#include "sim/io/save_state.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/solver/solver.hpp"
#include "sim/solver/solver_helpers.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/state/state_views.hpp"
#include "sim/time/time_callbacks.hpp"
#include "sim/time/time_defaults.hpp"
#include "sim/time/time_loop.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/wiring/dg_operator.hpp"
#include "sim/wiring/flux_bundle.hpp"
#include "sim/wiring/igr_physics.hpp"
