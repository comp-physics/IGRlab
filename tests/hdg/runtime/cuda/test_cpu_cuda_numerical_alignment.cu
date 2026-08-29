#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/hdg/state_io.hpp"
#include "test_cuda_trace_solver_routing.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu::detail
{

// Test-only access to the conservation gate already retained by the generic
// CPU numerics core.  No solver state or diagnostic surface is published.
struct CpuQualificationTestAccess
{
  static bool AuditLastConservation(
      const CpuNumericsCore& core, double& maximum_error,
      double& maximum_relative_error, bool& gate) noexcept
  {
    return core.AuditLastBorrowedEulerConservation(
        maximum_error, maximum_relative_error, gate);
  }
};

} // namespace dg::hdg::runtime::cpu::detail

namespace dg::hdg::runtime::cuda_w8a::detail
{

constexpr bool CudaPreparedIluFactorAccountingValid(
    const CudaIluFactorReusePolicy policy, const int trace_solves,
    const int linear_solve_attempts, const std::uint64_t setups,
    const std::uint64_t reuses,
    const std::uint64_t maximum_factor_age) noexcept
{
  if (trace_solves <= 0 || linear_solve_attempts < trace_solves)
    return false;
  const std::uint64_t solves =
      static_cast<std::uint64_t>(linear_solve_attempts);
  switch (policy)
  {
  case CudaIluFactorReusePolicy::refresh_each_trace_solve:
    return setups == solves && reuses == 0 && maximum_factor_age == 0;
  case CudaIluFactorReusePolicy::first_trace_solve_per_attempt:
    // A true-residual tightening is a same-matrix solve attempt: it reuses
    // the factor but cannot increase matrix-generation age.  Consequently
    // reuse count follows attempts while maximum age follows logical solves.
    return setups == 1 && reuses == solves - 1 &&
        maximum_factor_age ==
            static_cast<std::uint64_t>(trace_solves - 1);
  case CudaIluFactorReusePolicy::not_applicable:
    return setups == 0 && reuses == 0 && maximum_factor_age == 0;
  }
  return false;
}

static_assert(CudaPreparedIluFactorAccountingValid(
    CudaIluFactorReusePolicy::refresh_each_trace_solve,
    1, 2, 2, 0, 0));
static_assert(CudaPreparedIluFactorAccountingValid(
    CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
    1, 2, 1, 1, 0));
static_assert(CudaPreparedIluFactorAccountingValid(
    CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
    2, 3, 1, 2, 1));
static_assert(!CudaPreparedIluFactorAccountingValid(
    CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
    1, 2, 1, 1, 1));

constexpr bool CudaPreparedIluAccountingValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const int linear_iterations, const int linear_solve_attempts,
    const CudaTracePolicyDescriptor& policy) noexcept
{
  const auto preconditioner = policy.trace_preconditioner;
  const bool ilu = CudaTracePreconditionerIsIlu(preconditioner);
  const auto factor_reuse_policy = policy.ilu_factor_reuse_policy;
  const auto triangular_solve_kind = policy.ilu_triangular_solve_kind;
  const int lower_jacobi_iterations = policy.ilu_lower_jacobi_iterations;
  const int upper_jacobi_iterations = policy.ilu_upper_jacobi_iterations;
  if (!CudaTracePolicyDescriptorIsExact(policy)) return false;
  if (!ilu)
    return tensor.ilu_type == policy.ilu_type &&
        tensor.ilu_level_of_fill == policy.ilu_level_of_fill &&
        tensor.ilu_local_reordering == -1 &&
        tensor.ilu_application_iterations ==
            policy.ilu_application_iterations &&
        triangular_solve_kind ==
            CudaIluTriangularSolveKind::not_applicable &&
        tensor.ilu_triangular_solve_kind == triangular_solve_kind &&
        tensor.ilu_lower_jacobi_iterations == 0 &&
        tensor.ilu_upper_jacobi_iterations == 0 &&
        tensor.ilu_accounting_kind == policy.ilu_accounting_kind &&
        factor_reuse_policy == CudaIluFactorReusePolicy::not_applicable &&
        tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
        tensor.ilu_factor_setups == 0 &&
        tensor.ilu_factor_reuses == 0 &&
        tensor.ilu_preconditioner_callbacks_lower_bound == 0 &&
        tensor.ilu_preconditioner_callbacks_upper_bound == 0 &&
        tensor.ilu_inner_iterations_lower_bound == 0 &&
        tensor.ilu_inner_iterations_upper_bound == 0 &&
        tensor.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
        tensor.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
        tensor.ilu_factor_age_generations == 0 &&
        tensor.ilu_accounting_complete;
  if (trace_solves <= 0 || linear_iterations < 0 ||
      linear_solve_attempts < trace_solves || tensor.ilu_type != 0 ||
      tensor.ilu_level_of_fill != policy.ilu_level_of_fill ||
      tensor.ilu_local_reordering != policy.ilu_local_reordering ||
      tensor.ilu_application_iterations !=
          policy.ilu_application_iterations ||
      tensor.ilu_triangular_solve_kind != triangular_solve_kind ||
      tensor.ilu_lower_jacobi_iterations != lower_jacobi_iterations ||
      tensor.ilu_upper_jacobi_iterations != upper_jacobi_iterations ||
      (triangular_solve_kind == CudaIluTriangularSolveKind::direct &&
       (lower_jacobi_iterations != 0 || upper_jacobi_iterations != 0)) ||
      (triangular_solve_kind ==
           CudaIluTriangularSolveKind::iterative_jacobi &&
       (lower_jacobi_iterations != 5 || upper_jacobi_iterations != 5)) ||
      tensor.ilu_accounting_kind != policy.ilu_accounting_kind ||
      !tensor.ilu_accounting_complete ||
      tensor.ilu_preconditioner_callbacks_lower_bound >
          tensor.ilu_preconditioner_callbacks_upper_bound)
    return false;
  const bool factor_work =
      tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
      CudaPreparedIluFactorAccountingValid(
          factor_reuse_policy, trace_solves, linear_solve_attempts,
          tensor.ilu_factor_setups, tensor.ilu_factor_reuses,
          tensor.ilu_factor_age_generations);
  if (!factor_work) return false;
  const std::uint64_t iterations =
      static_cast<std::uint64_t>(linear_iterations);
  const std::uint64_t lower =
      tensor.ilu_preconditioner_callbacks_lower_bound;
  const std::uint64_t upper =
      tensor.ilu_preconditioner_callbacks_upper_bound;
  const bool callback_bounds = iterations == 0
      ? lower == 0 && upper == 0
      : iterations <= std::numeric_limits<std::uint64_t>::max()/2 &&
            iterations < lower && upper == 2*iterations;
  const std::uint64_t multiplier = static_cast<std::uint64_t>(
      tensor.ilu_application_iterations);
  const std::uint64_t factor_spmv_multiplier =
      triangular_solve_kind == CudaIluTriangularSolveKind::iterative_jacobi
      ? 8 : 0;
  return callback_bounds &&
      lower <= std::numeric_limits<std::uint64_t>::max()/multiplier &&
      upper <= std::numeric_limits<std::uint64_t>::max()/multiplier &&
      tensor.ilu_inner_iterations_lower_bound == lower*multiplier &&
      tensor.ilu_inner_iterations_upper_bound == upper*multiplier &&
      (factor_spmv_multiplier == 0 ||
       (tensor.ilu_inner_iterations_lower_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_multiplier &&
        tensor.ilu_inner_iterations_upper_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_multiplier)) &&
      tensor.ilu_jacobi_factor_spmv_updates_lower_bound ==
          tensor.ilu_inner_iterations_lower_bound*factor_spmv_multiplier &&
      tensor.ilu_jacobi_factor_spmv_updates_upper_bound ==
          tensor.ilu_inner_iterations_upper_bound*factor_spmv_multiplier;
}

bool CudaPreparedTracePolicyValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const int linear_iterations, const int linear_solve_attempts,
    const generic::CudaEngineKind engine, const bool igr_active,
    const CudaTracePolicyKind trace_policy) noexcept
{
  const CudaTracePolicyDescriptor expected = ResolveCudaTracePolicy(
      engine, igr_active, trace_policy);
  if (!CudaTracePolicyDescriptorIsExact(expected) || trace_solves <= 0 ||
      linear_iterations < 0 || linear_solve_attempts < trace_solves ||
      !CudaPreparedIluAccountingValid(
          tensor, trace_solves, linear_iterations, linear_solve_attempts,
          expected))
    return false;

  const std::uint64_t solves = static_cast<std::uint64_t>(trace_solves);
  const std::uint64_t attempts =
      static_cast<std::uint64_t>(linear_solve_attempts);
  const bool expected_mgr =
      trace_policy == CudaTracePolicyKind::equilibrated_mgr;
  const bool expected_matrix_free =
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(trace_policy);
  const bool expected_element_schwarz = trace_policy ==
          CudaTracePolicyKind::native_element_schwarz_fp64 ||
      expected_matrix_free || trace_policy == CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64;
  const bool expected_equilibrated_field_split = trace_policy ==
          CudaTracePolicyKind::
              native_element_schwarz_equilibrated_field_split_fp64 ||
      trace_policy == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      trace_policy == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64;
  const bool expected_native_solver =
      engine == generic::CudaEngineKind::tensor_core && !expected_mgr;
  const bool mgr_work = expected_mgr
      ? tensor.equilibration_builds == solves &&
            tensor.equilibration_passes == 2 * solves &&
            tensor.mgr_setup_calls == solves &&
            tensor.mgr_f_ilu_setups == tensor.mgr_setup_calls &&
            tensor.mgr_f_ilu_applications ==
                tensor.mgr_preconditioner_applications &&
            (linear_iterations == 0 ||
             tensor.mgr_preconditioner_applications > 0) &&
            std::isfinite(tensor.minimum_left_scale) &&
            std::isfinite(tensor.maximum_left_scale) &&
            std::isfinite(tensor.minimum_right_scale) &&
            std::isfinite(tensor.maximum_right_scale) &&
            tensor.minimum_left_scale > 0.0 &&
            tensor.minimum_left_scale <= tensor.maximum_left_scale &&
            tensor.minimum_right_scale > 0.0 &&
            tensor.minimum_right_scale <= tensor.maximum_right_scale
      : expected_equilibrated_field_split
          ? tensor.equilibration_builds == solves &&
                tensor.equilibration_passes ==
                    native_element_schwarz_equilibration_passes * solves &&
                tensor.mgr_setup_calls == 0 &&
                tensor.mgr_preconditioner_applications == 0 &&
                tensor.mgr_f_ilu_setups == 0 &&
                tensor.mgr_f_ilu_applications == 0
          : tensor.equilibration_builds == 0 &&
                tensor.equilibration_passes == 0 &&
                tensor.mgr_setup_calls == 0 &&
                tensor.mgr_preconditioner_applications == 0 &&
                tensor.mgr_f_ilu_setups == 0 &&
                tensor.mgr_f_ilu_applications == 0;
  const bool element_schwarz_work = expected_element_schwarz
      ? tensor.native_element_schwarz_configured &&
            tensor.element_schwarz_patch_setups == solves &&
            (linear_iterations == 0 ||
             tensor.element_schwarz_preconditioner_applications > 0) &&
            (tensor.element_schwarz_patch_rows == 40 ||
             tensor.element_schwarz_patch_rows == 60 ||
             tensor.element_schwarz_patch_rows == 80) &&
            tensor.element_schwarz_minimum_multiplicity >= 1 &&
            tensor.element_schwarz_maximum_multiplicity >=
                tensor.element_schwarz_minimum_multiplicity &&
            tensor.element_schwarz_maximum_multiplicity <= 2
      : !tensor.native_element_schwarz_configured &&
            tensor.element_schwarz_patch_setups == 0 &&
            tensor.element_schwarz_preconditioner_applications == 0 &&
            tensor.element_schwarz_patch_rows == 0 &&
            tensor.element_schwarz_minimum_multiplicity == 0 &&
            tensor.element_schwarz_maximum_multiplicity == 0;
  const bool matrix_free_work = expected_matrix_free
      ? tensor.native_matrix_free_trace_operator_configured &&
            tensor.native_matrix_free_trace_operator_applications >=
                static_cast<std::uint64_t>(linear_iterations) &&
            ((linear_iterations == 0) ==
             (tensor.native_matrix_free_trace_operator_applications == 0))
      : !tensor.native_matrix_free_trace_operator_configured &&
            tensor.native_matrix_free_trace_operator_applications == 0;
  return tensor.selected_engine == engine &&
      tensor.trace_policy == trace_policy &&
      tensor.trace_preconditioner_configured &&
      tensor.trace_preconditioner == expected.trace_preconditioner &&
      mgr_work && element_schwarz_work && matrix_free_work &&
      tensor.trace_solver_accounting_complete &&
      tensor.native_rank1_trace_solver_configured == expected_native_solver &&
      tensor.hypre_mgr_rank1_trace_solver_configured == expected_mgr &&
      !tensor.hypre_host_staged_rank2_trace_solver_configured &&
      tensor.native_rank1_trace_solve_calls ==
          (expected_native_solver ? attempts : 0) &&
      tensor.hypre_trace_solve_calls ==
          (expected_native_solver ? 0 : attempts) &&
      tensor.fallback_accounting_complete;
}

struct CrossBackendPreparedHealth
{
  double initial_residual = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::infinity();
  int nonlinear_updates = -1;
  int trace_solves = -1;
  int linear_iterations = -1;
  int linear_solve_attempts = -1;
  int true_residual_tightenings = -1;
  int line_search_trials = -1;
  int rejected_line_search_trials = -1;
  CudaTensorCoreEvidence tensor_core{};
  bool tensor_volume_evidence_valid = false;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool no_fallback = false;
  bool success = false;
};

// This accessor exists only in this qualification executable.  It observes
// the already-computed prepared gates and copies the logical committed true
// trace to host storage; it launches no numerical work and publishes no API.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool CapturePreparedHealth(
      Engine& engine, CrossBackendPreparedHealth& health) noexcept
  {
    health = {};
    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    health.initial_residual = numerical.initial_residual;
    health.final_residual = numerical.final_residual;
    health.true_relative_residual = numerical.true_relative_residual;
    health.maximum_conservation_error =
        numerical.maximum_conservation_error;
    health.maximum_conservation_relative_error =
        numerical.maximum_conservation_relative_error;
    health.nonlinear_updates = numerical.nonlinear_updates;
    health.trace_solves = numerical.trace_solves;
    health.linear_iterations = numerical.linear_iterations;
    health.linear_solve_attempts = numerical.linear_solve_attempts;
    health.true_residual_tightenings =
        numerical.true_residual_tightenings;
    health.line_search_trials = numerical.line_search_trials;
    health.rejected_line_search_trials =
        numerical.rejected_line_search_trials;
    health.tensor_core = pending.evidence.tensor_core;
    health.tensor_volume_evidence_valid =
        CudaPreparedTensorVolumeEvidenceValid(
            health.tensor_core, engine.numerics_.shape_,
            engine.config_.cuda_engine);
    health.finite = numerical.finite;
    health.admissible = numerical.admissible;
    health.recovery = numerical.recovery;
    health.true_residual = numerical.true_residual;
    health.conservation = numerical.conservation;
    health.no_fallback = pending.evidence.no_fallback;
    health.success = numerical.success;
    return engine.initialized_ && pending.active && pending.staging &&
        pending.prestage_allowed && numerical.candidate_ready;
  }

  static bool CapturePreparedHealth(
      cuda::GenericEulerHdgCudaSession& session,
      CrossBackendPreparedHealth& health) noexcept
  {
    return session.borrowed_rank1_ &&
        CapturePreparedHealth(*session.borrowed_rank1_, health);
  }

  static dg::hdg::generic::HdgEulerSessionResult InitializeTracePolicy(
      cuda::GenericEulerHdgCudaSession& session,
      const dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind trace_policy) noexcept
  {
    return session.InitializeRequest(request, trace_policy);
  }

  template <typename Engine>
  static bool ExportCommittedTrace(
      Engine& engine, std::vector<double>& trace,
      int& trace_components) noexcept
  {
    trace.clear();
    trace_components = 0;
    auto& numerics = engine.numerics_;
    const auto committed = numerics.CommittedSlot();
    bool ready = engine.initialized_ && !engine.pending_.active &&
        committed.trace.coefficients != nullptr &&
        committed.trace.count != 0 && numerics.shape_.trace_components > 0;
    try
    {
      if (ready) trace.resize(committed.trace.count);
    }
    catch (...) { ready = false; }
    if (!numerics.CollectiveContinue(ready)) return false;
    const bool copied = CudaTransportPolicy::DeviceToHost(
            trace.data(), committed.trace.coefficients,
            trace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(copied)) return false;
    trace_components = numerics.shape_.trace_components;
    return true;
  }

  static bool ExportCommittedTrace(
      cuda::GenericEulerHdgCudaSession& session,
      std::vector<double>& trace, int& trace_components) noexcept
  {
    return session.borrowed_rank1_ && ExportCommittedTrace(
        *session.borrowed_rank1_, trace, trace_components);
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerBoundaryPolicy;
using dg::hdg::generic::EulerBoundaryPolicyKind;
using dg::hdg::generic::EulerBoundaryPolicyTableView;
using dg::hdg::generic::EulerHdgAssemblyQuadratureOrders;
using dg::hdg::generic::EulerHdgNumericalPolicy;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::StateCoordinates;
using dg::hdg::runtime::cpu::CpuAttemptStatus;
using dg::hdg::runtime::cpu::detail::CpuNumericsCore;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::detail::CrossBackendPreparedHealth;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKindName;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr double comparison_relative_tolerance = 5.0e-8;
constexpr double comparison_absolute_tolerance = 5.0e-9;
constexpr double sigma_relative_tolerance = 5.0e-6;
constexpr double sigma_absolute_tolerance = 8192.0 *
    std::numeric_limits<double>::epsilon();
constexpr double igr_transition_x = 0.4;
constexpr double igr_transition_width = 0.06;
constexpr double igr_localization_half_width = 0.25;
constexpr double igr_minimum_near_far_energy_ratio = 1.5;
constexpr double igr_minimum_near_energy_fraction = 0.60;

struct AlignmentSelection
{
  int order = 1;
  EulerHdgAssemblyQuadratureOrders igr_assembly_quadrature_orders{0, 0};
  // Zero retains DefaultEulerHdgNumericalPolicy.  Parsing admits only finite
  // positive overrides and the complete policy validator enforces ordering.
  double newton_correction_linear_relative_tolerance = 0.0;
  double final_coherent_linear_relative_tolerance = 0.0;
  double true_residual_tightening_linear_relative_tolerance = 0.0;
  // This private selector is applied only to tensor-IGR cases.  Conventional
  // tensor cases and both controls retain engine-default trace policy.
  CudaTracePolicyKind tensor_igr_trace_policy =
      CudaTracePolicyKind::engine_default;
  bool validate_options_only = false;
};

constexpr std::uint32_t common_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

enum class InitialMode
{
  smooth_periodic,
  localized_igr_contact
};

struct Case
{
  std::string_view name;
  InitialMode initial_mode = InitialMode::smooth_periodic;
  RegularizationKind regularization = RegularizationKind::none;
  bool physical = false;
  int steps = 1;
  double dt = 1.0e-5;
  bool require_localized_sigma = false;
  double static_source_scale = 0.0;
  bool supply_initial_true_trace = false;
};

struct BoundaryContext
{
  std::array<EulerBoundaryPolicy, 4> policies{};
  std::uint64_t calls = 0;
};

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> initial_true_trace;
  std::vector<double> source;
  std::size_t scalar_count = 0;
};

struct ComparisonMetrics
{
  double relative_l2 = std::numeric_limits<double>::infinity();
  double maximum_absolute = std::numeric_limits<double>::infinity();
  double scale = 1.0;
  bool passed = false;
};

struct SigmaLocalization
{
  double global_l2_squared = 0.0;
  double global_maximum_absolute = 0.0;
  double near_mean_square = 0.0;
  double far_mean_square = 0.0;
  double near_energy_fraction = 0.0;
  bool available = false;
  bool finite = false;
  bool nonzero = false;
  bool localized = false;
};

struct StepState
{
  std::vector<double> physical_interior;
  std::vector<double> true_trace;
  double time = 0.0;
  std::uint64_t step = 0;
  std::uint64_t generation = 0;
  double maximum_wave_speed = 0.0;
  double backend_stable_dt = 0.0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double true_relative_residual = 0.0;
  double maximum_conservation_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  int nonlinear_updates = 0;
  int trace_solves = 0;
  int linear_iterations = 0;
  int line_search_trials = 0;
  int rejected_line_search_trials = 0;
  std::uint64_t tensor_condensation_launches = 0;
  std::uint64_t tensor_mma_tiles = 0;
  std::uint64_t tensor_volume_assembly_launches = 0;
  std::uint64_t tensor_volume_mma_tiles = 0;
  std::uint64_t fallback_count = 0;
  int trace_components = 0;
  SigmaLocalization sigma{};
};

struct Trajectory
{
  std::vector<StepState> states;
  bool passed = false;
};

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && output == 1;
}

mfem::ParMesh MakeMesh(const bool physical)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  if (physical) return mfem::ParMesh(MPI_COMM_WORLD, serial);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

void EvaluateInitial(const InitialMode mode, const mfem::Vector& point,
                     mfem::Vector& state)
{
  state.SetSize(components);
  if (mode == InitialMode::localized_igr_contact)
  {
    // Keep the density contact discontinuous, but move it off the 4x4 mesh
    // lines and superpose a small resolved normal-velocity transition.  The
    // old mesh-aligned zero-velocity contact had identically zero IGR q
    // forcing and therefore could not qualify localized regularization.
    const double density = point[0] < igr_transition_x ? 1.0 : 1.2;
    const double velocity_x = 0.02 * std::tanh(
        (point[0] - igr_transition_x) / igr_transition_width);
    constexpr double pressure = 1.0;
    state[0] = density;
    state[1] = density * velocity_x;
    state[2] = 0.0;
    state[3] = pressure / 0.4 +
        0.5 * density * velocity_x * velocity_x;
    return;
  }
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi * point[0]);
  const double sy = std::sin(2.0 * pi * point[1]);
  const double cx = std::cos(2.0 * pi * point[0]);
  const double cy = std::cos(2.0 * pi * point[1]);
  const double density = 1.0 + 0.04 * sx * sy;
  const double velocity_x = 0.18 + 0.02 * cx * sy;
  const double velocity_y = -0.09 + 0.015 * sx * cy;
  const double pressure = 1.0 + 0.03 * cx * cy;
  state[0] = density;
  state[1] = density * velocity_x;
  state[2] = density * velocity_y;
  state[3] = pressure / 0.4 +
      0.5 * density *
          (velocity_x * velocity_x + velocity_y * velocity_y);
}

void EvaluateSuppliedTrace(const InitialMode mode, const mfem::Vector& point,
                           mfem::Vector& state)
{
  EvaluateInitial(mode, point, state);
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double base_density = state[0];
  const double base_velocity_x = state[1] / base_density;
  const double base_velocity_y = state[2] / base_density;
  const double base_pressure = 0.4 *
      (state[3] - 0.5 * base_density *
          (base_velocity_x * base_velocity_x +
           base_velocity_y * base_velocity_y));
  const double density = base_density *
      (1.0 + 2.0e-4 * std::cos(2.0 * pi * point[0]));
  const double velocity_x = base_velocity_x +
      1.0e-4 * std::sin(2.0 * pi * point[1]);
  const double velocity_y = base_velocity_y -
      1.0e-4 * std::cos(2.0 * pi * (point[0] + point[1]));
  const double pressure = base_pressure *
      (1.0 + 1.5e-4 * std::sin(2.0 * pi * point[0]));
  state[0] = density;
  state[1] = density * velocity_x;
  state[2] = density * velocity_y;
  state[3] = pressure / 0.4 + 0.5 * density *
      (velocity_x * velocity_x + velocity_y * velocity_y);
}

BorrowedFields MakeFields(mfem::ParMesh& mesh, const InitialMode mode,
                          const int order, const double source_scale,
                          const bool supply_initial_true_trace)
{
  BorrowedFields fields;
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction projected(&space);
  const auto evaluator = [mode](const mfem::Vector& point,
                                mfem::Vector& state)
  { EvaluateInitial(mode, point, state); };
  mfem::VectorFunctionCoefficient coefficient(components, evaluator);
  projected.ProjectCoefficient(coefficient);
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() / components);
  const double* values = projected.HostRead();
  if (values != nullptr && projected.Size() > 0)
    fields.initial.assign(values, values + projected.Size());
  if (source_scale != 0.0 && !fields.initial.empty())
  {
    fields.source.resize(fields.initial.size(), 0.0);
    constexpr std::array<double, components> source_components{
        1.0e-3, 5.0e-4, -2.5e-4, 1.5e-3};
    for (int component = 0; component < components; ++component)
      for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
        fields.source[static_cast<std::size_t>(component) *
                          fields.scalar_count + scalar] =
            source_scale * source_components[component];
  }
  if (supply_initial_true_trace)
  {
    mfem::DG_Interface_FECollection trace_collection(
        order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, components, mfem::Ordering::byNODES);
    mfem::ParGridFunction local_trace(&trace_space);
    const auto trace_evaluator = [mode](const mfem::Vector& point,
                                        mfem::Vector& state)
    { EvaluateSuppliedTrace(mode, point, state); };
    mfem::VectorFunctionCoefficient trace_coefficient(
        components, trace_evaluator);
    mfem::Vector true_trace;
    const auto policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
    if (!dg_sim::hdg::ProjectEulerTrueTraceCoefficientL2(
            trace_space, local_trace, true_trace, trace_coefficient,
            policy.true_trace_projection_quadrature_order) ||
        true_trace.Size() <= 0)
      return {};
    const double* const trace_values = true_trace.HostRead();
    if (trace_values == nullptr) return {};
    fields.initial_true_trace.assign(
        trace_values, trace_values + true_trace.Size());

    // Independently project the volume initial-state function to the trace
    // space and prove that the supplied coefficients are not merely the
    // compatibility projection under another pointer.
    mfem::VectorFunctionCoefficient projected_coefficient(
        components, evaluator);
    mfem::Vector projected_true_trace;
    if (!dg_sim::hdg::ProjectEulerTrueTraceCoefficientL2(
            trace_space, local_trace, projected_true_trace,
            projected_coefficient,
            policy.true_trace_projection_quadrature_order) ||
        projected_true_trace.Size() != true_trace.Size())
      return {};
    const double* const projected_values = projected_true_trace.HostRead();
    if (projected_values == nullptr || std::equal(
            fields.initial_true_trace.begin(),
            fields.initial_true_trace.end(), projected_values))
      return {};
  }
  return fields;
}

bool EulerExterior(const void* opaque,
                   const dg::hdg::generic::BoundaryPointView& point,
                   const double* interior, double* exterior) noexcept
{
  auto* context = static_cast<BoundaryContext*>(const_cast<void*>(opaque));
  if (context == nullptr || point.position == nullptr ||
      point.outward_normal == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 4 ||
      interior == nullptr || exterior == nullptr)
    return false;
  ++context->calls;
  std::copy_n(interior, components, exterior);
  const double normal_momentum =
      interior[1] * point.outward_normal[0] +
      interior[2] * point.outward_normal[1];
  exterior[1] -= 2.0 * normal_momentum * point.outward_normal[0];
  exterior[2] -= 2.0 * normal_momentum * point.outward_normal[1];
  return true;
}

void ConfigureBoundary(BoundaryContext& boundary)
{
  const std::array<double, components> unused{};
  for (EulerBoundaryPolicy& policy : boundary.policies)
    policy = {EulerBoundaryPolicyKind::slip_wall, unused, 0.0};
}

EulerHdgNumericalPolicy SelectedNumericalPolicy(
    const AlignmentSelection& selection,
    const RegularizationKind regularization) noexcept
{
  EulerHdgNumericalPolicy policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(selection.order);
  if (selection.newton_correction_linear_relative_tolerance > 0.0)
    policy.newton_correction_linear_relative_tolerance =
        selection.newton_correction_linear_relative_tolerance;
  if (selection.final_coherent_linear_relative_tolerance > 0.0)
    policy.final_coherent_linear_relative_tolerance =
        selection.final_coherent_linear_relative_tolerance;
  if (selection.true_residual_tightening_linear_relative_tolerance > 0.0)
    policy.true_residual_tightening_linear_relative_tolerance =
        selection.true_residual_tightening_linear_relative_tolerance;
  if (regularization == RegularizationKind::igr &&
      selection.igr_assembly_quadrature_orders.volume_order > 0)
    policy.igr_assembly_quadrature_orders =
        selection.igr_assembly_quadrature_orders;
  policy.stabilization = regularization == RegularizationKind::igr
      ? dg::hdg::generic::RegularizationSelection{
            RegularizationKind::igr, 0.2, 0.8}
      : dg::hdg::generic::RegularizationSelection{
            RegularizationKind::none, 0.0, 0.0};
  return policy;
}

CudaTracePolicyKind SelectedCudaTracePolicy(
    const AlignmentSelection& selection,
    const CudaEngineKind engine,
    const RegularizationKind regularization) noexcept
{
  return engine == CudaEngineKind::tensor_core &&
          regularization == RegularizationKind::igr
      ? selection.tensor_igr_trace_policy
      : CudaTracePolicyKind::engine_default;
}

EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const BorrowedFields& fields,
    BoundaryContext& boundary, const Case& specification,
    const BackendKind backend, const AlignmentSelection& selection,
    const CudaEngineKind cuda_engine)
{
  ConfigureBoundary(boundary);
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = selection.order;
  request.backend = {backend, backend == BackendKind::cuda ? 0 : -1,
                     backend == BackendKind::cuda
                         ? cuda_engine : CudaEngineKind::ordinary};
  request.capabilities = common_capabilities | CapabilityBit(
      backend == BackendKind::cuda ? Capability::cuda : Capability::cpu);
  request.initial_state = {
      fields.initial.data(), fields.initial.size(), components};
  request.initial_true_trace = {
      fields.initial_true_trace.empty()
          ? nullptr : fields.initial_true_trace.data(),
      fields.initial_true_trace.size(), components};
  request.source = {
      fields.source.empty() ? nullptr : fields.source.data(),
      fields.source.size(), components};
  request.boundary.context = &boundary;
  request.boundary.exterior_state = EulerExterior;
  request.boundary.policies = specification.physical
      ? EulerBoundaryPolicyTableView{
            boundary.policies.data(), boundary.policies.size(), 2, 1.4}
      : EulerBoundaryPolicyTableView{nullptr, 0, 2, 1.4};
  request.numerical_policy = SelectedNumericalPolicy(
      selection, specification.regularization);
  return request;
}

ComparisonMetrics Compare(const std::vector<double>& left,
                          const std::vector<double>& right)
{
  ComparisonMetrics metrics;
  if (left.size() != right.size() || left.empty()) return metrics;
  long double difference_squared = 0.0L;
  long double reference_squared = 0.0L;
  metrics.maximum_absolute = 0.0;
  metrics.scale = 1.0;
  for (std::size_t index = 0; index < left.size(); ++index)
  {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index]))
      return metrics;
    const double difference = left[index] - right[index];
    difference_squared += static_cast<long double>(difference) * difference;
    reference_squared += std::max(
        static_cast<long double>(left[index]) * left[index],
        static_cast<long double>(right[index]) * right[index]);
    metrics.maximum_absolute = std::max(
        metrics.maximum_absolute, std::abs(difference));
    metrics.scale = std::max(
        metrics.scale, std::max(std::abs(left[index]),
                                std::abs(right[index])));
  }
  metrics.relative_l2 = static_cast<double>(
      std::sqrt(difference_squared) /
      std::max(1.0L, std::sqrt(reference_squared)));
  metrics.passed =
      metrics.relative_l2 <= comparison_relative_tolerance &&
      metrics.maximum_absolute <= comparison_absolute_tolerance +
          comparison_relative_tolerance * metrics.scale;
  return metrics;
}

bool PhysicalCoefficients(const std::vector<double>& values,
                          const std::size_t scalar_count)
{
  if (values.size() != scalar_count * components || scalar_count == 0)
    return false;
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
  {
    const double density = values[scalar];
    const double momentum_x = values[scalar_count + scalar];
    const double momentum_y = values[2 * scalar_count + scalar];
    const double energy = values[3 * scalar_count + scalar];
    const double pressure = 0.4 *
        (energy - 0.5 *
            (momentum_x * momentum_x + momentum_y * momentum_y) /
                density);
    if (!(density > 1.0e-8) || !(pressure > 1.0e-8) ||
        !std::isfinite(density) || !std::isfinite(pressure))
      return false;
  }
  return true;
}

bool ComputeSigmaLocalization(mfem::ParMesh& mesh, const int order,
                              const std::vector<double>& sigma,
                              SigmaLocalization& localization)
{
  localization = {};
  std::array<double, 5> local{};
  double local_maximum = 0.0;
  const bool local_ready = [&]() noexcept
  {
    try
    {
      mfem::L2_FECollection collection(
          order, 2, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      mfem::ParFiniteElementSpace space(
          &mesh, &collection, 1,
          mfem::Ordering::byNODES);
      if (space.GetVSize() < 0 ||
          sigma.size() != static_cast<std::size_t>(space.GetVSize()))
        return false;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
        const mfem::FiniteElement* finite_element = space.GetFE(element);
        mfem::ElementTransformation* transformation =
            mesh.GetElementTransformation(element);
        mfem::Array<int> vdofs;
        space.GetElementVDofs(element, vdofs);
        if (finite_element == nullptr || transformation == nullptr ||
            finite_element->GetDof() <= 0 ||
            vdofs.Size() != finite_element->GetDof())
          return false;
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), 2 * order + 4);
        mfem::Vector shape(finite_element->GetDof());
        mfem::Vector physical(mesh.SpaceDimension());
        for (int quadrature = 0; quadrature < rule.GetNPoints(); ++quadrature)
        {
          const mfem::IntegrationPoint& point = rule.IntPoint(quadrature);
          finite_element->CalcShape(point, shape);
          transformation->SetIntPoint(&point);
          transformation->Transform(point, physical);
          const double weight = point.weight * transformation->Weight();
          if (!(weight > 0.0) || !std::isfinite(weight) ||
              physical.Size() < 1 || !std::isfinite(physical[0]))
            return false;
          double value = 0.0;
          for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
          {
            double sign = 1.0;
            int dof = vdofs[scalar];
            if (dof < 0)
            {
              dof = -1 - dof;
              sign = -1.0;
            }
            if (dof < 0 || dof >= space.GetVSize() ||
                !std::isfinite(sigma[static_cast<std::size_t>(dof)]))
              return false;
            value += sign * shape[scalar] *
                sigma[static_cast<std::size_t>(dof)];
          }
          if (!std::isfinite(value)) return false;
          const double energy = weight * value * value;
          const bool near = std::abs(physical[0] - igr_transition_x) <=
              igr_localization_half_width;
          local[0] += energy;
          local[near ? 1 : 2] += energy;
          local[near ? 3 : 4] += weight;
          local_maximum = std::max(
              local_maximum, static_cast<double>(std::abs(value)));
        }
      }
      return std::all_of(local.begin(), local.end(),
                         [](const double value)
                         { return std::isfinite(value) && value >= 0.0; }) &&
          std::isfinite(local_maximum);
    }
    catch (...) { return false; }
  }();
  if (!All(local_ready, mesh.GetComm())) return false;

  std::array<double, 5> global{};
  double global_maximum = 0.0;
  const bool energy_reduced = MPI_Allreduce(
      local.data(), global.data(), static_cast<int>(global.size()),
      MPI_DOUBLE, MPI_SUM, mesh.GetComm()) == MPI_SUCCESS;
  const bool maximum_reduced = MPI_Allreduce(
      &local_maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX,
      mesh.GetComm()) == MPI_SUCCESS;
  if (!All(energy_reduced && maximum_reduced, mesh.GetComm())) return false;
  const double total_measure = global[3] + global[4];
  if (!(total_measure > 0.0) || !(global[3] > 0.0) ||
      !(global[4] > 0.0) || !std::isfinite(total_measure) ||
      !std::isfinite(global_maximum))
    return false;

  const double near_mean_square = global[1] / global[3];
  const double far_mean_square = global[2] / global[4];
  const double near_energy_fraction = global[0] > 0.0
      ? global[1] / global[0] : 0.0;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double energy_floor =
      4096.0 * epsilon * epsilon * total_measure;
  localization.global_l2_squared = global[0];
  localization.global_maximum_absolute = global_maximum;
  localization.near_mean_square = near_mean_square;
  localization.far_mean_square = far_mean_square;
  localization.near_energy_fraction = near_energy_fraction;
  localization.available = true;
  localization.finite = std::isfinite(localization.global_l2_squared) &&
      std::isfinite(localization.global_maximum_absolute) &&
      std::isfinite(localization.near_mean_square) &&
      std::isfinite(localization.far_mean_square) &&
      std::isfinite(localization.near_energy_fraction);
  localization.nonzero = localization.finite && global[0] > energy_floor &&
      global_maximum > 64.0 * std::numeric_limits<double>::epsilon();
  const double near_far_ratio = far_mean_square > 0.0
      ? near_mean_square / far_mean_square
      : std::numeric_limits<double>::infinity();
  // The p1-p3 control observations all had ratios >= 1.82 and near-energy
  // fractions >= 0.645.  Retain diagnosed margin while requiring meaningful
  // concentration instead of a numerical inequality at roundoff scale.
  localization.localized = localization.nonzero &&
      near_far_ratio >= igr_minimum_near_far_energy_ratio &&
      near_energy_fraction >= igr_minimum_near_energy_fraction;
  return localization.finite;
}

bool HasIgrContactAndVelocityTransition(const BorrowedFields& fields)
{
  if (fields.scalar_count == 0 ||
      fields.initial.size() != fields.scalar_count * components)
    return false;
  const auto density_bounds = std::minmax_element(
      fields.initial.begin(),
      fields.initial.begin() + static_cast<std::ptrdiff_t>(
          fields.scalar_count));
  double minimum_velocity = std::numeric_limits<double>::infinity();
  double maximum_velocity = -std::numeric_limits<double>::infinity();
  for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
  {
    const double density = fields.initial[scalar];
    const double velocity =
        fields.initial[fields.scalar_count + scalar] / density;
    minimum_velocity = std::min(minimum_velocity, velocity);
    maximum_velocity = std::max(maximum_velocity, velocity);
  }
  return *density_bounds.second - *density_bounds.first >= 0.19 &&
      std::isfinite(minimum_velocity) && std::isfinite(maximum_velocity) &&
      maximum_velocity - minimum_velocity >= 0.02;
}

bool CaptureCpuState(CpuNumericsCore& core,
                     const StateCoordinates& coordinates,
                     mfem::ParMesh& mesh, const int order,
                     const std::size_t scalar_count,
                     const RegularizationKind regularization,
                     StepState& state)
{
  state.physical_interior.assign(scalar_count * components, 0.0);
  std::vector<double> sigma(
      regularization == RegularizationKind::igr ? scalar_count : 0, 0.0);
  const auto snapshot = core.Snapshot(coordinates);
  if (!core.ExportCommittedFields({
          state.physical_interior.data(),
          sigma.empty() ? nullptr : sigma.data(), scalar_count}) ||
      snapshot.trace.data == nullptr || snapshot.trace.size == 0)
    return false;
  state.true_trace.assign(
      snapshot.trace.data, snapshot.trace.data + snapshot.trace.size);
  state.time = coordinates.time;
  state.step = coordinates.step;
  state.generation = coordinates.generation;
  return core.CommittedBorrowedEulerDiagnostics(
             state.maximum_wave_speed, state.backend_stable_dt) &&
      PhysicalCoefficients(state.physical_interior, scalar_count) &&
      (regularization != RegularizationKind::igr ||
       ComputeSigmaLocalization(mesh, order, sigma, state.sigma));
}

Trajectory RunCpu(const EulerHdgRequest& request,
                  mfem::ParMesh& mesh, const BorrowedFields& fields,
                  const Case& specification)
{
  Trajectory trajectory;
  CpuNumericsCore core;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
          ContractStatus::accepted ||
      !core.InitializeBorrowedEuler(request))
    return trajectory;
  StateCoordinates coordinates{};
  StepState initial;
  if (!CaptureCpuState(core, coordinates, mesh, request.polynomial_order,
                       fields.scalar_count, specification.regularization,
                       initial))
    return trajectory;
  initial.trace_components = specification.regularization ==
          RegularizationKind::igr ? 5 : components;
  if (initial.true_trace.size() %
          static_cast<std::size_t>(initial.trace_components) != 0)
    return trajectory;
  trajectory.states.push_back(std::move(initial));

  for (int step = 0; step < specification.steps; ++step)
  {
    if (!core.AdvanceBorrowedEuler(specification.dt, coordinates))
      return trajectory;
    const auto& attempt = core.evidence().last_attempt;
    bool conservation_gate = true;
    double conservation_error = 0.0;
    double conservation_relative_error = 0.0;
    const bool conservation_ready =
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditLastConservation(
                core, conservation_error, conservation_relative_error,
                conservation_gate);
    const bool health = conservation_ready && conservation_gate &&
        attempt.status == CpuAttemptStatus::success && attempt.committed &&
        attempt.finite_gate && attempt.admissibility_gate &&
        attempt.recovery_gate && attempt.nonlinear_gate &&
        attempt.true_residual_gate &&
        attempt.nonlinear_updates <=
            request.numerical_policy.maximum_nonlinear_updates &&
        std::isfinite(attempt.initial_nonlinear_residual) &&
        std::isfinite(attempt.final_nonlinear_residual) &&
        attempt.final_nonlinear_residual <=
            request.numerical_policy.nonlinear_relative_tolerance *
                (1.0 + attempt.initial_nonlinear_residual) &&
        attempt.maximum_true_relative_residual <=
            request.numerical_policy.true_residual_relative_tolerance &&
        conservation_relative_error <=
            request.numerical_policy.conservation_relative_tolerance;
    StepState state;
    if (!health ||
        !CaptureCpuState(core, coordinates, mesh, request.polynomial_order,
                         fields.scalar_count, specification.regularization,
                         state))
      return trajectory;
    state.initial_residual = attempt.initial_nonlinear_residual;
    state.final_residual = attempt.final_nonlinear_residual;
    state.true_relative_residual = attempt.maximum_true_relative_residual;
    state.maximum_conservation_error = conservation_error;
    state.maximum_conservation_relative_error =
        conservation_relative_error;
    state.nonlinear_updates = attempt.nonlinear_updates;
    state.trace_solves = static_cast<int>(
        attempt.work.complete_trace_solves);
    state.linear_iterations = attempt.linear_iterations;
    state.line_search_trials = static_cast<int>(
        attempt.work.trial_reassemblies);
    state.rejected_line_search_trials = static_cast<int>(
        attempt.work.trial_reassemblies >=
                attempt.work.accepted_newton_updates
            ? attempt.work.trial_reassemblies -
                  attempt.work.accepted_newton_updates
            : 0);
    state.trace_components = specification.regularization ==
            RegularizationKind::igr ? 5 : components;
    if (state.true_trace.size() %
            static_cast<std::size_t>(state.trace_components) != 0)
      return trajectory;
    trajectory.states.push_back(std::move(state));
  }
  trajectory.passed = trajectory.states.size() ==
      static_cast<std::size_t>(specification.steps + 1);
  return trajectory;
}

bool CaptureCudaState(GenericEulerHdgCudaSession& session,
                      mfem::ParMesh& mesh, const int order,
                      const std::size_t scalar_count,
                      const RegularizationKind regularization,
                      StepState& state)
{
  state.physical_interior.assign(scalar_count * components, 0.0);
  std::vector<double> sigma(
      regularization == RegularizationKind::igr ? scalar_count : 0, 0.0);
  if (!session.ExportCommittedFields({
          state.physical_interior.data(),
          sigma.empty() ? nullptr : sigma.data(), scalar_count}) ||
      !CudaQualificationTestAccess::ExportCommittedTrace(
          session, state.true_trace, state.trace_components) ||
      !PhysicalCoefficients(state.physical_interior, scalar_count) ||
      (regularization == RegularizationKind::igr &&
       !ComputeSigmaLocalization(mesh, order, sigma, state.sigma)))
    return false;
  const auto coordinates = session.coordinates();
  state.time = coordinates.time;
  state.step = coordinates.step;
  state.generation = coordinates.generation;
  const auto diagnostics = session.diagnostics();
  if (!diagnostics.available) return false;
  state.maximum_wave_speed = diagnostics.maximum_wave_speed;
  state.backend_stable_dt = diagnostics.backend_stable_dt;
  return true;
}

Trajectory RunCuda(const EulerHdgRequest& request,
                   mfem::ParMesh& mesh, const BorrowedFields& fields,
                   const Case& specification,
                   const CudaTracePolicyKind trace_policy)
{
  Trajectory trajectory;
  GenericEulerHdgCudaSession session;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
          ContractStatus::accepted ||
      CudaQualificationTestAccess::InitializeTracePolicy(
          session, request, trace_policy).status !=
          dg::hdg::generic::HdgEulerSessionStatus::initialized)
    return trajectory;
  StepState initial;
  if (!CaptureCudaState(session, mesh, request.polynomial_order,
                        fields.scalar_count, specification.regularization,
                        initial))
    return trajectory;
  trajectory.states.push_back(std::move(initial));

  for (int step = 0; step < specification.steps; ++step)
  {
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    if (session.Prepare(specification.dt, authority, output).status !=
        dg::hdg::generic::HdgEulerSessionStatus::prepared)
      return trajectory;
    CrossBackendPreparedHealth health;
    const bool captured =
        CudaQualificationTestAccess::CapturePreparedHealth(session, health);
    const bool expected_engine =
        health.tensor_core.selected_engine == request.backend.cuda_engine &&
        dg::hdg::runtime::cuda_w8a::detail::
            CudaPreparedTracePolicyValid(
                health.tensor_core, health.trace_solves,
                health.linear_iterations, health.linear_solve_attempts,
                request.backend.cuda_engine,
                specification.regularization == RegularizationKind::igr,
                trace_policy) &&
        health.tensor_core.fallback_count == 0 && health.no_fallback &&
        health.tensor_volume_evidence_valid &&
        (request.backend.cuda_engine == CudaEngineKind::ordinary
             ? health.tensor_core.condensation_launches == 0 &&
                   health.tensor_core.mma_tiles == 0
             : health.tensor_core.hardware_supported &&
                   health.tensor_core.condensation_launches > 0 &&
                   health.tensor_core.mma_tiles > 0);
    const bool healthy = captured && health.success && health.finite &&
        health.admissible && health.recovery && health.true_residual &&
        health.conservation && expected_engine &&
        health.nonlinear_updates >= 0 &&
        health.nonlinear_updates <=
            request.numerical_policy.maximum_nonlinear_updates &&
        health.trace_solves >= 1 &&
        health.linear_iterations >= 0 &&
        health.linear_solve_attempts ==
            health.trace_solves + health.true_residual_tightenings &&
        health.true_residual_tightenings >= 0 &&
        health.line_search_trials >= 0 &&
        health.rejected_line_search_trials >= 0 &&
        std::isfinite(health.initial_residual) &&
        std::isfinite(health.final_residual) &&
        health.final_residual <=
            request.numerical_policy.nonlinear_relative_tolerance *
                (1.0 + health.initial_residual) &&
        health.true_relative_residual <=
            request.numerical_policy.true_residual_relative_tolerance &&
        health.maximum_conservation_relative_error <=
            request.numerical_policy.conservation_relative_tolerance;
    if (!healthy || !session.AcknowledgePreparedOutput(authority, output) ||
        session.Commit(std::move(authority), output).status !=
            dg::hdg::generic::HdgEulerSessionStatus::committed)
      return trajectory;
    StepState state;
    if (!CaptureCudaState(session, mesh, request.polynomial_order,
                          fields.scalar_count, specification.regularization,
                          state))
      return trajectory;
    state.initial_residual = health.initial_residual;
    state.final_residual = health.final_residual;
    state.true_relative_residual = health.true_relative_residual;
    state.maximum_conservation_error = health.maximum_conservation_error;
    state.maximum_conservation_relative_error =
        health.maximum_conservation_relative_error;
    state.nonlinear_updates = health.nonlinear_updates;
    state.trace_solves = health.trace_solves;
    state.linear_iterations = health.linear_iterations;
    state.line_search_trials = health.line_search_trials;
    state.rejected_line_search_trials =
        health.rejected_line_search_trials;
    state.tensor_condensation_launches =
        health.tensor_core.condensation_launches;
    state.tensor_mma_tiles = health.tensor_core.mma_tiles;
    state.tensor_volume_assembly_launches =
        health.tensor_core.volume_assembly_launches;
    state.tensor_volume_mma_tiles = health.tensor_core.volume_mma_tiles;
    state.fallback_count = health.tensor_core.fallback_count;
    trajectory.states.push_back(std::move(state));
  }
  trajectory.passed = trajectory.states.size() ==
      static_cast<std::size_t>(specification.steps + 1);
  return trajectory;
}

bool CompareTrajectory(const Case& specification,
                       const std::string_view comparison,
                       const Trajectory& left, const Trajectory& right)
{
  if (!left.passed || !right.passed ||
      left.states.size() != right.states.size())
    return false;
  bool passed = true;
  for (std::size_t index = 0; index < left.states.size(); ++index)
  {
    const StepState& reference = left.states[index];
    const StepState& device = right.states[index];
    const ComparisonMetrics interior = Compare(
        reference.physical_interior, device.physical_interior);
    const ComparisonMetrics trace = Compare(
        reference.true_trace, device.true_trace);
    const auto near_scalar = [](const double left, const double right)
    {
      const double scale = std::max({1.0, std::abs(left), std::abs(right)});
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <=
              comparison_absolute_tolerance +
              comparison_relative_tolerance * scale;
    };
    const auto near_sigma = [](const double left, const double right)
    {
      const double scale = std::max(std::abs(left), std::abs(right));
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <= sigma_absolute_tolerance +
              sigma_relative_tolerance * scale;
    };
    // Both backends independently enforce the same 1e-10 acceptance policy.
    // CPU accumulates in element/quadrature order while CUDA uses a fixed
    // block/site reduction tree; MPI and host/device FP64 ordering may also
    // differ.  The existing 4096-epsilon comparison isolates that justified
    // reduction freedom without weakening either backend's gate.
    const auto near_conservation = [](const double left, const double right)
    {
      constexpr double tolerance = 4096.0 *
          std::numeric_limits<double>::epsilon();
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <= tolerance;
    };
    const bool igr_sigma = specification.regularization ==
        RegularizationKind::igr;
    const bool sigma_metrics = !igr_sigma ||
        (reference.sigma.available && device.sigma.available &&
         reference.sigma.finite && device.sigma.finite &&
         near_sigma(reference.sigma.global_l2_squared,
                    device.sigma.global_l2_squared) &&
         near_sigma(reference.sigma.global_maximum_absolute,
                    device.sigma.global_maximum_absolute) &&
         near_sigma(reference.sigma.near_mean_square,
                    device.sigma.near_mean_square) &&
         near_sigma(reference.sigma.far_mean_square,
                    device.sigma.far_mean_square) &&
         near_sigma(reference.sigma.near_energy_fraction,
                    device.sigma.near_energy_fraction));
    // Sigma is neutral at generation zero.  Every accepted state of the two
    // discontinuous IGR cases must thereafter carry finite nonzero energy
    // concentrated near the declared off-grid velocity transition.  These
    // are global quadrature integrals, so no coefficient or rank ordering is
    // part of the qualification.
    const bool require_localization =
        specification.require_localized_sigma && index != 0;
    const bool sigma_localization = !require_localization ||
        (reference.sigma.nonzero && reference.sigma.localized &&
         device.sigma.nonzero && device.sigma.localized);
    const bool state_passed = interior.passed && trace.passed &&
        sigma_metrics && sigma_localization &&
        reference.trace_components == device.trace_components &&
        reference.step == device.step &&
        reference.generation == device.generation &&
        reference.time == device.time &&
        near_scalar(reference.maximum_wave_speed,
                    device.maximum_wave_speed) &&
        near_scalar(reference.backend_stable_dt,
                    device.backend_stable_dt) &&
        near_conservation(reference.maximum_conservation_error,
                          device.maximum_conservation_error) &&
        near_conservation(
            reference.maximum_conservation_relative_error,
            device.maximum_conservation_relative_error);
    passed = state_passed && passed;
    if (!state_passed)
    {
      std::cerr << "cross-engine mismatch comparison=" << comparison
                << " case=" << specification.name
                << " state=" << index
                << " interior_rel=" << interior.relative_l2
                << " interior_inf=" << interior.maximum_absolute
                << " trace_rel=" << trace.relative_l2
                << " trace_inf=" << trace.maximum_absolute
                << " trace_components=" << reference.trace_components
                << '/' << device.trace_components
                << " wave=" << reference.maximum_wave_speed
                << '/' << device.maximum_wave_speed
                << " stable_dt=" << reference.backend_stable_dt
                << '/' << device.backend_stable_dt
                << " conservation="
                << reference.maximum_conservation_error << '/'
                << device.maximum_conservation_error
                << " conservation_relative="
                << reference.maximum_conservation_relative_error << '/'
                << device.maximum_conservation_relative_error
                << " residuals=" << reference.initial_residual << '/'
                << device.initial_residual << ','
                << reference.final_residual << '/'
                << device.final_residual << ','
                << reference.true_relative_residual << '/'
                << device.true_relative_residual
                << " work=" << reference.nonlinear_updates << '/'
                << device.nonlinear_updates << ','
                << reference.linear_iterations << '/'
                << device.linear_iterations << ','
                << reference.line_search_trials << '/'
                << device.line_search_trials << " tensor="
                << reference.tensor_condensation_launches << '/'
                << device.tensor_condensation_launches << ','
                << reference.tensor_mma_tiles << '/'
                << device.tensor_mma_tiles << " volume="
                << reference.tensor_volume_assembly_launches << '/'
                << device.tensor_volume_assembly_launches << ','
                << reference.tensor_volume_mma_tiles << '/'
                << device.tensor_volume_mma_tiles << " fallback="
                << reference.fallback_count << '/'
                << device.fallback_count << " sigma_l2="
                << reference.sigma.global_l2_squared << '/'
                << device.sigma.global_l2_squared << " sigma_max="
                << reference.sigma.global_maximum_absolute << '/'
                << device.sigma.global_maximum_absolute << " sigma_near="
                << reference.sigma.near_mean_square << '/'
                << device.sigma.near_mean_square << " sigma_far="
                << reference.sigma.far_mean_square << '/'
                << device.sigma.far_mean_square << " sigma_fraction="
                << reference.sigma.near_energy_fraction << '/'
                << device.sigma.near_energy_fraction << " sigma_flags="
                << reference.sigma.available << reference.sigma.finite
                << reference.sigma.nonzero << reference.sigma.localized << '/'
                << device.sigma.available << device.sigma.finite
                << device.sigma.nonzero << device.sigma.localized << '\n';
    }
  }
  return passed;
}

Trajectory RunCase(const Case& specification, const BackendKind backend,
                   const AlignmentSelection& selection,
                   const CudaEngineKind cuda_engine =
                       CudaEngineKind::ordinary)
{
  const int order = selection.order;
  mfem::ParMesh mesh = MakeMesh(specification.physical);
  BorrowedFields fields = MakeFields(
      mesh, specification.initial_mode, order,
      specification.static_source_scale,
      specification.supply_initial_true_trace);
  if (fields.initial.size() != fields.scalar_count * components ||
      !PhysicalCoefficients(fields.initial, fields.scalar_count) ||
      ((specification.static_source_scale != 0.0) !=
       !fields.source.empty()) ||
      (specification.supply_initial_true_trace !=
       !fields.initial_true_trace.empty()) ||
      (specification.supply_initial_true_trace &&
       (fields.initial_true_trace.size() % components != 0 ||
        !PhysicalCoefficients(
            fields.initial_true_trace,
            fields.initial_true_trace.size() / components))) ||
      (specification.initial_mode == InitialMode::localized_igr_contact &&
       !HasIgrContactAndVelocityTransition(fields)))
    return {};
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, fields, boundary, specification, backend, selection,
      cuda_engine);
  return backend == BackendKind::cpu
      ? RunCpu(request, mesh, fields, specification)
      : RunCuda(
            request, mesh, fields, specification,
            SelectedCudaTracePolicy(
                selection, cuda_engine, specification.regularization));
}

bool ParseOrder(const std::string_view text, int& order) noexcept
{
  order = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), order);
  return result.ec == std::errc{} &&
      result.ptr == text.data() + text.size() && order >= 1 && order <= 3;
}

bool ParseQuadratureOrder(const std::string_view text, int& order) noexcept
{
  order = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), order);
  return result.ec == std::errc{} &&
      result.ptr == text.data() + text.size() && order >= 1 && order <= 14;
}

bool ParsePositiveDouble(
    const std::string_view text, double& value) noexcept
{
  value = 0.0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), value,
      std::chars_format::general);
  return result.ec == std::errc{} &&
      result.ptr == text.data() + text.size() &&
      std::isfinite(value) && value > 0.0;
}

bool ParseCudaTracePolicy(
    const std::string_view text, CudaTracePolicyKind& policy) noexcept
{
  constexpr std::array<CudaTracePolicyKind, 8> policies{{
      CudaTracePolicyKind::engine_default,
      CudaTracePolicyKind::common_fast,
      CudaTracePolicyKind::equilibrated_mgr,
      CudaTracePolicyKind::native_element_schwarz_fp64,
      CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64}};
  for (const CudaTracePolicyKind candidate : policies)
    if (text == CudaTracePolicyKindName(candidate))
    {
      policy = candidate;
      return true;
    }
  return false;
}

bool ValidAlignmentSelection(
    const AlignmentSelection& selection) noexcept
{
  const EulerHdgNumericalPolicy conventional = SelectedNumericalPolicy(
      selection, RegularizationKind::none);
  const EulerHdgNumericalPolicy igr = SelectedNumericalPolicy(
      selection, RegularizationKind::igr);
  const auto tensor_igr_trace =
      dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
          CudaEngineKind::tensor_core, true,
          selection.tensor_igr_trace_policy);
  return dg::hdg::generic::ValidEulerHdgNumericalPolicy(conventional) &&
      dg::hdg::generic::ValidEulerHdgNumericalPolicy(igr) &&
      dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptorIsExact(
          tensor_igr_trace);
}

bool ParseSelection(const int argc, char* argv[],
                    AlignmentSelection& selection) noexcept
{
  selection = {};
  bool have_order = false;
  bool have_igr_volume_order = false;
  bool have_igr_face_order = false;
  bool have_newton_tolerance = false;
  bool have_final_tolerance = false;
  bool have_tightening_tolerance = false;
  bool have_tensor_igr_trace_policy = false;
  bool have_validate_options_only = false;
  for (int argument = 1; argument < argc; ++argument)
  {
    const std::string_view option{argv[argument]};
    if (option == "--order" && argument + 1 < argc && !have_order)
    {
      have_order = ParseOrder(argv[++argument], selection.order);
      if (!have_order) return false;
    }
    else if (option == "--igr-volume-quadrature-order" &&
             argument + 1 < argc && !have_igr_volume_order)
    {
      have_igr_volume_order = ParseQuadratureOrder(
          argv[++argument],
          selection.igr_assembly_quadrature_orders.volume_order);
      if (!have_igr_volume_order) return false;
    }
    else if (option == "--igr-face-quadrature-order" &&
             argument + 1 < argc && !have_igr_face_order)
    {
      have_igr_face_order = ParseQuadratureOrder(
          argv[++argument],
          selection.igr_assembly_quadrature_orders.face_order);
      if (!have_igr_face_order) return false;
    }
    else if (option == "--newton-linear-rtol" &&
             argument + 1 < argc && !have_newton_tolerance)
    {
      have_newton_tolerance = ParsePositiveDouble(
          argv[++argument],
          selection.newton_correction_linear_relative_tolerance);
      if (!have_newton_tolerance) return false;
    }
    else if (option == "--final-linear-rtol" &&
             argument + 1 < argc && !have_final_tolerance)
    {
      have_final_tolerance = ParsePositiveDouble(
          argv[++argument],
          selection.final_coherent_linear_relative_tolerance);
      if (!have_final_tolerance) return false;
    }
    else if (option == "--true-residual-tightening-linear-rtol" &&
             argument + 1 < argc && !have_tightening_tolerance)
    {
      have_tightening_tolerance = ParsePositiveDouble(
          argv[++argument],
          selection.true_residual_tightening_linear_relative_tolerance);
      if (!have_tightening_tolerance) return false;
    }
    else if (option == "--tensor-igr-trace-policy" &&
             argument + 1 < argc && !have_tensor_igr_trace_policy)
    {
      have_tensor_igr_trace_policy = ParseCudaTracePolicy(
          argv[++argument], selection.tensor_igr_trace_policy);
      if (!have_tensor_igr_trace_policy) return false;
    }
    else if (option == "--validate-options-only" &&
             !have_validate_options_only)
    {
      selection.validate_options_only = true;
      have_validate_options_only = true;
    }
    else
    {
      return false;
    }
  }
  if (have_igr_volume_order != have_igr_face_order) return false;
  return ValidAlignmentSelection(selection);
}

void PrintUsage(const char* const executable)
{
  std::cerr
      << "Usage: " << executable
      << " [--order 1|2|3]"
         " [--igr-volume-quadrature-order Q"
         " --igr-face-quadrature-order Q]"
         " [--newton-linear-rtol R] [--final-linear-rtol R]"
         " [--true-residual-tightening-linear-rtol R]"
         " [--tensor-igr-trace-policy engine-default|common-fast|"
         "equilibrated-mgr|native-element-schwarz-fp64|"
         "native-element-schwarz-matrix-free-fp64|"
         "native-element-schwarz-equilibrated-field-split-fp64|"
         "native-element-schwarz-matrix-free-equilibrated-field-split-fp64|"
         "native-element-schwarz-matrix-free-warp-equilibrated-field-split-"
         "fp64] [--validate-options-only]\n";
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  AlignmentSelection selection;
  if (!ParseSelection(argc, argv, selection))
  {
    PrintUsage(argv[0]);
    return 2;
  }
  const int order = selection.order;
  if (ranks != 1)
  {
    std::cerr << "CPU/CUDA numerical-alignment qualification is rank-one; "
                 "rank-two correctness has separate generic coverage\n";
    return 2;
  }
  if (selection.validate_options_only)
  {
    const EulerHdgNumericalPolicy igr = SelectedNumericalPolicy(
        selection, RegularizationKind::igr);
    std::cout << "HDG numerical-alignment option validation passed order="
              << order << " igr_quadrature="
              << igr.igr_assembly_quadrature_orders.volume_order << '/'
              << igr.igr_assembly_quadrature_orders.face_order
              << " linear_tolerances="
              << igr.newton_correction_linear_relative_tolerance << '/'
              << igr.final_coherent_linear_relative_tolerance << '/'
              << igr.true_residual_tightening_linear_relative_tolerance
              << " tensor_igr_trace_policy="
              << CudaTracePolicyKindName(
                     selection.tensor_igr_trace_policy) << '\n';
    return 0;
  }
  constexpr std::size_t case_count = 6;
  const std::array<Case, case_count> cases{{
      {"conventional-smooth-short-trajectory", InitialMode::smooth_periodic,
       RegularizationKind::none, false, 3, 1.0e-5, false, 0.0, false},
      {"conventional-smooth-slip-source-trajectory",
       InitialMode::smooth_periodic, RegularizationKind::none, true, 2,
       1.0e-5, false, 1.0, false},
      {"igr-smooth-short-trajectory", InitialMode::smooth_periodic,
       RegularizationKind::igr, false, 3, 1.0e-5, false, 0.0, false},
      {"igr-smooth-source-supplied-trace-trajectory",
       InitialMode::smooth_periodic, RegularizationKind::igr, false, 2,
       1.0e-5, false, 1.0, true},
      {"igr-periodic-localized-contact", InitialMode::localized_igr_contact,
       RegularizationKind::igr, false, 2, 1.0e-5, true, 0.0, false},
      {"igr-slip-boundary-localized-contact",
       InitialMode::localized_igr_contact,
       RegularizationKind::igr, true, 2, 1.0e-5, true, 0.0, false}}};

  // MFEM and HYPRE device selection is process-global.  Form every CPU
  // reference trajectory under the default host policy first, then select
  // CUDA once and form all device trajectories.  Interleaving the policies
  // in one process would make backend agreement depend on global setup order.
  std::array<Trajectory, case_count> cpu_trajectories{};
  for (std::size_t index = 0; index < cases.size(); ++index)
    cpu_trajectories[index] = RunCase(
        cases[index], BackendKind::cpu, selection);
  bool cpu_phase_ready = true;
  for (const Trajectory& trajectory : cpu_trajectories)
    cpu_phase_ready = trajectory.passed && cpu_phase_ready;
  if (!All(cpu_phase_ready))
  {
    std::cerr << "CPU reference trajectory construction failed\n";
    return 1;
  }

  // A successful host solve may leave HYPRE's process-global convergence bit
  // set even though the independently checked residual gates passed.  Clear
  // that library-global state at this explicit test-only backend boundary;
  // ordinary applications select exactly one backend per process.
  const bool hypre_transition_clean = HYPRE_ClearAllErrors() == 0 &&
      HYPRE_GetError() == 0;
  if (!All(hypre_transition_clean))
  {
    std::cerr << "HYPRE host-to-CUDA test transition failed\n";
    return 1;
  }

  mfem::Device device("cuda");
  std::array<Trajectory, case_count> ordinary_trajectories{};
  std::array<Trajectory, case_count> tensor_trajectories{};
  for (std::size_t index = 0; index < cases.size(); ++index)
    ordinary_trajectories[index] = RunCase(
        cases[index], BackendKind::cuda, selection,
        CudaEngineKind::ordinary);
  const bool ordinary_phase_ready = std::all_of(
      ordinary_trajectories.begin(), ordinary_trajectories.end(),
      [](const Trajectory& trajectory) { return trajectory.passed; });
  if (!All(ordinary_phase_ready))
  {
    std::cerr << "ordinary CUDA trajectory construction failed order="
              << order << '\n';
    return 1;
  }
  const bool cuda_engine_transition_clean = HYPRE_ClearAllErrors() == 0 &&
      HYPRE_GetError() == 0;
  if (!All(cuda_engine_transition_clean))
  {
    std::cerr << "ordinary-to-tensor CUDA test transition failed\n";
    return 1;
  }
  for (std::size_t index = 0; index < cases.size(); ++index)
    tensor_trajectories[index] = RunCase(
        cases[index], BackendKind::cuda, selection,
        CudaEngineKind::tensor_core);
  const bool tensor_phase_ready = std::all_of(
      tensor_trajectories.begin(), tensor_trajectories.end(),
      [](const Trajectory& trajectory) { return trajectory.passed; });
  if (!All(tensor_phase_ready))
  {
    std::cerr << "tensor CUDA trajectory construction failed order="
              << order << '\n';
    return 1;
  }

  bool passed = true;
  for (std::size_t index = 0; index < cases.size(); ++index)
  {
    const bool cpu_ordinary = CompareTrajectory(
        cases[index], "cpu-vs-ordinary", cpu_trajectories[index],
        ordinary_trajectories[index]);
    const bool cpu_tensor = CompareTrajectory(
        cases[index], "cpu-vs-tensor", cpu_trajectories[index],
        tensor_trajectories[index]);
    const bool ordinary_tensor = CompareTrajectory(
        cases[index], "ordinary-vs-tensor", ordinary_trajectories[index],
        tensor_trajectories[index]);
    const bool case_passed =
        cpu_ordinary && cpu_tensor && ordinary_tensor;
    if (!case_passed)
    {
      std::cerr << "cross-engine qualification failed case="
                << cases[index].name << " order=" << order
                << " comparisons=" << cpu_ordinary << '/'
                << cpu_tensor << '/' << ordinary_tensor
                << " cpu_states="
                << cpu_trajectories[index].states.size()
                << " ordinary_states="
                << ordinary_trajectories[index].states.size()
                << " tensor_states="
                << tensor_trajectories[index].states.size() << '\n';
    }
    passed = case_passed && passed;
  }
  passed = All(passed);
  if (passed)
  {
    const EulerHdgNumericalPolicy igr = SelectedNumericalPolicy(
        selection, RegularizationKind::igr);
    std::cout << "HDG CPU/ordinary-CUDA/tensor-CUDA numerical alignment and "
                 "IGR discontinuity qualification passed order="
              << order << " igr_quadrature="
              << igr.igr_assembly_quadrature_orders.volume_order << '/'
              << igr.igr_assembly_quadrature_orders.face_order
              << " linear_tolerances="
              << igr.newton_correction_linear_relative_tolerance << '/'
              << igr.final_coherent_linear_relative_tolerance << '/'
              << igr.true_residual_tightening_linear_relative_tolerance
              << " tensor_igr_trace_policy="
              << CudaTracePolicyKindName(
                     selection.tensor_igr_trace_policy) << '\n';
  }
  return passed ? 0 : 1;
}
