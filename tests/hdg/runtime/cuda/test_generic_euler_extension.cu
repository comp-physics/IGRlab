#include "igrlab/environment.hpp"
#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/euler/zhang_shu_u4.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "numerics/hdg/runtime/cuda/tensor_volume_layout.hpp"
#include "test_cuda_trace_solver_routing.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

constexpr double qualification_density_gate = 1.0e-8;

constexpr bool CudaPreparedIluAccountingValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const int linear_iterations, const generic::CudaEngineKind engine,
    const bool igr_active) noexcept
{
  const auto policy = ResolveCudaTracePolicy(
      engine, igr_active, tensor.trace_policy_rank_count,
      tensor.trace_policy_order, tensor.requested_trace_policy);
  if (!CudaTracePolicyDescriptorIsExact(policy) ||
      tensor.resolved_trace_policy != policy.resolved_selection)
    return false;
  const auto preconditioner = policy.trace_preconditioner;
  const bool ilu = CudaTracePreconditionerIsIlu(preconditioner);
  const auto factor_reuse_policy = policy.ilu_factor_reuse_policy;
  const auto triangular_solve_kind = policy.ilu_triangular_solve_kind;
  const int lower_jacobi_iterations = policy.ilu_lower_jacobi_iterations;
  const int upper_jacobi_iterations = policy.ilu_upper_jacobi_iterations;
  if (!ilu)
    return (preconditioner == CudaTracePreconditionerKind::hypre_diag_scale ||
            CudaTracePreconditionerIsNativeElementSchwarz(preconditioner)) &&
        tensor.ilu_type == -1 && tensor.ilu_level_of_fill == -1 &&
        tensor.ilu_local_reordering == -1 &&
        tensor.ilu_application_iterations == 0 &&
        triangular_solve_kind ==
            CudaIluTriangularSolveKind::not_applicable &&
        tensor.ilu_triangular_solve_kind == triangular_solve_kind &&
        tensor.ilu_lower_jacobi_iterations == 0 &&
        tensor.ilu_upper_jacobi_iterations == 0 &&
        tensor.ilu_accounting_kind == CudaIluAccountingKind::unavailable &&
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
  const int expected_fill = 0;
  const int expected_reordering = igr_active ? 1 : 0;
  if (trace_solves <= 0 || linear_iterations < 0 || tensor.ilu_type != 0 ||
      tensor.ilu_level_of_fill != expected_fill ||
      tensor.ilu_local_reordering != expected_reordering ||
      tensor.ilu_application_iterations != 1 ||
      tensor.ilu_triangular_solve_kind != triangular_solve_kind ||
      tensor.ilu_lower_jacobi_iterations != lower_jacobi_iterations ||
      tensor.ilu_upper_jacobi_iterations != upper_jacobi_iterations ||
      (triangular_solve_kind == CudaIluTriangularSolveKind::direct &&
       (lower_jacobi_iterations != 0 || upper_jacobi_iterations != 0)) ||
      (triangular_solve_kind ==
           CudaIluTriangularSolveKind::iterative_jacobi &&
       (lower_jacobi_iterations != 5 || upper_jacobi_iterations != 5)) ||
      tensor.ilu_accounting_kind !=
          CudaIluAccountingKind::derived_bounds_v1 ||
      !tensor.ilu_accounting_complete ||
      tensor.ilu_preconditioner_callbacks_lower_bound >
          tensor.ilu_preconditioner_callbacks_upper_bound)
    return false;
  const std::uint64_t solves = static_cast<std::uint64_t>(trace_solves);
  const bool factor_work = factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve
      ? tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == solves &&
            tensor.ilu_factor_reuses == 0 &&
            tensor.ilu_factor_age_generations == 0
      : factor_reuse_policy ==
              CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
            tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == 1 &&
            tensor.ilu_factor_reuses == solves - 1 &&
            tensor.ilu_factor_age_generations == solves - 1;
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

struct CudaPreparedIgrConservationCapture
{
  static constexpr double sigma_hat_probe = 0.125;
  int elements = 0;
  int scalar_interior_dofs = 0;
  int scalar_face_dofs = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  int volume_points = 0;
  int face_points = 0;
  int order = 0;
  int volume_assembly_order = 0;
  int face_assembly_order = 0;
  double alpha = 0.0;
  double beta = 0.0;
  double dt = 0.0;
  double lifecycle_maximum_error =
      std::numeric_limits<double>::infinity();
  double lifecycle_maximum_relative_error =
      std::numeric_limits<double>::infinity();
  std::vector<double> previous_interior{};
  std::vector<double> current_interior{};
  std::vector<double> current_local_trace{};
  std::array<double, euler_conservation_result_scalars> production_local{};
  std::array<double, euler_conservation_result_scalars>
      probed_production_local{};
};

struct CudaPreparedEulerConservationGateCapture
{
  std::array<double, euler_conservation_result_scalars> production_local{};
  double lifecycle_maximum_error =
      std::numeric_limits<double>::infinity();
  double lifecycle_maximum_relative_error =
      std::numeric_limits<double>::infinity();
  int nonlinear_updates = -1;
  FailureReason reason = FailureReason::none;
  CudaFailurePhase phase = CudaFailurePhase::setup;
  bool candidate_ready = false;
  bool conservation = false;
  bool success = false;
  bool pending_active = false;
  bool staging_retained = false;
  bool prestage_allowed = false;
};

struct CudaPreparedPartitionedClosureAudit
{
  double global_scalar_residual =
      std::numeric_limits<double>::infinity();
  double global_auxiliary_norm =
      std::numeric_limits<double>::infinity();
  double global_scalar_relative_residual =
      std::numeric_limits<double>::infinity();
  double relative_residual_tolerance = 0.0;
  bool auxiliary_interior_finite = false;
  bool auxiliary_true_trace_finite = false;
  bool auxiliary_local_trace_finite = false;
};

struct CudaInitializedRodas3IgrClosureAudit
{
  double global_scalar_residual =
      std::numeric_limits<double>::infinity();
  double global_scalar_relative_residual =
      std::numeric_limits<double>::infinity();
  double relative_residual_tolerance = 0.0;
  std::array<double, 3> global_auxiliary_norms{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  bool euler_interior_exact = false;
  bool euler_true_trace_exact = false;
  bool auxiliary_interior_finite_nontrivial = false;
  bool auxiliary_true_trace_finite_nontrivial = false;
  bool auxiliary_local_trace_finite_nontrivial = false;
  bool lifecycle_zero = false;
  bool combination_alias_exact = false;
};

struct CudaPreparedLocalCondensationCapture
{
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::uint64_t tensor_mma_tiles = 0;
  std::vector<double> ordinary_schur{};
  std::vector<double> ordinary_rhs{};
  std::vector<double> tensor_schur{};
  std::vector<double> tensor_rhs{};
};

struct CudaBorrowedAssemblyFusionAudit
{
  // residual_r, residual_g, A, B, C, D, f, g in that order.
  std::array<double, 8> maximum_absolute_errors{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<double, 8> maximum_scaled_errors{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  std::array<bool, 8> bitwise_equal{};
  int volume_points = 0;
  double fp64_comparison_tolerance = 0.0;
  bool passes_fp64_threshold = false;
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  std::uint64_t ordinary_volume_mma_tiles = 0;
  std::uint64_t tensor_volume_mma_tiles = 0;
  bool status_bitwise_equal = false;
  bool status_all_passed = false;
};

struct CudaPreparedRecoveryAudit
{
  int elements = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  double maximum_direct_scaled_error =
      std::numeric_limits<double>::infinity();
  double maximum_workspace_scaled_error =
      std::numeric_limits<double>::infinity();
  double maximum_factor_scaled_residual =
      std::numeric_limits<double>::infinity();
  double maximum_workspace_solve_scaled_residual =
      std::numeric_limits<double>::infinity();
  double maximum_scaled_residual =
      std::numeric_limits<double>::infinity();
  double tolerance = 0.0;
};

bool SolveRecoveryReference(std::vector<double>& matrix,
                            std::vector<double>& right_hand_side,
                            const int size) noexcept
{
  if (size <= 0 ||
      matrix.size() != static_cast<std::size_t>(size) * size ||
      right_hand_side.size() != static_cast<std::size_t>(size))
    return false;
  double scale = 0.0;
  for (const double value : matrix)
  {
    if (!std::isfinite(value)) return false;
    scale = std::max(scale, std::abs(value));
  }
  for (const double value : right_hand_side)
    if (!std::isfinite(value)) return false;
  for (int column = 0; column < size; ++column)
  {
    int pivot = column;
    for (int row = column + 1; row < size; ++row)
      if (std::abs(matrix[static_cast<std::size_t>(row) * size + column]) >
          std::abs(matrix[static_cast<std::size_t>(pivot) * size + column]))
        pivot = row;
    const double diagonal =
        matrix[static_cast<std::size_t>(pivot) * size + column];
    if (!(std::abs(diagonal) >
          64.0 * std::numeric_limits<double>::epsilon() * scale))
      return false;
    if (pivot != column)
    {
      for (int entry = 0; entry < size; ++entry)
        std::swap(matrix[static_cast<std::size_t>(column) * size + entry],
                  matrix[static_cast<std::size_t>(pivot) * size + entry]);
      std::swap(right_hand_side[column], right_hand_side[pivot]);
    }
    const double factor_diagonal =
        matrix[static_cast<std::size_t>(column) * size + column];
    for (int row = column + 1; row < size; ++row)
    {
      const double multiplier =
          matrix[static_cast<std::size_t>(row) * size + column] /
          factor_diagonal;
      matrix[static_cast<std::size_t>(row) * size + column] = multiplier;
      for (int entry = column + 1; entry < size; ++entry)
        matrix[static_cast<std::size_t>(row) * size + entry] -= multiplier *
            matrix[static_cast<std::size_t>(column) * size + entry];
      right_hand_side[row] -= multiplier * right_hand_side[column];
    }
  }
  for (int row = size - 1; row >= 0; --row)
  {
    double value = right_hand_side[row];
    for (int column = row + 1; column < size; ++column)
      value -= matrix[static_cast<std::size_t>(row) * size + column] *
          right_hand_side[column];
    right_hand_side[row] = value /
        matrix[static_cast<std::size_t>(row) * size + row];
    if (!std::isfinite(right_hand_side[row])) return false;
  }
  return true;
}

struct CudaRodas3ScreenObservation
{
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::infinity();
  bool candidate_ready = false;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool telemetry = false;
  bool success = false;
};

struct CudaRodas3LimiterEndpointAudit
{
  double maximum_candidate_formula_scaled_error =
      std::numeric_limits<double>::infinity();
  bool candidate_formula_nontrivial = false;
  bool candidate_formula_passed = false;
};

struct CudaDirectZhangShuAudit
{
  std::vector<double> limited_u4{};
  std::vector<double> theta_rho{};
  std::vector<double> theta_p{};
  int classification = 2;
  double correction_squared = std::numeric_limits<double>::infinity();
};


// This accessor is defined only in the qualification translation unit.  The
// production engine and generic facade grant it private friendship; its
// detail-only launch helpers publish no runtime diagnostic API or state.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool VerifyInitialStateHashesAndReferenceShapes(Engine& engine)
  {
    auto& numerics = engine.numerics_;
    const auto& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && shape.borrowed_host_inputs &&
        shape.elements > 0 && shape.interior_count > 0 &&
        shape.true_trace_count > 0 && numerics.ranks_ >= 1 &&
        numerics.ranks_ <= 2;
    if (!numerics.CollectiveContinue(local_valid)) return false;

    // Independently rebuild each reference basis. Reused reference values
    // must have exactly the original bits, including signed zeros.
    mfem::Vector expected(shape.scalar_interior_dofs);
    for (int element = 0; element < shape.elements; ++element)
    {
      const auto* fe = numerics.interior_space_->GetFE(element);
      const auto& rule = mfem::IntRules.Get(
          fe->GetGeomType(), shape.volume_assembly_order);
      for (int q = 0; q < shape.volume_points; ++q)
      {
        fe->CalcShape(rule.IntPoint(q), expected);
        const std::size_t offset =
            (static_cast<std::size_t>(element) * shape.volume_points + q) *
            shape.scalar_interior_dofs;
        local_valid = local_valid && std::memcmp(
            expected.HostRead(), numerics.host_volume_shape_.data() + offset,
            static_cast<std::size_t>(shape.scalar_interior_dofs) *
                sizeof(double)) == 0;
      }
    }

    // Read both actual device slots and hash their bytes on the host. Merely
    // comparing copied metadata would miss a stale or incorrectly seeded slot.
    const auto& committed = numerics.committed_slot_;
    const auto& candidate = numerics.candidate_slot_;
    for (int trace = 0; trace < 2; ++trace)
    {
      const std::size_t count = trace ? committed.trace.count
                                      : committed.interior.count;
      std::array<std::vector<double>, 2> values{{
          std::vector<double>(count), std::vector<double>(count)}};
      for (int slot_index = 0; slot_index < 2; ++slot_index)
      {
        const auto& slot = slot_index ? candidate : committed;
        const double* coefficients = trace ? slot.trace.coefficients
                                            : slot.interior.coefficients;
        const std::size_t actual_count = trace ? slot.trace.count
                                               : slot.interior.count;
        const StateHash& local_hash = trace ? slot.trace.local_hash
                                            : slot.interior.local_hash;
        const StateHash& aggregate_hash = trace ? slot.trace.hash
                                                : slot.interior.hash;
        const bool copied = actual_count == count && coefficients &&
            CudaTransportPolicy::DeviceToHost(
                values[slot_index].data(), coefficients,
                count * sizeof(double),
                CudaTransferCategory::generic_host_bridge,
                CudaTransferSite::generic_host_export) &&
            CudaTransportPolicy::Synchronize(
                CudaTransferSite::generic_host_export);
        if (!numerics.CollectiveContinue(copied)) return false;
        const StateHash observed = HashHostBytes(
            values[slot_index].data(), count * sizeof(double));
        std::array<unsigned char, 64> rank_hashes{};
        const bool gathered = MPI_Allgather(
            observed.bytes.data(), 32, MPI_UNSIGNED_CHAR,
            rank_hashes.data(), 32, MPI_UNSIGNED_CHAR,
            numerics.communicator_) == MPI_SUCCESS;
        local_valid = local_valid && gathered && observed == local_hash &&
            HashHostBytes(rank_hashes.data(),
                static_cast<std::size_t>(numerics.ranks_) * 32) ==
                aggregate_hash;
      }
      local_valid = local_valid && std::memcmp(
          values[0].data(), values[1].data(), count * sizeof(double)) == 0;
    }
    return numerics.CollectiveContinue(local_valid);
  }

  static bool VerifyInitialStateHashesAndReferenceShapes(
      cuda::GenericEulerHdgCudaSession& session)
  {
    if (session.borrowed_rank1_)
      return VerifyInitialStateHashesAndReferenceShapes(*session.borrowed_rank1_);
    if (session.borrowed_rank2_)
      return VerifyInitialStateHashesAndReferenceShapes(*session.borrowed_rank2_);
    if (session.partitioned_backward_euler_borrowed_rank1_)
      return VerifyInitialStateHashesAndReferenceShapes(
          *session.partitioned_backward_euler_borrowed_rank1_);
    if (session.partitioned_backward_euler_borrowed_rank2_)
      return VerifyInitialStateHashesAndReferenceShapes(
          *session.partitioned_backward_euler_borrowed_rank2_);
    if (session.rodas3_borrowed_rank1_)
      return VerifyInitialStateHashesAndReferenceShapes(
          *session.rodas3_borrowed_rank1_);
    if (session.rodas3_borrowed_rank2_)
      return VerifyInitialStateHashesAndReferenceShapes(
          *session.rodas3_borrowed_rank2_);
    return false;
  }

  template <typename Engine>
  static bool ApplyDirectZhangShu(
      Engine& engine, const std::vector<double>& input,
      CudaDirectZhangShuAudit& audit) noexcept
  {
    audit = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const int slot = numerics.candidate_limiter_diagnostics_;
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && numerics.positivity_limiter_enabled_ &&
        shape.borrowed_host_inputs && !shape.igr_active &&
        shape.interior_components == conserved_components &&
        shape.elements > 0 && shape.interior_count > 0 &&
        input.size() == static_cast<std::size_t>(shape.interior_count) &&
        slot >= 0 && slot < 2 && numerics.view_.work_interior != nullptr &&
        numerics.view_.status_flag != nullptr &&
        numerics.view_.scalar != nullptr;
    try
    {
      audit.limited_u4.resize(input.size(), 0.0);
      audit.theta_rho.resize(static_cast<std::size_t>(shape.elements), 0.0);
      audit.theta_p.resize(static_cast<std::size_t>(shape.elements), 0.0);
    }
    catch (...)
    {
      local_valid = false;
    }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const auto& policy = engine.config_.numerical_policy.positivity_limiter;
    const bool launched = CudaLaunchInitializeZhangShuDiagnostics(
            numerics.view_, shape) &&
        CudaTransportPolicy::HostToDevice(
            numerics.view_.work_interior, input.data(),
            input.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaLaunchZhangShuU4(
            numerics.view_, shape, policy, slot,
            numerics.view_.status_flag, numerics.view_.scalar);
    if (!numerics.CollectiveContinue(launched)) return false;

    const bool copied = CudaTransportPolicy::DeviceToHost(
            &audit.classification, numerics.view_.status_flag,
            sizeof(audit.classification),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            &audit.correction_squared, numerics.view_.scalar,
            sizeof(audit.correction_squared),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            audit.limited_u4.data(), numerics.view_.work_interior,
            audit.limited_u4.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            audit.theta_rho.data(), numerics.view_.positivity_theta_rho[slot],
            audit.theta_rho.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            audit.theta_p.data(), numerics.view_.positivity_theta_p[slot],
            audit.theta_p.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = copied && audit.classification >= 0 &&
        audit.classification <= 3 &&
        std::isfinite(audit.correction_squared) &&
        audit.correction_squared >= 0.0;
    return numerics.CollectiveContinue(local_valid);
  }

  static bool CorruptFacadePreparedAssessmentPolicy(
      cuda::GenericEulerHdgCudaSession& session) noexcept
  {
    if (!session.assessment_authority_) return false;
    session.assessed_policy_bits_[0] ^= 1U;
    return true;
  }

  static bool FacadePreparedAssessmentInactive(
      const cuda::GenericEulerHdgCudaSession& session) noexcept
  {
    return !session.assessment_authority_ && session.assessed_epoch_ == 0 &&
        session.assessed_facade_incarnation_ == 0 &&
        session.assessed_native_export_ == nullptr &&
        session.assessed_policy_bits_ == std::array<std::uint64_t, 3>{};
  }

  template <typename Selection>
  static std::unique_ptr<PersistentCudaSmoothEulerEngine<Selection>>
  CreateWithTracePolicy(
      const CudaConfig& config,
      const CudaTracePolicyKind requested_policy) noexcept
  {
    using Engine = PersistentCudaSmoothEulerEngine<Selection>;
    const bool igr = config.use_generic_numerical_policy &&
        config.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr;
    const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
        config.cuda_engine, igr, config.expected_rank_count, config.order,
        requested_policy);
    if (!CudaTracePolicyDescriptorIsExact(policy)) return {};
    try
    {
      return std::unique_ptr<Engine>(new Engine(config, policy));
    }
    catch (...) { return {}; }
  }

  template <typename Engine>
  static bool CapturePreparedRodas3(
      Engine& engine, CudaRodas3ScreenObservation& observation) noexcept
  {
    observation = {};
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging ||
        engine.config_.time_treatment != generic::EulerHdgTimeTreatment::rodas3)
      return false;
    const auto& numerical = engine.pending_.numerical;
    observation.maximum_conservation_error =
        numerical.maximum_conservation_error;
    observation.maximum_conservation_relative_error =
        numerical.maximum_conservation_relative_error;
    observation.candidate_ready = numerical.candidate_ready;
    observation.finite = numerical.finite;
    observation.admissible = numerical.admissible;
    observation.recovery = numerical.recovery;
    observation.true_residual = numerical.true_residual;
    observation.conservation = numerical.conservation;
    observation.telemetry = numerical.telemetry;
    observation.success = numerical.success;
    return true;
  }

  template <typename Engine>
  static bool VerifyLimiterDiagnosticAcceptedHistorySeed(
      Engine& engine, const std::vector<double>& expected_theta_rho,
      const std::vector<double>& expected_theta_p) noexcept
  {
    auto& numerics = engine.numerics_;
    const std::size_t elements =
        static_cast<std::size_t>(numerics.shape_.elements);
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && numerics.positivity_limiter_enabled_ &&
        elements > 0 && expected_theta_rho.size() == elements &&
        expected_theta_p.size() == elements &&
        numerics.committed_limiter_diagnostics_ >= 0 &&
        numerics.committed_limiter_diagnostics_ < 2 &&
        numerics.candidate_limiter_diagnostics_ >= 0 &&
        numerics.candidate_limiter_diagnostics_ < 2;
    std::vector<double> reset, observed;
    try
    {
      reset.assign(2 * elements, 1.0);
      observed.resize(2 * elements, 0.0);
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    double* const candidate = numerics.view_.positivity_theta_rho[
        numerics.candidate_limiter_diagnostics_];
    const bool reset_copied = CudaTransportPolicy::HostToDevice(
        candidate, reset.data(), reset.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(reset_copied)) return false;
    if (!numerics.SeedCandidateLimiterDiagnostics()) return false;
    const bool observed_copied = CudaTransportPolicy::DeviceToHost(
            observed.data(), candidate, observed.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = observed_copied;
    for (std::size_t element = 0;
         element < elements && local_valid; ++element)
      local_valid = observed[element] == expected_theta_rho[element] &&
          observed[elements + element] == expected_theta_p[element];
    return numerics.CollectiveContinue(local_valid);
  }

  template <typename Engine>
  static bool AuditPreparedRodas3LimiterEndpoint(
      Engine& engine, CudaRodas3LimiterEndpointAudit& audit) noexcept
  {
    audit = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging && numerics.initialized_ && numerics.rodas3_ &&
        engine.config_.numerical_policy.positivity_limiter.kind ==
            generic::EulerHdgPositivityLimiterKind::zhang_shu &&
        shape.borrowed_host_inputs && shape.elements > 0 &&
        shape.scalar_interior_dofs > 0 &&
        shape.interior_components >= conserved_components &&
        shape.interior_count > 0 && shape.true_trace_count > 0 &&
        numerics.view_.work_interior != nullptr &&
        numerics.view_.output_interior != nullptr &&
        numerics.view_.recovered_interior != nullptr;
    if (!numerics.CollectiveContinue(local_valid)) return false;

    std::array<std::vector<double>, 3> states;
    try
    {
      for (auto& state : states)
        state.resize(static_cast<std::size_t>(shape.interior_count));
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const bool copied = CudaTransportPolicy::DeviceToHost(
            states[0].data(), numerics.view_.work_interior,
            states[0].size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            states[1].data(), numerics.view_.output_interior,
            states[1].size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            states[2].data(), numerics.view_.recovered_interior,
            states[2].size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(copied)) return false;

    audit.maximum_candidate_formula_scaled_error = 0.0;
    double maximum_formula_increment = 0.0;
    for (int element = 0; element < shape.elements; ++element)
      for (int local = 0;
           local < conserved_components * shape.scalar_interior_dofs;
           ++local)
      {
        const std::size_t index = static_cast<std::size_t>(element) *
                static_cast<std::size_t>(shape.interior_dofs) +
            static_cast<std::size_t>(local);
        const double high = states[0][index];
        const double admitted_final_stage = states[1][index];
        const double final_stage_increment = states[2][index];
        const double expected = admitted_final_stage +
            0.5 * final_stage_increment;
        const double scale = std::max({1.0, std::abs(high),
                                       std::abs(expected)});
        const double scaled_error = std::abs(high - expected) / scale;
        maximum_formula_increment = std::max(
            maximum_formula_increment,
            std::abs(0.5 * final_stage_increment));
        local_valid = local_valid && std::isfinite(high) &&
            std::isfinite(admitted_final_stage) &&
            std::isfinite(final_stage_increment) &&
            std::isfinite(expected) && std::isfinite(scale) && scale > 0.0 &&
            std::isfinite(scaled_error);
        audit.maximum_candidate_formula_scaled_error = std::max(
            audit.maximum_candidate_formula_scaled_error, scaled_error);
      }
    audit.candidate_formula_nontrivial =
        std::isfinite(maximum_formula_increment) &&
        maximum_formula_increment > 0.0;
    audit.candidate_formula_passed = local_valid &&
        audit.candidate_formula_nontrivial &&
        audit.maximum_candidate_formula_scaled_error <=
            64.0 * std::numeric_limits<double>::epsilon();
    return numerics.CollectiveContinue(audit.candidate_formula_passed);
  }

  template <typename Engine>
  static bool AuditInitializedRodas3IgrClosure(
      Engine& engine, const double* supplied_interior,
      const std::size_t supplied_interior_count,
      const double* supplied_true_trace,
      const std::size_t supplied_true_trace_count,
      CudaInitializedRodas3IgrClosureAudit& audit) noexcept
  {
    audit = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    auto& committed = numerics.committed_slot_;
    auto& candidate = numerics.candidate_slot_;
    auto& workspace = numerics.workspace_slot_;
    const std::size_t element_trace_count = shape.elements > 0 &&
            shape.trace_dofs > 0 &&
            static_cast<std::size_t>(shape.elements) <=
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(shape.trace_dofs)
        ? static_cast<std::size_t>(shape.elements) * shape.trace_dofs : 0;
    const int true_scalar_count = shape.trace_components > 0
        ? shape.true_trace_count / shape.trace_components : 0;
    const std::size_t public_true_trace_count = true_scalar_count > 0
        ? static_cast<std::size_t>(conserved_components) *
              static_cast<std::size_t>(true_scalar_count)
        : 0;
    audit.relative_residual_tolerance =
        numerics.true_residual_relative_tolerance_;

    const auto& attempt = numerics.current_attempt_;
    const StateCoordinates coordinates = engine.contract_
        ? engine.contract_->coordinates() : StateCoordinates{};
    audit.lifecycle_zero = engine.initialized_ && engine.contract_ &&
        !engine.pending_.active && !engine.pending_.staging &&
        !engine.pending_.prepared && !engine.pending_.provisional &&
        engine.next_attempt_sequence_ == 0 && attempt.sequence == 0 &&
        attempt.committed_step == 0 && attempt.committed_generation == 0 &&
        attempt.retry_index == 0 && attempt.dt == 0.0 &&
        numerics.current_generation_ == 0 && numerics.ticket_count_ == 0 &&
        committed.interior.generation == 0 &&
        committed.trace.generation == 0 &&
        candidate.interior.generation == 0 && candidate.trace.generation == 0 &&
        coordinates.time == 0.0 && coordinates.step == 0 &&
        coordinates.generation == 0 && coordinates.cursor == 0 &&
        coordinates.dt == 0.0;
    audit.combination_alias_exact =
        numerics.view_.partitioned_base_interior != nullptr &&
        numerics.view_.partitioned_base_interior ==
            numerics.view_.rodas_combination_interior;

    bool local_valid = engine.initialized_ && engine.contract_ &&
        numerics.initialized_ && numerics.rodas3_ && shape.igr_active &&
        shape.borrowed_host_inputs &&
        engine.config_.cuda_engine == generic::CudaEngineKind::tensor_core &&
        shape.interior_components == igr_interior_components &&
        shape.trace_components == igr_trace_components &&
        shape.elements > 0 && shape.scalar_interior_dofs > 0 &&
        shape.scalar_face_dofs > 0 && shape.interior_count > 0 &&
        shape.true_trace_count > 0 && element_trace_count > 0 &&
        true_scalar_count > 0 &&
        shape.true_trace_count % shape.trace_components == 0 &&
        supplied_interior != nullptr && supplied_true_trace != nullptr &&
        supplied_interior_count ==
            static_cast<std::size_t>(conserved_components) *
                static_cast<std::size_t>(
                    numerics.interior_space_->GetVSize() /
                    shape.interior_components) &&
        supplied_true_trace_count == public_true_trace_count &&
        committed.interior.coefficients != nullptr &&
        committed.trace.coefficients != nullptr &&
        candidate.interior.coefficients != nullptr &&
        candidate.trace.coefficients != nullptr &&
        workspace.interior.coefficients == numerics.view_.work_interior &&
        workspace.trace.coefficients == numerics.view_.work_true_trace &&
        committed.interior.count == static_cast<std::size_t>(shape.interior_count) &&
        committed.trace.count == static_cast<std::size_t>(shape.true_trace_count) &&
        candidate.interior.count == committed.interior.count &&
        candidate.trace.count == committed.trace.count &&
        committed.interior.hash == candidate.interior.hash &&
        committed.trace.hash == candidate.trace.hash &&
        numerics.view_.old_interior != nullptr &&
        numerics.view_.current_local_trace != nullptr &&
        std::isfinite(engine.config_.dt) && engine.config_.dt > 0.0 &&
        std::isfinite(audit.relative_residual_tolerance) &&
        audit.relative_residual_tolerance > 0.0 && audit.lifecycle_zero &&
        audit.combination_alias_exact;

    std::array<std::vector<double>, 3> interiors;
    std::array<std::vector<double>, 3> true_traces;
    std::vector<double> local_trace;
    try
    {
      for (auto& values : interiors)
        values.resize(local_valid ? committed.interior.count : 0);
      for (auto& values : true_traces)
        values.resize(local_valid ? committed.trace.count : 0);
      local_trace.resize(local_valid ? element_trace_count : 0);
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const bool staged = CudaTransportPolicy::DeviceToDevice(
            workspace.interior.coefficients,
            committed.interior.coefficients,
            committed.interior.count * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToDevice(
            numerics.view_.old_interior,
            committed.interior.coefficients,
            committed.interior.count * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToDevice(
            workspace.trace.coefficients,
            committed.trace.coefficients,
            committed.trace.count * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        numerics.DistributeCurrentTrace(committed.trace.coefficients);
    if (!numerics.CollectiveContinue(staged)) return false;

    CudaTensorCoreEvidence scratch_tensor =
        numerics.initialization_tensor_core_evidence_;
    const bool assembled = numerics.AssemblePartitionedScalarClosure(
        workspace, engine.config_.dt, audit.global_scalar_residual,
        scratch_tensor);
    if (!numerics.CollectiveContinue(
            assembled && std::isfinite(audit.global_scalar_residual)))
      return false;

    const std::array<const double*, 3> interior_sources{{
        committed.interior.coefficients, candidate.interior.coefficients,
        numerics.view_.initial_current_interior}};
    const std::array<const double*, 3> true_trace_sources{{
        committed.trace.coefficients, candidate.trace.coefficients,
        numerics.view_.initial_current_true_trace}};
    bool copied = true;
    for (std::size_t state = 0; copied && state < interiors.size(); ++state)
      copied = CudaTransportPolicy::DeviceToHost(
          interiors[state].data(), interior_sources[state],
          interiors[state].size() * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    for (std::size_t state = 0; copied && state < true_traces.size(); ++state)
      copied = CudaTransportPolicy::DeviceToHost(
          true_traces[state].data(), true_trace_sources[state],
          true_traces[state].size() * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    copied = copied && CudaTransportPolicy::DeviceToHost(
        local_trace.data(), numerics.view_.current_local_trace,
        local_trace.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(copied)) return false;

    audit.euler_interior_exact = true;
    for (int element = 0;
         element < shape.elements && audit.euler_interior_exact; ++element)
      for (int component = 0;
           component < conserved_components && audit.euler_interior_exact;
           ++component)
        for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
        {
          const int local = component * shape.scalar_interior_dofs + scalar;
          double sign = 1.0;
          const int dof = CudaDecodeSigned(
              numerics.signed_element_interior_[
                  static_cast<std::size_t>(element) * shape.interior_dofs +
                  local],
              sign);
          if (dof < 0 || static_cast<std::size_t>(dof) >=
                  supplied_interior_count)
          {
            audit.euler_interior_exact = false;
            break;
          }
          const double expected = sign * supplied_interior[dof];
          const std::size_t index = static_cast<std::size_t>(element) *
              shape.interior_dofs + local;
          for (const auto& values : interiors)
            if (std::memcmp(&values[index], &expected, sizeof(double)) != 0)
            {
              audit.euler_interior_exact = false;
              break;
            }
          if (!audit.euler_interior_exact) break;
        }

    audit.euler_true_trace_exact = true;
    for (std::size_t index = 0;
         index < supplied_true_trace_count && audit.euler_true_trace_exact;
         ++index)
      for (const auto& values : true_traces)
        if (std::memcmp(
                &values[index], &supplied_true_trace[index],
                sizeof(double)) != 0)
        {
          audit.euler_true_trace_exact = false;
          break;
        }

    std::array<double, 3> local_auxiliary_squared{};
    bool auxiliary_interior_finite = true;
    for (int element = 0;
         element < shape.elements && auxiliary_interior_finite; ++element)
      for (int component = conserved_components;
           component < shape.interior_components; ++component)
        for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
        {
          const std::size_t index = static_cast<std::size_t>(element) *
              shape.interior_dofs +
              component * shape.scalar_interior_dofs + scalar;
          const double value = interiors[0][index];
          auxiliary_interior_finite = std::isfinite(value);
          if (!auxiliary_interior_finite) break;
          local_auxiliary_squared[0] += value * value;
        }

    bool auxiliary_true_trace_finite = true;
    for (int component = conserved_components;
         component < shape.trace_components &&
             auxiliary_true_trace_finite;
         ++component)
      for (int scalar = 0; scalar < true_scalar_count; ++scalar)
      {
        const double value = true_traces[0][
            static_cast<std::size_t>(component) * true_scalar_count + scalar];
        auxiliary_true_trace_finite = std::isfinite(value);
        if (!auxiliary_true_trace_finite) break;
        local_auxiliary_squared[1] += value * value;
      }

    const int face_stride =
        shape.trace_components * shape.scalar_face_dofs;
    bool auxiliary_local_trace_finite = true;
    for (int element = 0;
         element < shape.elements && auxiliary_local_trace_finite; ++element)
      for (int local = 0; local < shape.trace_dofs; ++local)
        if ((local % face_stride) / shape.scalar_face_dofs >=
            conserved_components)
        {
          const double value = local_trace[
              static_cast<std::size_t>(element) * shape.trace_dofs + local];
          auxiliary_local_trace_finite = std::isfinite(value);
          if (!auxiliary_local_trace_finite) break;
          local_auxiliary_squared[2] += value * value;
        }

    std::array<double, 3> global_auxiliary_squared{};
    const bool reduced = MPI_Allreduce(
        local_auxiliary_squared.data(), global_auxiliary_squared.data(),
        static_cast<int>(global_auxiliary_squared.size()), MPI_DOUBLE,
        MPI_SUM, numerics.communicator_) == MPI_SUCCESS;
    for (std::size_t category = 0;
         reduced && category < global_auxiliary_squared.size(); ++category)
      audit.global_auxiliary_norms[category] =
          global_auxiliary_squared[category] >= 0.0
          ? std::sqrt(global_auxiliary_squared[category])
          : std::numeric_limits<double>::infinity();
    audit.auxiliary_interior_finite_nontrivial = reduced &&
        auxiliary_interior_finite &&
        std::isfinite(audit.global_auxiliary_norms[0]) &&
        audit.global_auxiliary_norms[0] > 0.0;
    audit.auxiliary_true_trace_finite_nontrivial = reduced &&
        auxiliary_true_trace_finite &&
        std::isfinite(audit.global_auxiliary_norms[1]) &&
        audit.global_auxiliary_norms[1] > 0.0;
    audit.auxiliary_local_trace_finite_nontrivial = reduced &&
        auxiliary_local_trace_finite &&
        std::isfinite(audit.global_auxiliary_norms[2]) &&
        audit.global_auxiliary_norms[2] > 0.0;
    const double auxiliary_state_norm = std::hypot(
        audit.global_auxiliary_norms[0],
        audit.global_auxiliary_norms[1]);
    audit.global_scalar_relative_residual =
        audit.global_scalar_residual / std::max(auxiliary_state_norm, 1.0);
    const bool residual_exact =
        std::isfinite(audit.global_scalar_relative_residual) &&
        audit.global_scalar_relative_residual <=
            audit.relative_residual_tolerance;
    return numerics.CollectiveContinue(audit.euler_interior_exact &&
        audit.euler_true_trace_exact &&
        audit.auxiliary_interior_finite_nontrivial &&
        audit.auxiliary_true_trace_finite_nontrivial &&
        audit.auxiliary_local_trace_finite_nontrivial && residual_exact &&
        audit.lifecycle_zero && audit.combination_alias_exact);
  }


  template <typename Engine>
  static bool AuditBorrowedAssemblyFusion(
      Engine& engine, CudaBorrowedAssemblyFusionAudit& audit) noexcept
  {
    audit = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        engine.config_.use_generic_numerical_policy &&
        engine.config_.cuda_engine == generic::CudaEngineKind::tensor_core &&
        numerics.cuda_engine_ == generic::CudaEngineKind::tensor_core &&
        shape.borrowed_host_inputs && shape.elements > 0 &&
        shape.interior_dofs > 0 && shape.trace_dofs > 0 &&
        numerics.view_.initial_current_interior != nullptr &&
        numerics.view_.initial_current_true_trace != nullptr &&
        numerics.view_.work_interior != nullptr &&
        numerics.view_.old_interior != nullptr &&
        numerics.view_.work_true_trace != nullptr &&
        numerics.view_.current_local_trace != nullptr &&
        numerics.view_.residual_r != nullptr &&
        numerics.view_.residual_g != nullptr && numerics.view_.a != nullptr &&
        numerics.view_.b != nullptr && numerics.view_.c != nullptr &&
        numerics.view_.d != nullptr && numerics.view_.f != nullptr &&
        numerics.view_.g != nullptr && numerics.view_.status != nullptr;
    const std::size_t elements = local_valid
        ? static_cast<std::size_t>(shape.elements) : 0;
    const std::size_t interior = local_valid
        ? elements * static_cast<std::size_t>(shape.interior_dofs) : 0;
    const std::size_t trace = local_valid
        ? elements * static_cast<std::size_t>(shape.trace_dofs) : 0;
    const std::size_t a_count = local_valid
        ? interior * static_cast<std::size_t>(shape.interior_dofs) : 0;
    const std::size_t b_count = local_valid
        ? interior * static_cast<std::size_t>(shape.trace_dofs) : 0;
    const std::size_t c_count = local_valid
        ? trace * static_cast<std::size_t>(shape.interior_dofs) : 0;
    const std::size_t d_count = local_valid
        ? trace * static_cast<std::size_t>(shape.trace_dofs) : 0;

    struct Products
    {
      std::vector<double> residual_r, residual_g, a, b, c, d, f, g, status;
    };
    Products split, fused;
    const auto resize = [&](Products& products)
    {
      products.residual_r.resize(interior);
      products.residual_g.resize(trace);
      products.a.resize(a_count);
      products.b.resize(b_count);
      products.c.resize(c_count);
      products.d.resize(d_count);
      products.f.resize(interior);
      products.g.resize(trace);
      products.status.resize(elements);
    };
    try { resize(split); resize(fused); }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    // Recreate the first-attempt borrowed state in scratch without changing a
    // committed slot.  Both leaves consume these same immutable inputs.
    const bool seeded = CudaTransportPolicy::DeviceToDevice(
            numerics.view_.work_interior,
            numerics.view_.initial_current_interior,
            interior * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToDevice(
            numerics.view_.old_interior,
            numerics.view_.initial_current_interior,
            interior * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToDevice(
            numerics.view_.work_true_trace,
            numerics.view_.initial_current_true_trace,
            static_cast<std::size_t>(shape.true_trace_count) * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        numerics.DistributeCurrentTrace(
            numerics.view_.initial_current_true_trace);
    if (!numerics.CollectiveContinue(seeded)) return false;

    const auto d2h = [](double* destination, const double* source,
                        const std::size_t count) noexcept
    {
      return CudaTransportPolicy::DeviceToHost(
          destination, source, count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    };
    const auto launch_and_copy = [&](const CudaBorrowedAssemblyKind kind,
                                     Products& products,
                                     std::uint64_t& volume_mma_tiles) noexcept
    {
      const bool launched = CudaLaunchBorrowedAssembly(
              numerics.view_, shape, fixed_dt, kind, volume_mma_tiles) &&
          numerics.StatusPassed();
      if (!numerics.CollectiveContinue(launched)) return false;
      const bool copied = d2h(products.residual_r.data(),
                              numerics.view_.residual_r,
                              products.residual_r.size()) &&
          d2h(products.residual_g.data(), numerics.view_.residual_g,
              products.residual_g.size()) &&
          d2h(products.a.data(), numerics.view_.a, products.a.size()) &&
          d2h(products.b.data(), numerics.view_.b, products.b.size()) &&
          d2h(products.c.data(), numerics.view_.c, products.c.size()) &&
          d2h(products.d.data(), numerics.view_.d, products.d.size()) &&
          d2h(products.f.data(), numerics.view_.f, products.f.size()) &&
          d2h(products.g.data(), numerics.view_.g, products.g.size()) &&
          d2h(products.status.data(), numerics.view_.status,
              products.status.size()) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      return numerics.CollectiveContinue(copied);
    };
    if (!launch_and_copy(
            CudaBorrowedAssemblyKind::ordinary_split, split,
            audit.ordinary_volume_mma_tiles) ||
        !launch_and_copy(
            CudaBorrowedAssemblyKind::tensor_volume, fused,
            audit.tensor_volume_mma_tiles))
      return false;

    const auto planned_volume_work = PlanTensorVolumeMmaWork(
        shape.elements, shape.interior_components,
        shape.scalar_interior_dofs, shape.interior_dofs,
        shape.trace_dofs, shape.volume_points, shape.igr_active);
    const bool exact_volume_work = planned_volume_work.valid &&
        audit.ordinary_volume_mma_tiles == 0 &&
        audit.tensor_volume_mma_tiles == planned_volume_work.mma_tiles;

    const std::array<const std::vector<double>*, 8> split_products{{
        &split.residual_r, &split.residual_g, &split.a, &split.b,
        &split.c, &split.d, &split.f, &split.g}};
    const std::array<const std::vector<double>*, 8> fused_products{{
        &fused.residual_r, &fused.residual_g, &fused.a, &fused.b,
        &fused.c, &fused.d, &fused.f, &fused.g}};
    bool within_tolerance = true;
    // Only residual_r, A, and f contain the reordered volume contraction.
    // Their at-most three products per quadrature point stay under a strict
    // FP64, dimension-scaled roundoff envelope.  Face-only products retain
    // their existing bitwise gate, proving that the zero/face leaves did not
    // change.  This is a test comparison bound, not a solver tolerance.
    const double tensor_tolerance = 1024.0 *
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(1, shape.volume_points));
    for (std::size_t product = 0; product < split_products.size(); ++product)
    {
      const auto& left = *split_products[product];
      const auto& right = *fused_products[product];
      audit.maximum_absolute_errors[product] = 0.0;
      audit.maximum_scaled_errors[product] = 0.0;
      // Face-only blocks retain an exact gate, including signed zero.
      // Volume products still use the existing FP64 roundoff bound below.
      audit.bitwise_equal[product] = left.size() == right.size() &&
          (left.empty() || std::memcmp(left.data(), right.data(),
                                      left.size() * sizeof(double)) == 0);
      bool finite = left.size() == right.size();
      for (std::size_t index = 0; finite && index < left.size(); ++index)
      {
        finite = std::isfinite(left[index]) && std::isfinite(right[index]);
        if (finite)
        {
          const double difference = std::abs(left[index] - right[index]);
          audit.maximum_absolute_errors[product] = std::max(
              audit.maximum_absolute_errors[product],
              difference);
          const double scale = std::max(
              {1.0, std::abs(left[index]), std::abs(right[index])});
          audit.maximum_scaled_errors[product] = std::max(
              audit.maximum_scaled_errors[product], difference / scale);
        }
      }
      if (!finite)
      {
        audit.maximum_absolute_errors[product] =
            std::numeric_limits<double>::infinity();
        audit.maximum_scaled_errors[product] =
            std::numeric_limits<double>::infinity();
      }
      audit.bitwise_equal[product] = audit.bitwise_equal[product] && finite;
      const bool reordered_volume_product =
          product == 0 || product == 2 || product == 6;
      within_tolerance = within_tolerance && finite &&
          (reordered_volume_product
               ? audit.maximum_scaled_errors[product] <= tensor_tolerance
               : audit.bitwise_equal[product]);
    }
    audit.status_bitwise_equal = split.status.size() == fused.status.size() &&
        (split.status.empty() || std::memcmp(
            split.status.data(), fused.status.data(),
            split.status.size() * sizeof(double)) == 0);
    audit.status_all_passed = std::all_of(
        fused.status.begin(), fused.status.end(),
        [](const double value) { return value == 1.0; });
    audit.volume_points = shape.volume_points;
    audit.fp64_comparison_tolerance = tensor_tolerance;
    audit.passes_fp64_threshold = within_tolerance;
    audit.elements = shape.elements;
    audit.interior_dofs = shape.interior_dofs;
    audit.trace_dofs = shape.trace_dofs;
    return exact_volume_work && within_tolerance &&
        audit.status_bitwise_equal &&
        audit.status_all_passed;
  }

  static bool AuditBorrowedAssemblyFusion(
      cuda::GenericEulerHdgCudaSession& session,
      CudaBorrowedAssemblyFusionAudit& audit) noexcept
  {
    if (session.borrowed_rank1_)
      return AuditBorrowedAssemblyFusion(*session.borrowed_rank1_, audit);
    if (session.borrowed_rank2_)
      return AuditBorrowedAssemblyFusion(*session.borrowed_rank2_, audit);
    audit = {};
    return false;
  }

  template <typename Engine>
  static bool AuditConventionalP1Q12TensorVolumeMultiChunk(
      Engine& engine, CudaBorrowedAssemblyFusionAudit& audit) noexcept
  {
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const auto scratch = PlanTensorVolumeScratch(
        shape.interior_dofs, shape.trace_dofs,
        shape.volume_points, false);
    const auto mma_work = PlanTensorVolumeMmaWork(
        shape.elements, shape.interior_components,
        shape.scalar_interior_dofs, shape.interior_dofs,
        shape.trace_dofs, shape.volume_points, false);
    // For conventional p1, q12 is 49 quadrilateral points and the retained
    // solve workspace holds exactly 12 compact 44-double coefficients.  The
    // production launch loop must therefore execute 12+12+12+12+1.
    const bool exact_multi_chunk_shape = shape.order == 1 &&
        !shape.igr_active && shape.volume_assembly_order == 12 &&
        shape.volume_points == 49 && shape.scalar_interior_dofs == 4 &&
        shape.scalar_face_dofs == 2 && shape.interior_dofs == 16 &&
        shape.trace_dofs == 32 && scratch.valid &&
        scratch.workspace_doubles == 528 &&
        scratch.coefficient_stride == 44 &&
        scratch.capacity_points == 12 && scratch.chunk_points == 12 &&
        mma_work.valid && mma_work.contraction_launches == 5 &&
        mma_work.mma_tiles ==
            468U * static_cast<std::uint64_t>(shape.elements);
    return numerics.CollectiveContinue(exact_multi_chunk_shape) &&
        AuditBorrowedAssemblyFusion(engine, audit);
  }

  static bool AuditConventionalP1Q12TensorVolumeMultiChunk(
      cuda::GenericEulerHdgCudaSession& session,
      CudaBorrowedAssemblyFusionAudit& audit) noexcept
  {
    if (session.borrowed_rank1_)
      return AuditConventionalP1Q12TensorVolumeMultiChunk(
          *session.borrowed_rank1_, audit);
    if (session.borrowed_rank2_)
      return AuditConventionalP1Q12TensorVolumeMultiChunk(
          *session.borrowed_rank2_, audit);
    audit = {};
    return false;
  }

  template <typename Engine>
  static bool IluFactorAuthorityInactive(const Engine& engine) noexcept
  {
    const CudaIluAttemptFactorState& state =
        engine.numerics_.ilu_factor_state_;
    const AttemptIdentity empty_attempt{};
    return !state.active && !state.factor_ready &&
        state.policy == CudaIluFactorReusePolicy::not_applicable &&
        ::dg::hdg::runtime::detail::SameAttempt(
            state.attempt, empty_attempt) &&
        state.matrix_object_token == 0 && state.hypre_matrix_token == 0 &&
        state.diag_offsets_token == 0 && state.diag_columns_token == 0 &&
        state.diag_values_token == 0 && state.offd_values_token == 0 &&
        state.offd_offsets_token == 0 && state.offd_columns_token == 0 &&
        state.ilu_handle_token == 0 && state.factor_matrix_token == 0 &&
        state.factor_offsets_token == 0 &&
        state.factor_columns_token == 0 &&
        state.factor_values_token == 0 &&
        state.factor_permutation_token == 0 &&
        state.native_inverse_diagonal_token == 0 &&
        state.jacobi_diagonal_vector_token == 0 &&
        state.jacobi_diagonal_values_token == 0 &&
        state.matrix_rows == 0 && state.factor_rows == 0 &&
        state.factor_nonzeros == 0 &&
        state.diag_nonzeros == 0 && state.offd_nonzeros == 0 &&
        state.factor_generation == 0 && state.last_solve_generation == 0;
  }

  template <typename Engine>
  static bool VerifyPreparedCudaEngine(
      Engine& engine, const generic::CudaEngineKind expected) noexcept
  {
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging)
      return false;
    const auto& evidence = engine.pending_.evidence;
    const auto& tensor = evidence.tensor_core;
    const bool igr_active = engine.numerics_.shape_.igr_active;
    const CudaTracePolicyKind requested_policy =
        engine.numerics_.trace_policy_descriptor_.requested_selection;
    const auto trace_policy = ResolveCudaTracePolicy(
        expected, igr_active, engine.config_.expected_rank_count,
        engine.config_.order, requested_policy);
    return IluFactorAuthorityInactive(engine) &&
        tensor.selected_engine == expected &&
        CudaTracePolicyDescriptorIsExact(trace_policy) &&
        tensor.requested_trace_policy == requested_policy &&
        tensor.resolved_trace_policy == trace_policy.resolved_selection &&
        tensor.trace_policy_rank_count == engine.config_.expected_rank_count &&
        tensor.trace_policy_order == engine.config_.order &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner == trace_policy.trace_preconditioner &&
        CudaPreparedIluAccountingValid(
            tensor, evidence.trace_solves, evidence.linear_iterations,
            expected, igr_active) &&
        CudaPreparedTraceSolverRoutingValid(
            tensor, evidence.trace_solves, expected,
            engine.config_.expected_rank_count) &&
        CudaPreparedTensorVolumeEvidenceValid(
            tensor, engine.numerics_.shape_, expected) &&
        tensor.fallback_count == 0 && evidence.no_fallback &&
        (expected == generic::CudaEngineKind::ordinary
             ? tensor.condensation_launches == 0 && tensor.mma_tiles == 0
             : tensor.hardware_supported &&
                   tensor.condensation_launches > 0 && tensor.mma_tiles > 0);
  }

  static bool VerifyPreparedCudaEngine(
      cuda::GenericEulerHdgCudaSession& session,
      const generic::CudaEngineKind expected) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyPreparedCudaEngine(*session.borrowed_rank1_, expected);
    if (session.borrowed_rank2_)
      return VerifyPreparedCudaEngine(*session.borrowed_rank2_, expected);
    if (session.partitioned_backward_euler_borrowed_rank1_)
      return VerifyPreparedCudaEngine(
          *session.partitioned_backward_euler_borrowed_rank1_, expected);
    if (session.partitioned_backward_euler_borrowed_rank2_)
      return VerifyPreparedCudaEngine(
          *session.partitioned_backward_euler_borrowed_rank2_, expected);
    if (session.rodas3_borrowed_rank1_)
      return VerifyPreparedCudaEngine(
          *session.rodas3_borrowed_rank1_, expected);
    if (session.rodas3_borrowed_rank2_)
      return VerifyPreparedCudaEngine(
          *session.rodas3_borrowed_rank2_, expected);
    return false;
  }

  template <typename Engine>
  static bool AuditPreparedPartitionedClosure(
      Engine& engine, CudaPreparedPartitionedClosureAudit& audit) noexcept
  {
    audit = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const auto candidate = numerics.CandidateSlot();
    auto& workspace = numerics.WorkspaceSlot();
    const bool element_trace_extent_valid = shape.elements > 0 &&
        shape.trace_dofs > 0 &&
        static_cast<std::size_t>(shape.elements) <=
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(shape.trace_dofs);
    const std::size_t element_trace_count = element_trace_extent_valid
        ? static_cast<std::size_t>(shape.elements) * shape.trace_dofs : 0;
    audit.relative_residual_tolerance =
        numerics.true_residual_relative_tolerance_;
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging && engine.pending_.numerical.success &&
        numerics.partitioned_backward_euler_ &&
        shape.borrowed_host_inputs && shape.igr_active &&
        shape.interior_components == igr_interior_components &&
        shape.trace_components == igr_trace_components &&
        shape.scalar_interior_dofs > 0 && shape.scalar_face_dofs > 0 &&
        shape.interior_count > 0 && shape.local_trace_count > 0 &&
        shape.true_trace_count > 0 && element_trace_extent_valid &&
        shape.true_trace_count % shape.trace_components == 0 &&
        workspace.interior.coefficients == numerics.view_.work_interior &&
        workspace.trace.coefficients == numerics.view_.work_true_trace &&
        workspace.interior.count == shape.interior_count &&
        workspace.trace.count == shape.true_trace_count &&
        workspace.interior.hash == engine.pending_.candidate_interior_hash &&
        workspace.trace.hash == engine.pending_.candidate_trace_hash &&
        workspace.interior.local_hash ==
            engine.pending_.candidate_local_interior_hash &&
        workspace.trace.local_hash ==
            engine.pending_.candidate_local_trace_hash &&
        candidate.interior.coefficients != nullptr &&
        candidate.trace.coefficients != nullptr &&
        candidate.interior.count == workspace.interior.count &&
        candidate.trace.count == workspace.trace.count &&
        candidate.interior.hash == engine.pending_.candidate_interior_hash &&
        candidate.trace.hash == engine.pending_.candidate_trace_hash &&
        candidate.interior.local_hash ==
            engine.pending_.candidate_local_interior_hash &&
        candidate.trace.local_hash ==
            engine.pending_.candidate_local_trace_hash &&
        candidate.interior.generation == workspace.interior.generation &&
        candidate.trace.generation == workspace.trace.generation &&
        engine.config_.use_generic_numerical_policy &&
        std::isfinite(audit.relative_residual_tolerance) &&
        audit.relative_residual_tolerance > 0.0 &&
        audit.relative_residual_tolerance ==
            engine.config_.numerical_policy.true_residual_relative_tolerance &&
        numerics.view_.current_local_trace != nullptr;

    std::vector<double> interior;
    std::vector<double> true_trace;
    std::vector<double> local_trace;
    try
    {
      interior.resize(local_valid ? workspace.interior.count : 0);
      true_trace.resize(local_valid ? workspace.trace.count : 0);
      local_trace.resize(local_valid ? element_trace_count : 0);
    }
    catch (...)
    {
      local_valid = false;
    }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const bool copied = CudaTransportPolicy::DeviceToHost(
            interior.data(), workspace.interior.coefficients,
            interior.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            true_trace.data(), workspace.trace.coefficients,
            true_trace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            local_trace.data(), numerics.view_.current_local_trace,
            local_trace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(copied)) return false;

    double local_auxiliary_squared = 0.0;
    audit.auxiliary_interior_finite = true;
    for (int element = 0;
         element < shape.elements && audit.auxiliary_interior_finite;
         ++element)
      for (int local = 0; local < shape.interior_dofs; ++local)
        if (local / shape.scalar_interior_dofs >= conserved_components)
        {
          const std::size_t index = static_cast<std::size_t>(element) *
              shape.interior_dofs + local;
          const double value = interior[index];
          audit.auxiliary_interior_finite = std::isfinite(value);
          if (!audit.auxiliary_interior_finite) break;
          local_auxiliary_squared += value * value;
        }

    const int true_scalar_count =
        shape.true_trace_count / shape.trace_components;
    audit.auxiliary_true_trace_finite = true;
    for (auto iterator =
             true_trace.begin() + conserved_components * true_scalar_count;
         iterator != true_trace.end(); ++iterator)
    {
      audit.auxiliary_true_trace_finite = std::isfinite(*iterator);
      if (!audit.auxiliary_true_trace_finite) break;
      local_auxiliary_squared += *iterator * *iterator;
    }

    const int face_stride = shape.trace_components * shape.scalar_face_dofs;
    audit.auxiliary_local_trace_finite = true;
    for (int element = 0;
         element < shape.elements && audit.auxiliary_local_trace_finite;
         ++element)
      for (int local = 0; local < shape.trace_dofs; ++local)
        if ((local % face_stride) / shape.scalar_face_dofs >=
            conserved_components)
        {
          const std::size_t index = static_cast<std::size_t>(element) *
              shape.trace_dofs + local;
          audit.auxiliary_local_trace_finite =
              std::isfinite(local_trace[index]);
          if (!audit.auxiliary_local_trace_finite) break;
        }
    const bool auxiliary_finite = audit.auxiliary_interior_finite &&
        audit.auxiliary_true_trace_finite &&
        audit.auxiliary_local_trace_finite &&
        std::isfinite(local_auxiliary_squared);
    if (!numerics.CollectiveContinue(auxiliary_finite)) return false;

    // Reassemble the already frozen prepared candidate without changing its
    // state.  This test-only full scalar HDG residual is independent of the
    // condensed trace-system residual reported by Execute.  Normalize it by
    // the global auxiliary coefficient norm and state that denominator
    // explicitly; it is not the trace solver's ||b|| normalization.
    CudaTensorCoreEvidence scratch_tensor =
        engine.pending_.numerical.tensor_core;
    double global_residual = std::numeric_limits<double>::infinity();
    const bool assembled = numerics.AssemblePartitionedScalarClosure(
        workspace, engine.pending_.attempt.dt, global_residual,
        scratch_tensor);
    if (!numerics.CollectiveContinue(
            assembled && std::isfinite(global_residual)))
      return false;
    double global_auxiliary_squared = 0.0;
    const bool auxiliary_reduced = MPI_Allreduce(
        &local_auxiliary_squared, &global_auxiliary_squared, 1, MPI_DOUBLE,
        MPI_SUM, numerics.communicator_) == MPI_SUCCESS;
    audit.global_scalar_residual = global_residual;
    audit.global_auxiliary_norm =
        auxiliary_reduced && global_auxiliary_squared >= 0.0
        ? std::sqrt(global_auxiliary_squared)
        : std::numeric_limits<double>::infinity();
    audit.global_scalar_relative_residual = audit.global_scalar_residual /
        std::max(audit.global_auxiliary_norm, 1.0);
    const bool residual_gate = auxiliary_reduced &&
        std::isfinite(audit.global_scalar_residual) &&
        std::isfinite(audit.global_auxiliary_norm) &&
        std::isfinite(audit.global_scalar_relative_residual) &&
        std::isfinite(audit.relative_residual_tolerance) &&
        audit.relative_residual_tolerance > 0.0 &&
        audit.global_scalar_relative_residual <=
            audit.relative_residual_tolerance;
    return numerics.CollectiveContinue(residual_gate);
  }

  static bool AuditPreparedPartitionedClosure(
      cuda::GenericEulerHdgCudaSession& session,
      CudaPreparedPartitionedClosureAudit& audit) noexcept
  {
    if (session.partitioned_backward_euler_borrowed_rank1_)
      return AuditPreparedPartitionedClosure(
          *session.partitioned_backward_euler_borrowed_rank1_, audit);
    if (session.partitioned_backward_euler_borrowed_rank2_)
      return AuditPreparedPartitionedClosure(
          *session.partitioned_backward_euler_borrowed_rank2_, audit);
    audit = {};
    return false;
  }

  template <typename Engine>
  static bool CorruptHashConsistentPartitionedAuxiliaryCheckpoint(
      Engine& engine, CudaSerializedCheckpointV1& checkpoint) noexcept
  {
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const int ranks = engine.config_.expected_rank_count;
    const bool local_shape_valid = engine.initialized_ &&
        !engine.pending_.active && numerics.partitioned_backward_euler_ &&
        shape.igr_active && shape.elements > 0 &&
        shape.interior_components == igr_interior_components &&
        shape.scalar_interior_dofs > 0 &&
        ranks >= 1 && ranks <= 2 &&
        checkpoint.successful_commit &&
        checkpoint.descriptor.successful_commit &&
        checkpoint.interior_count <= checkpoint.interior.size() &&
        checkpoint.interior_count ==
            static_cast<std::uint64_t>(shape.interior_count);
    if (!numerics.CollectiveContinue(local_shape_valid)) return false;

    // Corrupt sigma (the first auxiliary component), never U4.  Recompute
    // both the rank-local and rank-common hashes so the production import and
    // device rehash gates accept this payload; only the independent scalar
    // HDG closure validation is then allowed to reject it.
    const std::size_t auxiliary_index =
        static_cast<std::size_t>(conserved_components) *
        static_cast<std::size_t>(shape.scalar_interior_dofs);
    const StateHash original_local_hash = checkpoint.local_interior_hash;
    const StateHash original_aggregate_hash =
        checkpoint.descriptor.interior_hash;
    const double original = checkpoint.interior[auxiliary_index];
    const double perturbation = 0.25 * std::max(1.0, std::abs(original));
    checkpoint.interior[auxiliary_index] = original >= 0.0
        ? original + perturbation : original - perturbation;
    const bool local_corruption_valid = std::isfinite(original) &&
        std::isfinite(checkpoint.interior[auxiliary_index]) &&
        checkpoint.interior[auxiliary_index] != original;
    if (!numerics.CollectiveContinue(local_corruption_valid)) return false;

    checkpoint.local_interior_hash = HashHostBytes(
        checkpoint.interior.data(),
        static_cast<std::size_t>(checkpoint.interior_count) * sizeof(double));
    std::array<unsigned char, 2 * 32> hash_pieces{};
    const bool gathered = MPI_Allgather(
        checkpoint.local_interior_hash.bytes.data(), 32, MPI_UNSIGNED_CHAR,
        hash_pieces.data(), 32, MPI_UNSIGNED_CHAR,
        numerics.communicator_) == MPI_SUCCESS;
    if (!numerics.CollectiveContinue(gathered)) return false;
    checkpoint.descriptor.interior_hash = HashHostBytes(
        hash_pieces.data(), static_cast<std::size_t>(ranks) * 32);
    return numerics.CollectiveContinue(
        checkpoint.local_interior_hash != original_local_hash &&
        checkpoint.descriptor.interior_hash != original_aggregate_hash);
  }

  template <typename Engine>
  static bool HashConsistentCheckpointReachedScalarValidation(
      Engine& engine, const CudaSerializedCheckpointV1& checkpoint) noexcept
  {
    auto& numerics = engine.numerics_;
    const auto& workspace = numerics.WorkspaceSlot();
    const auto& committed = numerics.CommittedSlot();
    return numerics.CollectiveContinue(
        engine.initialized_ && !engine.pending_.active &&
        numerics.partitioned_backward_euler_ &&
        workspace.interior.count == checkpoint.interior_count &&
        workspace.trace.count == checkpoint.owned_true_trace_count &&
        workspace.interior.local_hash == checkpoint.local_interior_hash &&
        workspace.trace.local_hash == checkpoint.local_trace_hash &&
        workspace.interior.hash == checkpoint.descriptor.interior_hash &&
        workspace.trace.hash == checkpoint.descriptor.trace_hash &&
        workspace.interior.generation ==
            checkpoint.descriptor.coordinates.generation &&
        workspace.trace.generation ==
            checkpoint.descriptor.coordinates.generation &&
        committed.interior.hash != workspace.interior.hash);
  }

  template <typename Engine>
  static bool AuditPreparedRecovery(
      Engine& engine, CudaPreparedRecoveryAudit& audit) noexcept
  {
    // Test-only D2H capture: check P*A=L*U, every genuine [B f] solve, and the
    // selected ordinary direct-LU or tensor retained-W recovery against an
    // independent pivoted FP64 solve and the original local equation.
    audit = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging && engine.pending_.numerical.success &&
        shape.borrowed_host_inputs && shape.elements > 0 &&
        shape.interior_dofs > 0 && shape.trace_dofs > 0 &&
        numerics.view_.a != nullptr && numerics.view_.b != nullptr &&
        numerics.view_.f != nullptr && numerics.view_.lu != nullptr &&
        numerics.view_.pivots != nullptr &&
        numerics.view_.solve_workspace != nullptr &&
        numerics.view_.trace_increment_local != nullptr &&
        numerics.view_.recovered_interior != nullptr;
    const std::size_t elements = local_valid
        ? static_cast<std::size_t>(shape.elements) : 0;
    const std::size_t ni = local_valid
        ? static_cast<std::size_t>(shape.interior_dofs) : 0;
    const std::size_t nt = local_valid
        ? static_cast<std::size_t>(shape.trace_dofs) : 0;
    std::vector<double> matrix, coupling, forcing, factors, pivots, workspace;
    std::vector<double> trace_increment, recovered;
    std::vector<double> direct_matrix, direct_solution, permuted_matrix;
    try
    {
      matrix.resize(elements * ni * ni);
      coupling.resize(elements * ni * nt);
      forcing.resize(elements * ni);
      factors.resize(elements * ni * ni);
      pivots.resize(elements * ni);
      workspace.resize(elements * ni * (nt + 1));
      trace_increment.resize(elements * nt);
      recovered.resize(elements * ni);
      direct_matrix.resize(ni * ni);
      direct_solution.resize(ni);
      permuted_matrix.resize(ni * ni);
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;
    const bool copied = CudaTransportPolicy::DeviceToHost(
            matrix.data(), numerics.view_.a, matrix.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            coupling.data(), numerics.view_.b,
            coupling.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            forcing.data(), numerics.view_.f,
            forcing.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            factors.data(), numerics.view_.lu,
            factors.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            pivots.data(), numerics.view_.pivots,
            pivots.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            workspace.data(), numerics.view_.solve_workspace,
            workspace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            trace_increment.data(), numerics.view_.trace_increment_local,
            trace_increment.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            recovered.data(), numerics.view_.recovered_interior,
            recovered.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(copied)) return false;

    audit.elements = shape.elements;
    audit.interior_dofs = shape.interior_dofs;
    audit.trace_dofs = shape.trace_dofs;
    audit.maximum_direct_scaled_error = 0.0;
    audit.maximum_workspace_scaled_error = 0.0;
    audit.maximum_factor_scaled_residual = 0.0;
    audit.maximum_workspace_solve_scaled_residual = 0.0;
    audit.maximum_scaled_residual = 0.0;
    // A dimension-scaled roundoff envelope; this does not modify any solver or
    // lifecycle tolerance and remains below the legacy 3e-10 oracle gate.
    audit.tolerance = 4096.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(1, shape.interior_dofs + shape.trace_dofs));
    bool finite = true;
    for (std::size_t element = 0; finite && element < elements; ++element)
    {
      const std::size_t matrix_offset = element * ni * ni;
      const std::size_t coupling_offset = element * ni * nt;
      const std::size_t interior_offset = element * ni;
      const std::size_t trace_offset = element * nt;
      const std::size_t workspace_offset = element * ni * (nt + 1);
      const std::size_t factor_offset = element * ni * ni;
      const std::size_t pivot_offset = element * ni;
      std::copy_n(matrix.data() + matrix_offset, ni * ni,
                  permuted_matrix.data());
      for (std::size_t column = 0; finite && column < ni; ++column)
      {
        const double encoded_pivot = pivots[pivot_offset + column];
        const bool pivot_representable = std::isfinite(encoded_pivot) &&
            encoded_pivot >= static_cast<double>(column) &&
            encoded_pivot < static_cast<double>(ni);
        const int pivot = pivot_representable
            ? static_cast<int>(encoded_pivot) : -1;
        finite = pivot_representable &&
            encoded_pivot == static_cast<double>(pivot) &&
            pivot >= static_cast<int>(column);
        if (finite && pivot != static_cast<int>(column))
          for (std::size_t entry = 0; entry < ni; ++entry)
            std::swap(permuted_matrix[column * ni + entry],
                      permuted_matrix[static_cast<std::size_t>(pivot) * ni +
                                      entry]);
      }
      for (std::size_t row = 0; finite && row < ni; ++row)
        for (std::size_t column = 0; finite && column < ni; ++column)
        {
          const long double reference = static_cast<long double>(
              permuted_matrix[row * ni + column]);
          long double reconstruction = 0.0L;
          long double factor_scale = std::abs(reference);
          const std::size_t last = std::min(row, column);
          for (std::size_t inner = 0; inner <= last; ++inner)
          {
            const double lower = inner == row ? 1.0 :
                factors[factor_offset + row * ni + inner];
            const double upper =
                factors[factor_offset + inner * ni + column];
            finite = finite && std::isfinite(lower) && std::isfinite(upper);
            const long double term =
                static_cast<long double>(lower) * upper;
            reconstruction += term;
            factor_scale += std::abs(term);
          }
          const double scaled_factor_residual = static_cast<double>(
              std::abs(reference - reconstruction) /
              std::max(1.0L, factor_scale));
          finite = finite && std::isfinite(scaled_factor_residual);
          audit.maximum_factor_scaled_residual = std::max(
              audit.maximum_factor_scaled_residual,
              scaled_factor_residual);
        }
      for (std::size_t rhs = 0; finite && rhs < nt + 1; ++rhs)
        for (std::size_t row = 0; finite && row < ni; ++row)
        {
          const double reference = rhs < nt
              ? coupling[coupling_offset + row * nt + rhs]
              : forcing[interior_offset + row];
          long double residual = -static_cast<long double>(reference);
          long double residual_scale = std::abs(
              static_cast<long double>(reference));
          for (std::size_t column = 0; column < ni; ++column)
          {
            const long double term = static_cast<long double>(
                matrix[matrix_offset + row * ni + column]) *
                workspace[workspace_offset + column * (nt + 1) + rhs];
            residual += term;
            residual_scale += std::abs(term);
          }
          const double scaled_solve_residual = static_cast<double>(
              std::abs(residual) / std::max(1.0L, residual_scale));
          finite = finite && std::isfinite(scaled_solve_residual);
          audit.maximum_workspace_solve_scaled_residual = std::max(
              audit.maximum_workspace_solve_scaled_residual,
              scaled_solve_residual);
        }
      if (!finite) break;
      std::copy_n(matrix.data() + matrix_offset, ni * ni,
                  direct_matrix.data());
      for (std::size_t row = 0; row < ni; ++row)
      {
        double value = forcing[interior_offset + row];
        for (std::size_t column = 0; column < nt; ++column)
          value -= coupling[coupling_offset + row * nt + column] *
              trace_increment[trace_offset + column];
        direct_solution[row] = value;
      }
      if (!SolveRecoveryReference(direct_matrix, direct_solution,
                                  shape.interior_dofs))
      {
        finite = false;
        break;
      }
      for (std::size_t row = 0; row < ni; ++row)
      {
        double workspace_value =
            workspace[workspace_offset + row * (nt + 1) + nt];
        for (std::size_t column = 0; column < nt; ++column)
          workspace_value -=
              workspace[workspace_offset + row * (nt + 1) + column] *
              trace_increment[trace_offset + column];
        const double production = recovered[interior_offset + row];
        finite = finite && std::isfinite(workspace_value) &&
            std::isfinite(production) && std::isfinite(direct_solution[row]);
        if (!finite) break;
        const double scale = std::max(
            {1.0, std::abs(production), std::abs(workspace_value),
             std::abs(direct_solution[row])});
        audit.maximum_direct_scaled_error = std::max(
            audit.maximum_direct_scaled_error,
            std::abs(production - direct_solution[row]) / scale);
        audit.maximum_workspace_scaled_error = std::max(
            audit.maximum_workspace_scaled_error,
            std::abs(production - workspace_value) / scale);

        long double residual = -static_cast<long double>(
            forcing[interior_offset + row]);
        long double residual_scale = std::abs(static_cast<long double>(
            forcing[interior_offset + row]));
        for (std::size_t column = 0; column < ni; ++column)
        {
          const long double term = static_cast<long double>(
              matrix[matrix_offset + row * ni + column]) *
              recovered[interior_offset + column];
          residual += term;
          residual_scale += std::abs(term);
        }
        for (std::size_t column = 0; column < nt; ++column)
        {
          const long double term = static_cast<long double>(
              coupling[coupling_offset + row * nt + column]) *
              trace_increment[trace_offset + column];
          residual += term;
          residual_scale += std::abs(term);
        }
        const double scaled_residual = static_cast<double>(
            std::abs(residual) / std::max(1.0L, residual_scale));
        finite = finite && std::isfinite(scaled_residual);
        audit.maximum_scaled_residual = std::max(
            audit.maximum_scaled_residual, scaled_residual);
      }
    }
    return finite &&
        audit.maximum_direct_scaled_error <= audit.tolerance &&
        audit.maximum_workspace_scaled_error <= audit.tolerance &&
        audit.maximum_factor_scaled_residual <= audit.tolerance &&
        audit.maximum_workspace_solve_scaled_residual <= audit.tolerance &&
        audit.maximum_scaled_residual <= audit.tolerance;
  }

  // Compare the two production condensation leaves from one coherent
  // prepared tensor candidate.  The relaunches reuse exactly the same
  // element-local A/B/C/D/F/G products, so nonlinear trajectory differences
  // cannot obscure Schur/RHS agreement.  The tensor result is restored before
  // the ordinary prepared/commit qualification resumes.
  template <typename Engine>
  static bool CapturePreparedLocalCondensationPair(
      Engine& engine,
      CudaPreparedLocalCondensationCapture& capture) noexcept
  {
    capture = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const auto& tensor = engine.pending_.evidence.tensor_core;
    const auto trace_policy = ResolveCudaTracePolicy(
        generic::CudaEngineKind::tensor_core, shape.igr_active,
        engine.config_.expected_rank_count, engine.config_.order,
        CudaTracePolicyKind::engine_default);
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging && engine.pending_.numerical.success &&
        engine.config_.cuda_engine == generic::CudaEngineKind::tensor_core &&
        numerics.cuda_engine_ == generic::CudaEngineKind::tensor_core &&
        shape.borrowed_host_inputs && shape.elements > 0 &&
        shape.interior_dofs > 0 && shape.trace_dofs > 0 &&
        numerics.view_.schur != nullptr &&
        numerics.view_.condensed_rhs != nullptr &&
        tensor.selected_engine == generic::CudaEngineKind::tensor_core &&
        CudaTracePolicyDescriptorIsExact(trace_policy) &&
        tensor.requested_trace_policy ==
            CudaTracePolicyKind::engine_default &&
        tensor.resolved_trace_policy == trace_policy.resolved_selection &&
        tensor.trace_policy_rank_count == engine.config_.expected_rank_count &&
        tensor.trace_policy_order == engine.config_.order &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner == trace_policy.trace_preconditioner &&
        tensor.fallback_count == 0 &&
        engine.pending_.evidence.no_fallback;
    const std::size_t schur_count = local_valid
        ? static_cast<std::size_t>(shape.elements) * shape.trace_dofs *
              shape.trace_dofs
        : 0;
    const std::size_t rhs_count = local_valid
        ? static_cast<std::size_t>(shape.elements) * shape.trace_dofs
        : 0;
    try
    {
      capture.tensor_schur.resize(schur_count);
      capture.ordinary_schur.resize(schur_count);
      capture.tensor_rhs.resize(rhs_count);
      capture.ordinary_rhs.resize(rhs_count);
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const auto copy_products = [&](std::vector<double>& schur,
                                   std::vector<double>& rhs) noexcept
    {
      return CudaTransportPolicy::DeviceToHost(
                 schur.data(), numerics.view_.schur,
                 schur.size() * sizeof(double),
                 CudaTransferCategory::generic_host_bridge,
                 CudaTransferSite::generic_host_export) &&
          CudaTransportPolicy::DeviceToHost(
                 rhs.data(), numerics.view_.condensed_rhs,
                 rhs.size() * sizeof(double),
                 CudaTransferCategory::generic_host_bridge,
                 CudaTransferSite::generic_host_export) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
    };

    const bool tensor_copied = numerics.CollectiveContinue(
        copy_products(capture.tensor_schur, capture.tensor_rhs));

    std::uint64_t ordinary_tiles = 0;
    const bool ordinary_launched = numerics.CollectiveContinue(
        CudaLaunchCondensation(
            numerics.view_, shape, generic::CudaEngineKind::ordinary,
            ordinary_tiles) &&
        numerics.StatusPassed());
    const bool ordinary_copied = ordinary_launched &&
        numerics.CollectiveContinue(
            copy_products(capture.ordinary_schur, capture.ordinary_rhs));

    std::uint64_t restored_tensor_tiles = 0;
    const bool tensor_restored = numerics.CollectiveContinue(
        CudaLaunchCondensation(
            numerics.view_, shape, generic::CudaEngineKind::tensor_core,
            restored_tensor_tiles) &&
        numerics.StatusPassed());

    capture.elements = shape.elements;
    capture.interior_dofs = shape.interior_dofs;
    capture.trace_dofs = shape.trace_dofs;
    capture.tensor_mma_tiles = restored_tensor_tiles;
    return tensor_copied && ordinary_launched && ordinary_copied &&
        tensor_restored && ordinary_tiles == 0 &&
        restored_tensor_tiles > 0;
  }

  template <typename Engine>
  static bool AuditNativeRankOneTrueRhsRefresh(Engine& engine) noexcept
  {
    // Build one coherent current operator/RHS pair, refresh only that same RHS,
    // and prove the independently retained ParCSR values remain byte-exact.
    // This is a private qualification seam, not a production diagnostic API.
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging && engine.pending_.numerical.success &&
        engine.config_.expected_rank_count == 1 && shape.igr_active &&
        engine.config_.cuda_engine == generic::CudaEngineKind::tensor_core &&
        numerics.cuda_engine_ == generic::CudaEngineKind::tensor_core &&
        numerics.trace_solver_kind_ == CudaTraceSolverKind::native_rank1_fp64 &&
        CudaTracePolicyUsesNativeElementMatrixFreeOperator(
            numerics.trace_policy_descriptor_.resolved_selection) &&
        numerics.true_device_.diag_nonzeros > 0 &&
        numerics.true_device_.offd_nonzeros == 0 &&
        numerics.true_device_.diag_values != nullptr &&
        numerics.view_.true_rhs_scratch != nullptr;
    std::vector<double> matrix_before, matrix_after, rhs_before, rhs_after;
    try
    {
      const std::size_t matrix_values = local_valid
          ? static_cast<std::size_t>(numerics.true_device_.diag_nonzeros) : 0;
      const std::size_t rhs_values = local_valid
          ? static_cast<std::size_t>(shape.true_trace_count) : 0;
      matrix_before.resize(matrix_values);
      matrix_after.resize(matrix_values);
      rhs_before.resize(rhs_values);
      rhs_after.resize(rhs_values);
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;
    const auto copy = [](double* destination, const double* source,
                         const std::size_t count) noexcept
    {
      return CudaTransportPolicy::DeviceToHost(
          destination, source, count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    };
    const bool assembled = numerics.AssembleTrueSystem() &&
        copy(matrix_before.data(), numerics.true_device_.diag_values,
             matrix_before.size()) &&
        copy(rhs_before.data(), numerics.view_.true_rhs_scratch,
             rhs_before.size()) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(assembled) ||
        !numerics.RefreshRetainedTrueRhs())
      return false;
    const bool copied =
        copy(matrix_after.data(), numerics.true_device_.diag_values,
             matrix_after.size()) &&
        copy(rhs_after.data(), numerics.view_.true_rhs_scratch,
             rhs_after.size()) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    return numerics.CollectiveContinue(copied &&
        matrix_before == matrix_after && rhs_before == rhs_after);
  }

  template <typename Engine>
  static bool VerifyAssemblerJacobian(
      Engine& engine, const double relative_tolerance,
      std::array<double, 6>& relative_errors,
      const bool differentiate_nonlinearities = false) noexcept
  {
    relative_errors.fill(std::numeric_limits<double>::infinity());
    return engine.initialized_ && !engine.pending_.active &&
        engine.config_.use_generic_numerical_policy &&
        engine.numerics_.AuditBorrowedEulerAssemblerJacobian(
            relative_tolerance, relative_errors,
            differentiate_nonlinearities);
  }

  static bool VerifyAssemblerJacobian(
      cuda::GenericEulerHdgCudaSession& session,
      const double relative_tolerance,
      std::array<double, 6>& relative_errors,
      const bool differentiate_nonlinearities = false) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyAssemblerJacobian(
          *session.borrowed_rank1_, relative_tolerance, relative_errors,
          differentiate_nonlinearities);
    if (session.borrowed_rank2_)
      return VerifyAssemblerJacobian(
          *session.borrowed_rank2_, relative_tolerance, relative_errors,
          differentiate_nonlinearities);
    if (session.rodas3_borrowed_rank1_)
      return VerifyAssemblerJacobian(
          *session.rodas3_borrowed_rank1_, relative_tolerance,
          relative_errors, differentiate_nonlinearities);
    relative_errors.fill(std::numeric_limits<double>::infinity());
    return false;
  }

  // Independent current-candidate LLF oracle.  It proves that production
  // assembly agrees with tau=max(lambda_n(U),lambda_n(Uhat)), refreshes that
  // value when either side changes, applies no historical fixed-tau floor,
  // and uses the refreshed coefficient as frozen data in A/B/C/D.
  template <typename Engine>
  static bool VerifyCurrentCandidateLlfPolicy(
      Engine& engine, std::array<double, 7>& evidence) noexcept
  {
    evidence.fill(std::numeric_limits<double>::infinity());
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const auto committed = numerics.CommittedSlot();
    const std::size_t tau_count = static_cast<std::size_t>(shape.elements) *
        faces_per_element * shape.face_points;
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        engine.config_.use_generic_numerical_policy &&
        engine.config_.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::none &&
        shape.borrowed_host_inputs && !shape.igr_active &&
        shape.interior_components == conserved_components &&
        shape.trace_components == conserved_components &&
        numerics.view_.borrowed_lagged_llf_tau != nullptr && tau_count != 0 &&
        committed.interior.coefficients != nullptr &&
        committed.trace.coefficients != nullptr;
    if (!numerics.CollectiveContinue(local_valid)) return false;

    std::vector<double> base_interior, base_true_trace;
    std::vector<double> shifted_interior, shifted_true_trace;
    std::vector<double> local_trace;
    std::vector<double> base_tau, interior_tau, trace_tau;
    try
    {
      base_interior.resize(committed.interior.count);
      base_true_trace.resize(committed.trace.count);
      shifted_interior.resize(committed.interior.count);
      shifted_true_trace.resize(committed.trace.count);
      local_trace.resize(static_cast<std::size_t>(shape.elements) *
                         shape.trace_dofs);
      base_tau.resize(tau_count);
      interior_tau.resize(tau_count);
      trace_tau.resize(tau_count);
    }
    catch (...) { local_valid = false; }
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const auto d2h = [](double* destination, const double* source,
                        const std::size_t count) noexcept
    {
      return CudaTransportPolicy::DeviceToHost(
          destination, source, count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    };
    const auto h2d = [](double* destination, const double* source,
                        const std::size_t count) noexcept
    {
      return CudaTransportPolicy::HostToDevice(
          destination, source, count * sizeof(double),
          CudaTransferCategory::generic_host_bridge,
          CudaTransferSite::generic_host_export);
    };
    local_valid = d2h(base_interior.data(),
                      committed.interior.coefficients,
                      base_interior.size()) &&
        d2h(base_true_trace.data(), committed.trace.coefficients,
            base_true_trace.size()) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(local_valid)) return false;

    std::copy(base_interior.begin(), base_interior.end(),
              shifted_interior.begin());
    std::copy(base_true_trace.begin(), base_true_trace.end(),
              shifted_true_trace.begin());
    local_valid = h2d(numerics.view_.old_interior,
                      base_interior.data(), base_interior.size()) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(local_valid)) return false;
    const auto galilean_shift = [](double& rho, double& momentum_x,
                                   double& momentum_y, double& energy,
                                   const double shift_x,
                                   const double shift_y) noexcept
    {
      const double old_momentum_x = momentum_x;
      const double old_momentum_y = momentum_y;
      momentum_x += rho * shift_x;
      momentum_y += rho * shift_y;
      energy += shift_x * old_momentum_x + shift_y * old_momentum_y +
          0.5 * rho * (shift_x * shift_x + shift_y * shift_y);
    };
    for (int element = 0; element < shape.elements; ++element)
      for (int scalar = 0; scalar < shape.scalar_interior_dofs; ++scalar)
      {
        const std::size_t offset = static_cast<std::size_t>(element) *
            shape.interior_dofs;
        galilean_shift(
            shifted_interior[offset + scalar],
            shifted_interior[offset + shape.scalar_interior_dofs + scalar],
            shifted_interior[offset + 2 * shape.scalar_interior_dofs + scalar],
            shifted_interior[offset + 3 * shape.scalar_interior_dofs + scalar],
            0.78, 0.34);
      }
    const int true_trace_scalars = shape.true_trace_count /
        conserved_components;
    local_valid = true_trace_scalars * conserved_components ==
        shape.true_trace_count;
    for (int scalar = 0; local_valid && scalar < true_trace_scalars; ++scalar)
      galilean_shift(
          shifted_true_trace[scalar],
          shifted_true_trace[true_trace_scalars + scalar],
          shifted_true_trace[2 * true_trace_scalars + scalar],
          shifted_true_trace[3 * true_trace_scalars + scalar],
          -0.73, 0.41);
    if (!numerics.CollectiveContinue(local_valid)) return false;

    const auto install = [&](const std::vector<double>& interior,
                             const std::vector<double>& trace) noexcept
    {
      const bool uploaded = h2d(numerics.view_.work_interior,
                                interior.data(), interior.size()) &&
          h2d(numerics.view_.work_true_trace,
              trace.data(), trace.size()) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      if (!numerics.CollectiveContinue(uploaded)) return false;
      return numerics.DistributeCurrentTrace(
          numerics.view_.work_true_trace);
    };
    const auto local_speed = [](const double* state, const double nx,
                                const double ny, double& speed) noexcept
    {
      if (!(state[0] > 0.0) || !std::isfinite(state[0]) ||
          !std::isfinite(state[1]) || !std::isfinite(state[2]) ||
          !std::isfinite(state[3]) || !std::isfinite(nx) ||
          !std::isfinite(ny) || (nx == 0.0 && ny == 0.0))
        return false;
      const double pressure = 0.4 *
          (state[3] - 0.5 *
              (state[1] * state[1] + state[2] * state[2]) / state[0]);
      const double sound_squared = 1.4 * pressure / state[0];
      if (!(pressure > 0.0) || !(sound_squared > 0.0) ||
          !std::isfinite(pressure) || !std::isfinite(sound_squared))
        return false;
      speed = std::abs((state[1] * nx + state[2] * ny) / state[0]) +
          std::sqrt(sound_squared);
      return std::isfinite(speed) && speed > 0.0;
    };
    const auto capture_tau = [&](const std::vector<double>& interior,
                                 const std::vector<double>& trace,
                                 std::vector<double>& tau,
                                 double& maximum_error) noexcept
    {
      maximum_error = 0.0;
      if (!install(interior, trace)) return false;
      const bool launched = CudaLaunchFreezeBorrowedCurrentLlfTau(
              numerics.view_, shape) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::scalar_status) &&
          numerics.StatusPassed();
      if (!numerics.CollectiveContinue(launched)) return false;
      const bool copied = d2h(tau.data(),
                              numerics.view_.borrowed_lagged_llf_tau,
                              tau.size()) &&
          d2h(local_trace.data(), numerics.view_.current_local_trace,
              local_trace.size()) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      if (!numerics.CollectiveContinue(copied)) return false;
      bool valid = true;
      for (int element = 0; valid && element < shape.elements; ++element)
        for (int face = 0; valid && face < faces_per_element; ++face)
          for (int q = 0; valid && q < shape.face_points; ++q)
          {
            const int qp = (element * faces_per_element + face) *
                shape.face_points + q;
            const std::size_t interior_offset =
                static_cast<std::size_t>(element) * shape.interior_dofs;
            const std::size_t trace_offset =
                static_cast<std::size_t>(element) * shape.trace_dofs +
                face * shape.scalar_face_dofs * conserved_components;
            double interior_state[conserved_components]{};
            double trace_state[conserved_components]{};
            for (int scalar = 0;
                 scalar < shape.scalar_interior_dofs; ++scalar)
              for (int component = 0;
                   component < conserved_components; ++component)
                interior_state[component] += numerics.host_face_interior_shape_[
                    static_cast<std::size_t>(qp) *
                        shape.scalar_interior_dofs + scalar] *
                    interior[interior_offset + scalar +
                        shape.scalar_interior_dofs * component];
            for (int scalar = 0; scalar < shape.scalar_face_dofs; ++scalar)
              for (int component = 0;
                   component < conserved_components; ++component)
                trace_state[component] += numerics.host_face_trace_shape_[
                    static_cast<std::size_t>(qp) *
                        shape.scalar_face_dofs + scalar] *
                    local_trace[trace_offset + scalar +
                        shape.scalar_face_dofs * component];
            const double nx = numerics.host_face_normal_[
                static_cast<std::size_t>(qp) * dimension];
            const double ny = numerics.host_face_normal_[
                static_cast<std::size_t>(qp) * dimension + 1];
            double interior_speed = 0.0, trace_speed = 0.0;
            valid = local_speed(interior_state, nx, ny, interior_speed) &&
                local_speed(trace_state, nx, ny, trace_speed);
            const double expected = std::max(interior_speed, trace_speed);
            valid = valid && std::isfinite(tau[qp]) && tau[qp] > 0.0;
            maximum_error = std::max(
                maximum_error, std::abs(tau[qp] - expected));
          }
      return numerics.CollectiveContinue(valid);
    };
    const auto assemble = [&](const std::vector<double>& interior,
                              const std::vector<double>& trace,
                              const bool frozen,
                              std::vector<double>& residual_r,
                              std::vector<double>& residual_g) noexcept
    {
      if (!install(interior, trace)) return false;
      std::uint64_t ignored_volume_mma_tiles = 0;
      const bool launched = (frozen
              ? CudaLaunchFrozenBorrowedLlfAssembly(
                    numerics.view_, shape, fixed_dt)
              : CudaLaunchAssembly(
                    numerics.view_, shape, fixed_dt,
                    numerics.cuda_engine_, ignored_volume_mma_tiles)) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::scalar_status) &&
          numerics.StatusPassed();
      if (!numerics.CollectiveContinue(launched)) return false;
      bool ready = true;
      try
      {
        residual_r.resize(static_cast<std::size_t>(shape.interior_count));
        residual_g.resize(static_cast<std::size_t>(shape.elements) *
                          shape.trace_dofs);
      }
      catch (...) { ready = false; }
      if (!numerics.CollectiveContinue(ready)) return false;
      const bool copied = d2h(residual_r.data(), numerics.view_.residual_r,
                              residual_r.size()) &&
          d2h(residual_g.data(), numerics.view_.residual_g,
              residual_g.size()) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      return numerics.CollectiveContinue(copied);
    };
    const auto upload_tau = [&](const std::vector<double>& tau) noexcept
    {
      const bool uploaded = h2d(numerics.view_.borrowed_lagged_llf_tau,
                                tau.data(), tau.size()) &&
          CudaTransportPolicy::Synchronize(
              CudaTransferSite::generic_host_export);
      return numerics.CollectiveContinue(uploaded);
    };

    double base_error = 0.0, interior_error = 0.0, trace_error = 0.0;
    std::vector<double> interior_stale_r, interior_stale_g;
    std::vector<double> interior_current_r, interior_current_g;
    std::vector<double> interior_frozen_r, interior_frozen_g;
    std::vector<double> trace_stale_r, trace_stale_g;
    std::vector<double> trace_current_r, trace_current_g;
    std::vector<double> trace_frozen_r, trace_frozen_g;
    const bool captured =
        capture_tau(base_interior, base_true_trace, base_tau, base_error) &&
        upload_tau(base_tau) &&
        assemble(shifted_interior, base_true_trace, true,
                 interior_stale_r, interior_stale_g) &&
        capture_tau(shifted_interior, base_true_trace, interior_tau,
                    interior_error) &&
        assemble(shifted_interior, base_true_trace, false,
                 interior_current_r, interior_current_g) &&
        assemble(shifted_interior, base_true_trace, true,
                 interior_frozen_r, interior_frozen_g) &&
        upload_tau(base_tau) &&
        assemble(base_interior, shifted_true_trace, true,
                 trace_stale_r, trace_stale_g) &&
        capture_tau(base_interior, shifted_true_trace, trace_tau,
                    trace_error) &&
        assemble(base_interior, shifted_true_trace, false,
                 trace_current_r, trace_current_g) &&
        assemble(base_interior, shifted_true_trace, true,
                 trace_frozen_r, trace_frozen_g);
    const bool restore_uploaded = h2d(
            numerics.view_.work_interior, base_interior.data(),
            base_interior.size()) &&
        h2d(numerics.view_.work_true_trace, base_true_trace.data(),
            base_true_trace.size()) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    const bool restored = numerics.CollectiveContinue(restore_uploaded) &&
        numerics.DistributeCurrentTrace(numerics.view_.work_true_trace);
    if (!captured || !restored) return false;

    double local_base_maximum = 0.0;
    double local_interior_tau_change = 0.0;
    double local_trace_tau_change = 0.0;
    for (std::size_t index = 0; index < base_tau.size(); ++index)
    {
      local_base_maximum = std::max(local_base_maximum, base_tau[index]);
      local_interior_tau_change = std::max(
          local_interior_tau_change,
          std::abs(interior_tau[index] - base_tau[index]));
      local_trace_tau_change = std::max(
          local_trace_tau_change,
          std::abs(trace_tau[index] - base_tau[index]));
    }
    const auto squared_pair = [](const std::vector<double>& first_r,
                                 const std::vector<double>& first_g,
                                 const std::vector<double>& second_r,
                                 const std::vector<double>& second_g,
                                 double& difference,
                                 double& reference) noexcept
    {
      difference = 0.0;
      reference = 0.0;
      if (first_r.size() != second_r.size() ||
          first_g.size() != second_g.size()) return false;
      for (std::size_t index = 0; index < first_r.size(); ++index)
      {
        const double delta = first_r[index] - second_r[index];
        difference += delta * delta;
        reference += first_r[index] * first_r[index];
      }
      for (std::size_t index = 0; index < first_g.size(); ++index)
      {
        const double delta = first_g[index] - second_g[index];
        difference += delta * delta;
        reference += first_g[index] * first_g[index];
      }
      return std::isfinite(difference) && std::isfinite(reference);
    };
    std::array<double, 8> local_squares{};
    local_valid = squared_pair(
            interior_current_r, interior_current_g,
            interior_stale_r, interior_stale_g,
            local_squares[0], local_squares[1]) &&
        squared_pair(trace_current_r, trace_current_g,
                     trace_stale_r, trace_stale_g,
                     local_squares[2], local_squares[3]) &&
        squared_pair(interior_current_r, interior_current_g,
                     interior_frozen_r, interior_frozen_g,
                     local_squares[4], local_squares[5]) &&
        squared_pair(trace_current_r, trace_current_g,
                     trace_frozen_r, trace_frozen_g,
                     local_squares[6], local_squares[7]);
    if (!numerics.CollectiveContinue(local_valid)) return false;

    std::array<double, 4> local_maxima{{
        local_base_maximum,
        std::max({base_error, interior_error, trace_error}),
        local_interior_tau_change,
        local_trace_tau_change}};
    std::array<double, 4> global_maxima{};
    std::array<double, 8> global_squares{};
    const bool maxima_reduced = MPI_Allreduce(
        local_maxima.data(), global_maxima.data(),
        static_cast<int>(local_maxima.size()), MPI_DOUBLE, MPI_MAX,
        numerics.communicator_) == MPI_SUCCESS;
    const bool squares_reduced = MPI_Allreduce(
        local_squares.data(), global_squares.data(),
        static_cast<int>(local_squares.size()), MPI_DOUBLE, MPI_SUM,
        numerics.communicator_) == MPI_SUCCESS;
    if (!numerics.CollectiveContinue(maxima_reduced && squares_reduced))
      return false;
    const auto relative_norm = [](const double difference,
                                  const double reference) noexcept
    {
      return std::sqrt(difference) /
          std::max(1.0, std::sqrt(reference));
    };
    evidence[0] = global_maxima[0];
    evidence[1] = global_maxima[1];
    evidence[2] = global_maxima[2];
    evidence[3] = global_maxima[3];
    evidence[4] = relative_norm(global_squares[0], global_squares[1]);
    evidence[5] = relative_norm(global_squares[2], global_squares[3]);
    evidence[6] = std::max(
        relative_norm(global_squares[4], global_squares[5]),
        relative_norm(global_squares[6], global_squares[7]));
    constexpr double historical_fixed_tau = 2.5;
    const double arithmetic_tolerance = 4096.0 *
        std::numeric_limits<double>::epsilon() *
        (1.0 + evidence[0]);
    return numerics.CollectiveContinue(
        std::all_of(evidence.begin(), evidence.end(),
                    [](const double value) { return std::isfinite(value); }) &&
        evidence[0] < historical_fixed_tau &&
        evidence[1] <= arithmetic_tolerance &&
        evidence[2] > 1.0e-6 && evidence[3] > 1.0e-6 &&
        evidence[4] > 1.0e-10 && evidence[5] > 1.0e-10 &&
        evidence[6] <= arithmetic_tolerance);
  }

  static bool VerifyCurrentCandidateLlfPolicy(
      cuda::GenericEulerHdgCudaSession& session,
      std::array<double, 7>& evidence) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyCurrentCandidateLlfPolicy(
          *session.borrowed_rank1_, evidence);
    if (session.borrowed_rank2_)
      return VerifyCurrentCandidateLlfPolicy(
          *session.borrowed_rank2_, evidence);
    evidence.fill(std::numeric_limits<double>::infinity());
    return false;
  }

  template <typename Engine>
  static bool VerifyPreparedWorkAccounting(
      Engine& engine, const CudaStepEvidence& evidence,
      const bool require_linear_activity,
      std::array<int, 6>& counts) noexcept
  {
    counts.fill(-1);
    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    counts = {{numerical.nonlinear_updates,
               numerical.trace_solves,
               numerical.linear_iterations,
               numerical.line_search_trials,
               numerical.rejected_line_search_trials,
               engine.numerics_.maximum_linear_iterations_}};
    const bool evidence_exact =
        evidence.nonlinear_updates == numerical.nonlinear_updates &&
        evidence.trace_solves == numerical.trace_solves &&
        evidence.linear_iterations == numerical.linear_iterations &&
        evidence.line_search_trials == numerical.line_search_trials &&
        evidence.rejected_line_search_trials ==
            numerical.rejected_line_search_trials &&
        pending.evidence.nonlinear_updates == numerical.nonlinear_updates &&
        pending.evidence.trace_solves == numerical.trace_solves &&
        pending.evidence.linear_iterations == numerical.linear_iterations &&
        pending.evidence.line_search_trials == numerical.line_search_trials &&
        pending.evidence.rejected_line_search_trials ==
            numerical.rejected_line_search_trials;
    const bool complete_work_relations =
        numerical.nonlinear_updates >= 0 && numerical.trace_solves > 0 &&
        numerical.linear_iterations >= 0 &&
        numerical.line_search_trials >= 0 &&
        numerical.rejected_line_search_trials >= 0 &&
        numerical.trace_solves == numerical.nonlinear_updates + 1 &&
        numerical.line_search_trials == numerical.nonlinear_updates +
            numerical.rejected_line_search_trials;
    const long long maximum_linear_work =
        static_cast<long long>(engine.numerics_.maximum_linear_iterations_) *
        static_cast<long long>(numerical.trace_solves);
    const bool linear_work_bounded =
        engine.numerics_.maximum_linear_iterations_ > 0 &&
        static_cast<long long>(numerical.linear_iterations) <=
            maximum_linear_work &&
        (!require_linear_activity || numerical.linear_iterations > 0);
    std::array<int, 6> minimum = counts;
    std::array<int, 6> maximum = counts;
    const bool minimum_ready = MPI_Allreduce(
        counts.data(), minimum.data(), static_cast<int>(counts.size()),
        MPI_INT, MPI_MIN, engine.numerics_.communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        counts.data(), maximum.data(), static_cast<int>(counts.size()),
        MPI_INT, MPI_MAX, engine.numerics_.communicator_) == MPI_SUCCESS;
    return engine.numerics_.CollectiveContinue(
        engine.initialized_ && pending.active && pending.staging &&
        pending.prestage_allowed && numerical.success && evidence_exact &&
        complete_work_relations && linear_work_bounded && minimum_ready &&
        maximum_ready && minimum == maximum);
  }

  static bool VerifyPreparedWorkAccounting(
      cuda::GenericEulerHdgCudaSession& session,
      const bool require_linear_activity,
      std::array<int, 6>& counts) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyPreparedWorkAccounting(
          *session.borrowed_rank1_,
          session.borrowed_rank1_->pending_.evidence,
          require_linear_activity, counts);
    if (session.borrowed_rank2_)
      return VerifyPreparedWorkAccounting(
          *session.borrowed_rank2_,
          session.borrowed_rank2_->pending_.evidence,
          require_linear_activity, counts);
    counts.fill(-1);
    return false;
  }

  template <typename Engine>
  static bool VerifyZeroUpdateFinalReconstruction(Engine& engine) noexcept
  {
    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    std::array<int, 6> work{};
    const bool work_accounted = VerifyPreparedWorkAccounting(
        engine, pending.evidence, false, work);
    return work_accounted && engine.initialized_ && pending.active &&
        pending.staging &&
        pending.prestage_allowed && numerical.success &&
        numerical.candidate_ready && numerical.finite &&
        numerical.admissible && numerical.recovery &&
        numerical.true_residual && numerical.nonlinear_updates == 0 &&
        numerical.trace_solves == 1 && numerical.linear_iterations >= 0 &&
        numerical.line_search_trials == 0 &&
        numerical.rejected_line_search_trials == 0 &&
        numerical.products.CoupledAndOrdered();
  }

  static bool VerifyZeroUpdateFinalReconstruction(
      cuda::GenericEulerHdgCudaSession& session) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyZeroUpdateFinalReconstruction(
          *session.borrowed_rank1_);
    if (session.borrowed_rank2_)
      return VerifyZeroUpdateFinalReconstruction(
          *session.borrowed_rank2_);
    return false;
  }

  template <typename Engine>
  static bool CapturePreparedIgrConservation(
      Engine& engine, CudaPreparedIgrConservationCapture& capture) noexcept
  {
    capture = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    bool local_valid = engine.initialized_ && engine.pending_.active &&
        engine.pending_.staging &&
        engine.config_.use_generic_numerical_policy &&
        engine.config_.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::igr && shape.igr_active &&
        std::isfinite(engine.pending_.attempt.dt) &&
        engine.pending_.attempt.dt > 0.0 &&
        numerics.view_.work_interior != nullptr &&
        numerics.view_.old_interior != nullptr &&
        numerics.view_.current_local_trace != nullptr &&
        numerics.view_.scalar != nullptr &&
        shape.interior_components == igr_interior_components &&
        shape.trace_components == igr_trace_components &&
        shape.scalar_interior_dofs > 0 && shape.scalar_face_dofs > 0 &&
        shape.interior_dofs ==
            shape.scalar_interior_dofs * igr_interior_components &&
        shape.trace_dofs == faces_per_element * shape.scalar_face_dofs *
            igr_trace_components;
    std::vector<double> probed_trace;
    try
    {
      capture.current_interior.resize(
          static_cast<std::size_t>(shape.interior_count));
      capture.previous_interior.resize(capture.current_interior.size());
      capture.current_local_trace.resize(
          static_cast<std::size_t>(shape.elements) * shape.trace_dofs);
      probed_trace.resize(capture.current_local_trace.size());
    }
    catch (...) { local_valid = false; }

    if (!numerics.CollectiveContinue(local_valid)) return false;
    const bool launched = CudaLaunchEulerConservation(
            numerics.view_, shape, engine.pending_.attempt.dt,
            numerics.view_.scalar) &&
        CudaTransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!numerics.CollectiveContinue(launched)) return false;

    const bool copied = CudaTransportPolicy::DeviceToHost(
        capture.current_interior.data(), numerics.view_.work_interior,
        capture.current_interior.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            capture.previous_interior.data(), numerics.view_.old_interior,
            capture.previous_interior.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            capture.current_local_trace.data(),
            numerics.view_.current_local_trace,
            capture.current_local_trace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            capture.production_local.data(), numerics.view_.scalar,
            capture.production_local.size() * sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    capture.elements = shape.elements;
    capture.scalar_interior_dofs = shape.scalar_interior_dofs;
    capture.scalar_face_dofs = shape.scalar_face_dofs;
    capture.interior_dofs = shape.interior_dofs;
    capture.trace_dofs = shape.trace_dofs;
    capture.volume_points = shape.volume_points;
    capture.face_points = shape.face_points;
    capture.order = shape.order;
    capture.volume_assembly_order = shape.volume_assembly_order;
    capture.face_assembly_order = shape.face_assembly_order;
    capture.alpha = shape.igr_alpha;
    capture.beta = shape.igr_beta;
    capture.dt = engine.pending_.attempt.dt;
    capture.lifecycle_maximum_error =
        engine.pending_.numerical.maximum_conservation_error;
    capture.lifecycle_maximum_relative_error =
        engine.pending_.numerical.maximum_conservation_relative_error;
    local_valid = copied &&
        capture.production_local[
            euler_conservation_result_scalars - 1] == 1.0 &&
        std::isfinite(capture.production_local[2 * igr_trace_components]) &&
        capture.production_local[2 * igr_trace_components] >= 0.0 &&
        std::all_of(capture.production_local.begin(),
                    capture.production_local.end(),
                    [](const double value) { return std::isfinite(value); });
    if (!numerics.CollectiveContinue(local_valid)) return false;

    std::copy(capture.current_local_trace.begin(),
              capture.current_local_trace.end(), probed_trace.begin());
    for (int element = 0; element < shape.elements; ++element)
      for (int face = 0; face < faces_per_element; ++face)
        for (int scalar = 0; scalar < shape.scalar_face_dofs; ++scalar)
        {
          const std::size_t index = static_cast<std::size_t>(element) *
                  shape.trace_dofs +
              static_cast<std::size_t>(face) * shape.trace_components *
                  shape.scalar_face_dofs +
              4 * shape.scalar_face_dofs + scalar;
          probed_trace[index] +=
              CudaPreparedIgrConservationCapture::sigma_hat_probe;
        }

    const bool probe_installed = CudaTransportPolicy::HostToDevice(
        numerics.view_.current_local_trace, probed_trace.data(),
        probed_trace.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export);
    const bool probe_installed_everywhere =
        numerics.CollectiveContinue(probe_installed);
    const bool probe_launched = probe_installed_everywhere &&
        CudaLaunchEulerConservation(
            numerics.view_, shape, engine.pending_.attempt.dt,
            numerics.view_.scalar) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::scalar_status);
    const bool probe_launched_everywhere =
        numerics.CollectiveContinue(probe_launched);
    const bool probe_copied = probe_launched_everywhere &&
        CudaTransportPolicy::DeviceToHost(
            capture.probed_production_local.data(), numerics.view_.scalar,
            capture.probed_production_local.size() * sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::scalar_status);
    const bool probe_copied_everywhere =
        numerics.CollectiveContinue(probe_copied);

    // The probe is test-only and must leave the live prepared transaction
    // byte-for-byte unchanged regardless of a failed launch or transfer.
    const bool trace_restored = CudaTransportPolicy::HostToDevice(
        numerics.view_.current_local_trace,
        capture.current_local_trace.data(),
        capture.current_local_trace.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export);
    const bool scalar_restored = CudaTransportPolicy::HostToDevice(
        numerics.view_.scalar, capture.production_local.data(),
        capture.production_local.size() * sizeof(double),
        CudaTransferCategory::scalar_diagnostic,
        CudaTransferSite::scalar_status);
    const bool restore_synchronized = CudaTransportPolicy::Synchronize(
        CudaTransferSite::generic_host_export);
    const bool restored_everywhere = numerics.CollectiveContinue(
        trace_restored && scalar_restored && restore_synchronized);
    const bool probe_finite = probe_copied_everywhere &&
        capture.probed_production_local[
            euler_conservation_result_scalars - 1] == 1.0 &&
        capture.probed_production_local[2 * igr_trace_components] ==
            capture.production_local[2 * igr_trace_components] &&
        std::all_of(capture.probed_production_local.begin(),
                    capture.probed_production_local.end(),
                    [](const double value) { return std::isfinite(value); });
    return numerics.CollectiveContinue(
        probe_finite && restored_everywhere);
  }

  static bool CapturePreparedIgrConservation(
      cuda::GenericEulerHdgCudaSession& session,
      CudaPreparedIgrConservationCapture& capture) noexcept
  {
    if (session.partitioned_backward_euler_borrowed_rank1_)
      return CapturePreparedIgrConservation(
          *session.partitioned_backward_euler_borrowed_rank1_, capture);
    if (session.partitioned_backward_euler_borrowed_rank2_)
      return CapturePreparedIgrConservation(
          *session.partitioned_backward_euler_borrowed_rank2_, capture);
    if (session.borrowed_rank1_)
      return CapturePreparedIgrConservation(
          *session.borrowed_rank1_, capture);
    if (session.borrowed_rank2_)
      return CapturePreparedIgrConservation(
          *session.borrowed_rank2_, capture);
    return false;
  }

  template <typename Engine>
  static bool CapturePreparedEulerConservationGate(
      Engine& engine,
      CudaPreparedEulerConservationGateCapture& capture) noexcept
  {
    capture = {};
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const auto& pending = engine.pending_;
    bool local_valid = engine.initialized_ && pending.active &&
        pending.staging.has_value() &&
        engine.config_.use_generic_numerical_policy &&
        engine.config_.numerical_policy.stabilization.kind ==
            generic::RegularizationKind::none &&
        shape.borrowed_host_inputs && !shape.igr_active &&
        std::isfinite(pending.attempt.dt) && pending.attempt.dt > 0.0 &&
        numerics.view_.scalar != nullptr;
    if (!numerics.CollectiveContinue(local_valid)) return false;
    const bool launched = CudaLaunchEulerConservation(
            numerics.view_, shape, pending.attempt.dt,
            numerics.view_.scalar) &&
        CudaTransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    if (!numerics.CollectiveContinue(launched)) return false;
    const bool copied = CudaTransportPolicy::DeviceToHost(
            capture.production_local.data(), numerics.view_.scalar,
            capture.production_local.size() * sizeof(double),
            CudaTransferCategory::scalar_diagnostic,
            CudaTransferSite::scalar_status) &&
        CudaTransportPolicy::Synchronize(CudaTransferSite::scalar_status);
    capture.lifecycle_maximum_error =
        pending.numerical.maximum_conservation_error;
    capture.lifecycle_maximum_relative_error =
        pending.numerical.maximum_conservation_relative_error;
    capture.nonlinear_updates = pending.numerical.nonlinear_updates;
    capture.reason = pending.numerical.reason;
    capture.phase = pending.numerical.phase;
    capture.candidate_ready = pending.numerical.candidate_ready;
    capture.conservation = pending.numerical.conservation;
    capture.success = pending.numerical.success;
    capture.pending_active = pending.active;
    capture.staging_retained = pending.staging.has_value();
    capture.prestage_allowed = pending.prestage_allowed;
    local_valid = copied &&
        capture.production_local[euler_conservation_result_scalars - 1] ==
            1.0 &&
        std::all_of(capture.production_local.begin(),
                    capture.production_local.end(),
                    [](const double value) { return std::isfinite(value); });
    return numerics.CollectiveContinue(local_valid);
  }

  template <typename Engine>
  static bool VerifyIgrCouplingBlocks(
      Engine& engine, std::array<double, 2>& relative_norms) noexcept
  {
    relative_norms.fill(0.0);
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const std::size_t a_count = static_cast<std::size_t>(shape.elements) *
        static_cast<std::size_t>(shape.interior_dofs) *
        static_cast<std::size_t>(shape.interior_dofs);
    const std::size_t b_count = static_cast<std::size_t>(shape.elements) *
        static_cast<std::size_t>(shape.interior_dofs) *
        static_cast<std::size_t>(shape.trace_dofs);
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        shape.igr_active &&
        shape.interior_components == igr_interior_components &&
        shape.trace_components == igr_trace_components &&
        shape.scalar_interior_dofs > 0 && shape.scalar_face_dofs > 0 &&
        numerics.view_.a != nullptr && numerics.view_.b != nullptr;
    std::vector<double> a, b;
    try
    {
      a.resize(a_count);
      b.resize(b_count);
    }
    catch (...) { local_valid = false; }
    const bool copied = local_valid &&
        CudaTransportPolicy::DeviceToHost(
            a.data(), numerics.view_.a, a.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            b.data(), numerics.view_.b, b.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = local_valid && copied;

    // Norm 0 is the production Euler residual's sensitivity to IGR unknowns;
    // norm 1 is the auxiliary residual's sensitivity to Euler unknowns.  Both
    // directions must survive the live 7x5 assembler, independently of any
    // nonlinear-solver stopping tolerance.
    std::array<double, 3> local{};
    for (int element = 0; local_valid && element < shape.elements; ++element)
    {
      const std::size_t a_offset = static_cast<std::size_t>(element) *
          static_cast<std::size_t>(shape.interior_dofs) *
          static_cast<std::size_t>(shape.interior_dofs);
      for (int row = 0; row < shape.interior_dofs; ++row)
        for (int column = 0; column < shape.interior_dofs; ++column)
        {
          const double value = a[a_offset +
              static_cast<std::size_t>(row) * shape.interior_dofs + column];
          local_valid = local_valid && std::isfinite(value);
          local[2] += value * value;
          const int row_component = row / shape.scalar_interior_dofs;
          const int column_component = column /
              shape.scalar_interior_dofs;
          if (row_component < conserved_components &&
              column_component >= conserved_components)
            local[0] += value * value;
          if (row_component >= conserved_components &&
              column_component < conserved_components)
            local[1] += value * value;
        }

      const std::size_t b_offset = static_cast<std::size_t>(element) *
          static_cast<std::size_t>(shape.interior_dofs) *
          static_cast<std::size_t>(shape.trace_dofs);
      for (int row = 0; row < shape.interior_dofs; ++row)
        for (int column = 0; column < shape.trace_dofs; ++column)
        {
          const double value = b[b_offset +
              static_cast<std::size_t>(row) * shape.trace_dofs + column];
          local_valid = local_valid && std::isfinite(value);
          local[2] += value * value;
          const int row_component = row / shape.scalar_interior_dofs;
          const int face_column = column %
              (shape.trace_components * shape.scalar_face_dofs);
          const int column_component = face_column /
              shape.scalar_face_dofs;
          if (row_component < conserved_components &&
              column_component >= conserved_components)
            local[0] += value * value;
          if (row_component >= conserved_components &&
              column_component < conserved_components)
            local[1] += value * value;
        }
    }
    std::array<double, 3> global{};
    const bool reduced = MPI_Allreduce(
        local.data(), global.data(), static_cast<int>(global.size()),
        MPI_DOUBLE, MPI_SUM, engine.config_.borrowed_communicator) ==
        MPI_SUCCESS;
    const double denominator = reduced && global[2] > 0.0
        ? std::sqrt(global[2]) : 0.0;
    if (denominator > 0.0)
    {
      relative_norms[0] = std::sqrt(global[0]) / denominator;
      relative_norms[1] = std::sqrt(global[1]) / denominator;
    }
    return reduced && numerics.CollectiveContinue(
        local_valid && std::isfinite(relative_norms[0]) &&
        std::isfinite(relative_norms[1]) && relative_norms[0] > 1.0e-12 &&
        relative_norms[1] > 1.0e-12);
  }

  template <typename Engine>
  static bool VerifyBorrowedInitialTrueTrace(
      Engine& engine, const CudaBorrowedHostL2Input expected) noexcept
  {
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const bool shape_valid = shape.trace_components >= conserved_components &&
        shape.true_trace_count >= 0 && shape.trace_components != 0 &&
        shape.true_trace_count % shape.trace_components == 0;
    const std::size_t scalar_count = shape_valid
        ? static_cast<std::size_t>(shape.true_trace_count /
                                   shape.trace_components)
        : 0;
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && numerics.borrowed_host_inputs_ &&
        shape_valid && expected.coefficients != nullptr &&
        expected.scalar_count == scalar_count &&
        expected.components == conserved_components &&
        numerics.host_exact_true_trace_.size() ==
            static_cast<std::size_t>(shape.true_trace_count) &&
        numerics.committed_slot_.trace.coefficients != nullptr &&
        numerics.committed_slot_.trace.count ==
            static_cast<std::size_t>(shape.true_trace_count);
    std::vector<double> observed;
    try
    {
      observed.resize(static_cast<std::size_t>(shape.true_trace_count));
    }
    catch (...) { local_valid = false; }
    const bool copied = local_valid && CudaTransportPolicy::DeviceToHost(
        observed.data(), numerics.committed_slot_.trace.coefficients,
        observed.size() * sizeof(double),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = local_valid && copied;
    for (int component = 0;
         local_valid && component < conserved_components; ++component)
      for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
      {
        const std::size_t index = scalar_count *
            static_cast<std::size_t>(component) + scalar;
        local_valid = std::isfinite(expected.coefficients[index]) &&
            numerics.host_exact_true_trace_[index] ==
                expected.coefficients[index] &&
            observed[index] == expected.coefficients[index];
      }
    for (int component = conserved_components;
         local_valid && component < shape.trace_components; ++component)
      for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
      {
        const std::size_t index = scalar_count *
            static_cast<std::size_t>(component) + scalar;
        const double host_value = numerics.host_exact_true_trace_[index];
        const double device_value = observed[index];
        local_valid = host_value == 0.0 && !std::signbit(host_value) &&
            device_value == 0.0 && !std::signbit(device_value);
      }
    return numerics.CollectiveContinue(local_valid);
  }

  static int LoweredBoundaryKind(
      const generic::EulerBoundaryPolicyKind kind) noexcept
  {
    switch (kind)
    {
    case generic::EulerBoundaryPolicyKind::supersonic_inlet:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::supersonic_inlet);
    case generic::EulerBoundaryPolicyKind::subsonic_inlet:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::subsonic_inlet);
    case generic::EulerBoundaryPolicyKind::supersonic_outlet:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::supersonic_outlet);
    case generic::EulerBoundaryPolicyKind::subsonic_outlet_pressure:
      return static_cast<int>(
          CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure);
    case generic::EulerBoundaryPolicyKind::far_field:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::far_field);
    case generic::EulerBoundaryPolicyKind::nonreflecting_outlet:
      return static_cast<int>(
          CudaStaticBoundaryPolicyKind::nonreflecting_outlet);
    case generic::EulerBoundaryPolicyKind::slip_wall:
      return static_cast<int>(CudaStaticBoundaryPolicyKind::slip_wall);
    case generic::EulerBoundaryPolicyKind::periodic_topology:
      return 0;
    }
    return -1;
  }

  static bool SameDouble(const double left, const double right) noexcept
  {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
  }

  template <typename Engine>
  static bool VerifyBoundaryPolicyTable(
      Engine& engine, const generic::EulerBoundaryPolicy* const expected,
      const std::size_t expected_count, const double expected_gamma) noexcept
  {
    auto& numerics = engine.numerics_;
    const CudaMeshShape& shape = numerics.shape_;
    const std::size_t face_count = static_cast<std::size_t>(shape.elements) *
        static_cast<std::size_t>(faces_per_element);
    bool local_valid = engine.initialized_ && !engine.pending_.active &&
        numerics.initialized_ && numerics.mesh_ != nullptr &&
        shape.borrowed_static_boundary_policies && expected != nullptr &&
        expected_count > 0 && expected_count <= 7 &&
        SameDouble(shape.boundary_gamma, expected_gamma) &&
        numerics.host_boundary_policy_kind_.size() == face_count &&
        numerics.host_boundary_pressure_.size() == face_count &&
        numerics.host_boundary_reference_.size() ==
            face_count * conserved_components &&
        numerics.view_.boundary_policy_kind != nullptr &&
        numerics.view_.boundary_pressure != nullptr &&
        numerics.view_.boundary_reference != nullptr;
    std::vector<int> device_kinds;
    std::vector<double> device_pressures, device_references;
    try
    {
      device_kinds.resize(face_count);
      device_pressures.resize(face_count);
      device_references.resize(face_count * conserved_components);
    }
    catch (...) { local_valid = false; }
    const bool copied = local_valid && CudaTransportPolicy::DeviceToHost(
        device_kinds.data(), numerics.view_.boundary_policy_kind,
        device_kinds.size() * sizeof(int),
        CudaTransferCategory::generic_host_bridge,
        CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            device_pressures.data(), numerics.view_.boundary_pressure,
            device_pressures.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::DeviceToHost(
            device_references.data(), numerics.view_.boundary_reference,
            device_references.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    local_valid = local_valid && copied;

    std::array<long long, 7> local_uses{};
    std::vector<unsigned char> shared;
    try
    {
      shared.resize(static_cast<std::size_t>(
          numerics.mesh_->GetNumFaces()), 0);
      for (int index = 0; index < numerics.mesh_->GetNSharedFaces(); ++index)
      {
        const int face = numerics.mesh_->GetSharedFace(index);
        if (face < 0 || face >= static_cast<int>(shared.size()))
          local_valid = false;
        else
          shared[static_cast<std::size_t>(face)] = 1;
      }
      for (int element = 0; local_valid && element < shape.elements; ++element)
      {
        mfem::Array<int> faces, orientations;
        numerics.mesh_->GetElementEdges(element, faces, orientations);
        local_valid = faces.Size() == faces_per_element;
        for (int local_face = 0;
             local_valid && local_face < faces.Size(); ++local_face)
        {
          const int face = faces[local_face];
          mfem::FaceElementTransformations* transformations =
              numerics.mesh_->GetFaceElementTransformations(face);
          const int boundary_element = face >= 0 &&
                  face < static_cast<int>(shared.size())
              ? numerics.mesh_->GetFaceToBdrElMap()[face] : -1;
          const bool physical = transformations != nullptr &&
              shared[static_cast<std::size_t>(face)] == 0 &&
              transformations->Elem2No < 0 && boundary_element >= 0;
          const std::size_t face_index = static_cast<std::size_t>(element) *
              faces_per_element + local_face;
          int expected_kind = 0;
          double expected_pressure = 0.0;
          std::array<double, conserved_components> expected_reference{};
          if (physical)
          {
            const int attribute =
                numerics.mesh_->GetBdrAttribute(boundary_element);
            local_valid = attribute >= 1 &&
                static_cast<std::size_t>(attribute) <= expected_count;
            if (!local_valid) break;
            const generic::EulerBoundaryPolicy& policy =
                expected[static_cast<std::size_t>(attribute - 1)];
            expected_kind = LoweredBoundaryKind(policy.kind);
            expected_pressure = policy.pressure_out;
            expected_reference = policy.reference_state;
            ++local_uses[static_cast<std::size_t>(attribute - 1)];
          }
          local_valid = expected_kind >= 0 &&
              numerics.host_boundary_policy_kind_[face_index] ==
                  expected_kind &&
              device_kinds[face_index] == expected_kind &&
              SameDouble(numerics.host_boundary_pressure_[face_index],
                         expected_pressure) &&
              SameDouble(device_pressures[face_index], expected_pressure);
          for (int component = 0;
               local_valid && component < conserved_components; ++component)
          {
            const std::size_t index = face_index * conserved_components +
                static_cast<std::size_t>(component);
            local_valid = SameDouble(
                    numerics.host_boundary_reference_[index],
                    expected_reference[static_cast<std::size_t>(component)]) &&
                SameDouble(
                    device_references[index],
                    expected_reference[static_cast<std::size_t>(component)]);
          }
        }
      }
    }
    catch (...) { local_valid = false; }

    std::array<long long, 7> global_uses{};
    const bool reduced = MPI_Allreduce(
        local_uses.data(), global_uses.data(),
        static_cast<int>(global_uses.size()), MPI_LONG_LONG_INT, MPI_SUM,
        engine.config_.borrowed_communicator) == MPI_SUCCESS;
    bool semantic_coverage = reduced;
    for (std::size_t index = 0;
         semantic_coverage && index < global_uses.size(); ++index)
    {
      const bool present = index < expected_count;
      const bool periodic = present && expected[index].kind ==
          generic::EulerBoundaryPolicyKind::periodic_topology;
      semantic_coverage = present
          ? (periodic ? global_uses[index] == 0 : global_uses[index] > 0)
          : global_uses[index] == 0;
    }
    return numerics.CollectiveContinue(
        local_valid && semantic_coverage);
  }

  static bool VerifyBorrowedInitialTrueTrace(
      cuda::GenericEulerHdgCudaSession& session,
      const CudaBorrowedHostL2Input expected) noexcept
  {
    if (session.partitioned_backward_euler_borrowed_rank1_)
      return VerifyBorrowedInitialTrueTrace(
          *session.partitioned_backward_euler_borrowed_rank1_, expected);
    if (session.partitioned_backward_euler_borrowed_rank2_)
      return VerifyBorrowedInitialTrueTrace(
          *session.partitioned_backward_euler_borrowed_rank2_, expected);
    if (session.borrowed_rank1_)
      return VerifyBorrowedInitialTrueTrace(
          *session.borrowed_rank1_, expected);
    if (session.borrowed_rank2_)
      return VerifyBorrowedInitialTrueTrace(
          *session.borrowed_rank2_, expected);
    return false;
  }

  static bool VerifyBoundaryPolicyTable(
      cuda::GenericEulerHdgCudaSession& session,
      const generic::EulerBoundaryPolicy* const expected,
      const std::size_t expected_count, const double expected_gamma) noexcept
  {
    if (session.partitioned_backward_euler_borrowed_rank1_)
      return VerifyBoundaryPolicyTable(
          *session.partitioned_backward_euler_borrowed_rank1_, expected,
          expected_count, expected_gamma);
    if (session.partitioned_backward_euler_borrowed_rank2_)
      return VerifyBoundaryPolicyTable(
          *session.partitioned_backward_euler_borrowed_rank2_, expected,
          expected_count, expected_gamma);
    if (session.borrowed_rank1_)
      return VerifyBoundaryPolicyTable(
          *session.borrowed_rank1_, expected, expected_count, expected_gamma);
    if (session.borrowed_rank2_)
      return VerifyBoundaryPolicyTable(
          *session.borrowed_rank2_, expected, expected_count, expected_gamma);
    if (session.rodas3_borrowed_rank1_)
      return VerifyBoundaryPolicyTable(
          *session.rodas3_borrowed_rank1_, expected,
          expected_count, expected_gamma);
    if (session.rodas3_borrowed_rank2_)
      return VerifyBoundaryPolicyTable(
          *session.rodas3_borrowed_rank2_, expected,
          expected_count, expected_gamma);
    return false;
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
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::HdgEulerHostInteriorFields;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::BorrowedCudaRank1OneGpuRuntimeSelection;
using dg::hdg::runtime::BorrowedCudaRank2OneGpuHostStagedRuntimeSelection;
using dg::hdg::runtime::FailureReason;
using dg::hdg::runtime::RunStatus;
using dg::hdg::runtime::StableIdentity;
using dg::hdg::runtime::cuda_w8a::CudaStaticBoundaryPolicy;
using dg::hdg::runtime::cuda_w8a::CudaStaticBoundaryPolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaConfig;
using dg::hdg::runtime::cuda_w8a::CudaFailurePhase;
using dg::hdg::runtime::cuda_w8a::CudaBorrowedHostL2Input;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaPrecommitSnapshot;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::CudaRunEvidence;
using dg::hdg::runtime::cuda_w8a::CudaRunStatus;
using dg::hdg::runtime::cuda_w8a::CudaStepEvidence;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaTransferTotals;
using dg::hdg::runtime::cuda_w8a::CudaTransportPolicy;
using dg::hdg::runtime::cuda_w8a::PersistentCudaSmoothEulerEngine;
using dg::hdg::runtime::cuda_w8a::detail::CudaBorrowedRecoveryForEngine;
using dg::hdg::runtime::cuda_w8a::detail::CudaBorrowedRecoveryKind;
using dg::hdg::runtime::cuda_w8a::detail::CudaDirectZhangShuAudit;
using dg::hdg::runtime::cuda_w8a::detail::CudaRecoveryRouteSupported;
using dg::hdg::runtime::cuda_w8a::detail::CudaRodas3LimiterEndpointAudit;
using dg::hdg::runtime::cuda_w8a::detail::CudaRodas3ScreenObservation;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaPreparedEulerConservationGateCapture;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaPreparedIgrConservationCapture;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaPreparedPartitionedClosureAudit;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaInitializedRodas3IgrClosureAudit;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaPreparedLocalCondensationCapture;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaBorrowedAssemblyFusionAudit;
using dg::hdg::runtime::cuda_w8a::detail::CudaPreparedRecoveryAudit;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using dg::hdg::runtime::cuda_w8a::igr_interior_components;
using dg::hdg::runtime::cuda_w8a::igr_trace_components;
using dg::hdg::runtime::cuda_w8a::detail::qualification_density_gate;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr int hidden_corner_order = 3;
constexpr int hidden_corner_scalar_dofs = 16;
constexpr int hidden_corner_quadrature_order = 12;
constexpr int hidden_corner_pressure_bisections = 24;
constexpr double hidden_corner_density_floor = 1.0e-8;
constexpr double hidden_corner_pressure_floor = 1.0e-6;
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);
constexpr std::uint32_t cpu_capabilities =
    (cuda_capabilities & ~CapabilityBit(Capability::cuda)) |
    CapabilityBit(Capability::cpu);

CudaEngineKind selected_cuda_engine = CudaEngineKind::ordinary;
EulerHdgTimeTreatment selected_time_treatment =
    EulerHdgTimeTreatment::backward_euler;

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
             MPI_SUCCESS &&
      output == 1;
}

mfem::ParMesh MakePeriodicMesh()
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakeMixedPeriodicMesh()
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(1, mfem::Vector(2));
  translations[0] = 0.0;
  translations[0][0] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

mfem::ParMesh MakePhysicalMesh(const int elements_per_direction = 3)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      elements_per_direction, elements_per_direction,
      mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_WORLD, serial);
}

mfem::ParMesh MakeAllSevenPhysicalMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      4, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  for (int boundary = 0; boundary < serial.GetNBE(); ++boundary)
    serial.SetBdrAttribute(boundary, 1 + boundary % 7);
  serial.SetAttributes(false, true);
  return mfem::ParMesh(MPI_COMM_WORLD, serial);
}

struct BoundaryContext
{
  std::array<int, 4> calls{};
  std::array<EulerBoundaryPolicy, 4> policies{};
};

struct AllSevenBoundaryContext
{
  std::array<int, 7> calls{};
  std::array<EulerBoundaryPolicy, 7> policies{};
};

void ConfigureAllSevenPolicies(AllSevenBoundaryContext& context)
{
  const std::array<double, components> inlet{{
      1.0, 0.20, -0.05, 2.52125}};
  const std::array<double, components> subsonic_inlet{{
      0.95, 0.095, 0.019, 2.3812}};
  const std::array<double, components> far_field{{
      1.05, -0.105, 0.0525, 2.5078125}};
  const std::array<double, components> nonreflecting{{
      0.98, 0.049, -0.0294, 2.4522}};
  const std::array<double, components> unused{};
  context.policies = {{
      {EulerBoundaryPolicyKind::supersonic_inlet, inlet, 0.0},
      {EulerBoundaryPolicyKind::subsonic_inlet, subsonic_inlet, 0.0},
      {EulerBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
      {EulerBoundaryPolicyKind::subsonic_outlet_pressure, unused, 0.93},
      {EulerBoundaryPolicyKind::far_field, far_field, 0.0},
      {EulerBoundaryPolicyKind::nonreflecting_outlet, nonreflecting, 0.0},
      {EulerBoundaryPolicyKind::slip_wall, unused, 0.0}}};
}

bool AllSevenPhysicalEulerExterior(
    const void* opaque, const dg::hdg::generic::BoundaryPointView& point,
    const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<AllSevenBoundaryContext*>(
      const_cast<void*>(opaque));
  if (context == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 7 ||
      interior == nullptr || exterior == nullptr)
    return false;
  ++context->calls[static_cast<std::size_t>(point.boundary_attribute - 1)];
  // Explicit initial_true_trace must bypass compatibility projection and this
  // deliberately failing callback.
  return false;
}

void ConfigurePolicies(BoundaryContext& context, const bool physical)
{
  const std::array<double, components> reference{{1.0, 0.05, -0.02, 2.60}};
  const std::array<double, components> unused{};
  if (!physical)
  {
    for (EulerBoundaryPolicy& policy : context.policies)
    {
      policy = {EulerBoundaryPolicyKind::periodic_topology, unused, 0.0};
    }
    return;
  }
  context.policies = {{
      {EulerBoundaryPolicyKind::slip_wall, unused, 0.0},
      {EulerBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
      {EulerBoundaryPolicyKind::far_field, reference, 0.0},
      {EulerBoundaryPolicyKind::supersonic_inlet, reference, 0.0}}};
}

bool PhysicalEulerExterior(
    const void* opaque, const dg::hdg::generic::BoundaryPointView& point,
    const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(
      const_cast<void*>(opaque));
  if (context == nullptr || point.position == nullptr ||
      point.outward_normal == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 4 ||
      interior == nullptr || exterior == nullptr)
  {
    return false;
  }
  ++context->calls[static_cast<std::size_t>(point.boundary_attribute - 1)];
  std::copy_n(interior, components, exterior);
  switch (point.boundary_attribute)
  {
    case 1:
    {
      // Slip-wall exterior state.
      const double normal_momentum =
          interior[1] * point.outward_normal[0] +
          interior[2] * point.outward_normal[1];
      exterior[1] -= 2.0 * normal_momentum * point.outward_normal[0];
      exterior[2] -= 2.0 * normal_momentum * point.outward_normal[1];
      break;
    }
    case 2:
      // Supersonic outlet: the interior state is the exterior state.
      break;
    case 3:
      // Far-field/nonreflecting representative state.
      exterior[0] = 1.0;
      exterior[1] = 0.05;
      exterior[2] = -0.02;
      exterior[3] = 2.60;
      break;
    case 4:
      // Inlet representative state.
      exterior[0] = 1.0;
      exterior[1] = 0.05;
      exterior[2] = -0.02;
      exterior[3] = 2.60;
      break;
  }
  return true;
}

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> initial_true_trace;
  std::vector<double> source;
  std::size_t volume_scalar_count = 0;
};

enum class SourceMode
{
  empty,
  explicit_zero,
  nonzero
};

enum class InitialMode
{
  constant,
  smooth_periodic,
  affine_physical,
  hidden_corner_pressure
};

void EvaluateInitialState(const InitialMode mode, const mfem::Vector& point,
                          mfem::Vector& state)
{
  state.SetSize(components);
  if (mode == InitialMode::constant)
  {
    state[0] = 1.0;
    state[1] = 0.05;
    state[2] = -0.02;
    state[3] = 2.60;
    return;
  }
  if (mode == InitialMode::affine_physical)
  {
    const double rho = 1.0 + 0.05 * point[0] + 0.03 * point[1];
    const double u = 0.20 + 0.02 * point[0];
    const double v = -0.10 + 0.015 * point[1];
    const double pressure = 1.0 + 0.04 * point[0] - 0.02 * point[1];
    state[0] = rho;
    state[1] = rho * u;
    state[2] = rho * v;
    state[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
    return;
  }
  if (mode == InitialMode::hidden_corner_pressure)
  {
    constexpr double pressure_delta = 1.0e-2;
    const double pressure = point[0] + point[1] - pressure_delta;
    state[0] = 1.0;
    state[1] = 0.0;
    state[2] = 0.0;
    state[3] = pressure / 0.4;
    return;
  }
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi * point[0]);
  const double sy = std::sin(2.0 * pi * point[1]);
  const double cx = std::cos(2.0 * pi * point[0]);
  const double cy = std::cos(2.0 * pi * point[1]);
  const double rho = 1.0 + 0.08 * sx * sy;
  const double u = 0.22 + 0.04 * cx * sy;
  const double v = -0.11 + 0.03 * sx * cy;
  const double pressure = 1.0 + 0.06 * cx * cy;
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / 0.4 + 0.5 * rho * (u * u + v * v);
}

int DecodeSignedDof(int encoded, double& sign) noexcept;

template <typename Evaluator>
bool InterpolateVolumeFieldHost(
    mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space,
    Evaluator&& evaluate, std::vector<double>& coefficients)
{
  try
  {
    const int vector_size = space.GetVSize();
    if (vector_size < 0 || space.GetVDim() != components) return false;
    coefficients.assign(static_cast<std::size_t>(vector_size), 0.0);
    std::vector<unsigned char> assigned(
        static_cast<std::size_t>(vector_size), 0);
    for (int element = 0; element < mesh.GetNE(); ++element)
    {
      const mfem::FiniteElement* finite_element = space.GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh.GetElementTransformation(element);
      mfem::Array<int> vdofs;
      space.GetElementVDofs(element, vdofs);
      if (finite_element == nullptr || transformation == nullptr ||
          finite_element->GetDof() <= 0 ||
          vdofs.Size() != finite_element->GetDof() * components)
        return false;
      const mfem::IntegrationRule& nodes = finite_element->GetNodes();
      if (nodes.GetNPoints() != finite_element->GetDof()) return false;
      for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
      {
        mfem::Vector position(2), state(components);
        transformation->Transform(nodes.IntPoint(scalar), position);
        evaluate(position, state);
        if (state.Size() != components) return false;
        for (int component = 0; component < components; ++component)
        {
          double sign = 1.0;
          const int dof = DecodeSignedDof(
              vdofs[scalar + finite_element->GetDof() * component], sign);
          const double value = sign * state[component];
          if (dof < 0 || dof >= vector_size || !std::isfinite(value))
            return false;
          const std::size_t index = static_cast<std::size_t>(dof);
          if (assigned[index] != 0 && coefficients[index] != value)
            return false;
          coefficients[index] = value;
          assigned[index] = 1;
        }
      }
    }
    return std::find(assigned.begin(), assigned.end(), 0) == assigned.end();
  }
  catch (...) { return false; }
}

template <typename Evaluator>
bool InterpolateTrueTraceHost(
    mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space,
    Evaluator&& evaluate, std::vector<double>& true_coefficients)
{
  try
  {
    const int true_vector_size = space.GetTrueVSize();
    if (true_vector_size < 0 || space.GetVDim() != components)
      return false;
    true_coefficients.assign(
        static_cast<std::size_t>(true_vector_size), 0.0);
    std::vector<unsigned char> assigned(
        static_cast<std::size_t>(true_vector_size), 0);
    for (int face = 0; face < mesh.GetNumFaces(); ++face)
    {
      mfem::FaceElementTransformations* transformation =
          mesh.GetFaceElementTransformations(face);
      const mfem::FiniteElement* finite_element = space.GetFaceElement(face);
      mfem::Array<int> vdofs;
      space.GetFaceVDofs(face, vdofs);
      if (transformation == nullptr || transformation->Face == nullptr ||
          finite_element == nullptr || finite_element->GetDof() <= 0 ||
          vdofs.Size() != finite_element->GetDof() * components)
        return false;
      const mfem::IntegrationRule& nodes = finite_element->GetNodes();
      if (nodes.GetNPoints() != finite_element->GetDof()) return false;
      for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
      {
        mfem::Vector position(2), state(components);
        transformation->Face->Transform(nodes.IntPoint(scalar), position);
        evaluate(position, state);
        if (state.Size() != components) return false;
        for (int component = 0; component < components; ++component)
        {
          double sign = 1.0;
          const int local_dof = DecodeSignedDof(
              vdofs[scalar + finite_element->GetDof() * component], sign);
          const int true_dof = local_dof >= 0 && local_dof < space.GetVSize()
              ? space.GetLocalTDofNumber(local_dof) : -2;
          const double value = sign * state[component];
          if (true_dof == -1) continue;
          if (true_dof < 0 || true_dof >= true_vector_size ||
              !std::isfinite(value))
            return false;
          const std::size_t index = static_cast<std::size_t>(true_dof);
          if (assigned[index] != 0 && true_coefficients[index] != value)
            return false;
          true_coefficients[index] = value;
          assigned[index] = 1;
        }
      }
    }
    return std::find(assigned.begin(), assigned.end(), 0) == assigned.end();
  }
  catch (...) { return false; }
}

BorrowedFields MakeBorrowedFields(
    mfem::ParMesh& mesh, const int order, const SourceMode source_mode,
    const InitialMode initial_mode = InitialMode::constant)
{
  BorrowedFields fields;
  mfem::L2_FECollection volume_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &volume_collection, components, mfem::Ordering::byNODES);
  fields.volume_scalar_count = static_cast<std::size_t>(
      volume_space.GetVSize() / components);
  const auto initial_evaluator =
      [initial_mode](const mfem::Vector& point, mfem::Vector& state)
      { EvaluateInitialState(initial_mode, point, state); };
  if (!InterpolateVolumeFieldHost(
          mesh, volume_space, initial_evaluator, fields.initial))
    return {};
  if (source_mode != SourceMode::empty)
  {
    fields.source.resize(fields.initial.size(), 0.0);
  }
  for (std::size_t scalar = 0; scalar < fields.volume_scalar_count; ++scalar)
  {
    if (source_mode == SourceMode::nonzero)
    {
      // Keep dt*S far above the nonlinear stopping floor so this checks the
      // source coefficient quantitatively, not merely its sign.
      fields.source[scalar] = 0.25;
      fields.source[fields.volume_scalar_count + scalar] = 0.125;
      fields.source[2 * fields.volume_scalar_count + scalar] = -0.1875;
      fields.source[3 * fields.volume_scalar_count + scalar] = 0.375;
    }
  }

  mfem::DG_Interface_FECollection trace_collection(
      order, 2, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, components, mfem::Ordering::byNODES);
  if (!InterpolateTrueTraceHost(
          mesh, trace_space, initial_evaluator, fields.initial_true_trace))
    return {};
  return fields;
}

struct HiddenCornerFieldAudit
{
  std::array<double, components> mean{};
  double minimum_selected_pressure =
      std::numeric_limits<double>::infinity();
  double minimum_vertex_pressure =
      std::numeric_limits<double>::infinity();
  bool selected_points_admissible = false;
  bool vertices_admissible = false;
};

bool AuditHiddenCornerField(
    mfem::ParMesh& mesh, const std::vector<double>& coefficients,
    HiddenCornerFieldAudit& audit)
{
  audit = {};
  bool ready = mesh.GetNE() == 1;
  try
  {
    mfem::L2_FECollection collection(
        hidden_corner_order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, components, mfem::Ordering::byNODES);
    const mfem::FiniteElement* const finite_element =
        ready ? space.GetFE(0) : nullptr;
    mfem::ElementTransformation* const transformation =
        ready ? mesh.GetElementTransformation(0) : nullptr;
    mfem::Array<int> vdofs;
    if (ready) space.GetElementVDofs(0, vdofs);
    ready = ready && finite_element && transformation &&
        finite_element->GetDof() == hidden_corner_scalar_dofs &&
        vdofs.Size() == components * hidden_corner_scalar_dofs &&
        coefficients.size() == static_cast<std::size_t>(space.GetVSize());

    const auto evaluate = [&](const mfem::IntegrationPoint& point,
                              std::array<double, components>& state)
    {
      if (!ready) return false;
      mfem::Vector shape(hidden_corner_scalar_dofs);
      finite_element->CalcShape(point, shape);
      state = {};
      for (int scalar = 0; scalar < hidden_corner_scalar_dofs; ++scalar)
        for (int component = 0; component < components; ++component)
        {
          double sign = 1.0;
          const int dof = DecodeSignedDof(
              vdofs[component * hidden_corner_scalar_dofs + scalar], sign);
          if (dof < 0 ||
              static_cast<std::size_t>(dof) >= coefficients.size())
          {
            return false;
          }
          state[component] += sign * shape[scalar] *
              coefficients[static_cast<std::size_t>(dof)];
        }
      return std::all_of(
          state.begin(), state.end(),
          [](const double value) { return std::isfinite(value); });
    };
    const auto admit = [&](const mfem::IntegrationPoint& point,
                           double& minimum_pressure)
    {
      std::array<double, components> state{};
      double pressure = std::numeric_limits<double>::quiet_NaN();
      const bool pressure_ready = evaluate(point, state) &&
          dg::hdg::euler::zhang_shu::Pressure(
              state.data(), 1.4, pressure);
      if (pressure_ready)
        minimum_pressure = std::min(minimum_pressure, pressure);
      return pressure_ready && state[0] >= hidden_corner_density_floor &&
          pressure >= hidden_corner_pressure_floor;
    };

    bool selected = ready;
    double measure = 0.0;
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, hidden_corner_quadrature_order);
    for (int q = 0; q < volume_rule.GetNPoints() && ready; ++q)
    {
      const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
      std::array<double, components> state{};
      transformation->SetIntPoint(&point);
      const double weight = point.weight * transformation->Weight();
      ready = std::isfinite(weight) && weight > 0.0 &&
          evaluate(point, state);
      if (!ready) break;
      measure += weight;
      for (int component = 0; component < components; ++component)
        audit.mean[component] += weight * state[component];
      selected = admit(point, audit.minimum_selected_pressure) && selected;
    }
    ready = ready && std::isfinite(measure) && measure > 0.0;
    for (double& value : audit.mean)
    {
      if (ready) value /= measure;
      ready = ready && std::isfinite(value);
    }

    const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
        mfem::Geometry::SEGMENT, hidden_corner_quadrature_order);
    for (int q = 0; q < face_rule.GetNPoints() && ready; ++q)
    {
      const double s = face_rule.IntPoint(q).x;
      std::array<mfem::IntegrationPoint, 4> points{};
      points[0].Set2(s, 0.0);
      points[1].Set2(1.0, s);
      points[2].Set2(s, 1.0);
      points[3].Set2(0.0, s);
      for (const mfem::IntegrationPoint& point : points)
        selected = admit(point, audit.minimum_selected_pressure) && selected;
    }
    const mfem::IntegrationRule& nodes = finite_element->GetNodes();
    ready = ready && nodes.GetNPoints() == hidden_corner_scalar_dofs;
    for (int scalar = 0; scalar < nodes.GetNPoints() && ready; ++scalar)
      selected = admit(
          nodes.IntPoint(scalar), audit.minimum_selected_pressure) && selected;

    const mfem::IntegrationRule* const vertices =
        mfem::Geometries.GetVertices(mfem::Geometry::SQUARE);
    bool vertices_admissible = ready && vertices &&
        vertices->GetNPoints() == 4;
    for (int vertex = 0;
         vertices && vertex < vertices->GetNPoints() && ready; ++vertex)
    {
      vertices_admissible = admit(
          vertices->IntPoint(vertex), audit.minimum_vertex_pressure) &&
          vertices_admissible;
    }
    audit.selected_points_admissible = selected;
    audit.vertices_admissible = vertices_admissible;
  }
  catch (...)
  {
    ready = false;
  }
  return ready &&
      std::isfinite(audit.minimum_selected_pressure) &&
      std::isfinite(audit.minimum_vertex_pressure);
}

EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const int order, BorrowedFields& fields,
    BoundaryContext& boundary, const bool physical,
    const RegularizationKind kind)
{
  ConfigurePolicies(boundary, physical);
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cuda, 0, selected_cuda_engine};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.empty() ? nullptr : fields.initial.data(),
      fields.initial.size(), components};
  request.initial_true_trace = {
      fields.initial_true_trace.empty() ? nullptr
                                        : fields.initial_true_trace.data(),
      fields.initial_true_trace.size(), components};
  request.source = {fields.source.empty() ? nullptr : fields.source.data(),
                    fields.source.size(), components};
  request.boundary.context = &boundary;
  request.boundary.exterior_state = PhysicalEulerExterior;
  request.boundary.policies = physical
      ? dg::hdg::generic::EulerBoundaryPolicyTableView{
            boundary.policies.data(), boundary.policies.size(), 2, 1.4}
      : dg::hdg::generic::EulerBoundaryPolicyTableView{
            nullptr, 0, 2, 1.4};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  request.numerical_policy.stabilization = kind == RegularizationKind::igr
      ? dg::hdg::generic::RegularizationSelection{
            RegularizationKind::igr, 0.2, 0.8}
      : dg::hdg::generic::RegularizationSelection{
            RegularizationKind::none, 0.0, 0.0};
  request.time_treatment = selected_time_treatment;
  return request;
}

CudaConfig MakeDirectConfig(const EulerHdgRequest& request,
                            const int ranks, const int order)
{
  CudaConfig config{};
  config.order = order;
  config.mesh_n = 0;
  config.requested_device = 0;
  config.cuda_engine = request.backend.cuda_engine;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = const_cast<mfem::ParMesh*>(request.mesh);
  config.borrowed_communicator = request.communicator;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {
      request.initial_state.data,
      request.initial_state.scalar_count / static_cast<std::size_t>(components),
      request.initial_state.components};
  config.borrowed_initial_true_trace = {
      request.initial_true_trace.data,
      request.initial_true_trace.scalar_count /
          static_cast<std::size_t>(components),
      request.initial_true_trace.components};
  config.borrowed_source = {
      request.source.data,
      request.source.scalar_count / static_cast<std::size_t>(components),
      request.source.components};
  config.borrowed_boundary = request.boundary;
  config.use_generic_numerical_policy = true;
  config.numerical_policy = request.numerical_policy;
  config.time_treatment = request.time_treatment;
  return config;
}

struct IndependentIgrConservationResult
{
  std::array<double, 5> production_balance{};
  std::array<double, 5> production_scale{};
  std::array<double, 5> oracle_balance{};
  std::array<double, 5> oracle_scale{};
  double production_domain_measure =
      std::numeric_limits<double>::infinity();
  double oracle_domain_measure =
      std::numeric_limits<double>::infinity();
  double domain_measure_discrepancy =
      std::numeric_limits<double>::infinity();
  // Source components, Euler change, physical Euler flux, sigma/rho,
  // alpha*Q, and the physical auxiliary flux, respectively.
  std::array<double, 9> activity{};
  double maximum_balance_discrepancy =
      std::numeric_limits<double>::infinity();
  double maximum_scale_discrepancy =
      std::numeric_limits<double>::infinity();
  double production_probe_delta =
      std::numeric_limits<double>::infinity();
  double oracle_probe_delta =
      std::numeric_limits<double>::infinity();
  double probe_discrepancy =
      std::numeric_limits<double>::infinity();
  bool passed = false;
};

int DecodeSignedDof(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

bool IndependentRegularizedNormalFlux(
    const std::array<long double, 5>& trace,
    const long double normal_x, const long double normal_y,
    std::array<long double, 4>& flux) noexcept
{
  const long double rho = trace[0];
  const long double momentum_x = trace[1];
  const long double momentum_y = trace[2];
  const long double energy = trace[3];
  const long double sigma_hat = trace[4];
  if (!(rho > 0.0L)) return false;
  const long double normal_momentum =
      momentum_x * normal_x + momentum_y * normal_y;
  const long double pressure = 0.4L *
      (energy - 0.5L *
          (momentum_x * momentum_x + momentum_y * momentum_y) / rho);
  if (!(pressure > 0.0L) || !std::isfinite(pressure)) return false;
  const long double normal_velocity = normal_momentum / rho;
  flux = {{
      normal_momentum,
      momentum_x * normal_velocity + pressure * normal_x +
          sigma_hat * normal_x,
      momentum_y * normal_velocity + pressure * normal_y +
          sigma_hat * normal_y,
      (energy + pressure) * normal_velocity +
          sigma_hat * normal_velocity}};
  return std::all_of(flux.begin(), flux.end(),
                     [](const long double value)
                     { return std::isfinite(value); });
}

bool VerifyIndependentIgrConservation(
    const CudaPreparedIgrConservationCapture& capture,
    const EulerHdgRequest& request,
    IndependentIgrConservationResult& result)
{
  result = {};
  const auto requested_orders =
      request.numerical_policy.igr_assembly_quadrature_orders;
  const bool locally_admitted = request.mesh != nullptr &&
      request.source.data != nullptr &&
      request.source.components == components &&
      request.source.scalar_count != 0 &&
      capture.elements >= 0 && capture.scalar_interior_dofs > 0 &&
      capture.scalar_face_dofs > 0 &&
      capture.interior_dofs == capture.scalar_interior_dofs * 7 &&
      capture.trace_dofs == 4 * capture.scalar_face_dofs * 5 &&
      capture.order == request.polynomial_order &&
      capture.volume_assembly_order == requested_orders.volume_order &&
      capture.face_assembly_order == requested_orders.face_order &&
      capture.alpha == request.numerical_policy.stabilization.alpha &&
      capture.beta == request.numerical_policy.stabilization.beta &&
      capture.dt > 0.0 && std::isfinite(capture.dt);
  if (!All(locally_admitted, request.communicator)) return false;

  mfem::ParMesh& mesh = *const_cast<mfem::ParMesh*>(request.mesh);
  bool local_valid = capture.elements == mesh.GetNE() &&
      capture.current_interior.size() ==
          static_cast<std::size_t>(capture.elements) *
              capture.interior_dofs &&
      capture.previous_interior.size() ==
          capture.current_interior.size() &&
      capture.current_local_trace.size() ==
          static_cast<std::size_t>(capture.elements) * capture.trace_dofs;
  std::array<long double, 5> local_balance{};
  std::array<long double, 5> local_scale{};
  std::array<long double, 9> local_activity{};
  std::array<long double, 4> local_change_integral{};
  std::array<long double, 4> local_source_integral{};
  std::array<long double, 4> local_boundary_integral{};
  std::array<long double, 3> local_auxiliary_integral{};
  long double local_domain_measure = 0.0L;
  long double local_probe_delta = 0.0L;

  try
  {
    mfem::L2_FECollection volume_collection(
        capture.order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace source_space(
        &mesh, &volume_collection, components, mfem::Ordering::byNODES);
    mfem::DG_Interface_FECollection trace_collection(
        capture.order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace trace_space(
        &mesh, &trace_collection, 5, mfem::Ordering::byNODES);
    local_valid = local_valid &&
        request.source.scalar_count ==
            static_cast<std::size_t>(source_space.GetVSize());

    std::vector<unsigned char> shared_faces(
        static_cast<std::size_t>(mesh.GetNumFaces()), 0);
    for (int shared = 0; local_valid && shared < mesh.GetNSharedFaces();
         ++shared)
    {
      const int face = mesh.GetSharedFace(shared);
      local_valid = face >= 0 &&
          face < static_cast<int>(shared_faces.size());
      if (local_valid) shared_faces[static_cast<std::size_t>(face)] = 1;
    }

    for (int element = 0;
         local_valid && element < capture.elements; ++element)
    {
      const mfem::FiniteElement* finite_element =
          source_space.GetFE(element);
      mfem::ElementTransformation* transformation =
          mesh.GetElementTransformation(element);
      mfem::Array<int> source_vdofs;
      source_space.GetElementVDofs(element, source_vdofs);
      local_valid = finite_element != nullptr && transformation != nullptr &&
          finite_element->GetDof() == capture.scalar_interior_dofs &&
          source_vdofs.Size() ==
              capture.scalar_interior_dofs * components;
      if (!local_valid) break;

      const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
          finite_element->GetGeomType(), capture.volume_assembly_order);
      local_valid = volume_rule.GetNPoints() == capture.volume_points;
      mfem::Vector volume_shape(capture.scalar_interior_dofs);
      mfem::DenseMatrix volume_derivative(
          capture.scalar_interior_dofs, 2);
      const std::size_t interior_offset =
          static_cast<std::size_t>(element) * capture.interior_dofs;
      for (int q = 0; local_valid && q < volume_rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
        transformation->SetIntPoint(&point);
        finite_element->CalcShape(point, volume_shape);
        finite_element->CalcPhysDShape(*transformation, volume_derivative);
        const long double weight = static_cast<long double>(
            point.weight * transformation->Weight());
        std::array<long double, 4> current{};
        std::array<long double, 4> previous{};
        std::array<long double, 4> source{};
        long double sigma = 0.0L;
        long double conservative_gradient[3][2] = {};
        local_valid = weight > 0.0L && std::isfinite(weight);
        if (local_valid)
        {
          local_domain_measure += weight;
          local_valid = std::isfinite(local_domain_measure);
        }
        for (int scalar = 0;
             local_valid && scalar < capture.scalar_interior_dofs; ++scalar)
        {
          const long double phi = volume_shape[scalar];
          for (int component = 0; component < components; ++component)
          {
            const std::size_t state_index = interior_offset + scalar +
                static_cast<std::size_t>(capture.scalar_interior_dofs) *
                    component;
            current[static_cast<std::size_t>(component)] +=
                phi * capture.current_interior[state_index];
            previous[static_cast<std::size_t>(component)] +=
                phi * capture.previous_interior[state_index];
            double sign = 1.0;
            const int source_dof = DecodeSignedDof(
                source_vdofs[scalar +
                    capture.scalar_interior_dofs * component], sign);
            local_valid = source_dof >= 0 &&
                static_cast<std::size_t>(source_dof) <
                    request.source.scalar_count &&
                std::isfinite(request.source.data[source_dof]);
            if (!local_valid) break;
            source[static_cast<std::size_t>(component)] += phi * sign *
                request.source.data[source_dof];
          }
          sigma += phi * capture.current_interior[
              interior_offset + scalar +
              static_cast<std::size_t>(capture.scalar_interior_dofs) * 4];
          for (int component = 0; component < 3; ++component)
            for (int direction = 0; direction < 2; ++direction)
              conservative_gradient[component][direction] +=
                  static_cast<long double>(
                      volume_derivative(scalar, direction)) *
                  capture.current_interior[
                      interior_offset + scalar +
                      static_cast<std::size_t>(
                          capture.scalar_interior_dofs) * component];
        }
        local_valid = local_valid && current[0] >
            static_cast<long double>(qualification_density_gate) &&
            std::isfinite(sigma);
        for (int component = 0;
             local_valid && component < components; ++component)
        {
          const long double change = weight *
              (current[static_cast<std::size_t>(component)] -
               previous[static_cast<std::size_t>(component)]);
          const long double forcing =
              static_cast<long double>(capture.dt) * weight *
              source[static_cast<std::size_t>(component)];
          local_valid = std::isfinite(change) && std::isfinite(forcing);
          local_balance[static_cast<std::size_t>(component)] +=
              change - forcing;
          local_scale[static_cast<std::size_t>(component)] +=
              std::abs(change) + std::abs(forcing);
          local_change_integral[static_cast<std::size_t>(component)] +=
              change;
          local_source_integral[static_cast<std::size_t>(component)] +=
              forcing;
          local_activity[static_cast<std::size_t>(component)] +=
              std::abs(forcing);
          local_activity[4] += std::abs(change);
        }

        long double velocity_gradient[2][2] = {};
        for (int velocity = 0; velocity < 2; ++velocity)
          for (int direction = 0; direction < 2; ++direction)
            velocity_gradient[velocity][direction] =
                conservative_gradient[1 + velocity][direction] /
                    current[0] -
                current[static_cast<std::size_t>(1 + velocity)] *
                    conservative_gradient[0][direction] /
                    (current[0] * current[0]);
        const long double velocity_trace =
            velocity_gradient[0][0] + velocity_gradient[1][1];
        const long double q_value =
            velocity_gradient[0][0] * velocity_gradient[0][0] +
            2.0L * velocity_gradient[0][1] * velocity_gradient[1][0] +
            velocity_gradient[1][1] * velocity_gradient[1][1] +
            velocity_trace * velocity_trace;
        const long double sigma_term = weight * sigma / current[0];
        const long double q_term = weight *
            static_cast<long double>(capture.alpha) * q_value;
        local_valid = local_valid && std::isfinite(q_value) &&
            std::isfinite(sigma_term) && std::isfinite(q_term);
        local_balance[4] += sigma_term - q_term;
        local_scale[4] += std::abs(sigma_term) + std::abs(q_term);
        local_auxiliary_integral[0] += sigma_term;
        local_auxiliary_integral[1] += q_term;
        local_activity[6] += std::abs(sigma_term);
        local_activity[7] += std::abs(q_term);
      }

      mfem::Array<int> faces;
      mfem::Array<int> orientations;
      mesh.GetElementEdges(element, faces, orientations);
      local_valid = local_valid && faces.Size() == 4;
      for (int local_face = 0;
           local_valid && local_face < faces.Size(); ++local_face)
      {
        const int face = faces[local_face];
        mfem::FaceElementTransformations* face_transformation =
            mesh.GetFaceElementTransformations(face);
        const mfem::FiniteElement* trace_element =
            trace_space.GetFaceElement(face);
        const int boundary_element = face >= 0 &&
                face < static_cast<int>(shared_faces.size())
            ? mesh.GetFaceToBdrElMap()[face] : -1;
        const bool physical = face >= 0 &&
            face < static_cast<int>(shared_faces.size()) &&
            shared_faces[static_cast<std::size_t>(face)] == 0 &&
            face_transformation != nullptr &&
            face_transformation->Elem2No < 0 && boundary_element >= 0;
        if (!physical) continue;
        local_valid = face_transformation->Face != nullptr &&
            trace_element != nullptr &&
            trace_element->GetDof() == capture.scalar_face_dofs &&
            mesh.GetBdrAttribute(boundary_element) >= 1 &&
            mesh.GetBdrAttribute(boundary_element) <= 7;
        if (!local_valid) break;
        const bool element_one =
            face_transformation->Elem1No == element;
        local_valid = element_one ||
            face_transformation->Elem2No == element;
        if (!local_valid) break;

        const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
            trace_element->GetGeomType(), capture.face_assembly_order);
        local_valid = face_rule.GetNPoints() == capture.face_points;
        long double face_measure = 0.0L;
        for (int q = 0; local_valid && q < face_rule.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
          face_transformation->SetAllIntPoints(&point);
          const long double contribution = point.weight *
              face_transformation->Face->Weight();
          local_valid = contribution > 0.0L &&
              std::isfinite(contribution);
          face_measure += contribution;
        }
        const long double h =
            static_cast<long double>(mesh.GetElementVolume(element)) /
            face_measure;
        local_valid = local_valid && face_measure > 0.0L && h > 0.0L &&
            std::isfinite(h);
        mfem::Vector interior_shape(capture.scalar_interior_dofs);
        mfem::Vector trace_shape(capture.scalar_face_dofs);
        const std::size_t trace_offset =
            static_cast<std::size_t>(element) * capture.trace_dofs +
            static_cast<std::size_t>(local_face) * 5 *
                capture.scalar_face_dofs;
        for (int q = 0;
             local_valid && q < face_rule.GetNPoints(); ++q)
        {
          const mfem::IntegrationPoint& point = face_rule.IntPoint(q);
          face_transformation->SetAllIntPoints(&point);
          const mfem::IntegrationPoint& element_point = element_one
              ? face_transformation->GetElement1IntPoint()
              : face_transformation->GetElement2IntPoint();
          mfem::Vector scaled_normal(2);
          mfem::CalcOrtho(
              face_transformation->Face->Jacobian(), scaled_normal);
          const long double normal_norm = scaled_normal.Norml2();
          const long double measure =
              face_transformation->Face->Weight();
          local_valid = normal_norm > 0.0L && measure > 0.0L &&
              std::isfinite(normal_norm) && std::isfinite(measure);
          if (!local_valid) break;
          const long double side = element_one ? 1.0L : -1.0L;
          const long double normal_x =
              side * scaled_normal[0] / normal_norm;
          const long double normal_y =
              side * scaled_normal[1] / normal_norm;
          finite_element->CalcShape(element_point, interior_shape);
          trace_element->CalcShape(point, trace_shape);
          std::array<long double, 7> interior{};
          std::array<long double, 5> trace{};
          for (int scalar = 0;
               scalar < capture.scalar_interior_dofs; ++scalar)
            for (int component = 0; component < 7; ++component)
              interior[static_cast<std::size_t>(component)] +=
                  static_cast<long double>(interior_shape[scalar]) *
                  capture.current_interior[
                      interior_offset + scalar +
                      static_cast<std::size_t>(
                          capture.scalar_interior_dofs) * component];
          for (int scalar = 0; scalar < capture.scalar_face_dofs; ++scalar)
            for (int component = 0; component < 5; ++component)
              trace[static_cast<std::size_t>(component)] +=
                  static_cast<long double>(trace_shape[scalar]) *
                  capture.current_local_trace[
                      trace_offset + scalar +
                      static_cast<std::size_t>(capture.scalar_face_dofs) *
                          component];
          std::array<long double, 4> flux{};
          local_valid = IndependentRegularizedNormalFlux(
              trace, normal_x, normal_y, flux);
          const long double weight = point.weight * measure;
          for (int component = 0;
               local_valid && component < components; ++component)
          {
            const long double numerical_flux =
                flux[static_cast<std::size_t>(component)] +
                static_cast<long double>(capture.beta) *
                    (interior[static_cast<std::size_t>(component)] -
                     trace[static_cast<std::size_t>(component)]);
            const long double contribution =
                static_cast<long double>(capture.dt) * weight *
                numerical_flux;
            local_valid = std::isfinite(contribution);
            local_balance[static_cast<std::size_t>(component)] +=
                contribution;
            local_scale[static_cast<std::size_t>(component)] +=
                std::abs(contribution);
            local_boundary_integral[static_cast<std::size_t>(component)] +=
                contribution;
            local_activity[5] += std::abs(contribution);
          }
          local_valid = local_valid && trace[0] >
              static_cast<long double>(qualification_density_gate);
          const long double degree_scale =
              static_cast<long double>((capture.order + 1) *
                                       (capture.order + 1));
          const long double tau = degree_scale / (trace[0] * h);
          const long double z_hat_normal =
              interior[5] * normal_x + interior[6] * normal_y -
              tau * (interior[4] - trace[4]);
          const long double auxiliary_contribution =
              static_cast<long double>(capture.alpha) * weight *
              z_hat_normal;
          local_valid = local_valid && tau > 0.0L &&
              std::isfinite(tau) && std::isfinite(auxiliary_contribution);
          local_balance[4] -= auxiliary_contribution;
          local_scale[4] += std::abs(auxiliary_contribution);
          local_auxiliary_integral[2] += auxiliary_contribution;
          local_activity[8] += std::abs(auxiliary_contribution);
          long double trace_partition = 0.0L;
          for (int scalar = 0; scalar < capture.scalar_face_dofs; ++scalar)
            trace_partition += trace_shape[scalar];
          const long double probe_delta =
              -static_cast<long double>(capture.alpha) * weight * tau *
              static_cast<long double>(
                  CudaPreparedIgrConservationCapture::sigma_hat_probe) *
              trace_partition;
          local_valid = local_valid && std::isfinite(trace_partition) &&
              std::isfinite(probe_delta);
          local_probe_delta += probe_delta;
        }
      }
    }
  }
  catch (...) { local_valid = false; }

  if (!All(local_valid, request.communicator)) return false;

  std::array<double, 5> production_local_balance{};
  std::array<double, 5> production_local_scale{};
  std::copy_n(capture.production_local.begin(), 5,
              production_local_balance.begin());
  std::copy_n(capture.production_local.begin() + 5, 5,
              production_local_scale.begin());
  std::array<long double, 5> global_balance{};
  std::array<long double, 5> global_scale{};
  std::array<long double, 9> global_activity{};
  std::array<long double, 4> global_change_integral{};
  std::array<long double, 4> global_source_integral{};
  std::array<long double, 4> global_boundary_integral{};
  std::array<long double, 3> global_auxiliary_integral{};
  const double production_local_domain_measure =
      capture.production_local[2 * igr_trace_components];
  long double global_domain_measure = 0.0L;
  const double production_local_probe_delta =
      capture.probed_production_local[4] - capture.production_local[4];
  long double global_probe_delta = 0.0L;
  const int production_balance_status = MPI_Allreduce(
      production_local_balance.data(), result.production_balance.data(), 5,
      MPI_DOUBLE, MPI_SUM, request.communicator);
  const int production_scale_status = MPI_Allreduce(
      production_local_scale.data(), result.production_scale.data(), 5,
      MPI_DOUBLE, MPI_SUM, request.communicator);
  const int oracle_balance_status = MPI_Allreduce(
      local_balance.data(), global_balance.data(), 5, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int oracle_scale_status = MPI_Allreduce(
      local_scale.data(), global_scale.data(), 5, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int production_measure_status = MPI_Allreduce(
      &production_local_domain_measure, &result.production_domain_measure,
      1, MPI_DOUBLE, MPI_SUM, request.communicator);
  const int oracle_measure_status = MPI_Allreduce(
      &local_domain_measure, &global_domain_measure, 1, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int activity_status = MPI_Allreduce(
      local_activity.data(), global_activity.data(), 9, MPI_LONG_DOUBLE,
      MPI_SUM, request.communicator);
  const int change_status = MPI_Allreduce(
      local_change_integral.data(), global_change_integral.data(), 4,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int source_status = MPI_Allreduce(
      local_source_integral.data(), global_source_integral.data(), 4,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int boundary_status = MPI_Allreduce(
      local_boundary_integral.data(), global_boundary_integral.data(), 4,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int auxiliary_status = MPI_Allreduce(
      local_auxiliary_integral.data(), global_auxiliary_integral.data(), 3,
      MPI_LONG_DOUBLE, MPI_SUM, request.communicator);
  const int production_probe_status = MPI_Allreduce(
      &production_local_probe_delta, &result.production_probe_delta, 1,
      MPI_DOUBLE, MPI_SUM, request.communicator);
  const int oracle_probe_status = MPI_Allreduce(
      &local_probe_delta, &global_probe_delta, 1, MPI_LONG_DOUBLE, MPI_SUM,
      request.communicator);
  const bool reduced = production_balance_status == MPI_SUCCESS &&
      production_scale_status == MPI_SUCCESS &&
      oracle_balance_status == MPI_SUCCESS &&
      oracle_scale_status == MPI_SUCCESS &&
      production_measure_status == MPI_SUCCESS &&
      oracle_measure_status == MPI_SUCCESS &&
      activity_status == MPI_SUCCESS && change_status == MPI_SUCCESS &&
      source_status == MPI_SUCCESS && boundary_status == MPI_SUCCESS &&
      auxiliary_status == MPI_SUCCESS &&
      production_probe_status == MPI_SUCCESS &&
      oracle_probe_status == MPI_SUCCESS;
  if (!All(reduced, request.communicator)) return false;

  bool parity = true;
  double production_maximum_error = 0.0;
  double production_maximum_relative_error = 0.0;
  result.maximum_balance_discrepancy = 0.0;
  result.maximum_scale_discrepancy = 0.0;
  result.oracle_domain_measure = static_cast<double>(global_domain_measure);
  result.domain_measure_discrepancy = std::abs(
      result.production_domain_measure - result.oracle_domain_measure);
  const double domain_measure_tolerance = 2.0e-11 *
      (1.0 + std::abs(result.oracle_domain_measure));
  parity = parity && std::isfinite(result.production_domain_measure) &&
      result.production_domain_measure > 0.0 &&
      std::isfinite(result.oracle_domain_measure) &&
      result.oracle_domain_measure > 0.0 &&
      result.domain_measure_discrepancy <= domain_measure_tolerance;
  std::array<double, 5> comparison_tolerance{};
  for (std::size_t component = 0; component < 5; ++component)
  {
    result.oracle_balance[component] =
        static_cast<double>(global_balance[component]);
    result.oracle_scale[component] =
        static_cast<double>(global_scale[component]);
    const double balance_discrepancy = std::abs(
        result.production_balance[component] -
        result.oracle_balance[component]);
    const double scale_discrepancy = std::abs(
        result.production_scale[component] - result.oracle_scale[component]);
    comparison_tolerance[component] =
        2.0e-11 * (1.0 + std::abs(result.oracle_scale[component]));
    result.maximum_balance_discrepancy = std::max(
        result.maximum_balance_discrepancy, balance_discrepancy);
    result.maximum_scale_discrepancy = std::max(
        result.maximum_scale_discrepancy, scale_discrepancy);
    parity = parity && std::isfinite(result.production_balance[component]) &&
        std::isfinite(result.production_scale[component]) &&
        std::isfinite(result.oracle_balance[component]) &&
        std::isfinite(result.oracle_scale[component]) &&
        balance_discrepancy <= comparison_tolerance[component] &&
        scale_discrepancy <= comparison_tolerance[component] &&
        result.production_scale[component] > 0.0;
    const double error = std::abs(result.production_balance[component]);
    production_maximum_error = std::max(
        production_maximum_error, error);
    production_maximum_relative_error = std::max(
        production_maximum_relative_error,
        dg::hdg::generic::EulerHdgConservationRelativeError(
            error, result.production_scale[component],
            result.production_domain_measure));
  }
  for (std::size_t index = 0; index < result.activity.size(); ++index)
    result.activity[index] = static_cast<double>(global_activity[index]);

  const double lifecycle_error_tolerance = 64.0 *
      std::numeric_limits<double>::epsilon() *
      (1.0 + production_maximum_error);
  const double lifecycle_relative_tolerance = 64.0 *
      std::numeric_limits<double>::epsilon() *
      (1.0 + production_maximum_relative_error);
  // A partitioned step admits the conservative Euler solve against the
  // lagged auxiliary state a^n, then projects the coherent output auxiliary
  // state a^{n+1}.  The live oracle below intentionally recomputes balance on
  // that final state, so it must not be bitwise compared with the lifecycle
  // witness for the earlier (and different) Euler equation.  Preserve exact
  // witness parity for the monolithic path and independently require the
  // partitioned lifecycle witness to have passed its configured gate.
  const bool partitioned_lifecycle = request.time_treatment ==
      EulerHdgTimeTreatment::partitioned_backward_euler;
  const bool lifecycle_parity = partitioned_lifecycle ?
      std::isfinite(capture.lifecycle_maximum_error) &&
          capture.lifecycle_maximum_error >= 0.0 &&
          std::isfinite(capture.lifecycle_maximum_relative_error) &&
          capture.lifecycle_maximum_relative_error >= 0.0 &&
          capture.lifecycle_maximum_relative_error <=
              request.numerical_policy.conservation_relative_tolerance :
      std::abs(production_maximum_error -
               capture.lifecycle_maximum_error) <=
          lifecycle_error_tolerance &&
      std::abs(production_maximum_relative_error -
               capture.lifecycle_maximum_relative_error) <=
          lifecycle_relative_tolerance;
  const bool active =
      std::all_of(result.activity.begin(), result.activity.begin() + 4,
                  [](const double value)
                  { return std::isfinite(value) && value > 1.0e-12; }) &&
      result.activity[4] > 1.0e-10 &&
      result.activity[5] > 1.0e-10 &&
      result.activity[6] > 1.0e-10 &&
      result.activity[7] > 1.0e-10 &&
      result.activity[8] > 1.0e-12;
  long double maximum_change = 0.0L;
  long double maximum_source = 0.0L;
  long double maximum_boundary = 0.0L;
  for (const long double value : global_change_integral)
    maximum_change = std::max(maximum_change, std::abs(value));
  for (const long double value : global_source_integral)
    maximum_source = std::max(maximum_source, std::abs(value));
  for (const long double value : global_boundary_integral)
    maximum_boundary = std::max(maximum_boundary, std::abs(value));
  const long double euler_omission_tolerance = 100.0L *
      *std::max_element(comparison_tolerance.begin(),
                        comparison_tolerance.begin() + 4);
  const long double auxiliary_omission_tolerance =
      100.0L * comparison_tolerance[4];
  result.oracle_probe_delta = static_cast<double>(global_probe_delta);
  result.probe_discrepancy = std::abs(
      result.production_probe_delta - result.oracle_probe_delta);
  const double probe_tolerance = 2.0e-11 *
      (1.0 + std::abs(result.oracle_probe_delta));
  const bool signed_probe_sensitive =
      std::isfinite(result.production_probe_delta) &&
      std::isfinite(result.oracle_probe_delta) &&
      result.oracle_probe_delta < -probe_tolerance &&
      result.production_probe_delta < -probe_tolerance &&
      result.probe_discrepancy <= probe_tolerance;
  const bool omission_detectable =
      maximum_change > euler_omission_tolerance &&
      maximum_source > euler_omission_tolerance &&
      maximum_boundary > euler_omission_tolerance &&
      std::abs(global_auxiliary_integral[0]) >
          auxiliary_omission_tolerance &&
      std::abs(global_auxiliary_integral[1]) >
          auxiliary_omission_tolerance &&
      global_activity[8] > 1.0e-12L && signed_probe_sensitive;
  result.passed = parity && lifecycle_parity && active &&
      omission_detectable;
  if (!result.passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "independent IGR conservation gate detail parity="
                << parity << " lifecycle=" << lifecycle_parity
                << " active=" << active
                << " omission=" << omission_detectable
                << " production_error=" << production_maximum_error
                << " lifecycle_error=" << capture.lifecycle_maximum_error
                << " production_relative="
                << production_maximum_relative_error
                << " lifecycle_relative="
                << capture.lifecycle_maximum_relative_error << '\n';
  }
  return All(result.passed, request.communicator);
}

bool VerifyLocalCondensationAgreement(
    const CudaPreparedLocalCondensationCapture& capture,
    double& maximum_schur_scaled_error,
    double& maximum_rhs_scaled_error,
    double& maximum_trace_action_scaled_error) noexcept
{
  maximum_schur_scaled_error = std::numeric_limits<double>::infinity();
  maximum_rhs_scaled_error = std::numeric_limits<double>::infinity();
  maximum_trace_action_scaled_error =
      std::numeric_limits<double>::infinity();
  if (capture.elements <= 0 || capture.interior_dofs <= 0 ||
      capture.trace_dofs <= 0 || capture.tensor_mma_tiles == 0 ||
      capture.ordinary_schur.empty() || capture.ordinary_rhs.empty() ||
      capture.ordinary_schur.size() != capture.tensor_schur.size() ||
      capture.ordinary_rhs.size() != capture.tensor_rhs.size())
    return false;
  const std::uint64_t schur_only_tiles =
      static_cast<std::uint64_t>(capture.elements) *
      static_cast<std::uint64_t>((capture.trace_dofs + 7) / 8) *
      static_cast<std::uint64_t>((capture.trace_dofs + 7) / 8) *
      static_cast<std::uint64_t>((capture.interior_dofs + 3) / 4);
  if (capture.tensor_mma_tiles <= schur_only_tiles) return false;

  const auto maximum_scaled_error = [](const std::vector<double>& ordinary,
                                       const std::vector<double>& tensor,
                                       double& maximum) noexcept
  {
    maximum = 0.0;
    for (std::size_t index = 0; index < ordinary.size(); ++index)
    {
      if (!std::isfinite(ordinary[index]) || !std::isfinite(tensor[index]))
        return false;
      const double scale = std::max(
          {1.0, std::abs(ordinary[index]), std::abs(tensor[index])});
      maximum = std::max(
          maximum, std::abs(ordinary[index] - tensor[index]) / scale);
    }
    return true;
  };
  const bool finite = maximum_scaled_error(
          capture.ordinary_schur, capture.tensor_schur,
          maximum_schur_scaled_error) &&
      maximum_scaled_error(
          capture.ordinary_rhs, capture.tensor_rhs,
          maximum_rhs_scaled_error);
  bool action_finite = finite;
  maximum_trace_action_scaled_error = 0.0;
  const std::size_t trace_dofs =
      static_cast<std::size_t>(capture.trace_dofs);
  for (int element = 0; action_finite && element < capture.elements;
       ++element)
  {
    const std::size_t matrix_offset =
        static_cast<std::size_t>(element) * trace_dofs * trace_dofs;
    for (std::size_t row = 0; row < trace_dofs; ++row)
    {
      double ordinary_action = 0.0;
      double tensor_action = 0.0;
      for (std::size_t column = 0; column < trace_dofs; ++column)
      {
        const double probe =
            0.125 * static_cast<double>(static_cast<int>(column % 11) - 5) +
            0.01 * static_cast<double>(element % 3);
        const std::size_t index = matrix_offset + row * trace_dofs + column;
        ordinary_action += capture.ordinary_schur[index] * probe;
        tensor_action += capture.tensor_schur[index] * probe;
      }
      action_finite = action_finite && std::isfinite(ordinary_action) &&
          std::isfinite(tensor_action);
      const double scale = std::max(
          {1.0, std::abs(ordinary_action), std::abs(tensor_action)});
      maximum_trace_action_scaled_error = std::max(
          maximum_trace_action_scaled_error,
          std::abs(ordinary_action - tensor_action) / scale);
    }
  }
  // The tensor leaf reorders the FP64 factor, multi-RHS solves, and Schur
  // contraction into scalar diagonal panels plus m8n8k4 updates.  The
  // deterministic Schur-vector probe also qualifies the local true-trace
  // action used by global assembly; lifecycle tests independently compare the
  // assembled solve and true residual.  Retain the established strict
  // dimension-scaled roundoff envelopes for both leaves.
  const double tolerance = 512.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(1, capture.interior_dofs));
  const double action_tolerance = tolerance *
      static_cast<double>(std::max(1, capture.trace_dofs));
  return finite && action_finite &&
      maximum_schur_scaled_error <= tolerance &&
      maximum_rhs_scaled_error <= tolerance &&
      maximum_trace_action_scaled_error <= action_tolerance;
}

template <typename Selection>
bool CheckEulerAlgebraForSelection(const EulerHdgRequest& request,
                                   const int ranks,
                                   const int order)
{
  CudaConfig config = MakeDirectConfig(request, ranks, order);

  auto engine = std::make_unique<
      PersistentCudaSmoothEulerEngine<Selection>>(config);
  CudaRunEvidence initialization{};
  std::array<double, 6> jacobian_errors{};
  std::array<double, 7> llf_evidence{};
  std::array<double, 2> coupling_norms{};
  std::array<int, 6> work_counts{};
  constexpr double jacobian_tolerance = 5.0e-5;
  const bool initialized = engine->Initialize(initialization);
  const bool initialized_everywhere = All(initialized, request.communicator);
  if (!initialized_everywhere) return false;
  const bool igr = request.numerical_policy.stabilization.kind ==
      RegularizationKind::igr;
  CudaBorrowedAssemblyFusionAudit assembly_fusion_audit{};
  const bool assembly_fusion =
      config.cuda_engine != CudaEngineKind::tensor_core ||
      CudaQualificationTestAccess::AuditBorrowedAssemblyFusion(
          *engine, assembly_fusion_audit);
  const bool assembly_fusion_everywhere = All(
      assembly_fusion, request.communicator);
  if (!assembly_fusion_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA split/tensor-volume assembly mismatch order="
                << order << " regularization=" << (igr ? "igr" : "none")
                << " dimensions=" << assembly_fusion_audit.elements << 'x'
                << assembly_fusion_audit.interior_dofs << 'x'
                << assembly_fusion_audit.trace_dofs << " maxima=";
      for (const double error :
           assembly_fusion_audit.maximum_absolute_errors)
        std::cerr << ' ' << error;
      std::cerr << " exact=";
      for (const bool exact : assembly_fusion_audit.bitwise_equal)
        std::cerr << ' ' << exact;
      std::cerr << " scaled=";
      for (const double error :
           assembly_fusion_audit.maximum_scaled_errors)
        std::cerr << ' ' << error;
      std::cerr << " status="
                << assembly_fusion_audit.status_bitwise_equal << '/'
                << assembly_fusion_audit.status_all_passed << '\n';
    }
    return false;
  }
  const bool llf_policy = igr ||
      CudaQualificationTestAccess::VerifyCurrentCandidateLlfPolicy(
          *engine, llf_evidence);
  const bool llf_policy_everywhere = All(llf_policy, request.communicator);
  if (!llf_policy_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA current-candidate LLF qualification failed order="
                << order << " evidence=";
      for (const double value : llf_evidence) std::cerr << ' ' << value;
      std::cerr << '\n';
    }
    return false;
  }
  const bool jacobian = CudaQualificationTestAccess::VerifyAssemblerJacobian(
      *engine, jacobian_tolerance, jacobian_errors);
  const bool jacobian_everywhere = All(jacobian, request.communicator);
  if (!jacobian_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA Euler Jacobian qualification failed order=" << order
                << " regularization="
                << (request.numerical_policy.stabilization.kind ==
                        RegularizationKind::igr ? "igr" : "none")
                << " relative_errors=";
      for (const double error : jacobian_errors) std::cerr << ' ' << error;
      std::cerr << '\n';
    }
    return false;
  }
  const bool coupled = !igr ||
      CudaQualificationTestAccess::VerifyIgrCouplingBlocks(
          *engine, coupling_norms);
  const bool coupled_everywhere = All(coupled, request.communicator);
  if (!coupled_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA IGR coupling qualification failed order=" << order
                << " norms=" << coupling_norms[0] << ' '
                << coupling_norms[1] << '\n';
    return false;
  }

  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence step{};
  constexpr double dt = 1.0e-5;
  const bool prepared = engine->PrepareStep(false, dt, snapshot, step);
  const bool prepared_everywhere = All(prepared, request.communicator);
  if (!prepared_everywhere) return false;
  CudaPreparedRecoveryAudit recovery_audit{};
  const bool recovery = CudaQualificationTestAccess::AuditPreparedRecovery(
      *engine, recovery_audit);
  const bool recovery_everywhere = All(recovery, request.communicator);
  if (!recovery_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA local-algebra/recovery mismatch order=" << order
                << " regularization=" << (igr ? "igr" : "none")
                << " engine="
                << (config.cuda_engine == CudaEngineKind::tensor_core
                        ? "tensor" : "ordinary")
                << " dimensions=" << recovery_audit.elements << 'x'
                << recovery_audit.interior_dofs << 'x'
                << recovery_audit.trace_dofs
                << " direct_scaled_error="
                << recovery_audit.maximum_direct_scaled_error
                << " workspace_scaled_error="
                << recovery_audit.maximum_workspace_scaled_error
                << " factor_scaled_residual="
                << recovery_audit.maximum_factor_scaled_residual
                << " workspace_solve_scaled_residual="
                << recovery_audit.maximum_workspace_solve_scaled_residual
                << " scaled_residual="
                << recovery_audit.maximum_scaled_residual
                << " tolerance=" << recovery_audit.tolerance << '\n';
    return false;
  }
  CudaPreparedLocalCondensationCapture condensation_capture{};
  double schur_scaled_error = 0.0;
  double rhs_scaled_error = 0.0;
  double trace_action_scaled_error = 0.0;
  const bool condensation_pair =
      config.cuda_engine != CudaEngineKind::tensor_core ||
      (CudaQualificationTestAccess::CapturePreparedLocalCondensationPair(
           *engine, condensation_capture) &&
       VerifyLocalCondensationAgreement(
           condensation_capture, schur_scaled_error, rhs_scaled_error,
           trace_action_scaled_error));
  const bool condensation_pair_everywhere = All(
      condensation_pair, request.communicator);
  if (!condensation_pair_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA ordinary/tensor local condensation mismatch order="
                << order << " regularization=" << (igr ? "igr" : "none")
                << " schur_scaled_error=" << schur_scaled_error
                << " rhs_scaled_error=" << rhs_scaled_error
                << " trace_action_scaled_error="
                << trace_action_scaled_error
                << " tensor_mma_tiles="
                << condensation_capture.tensor_mma_tiles << '\n';
    return false;
  }
  const bool native_rhs_refresh =
      config.cuda_engine != CudaEngineKind::tensor_core || !igr || ranks != 1 ||
      CudaQualificationTestAccess::AuditNativeRankOneTrueRhsRefresh(*engine);
  const bool native_rhs_refresh_everywhere = All(
      native_rhs_refresh, request.communicator);
  if (!native_rhs_refresh_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA native rank-one RHS-only refresh failed order="
                << order << '\n';
    return false;
  }
  const bool work_accounting =
      CudaQualificationTestAccess::VerifyPreparedWorkAccounting(
          *engine, step, true, work_counts);
  const bool work_accounting_everywhere = All(
      work_accounting, request.communicator);
  if (!work_accounting_everywhere)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA prepared work accounting failed order=" << order
                << " regularization=" << (igr ? "igr" : "none")
                << " counts=";
      for (const int count : work_counts) std::cerr << ' ' << count;
      std::cerr << '\n';
    }
    return false;
  }

  CudaPreparedOutputMetadataV1 output{};
  const bool staged = engine->StagePreparedOutputMetadata(output, step);
  const bool staged_everywhere = All(staged, request.communicator);
  if (!staged_everywhere) return false;
  const bool acknowledged = engine->MarkOutputStaged(output);
  const bool acknowledged_everywhere = All(
      acknowledged, request.communicator);
  if (!acknowledged_everywhere) return false;
  const bool commit_prepared = engine->PrepareCommit(output, step);
  const bool commit_prepared_everywhere = All(
      commit_prepared, request.communicator);
  if (!commit_prepared_everywhere) return false;
  const bool provisional = engine->CommitPrepared(output, step);
  const bool provisional_everywhere = All(
      provisional, request.communicator);
  if (!provisional_everywhere) return false;
  const bool committed = engine->FinalizeCommit(output, step);
  const bool committed_work_accounting =
      step.nonlinear_updates == work_counts[0] &&
      step.trace_solves == work_counts[1] &&
      step.linear_iterations == work_counts[2] &&
      step.line_search_trials == work_counts[3] &&
      step.rejected_line_search_trials == work_counts[4];
  const bool passed = committed && output.successful_commit &&
      committed_work_accounting;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA Euler algebra qualification failed order=" << order
                << " regularization=" << (igr ? "igr" : "none")
                << " initialized=" << initialized
                << " llf_policy=" << llf_policy
                << " jacobian=" << jacobian
                << " coupling=" << coupled
                << " prepared=" << prepared
                << " condensation_pair=" << condensation_pair
                << " work_accounting=" << work_accounting
                << " committed_work_accounting="
                << committed_work_accounting
                << " jacobian_errors=";
      for (const double error : jacobian_errors) std::cerr << ' ' << error;
      std::cerr << " coupling_norms=" << coupling_norms[0] << ' '
                << coupling_norms[1];
      std::cerr << '\n';
    }
  }
  return All(passed, request.communicator);
}

bool CheckCopiedTraceForGenericSession(
    const EulerHdgRequest& request, BorrowedFields& fields,
    const std::vector<double>& expected_trace, const int order)
{
  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized =
      initialization.status == HdgEulerSessionStatus::initialized;
  const bool initialized_everywhere = All(
      initialized, request.communicator);
  if (!initialized_everywhere) return false;

  const double poisoned = std::numeric_limits<double>::quiet_NaN();
  std::fill(fields.initial_true_trace.begin(),
            fields.initial_true_trace.end(), poisoned);
  const bool copied_trace =
      CudaQualificationTestAccess::VerifyBorrowedInitialTrueTrace(
          session,
      CudaBorrowedHostL2Input{
          expected_trace.data(), expected_trace.size() /
              static_cast<std::size_t>(components), components});
  const bool passed = initialized && copied_trace;
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA copied-trace qualification failed order="
                << order << " initialized=" << initialized
                << " copied_trace=" << copied_trace << '\n';
    }
  }
  return All(passed, request.communicator);
}

bool CheckCopiedInitialTrueTrace(const int order)
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty);
  const std::vector<double> restricted_trace = fields.initial_true_trace;
  const std::size_t trace_scalars =
      fields.initial_true_trace.size() / static_cast<std::size_t>(components);
  for (std::size_t scalar = 0; scalar < trace_scalars; ++scalar)
  {
    fields.initial_true_trace[scalar] +=
        5.0e-4 * static_cast<double>(1 + scalar % 3);
    fields.initial_true_trace[trace_scalars + scalar] -=
        3.0e-4 * static_cast<double>(1 + scalar % 2);
    fields.initial_true_trace[2 * trace_scalars + scalar] +=
        2.0e-4 * static_cast<double>(1 + scalar % 4);
    fields.initial_true_trace[3 * trace_scalars + scalar] += 7.0e-4;
  }
  const std::vector<double> expected_trace = fields.initial_true_trace;
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  const bool intentionally_distinct = restricted_trace != expected_trace;
  const bool seam = CheckCopiedTraceForGenericSession(
      request, fields, expected_trace, order);
  return All(intentionally_distinct && seam, request.communicator);
}

bool CheckAllSevenLoweredBoundaryPolicies(const int order)
{
  mfem::ParMesh mesh = MakeAllSevenPhysicalMesh();
  const auto check = [&](const RegularizationKind kind)
  {
    BorrowedFields fields = MakeBorrowedFields(
        mesh, order, SourceMode::nonzero);
    BoundaryContext unused_boundary{};
    EulerHdgRequest request = MakeRequest(
        mesh, order, fields, unused_boundary, false, kind);
    AllSevenBoundaryContext boundary{};
    ConfigureAllSevenPolicies(boundary);
    const std::array<EulerBoundaryPolicy, 7> expected = boundary.policies;
    request.boundary.context = &boundary;
    request.boundary.exterior_state = AllSevenPhysicalEulerExterior;
    request.boundary.policies = {
        boundary.policies.data(), boundary.policies.size(), 2, 1.4};
    if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
        ContractStatus::accepted)
      return false;

    GenericEulerHdgCudaSession session;
    const auto initialization = session.Initialize(request);
    const bool initialized = All(
        initialization.status == HdgEulerSessionStatus::initialized,
        request.communicator);
    if (!initialized) return false;

    // The facade's temporary lowered table and the caller's generic table are
    // both gone or poisoned before inspection and execution.  The engine must
    // use its copied host/device policy maps in both 4/4 and 7/5 modes.
    for (EulerBoundaryPolicy& policy : boundary.policies)
    {
      policy.kind = static_cast<EulerBoundaryPolicyKind>(255);
      policy.reference_state.fill(
          std::numeric_limits<double>::quiet_NaN());
      policy.pressure_out = std::numeric_limits<double>::quiet_NaN();
    }
    const bool copied_and_associated =
        CudaQualificationTestAccess::VerifyBoundaryPolicyTable(
            session, expected.data(), expected.size(), 1.4);

    constexpr double dt = 1.0e-5;
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    const bool prepared = All(
        session.Prepare(dt, authority, output).status ==
            HdgEulerSessionStatus::prepared,
        request.communicator);
    bool aborted = false;
    if (prepared)
      aborted = All(
          session.Abort(std::move(authority), output).status ==
              HdgEulerSessionStatus::aborted,
          request.communicator);

    std::array<int, 7> global_callback_calls{};
    const bool callback_census = MPI_Allreduce(
        boundary.calls.data(), global_callback_calls.data(),
        static_cast<int>(global_callback_calls.size()), MPI_INT, MPI_SUM,
        request.communicator) == MPI_SUCCESS;
    const bool callback_bypassed = callback_census && std::all_of(
        global_callback_calls.begin(), global_callback_calls.end(),
        [](const int calls) { return calls == 0; });
    const bool passed = All(
        copied_and_associated && prepared && aborted && callback_bypassed,
        request.communicator);
    if (!passed)
    {
      int rank = 0;
      MPI_Comm_rank(request.communicator, &rank);
      if (rank == 0)
        std::cerr << "CUDA all-seven boundary execution failed order="
                  << order << " regularization="
                  << (kind == RegularizationKind::igr ? "igr" : "none")
                  << " copied=" << copied_and_associated
                  << " prepared=" << prepared << " aborted=" << aborted
                  << " callback_bypassed=" << callback_bypassed << '\n';
    }
    return passed;
  };
  const bool conventional = check(RegularizationKind::none);
  const bool igr = check(RegularizationKind::igr);
  return conventional && igr;
}

bool CheckMixedLoweredBoundaryPolicies(const int order)
{
  mfem::ParMesh mesh = MakeMixedPeriodicMesh();
  const auto check = [&](const RegularizationKind kind)
  {
    BorrowedFields fields = MakeBorrowedFields(
        mesh, order, SourceMode::nonzero);
    BoundaryContext boundary{};
    EulerHdgRequest request = MakeRequest(
        mesh, order, fields, boundary, true, kind);
    // Periodic x-identification retains attributes 2 and 4, while attributes
    // 1 and 3 remain physical.  The facade must omit the retained periodic
    // entries and lower only the two physical closures.
    boundary.policies[1] = {
        EulerBoundaryPolicyKind::periodic_topology, {}, 0.0};
    boundary.policies[3] = {
        EulerBoundaryPolicyKind::periodic_topology, {}, 0.0};
    const std::array<EulerBoundaryPolicy, 4> expected = boundary.policies;
    if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
        ContractStatus::accepted)
      return false;

    GenericEulerHdgCudaSession session;
    const auto initialization = session.Initialize(request);
    const bool initialized = All(
        initialization.status == HdgEulerSessionStatus::initialized,
        request.communicator);
    if (!initialized) return false;

    for (EulerBoundaryPolicy& policy : boundary.policies)
    {
      policy.kind = static_cast<EulerBoundaryPolicyKind>(255);
      policy.reference_state.fill(
          std::numeric_limits<double>::quiet_NaN());
      policy.pressure_out = std::numeric_limits<double>::quiet_NaN();
    }
    const bool copied_and_associated =
        CudaQualificationTestAccess::VerifyBoundaryPolicyTable(
            session, expected.data(), expected.size(), 1.4);

    constexpr double dt = 1.0e-5;
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    const bool prepared = All(
        session.Prepare(dt, authority, output).status ==
            HdgEulerSessionStatus::prepared,
        request.communicator);
    bool aborted = false;
    if (prepared)
      aborted = All(
          session.Abort(std::move(authority), output).status ==
              HdgEulerSessionStatus::aborted,
          request.communicator);

    std::array<int, 4> global_callback_calls{};
    const bool callback_census = MPI_Allreduce(
        boundary.calls.data(), global_callback_calls.data(),
        static_cast<int>(global_callback_calls.size()), MPI_INT, MPI_SUM,
        request.communicator) == MPI_SUCCESS;
    const bool callback_bypassed = callback_census && std::all_of(
        global_callback_calls.begin(), global_callback_calls.end(),
        [](const int calls) { return calls == 0; });
    const bool passed = All(
        copied_and_associated && prepared && aborted && callback_bypassed,
        request.communicator);
    if (!passed)
    {
      int rank = 0;
      MPI_Comm_rank(request.communicator, &rank);
      if (rank == 0)
        std::cerr << "CUDA mixed boundary execution failed order="
                  << order << " regularization="
                  << (kind == RegularizationKind::igr ? "igr" : "none")
                  << " copied=" << copied_and_associated
                  << " prepared=" << prepared << " aborted=" << aborted
                  << " callback_bypassed=" << callback_bypassed << '\n';
    }
    return passed;
  };
  const bool conventional = check(RegularizationKind::none);
  const bool igr = check(RegularizationKind::igr);
  return conventional && igr;
}

bool CheckIndependentIgrConservation(const int order)
{
  mfem::ParMesh mesh = MakeAllSevenPhysicalMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::affine_physical);
  mfem::L2_FECollection source_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace source_space(
      &mesh, &source_collection, components, mfem::Ordering::byNODES);
  const auto source_evaluator =
      [](const mfem::Vector& point, mfem::Vector& source)
      {
        const double modulation = 1.0 + 0.1 * point[0] - 0.05 * point[1];
        source.SetSize(components);
        source[0] = 0.25 * modulation;
        source[1] = 0.125 * modulation;
        source[2] = -0.1875 * modulation;
        source[3] = 0.375 * modulation;
      };
  if (!InterpolateVolumeFieldHost(
          mesh, source_space, source_evaluator, fields.source))
    return false;

  BoundaryContext unused_boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, unused_boundary, false,
      RegularizationKind::igr);
  // A deliberately split policy proves that the production CUDA route does
  // not silently reuse the element rule for faces.
  request.numerical_policy.igr_assembly_quadrature_orders = {10, 8};
  AllSevenBoundaryContext boundary{};
  ConfigureAllSevenPolicies(boundary);
  request.boundary.context = &boundary;
  request.boundary.exterior_state = AllSevenPhysicalEulerExterior;
  request.boundary.policies = {
      boundary.policies.data(), boundary.policies.size(), 2, 1.4};
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized = All(
      initialization.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  if (!initialized) return false;

  constexpr double dt = 1.0e-5;
  GenericEulerCudaPreparedStep authority;
  GenericEulerHdgCudaSession::prepared_output_type output{};
  const auto prepared_result = session.Prepare(dt, authority, output);
  const bool prepared = All(
      prepared_result.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!prepared) return false;
  CudaPreparedIgrConservationCapture capture;
  const bool captured =
      CudaQualificationTestAccess::CapturePreparedIgrConservation(
          session, capture);
  IndependentIgrConservationResult oracle;
  const bool verified = captured && VerifyIndependentIgrConservation(
      capture, request, oracle);
  const auto aborted = session.Abort(std::move(authority), output);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted,
      request.communicator);
  const bool passed = All(
      captured && verified && aborted_everywhere, request.communicator);
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA independent IGR conservation oracle failed order="
                << order << " captured=" << captured
                << " verified=" << verified
                << " aborted=" << aborted_everywhere
                << " balance_discrepancy="
                << oracle.maximum_balance_discrepancy
                << " scale_discrepancy="
                << oracle.maximum_scale_discrepancy
                << " signed_probe=" << oracle.production_probe_delta << '/'
                << oracle.oracle_probe_delta
                << " probe_discrepancy=" << oracle.probe_discrepancy
                << " activity=";
      for (const double value : oracle.activity) std::cerr << ' ' << value;
      std::cerr << '\n';
    }
  }
  return passed;
}

bool CheckConventionalConservationGateRollback(const int order,
                                               const int ranks)
{
  // One focused rank-one/p1 case gives an analytic, source-only defect.  The
  // ordinary p1-p3/rank-one/rank-two lifecycle cases below retain the broad
  // supported-surface coverage without repeating this rollback fixture.
  if (order != 1 || ranks != 1) return true;

  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::constant);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  // This admits the initial iterate as the nonlinear candidate.  With a
  // periodic constant state, no physical boundary flux exists and U^n=U^{n+1};
  // the four conservation balances are therefore exactly -dt*integral(S).
  request.numerical_policy.nonlinear_relative_tolerance = 2.0;
  constexpr double dt = 1.0e-5;

  const auto run = [&](const double conservation_tolerance,
                       const bool expect_conservation,
                       CudaPreparedEulerConservationGateCapture& capture,
                       CudaStepEvidence& rollback) -> bool
  {
    request.numerical_policy.conservation_relative_tolerance =
        conservation_tolerance;
    if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
        ContractStatus::accepted)
      return false;
    CudaConfig config = MakeDirectConfig(request, ranks, order);
    auto engine = std::make_unique<PersistentCudaSmoothEulerEngine<
        BorrowedCudaRank1OneGpuRuntimeSelection>>(config);
    CudaRunEvidence initialization{};
    if (!All(engine->Initialize(initialization), request.communicator) ||
        engine->contract() == nullptr)
      return false;
    std::vector<double> committed_before(fields.initial.size(), -1.0);
    const bool exported_before = engine->ExportCommittedInterior(
        CudaMutableHostL2Output{
            committed_before.data(), fields.volume_scalar_count, components});
    const auto before = engine->contract()->coordinates();
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence step{};
    const bool prepared = engine->PrepareStep(false, dt, snapshot, step);
    if (!All(prepared, request.communicator)) return false;
    const bool factor_authority_inactive =
        CudaQualificationTestAccess::IluFactorAuthorityInactive(*engine);
    const bool captured = prepared &&
        CudaQualificationTestAccess::CapturePreparedEulerConservationGate(
            *engine, capture);
    const bool gate_state = factor_authority_inactive && captured &&
        snapshot.numerical_candidate_ready &&
        capture.pending_active && capture.staging_retained &&
        capture.candidate_ready && capture.nonlinear_updates == 0 &&
        capture.conservation == expect_conservation &&
        capture.success == expect_conservation &&
        capture.prestage_allowed == expect_conservation &&
        capture.reason == (expect_conservation
            ? FailureReason::none : FailureReason::success_gate);

    CudaPreparedOutputMetadataV1 output{};
    const bool staged = engine->StagePreparedOutputMetadata(output, step);
    const bool staging_state = staged == expect_conservation;
    const bool aborted = engine->AbortPrepared(
        output, FailureReason::backend_failure, rollback);
    const auto after = engine->contract()->coordinates();
    std::vector<double> committed_after(fields.initial.size(), -2.0);
    const bool exported_after = engine->ExportCommittedInterior(
        CudaMutableHostL2Output{
            committed_after.data(), fields.volume_scalar_count, components});
    const auto& telemetry = engine->contract()->telemetry();
    const bool rolled_back = aborted && !rollback.committed &&
        rollback.retry_available && rollback.coordinates_preserved &&
        rollback.committed_bytes_preserved &&
        rollback.before_interior_hash == rollback.after_interior_hash &&
        rollback.before_trace_hash == rollback.after_trace_hash &&
        dg::hdg::runtime::detail::SameCoordinates(before, after) &&
        dg::hdg::runtime::detail::SameCoordinates(
            rollback.before, rollback.after) &&
        exported_before && exported_after &&
        committed_before == fields.initial &&
        committed_after == committed_before &&
        engine->contract()->status() == RunStatus::retry_available &&
        telemetry.commits == 0 && telemetry.rollbacks == 1 &&
        !output.candidate_frozen && !output.checkpoint_staged &&
        !output.output_staged && !output.successful_commit;
    return All(
        prepared && captured && gate_state && staging_state && rolled_back,
        request.communicator);
  };

  CudaPreparedEulerConservationGateCapture relaxed{};
  CudaStepEvidence relaxed_rollback{};
  const bool relaxed_run = run(1.0e-4, true, relaxed, relaxed_rollback);
  const std::array<double, 5> expected_balance{{
      -2.5e-6, -1.25e-6, 1.875e-6, -3.75e-6, 0.0}};
  const std::array<double, 5> expected_scale{{
      2.5e-6, 1.25e-6, 1.875e-6, 3.75e-6, 0.0}};
  const auto close = [](const double actual, const double expected)
  {
    return std::isfinite(actual) &&
        std::abs(actual - expected) <=
            2.0e-12 * (1.0 + std::abs(expected));
  };
  bool analytic = relaxed_run && close(relaxed.production_local[10], 1.0) &&
      relaxed.production_local[11] == 1.0;
  double production_maximum_error = 0.0;
  double production_maximum_relative_error = 0.0;
  for (std::size_t component = 0; analytic && component < 5; ++component)
  {
    analytic = close(relaxed.production_local[component],
                     expected_balance[component]) &&
        close(relaxed.production_local[5 + component],
              expected_scale[component]);
    if (component < 4)
    {
      const double error = std::abs(relaxed.production_local[component]);
      production_maximum_error = std::max(
          production_maximum_error, error);
      production_maximum_relative_error = std::max(
          production_maximum_relative_error,
          dg::hdg::generic::EulerHdgConservationRelativeError(
              error, relaxed.production_local[5 + component],
              relaxed.production_local[10]));
    }
  }
  analytic = analytic && production_maximum_relative_error > 1.0e-8 &&
      production_maximum_relative_error < 1.0e-4 &&
      relaxed.production_local[4] == 0.0 &&
      relaxed.production_local[9] == 0.0 &&
      close(relaxed.lifecycle_maximum_error, production_maximum_error) &&
      close(relaxed.lifecycle_maximum_relative_error,
            production_maximum_relative_error);

  CudaPreparedEulerConservationGateCapture tight{};
  CudaStepEvidence tight_rollback{};
  const double tight_tolerance =
      0.25 * production_maximum_relative_error;
  const bool tight_run = analytic && tight_tolerance > 0.0 &&
      run(tight_tolerance, false, tight, tight_rollback);
  bool tight_balance_parity = tight_run;
  for (std::size_t index = 0;
       tight_balance_parity && index < tight.production_local.size(); ++index)
    tight_balance_parity = close(
        tight.production_local[index], relaxed.production_local[index]);
  const bool rejected_by_measured_gate = tight_balance_parity &&
      close(tight.lifecycle_maximum_error, production_maximum_error) &&
      close(tight.lifecycle_maximum_relative_error,
            production_maximum_relative_error) &&
      tight.lifecycle_maximum_relative_error > tight_tolerance;
  const bool passed = All(
      analytic && rejected_by_measured_gate, request.communicator);
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA conventional conservation gate/rollback failed"
                << " relaxed_run=" << relaxed_run
                << " analytic=" << analytic
                << " tight_run=" << tight_run
                << " measured=" << production_maximum_relative_error
                << " tight_tolerance=" << tight_tolerance
                << " tight_lifecycle="
                << tight.lifecycle_maximum_relative_error << '\n';
  }
  return passed;
}

template <typename Selection>
bool CheckTensorDiagonalConvergenceWarningRollbackForSelection(
    const EulerHdgRequest& request, const BorrowedFields& fields,
    const int order, const int ranks)
{
  CudaConfig config = MakeDirectConfig(request, ranks, order);
  auto engine = std::make_unique<
      PersistentCudaSmoothEulerEngine<Selection>>(config);
  CudaRunEvidence initialization{};
  const bool initialized = engine->Initialize(initialization);
  const bool initialization_hypre_clean = HYPRE_GetError() == 0;
  if (!initialization_hypre_clean) (void)HYPRE_ClearAllErrors();
  if (!All(initialized && initialization_hypre_clean,
           request.communicator))
    return false;

  std::vector<double> committed_before(fields.initial.size(), -1.0);
  const bool exported_before = engine->ExportCommittedInterior(
      CudaMutableHostL2Output{committed_before.data(),
                              fields.volume_scalar_count, components});
  const auto before = engine->contract()->coordinates();

  CudaPrecommitSnapshot first_snapshot{};
  CudaStepEvidence first{};
  constexpr double dt = 1.0e-5;
  const bool first_prepared = engine->PrepareStep(
      false, dt, first_snapshot, first);
  const bool first_hypre_clean = HYPRE_GetError() == 0;
  if (!first_hypre_clean) (void)HYPRE_ClearAllErrors();
  std::vector<double> committed_after_first(fields.initial.size(), -2.0);
  const bool exported_after_first = engine->ExportCommittedInterior(
      CudaMutableHostL2Output{committed_after_first.data(),
                              fields.volume_scalar_count, components});
  const auto after_first = engine->contract()->coordinates();
  const auto& first_telemetry = engine->contract()->telemetry();
  const bool first_rollback = !first_prepared && first_hypre_clean &&
      first.status == dg::hdg::runtime::cuda_w8a::CudaRunStatus::retry_available &&
      first.contract_reason == FailureReason::true_residual_gate &&
      first.phase == dg::hdg::runtime::cuda_w8a::CudaFailurePhase::trace_solve &&
      first.attempt.retry_index == 0 && first.linear_iterations == 1 &&
      // The bounded Krylov work is retained even though the rebuilt true-
      // residual gate prevents this solve from becoming a complete trace
      // solve in the lifecycle accounting.
      first.trace_solves == 0 &&
      std::isfinite(first.true_relative_residual) &&
      first.true_relative_residual >
          request.numerical_policy.true_residual_relative_tolerance &&
      first.retry_available && !first.committed &&
      first.coordinates_preserved && first.committed_bytes_preserved &&
      first.tensor_core.trace_preconditioner ==
          dg::hdg::runtime::cuda_w8a::CudaTracePreconditionerKind::
              hypre_diag_scale &&
      first.tensor_core.trace_preconditioner_configured &&
      engine->contract()->status() == RunStatus::retry_available &&
      first_telemetry.commits == 0 && first_telemetry.rollbacks == 1 &&
      exported_before && exported_after_first &&
      committed_before == fields.initial &&
      committed_after_first == committed_before &&
      dg::hdg::runtime::detail::SameCoordinates(before, after_first);
  if (!All(first_rollback, request.communicator))
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "tensor diagonal convergence-warning first rollback"
                << " prepared=" << first_prepared
                << " hypre_clean=" << first_hypre_clean
                << " status=" << static_cast<int>(first.status)
                << " reason=" << static_cast<int>(first.contract_reason)
                << " phase=" << static_cast<int>(first.phase)
                << " retry_index=" << first.attempt.retry_index
                << " linear=" << first.linear_iterations
                << " solves=" << first.trace_solves
                << " true_relative=" << first.true_relative_residual
                << " true_tolerance="
                << request.numerical_policy.true_residual_relative_tolerance
                << " retry_available=" << first.retry_available
                << " committed=" << first.committed
                << " coordinates=" << first.coordinates_preserved
                << " bytes=" << first.committed_bytes_preserved
                << " trace_kind="
                << static_cast<int>(first.tensor_core.trace_preconditioner)
                << " trace_configured="
                << first.tensor_core.trace_preconditioner_configured
                << " contract_status="
                << static_cast<int>(engine->contract()->status())
                << " commits=" << first_telemetry.commits
                << " rollbacks=" << first_telemetry.rollbacks
                << " export_before=" << exported_before
                << " export_after=" << exported_after_first
                << " initial_equal=" << (committed_before == fields.initial)
                << " rollback_equal="
                << (committed_after_first == committed_before)
                << " coordinate_equal="
                << dg::hdg::runtime::detail::SameCoordinates(
                       before, after_first) << '\n';
    return false;
  }

  CudaPrecommitSnapshot retry_snapshot{};
  CudaStepEvidence retry{};
  const bool retry_prepared = engine->PrepareStep(
      true, dt, retry_snapshot, retry);
  const bool retry_hypre_clean = HYPRE_GetError() == 0;
  if (!retry_hypre_clean) (void)HYPRE_ClearAllErrors();
  std::vector<double> committed_after_retry(fields.initial.size(), -3.0);
  const bool exported_after_retry = engine->ExportCommittedInterior(
      CudaMutableHostL2Output{committed_after_retry.data(),
                              fields.volume_scalar_count, components});
  const auto after_retry = engine->contract()->coordinates();
  const auto& retry_telemetry = engine->contract()->telemetry();
  const bool retry_rollback = !retry_prepared && retry_hypre_clean &&
      retry.status == dg::hdg::runtime::cuda_w8a::CudaRunStatus::terminal_failure &&
      retry.contract_reason == FailureReason::true_residual_gate &&
      retry.phase == dg::hdg::runtime::cuda_w8a::CudaFailurePhase::trace_solve &&
      retry.attempt.retry_index == 1 && retry.linear_iterations == 1 &&
      retry.trace_solves == 0 &&
      std::isfinite(retry.true_relative_residual) &&
      retry.true_relative_residual >
          request.numerical_policy.true_residual_relative_tolerance &&
      !retry.retry_available && !retry.committed &&
      retry.coordinates_preserved && retry.committed_bytes_preserved &&
      engine->contract()->status() == RunStatus::terminal_failure &&
      retry_telemetry.commits == 0 && retry_telemetry.rollbacks == 2 &&
      exported_after_retry && committed_after_retry == committed_before &&
      dg::hdg::runtime::detail::SameCoordinates(before, after_retry);
  const bool passed = All(retry_rollback, request.communicator);
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "tensor diagonal convergence-warning retry rollback"
                << " prepared=" << retry_prepared
                << " hypre_clean=" << retry_hypre_clean
                << " status=" << static_cast<int>(retry.status)
                << " reason=" << static_cast<int>(retry.contract_reason)
                << " phase=" << static_cast<int>(retry.phase)
                << " retry_index=" << retry.attempt.retry_index
                << " linear=" << retry.linear_iterations
                << " solves=" << retry.trace_solves
                << " true_relative=" << retry.true_relative_residual
                << " true_tolerance="
                << request.numerical_policy.true_residual_relative_tolerance
                << " retry_available=" << retry.retry_available
                << " committed=" << retry.committed
                << " coordinates=" << retry.coordinates_preserved
                << " bytes=" << retry.committed_bytes_preserved
                << " contract_status="
                << static_cast<int>(engine->contract()->status())
                << " commits=" << retry_telemetry.commits
                << " rollbacks=" << retry_telemetry.rollbacks
                << " export=" << exported_after_retry
                << " rollback_equal="
                << (committed_after_retry == committed_before)
                << " coordinate_equal="
                << dg::hdg::runtime::detail::SameCoordinates(
                       before, after_retry) << '\n';
  }
  engine.reset();
  const bool teardown_hypre_clean = HYPRE_GetError() == 0;
  if (!teardown_hypre_clean) (void)HYPRE_ClearAllErrors();
  return passed && All(teardown_hypre_clean, request.communicator);
}

bool CheckTensorDiagonalConvergenceWarningRollback(const int order,
                                                   const int ranks)
{
  // One p1 case per supported rank surface exercises the exact tensor-
  // conventional diagonal policy.  Other engines/orders keep their existing
  // lifecycle matrix without repeating this forced nonconvergence fixture.
  if (selected_cuda_engine != CudaEngineKind::tensor_core || order != 1)
    return true;
  const bool incoming_hypre_clean = HYPRE_ClearAllErrors() == 0 &&
      HYPRE_GetError() == 0;
  if (!All(incoming_hypre_clean)) return false;

  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  request.numerical_policy.maximum_linear_iterations = 1;
  request.numerical_policy.newton_correction_linear_relative_tolerance =
      1.0e-30;
  request.numerical_policy.final_coherent_linear_relative_tolerance =
      1.0e-30;
  request.numerical_policy.
      true_residual_tightening_linear_relative_tolerance = 1.0e-30;
  request.numerical_policy.true_residual_relative_tolerance = 1.0e-30;
  const bool admitted =
      dg::hdg::generic::ValidateEulerHdgRequest(request) ==
          ContractStatus::accepted;
  if (!All(admitted, request.communicator)) return false;
  return ranks == 1
      ? CheckTensorDiagonalConvergenceWarningRollbackForSelection<
            BorrowedCudaRank1OneGpuRuntimeSelection>(
            request, fields, order, ranks)
      : CheckTensorDiagonalConvergenceWarningRollbackForSelection<
            BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>(
            request, fields, order, ranks);
}

bool CheckEulerAlgebra(const int order, const int ranks)
{
  mfem::ParMesh periodic_mesh = MakePeriodicMesh();
  BorrowedFields periodic_fields = MakeBorrowedFields(
      periodic_mesh, order, SourceMode::nonzero,
      InitialMode::smooth_periodic);
  const auto check_periodic = [&](const RegularizationKind kind)
  {
    BoundaryContext boundary{};
    EulerHdgRequest request = MakeRequest(
        periodic_mesh, order, periodic_fields, boundary, false, kind);
    const bool admitted =
        dg::hdg::generic::ValidateEulerHdgRequest(request) ==
            ContractStatus::accepted;
    if (!All(admitted, request.communicator)) return false;
    return ranks == 1
        ? CheckEulerAlgebraForSelection<
              BorrowedCudaRank1OneGpuRuntimeSelection>(request, ranks, order)
        : CheckEulerAlgebraForSelection<
              BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>(
                  request, ranks, order);
  };

  const auto check_physical = [&](const RegularizationKind kind)
  {
    mfem::ParMesh mesh = MakePhysicalMesh();
    BorrowedFields fields = MakeBorrowedFields(
        mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
    BoundaryContext boundary{};
    EulerHdgRequest request = MakeRequest(
        mesh, order, fields, boundary, true, kind);
    // A fixed supersonic-inlet exterior is the exact CUDA analogue of the CPU
    // Picard-row audit fixture.  It makes the physical Euler row's frozen
    // exterior differentiable while retaining the production IGR Neumann
    // auxiliary row on the same faces.
    const std::array<double, components> reference{{
        1.0, 0.05, -0.02, 2.60}};
    for (EulerBoundaryPolicy& policy : boundary.policies)
      policy = {EulerBoundaryPolicyKind::supersonic_inlet,
                reference, 0.0};

    GenericEulerHdgCudaSession session;
    const auto initialization = session.Initialize(request);
    std::array<double, 6> relative_errors{};
    constexpr double jacobian_tolerance = 5.0e-5;
    const bool initialized = All(
        initialization.status == HdgEulerSessionStatus::initialized,
        request.communicator);
    CudaBorrowedAssemblyFusionAudit assembly_fusion_audit{};
    const bool assembly_fusion = initialized &&
        (selected_cuda_engine != CudaEngineKind::tensor_core ||
         CudaQualificationTestAccess::AuditBorrowedAssemblyFusion(
             session, assembly_fusion_audit));
    const bool verified = assembly_fusion &&
        CudaQualificationTestAccess::VerifyAssemblerJacobian(
            session, jacobian_tolerance, relative_errors);
    const bool passed = All(verified, request.communicator);
    if (!passed)
    {
      int rank = 0;
      MPI_Comm_rank(request.communicator, &rank);
      if (rank == 0)
      {
        std::cerr << "CUDA physical Euler Jacobian qualification failed order="
                  << order << " regularization="
                  << (kind == RegularizationKind::igr ? "igr" : "none")
                  << " initialized=" << initialized
                  << " assembly_fusion=" << assembly_fusion
                  << " assembly_maxima=";
        for (const double error :
             assembly_fusion_audit.maximum_absolute_errors)
          std::cerr << ' ' << error;
        std::cerr
                  << " relative_errors=";
        for (const double error : relative_errors) std::cerr << ' ' << error;
        std::cerr << '\n';
      }
    }
    return passed;
  };

  const bool conventional_periodic =
      check_periodic(RegularizationKind::none);
  const bool conventional_physical =
      check_physical(RegularizationKind::none);
  const bool igr_periodic = check_periodic(RegularizationKind::igr);
  const bool igr_physical = check_physical(RegularizationKind::igr);
  return conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical;
}

bool CheckConventionalP1Q12TensorVolumeMultiChunkParity(const int order)
{
  if (selected_cuda_engine != CudaEngineKind::tensor_core || order != 1)
    return true;

  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  request.numerical_policy.conventional_assembly_quadrature_order = 12;
  const bool admitted =
      dg::hdg::generic::ValidateEulerHdgRequest(request) ==
          ContractStatus::accepted;
  if (!All(admitted, request.communicator)) return false;

  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized = All(
      initialization.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  CudaBorrowedAssemblyFusionAudit audit{};
  const bool parity = initialized &&
      CudaQualificationTestAccess::
          AuditConventionalP1Q12TensorVolumeMultiChunk(session, audit);
  const bool passed = All(parity, request.communicator);
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA conventional p1 q12 tensor-volume multi-chunk "
                   "parity failed initialized="
                << initialized << " dimensions=" << audit.elements << 'x'
                << audit.interior_dofs << 'x' << audit.trace_dofs
                << " maxima=";
      for (const double error : audit.maximum_absolute_errors)
        std::cerr << ' ' << error;
      std::cerr << " scaled=";
      for (const double error : audit.maximum_scaled_errors)
        std::cerr << ' ' << error;
      std::cerr << " status=" << audit.status_bitwise_equal << '/'
                << audit.status_all_passed << '\n';
    }
  }
  return passed;
}

bool CheckZeroUpdateFinalReconstruction(const int order)
{
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty, InitialMode::constant);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  GenericEulerHdgCudaSession session;
  const auto initialization = session.Initialize(request);
  const bool initialized = All(
      initialization.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  if (!initialized) return false;

  GenericEulerCudaPreparedStep authority;
  GenericEulerHdgCudaSession::prepared_output_type output{};
  constexpr double dt = 1.0e-5;
  const auto prepared_result = session.Prepare(dt, authority, output);
  const bool prepared = All(
      prepared_result.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!prepared) return false;
  const bool reconstruction =
      CudaQualificationTestAccess::VerifyZeroUpdateFinalReconstruction(
          session);
  std::vector<double> candidate(fields.initial.size(),
                                std::numeric_limits<double>::quiet_NaN());
  const bool exported = session.ExportPreparedInterior(
      authority, output,
      CudaMutableHostL2Output{
          candidate.data(), fields.volume_scalar_count, components});
  const bool unchanged = candidate == fields.initial;
  const auto aborted = session.Abort(std::move(authority), output);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted,
      request.communicator);
  const bool passed = All(
      reconstruction && exported && unchanged && aborted_everywhere,
      request.communicator);
  if (!passed)
  {
    int rank = 0;
    MPI_Comm_rank(request.communicator, &rank);
    if (rank == 0)
      std::cerr << "CUDA zero-update final reconstruction failed order="
                << order << " reconstruction=" << reconstruction
                << " exported=" << exported << " unchanged=" << unchanged
                << " aborted=" << aborted_everywhere << '\n';
  }
  return passed;
}

bool SameValues(const std::vector<double>& left,
                const std::vector<double>& right)
{
  return left == right;
}

bool NearValues(const std::vector<double>& left,
                const std::vector<double>& right)
{
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
  {
    const double scale = std::max({1.0, std::abs(left[i]),
                                   std::abs(right[i])});
    if (!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
        std::abs(left[i] - right[i]) > 1.0e-12 * scale)
    {
      return false;
    }
  }
  return true;
}

bool DifferentValues(const std::vector<double>& left,
                     const std::vector<double>& right)
{
  return left.size() == right.size() &&
      !std::equal(left.begin(), left.end(), right.begin());
}

bool SameTransferAccounting(const CudaTransferTotals& before,
                            const CudaTransferTotals& after)
{
  return before.host_to_device_bytes == after.host_to_device_bytes &&
      before.device_to_host_bytes == after.device_to_host_bytes &&
      before.device_to_device_bytes == after.device_to_device_bytes &&
      before.mpi_staging_bytes == after.mpi_staging_bytes &&
      before.synchronization_events == after.synchronization_events &&
      before.event_count == after.event_count &&
      before.complete == after.complete;
}

bool ExactIgrFieldExportTransfer(const CudaTransferTotals& before,
                                 const CudaTransferTotals& after,
                                 const std::size_t scalar_count)
{
  constexpr std::size_t published_components = components + 1;
  const std::size_t expected_bytes =
      published_components * scalar_count * sizeof(double);
  return before.complete && after.complete &&
      after.host_to_device_bytes == before.host_to_device_bytes &&
      after.device_to_host_bytes ==
          before.device_to_host_bytes + expected_bytes &&
      after.device_to_device_bytes == before.device_to_device_bytes &&
      after.mpi_staging_bytes == before.mpi_staging_bytes &&
      after.synchronization_events == before.synchronization_events + 1 &&
      after.event_count == before.event_count + 1;
}

bool GlobalSourceOnlyResponseNorm(
    mfem::ParMesh& mesh, const int order,
    const std::vector<double>& candidate,
    const std::vector<double>& initial,
    const std::vector<double>& source,
    const double dt,
    long double& norm)
{
  long double local_squared_norm = 0.0L;
  const bool local_ready = [&]() noexcept
  {
    try
    {
      mfem::L2_FECollection collection(
          order, 2, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      mfem::ParFiniteElementSpace space(
          &mesh, &collection, components, mfem::Ordering::byNODES);
      const std::size_t vector_size =
          static_cast<std::size_t>(space.GetVSize());
      if (space.GetVSize() < 0 || candidate.size() != vector_size ||
          initial.size() != vector_size || source.size() != vector_size ||
          !std::isfinite(dt) || !(dt > 0.0))
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
            vdofs.Size() != finite_element->GetDof() * components)
          return false;
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), 2 * order + 2);
        mfem::Vector shape(finite_element->GetDof());
        for (int quadrature = 0;
             quadrature < rule.GetNPoints(); ++quadrature)
        {
          const mfem::IntegrationPoint& point = rule.IntPoint(quadrature);
          finite_element->CalcShape(point, shape);
          transformation->SetIntPoint(&point);
          const long double weight = static_cast<long double>(
              point.weight * transformation->Weight());
          if (!(weight > 0.0L) || !std::isfinite(weight)) return false;
          for (int component = 0; component < components; ++component)
          {
            long double response = 0.0L;
            for (int scalar = 0;
                 scalar < finite_element->GetDof(); ++scalar)
            {
              double sign = 1.0;
              const int dof = DecodeSignedDof(
                  vdofs[scalar + finite_element->GetDof() * component],
                  sign);
              if (dof < 0 || dof >= space.GetVSize()) return false;
              const std::size_t index = static_cast<std::size_t>(dof);
              const double coefficient = candidate[index] - initial[index] -
                  dt * source[index];
              if (!std::isfinite(coefficient)) return false;
              response += static_cast<long double>(
                  sign * coefficient * shape[scalar]);
            }
            local_squared_norm += weight * response * response;
          }
        }
      }
      return std::isfinite(local_squared_norm) &&
          local_squared_norm >= 0.0L;
    }
    catch (...) { return false; }
  }();
  if (!All(local_ready, mesh.GetComm())) return false;
  long double global_squared_norm = 0.0L;
  const bool reduced = MPI_Allreduce(
      &local_squared_norm, &global_squared_norm, 1, MPI_LONG_DOUBLE,
      MPI_SUM, mesh.GetComm()) == MPI_SUCCESS;
  const bool global_ready = reduced && std::isfinite(global_squared_norm) &&
      global_squared_norm >= 0.0L;
  if (!All(global_ready, mesh.GetComm())) return false;
  norm = std::sqrt(global_squared_norm);
  return All(std::isfinite(norm), mesh.GetComm());
}

bool BoundaryResponseSeparated(const long double periodic_response,
                               const long double physical_response)
{
  constexpr long double periodic_roundoff_ceiling = 1.0e-9L;
  constexpr long double physical_activity_floor = 1.0e-10L;
  constexpr long double separation_factor = 8.0L;
  return std::isfinite(periodic_response) &&
      std::isfinite(physical_response) && periodic_response >= 0.0L &&
      physical_response > physical_activity_floor &&
      periodic_response <= periodic_roundoff_ceiling &&
      physical_response > separation_factor *
          std::max(periodic_response, periodic_roundoff_ceiling);
}

bool ExpectedPeriodicSourceResponse(
    const std::vector<double>& candidate,
    const std::vector<double>& initial,
    const std::vector<double>& source,
    const std::size_t scalar_count,
    const SourceMode source_mode,
    const InitialMode initial_mode,
    const double dt)
{
  if (initial_mode != InitialMode::constant) return true;
  if (source_mode == SourceMode::empty ||
      source_mode == SourceMode::explicit_zero)
    return NearValues(candidate, initial);
  if (candidate.size() != initial.size() ||
      source.size() != initial.size() ||
      initial.size() != scalar_count * static_cast<std::size_t>(components))
    return false;
  for (int component = 0; component < components; ++component)
    for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
    {
      const std::size_t index = scalar_count *
          static_cast<std::size_t>(component) + scalar;
      const double expected_delta = dt * source[index];
      const double actual_delta = candidate[index] - initial[index];
      const double state_scale = std::max(
          {1.0, std::abs(initial[index]), std::abs(candidate[index])});
      const double roundoff_floor = 4096.0 *
          std::numeric_limits<double>::epsilon() * state_scale;
      const double tolerance = roundoff_floor +
          5.0e-5 * std::abs(expected_delta);
      if (!std::isfinite(actual_delta) || expected_delta == 0.0 ||
          std::signbit(actual_delta) != std::signbit(expected_delta) ||
          std::abs(actual_delta - expected_delta) > tolerance)
        return false;
    }
  return true;
}

bool BoundaryCoverage(const BoundaryContext& context,
                      const bool physical,
                      const MPI_Comm communicator)
{
  // CUDA consumes the copied POD table directly; it is not required to call
  // the host callback for physical faces.  Periodic topology, however, must
  // never be reclassified as a physical callback closure.
  if (physical) return true;
  std::array<int, 4> global{};
  if (MPI_Allreduce(context.calls.data(), global.data(),
                    static_cast<int>(global.size()), MPI_INT, MPI_SUM,
                    communicator) != MPI_SUCCESS)
  {
    return false;
  }
  return std::all_of(global.begin(), global.end(),
                     [](const int calls) { return calls == 0; });
}

bool RejectedBeforeMfemDevice(const EulerHdgRequest& request)
{
  const bool unconfigured_before = !mfem::Device::IsConfigured();
  GenericEulerHdgCudaSession session;
  const auto result = session.Initialize(request);
  const auto coordinates = session.coordinates();
  return unconfigured_before &&
      result.status == HdgEulerSessionStatus::rejected &&
      coordinates.step == 0 && coordinates.generation == 0 &&
      coordinates.time == 0.0 && !mfem::Device::IsConfigured();
}

bool CheckContractAdmission(const int order)
{
  mfem::ParMesh mesh = MakePhysicalMesh();
  // Keep this pre-device contract fixture host-only.  ProjectCoefficient on a
  // CUDA-enabled MFEM build requires an admitted Device, while the malformed
  // session cases below deliberately prove rejection before Device setup.
  BorrowedFields fields{};
  fields.volume_scalar_count = 1;
  fields.initial = {1.0, 0.05, -0.02, 2.60};
  fields.initial_true_trace = fields.initial;
  fields.source = {0.25, 0.125, -0.1875, 0.375};
  BoundaryContext boundary{};
  EulerHdgRequest conventional = MakeRequest(
      mesh, order, fields, boundary, true, RegularizationKind::none);
  EulerHdgRequest igr = MakeRequest(
      mesh, order, fields, boundary, true, RegularizationKind::igr);
  EulerHdgRequest invalid_igr = igr;
  invalid_igr.numerical_policy.stabilization.alpha = 0.0;
  EulerHdgRequest null_table = conventional;
  null_table.boundary.policies.policies = nullptr;
  EulerHdgRequest short_table = conventional;
  short_table.boundary.policies.count = boundary.policies.size() - 1;
  EulerHdgRequest wrong_dimension = conventional;
  wrong_dimension.boundary.policies.space_dimension = 3;
  EulerHdgRequest nonfinite_gamma = conventional;
  nonfinite_gamma.boundary.policies.gamma =
      std::numeric_limits<double>::quiet_NaN();

  std::array<EulerBoundaryPolicy, 4> invalid_kind_policies = boundary.policies;
  invalid_kind_policies[0].kind = static_cast<EulerBoundaryPolicyKind>(255);
  EulerHdgRequest invalid_kind = conventional;
  invalid_kind.boundary.policies.policies = invalid_kind_policies.data();

  int ranks = 1;
  int rank = 0;
  MPI_Comm_size(mesh.GetComm(), &ranks);
  MPI_Comm_rank(mesh.GetComm(), &rank);
  std::array<EulerBoundaryPolicy, 4> mismatched_policies = boundary.policies;
  if (ranks == 2 && rank == 0)
  {
    mismatched_policies[0].kind =
        EulerBoundaryPolicyKind::supersonic_outlet;
  }
  EulerHdgRequest mismatched = conventional;
  mismatched.boundary.policies.policies = mismatched_policies.data();

  BorrowedFields crossed_fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty);
  BoundaryContext crossed_boundary{};
  EulerHdgRequest crossed_backend = MakeRequest(
      mesh, order, crossed_fields, crossed_boundary, true,
      RegularizationKind::none);
  if (ranks == 2 && rank == 0)
  {
    crossed_backend.backend = {
        BackendKind::cpu, -1, CudaEngineKind::ordinary};
    crossed_backend.capabilities = cpu_capabilities;
  }

  const bool valid_contracts =
      dg::hdg::generic::ValidateEulerHdgRequest(conventional) ==
          ContractStatus::accepted &&
      dg::hdg::generic::ValidateEulerHdgRequest(igr) ==
          ContractStatus::accepted;
  const bool invalid_igr_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(invalid_igr) !=
          ContractStatus::accepted;
  const bool null_table_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(null_table) !=
          ContractStatus::accepted;
  const bool short_table_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(short_table) !=
          ContractStatus::accepted;
  const bool wrong_dimension_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(wrong_dimension) !=
          ContractStatus::accepted;
  const bool nonfinite_gamma_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(nonfinite_gamma) !=
          ContractStatus::accepted;
  const bool invalid_kind_contract =
      dg::hdg::generic::ValidateEulerHdgRequest(invalid_kind) !=
          ContractStatus::accepted;
  const bool mismatched_contract = ranks != 2 ||
      dg::hdg::generic::ValidateEulerHdgRequest(mismatched) !=
          ContractStatus::accepted;
  const bool crossed_backend_contract = ranks != 2 ||
      dg::hdg::generic::ValidateEulerHdgRequest(crossed_backend) !=
          ContractStatus::accepted;

  // The public session repeats the generic contract before any device or
  // engine allocation.  Each malformed request owns a fresh one-shot facade.
  const bool invalid_igr_session = RejectedBeforeMfemDevice(invalid_igr);
  const bool null_table_session = RejectedBeforeMfemDevice(null_table);
  const bool short_table_session = RejectedBeforeMfemDevice(short_table);
  const bool wrong_dimension_session =
      RejectedBeforeMfemDevice(wrong_dimension);
  const bool nonfinite_gamma_session =
      RejectedBeforeMfemDevice(nonfinite_gamma);
  const bool invalid_kind_session = RejectedBeforeMfemDevice(invalid_kind);
  const bool mismatched_session = ranks != 2 ||
      RejectedBeforeMfemDevice(mismatched);
  const bool crossed_backend_session = ranks != 2 ||
      RejectedBeforeMfemDevice(crossed_backend);
  return All(valid_contracts && invalid_igr_contract &&
                 null_table_contract && short_table_contract &&
                 wrong_dimension_contract && nonfinite_gamma_contract &&
                 invalid_kind_contract && mismatched_contract &&
                 crossed_backend_contract && invalid_igr_session &&
                 null_table_session && short_table_session &&
                 wrong_dimension_session && nonfinite_gamma_session &&
                 invalid_kind_session && mismatched_session &&
                 crossed_backend_session,
             mesh.GetComm());
}

bool CheckLifecycle(const int order, const bool physical,
                    const RegularizationKind kind,
                    const SourceMode source_mode,
                    const InitialMode initial_mode,
                    std::vector<double>& committed_result,
                    long double* source_only_response = nullptr,
                    const int physical_elements_per_direction = 3)
{
  mfem::ParMesh mesh = physical
      ? MakePhysicalMesh(physical_elements_per_direction) : MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, source_mode, initial_mode);
  const std::vector<double> expected_initial = fields.initial;
  const std::vector<double> expected_source = fields.source;
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, physical, kind);
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
  {
    return false;
  }

  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(request);
  const bool igr = kind == RegularizationKind::igr;
  const double unwritten = std::numeric_limits<double>::quiet_NaN();
  const auto finite = [](const std::vector<double>& values)
  {
    return std::all_of(values.begin(), values.end(),
        [](const double value) { return std::isfinite(value); });
  };
  const auto export_committed =
      [&](std::vector<double>& conservative, std::vector<double>& sigma,
          bool& exact_transfer)
  {
    if (!igr)
    {
      exact_transfer = true;
      return session.ExportCommittedInterior(
          CudaMutableHostL2Output{conservative.data(),
                                  fields.volume_scalar_count, components});
    }
    const CudaTransferTotals before = CudaTransportPolicy::Accounting();
    const bool exported = session.ExportCommittedFields(
        HdgEulerHostInteriorFields{conservative.data(), sigma.data(),
                                   fields.volume_scalar_count});
    const CudaTransferTotals after = CudaTransportPolicy::Accounting();
    exact_transfer = ExactIgrFieldExportTransfer(
        before, after, fields.volume_scalar_count);
    return exported;
  };
  const auto export_prepared =
      [&](const GenericEulerCudaPreparedStep& authority,
          const GenericEulerHdgCudaSession::prepared_output_type& output,
          std::vector<double>& conservative, std::vector<double>& sigma,
          bool& exact_transfer)
  {
    if (!igr)
    {
      exact_transfer = true;
      return session.ExportPreparedInterior(
          authority, output,
          CudaMutableHostL2Output{conservative.data(),
                                  fields.volume_scalar_count, components});
    }
    const CudaTransferTotals before = CudaTransportPolicy::Accounting();
    const bool exported = session.ExportPreparedFields(
        authority, output,
        HdgEulerHostInteriorFields{conservative.data(), sigma.data(),
                                   fields.volume_scalar_count});
    const CudaTransferTotals after = CudaTransportPolicy::Accounting();
    exact_transfer = ExactIgrFieldExportTransfer(
        before, after, fields.volume_scalar_count);
    return exported;
  };
  std::vector<double> committed_initial(expected_initial.size(), -1.0);
  std::vector<double> sigma_initial(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool initialized_everywhere = All(
      initialized.status == HdgEulerSessionStatus::initialized,
      request.communicator);
  if (!initialized_everywhere) return false;
  if (!CudaQualificationTestAccess::VerifyInitialStateHashesAndReferenceShapes(
          session))
  {
    std::cerr << "CUDA initial state hash/reference shape mismatch order="
              << order << '\n';
    return false;
  }
  bool initial_transfer = false;
  const bool initial_export = export_committed(
      committed_initial, sigma_initial, initial_transfer);
  bool rejected_bad_extent = true;
  if (igr)
  {
    std::vector<double> bad_conservative(expected_initial.size(), 17.0);
    std::vector<double> bad_sigma(fields.volume_scalar_count, 19.0);
    const CudaTransferTotals before = CudaTransportPolicy::Accounting();
    const bool bad_export = fields.volume_scalar_count == 0 ||
        session.ExportCommittedFields(HdgEulerHostInteriorFields{
            bad_conservative.data(), bad_sigma.data(),
            fields.volume_scalar_count - 1});
    const CudaTransferTotals after = CudaTransportPolicy::Accounting();
    rejected_bad_extent = fields.volume_scalar_count > 0 && !bad_export &&
        std::all_of(bad_conservative.begin(), bad_conservative.end(),
                    [](const double value) { return value == 17.0; }) &&
        std::all_of(bad_sigma.begin(), bad_sigma.end(),
                    [](const double value) { return value == 19.0; }) &&
        SameTransferAccounting(before, after);
  }
  if (!All(initial_export && initial_transfer && rejected_bad_extent &&
               SameValues(committed_initial, expected_initial) &&
               (!igr || finite(sigma_initial)),
           request.communicator))
  {
    return false;
  }

  // Every borrowed numerical view must have been copied by Initialize.
  const double poisoned = std::numeric_limits<double>::quiet_NaN();
  std::fill(fields.initial.begin(), fields.initial.end(), poisoned);
  std::fill(fields.initial_true_trace.begin(),
            fields.initial_true_trace.end(), poisoned);
  std::fill(fields.source.begin(), fields.source.end(), poisoned);

  constexpr double dt = 1.0e-5;
  GenericEulerCudaPreparedStep first_authority;
  GenericEulerHdgCudaSession::prepared_output_type first_output{};
  const auto before = session.coordinates();
  const auto first_prepared = session.Prepare(
      dt, first_authority, first_output);
  std::vector<double> first_candidate(expected_initial.size(), -2.0);
  std::vector<double> sigma_first_candidate(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool first_prepared_everywhere = All(
      first_prepared.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!first_prepared_everywhere) return false;
  const bool first_engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(
          session, request.backend.cuda_engine);
  if (!All(first_engine_evidence, request.communicator)) return false;
  bool first_transfer = false;
  const bool first_export = export_prepared(
      first_authority, first_output, first_candidate,
      sigma_first_candidate, first_transfer);
  const bool partitioned_igr = igr && request.time_treatment ==
      EulerHdgTimeTreatment::partitioned_backward_euler;
  CudaPreparedPartitionedClosureAudit partitioned_closure_audit{};
  const bool partitioned_closure = !partitioned_igr ||
      CudaQualificationTestAccess::AuditPreparedPartitionedClosure(
          session, partitioned_closure_audit);
  const bool expected_update = physical
      ? (source_mode != SourceMode::nonzero ||
         DifferentValues(first_candidate, committed_initial))
      : ExpectedPeriodicSourceResponse(
            first_candidate, committed_initial, expected_source,
            fields.volume_scalar_count, source_mode, initial_mode, dt);
  if (!All(first_export && first_transfer && expected_update &&
               partitioned_closure &&
               (!igr || finite(sigma_first_candidate)),
           request.communicator))
  {
    if (partitioned_igr)
      std::cerr << "CUDA partitioned closure audit failed order=" << order
                << " physical=" << physical
                << " residual="
                << partitioned_closure_audit.global_scalar_residual
                << " auxiliary_norm="
                << partitioned_closure_audit.global_auxiliary_norm
                << " relative_residual="
                << partitioned_closure_audit.global_scalar_relative_residual
                << " tolerance="
                << partitioned_closure_audit.relative_residual_tolerance
                << " auxiliary_finite="
                << partitioned_closure_audit.auxiliary_interior_finite << '/'
                << partitioned_closure_audit.auxiliary_true_trace_finite << '/'
                << partitioned_closure_audit.auxiliary_local_trace_finite
                << '\n';
    return false;
  }
  const auto aborted = session.Abort(
      std::move(first_authority), first_output);
  std::vector<double> after_abort(expected_initial.size(), -3.0);
  std::vector<double> sigma_after_abort(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted,
      request.communicator);
  if (!aborted_everywhere) return false;
  bool abort_transfer = false;
  const bool abort_export = export_committed(
      after_abort, sigma_after_abort, abort_transfer);
  if (!All(abort_export && abort_transfer &&
               SameValues(after_abort, committed_initial) &&
               (!igr || SameValues(sigma_after_abort, sigma_initial)) &&
               session.coordinates().step == before.step &&
               session.coordinates().generation == before.generation &&
               session.coordinates().time == before.time,
           request.communicator))
  {
    return false;
  }

  GenericEulerCudaPreparedStep retry_authority;
  GenericEulerHdgCudaSession::prepared_output_type retry_output{};
  const auto retry_prepared = session.Prepare(
      dt, retry_authority, retry_output);
  std::vector<double> retry_candidate(expected_initial.size(), -4.0);
  std::vector<double> sigma_retry_candidate(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool retry_prepared_everywhere = All(
      retry_prepared.status == HdgEulerSessionStatus::prepared,
      request.communicator);
  if (!retry_prepared_everywhere) return false;
  const bool retry_engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(
          session, request.backend.cuda_engine);
  if (!All(retry_engine_evidence, request.communicator)) return false;
  bool retry_transfer = false;
  const bool retry_export = export_prepared(
      retry_authority, retry_output, retry_candidate,
      sigma_retry_candidate, retry_transfer);
  CudaPreparedPartitionedClosureAudit retry_closure_audit{};
  const bool retry_partitioned_closure = !partitioned_igr ||
      CudaQualificationTestAccess::AuditPreparedPartitionedClosure(
          session, retry_closure_audit);
  if (!All(retry_export && retry_transfer &&
               retry_partitioned_closure &&
               NearValues(first_candidate, retry_candidate) &&
               (!igr || NearValues(sigma_first_candidate,
                                   sigma_retry_candidate)),
           request.communicator))
  {
    if (partitioned_igr)
      std::cerr << "CUDA partitioned retry closure audit failed order="
                << order << " physical=" << physical
                << " residual=" << retry_closure_audit.global_scalar_residual
                << " auxiliary_norm="
                << retry_closure_audit.global_auxiliary_norm
                << " relative_residual="
                << retry_closure_audit.global_scalar_relative_residual
                << " tolerance="
                << retry_closure_audit.relative_residual_tolerance
                << " auxiliary_finite="
                << retry_closure_audit.auxiliary_interior_finite << '/'
                << retry_closure_audit.auxiliary_true_trace_finite << '/'
                << retry_closure_audit.auxiliary_local_trace_finite << '\n';
    return false;
  }
  const bool acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_output);
  if (!All(acknowledged, request.communicator)) return false;
  const auto committed = session.Commit(
      std::move(retry_authority), retry_output);
  std::vector<double> committed_final(expected_initial.size(), -5.0);
  std::vector<double> sigma_committed_final(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool committed_everywhere = All(
      committed.status == HdgEulerSessionStatus::committed,
      request.communicator);
  if (!committed_everywhere) return false;
  bool final_transfer = false;
  const bool final_export = export_committed(
      committed_final, sigma_committed_final, final_transfer);
  const auto after = session.coordinates();
  const bool boundary_coverage = BoundaryCoverage(
      boundary, physical, request.communicator);
  const bool passed = All(
      final_export && final_transfer &&
          SameValues(committed_final, retry_candidate) &&
          (!igr || SameValues(sigma_committed_final,
                              sigma_retry_candidate)) &&
          after.step == before.step + 1 &&
          after.generation == before.generation + 1 &&
          after.time == before.time + dt &&
          retry_output.successful_commit && boundary_coverage,
      request.communicator);
  const bool response_ready = source_only_response == nullptr ||
      (passed && GlobalSourceOnlyResponseNorm(
          mesh, order, committed_final, expected_initial, expected_source,
          dt, *source_only_response));
  const bool qualified = All(passed && response_ready, request.communicator);
  if (qualified) committed_result = std::move(committed_final);
  return qualified;
}

bool CheckRejectedArmijoRollbackForSelection(
    const int order, const int ranks, const RegularizationKind kind)
{
  // The direct-to-workspace trial kernel is shared by every generic order and
  // rank count.  Exercise its rejected-trial restoration once for each
  // mathematical mode, without duplicating the existing p1-p3/rank1-rank2
  // lifecycle matrix.
  if (order != 1 || ranks != 1) return true;

  // The IGR control at dt=0.2 completes every numerical gate but misses the
  // unchanged 1e-10 conservation gate by about 3.5%.  Keep the same large-step
  // source stimulus and strict Armijo comparison, with only enough IGR step
  // reduction to separate the intended line-search rejection from that gate.
  const double dt = kind == RegularizationKind::igr ? 0.18 : 0.2;
  constexpr double control_armijo_constant = 1.0e-4;
  constexpr double strict_armijo_constant = 0.99;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
  const std::vector<double> expected_initial = fields.initial;
  const bool igr = kind == RegularizationKind::igr;
  const char* const mode_name = igr ? "igr" : "conventional";
  const double unwritten = std::numeric_limits<double>::quiet_NaN();
  const auto finite = [](const std::vector<double>& values)
  {
    return std::all_of(values.begin(), values.end(),
        [](const double value) { return std::isfinite(value); });
  };
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, kind);
  request.numerical_policy.armijo_constant = control_armijo_constant;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  std::array<int, 6> control_work{};
  {
    GenericEulerHdgCudaSession control;
    const auto initialized = control.Initialize(request);
    if (!All(initialized.status == HdgEulerSessionStatus::initialized,
             request.communicator))
      return false;

    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    const auto prepared = control.Prepare(dt, authority, output);
    if (!All(prepared.status == HdgEulerSessionStatus::prepared,
             request.communicator))
    {
      std::cerr << "CUDA rejected-Armijo control mode=" << mode_name
                << " prepare_status="
                << static_cast<int>(prepared.status) << '\n';
      return false;
    }
    const bool accounted =
        CudaQualificationTestAccess::VerifyPreparedWorkAccounting(
            control, true, control_work);
    const auto aborted = control.Abort(std::move(authority), output);
    if (!All(accounted && control_work[0] > 0 && control_work[2] > 0 &&
                 control_work[3] == control_work[0] &&
                 control_work[4] == 0 &&
                 aborted.status == HdgEulerSessionStatus::aborted,
             request.communicator))
    {
      std::cerr << "CUDA rejected-Armijo control mode=" << mode_name
                << " work="
                << control_work[0] << '/' << control_work[1] << '/'
                << control_work[2] << '/' << control_work[3] << '/'
                << control_work[4] << '/' << control_work[5]
                << " accounted=" << accounted << " abort="
                << static_cast<int>(aborted.status) << '\n';
      return false;
    }
  }

  // The control and strict runs are otherwise identical.  Therefore their
  // first differing line-search decision is an Armijo decision: candidate
  // admissibility does not depend on this coefficient.
  request.numerical_policy.armijo_constant = strict_armijo_constant;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(request);
  if (!All(initialized.status == HdgEulerSessionStatus::initialized,
           request.communicator))
    return false;

  const auto export_committed =
      [&](std::vector<double>& conservative, std::vector<double>& sigma)
  {
    return igr
        ? session.ExportCommittedFields(HdgEulerHostInteriorFields{
              conservative.data(), sigma.data(),
              fields.volume_scalar_count})
        : session.ExportCommittedInterior(CudaMutableHostL2Output{
              conservative.data(), fields.volume_scalar_count, components});
  };
  const auto export_prepared =
      [&](const GenericEulerCudaPreparedStep& authority,
          const GenericEulerHdgCudaSession::prepared_output_type& output,
          std::vector<double>& conservative, std::vector<double>& sigma)
  {
    return igr
        ? session.ExportPreparedFields(
              authority, output, HdgEulerHostInteriorFields{
                  conservative.data(), sigma.data(),
                  fields.volume_scalar_count})
        : session.ExportPreparedInterior(
              authority, output, CudaMutableHostL2Output{
                  conservative.data(), fields.volume_scalar_count,
                  components});
  };

  std::vector<double> committed_initial(expected_initial.size(), -1.0);
  std::vector<double> sigma_initial(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool initial_export = export_committed(
      committed_initial, sigma_initial);
  if (!All(initial_export && SameValues(committed_initial, expected_initial) &&
               (!igr || finite(sigma_initial)),
           request.communicator))
    return false;

  const auto before = session.coordinates();
  GenericEulerCudaPreparedStep first_authority;
  GenericEulerHdgCudaSession::prepared_output_type first_output{};
  const auto first_prepared = session.Prepare(
      dt, first_authority, first_output);
  if (!All(first_prepared.status == HdgEulerSessionStatus::prepared,
           request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo strict mode=" << mode_name
              << " prepare_status="
              << static_cast<int>(first_prepared.status) << '\n';
    return false;
  }

  std::array<int, 6> first_work{};
  const bool first_work_accounted =
      CudaQualificationTestAccess::VerifyPreparedWorkAccounting(
          session, true, first_work);
  std::vector<double> first_candidate(expected_initial.size(), -2.0);
  std::vector<double> sigma_first_candidate(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool first_export = export_prepared(
      first_authority, first_output, first_candidate,
      sigma_first_candidate);
  const auto first_descriptor = first_output.descriptor;
  const auto first_local_interior_hash = first_output.local_interior_hash;
  const auto first_local_trace_hash = first_output.local_trace_hash;
  const dg::hdg::runtime::StateHash empty_hash{};
  const bool rejected_armijo = first_work_accounted &&
      first_work[0] > 0 && first_work[2] > 0 && first_work[4] > 0 &&
      first_work[3] == first_work[0] + first_work[4] &&
      first_work[1] == first_work[0] + 1;
  const bool first_identity = first_export &&
      (!igr || finite(sigma_first_candidate)) &&
      first_output.candidate_frozen && first_output.checkpoint_staged &&
      first_output.staging_range.complete &&
      !first_output.output_staged && !first_output.successful_commit &&
      first_descriptor.successful_commit &&
      first_descriptor.interior_hash != empty_hash &&
      first_descriptor.trace_hash != empty_hash &&
      first_local_interior_hash != empty_hash &&
      first_local_trace_hash != empty_hash;
  const auto aborted = session.Abort(
      std::move(first_authority), first_output);
  if (!All(rejected_armijo && first_identity &&
               aborted.status == HdgEulerSessionStatus::aborted,
           request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo strict mode=" << mode_name
              << " work="
              << first_work[0] << '/' << first_work[1] << '/'
              << first_work[2] << '/' << first_work[3] << '/'
              << first_work[4] << '/' << first_work[5]
              << " accounted=" << first_work_accounted
              << " identity=" << first_identity << " abort="
              << static_cast<int>(aborted.status) << '\n';
    return false;
  }

  std::vector<double> after_abort(expected_initial.size(), -3.0);
  std::vector<double> sigma_after_abort(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool abort_export = export_committed(after_abort, sigma_after_abort);
  const auto after_abort_coordinates = session.coordinates();
  if (!All(abort_export && SameValues(after_abort, committed_initial) &&
               (!igr || SameValues(sigma_after_abort, sigma_initial)) &&
               after_abort_coordinates.time == before.time &&
               after_abort_coordinates.dt == before.dt &&
               after_abort_coordinates.step == before.step &&
               after_abort_coordinates.generation == before.generation,
           request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo abort mode=" << mode_name
              << " rollback=" << abort_export
              << '/' << SameValues(after_abort, committed_initial) << ' '
              << (!igr || SameValues(sigma_after_abort, sigma_initial)) << ' '
              << after_abort_coordinates.time << '/'
              << after_abort_coordinates.dt << '/'
              << after_abort_coordinates.step << '/'
              << after_abort_coordinates.generation << " expected="
              << before.time << '/' << before.dt << '/' << before.step
              << '/' << before.generation << '\n';
    return false;
  }

  GenericEulerCudaPreparedStep retry_authority;
  GenericEulerHdgCudaSession::prepared_output_type retry_output{};
  const auto retry_prepared = session.Prepare(
      dt, retry_authority, retry_output);
  if (!All(retry_prepared.status == HdgEulerSessionStatus::prepared,
           request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo retry mode=" << mode_name
              << " prepare_status="
              << static_cast<int>(retry_prepared.status) << '\n';
    return false;
  }

  std::array<int, 6> retry_work{};
  const bool retry_work_accounted =
      CudaQualificationTestAccess::VerifyPreparedWorkAccounting(
          session, true, retry_work);
  std::vector<double> retry_candidate(expected_initial.size(), -4.0);
  std::vector<double> sigma_retry_candidate(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool retry_export = export_prepared(
      retry_authority, retry_output, retry_candidate,
      sigma_retry_candidate);
  auto normalized_retry_descriptor = retry_output.descriptor;
  normalized_retry_descriptor.cursor = first_descriptor.cursor;
  normalized_retry_descriptor.coordinates.cursor =
      first_descriptor.coordinates.cursor;
  normalized_retry_descriptor.coordinate_fingerprint =
      first_descriptor.coordinate_fingerprint;
  const bool expected_retry_identity =
      first_output.attempt.retry_index == 0 &&
      retry_output.attempt.retry_index == 1 &&
      retry_output.attempt.sequence == first_output.attempt.sequence + 1 &&
      retry_output.attempt.committed_step ==
          first_output.attempt.committed_step &&
      retry_output.attempt.committed_generation ==
          first_output.attempt.committed_generation &&
      dg::hdg::runtime::detail::DoubleBits(retry_output.attempt.dt) ==
          dg::hdg::runtime::detail::DoubleBits(first_output.attempt.dt) &&
      first_descriptor.cursor == first_output.attempt.sequence &&
      first_descriptor.coordinates.cursor == first_output.attempt.sequence &&
      retry_output.descriptor.cursor == retry_output.attempt.sequence &&
      retry_output.descriptor.coordinates.cursor ==
          retry_output.attempt.sequence;
  const bool retry_identity = retry_work_accounted &&
      retry_work == first_work && retry_export &&
      SameValues(retry_candidate, first_candidate) &&
      (!igr || NearValues(sigma_retry_candidate,
                          sigma_first_candidate)) &&
      expected_retry_identity &&
      dg::hdg::runtime::detail::SameCheckpoint(
          normalized_retry_descriptor, first_descriptor) &&
      retry_output.local_interior_hash == first_local_interior_hash &&
      retry_output.local_trace_hash == first_local_trace_hash;
  if (!All(retry_identity, request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo retry mode=" << mode_name
              << " identity="
              << retry_work_accounted << '/' << (retry_work == first_work)
              << '/' << retry_export << '/'
              << SameValues(retry_candidate, first_candidate) << '/'
              << (!igr || NearValues(sigma_retry_candidate,
                                     sigma_first_candidate)) << '/'
              << expected_retry_identity << '/'
              << dg::hdg::runtime::detail::SameCheckpoint(
                     normalized_retry_descriptor, first_descriptor)
              << '/' << (retry_output.local_interior_hash ==
                          first_local_interior_hash)
              << '/' << (retry_output.local_trace_hash ==
                          first_local_trace_hash) << '\n';
    (void)session.Abort(std::move(retry_authority), retry_output);
    return false;
  }

  const bool acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_output);
  if (!All(acknowledged, request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo retry mode=" << mode_name
              << " acknowledge failed\n";
    (void)session.Abort(std::move(retry_authority), retry_output);
    return false;
  }
  const auto committed = session.Commit(
      std::move(retry_authority), retry_output);
  if (!All(committed.status == HdgEulerSessionStatus::committed,
           request.communicator))
  {
    std::cerr << "CUDA rejected-Armijo mode=" << mode_name
              << " commit_status="
              << static_cast<int>(committed.status) << '\n';
    return false;
  }

  std::vector<double> committed_final(expected_initial.size(), -5.0);
  std::vector<double> sigma_committed_final(
      igr ? fields.volume_scalar_count : 0, unwritten);
  const bool final_export = export_committed(
      committed_final, sigma_committed_final);
  const auto after = session.coordinates();
  const bool final_ready = final_export &&
      SameValues(committed_final, retry_candidate) &&
      (!igr || SameValues(sigma_committed_final,
                          sigma_retry_candidate)) &&
      retry_output.successful_commit && after.step == before.step + 1 &&
      after.generation == before.generation + 1 &&
      after.time == before.time + dt && after.dt == dt;
  if (!All(final_ready, request.communicator))
    std::cerr << "CUDA rejected-Armijo final mode=" << mode_name
              << " values=" << final_export << '/'
              << SameValues(committed_final, retry_candidate) << '/'
              << (!igr || SameValues(sigma_committed_final,
                                     sigma_retry_candidate)) << '/'
              << retry_output.successful_commit << ' ' << after.time << '/'
              << after.dt << '/' << after.step << '/' << after.generation
              << '\n';
  return All(final_ready, request.communicator);
}

bool CheckRejectedArmijoRollback(const int order, const int ranks)
{
  return CheckRejectedArmijoRollbackForSelection(
             order, ranks, RegularizationKind::none) &&
      CheckRejectedArmijoRollbackForSelection(
             order, ranks, RegularizationKind::igr);
}

template <typename Engine>
bool CaptureAndCommitPartitionedCheckpoint(
    Engine& engine, const double dt,
    dg::hdg::runtime::cuda_w8a::CudaSerializedCheckpointV1& checkpoint,
    CudaStepEvidence& evidence)
{
  CudaPrecommitSnapshot snapshot{};
  const bool prepared = engine.PrepareStep(false, dt, snapshot, evidence);
  const bool staged = prepared &&
      engine.StagePreparedCheckpoint(checkpoint, evidence);
  const bool marked = staged && engine.MarkOutputStaged(checkpoint);
  const bool commit_prepared = marked &&
      engine.PrepareCommit(checkpoint, evidence);
  const bool provisional = commit_prepared &&
      engine.CommitPrepared(checkpoint, evidence);
  const bool committed = provisional &&
      engine.FinalizeCommit(checkpoint, evidence);
  return prepared && snapshot.numerical_candidate_ready && staged && marked &&
      commit_prepared && provisional && committed &&
      checkpoint.candidate_frozen && checkpoint.checkpoint_staged &&
      checkpoint.output_staged && checkpoint.successful_commit &&
      checkpoint.descriptor.successful_commit &&
      checkpoint.staging_range.complete &&
      evidence.status == CudaRunStatus::committed && evidence.committed &&
      !evidence.retry_available && evidence.scalar_closure_solves == 1 &&
      evidence.scalar_closure_linear_solve_attempts > 0 &&
      evidence.no_fallback;
}

template <typename Engine>
bool CommitPartitionedContinuation(Engine& engine, const double dt,
                                   CudaStepEvidence& evidence)
{
  CudaPrecommitSnapshot snapshot{};
  CudaPreparedOutputMetadataV1 output{};
  const bool prepared = engine.PrepareStep(false, dt, snapshot, evidence);
  const bool staged = prepared &&
      engine.StagePreparedOutputMetadata(output, evidence);
  const bool marked = staged && engine.MarkOutputStaged(output);
  const bool commit_prepared = marked &&
      engine.PrepareCommit(output, evidence);
  const bool provisional = commit_prepared &&
      engine.CommitPrepared(output, evidence);
  const bool committed = provisional &&
      engine.FinalizeCommit(output, evidence);
  return prepared && snapshot.numerical_candidate_ready && staged && marked &&
      commit_prepared && provisional && committed &&
      output.candidate_frozen && output.checkpoint_staged &&
      output.output_staged && output.successful_commit &&
      output.staging_range.complete &&
      evidence.status == CudaRunStatus::committed && evidence.committed &&
      !evidence.retry_available && evidence.scalar_closure_solves == 1 &&
      evidence.scalar_closure_linear_solve_attempts > 0 &&
      evidence.no_fallback;
}

template <typename Engine>
bool ExportPartitionedCommittedFields(
    Engine& engine, const std::size_t scalar_count,
    std::vector<double>& conservative, std::vector<double>& sigma)
{
  conservative.assign(scalar_count * components,
                      std::numeric_limits<double>::quiet_NaN());
  sigma.assign(scalar_count, std::numeric_limits<double>::quiet_NaN());
  return engine.ExportCommittedFields(HdgEulerHostInteriorFields{
      conservative.data(), sigma.data(), scalar_count});
}

template <typename Selection>
bool CheckPartitionedCheckpointRestartForSelection(
    const int order, const int ranks)
{
  // Keep this restart gate bounded to the representative p2 selected smoke.
  // Each p2 ordinary/tensor executable covers its configured rank count.
  if (order != 2) return true;

  using Engine = PersistentCudaSmoothEulerEngine<Selection>;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  CudaConfig config = MakeDirectConfig(request, ranks, order);
  constexpr double dt = 1.0e-5;
  auto checkpoint = std::make_unique<
      dg::hdg::runtime::cuda_w8a::CudaSerializedCheckpointV1>();
  std::vector<double> first_conservative, first_sigma;
  std::vector<double> continued_conservative, continued_sigma;
  dg::hdg::runtime::StateCoordinates first_coordinates{};
  dg::hdg::runtime::StateCoordinates continued_coordinates{};

  {
    auto source = std::make_unique<Engine>(config);
    CudaRunEvidence initialization{};
    if (!All(source->Initialize(initialization) &&
                 source->contract() != nullptr,
             request.communicator))
      return false;

    CudaStepEvidence captured{};
    const bool checkpoint_committed = CaptureAndCommitPartitionedCheckpoint(
        *source, dt, *checkpoint, captured);
    first_coordinates = source->contract() != nullptr
        ? source->contract()->coordinates()
        : dg::hdg::runtime::StateCoordinates{};
    const bool checkpoint_identity = checkpoint_committed &&
        checkpoint->attempt.sequence == checkpoint->descriptor.cursor &&
        checkpoint->attempt.sequence ==
            checkpoint->descriptor.coordinates.cursor &&
        checkpoint->local_interior_hash != dg::hdg::runtime::StateHash{} &&
        checkpoint->local_trace_hash != dg::hdg::runtime::StateHash{} &&
        dg::hdg::runtime::detail::SameCoordinates(
            first_coordinates, checkpoint->descriptor.coordinates);
    const bool first_export = ExportPartitionedCommittedFields(
        *source, fields.volume_scalar_count,
        first_conservative, first_sigma);
    if (!All(checkpoint_identity && first_export,
             request.communicator))
      return false;

    CudaStepEvidence source_continuation{};
    const bool source_continued = CommitPartitionedContinuation(
        *source, dt, source_continuation);
    continued_coordinates = source->contract() != nullptr
        ? source->contract()->coordinates()
        : dg::hdg::runtime::StateCoordinates{};
    const bool continuation_export = ExportPartitionedCommittedFields(
        *source, fields.volume_scalar_count,
        continued_conservative, continued_sigma);
    if (!All(source_continued && continuation_export &&
                 continued_coordinates.step == first_coordinates.step + 1 &&
                 continued_coordinates.generation ==
                     first_coordinates.generation + 1 &&
                 continued_coordinates.time == first_coordinates.time + dt,
             request.communicator))
      return false;
  }

  {
    auto restarted = std::make_unique<Engine>(config);
    CudaRunEvidence initialization{};
    if (!All(restarted->Initialize(initialization) &&
                 restarted->contract() != nullptr,
             request.communicator))
      return false;
    CudaStepEvidence restored_evidence{};
    const bool restored = restarted->Restore(*checkpoint, restored_evidence);
    std::vector<double> restored_conservative, restored_sigma;
    const bool restored_export = ExportPartitionedCommittedFields(
        *restarted, fields.volume_scalar_count,
        restored_conservative, restored_sigma);
    const auto restored_coordinates = restarted->contract() != nullptr
        ? restarted->contract()->coordinates()
        : dg::hdg::runtime::StateCoordinates{};
    const bool restored_state = restored && restored_export &&
        restored_evidence.status == CudaRunStatus::committed &&
        restored_evidence.committed &&
        restored_evidence.collective_consensus &&
        restored_evidence.contract_reason == FailureReason::none &&
        restored_evidence.after_interior_hash ==
            checkpoint->descriptor.interior_hash &&
        restored_evidence.after_trace_hash ==
            checkpoint->descriptor.trace_hash &&
        dg::hdg::runtime::detail::SameCoordinates(
            restored_coordinates, first_coordinates) &&
        SameValues(restored_conservative, first_conservative) &&
        SameValues(restored_sigma, first_sigma);
    if (!All(restored_state, request.communicator)) return false;

    CudaStepEvidence restarted_continuation{};
    const bool continued = CommitPartitionedContinuation(
        *restarted, dt, restarted_continuation);
    std::vector<double> restart_continued_conservative,
        restart_continued_sigma;
    const bool continued_export = ExportPartitionedCommittedFields(
        *restarted, fields.volume_scalar_count,
        restart_continued_conservative, restart_continued_sigma);
    const auto restart_continued_coordinates =
        restarted->contract() != nullptr
        ? restarted->contract()->coordinates()
        : dg::hdg::runtime::StateCoordinates{};
    const bool continuation_agreement = continued && continued_export &&
        dg::hdg::runtime::detail::SameCoordinates(
            restart_continued_coordinates, continued_coordinates) &&
        NearValues(restart_continued_conservative,
                   continued_conservative) &&
        NearValues(restart_continued_sigma, continued_sigma);
    if (!All(continuation_agreement, request.communicator)) return false;
  }

  // The corrupted checkpoint remains self-consistent at both hash layers.
  // Restore must import it into scratch, reject the scalar closure, and leave
  // the pristine committed state and closure certificate publishable.
  auto corrupted = std::make_unique<
      dg::hdg::runtime::cuda_w8a::CudaSerializedCheckpointV1>(*checkpoint);
  {
    auto rejected = std::make_unique<Engine>(config);
    CudaRunEvidence initialization{};
    if (!All(rejected->Initialize(initialization) &&
                 rejected->contract() != nullptr,
             request.communicator))
      return false;
    std::vector<double> initial_conservative, initial_sigma;
    const bool initial_export = ExportPartitionedCommittedFields(
        *rejected, fields.volume_scalar_count,
        initial_conservative, initial_sigma);
    const auto before = rejected->contract()->coordinates();
    const bool corrupted_hashes =
        CudaQualificationTestAccess::
            CorruptHashConsistentPartitionedAuxiliaryCheckpoint(
                *rejected, *corrupted) &&
        corrupted->local_trace_hash == checkpoint->local_trace_hash &&
        corrupted->descriptor.trace_hash ==
            checkpoint->descriptor.trace_hash;
    if (!All(initial_export && corrupted_hashes, request.communicator))
      return false;

    CudaStepEvidence rejection{};
    const bool unexpectedly_restored = rejected->Restore(*corrupted, rejection);
    const bool reached_scalar_validation =
        CudaQualificationTestAccess::
            HashConsistentCheckpointReachedScalarValidation(
                *rejected, *corrupted);
    std::vector<double> after_conservative, after_sigma;
    const bool after_export = ExportPartitionedCommittedFields(
        *rejected, fields.volume_scalar_count,
        after_conservative, after_sigma);
    const auto after = rejected->contract() != nullptr
        ? rejected->contract()->coordinates()
        : dg::hdg::runtime::StateCoordinates{};
    const bool rejected_without_publication = !unexpectedly_restored &&
        reached_scalar_validation && after_export &&
        rejection.status == CudaRunStatus::runtime_failure &&
        rejection.contract_reason == FailureReason::restart_incompatible &&
        !rejection.committed &&
        dg::hdg::runtime::detail::SameCoordinates(before, after) &&
        SameValues(after_conservative, initial_conservative) &&
        SameValues(after_sigma, initial_sigma) &&
        rejected->contract() != nullptr &&
        rejected->contract()->status() == RunStatus::ready;
    if (!All(rejected_without_publication, request.communicator)) return false;
  }
  return true;
}

bool CheckPartitionedCheckpointRestart(const int order, const int ranks)
{
  if (ranks == 1)
    return CheckPartitionedCheckpointRestartForSelection<
        dg::hdg::runtime::PartitionedBackwardEulerCudaRank1RuntimeSelection>(
            order, ranks);
  return CheckPartitionedCheckpointRestartForSelection<
      dg::hdg::runtime::PartitionedBackwardEulerCudaRank2RuntimeSelection>(
          order, ranks);
}

template <typename Selection>
bool CheckPartitionedFailureInjectionForSelection(
    const int order, const int ranks)
{
  // This direct-engine fixture isolates abort/retry and closure-certificate
  // semantics.  Physical-policy lowering is owned by the generic facade and
  // is already exercised by the physical lifecycle cases above.
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::nonzero, InitialMode::constant);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  CudaConfig config = MakeDirectConfig(request, ranks, order);
  config.injection.enabled = true;
  config.injection.phase = CudaFailurePhase::diagnostics;
  config.injection.target_step = 1;
  config.injection.source_rank = ranks - 1;

  auto engine = std::make_unique<
      PersistentCudaSmoothEulerEngine<Selection>>(config);
  CudaRunEvidence initialization{};
  if (!All(engine->Initialize(initialization), request.communicator))
    return false;

  constexpr double dt = 1.0e-5;
  CudaPrecommitSnapshot injected_snapshot{};
  CudaStepEvidence injected{};
  const bool injected_prepared = engine->PrepareStep(
      false, dt, injected_snapshot, injected);
  if (!All(injected_prepared &&
               injected_snapshot.numerical_candidate_ready &&
               injected.source_rank == ranks - 1,
           request.communicator))
    return false;

  CudaPreparedOutputMetadataV1 denied_output{};
  const bool rolled_back = engine->PrepareCommit(
      denied_output, injected);
  if (!All(rolled_back && injected.retry_available && !injected.committed &&
               injected.status == CudaRunStatus::retry_available &&
               injected.contract_reason == FailureReason::backend_failure &&
               injected.coordinates_preserved &&
               injected.committed_bytes_preserved &&
               engine->contract() != nullptr &&
               engine->contract()->status() == RunStatus::retry_available,
           request.communicator))
    return false;

  CudaPrecommitSnapshot retry_snapshot{};
  CudaStepEvidence retry{};
  CudaPreparedOutputMetadataV1 retry_output{};
  const bool retry_prepared = engine->PrepareStep(
      true, dt, retry_snapshot, retry);
  const bool staged = retry_prepared &&
      engine->StagePreparedOutputMetadata(retry_output, retry);
  const bool marked = staged && engine->MarkOutputStaged(retry_output);
  const bool prepared_commit = marked &&
      engine->PrepareCommit(retry_output, retry);
  const bool provisional = prepared_commit &&
      engine->CommitPrepared(retry_output, retry);
  const bool committed = provisional &&
      engine->FinalizeCommit(retry_output, retry);
  return All(retry_prepared && retry_snapshot.numerical_candidate_ready &&
                 staged && marked && prepared_commit && provisional &&
                 committed && retry_output.successful_commit &&
                 retry.status == CudaRunStatus::committed &&
                 retry.committed && !retry.retry_available &&
                 retry.source_rank == -1,
             request.communicator);
}

bool CheckPartitionedFailureInjection(const int order, const int ranks)
{
  if (ranks == 1)
    return CheckPartitionedFailureInjectionForSelection<
        dg::hdg::runtime::PartitionedBackwardEulerCudaRank1RuntimeSelection>(
            order, ranks);
  return CheckPartitionedFailureInjectionForSelection<
      dg::hdg::runtime::PartitionedBackwardEulerCudaRank2RuntimeSelection>(
      order, ranks);
}

struct Rodas3ScreenRecord
{
  double setup_seconds = std::numeric_limits<double>::infinity();
  double step_seconds = std::numeric_limits<double>::infinity();
  CudaRunEvidence initialization{};
  CudaStepEvidence evidence{};
  CudaStepEvidence retry_evidence{};
  CudaStepEvidence rollback{};
  CudaRodas3ScreenObservation observation{};
  CudaRodas3ScreenObservation retry_observation{};
  CudaInitializedRodas3IgrClosureAudit initial_closure_audit{};
  dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
  dg::hdg::generic::HdgEulerPreparedAssessment retry_assessment{};
  bool state_changed = false;
  bool architecture_passed = false;
  bool numerical_passed = false;
  bool lifecycle_passed = false;
  bool host_staged_passed = false;
  bool rollback_passed = false;
  bool retry_passed = false;
  bool exact_jacobian_passed = false;
  bool initial_closure_audited = false;
  bool initial_closure_passed = true;
  std::array<double, 6> exact_jacobian_relative_errors{};
  bool passed = false;
};

bool RunRodas3RankOneStep(
    const CudaEngineKind engine_kind,
    const RegularizationKind regularization,
    Rodas3ScreenRecord& record,
    const bool physical = false)
{
  record = {};
  constexpr int order = 1;
  constexpr double dt = 1.0e-6;
  mfem::ParMesh mesh = physical ? MakePhysicalMesh() : MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, physical ? SourceMode::nonzero : SourceMode::empty,
      InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, physical, regularization);
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;
  CudaConfig config = MakeDirectConfig(request, 1, order);
  config.dt = dt;
  config.cuda_engine = engine_kind;
  std::array<CudaStaticBoundaryPolicy, 4> device_policies{};
  if (physical)
  {
    const std::array<double, components> reference{{
        1.0, 0.05, -0.02, 2.60}};
    const std::array<double, components> unused{};
    device_policies = {{
        {1, CudaStaticBoundaryPolicyKind::slip_wall, unused, 0.0},
        {2, CudaStaticBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
        {3, CudaStaticBoundaryPolicyKind::far_field, reference, 0.0},
        {4, CudaStaticBoundaryPolicyKind::supersonic_inlet, reference, 0.0}}};
    config.borrowed_global_topology = StableIdentity::Copy(
        "generic.euler.request.topology.v1");
    config.borrowed_boundary_policies = {
        device_policies.data(), device_policies.size(), 1.4};
  }

  using Engine = PersistentCudaSmoothEulerEngine<
      dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>;
  const auto setup_begin = std::chrono::steady_clock::now();
  std::unique_ptr<Engine> engine;
  if (engine_kind == CudaEngineKind::tensor_core)
  {
    const CudaTracePolicyKind requested_policy =
        dg::hdg::runtime::cuda_w8a::DefaultCudaTracePolicyRequest(
            engine_kind, regularization == RegularizationKind::igr, 1,
            EulerHdgTimeTreatment::rodas3);
    engine = CudaQualificationTestAccess::CreateWithTracePolicy<
        dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>(
            config, requested_policy);
  }
  else if (engine_kind == CudaEngineKind::ordinary)
  {
    engine = std::make_unique<Engine>(config);
  }
  CudaRunEvidence initialization{};
  const bool initialized = engine && engine->Initialize(initialization) &&
      engine->contract() != nullptr;
  const auto setup_end = std::chrono::steady_clock::now();
  record.setup_seconds = std::chrono::duration<double>(
      setup_end - setup_begin).count();
  if (!All(initialized, request.communicator))
    return false;

  const bool initial_closure_audit_required =
      engine_kind == CudaEngineKind::tensor_core &&
      regularization == RegularizationKind::igr && !physical;
  record.initial_closure_audited = initial_closure_audit_required;
  record.initial_closure_passed = !initial_closure_audit_required ||
      CudaQualificationTestAccess::AuditInitializedRodas3IgrClosure(
          *engine, fields.initial.data(), fields.initial.size(),
          fields.initial_true_trace.data(), fields.initial_true_trace.size(),
          record.initial_closure_audit);
  if (!All(record.initial_closure_passed, request.communicator)) return false;

  constexpr double exact_jacobian_tolerance = 2.5e-4;
  record.exact_jacobian_passed = !physical ||
      CudaQualificationTestAccess::VerifyAssemblerJacobian(
          *engine, exact_jacobian_tolerance,
          record.exact_jacobian_relative_errors, true);
  if (!All(record.exact_jacobian_passed, request.communicator)) return false;

  CudaPrecommitSnapshot snapshot{};
  CudaPreparedOutputMetadataV1 output{};
  const auto step_begin = std::chrono::steady_clock::now();
  const bool prepared = engine->PrepareStep(
      false, dt, snapshot, record.evidence);
  const bool captured = prepared &&
      CudaQualificationTestAccess::CapturePreparedRodas3(
          *engine, record.observation);
  const bool staged = captured &&
      engine->StagePreparedOutputMetadata(output, record.evidence);
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      fixed_policy{1.0e-8, 1.0e-5, 0.25};
  const bool assessed = staged && engine->AssessPrepared(
      output, fixed_policy, record.assessment);
  const bool marked = assessed &&
      record.assessment.algebraic_correction_gate &&
      engine->MarkOutputStaged(output);
  const bool commit_prepared = marked &&
      engine->PrepareCommit(output, record.evidence);
  const bool provisional = commit_prepared &&
      engine->CommitPrepared(output, record.evidence);
  const bool committed = provisional &&
      engine->FinalizeCommit(output, record.evidence);
  const auto step_end = std::chrono::steady_clock::now();
  record.step_seconds = std::chrono::duration<double>(
      step_end - step_begin).count();

  const auto& evidence = record.evidence;
  const auto& observation = record.observation;
  record.state_changed = evidence.before_interior_hash !=
      evidence.after_interior_hash;
  const bool preconditioner_architecture =
      engine_kind == CudaEngineKind::tensor_core
      ? evidence.common_shift_preconditioner_setups == 1 &&
          evidence.common_shift_preconditioner_reuses == 3
      : evidence.common_shift_preconditioner_setups == 4 &&
          evidence.common_shift_preconditioner_reuses == 0 &&
          evidence.tensor_core.ilu_factor_setups >= 4 &&
          evidence.tensor_core.ilu_factor_reuses == 0;
  record.architecture_passed =
      evidence.common_shift_operator_builds == 1 &&
      evidence.common_shift_operator_reuses == 3 &&
      evidence.condensed_rhs_refreshes == 3 &&
      evidence.common_shift_trace_solves == 4 &&
      evidence.positivity_limiter_calls == 0 &&
      evidence.positivity_limiter_active_calls == 0 &&
      evidence.positivity_limiter_inadmissible_mean_rejections == 0 &&
      evidence.positivity_algebraic_projections == 0 &&
      evidence.common_shift_operator_rebuilds == 0 &&
      evidence.maximum_positivity_limiter_correction_norm == 0.0 &&
      preconditioner_architecture;
  record.numerical_passed = observation.candidate_ready &&
      observation.finite && observation.admissible &&
      observation.recovery && observation.true_residual &&
      observation.conservation && observation.telemetry &&
      observation.success && std::isfinite(evidence.temporal_error_norm) &&
      evidence.temporal_error_norm >= 0.0 &&
      std::isfinite(evidence.final_algebraic_residual) &&
      evidence.final_algebraic_residual >= 0.0 &&
      std::isfinite(evidence.final_algebraic_correction_norm) &&
      evidence.final_algebraic_correction_norm >= 0.0 &&
      std::isfinite(evidence.true_relative_residual) &&
      evidence.true_relative_residual <=
          request.numerical_policy.true_residual_relative_tolerance &&
      std::isfinite(observation.maximum_conservation_error) &&
      std::isfinite(observation.maximum_conservation_relative_error) &&
      observation.maximum_conservation_relative_error <=
          request.numerical_policy.conservation_relative_tolerance;
  record.lifecycle_passed = prepared && captured && staged && assessed &&
      record.assessment.finite_gate && record.assessment.solver_gate &&
      record.assessment.residual_gate &&
      record.assessment.admissibility_gate &&
      record.assessment.algebraic_closure_gate &&
      record.assessment.algebraic_correction_gate && marked &&
      commit_prepared && provisional && committed &&
      snapshot.numerical_candidate_ready && output.candidate_frozen &&
      output.checkpoint_staged && output.output_staged &&
      output.successful_commit && evidence.status == CudaRunStatus::committed &&
      evidence.committed && !evidence.retry_available &&
      evidence.no_fallback && evidence.device_resident &&
      evidence.after.step == evidence.before.step + 1 &&
      evidence.after.generation == evidence.before.generation + 1 &&
      evidence.after.time == evidence.before.time + dt;
  record.passed = record.architecture_passed && record.numerical_passed &&
      record.lifecycle_passed && record.exact_jacobian_passed &&
      record.initial_closure_passed &&
      record.state_changed && std::isfinite(record.setup_seconds) &&
      record.setup_seconds >= 0.0 && std::isfinite(record.step_seconds) &&
      record.step_seconds > 0.0;
  return All(record.passed, request.communicator);
}

bool RunRodas3RankTwoStep(
    const CudaEngineKind engine_kind,
    const RegularizationKind regularization,
    Rodas3ScreenRecord& record)
{
  record = {};
  constexpr int order = 1;
  constexpr int ranks = 2;
  constexpr double dt = 1.0e-6;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, regularization);
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;
  CudaConfig config = MakeDirectConfig(request, ranks, order);
  config.dt = dt;
  config.cuda_engine = engine_kind;
  using Engine = PersistentCudaSmoothEulerEngine<
      dg::hdg::runtime::Rodas3CudaRank2RuntimeSelection>;
  const auto setup_begin = std::chrono::steady_clock::now();
  auto engine = std::make_unique<Engine>(config);
  const bool initialized = engine && engine->Initialize(record.initialization) &&
      engine->contract() != nullptr;
  const auto setup_end = std::chrono::steady_clock::now();
  record.setup_seconds = std::chrono::duration<double>(
      setup_end - setup_begin).count();
  if (!All(initialized, request.communicator)) return false;

  CudaPrecommitSnapshot snapshot{};
  CudaPreparedOutputMetadataV1 output{};
  const auto step_begin = std::chrono::steady_clock::now();
  const bool prepared = engine->PrepareStep(
      false, dt, snapshot, record.evidence);
  const bool captured = prepared &&
      CudaQualificationTestAccess::CapturePreparedRodas3(
          *engine, record.observation);
  const bool staged = captured &&
      engine->StagePreparedOutputMetadata(output, record.evidence);
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      fixed_policy{1.0e-8, 1.0e-5, 0.25};
  const bool assessed = staged && engine->AssessPrepared(
      output, fixed_policy, record.assessment);
  const bool staged_artifact = output.candidate_frozen &&
      output.checkpoint_staged && !output.output_staged &&
      !output.successful_commit;
  const bool rolled_back = assessed &&
      engine->AbortPrepared(output, FailureReason::backend_failure,
                            record.rollback);
  const bool aborted_artifacts_cleared = !output.candidate_frozen &&
      !output.checkpoint_staged && !output.output_staged &&
      !output.successful_commit;
  record.rollback_passed = rolled_back && !record.rollback.committed &&
      record.rollback.retry_available && record.rollback.collective_consensus &&
      record.rollback.coordinates_preserved &&
      record.rollback.committed_bytes_preserved &&
      record.rollback.status == CudaRunStatus::retry_available &&
      record.rollback.contract_reason == FailureReason::backend_failure &&
      record.rollback.before_interior_hash ==
          record.rollback.after_interior_hash &&
      record.rollback.before_trace_hash == record.rollback.after_trace_hash &&
      engine->contract() != nullptr &&
      engine->contract()->status() == RunStatus::retry_available &&
      aborted_artifacts_cleared;

  CudaPrecommitSnapshot retry_snapshot{};
  CudaPreparedOutputMetadataV1 retry_output{};
  const bool retry_prepared = record.rollback_passed && engine->PrepareStep(
      true, dt, retry_snapshot, record.retry_evidence);
  const bool retry_captured = retry_prepared &&
      CudaQualificationTestAccess::CapturePreparedRodas3(
          *engine, record.retry_observation);
  const bool retry_staged = retry_captured &&
      engine->StagePreparedOutputMetadata(
          retry_output, record.retry_evidence);
  const bool retry_assessed = retry_staged && engine->AssessPrepared(
      retry_output, fixed_policy, record.retry_assessment);
  const bool retry_marked = retry_assessed &&
      record.retry_assessment.algebraic_correction_gate &&
      engine->MarkOutputStaged(retry_output);
  const bool retry_commit_prepared = retry_marked &&
      engine->PrepareCommit(retry_output, record.retry_evidence);
  const bool retry_provisional = retry_commit_prepared &&
      engine->CommitPrepared(retry_output, record.retry_evidence);
  const bool retry_committed = retry_provisional &&
      engine->FinalizeCommit(retry_output, record.retry_evidence);
  const auto step_end = std::chrono::steady_clock::now();
  record.step_seconds = std::chrono::duration<double>(
      step_end - step_begin).count();

  const auto& evidence = record.evidence;
  const auto& observation = record.observation;
  const bool retained_ilu = engine_kind == CudaEngineKind::tensor_core &&
      regularization == RegularizationKind::igr;
  const bool diagonal_scale = engine_kind == CudaEngineKind::tensor_core &&
      regularization == RegularizationKind::none;
  const std::uint64_t expected_physical_trace_solves =
      regularization == RegularizationKind::igr ? 6U : 7U;
  const auto architecture_exact = [&](const CudaStepEvidence& value)
  {
    const bool preconditioner_exact = retained_ilu
        ? value.common_shift_preconditioner_setups == 1 &&
            value.common_shift_preconditioner_reuses == 3 &&
            value.tensor_core.ilu_factor_setups == 1 &&
            value.tensor_core.ilu_factor_reuses == 5
        : value.common_shift_preconditioner_setups == 4 &&
            value.common_shift_preconditioner_reuses == 0 &&
            (diagonal_scale
                 ? value.tensor_core.ilu_factor_setups == 0 &&
                     value.tensor_core.ilu_factor_reuses == 0
                 : value.tensor_core.ilu_factor_setups ==
                       expected_physical_trace_solves &&
                     value.tensor_core.ilu_factor_reuses == 0);
    return value.common_shift_operator_builds == 1 &&
        value.common_shift_operator_reuses == 3 &&
        value.condensed_rhs_refreshes == 3 &&
        value.common_shift_trace_solves == 4 && preconditioner_exact;
  };
  const auto numerical_exact = [&](
      const CudaStepEvidence& value,
      const CudaRodas3ScreenObservation& observed,
      const dg::hdg::generic::HdgEulerPreparedAssessment& assessed_value)
  {
    return observed.candidate_ready && observed.finite &&
        observed.admissible && observed.recovery && observed.true_residual &&
        observed.conservation && observed.telemetry && observed.success &&
        assessed_value.finite_gate && assessed_value.solver_gate &&
        assessed_value.residual_gate && assessed_value.admissibility_gate &&
        assessed_value.algebraic_closure_gate &&
        assessed_value.algebraic_correction_gate &&
        std::isfinite(value.temporal_error_norm) &&
        value.temporal_error_norm >= 0.0 &&
        std::isfinite(value.final_algebraic_residual) &&
        value.final_algebraic_residual >= 0.0 &&
        std::isfinite(value.final_algebraic_correction_norm) &&
        value.final_algebraic_correction_norm >= 0.0 &&
        std::isfinite(value.true_relative_residual) &&
        value.true_relative_residual <=
            request.numerical_policy.true_residual_relative_tolerance &&
        std::isfinite(observed.maximum_conservation_error) &&
        std::isfinite(observed.maximum_conservation_relative_error) &&
        observed.maximum_conservation_relative_error <=
            request.numerical_policy.conservation_relative_tolerance;
  };
  const auto host_staged_exact = [&](const CudaStepEvidence& value)
  {
    return value.tensor_core.
            hypre_host_staged_rank2_trace_solver_configured &&
        value.tensor_core.hypre_trace_solve_calls ==
            expected_physical_trace_solves &&
        value.transfers.complete && value.transfers.host_to_device_bytes > 0 &&
        value.transfers.device_to_host_bytes > 0 &&
        value.transfers.mpi_staging_bytes > 0;
  };
  record.architecture_passed = architecture_exact(evidence) &&
      architecture_exact(record.retry_evidence);
  record.numerical_passed = numerical_exact(
      evidence, observation, record.assessment) && numerical_exact(
      record.retry_evidence, record.retry_observation,
      record.retry_assessment);
  record.state_changed = snapshot.candidate_interior_hash !=
          evidence.before_interior_hash &&
      retry_snapshot.candidate_interior_hash !=
          record.retry_evidence.before_interior_hash;
  record.host_staged_passed = record.initialization.ranks == ranks &&
      record.initialization.same_physical_gpu &&
      record.initialization.host_staged_rank2 &&
      record.initialization.no_host_solver_fallback &&
      record.initialization.mfem_device_valid &&
      record.initialization.hypre_cuda_umpire_device &&
      record.initialization.signed_owner_ghost_semantics &&
      record.initialization.tensor_core.
          hypre_host_staged_rank2_trace_solver_configured &&
      host_staged_exact(evidence) && host_staged_exact(record.retry_evidence);
  record.retry_passed = retry_prepared && retry_captured && retry_staged &&
      retry_assessed && retry_marked && retry_commit_prepared &&
      retry_provisional && retry_committed &&
      retry_snapshot.numerical_candidate_ready &&
      retry_output.candidate_frozen && retry_output.checkpoint_staged &&
      retry_output.output_staged && retry_output.successful_commit &&
      record.retry_evidence.status == CudaRunStatus::committed &&
      record.retry_evidence.committed &&
      !record.retry_evidence.retry_available &&
      record.retry_evidence.no_fallback &&
      record.retry_evidence.device_resident &&
      record.retry_evidence.before.step == evidence.before.step &&
      record.retry_evidence.before.generation == evidence.before.generation &&
      record.retry_evidence.before.time == evidence.before.time &&
      record.retry_evidence.after.step == evidence.before.step + 1 &&
      record.retry_evidence.after.generation ==
          evidence.before.generation + 1 &&
      record.retry_evidence.after.time == evidence.before.time + dt;
  record.lifecycle_passed = prepared && captured && staged && assessed &&
      staged_artifact && snapshot.numerical_candidate_ready &&
      evidence.no_fallback && evidence.device_resident &&
      record.host_staged_passed && record.rollback_passed &&
      record.retry_passed;
  record.passed = record.architecture_passed && record.numerical_passed &&
      record.lifecycle_passed && record.state_changed &&
      std::isfinite(record.setup_seconds) && record.setup_seconds >= 0.0 &&
      std::isfinite(record.step_seconds) && record.step_seconds > 0.0;
  return All(record.passed, request.communicator);
}

bool CheckRodas3PreparedAssessmentAuthority()
{
  constexpr int order = 1;
  constexpr double first_dt = 1.0e-6;
  constexpr double fresh_dt = 7.5e-7;
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      fixed_policy{1.0e-8, 1.0e-5, 0.25};
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      invalid_policy{0.0, 0.0, 0.25};
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::igr);
  request.time_treatment = EulerHdgTimeTreatment::rodas3;

  GenericEulerHdgCudaSession session;
  const bool initialized = session.Initialize(request).status ==
      HdgEulerSessionStatus::initialized;
  const bool capability = dg::hdg::generic::HasHdgEulerSessionCapabilities(
      session.capabilities(),
      dg::hdg::generic::HdgEulerSessionCapabilityBit(
          dg::hdg::generic::HdgEulerSessionCapability::
              prepared_embedded_assessment));
  if (!All(initialized && capability, request.communicator)) return false;

  const auto initial_coordinates = session.coordinates();
  GenericEulerCudaPreparedStep first_authority;
  GenericEulerHdgCudaSession::prepared_output_type first_output{};
  const bool first_prepared = session.Prepare(
      first_dt, first_authority, first_output).status ==
          HdgEulerSessionStatus::prepared;
  const bool first_inactive =
      CudaQualificationTestAccess::FacadePreparedAssessmentInactive(session);
  const bool first_acknowledged = first_prepared &&
      session.AcknowledgePreparedOutput(first_authority, first_output);
  const auto missing_assessment = first_acknowledged
      ? session.Commit(std::move(first_authority), first_output)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const bool missing_rejected = missing_assessment.status ==
          HdgEulerSessionStatus::rejected &&
      session.coordinates().time == initial_coordinates.time &&
      session.coordinates().step == initial_coordinates.step &&
      session.coordinates().generation == initial_coordinates.generation;

  GenericEulerCudaPreparedStep empty_authority;
  dg::hdg::generic::HdgEulerPreparedAssessment mismatch_assessment{};
  const bool mismatched_token_rejected = !session.AssessPrepared(
      empty_authority, first_output, fixed_policy, mismatch_assessment);
  dg::hdg::generic::HdgEulerPreparedAssessment invalid_assessment{};
  const bool invalid_policy_rejected = !session.AssessPrepared(
      first_authority, first_output, invalid_policy, invalid_assessment);
  dg::hdg::generic::HdgEulerPreparedAssessment first_assessment{};
  const bool first_assessed = session.AssessPrepared(
      first_authority, first_output, fixed_policy, first_assessment);
  const bool first_gates = first_assessed &&
      dg::hdg::generic::IsValidHdgEulerPreparedAssessment(
          first_assessment) &&
      first_assessment.finite_gate && first_assessment.solver_gate &&
      first_assessment.residual_gate &&
      first_assessment.admissibility_gate &&
      first_assessment.algebraic_closure_gate &&
      first_assessment.algebraic_correction_gate;
  const auto declined = first_gates
      ? session.RejectPrepared(
            std::move(first_authority), first_output)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const bool decline_exact = declined.status ==
          HdgEulerSessionStatus::aborted &&
      session.coordinates().time == initial_coordinates.time &&
      session.coordinates().step == initial_coordinates.step &&
      session.coordinates().generation == initial_coordinates.generation &&
      !first_output.candidate_frozen &&
      !first_output.checkpoint_staged && !first_output.output_staged &&
      CudaQualificationTestAccess::FacadePreparedAssessmentInactive(session);
  if (!All(first_prepared && first_inactive && first_acknowledged &&
               missing_rejected && mismatched_token_rejected &&
               invalid_policy_rejected && first_gates && decline_exact,
           request.communicator))
    return false;

  GenericEulerCudaPreparedStep fresh_authority;
  GenericEulerHdgCudaSession::prepared_output_type fresh_output{};
  const bool fresh_prepared = session.Prepare(
      fresh_dt, fresh_authority, fresh_output).status ==
          HdgEulerSessionStatus::prepared;
  dg::hdg::generic::HdgEulerPreparedAssessment fresh_assessment{};
  const bool fresh_assessed = fresh_prepared && session.AssessPrepared(
      fresh_authority, fresh_output, fixed_policy, fresh_assessment) &&
      fresh_assessment.algebraic_correction_gate;
  const bool fresh_acknowledged = fresh_assessed &&
      session.AcknowledgePreparedOutput(fresh_authority, fresh_output);
  const bool policy_corrupted = fresh_acknowledged &&
      CudaQualificationTestAccess::CorruptFacadePreparedAssessmentPolicy(
          session);
  const auto mismatched_policy = policy_corrupted
      ? session.Commit(std::move(fresh_authority), fresh_output)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const bool mismatched_policy_rejected = mismatched_policy.status ==
      HdgEulerSessionStatus::rejected;
  dg::hdg::generic::HdgEulerPreparedAssessment reassessment{};
  const bool reassessed = mismatched_policy_rejected &&
      session.AssessPrepared(
          fresh_authority, fresh_output, fixed_policy, reassessment) &&
      reassessment.algebraic_correction_gate;
  const auto committed = reassessed
      ? session.Commit(std::move(fresh_authority), fresh_output)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const bool committed_exact = committed.status ==
          HdgEulerSessionStatus::committed &&
      committed.coordinates.time == initial_coordinates.time + fresh_dt &&
      committed.coordinates.step == initial_coordinates.step + 1 &&
      committed.coordinates.generation ==
          initial_coordinates.generation + 1 &&
      fresh_output.successful_commit &&
      CudaQualificationTestAccess::FacadePreparedAssessmentInactive(session);
  return All(fresh_prepared && fresh_assessed && fresh_acknowledged &&
                 policy_corrupted && mismatched_policy_rejected &&
                 reassessed && committed_exact,
             request.communicator);
}

bool CheckRodas3DefaultTraceArchitecture()
{
  constexpr int order = 1;
  constexpr double dt = 1.0e-6;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty, InitialMode::smooth_periodic);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  request.backend.cuda_engine = CudaEngineKind::tensor_core;
  request.time_treatment = EulerHdgTimeTreatment::rodas3;

  GenericEulerHdgCudaSession session;
  if (!All(session.Initialize(request).status ==
               HdgEulerSessionStatus::initialized,
           request.communicator))
    return false;

  const auto initial = session.coordinates();
  GenericEulerCudaPreparedStep authority;
  GenericEulerHdgCudaSession::prepared_output_type output{};
  const bool prepared = session.Prepare(dt, authority, output).status ==
      HdgEulerSessionStatus::prepared;
  const bool exact_native_route = prepared &&
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(
          session, CudaEngineKind::tensor_core);
  const auto rejected = exact_native_route
      ? session.RejectPrepared(std::move(authority), output)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const auto final = session.coordinates();
  return All(prepared && exact_native_route &&
                 rejected.status == HdgEulerSessionStatus::aborted &&
                 final.time == initial.time && final.step == initial.step &&
                 final.generation == initial.generation,
             request.communicator);
}

bool CheckRodas3EntryProjectionGate()
{
  constexpr int order = 1;
  constexpr double dt = 1.0e-6;
  mfem::ParMesh mesh = MakePeriodicMesh();
  const BorrowedFields coherent = MakeBorrowedFields(
      mesh, order, SourceMode::empty, InitialMode::smooth_periodic);
  const auto run = [&](const double trace_scale,
                       const bool expect_prepared)
  {
    BorrowedFields fields = coherent;
    for (double& value : fields.initial_true_trace) value *= trace_scale;
    BoundaryContext boundary{};
    EulerHdgRequest request = MakeRequest(
        mesh, order, fields, boundary, false, RegularizationKind::igr);
    request.time_treatment = EulerHdgTimeTreatment::rodas3;
    CudaConfig config = MakeDirectConfig(request, 1, order);
    config.dt = dt;
    config.cuda_engine = CudaEngineKind::tensor_core;
    using Engine = PersistentCudaSmoothEulerEngine<
        dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>;
    const CudaTracePolicyKind requested_policy =
        dg::hdg::runtime::cuda_w8a::DefaultCudaTracePolicyRequest(
            CudaEngineKind::tensor_core, true, 1,
            EulerHdgTimeTreatment::rodas3);
    auto engine = CudaQualificationTestAccess::CreateWithTracePolicy<
        dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>(
            config, requested_policy);
    CudaRunEvidence initialization{};
    if (!engine || !engine->Initialize(initialization)) return false;

    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    const bool prepared = engine->PrepareStep(false, dt, snapshot, evidence);
    const bool projected = evidence.nonlinear_updates > 0 &&
        std::isfinite(evidence.initial_residual) &&
        evidence.initial_residual >
            request.numerical_policy.nonlinear_relative_tolerance;
    if (!expect_prepared)
    {
      return !prepared && projected &&
          evidence.phase == CudaFailurePhase::recovery &&
          evidence.contract_reason == FailureReason::recovery_gate &&
          evidence.common_shift_operator_builds == 0 &&
          evidence.before.time == evidence.after.time &&
          evidence.before.step == evidence.after.step &&
          evidence.before.generation == evidence.after.generation &&
          evidence.coordinates_preserved &&
          evidence.committed_bytes_preserved;
    }

    CudaPreparedOutputMetadataV1 output{};
    CudaStepEvidence staged_evidence{};
    const bool staged = prepared &&
        engine->StagePreparedOutputMetadata(output, staged_evidence);
    CudaStepEvidence rollback{};
    const bool rolled_back = staged && engine->AbortPrepared(
        output, FailureReason::backend_failure, rollback);
    return projected && rolled_back && rollback.coordinates_preserved &&
        rollback.committed_bytes_preserved;
  };

  // The first state is intentionally inconsistent but remains within the
  // CPU/CUDA shared 0.25 normalized entry-correction allowance. The second
  // reaches the same constraint surface only by replacing the supplied trace
  // wholesale and must be rejected transactionally before causal stages.
  return All(run(1.05, true) && run(64.0, false), mesh.GetComm());
}

bool CheckRodas3ZhangShuActiveStageLifecycle()
{
  constexpr int order = 1;
  constexpr int scalar_dofs = 4;
  constexpr double dt = 1.0e-6;
  constexpr double density_floor = 0.999;
  constexpr double pressure_floor = 1.039;
  constexpr double source_amplitude = 2500.0;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::explicit_zero, InitialMode::constant);
  if (fields.initial.empty() || fields.source.size() != fields.initial.size())
    return false;

  // For Q1 Gauss--Legendre L2, the four nodal weights are equal.  These
  // density-and-energy checkerboards therefore have exactly zero cell mean
  // while stage 2, Y_3 = y_n + k_1, crosses both consumer floors.  The
  // pressure pass contracts the density-limited polynomial strictly away
  // from its corner density root, so coefficient-space vertex interpolation
  // remains independently admissible rather than relying on root rounding.
  // Stages 0 and 1 remain the admissible coherent entry state, so activation
  // specifically reaches the causal-stage projection/rebuild path.
  bool source_ready = true;
  try
  {
    mfem::L2_FECollection collection(
        order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::ParFiniteElementSpace space(
        &mesh, &collection, components, mfem::Ordering::byNODES);
    for (int element = 0; element < mesh.GetNE() && source_ready; ++element)
    {
      const mfem::FiniteElement* const finite_element = space.GetFE(element);
      mfem::Array<int> vdofs;
      space.GetElementVDofs(element, vdofs);
      source_ready = finite_element != nullptr &&
          finite_element->GetDof() == scalar_dofs &&
          vdofs.Size() == components * scalar_dofs;
      for (int scalar = 0; scalar < scalar_dofs && source_ready; ++scalar)
      {
        double density_sign = 1.0;
        double energy_sign = 1.0;
        const int density_dof = DecodeSignedDof(
            vdofs[scalar], density_sign);
        const int energy_dof = DecodeSignedDof(
            vdofs[3 * scalar_dofs + scalar], energy_sign);
        const double local_value = scalar % 2 == 0
            ? -source_amplitude : source_amplitude;
        source_ready = density_dof >= 0 && energy_dof >= 0 &&
            static_cast<std::size_t>(density_dof) < fields.source.size() &&
            static_cast<std::size_t>(energy_dof) < fields.source.size();
        if (source_ready)
        {
          fields.source[static_cast<std::size_t>(density_dof)] =
              density_sign * local_value;
          fields.source[static_cast<std::size_t>(energy_dof)] =
              energy_sign * local_value;
        }
      }
    }
  }
  catch (...) { source_ready = false; }
  if (!All(source_ready, mesh.GetComm())) return false;

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  request.backend.cuda_engine = CudaEngineKind::tensor_core;
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  request.numerical_policy.positivity_limiter = {
      dg::hdg::generic::EulerHdgPositivityLimiterKind::zhang_shu,
      dg::hdg::generic::EulerHdgZhangShuPointSet::consumer,
      density_floor,
      pressure_floor,
      12};
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  CudaConfig config = MakeDirectConfig(request, 1, order);
  config.dt = dt;
  config.cuda_engine = CudaEngineKind::tensor_core;
  const CudaTracePolicyKind requested_policy =
      dg::hdg::runtime::cuda_w8a::DefaultCudaTracePolicyRequest(
          CudaEngineKind::tensor_core, false, 1,
          EulerHdgTimeTreatment::rodas3);
  auto engine = CudaQualificationTestAccess::CreateWithTracePolicy<
      dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>(
          config, requested_policy);
  CudaRunEvidence initialization{};
  if (!engine || !All(engine->Initialize(initialization), mesh.GetComm()))
    return false;

  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  CudaRodas3ScreenObservation observation{};
  CudaPreparedOutputMetadataV1 output{};
  dg::hdg::generic::HdgEulerPreparedAssessment measured_assessment{};
  dg::hdg::generic::HdgEulerPreparedAssessment strict_assessment{};
  dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      measurement_policy{1.0e-8, 1.0e-5, 1.0};
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      acceptance_policy{1.0e-8, 1.0e-5, 0.25};
  CudaRodas3LimiterEndpointAudit endpoint_audit{};
  const bool prepared = engine->PrepareStep(false, dt, snapshot, evidence);
  const CudaStepEvidence prepared_evidence = evidence;
  const bool captured = prepared &&
      CudaQualificationTestAccess::CapturePreparedRodas3(
          *engine, observation);
  // Before metadata staging repacks output_interior, verify that the selected
  // stage-consistent candidate is exactly admitted Y4 + gamma*q4 in U4.
  const bool endpoint_audited = captured &&
      CudaQualificationTestAccess::AuditPreparedRodas3LimiterEndpoint(
          *engine, endpoint_audit);
  const bool staged = endpoint_audited &&
      engine->StagePreparedOutputMetadata(output, evidence);
  // Measure the real post-limiter algebraic reconstruction.  When it is
  // nonzero, set a valid bound strictly below it and prove rejection without
  // fabricating a correction in private device storage.  Exact zero is also
  // a legitimate fully coherent endpoint and must remain admissible.
  const bool measured = staged && engine->AssessPrepared(
      output, measurement_policy, measured_assessment);
  const double measured_correction =
      measured_assessment.normalized_algebraic_correction;
  const bool nonzero_correction = measured_correction > 0.0;
  const double strict_fraction = nonzero_correction
      ? 0.5 * std::min(measured_correction, 1.0) : 0.0;
  const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      strict_assessment_policy{1.0e-8, 1.0e-5, strict_fraction};
  const bool strict_policy_ready = measured &&
      std::isfinite(measured_correction) && measured_correction >= 0.0 &&
      (!nonzero_correction ||
       (strict_fraction > 0.0 && strict_fraction < measured_correction &&
        dg::hdg::generic::IsValidHdgEulerPreparedAssessmentPolicy(
            strict_assessment_policy)));
  const bool strict_assessed = strict_policy_ready &&
      (!nonzero_correction || engine->AssessPrepared(
          output, strict_assessment_policy, strict_assessment));
  const bool strict_primary_gates = strict_assessed &&
      (nonzero_correction
          ? strict_assessment.finite_gate &&
                strict_assessment.solver_gate &&
                strict_assessment.residual_gate &&
                strict_assessment.admissibility_gate &&
                strict_assessment.algebraic_closure_gate &&
                !strict_assessment.algebraic_correction_gate &&
                strict_assessment.normalized_algebraic_correction >
                    strict_assessment_policy.algebraic_correction_fraction
          : measured_assessment.finite_gate &&
                measured_assessment.solver_gate &&
                measured_assessment.residual_gate &&
                measured_assessment.admissibility_gate &&
                measured_assessment.algebraic_closure_gate &&
                measured_assessment.algebraic_correction_gate &&
                measured_correction == 0.0) &&
      !output.output_staged && !output.successful_commit;
  const bool assessed = strict_primary_gates &&
      engine->AssessPrepared(
          output, acceptance_policy, assessment);
  const bool marked = assessed && assessment.algebraic_correction_gate &&
      engine->MarkOutputStaged(output);
  const bool commit_prepared = marked &&
      engine->PrepareCommit(output, evidence);
  const bool provisional = commit_prepared &&
      engine->CommitPrepared(output, evidence);
  const bool committed = provisional &&
      engine->FinalizeCommit(output, evidence);

  std::vector<double> committed_u4(fields.initial.size(), 0.0);
  std::vector<double> theta_rho(
      static_cast<std::size_t>(mesh.GetNE()), 0.0);
  std::vector<double> theta_p(
      static_cast<std::size_t>(mesh.GetNE()), 0.0);
  const bool exported = committed && engine->ExportCommittedFields(
      dg::hdg::generic::HdgEulerHostInteriorFields{
          committed_u4.data(), nullptr, fields.volume_scalar_count,
          theta_rho.data(), theta_p.data(),
          static_cast<std::size_t>(mesh.GetNE())});
  const bool limiter_diagnostics_active = exported &&
      std::all_of(theta_rho.begin(), theta_rho.end(),
                  [](const double value)
                  { return std::isfinite(value) && value >= 0.0 && value <= 1.0; }) &&
      std::all_of(theta_p.begin(), theta_p.end(),
                  [](const double value)
                  { return std::isfinite(value) && value >= 0.0 && value <= 1.0; }) &&
      (std::any_of(theta_rho.begin(), theta_rho.end(),
                   [](const double value) { return value < 1.0; }) ||
       std::any_of(theta_p.begin(), theta_p.end(),
                   [](const double value) { return value < 1.0; }));
  const bool diagnostic_history_seeded = limiter_diagnostics_active &&
      CudaQualificationTestAccess::
          VerifyLimiterDiagnosticAcceptedHistorySeed(
              *engine, theta_rho, theta_p);
  bool consumer_coefficients_admissible = exported;
  for (std::size_t scalar = 0;
       scalar < fields.volume_scalar_count &&
           consumer_coefficients_admissible;
       ++scalar)
  {
    const double density = committed_u4[scalar];
    const double momentum_x =
        committed_u4[fields.volume_scalar_count + scalar];
    const double momentum_y =
        committed_u4[2 * fields.volume_scalar_count + scalar];
    const double energy =
        committed_u4[3 * fields.volume_scalar_count + scalar];
    const double pressure = 0.4 *
        (energy - 0.5 *
            (momentum_x * momentum_x + momentum_y * momentum_y) / density);
    consumer_coefficients_admissible = std::isfinite(density) &&
        density >= density_floor && std::isfinite(pressure) &&
        pressure >= pressure_floor;
  }
  const bool passed = prepared && captured && endpoint_audited && staged &&
      measured && strict_policy_ready && strict_assessed &&
      strict_primary_gates && assessed && marked && commit_prepared &&
      provisional && committed && endpoint_audit.candidate_formula_passed &&
      endpoint_audit.candidate_formula_nontrivial &&
      snapshot.numerical_candidate_ready && observation.candidate_ready &&
      observation.finite && observation.admissible && observation.recovery &&
      observation.true_residual && observation.conservation &&
      observation.telemetry && observation.success &&
      assessment.finite_gate && assessment.solver_gate &&
      assessment.residual_gate && assessment.admissibility_gate &&
      assessment.algebraic_closure_gate &&
      assessment.algebraic_correction_gate &&
      evidence.status == CudaRunStatus::committed && evidence.committed &&
      evidence.positivity_limiter_calls ==
          prepared_evidence.positivity_limiter_calls &&
      evidence.positivity_limiter_active_calls ==
          prepared_evidence.positivity_limiter_active_calls &&
      evidence.positivity_limiter_inadmissible_mean_rejections ==
          prepared_evidence.positivity_limiter_inadmissible_mean_rejections &&
      evidence.positivity_algebraic_projections ==
          prepared_evidence.positivity_algebraic_projections &&
      evidence.common_shift_operator_rebuilds ==
          prepared_evidence.common_shift_operator_rebuilds &&
      evidence.maximum_positivity_limiter_correction_norm ==
          prepared_evidence.maximum_positivity_limiter_correction_norm &&
      prepared_evidence.no_fallback && prepared_evidence.device_resident &&
      prepared_evidence.positivity_limiter_calls == 6 &&
      // Only causal stages 2 and 3 are eligible for reconstruction.  Two
      // rebuilds therefore prove that the final-stage branch was exercised.
      prepared_evidence.positivity_limiter_active_calls == 2 &&
      prepared_evidence.positivity_limiter_inadmissible_mean_rejections == 0 &&
      prepared_evidence.positivity_algebraic_projections ==
          prepared_evidence.positivity_limiter_active_calls &&
      prepared_evidence.common_shift_operator_rebuilds == 2 &&
      prepared_evidence.common_shift_operator_builds ==
          1 + prepared_evidence.common_shift_operator_rebuilds &&
      prepared_evidence.common_shift_operator_reuses == 3 &&
      prepared_evidence.condensed_rhs_refreshes == 3 &&
      prepared_evidence.maximum_positivity_limiter_correction_norm > 0.0 &&
      std::isfinite(prepared_evidence.temporal_error_norm) &&
      std::isfinite(prepared_evidence.final_algebraic_residual) &&
      std::isfinite(prepared_evidence.final_algebraic_correction_norm) &&
      consumer_coefficients_admissible && limiter_diagnostics_active &&
      diagnostic_history_seeded &&
      output.successful_commit &&
      evidence.after.step == 1 && evidence.after.generation == 1 &&
      evidence.after.time == dt;
  if (!passed)
  {
    int rank = -1;
    MPI_Comm_rank(mesh.GetComm(), &rank);
    if (rank == 0)
    {
      std::cerr << "CUDA Zhang--Shu active-stage lifecycle failed: prepared="
                << prepared << " captured=" << captured
                << " staged=" << staged
                << " strict_assessed=" << strict_assessed
                << " strict_primary=" << strict_primary_gates
                << " reassessed=" << assessed
                << " marked=" << marked
                << " commit_prepared=" << commit_prepared
                << " provisional=" << provisional
                << " committed=" << committed << " status="
                << static_cast<int>(evidence.status) << " phase="
                << static_cast<int>(prepared_evidence.phase) << " reason="
                << static_cast<int>(prepared_evidence.contract_reason)
                << " calls="
                << prepared_evidence.positivity_limiter_calls << " active="
                << prepared_evidence.positivity_limiter_active_calls
                << " mean_rejections="
                << prepared_evidence.
                    positivity_limiter_inadmissible_mean_rejections
                << " projections="
                << prepared_evidence.positivity_algebraic_projections
                << " rebuilds="
                << prepared_evidence.common_shift_operator_rebuilds
                << " builds="
                << prepared_evidence.common_shift_operator_builds
                << " reuses="
                << prepared_evidence.common_shift_operator_reuses
                << " correction="
                << prepared_evidence.maximum_positivity_limiter_correction_norm
                << " conservation=" << observation.conservation
                << " endpoint_formula="
                << endpoint_audit.candidate_formula_passed
                << " endpoint_formula_error="
                << endpoint_audit.maximum_candidate_formula_scaled_error
                << " diagnostic_history_seeded="
                << diagnostic_history_seeded
                << " measured_correction=" << measured_correction
                << " strict_fraction=" << strict_fraction
                << " strict_policy_ready=" << strict_policy_ready
                << " strict_differential="
                << strict_assessment.normalized_differential_error
                << " strict_algebraic="
                << strict_assessment.normalized_algebraic_correction
                << " strict_algebraic_gate="
                << strict_assessment.algebraic_correction_gate
                << " normalized_differential="
                << assessment.normalized_differential_error
                << " normalized_algebraic="
                << assessment.normalized_algebraic_correction
                << " algebraic_gate="
                << assessment.algebraic_correction_gate
                << '\n';
    }
  }
  return All(passed, mesh.GetComm());
}

bool CheckRodas3ZhangShuMeanRejectionAttribution()
{
  constexpr int order = 1;
  constexpr double dt = 1.0e-6;
  mfem::ParMesh mesh = MakePeriodicMesh();
  BorrowedFields fields = MakeBorrowedFields(
      mesh, order, SourceMode::empty, InitialMode::constant);
  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, order, fields, boundary, false, RegularizationKind::none);
  request.backend.cuda_engine = CudaEngineKind::tensor_core;
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  // The constant state has rho_bar=1.  A floor strictly above that mean is a
  // deterministic no-repair limiter rejection, while the Euler state itself
  // remains finite with positive physical density and pressure.
  request.numerical_policy.positivity_limiter = {
      dg::hdg::generic::EulerHdgPositivityLimiterKind::zhang_shu,
      dg::hdg::generic::EulerHdgZhangShuPointSet::consumer,
      1.001,
      1.0e-8,
      12};
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
      ContractStatus::accepted)
    return false;

  CudaConfig config = MakeDirectConfig(request, 1, order);
  config.dt = dt;
  config.cuda_engine = CudaEngineKind::tensor_core;
  const CudaTracePolicyKind requested_policy =
      dg::hdg::runtime::cuda_w8a::DefaultCudaTracePolicyRequest(
          CudaEngineKind::tensor_core, false, 1,
          EulerHdgTimeTreatment::rodas3);
  auto engine = CudaQualificationTestAccess::CreateWithTracePolicy<
      dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>(
          config, requested_policy);
  CudaRunEvidence initialization{};
  if (!engine || !All(engine->Initialize(initialization), mesh.GetComm()))
    return false;

  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  const bool prepared = engine->PrepareStep(false, dt, snapshot, evidence);
  const CudaStepEvidence failed_evidence = evidence;
  CudaStepEvidence rollback{};
  const bool abandoned = !prepared && engine->RejectFailedAttempt(rollback);
  std::vector<double> committed_u4(fields.initial.size(), 0.0);
  std::vector<double> theta_rho(
      static_cast<std::size_t>(mesh.GetNE()), 0.0);
  std::vector<double> theta_p(
      static_cast<std::size_t>(mesh.GetNE()), 0.0);
  const bool diagnostics_rolled_back = abandoned &&
      engine->ExportCommittedFields(
          dg::hdg::generic::HdgEulerHostInteriorFields{
              committed_u4.data(), nullptr, fields.volume_scalar_count,
              theta_rho.data(), theta_p.data(),
              static_cast<std::size_t>(mesh.GetNE())}) &&
      std::all_of(theta_rho.begin(), theta_rho.end(),
                  [](const double value) { return value == 1.0; }) &&
      std::all_of(theta_p.begin(), theta_p.end(),
                  [](const double value) { return value == 1.0; });
  const bool passed = abandoned &&
      failed_evidence.status == CudaRunStatus::retry_available &&
      failed_evidence.phase == CudaFailurePhase::assembly &&
      failed_evidence.contract_reason == FailureReason::admissibility_gate &&
      failed_evidence.positivity_limiter_calls == 1 &&
      failed_evidence.positivity_limiter_active_calls == 0 &&
      failed_evidence.positivity_limiter_inadmissible_mean_rejections == 1 &&
      failed_evidence.positivity_algebraic_projections == 0 &&
      failed_evidence.common_shift_operator_builds == 0 &&
      failed_evidence.common_shift_trace_solves == 0 &&
      failed_evidence.retry_available &&
      failed_evidence.coordinates_preserved &&
      failed_evidence.committed_bytes_preserved &&
      rollback.coordinates_preserved && rollback.committed_bytes_preserved &&
      !rollback.retry_available && diagnostics_rolled_back;
  if (!passed)
  {
    int rank = -1;
    MPI_Comm_rank(mesh.GetComm(), &rank);
    if (rank == 0)
      std::cerr << "CUDA Zhang--Shu mean-rejection attribution failed: "
                << "prepared=" << prepared << " abandoned=" << abandoned
                << " status=" << static_cast<int>(failed_evidence.status)
                << " phase=" << static_cast<int>(failed_evidence.phase)
                << " reason="
                << static_cast<int>(failed_evidence.contract_reason)
                << " calls="
                << failed_evidence.positivity_limiter_calls
                << " active="
                << failed_evidence.positivity_limiter_active_calls
                << " mean_rejections="
                << failed_evidence.
                    positivity_limiter_inadmissible_mean_rejections
                << " retry=" << failed_evidence.retry_available
                << '\n';
  }
  return All(passed, mesh.GetComm());
}

bool CheckP3ZhangShuConsumerVertices(const CudaEngineKind engine_kind)
{
  mfem::ParMesh mesh = MakePhysicalMesh(1);
  BorrowedFields fields = MakeBorrowedFields(
      mesh, hidden_corner_order, SourceMode::explicit_zero,
      InitialMode::hidden_corner_pressure);
  HiddenCornerFieldAudit input{};
  const bool input_audited = !fields.initial.empty() &&
      fields.volume_scalar_count == hidden_corner_scalar_dofs &&
      AuditHiddenCornerField(mesh, fields.initial, input);
  double mean_pressure = 0.0;
  const bool hidden_corner_witness = input_audited &&
      input.selected_points_admissible && !input.vertices_admissible &&
      input.minimum_selected_pressure > hidden_corner_pressure_floor &&
      input.minimum_vertex_pressure < hidden_corner_pressure_floor &&
      dg::hdg::euler::zhang_shu::Pressure(
          input.mean.data(), 1.4, mean_pressure) &&
      mean_pressure > hidden_corner_pressure_floor;

  BoundaryContext boundary{};
  EulerHdgRequest request = MakeRequest(
      mesh, hidden_corner_order, fields, boundary, true,
      RegularizationKind::none);
  request.backend.cuda_engine = engine_kind;
  request.time_treatment = EulerHdgTimeTreatment::rodas3;
  request.numerical_policy.conventional_assembly_quadrature_order =
      hidden_corner_quadrature_order;
  request.numerical_policy.positivity_limiter = {
      dg::hdg::generic::EulerHdgPositivityLimiterKind::zhang_shu,
      dg::hdg::generic::EulerHdgZhangShuPointSet::consumer,
      hidden_corner_density_floor,
      hidden_corner_pressure_floor,
      hidden_corner_pressure_bisections};
  const bool request_valid = hidden_corner_witness &&
      dg::hdg::generic::ValidateEulerHdgRequest(request) ==
          ContractStatus::accepted;

  CudaConfig config = MakeDirectConfig(request, 1, hidden_corner_order);
  config.dt = 1.0e-8;
  config.cuda_engine = engine_kind;
  const std::array<double, components> reference{{
      1.0, 0.05, -0.02, 2.60}};
  const std::array<double, components> unused{};
  const std::array<CudaStaticBoundaryPolicy, 4> device_policies{{
      {1, CudaStaticBoundaryPolicyKind::slip_wall, unused, 0.0},
      {2, CudaStaticBoundaryPolicyKind::supersonic_outlet, unused, 0.0},
      {3, CudaStaticBoundaryPolicyKind::far_field, reference, 0.0},
      {4, CudaStaticBoundaryPolicyKind::supersonic_inlet, reference, 0.0}}};
  config.borrowed_global_topology = StableIdentity::Copy(
      "generic.euler.request.topology.v1");
  config.borrowed_boundary_policies = {
      device_policies.data(), device_policies.size(), 1.4};
  const CudaTracePolicyKind requested_policy =
      dg::hdg::runtime::cuda_w8a::DefaultCudaTracePolicyRequest(
          engine_kind, false, 1, EulerHdgTimeTreatment::rodas3);
  using Engine = PersistentCudaSmoothEulerEngine<
      dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>;
  std::unique_ptr<Engine> engine{};
  if (request_valid)
  {
    engine = CudaQualificationTestAccess::CreateWithTracePolicy<
        dg::hdg::runtime::Rodas3CudaRank1RuntimeSelection>(
        config, requested_policy);
  }
  CudaRunEvidence initialization{};
  const bool initialized = engine && engine->Initialize(initialization);
  CudaDirectZhangShuAudit device_audit{};
  const bool limited = initialized &&
      CudaQualificationTestAccess::ApplyDirectZhangShu(
          *engine, fields.initial, device_audit);
  HiddenCornerFieldAudit admitted{};
  const bool output_audited = limited && AuditHiddenCornerField(
      mesh, device_audit.limited_u4, admitted);

  bool mean_preserved = output_audited;
  for (int component = 0; component < components && mean_preserved;
       ++component)
  {
    const double scale = std::max(
        {1.0, std::abs(input.mean[component]),
         std::abs(admitted.mean[component])});
    mean_preserved = std::abs(
        input.mean[component] - admitted.mean[component]) <=
        2.0e-13 * scale;
  }

  const bool theta_ready = device_audit.theta_rho.size() == 1 &&
      device_audit.theta_p.size() == 1 &&
      device_audit.theta_rho[0] == 1.0 &&
      std::isfinite(device_audit.theta_p[0]) &&
      device_audit.theta_p[0] >= 0.0 &&
      device_audit.theta_p[0] < 1.0;
  const double exact_pressure_theta = theta_ready
      ? (mean_pressure - hidden_corner_pressure_floor) /
          (mean_pressure - input.minimum_vertex_pressure)
      : 0.0;
  const double bisection_width = std::ldexp(
      1.0, -hidden_corner_pressure_bisections);
  const bool conservative_pressure_bracket = theta_ready &&
      device_audit.theta_p[0] <= exact_pressure_theta + 64.0 *
          std::numeric_limits<double>::epsilon() &&
      device_audit.theta_p[0] >=
          exact_pressure_theta - bisection_width - 1.0e-12;

  bool coefficient_formula = theta_ready &&
      device_audit.limited_u4.size() == fields.initial.size();
  for (int component = 0; component < components && coefficient_formula;
       ++component)
  {
    const double theta = component == 0
        ? device_audit.theta_rho[0] * device_audit.theta_p[0]
        : device_audit.theta_p[0];
    for (std::size_t scalar = 0;
         scalar < fields.volume_scalar_count && coefficient_formula; ++scalar)
    {
      const std::size_t index =
          static_cast<std::size_t>(component) *
              fields.volume_scalar_count + scalar;
      const double expected = (1.0 - theta) * input.mean[component] +
          theta * fields.initial[index];
      const double observed = device_audit.limited_u4[index];
      const double scale = std::max(
          {1.0, std::abs(expected), std::abs(observed)});
      coefficient_formula = std::isfinite(observed) &&
          std::abs(observed - expected) <= 512.0 *
              std::numeric_limits<double>::epsilon() * scale;
    }
  }

  const bool passed = request_valid && initialized && limited &&
      device_audit.classification == 1 &&
      device_audit.correction_squared > 0.0 && output_audited &&
      admitted.selected_points_admissible && admitted.vertices_admissible &&
      admitted.minimum_vertex_pressure >= hidden_corner_pressure_floor &&
      mean_preserved && conservative_pressure_bracket &&
      coefficient_formula;
  int rank = -1;
  MPI_Comm_rank(mesh.GetComm(), &rank);
  if (!passed && rank == 0)
  {
    std::cerr << "CUDA Zhang--Shu p3 consumer-vertex failure: engine="
              << (engine_kind == CudaEngineKind::tensor_core
                      ? "tensor" : "ordinary")
              << " witness=" << hidden_corner_witness
              << " request=" << request_valid
              << " initialized=" << initialized
              << " limited=" << limited
              << " classification=" << device_audit.classification
              << " correction2=" << device_audit.correction_squared
              << " input_selected=" << input.selected_points_admissible
              << " input_vertex=" << input.vertices_admissible
              << " input_vertex_p=" << input.minimum_vertex_pressure
              << " output_vertex=" << admitted.vertices_admissible
              << " output_vertex_p=" << admitted.minimum_vertex_pressure
              << " mean_preserved=" << mean_preserved
              << " theta_rho="
              << (device_audit.theta_rho.empty()
                      ? -1.0 : device_audit.theta_rho[0])
              << " theta_p="
              << (device_audit.theta_p.empty()
                      ? -1.0 : device_audit.theta_p[0])
              << " exact_theta=" << exact_pressure_theta
              << " coefficient_formula=" << coefficient_formula << '\n';
  }
  return All(passed, mesh.GetComm());
}

bool CheckRodas3RankOneScreen(const int rank)
{
  Rodas3ScreenRecord conventional{};
  Rodas3ScreenRecord igr{};
  Rodas3ScreenRecord conventional_physical{};
  Rodas3ScreenRecord igr_physical{};
  Rodas3ScreenRecord ordinary_conventional{};
  Rodas3ScreenRecord ordinary_igr{};
  const bool conventional_passed = RunRodas3RankOneStep(
      CudaEngineKind::tensor_core, RegularizationKind::none, conventional);
  const bool igr_passed = RunRodas3RankOneStep(
      CudaEngineKind::tensor_core, RegularizationKind::igr, igr);
  const bool conventional_physical_passed = RunRodas3RankOneStep(
      CudaEngineKind::tensor_core, RegularizationKind::none,
      conventional_physical, true);
  const bool igr_physical_passed = RunRodas3RankOneStep(
      CudaEngineKind::tensor_core, RegularizationKind::igr,
      igr_physical, true);
  const bool ordinary_conventional_passed = RunRodas3RankOneStep(
      CudaEngineKind::ordinary, RegularizationKind::none,
      ordinary_conventional);
  const bool ordinary_igr_passed = RunRodas3RankOneStep(
      CudaEngineKind::ordinary, RegularizationKind::igr, ordinary_igr);
  const bool assessment_authority =
      CheckRodas3PreparedAssessmentAuthority();
  const bool default_trace_architecture =
      CheckRodas3DefaultTraceArchitecture();
  const bool entry_projection_gate = CheckRodas3EntryProjectionGate();
  const bool positivity_lifecycle =
      CheckRodas3ZhangShuActiveStageLifecycle();
  const bool positivity_mean_rejection =
      CheckRodas3ZhangShuMeanRejectionAttribution();
  const bool all_seven_boundary_policies =
      CheckAllSevenLoweredBoundaryPolicies(1);
  if (rank == 0)
  {
    const auto print = [](const char* mode, const Rodas3ScreenRecord& value)
    {
      const auto& evidence = value.evidence;
      std::cout << "[cuda-rodas3-screen] mode=" << mode
                << " setup_seconds=" << value.setup_seconds
                << " step_seconds=" << value.step_seconds
                << " common_shift_builds="
                << evidence.common_shift_operator_builds
                << " common_shift_reuses="
                << evidence.common_shift_operator_reuses
                << " rhs_refreshes=" << evidence.condensed_rhs_refreshes
                << " trace_solves=" << evidence.common_shift_trace_solves
                << " preconditioner_setups="
                << evidence.common_shift_preconditioner_setups
                << " preconditioner_reuses="
                << evidence.common_shift_preconditioner_reuses
                << " assemblies=" << evidence.complete_assemblies
                << " condensations=" << evidence.complete_condensations
                << " recoveries=" << evidence.complete_recoveries
                << " linear_iterations=" << evidence.linear_iterations
                << " temporal_error=" << evidence.temporal_error_norm
                << " normalized_temporal_error="
                << value.assessment.normalized_differential_error
                << " algebraic_residual="
                << evidence.final_algebraic_residual
                << " algebraic_correction="
                << evidence.final_algebraic_correction_norm
                << " normalized_algebraic_correction="
                << value.assessment.normalized_algebraic_correction
                << " true_residual=" << evidence.true_relative_residual
                << " conservation_relative="
                << value.observation.maximum_conservation_relative_error
                << " state_changed=" << value.state_changed
                << " architecture_passed=" << value.architecture_passed
                << " numerical_passed=" << value.numerical_passed
                << " lifecycle_passed=" << value.lifecycle_passed
                << " exact_jacobian_passed="
                << value.exact_jacobian_passed
                << " initial_closure_audited="
                << value.initial_closure_audited
                << " initial_closure_passed="
                << value.initial_closure_passed;
      if (value.initial_closure_audited)
      {
        std::cout << " initial_u4_exact="
                  << value.initial_closure_audit.euler_interior_exact
                  << " initial_uhat4_exact="
                  << value.initial_closure_audit.euler_true_trace_exact
                  << " initial_auxiliary_norms=";
        for (const double norm :
             value.initial_closure_audit.global_auxiliary_norms)
          std::cout << norm << ',';
        std::cout << " initial_scalar_residual="
                  << value.initial_closure_audit.global_scalar_residual
                  << " initial_scalar_relative_residual="
                  << value.initial_closure_audit.
                      global_scalar_relative_residual
                  << " initial_lifecycle_zero="
                  << value.initial_closure_audit.lifecycle_zero
                  << " initial_combination_alias="
                  << value.initial_closure_audit.combination_alias_exact;
      }
      std::cout << " exact_jacobian_errors=";
      for (const double error : value.exact_jacobian_relative_errors)
        std::cout << error << ',';
      std::cout
                << " candidate_ready="
                << value.observation.candidate_ready
                << " finite=" << value.observation.finite
                << " admissible=" << value.observation.admissible
                << " recovery=" << value.observation.recovery
                << " true_residual_gate="
                << value.observation.true_residual
                << " conservation_gate="
                << value.observation.conservation
                << " telemetry=" << value.observation.telemetry
                << " numerical_success=" << value.observation.success
                << " passed=" << value.passed << '\n';
    };
    print("conventional", conventional);
    print("igr", igr);
    print("conventional-physical-source", conventional_physical);
    print("igr-physical-source", igr_physical);
    print("ordinary-conventional", ordinary_conventional);
    print("ordinary-igr", ordinary_igr);
    std::cout << "[cuda-rodas3-screen] prepared_assessment_authority="
              << assessment_authority
              << " default_trace_architecture="
              << default_trace_architecture
              << " entry_projection_gate=" << entry_projection_gate
              << " positivity_lifecycle=" << positivity_lifecycle
              << " positivity_mean_rejection="
              << positivity_mean_rejection
              << " all_seven_boundary_policies="
              << all_seven_boundary_policies << '\n';
  }
  return conventional_passed && igr_passed &&
      conventional_physical_passed && igr_physical_passed &&
      ordinary_conventional_passed && ordinary_igr_passed &&
      assessment_authority && default_trace_architecture &&
      entry_projection_gate && positivity_lifecycle &&
      positivity_mean_rejection &&
      all_seven_boundary_policies;
}

bool CheckRodas3RankTwoScreen(
    const int rank, const CudaEngineKind engine_kind)
{
  Rodas3ScreenRecord conventional{};
  Rodas3ScreenRecord igr{};
  const bool conventional_passed = RunRodas3RankTwoStep(
      engine_kind, RegularizationKind::none, conventional);
  const bool igr_passed = RunRodas3RankTwoStep(
      engine_kind, RegularizationKind::igr, igr);
  if (rank == 0)
  {
    const char* engine_name = engine_kind == CudaEngineKind::ordinary
        ? "ordinary" : "tensor";
    const auto print = [engine_name](
        const char* mode, const Rodas3ScreenRecord& value)
    {
      const auto& evidence = value.evidence;
      std::cout << "[cuda-rodas3-rank2-screen] engine=" << engine_name
                << " mode=" << mode
                << " setup_seconds=" << value.setup_seconds
                << " step_seconds=" << value.step_seconds
                << " common_shift_builds="
                << evidence.common_shift_operator_builds
                << " common_shift_reuses="
                << evidence.common_shift_operator_reuses
                << " rhs_refreshes=" << evidence.condensed_rhs_refreshes
                << " trace_solves=" << evidence.common_shift_trace_solves
                << " preconditioner_setups="
                << evidence.common_shift_preconditioner_setups
                << " preconditioner_reuses="
                << evidence.common_shift_preconditioner_reuses
                << " ilu_factor_setups="
                << evidence.tensor_core.ilu_factor_setups
                << " ilu_factor_reuses="
                << evidence.tensor_core.ilu_factor_reuses
                << " hypre_trace_calls="
                << evidence.tensor_core.hypre_trace_solve_calls
                << " temporal_error=" << evidence.temporal_error_norm
                << " normalized_temporal_error="
                << value.assessment.normalized_differential_error
                << " algebraic_residual="
                << evidence.final_algebraic_residual
                << " algebraic_correction="
                << evidence.final_algebraic_correction_norm
                << " normalized_algebraic_correction="
                << value.assessment.normalized_algebraic_correction
                << " true_residual=" << evidence.true_relative_residual
                << " conservation_relative="
                << value.observation.maximum_conservation_relative_error
                << " mpi_staging_bytes="
                << evidence.transfers.mpi_staging_bytes
                << " h2d_bytes=" << evidence.transfers.host_to_device_bytes
                << " d2h_bytes=" << evidence.transfers.device_to_host_bytes
                << " host_staged=" << value.host_staged_passed
                << " finite=" << value.observation.finite
                << " admissible=" << value.observation.admissible
                << " final_dae="
                << value.assessment.algebraic_closure_gate
                << " conservation=" << value.observation.conservation
                << " rollback=" << value.rollback_passed
                << " retry=" << value.retry_passed
                << " retry_common_shift="
                << value.retry_evidence.common_shift_operator_builds << '/'
                << value.retry_evidence.common_shift_operator_reuses << '/'
                << value.retry_evidence.condensed_rhs_refreshes << '/'
                << value.retry_evidence.common_shift_trace_solves
                << " retry_hypre_trace_calls="
                << value.retry_evidence.tensor_core.hypre_trace_solve_calls
                << " architecture=" << value.architecture_passed
                << " numerical=" << value.numerical_passed
                << " lifecycle=" << value.lifecycle_passed
                << " passed=" << value.passed << '\n';
    };
    print("conventional-rank2", conventional);
    print("igr-rank2", igr);
  }
  return conventional_passed && igr_passed;
}

// A measurement-only path for arithmetic experiments.  Success means the
// unchanged FP64 reference and candidate both produced finite local blocks;
// satisfying the original FP64 error envelope is reported separately.
bool MeasureAssemblyArithmetic(const int order, const int rank)
{
  if(selected_cuda_engine!=CudaEngineKind::tensor_core)return false;
  constexpr std::array<const char*,8> names{{
      "residual_r","residual_g","A","B","C","D","f","g"}};
  const char* requested=::igrlab::GetEnv("DG_IGR_HDG_VOLUME_ARITHMETIC");
  const char* mode=requested&&*requested?requested:"fp64-wmma";
  bool all_valid=true;
  for(const bool physical : {false,true})
    for(const auto kind : {RegularizationKind::none,RegularizationKind::igr})
    {
      mfem::ParMesh mesh=physical?MakePhysicalMesh():MakePeriodicMesh();
      BorrowedFields fields=MakeBorrowedFields(
          mesh,order,SourceMode::nonzero,InitialMode::smooth_periodic);
      BoundaryContext boundary{};
      EulerHdgRequest request=MakeRequest(
          mesh,order,fields,boundary,physical,kind);
      GenericEulerHdgCudaSession session;
      const auto initialized=session.Initialize(request);
      const bool ready=All(
          initialized.status==HdgEulerSessionStatus::initialized,
          request.communicator);
      CudaBorrowedAssemblyFusionAudit audit{};
      const bool original_gate=ready&&
          CudaQualificationTestAccess::AuditBorrowedAssemblyFusion(session,audit);
      const bool finite=ready&&audit.elements>0&&audit.status_bitwise_equal&&
          audit.status_all_passed&&
          std::all_of(audit.maximum_absolute_errors.begin(),
              audit.maximum_absolute_errors.end(),
              [](const double x){return std::isfinite(x);})&&
          std::all_of(audit.maximum_scaled_errors.begin(),
              audit.maximum_scaled_errors.end(),
              [](const double x){return std::isfinite(x);});
      all_valid=All(finite,request.communicator)&&all_valid;
      if(rank==0)
      {
        std::cout<<std::setprecision(17)
            <<"[assembly-arithmetic] mode="<<mode<<" order="<<order
            <<" boundary="<<(physical?"physical":"periodic")
            <<" regularization="<<(kind==RegularizationKind::igr?"igr":"none")
            <<" initial=smooth_periodic elements="<<audit.elements
            <<" volume_points="<<audit.volume_points
            <<" interior_dofs="<<audit.interior_dofs
            <<" trace_dofs="<<audit.trace_dofs
            <<" fp64_threshold="<<audit.fp64_comparison_tolerance
            <<" passes_fp64_threshold="<<audit.passes_fp64_threshold
            <<" passes_original_fp64_gate="<<original_gate
            <<" candidate_wmma_tiles="<<audit.tensor_volume_mma_tiles
            <<" status_equal="<<audit.status_bitwise_equal
            <<" status_passed="<<audit.status_all_passed
            <<" measurements_valid="<<finite<<'\n';
        for(std::size_t block=0;block<names.size();++block)
          std::cout<<"[assembly-arithmetic-block] mode="<<mode
              <<" boundary="<<(physical?"physical":"periodic")
              <<" regularization="<<(kind==RegularizationKind::igr?"igr":"none")
              <<" block="<<names[block]
              <<" max_absolute="<<audit.maximum_absolute_errors[block]
              <<" max_scaled="<<audit.maximum_scaled_errors[block]
              <<" bitwise_equal="<<audit.bitwise_equal[block]<<'\n';
      }
    }
  return all_valid;
}

struct Options
{
  int order = -1;
  CudaEngineKind engine = CudaEngineKind::ordinary;
  EulerHdgTimeTreatment time_treatment =
      EulerHdgTimeTreatment::backward_euler;
  bool assembly_arithmetic_only = false;
  bool partitioned_smoke = false;
  bool high_order_smoke = false;
  bool rodas3_screen = false;
  bool rodas3_rank2_screen = false;
};

bool ParseOptions(const int argc, char* argv[], Options& options)
{
  for (int index = 1; index < argc; ++index)
  {
    const std::string argument(argv[index]);
    if (argument == "--order")
    {
      if (++index >= argc) return false;
      try { options.order = std::stoi(argv[index]); }
      catch (...) { return false; }
      continue;
    }
    if (argument == "--engine")
    {
      if (++index >= argc) return false;
      const std::string engine(argv[index]);
      if (engine == "ordinary")
        options.engine = CudaEngineKind::ordinary;
      else if (engine == "tensor")
        options.engine = CudaEngineKind::tensor_core;
      else
        return false;
      continue;
    }
    if (argument == "--time-treatment")
    {
      if (++index >= argc) return false;
      const std::string treatment(argv[index]);
      if (treatment == "backward-euler")
        options.time_treatment = EulerHdgTimeTreatment::backward_euler;
      else if (treatment == "partitioned-backward-euler")
        options.time_treatment =
            EulerHdgTimeTreatment::partitioned_backward_euler;
      else if (treatment == "rodas3")
        options.time_treatment = EulerHdgTimeTreatment::rodas3;
      else
        return false;
      continue;
    }
    if (argument == "--assembly-arithmetic-only")
    {
      options.assembly_arithmetic_only = true;
      continue;
    }
    if (argument == "--partitioned-smoke")
    {
      options.partitioned_smoke = true;
      continue;
    }
    if (argument == "--high-order-smoke")
    {
      if (options.high_order_smoke) return false;
      options.high_order_smoke = true;
      continue;
    }
    if (argument == "--rodas3-screen")
    {
      options.rodas3_screen = true;
      continue;
    }
    if (argument == "--rodas3-rank2-screen")
    {
      options.rodas3_rank2_screen = true;
      continue;
    }
    return false;
  }
  const bool partitioned_selected = options.time_treatment ==
      EulerHdgTimeTreatment::partitioned_backward_euler;
  const bool rodas3_selected = options.time_treatment ==
      EulerHdgTimeTreatment::rodas3;
  const bool any_rodas3_screen = options.rodas3_screen ||
      options.rodas3_rank2_screen;
  return dg::hdg::generic::PlanPolynomialShape(options.order, true).valid &&
      options.partitioned_smoke == partitioned_selected &&
      any_rodas3_screen == rodas3_selected &&
      !(options.rodas3_screen && options.rodas3_rank2_screen) &&
      (!options.high_order_smoke ||
       (options.order > 3 && !options.partitioned_smoke &&
        !any_rodas3_screen && !options.assembly_arithmetic_only)) &&
      (!options.assembly_arithmetic_only ||
       (options.engine == CudaEngineKind::tensor_core &&
        !options.partitioned_smoke && !any_rodas3_screen));
}

bool CheckRecoveryDispatchAdmission() noexcept
{
  static_assert(CudaBorrowedRecoveryForEngine(CudaEngineKind::ordinary) ==
                CudaBorrowedRecoveryKind::ordinary_direct_lu);
  static_assert(CudaBorrowedRecoveryForEngine(CudaEngineKind::tensor_core) ==
                CudaBorrowedRecoveryKind::tensor_retained_solve);
  constexpr auto invalid_engine = static_cast<CudaEngineKind>(
      std::numeric_limits<std::uint8_t>::max());
  static_assert(CudaBorrowedRecoveryForEngine(invalid_engine) ==
                CudaBorrowedRecoveryKind::invalid);
  static_assert(CudaRecoveryRouteSupported(CudaEngineKind::ordinary, true));
  static_assert(CudaRecoveryRouteSupported(CudaEngineKind::ordinary, false));
  static_assert(CudaRecoveryRouteSupported(CudaEngineKind::tensor_core, true));
  static_assert(!CudaRecoveryRouteSupported(CudaEngineKind::tensor_core,
                                            false));
  static_assert(!CudaRecoveryRouteSupported(invalid_engine, true));
  static_assert(!CudaRecoveryRouteSupported(invalid_engine, false));
  return true;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  int rank = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Options options;
  const bool options_ready = ParseOptions(argc, argv, options);
  if (!options_ready)
  {
    if (rank == 0)
      std::cerr << "Usage: " << argv[0]
                << " --order P [--engine ordinary|tensor]"
                   " [--time-treatment backward-euler|"
                   "partitioned-backward-euler|rodas3]"
                   " [--high-order-smoke|--partitioned-smoke|--rodas3-screen|"
                   "--rodas3-rank2-screen|--assembly-arithmetic-only]\n";
    return 2;
  }
  const int order = options.order;
  if (options.high_order_smoke && ranks != 1)
  {
    if (rank == 0)
      std::cerr << "--high-order-smoke requires exactly one MPI rank\n";
    return 2;
  }
  selected_cuda_engine = options.engine;
  selected_time_treatment = options.time_treatment;
  const bool admission = options.rodas3_screen ||
      options.rodas3_rank2_screen ||
      (order > 0 && CheckContractAdmission(order));
  mfem::Device device("cuda");
  (void)device;
  if (options.assembly_arithmetic_only)
    return ranks == 1 && admission && MeasureAssemblyArithmetic(order,rank)
        ? EXIT_SUCCESS : EXIT_FAILURE;
  if (options.high_order_smoke)
  {
    std::vector<double> conventional;
    std::vector<double> igr;
    const bool conventional_passed = CheckLifecycle(
        order, true, RegularizationKind::none, SourceMode::nonzero,
        InitialMode::smooth_periodic, conventional, nullptr, 1);
    const bool igr_passed = CheckLifecycle(
        order, false, RegularizationKind::igr, SourceMode::empty,
        InitialMode::smooth_periodic, igr);
    const bool passed = All(ranks == 1 && admission &&
        conventional_passed && igr_passed);
    if (rank == 0)
      std::cout << "HDG high-order CUDA lifecycle smoke order=" << order
                << " engine="
                << (options.engine == CudaEngineKind::tensor_core
                        ? "tensor" : "ordinary")
                << " physical_elements=1 periodic_igr_elements=9"
                << " dt=1e-5 conventional=" << conventional_passed
                << " igr=" << igr_passed << " passed=" << passed << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (options.rodas3_screen)
    return ranks == 1 && CheckRodas3RankOneScreen(rank)
        ? EXIT_SUCCESS : EXIT_FAILURE;
  if (options.rodas3_rank2_screen)
    return ranks == 2 && CheckRodas3RankTwoScreen(rank, options.engine)
        ? EXIT_SUCCESS : EXIT_FAILURE;
  const bool recovery_dispatch_admission = CheckRecoveryDispatchAdmission();
  std::vector<double> conventional_periodic_state;
  std::vector<double> conventional_physical_state;
  std::vector<double> igr_periodic_state;
  std::vector<double> igr_physical_state;
  std::vector<double> conventional_empty_source_state;
  std::vector<double> conventional_zero_source_state;
  std::vector<double> igr_empty_source_state;
  std::vector<double> igr_zero_source_state;
  long double conventional_periodic_response =
      std::numeric_limits<long double>::infinity();
  long double conventional_physical_response =
      std::numeric_limits<long double>::infinity();
  long double igr_periodic_response =
      std::numeric_limits<long double>::infinity();
  long double igr_physical_response =
      std::numeric_limits<long double>::infinity();
  if (options.partitioned_smoke)
  {
    std::vector<double> conventional_periodic_smoke;
    std::vector<double> conventional_physical_smoke;
    std::vector<double> igr_periodic_smoke;
    std::vector<double> igr_physical_smoke;
    const bool conventional_periodic = CheckLifecycle(
        order, false, RegularizationKind::none, SourceMode::nonzero,
        InitialMode::constant, conventional_periodic_smoke);
    const bool conventional_physical = CheckLifecycle(
        order, true, RegularizationKind::none, SourceMode::nonzero,
        InitialMode::constant, conventional_physical_smoke);
    const bool igr_periodic = CheckLifecycle(
        order, false, RegularizationKind::igr, SourceMode::nonzero,
        InitialMode::constant, igr_periodic_smoke);
    const bool igr_physical = CheckLifecycle(
        order, true, RegularizationKind::igr, SourceMode::nonzero,
        InitialMode::constant, igr_physical_smoke);
    std::vector<double> igr_empty_source_smoke;
    std::vector<double> igr_zero_source_smoke;
    const bool bounded_deep_qualification = order != 2 ||
        (CheckLifecycle(
             order, false, RegularizationKind::igr, SourceMode::empty,
             InitialMode::constant, igr_empty_source_smoke) &&
         CheckLifecycle(
             order, false, RegularizationKind::igr, SourceMode::explicit_zero,
             InitialMode::constant, igr_zero_source_smoke) &&
         SameValues(igr_empty_source_smoke, igr_zero_source_smoke) &&
         CheckAllSevenLoweredBoundaryPolicies(order) &&
         CheckMixedLoweredBoundaryPolicies(order) &&
         CheckIndependentIgrConservation(order));
    const bool failure_injection =
        CheckPartitionedFailureInjection(order, ranks);
    const bool checkpoint_restart =
        CheckPartitionedCheckpointRestart(order, ranks);
    const bool passed = (ranks == 1 || ranks == 2) && All(
        admission && recovery_dispatch_admission &&
        conventional_periodic && conventional_physical &&
        igr_periodic && igr_physical && bounded_deep_qualification &&
        failure_injection && checkpoint_restart);
    if (!passed && rank == 0)
      std::cerr << "partitioned Backward-Euler CUDA smoke failed order="
                << order << " ranks=" << ranks << " engine="
                << (selected_cuda_engine == CudaEngineKind::tensor_core
                        ? "tensor" : "ordinary")
                << " admission=" << admission
                << " recovery_dispatch=" << recovery_dispatch_admission
                << " conventional=" << conventional_periodic << '/'
                << conventional_physical << " igr=" << igr_periodic << '/'
                << igr_physical << " deep=" << bounded_deep_qualification
                << " injection=" << failure_injection
                << " checkpoint_restart=" << checkpoint_restart
                << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  const bool copied_initial_trace = order > 0 &&
      CheckCopiedInitialTrueTrace(order);
  const bool all_seven_boundary_policies = order > 0 &&
      CheckAllSevenLoweredBoundaryPolicies(order);
  const bool mixed_boundary_policies = order > 0 &&
      CheckMixedLoweredBoundaryPolicies(order);
  const bool independent_igr_conservation = order > 0 &&
      CheckIndependentIgrConservation(order);
  const bool conventional_conservation_rollback = order > 0 &&
      CheckConventionalConservationGateRollback(order, ranks);
  const bool tensor_diagonal_convergence_warning_rollback = order > 0 &&
      CheckTensorDiagonalConvergenceWarningRollback(order, ranks);
  const bool conventional_p1_q12_tensor_volume_multi_chunk_parity =
      order > 0 &&
      CheckConventionalP1Q12TensorVolumeMultiChunkParity(order);
  const bool euler_algebra = order > 0 && CheckEulerAlgebra(order, ranks);
  const bool zero_update_final_reconstruction = order > 0 &&
      CheckZeroUpdateFinalReconstruction(order);
  const bool rejected_armijo_rollback = order > 0 &&
      CheckRejectedArmijoRollback(order, ranks);
  const bool p3_consumer_vertices = order != hidden_corner_order ||
      ranks != 1 || CheckP3ZhangShuConsumerVertices(options.engine);
  const bool conventional_periodic = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::nonzero,
      InitialMode::constant,
      conventional_periodic_state, &conventional_periodic_response);
  const bool conventional_physical = order > 0 && CheckLifecycle(
      order, true, RegularizationKind::none, SourceMode::nonzero,
      InitialMode::constant,
      conventional_physical_state, &conventional_physical_response);
  const bool igr_periodic = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::nonzero,
      InitialMode::constant,
      igr_periodic_state, &igr_periodic_response);
  const bool igr_physical = order > 0 && CheckLifecycle(
      order, true, RegularizationKind::igr, SourceMode::nonzero,
      InitialMode::constant,
      igr_physical_state, &igr_physical_response);
  const bool conventional_empty_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::empty,
      InitialMode::constant,
      conventional_empty_source_state);
  const bool conventional_zero_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::none, SourceMode::explicit_zero,
      InitialMode::constant,
      conventional_zero_source_state);
  const bool igr_empty_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::empty,
      InitialMode::constant,
      igr_empty_source_state);
  const bool igr_zero_source = order > 0 && CheckLifecycle(
      order, false, RegularizationKind::igr, SourceMode::explicit_zero,
      InitialMode::constant,
      igr_zero_source_state);
  const bool source_equivalence = conventional_empty_source &&
      conventional_zero_source && igr_empty_source && igr_zero_source &&
      NearValues(conventional_empty_source_state,
                 conventional_zero_source_state) &&
      NearValues(igr_empty_source_state, igr_zero_source_state);
  const bool boundary_effect = conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical &&
      BoundaryResponseSeparated(conventional_periodic_response,
                                conventional_physical_response) &&
      BoundaryResponseSeparated(igr_periodic_response,
                                igr_physical_response);
  const bool passed = (ranks == 1 || ranks == 2) && All(
      admission && recovery_dispatch_admission && copied_initial_trace &&
      all_seven_boundary_policies &&
      mixed_boundary_policies && independent_igr_conservation &&
      conventional_conservation_rollback &&
      tensor_diagonal_convergence_warning_rollback &&
      conventional_p1_q12_tensor_volume_multi_chunk_parity &&
      euler_algebra && zero_update_final_reconstruction &&
      rejected_armijo_rollback && p3_consumer_vertices &&
      conventional_periodic && conventional_physical &&
      igr_periodic && igr_physical && boundary_effect && source_equivalence);
  if (!passed)
  {
    std::cerr << "generic CUDA Euler-extension qualification failed rank="
              << rank << " ranks=" << ranks << " order=" << order
              << " engine="
              << (selected_cuda_engine == CudaEngineKind::tensor_core
                      ? "tensor" : "ordinary")
              << " admission=" << admission
              << " recovery_dispatch_admission="
              << recovery_dispatch_admission
              << " copied_initial_trace=" << copied_initial_trace
              << " all_seven_boundary_policies="
              << all_seven_boundary_policies
              << " mixed_boundary_policies=" << mixed_boundary_policies
              << " independent_igr_conservation="
              << independent_igr_conservation
              << " conventional_conservation_rollback="
              << conventional_conservation_rollback
              << " tensor_diagonal_convergence_warning_rollback="
              << tensor_diagonal_convergence_warning_rollback
              << " conventional_p1_q12_tensor_volume_multi_chunk_parity="
              << conventional_p1_q12_tensor_volume_multi_chunk_parity
              << " euler_algebra=" << euler_algebra
              << " zero_update_final_reconstruction="
              << zero_update_final_reconstruction
              << " rejected_armijo_rollback="
              << rejected_armijo_rollback
              << " p3_consumer_vertices=" << p3_consumer_vertices
              << " conventional_periodic=" << conventional_periodic
              << " conventional_physical=" << conventional_physical
              << " igr_periodic=" << igr_periodic
              << " igr_physical=" << igr_physical
              << " boundary_effect=" << boundary_effect
              << " conventional_boundary_response="
              << static_cast<double>(conventional_periodic_response) << '/'
              << static_cast<double>(conventional_physical_response)
              << " igr_boundary_response="
              << static_cast<double>(igr_periodic_response) << '/'
              << static_cast<double>(igr_physical_response)
              << " source_equivalence=" << source_equivalence << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
