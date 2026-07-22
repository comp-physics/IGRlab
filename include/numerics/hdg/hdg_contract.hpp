#pragma once

/**
 * @file numerics/hdg/hdg_contract.hpp
 * @brief Static, model-neutral HDG operation and storage contracts.
 */

#include "numerics/hdg/hdg_topology.hpp"
#include "numerics/hdg/hdg_trace.hpp"

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace dg
{
namespace hdg
{

/** A logical strided batch of local blocks.  No packing is prescribed. */
template <typename Scalar>
struct StridedBlockView
{
  Scalar* data = nullptr;
  LocalIndex batch_count = 0;
  LocalIndex rows = 0;
  LocalIndex columns = 0;
  std::ptrdiff_t batch_stride = 0;
  std::ptrdiff_t row_stride = 0;
  std::ptrdiff_t column_stride = 0;

  DG_IGR_HDG_HOST_DEVICE constexpr Scalar& operator()(
      const LocalIndex batch,
      const LocalIndex row,
      const LocalIndex column) const noexcept
  {
    return data[batch * batch_stride + row * row_stride +
                column * column_stride];
  }
};

template <typename Scalar>
DG_IGR_HDG_HOST_DEVICE constexpr StridedBlockView<const Scalar> AsConst(
    const StridedBlockView<Scalar> view) noexcept
{
  return {view.data,
          view.batch_count,
          view.rows,
          view.columns,
          view.batch_stride,
          view.row_stride,
          view.column_stride};
}

/** Oracle-compatible logical blocks:
 * R = A*x + B*lambda - f and G = C*x + D*lambda.
 */
template <typename ResidualScalar, typename JacobianScalar>
struct LocalLinearizationView
{
  StridedBlockView<ResidualScalar> interior_residual;
  StridedBlockView<ResidualScalar> trace_residual;
  StridedBlockView<JacobianScalar> A;
  StridedBlockView<JacobianScalar> B;
  StridedBlockView<JacobianScalar> C;
  StridedBlockView<JacobianScalar> D;
};

struct WorkspaceRegion
{
  std::size_t offset = 0;
  std::size_t size = 0;
};

/** Per-batch workspace regions.  A backend chooses the actual allocation. */
struct WorkspaceLayout
{
  WorkspaceRegion local_residual;
  WorkspaceRegion local_jacobian;
  WorkspaceRegion factorization;
  WorkspaceRegion condensed_system;
  WorkspaceRegion temporary;
  std::size_t scalars_per_batch = 0;
  std::size_t batch_count = 0;

  constexpr std::size_t ScalarCount() const noexcept
  {
    return scalars_per_batch * batch_count;
  }
};

/** Checked host-side layout construction.  Invalid local dimensions are
 * rejected before the execution views perform size_t conversion. */
inline bool TryMakeWorkspaceLayout(
    const LocalBlockShape shape,
    const std::size_t batch_count,
    const std::size_t temporary_scalars,
    WorkspaceLayout& layout) noexcept
{
  if (!IsValidLocalBlockShape(shape))
  {
    return false;
  }
  const std::size_t ni = static_cast<std::size_t>(shape.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape.TraceUnknowns());
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const auto add = [maximum](const std::size_t left,
                             const std::size_t right,
                             std::size_t& result) noexcept
  {
    if (left > maximum - right)
    {
      return false;
    }
    result = left + right;
    return true;
  };
  const auto multiply = [maximum](const std::size_t left,
                                  const std::size_t right,
                                  std::size_t& result) noexcept
  {
    if (left != 0 && right > maximum / left)
    {
      return false;
    }
    result = left * right;
    return true;
  };

  std::size_t residual_count = 0;
  std::size_t ni_squared = 0;
  std::size_t ni_nt = 0;
  std::size_t nt_ni = 0;
  std::size_t nt_squared = 0;
  std::size_t jacobian_count = 0;
  std::size_t condensed_count = 0;
  if (!add(ni, nt, residual_count) || !multiply(ni, ni, ni_squared) ||
      !multiply(ni, nt, ni_nt) || !multiply(nt, ni, nt_ni) ||
      !multiply(nt, nt, nt_squared) || !add(ni_squared, ni_nt, jacobian_count) ||
      !add(jacobian_count, nt_ni, jacobian_count) ||
      !add(jacobian_count, nt_squared, jacobian_count) ||
      !add(nt_squared, nt, condensed_count))
  {
    return false;
  }
  const std::size_t factor_count = ni_squared;
  const WorkspaceRegion residual{0, residual_count};
  std::size_t jacobian_offset = 0;
  std::size_t factor_offset = 0;
  std::size_t condensed_offset = 0;
  std::size_t temporary_offset = 0;
  if (!add(residual.offset, residual.size, jacobian_offset) ||
      !add(jacobian_offset, jacobian_count, factor_offset) ||
      !add(factor_offset, factor_count, condensed_offset) ||
      !add(condensed_offset, condensed_count, temporary_offset))
  {
    return false;
  }
  const WorkspaceRegion jacobian{jacobian_offset, jacobian_count};
  const WorkspaceRegion factor{factor_offset, factor_count};
  const WorkspaceRegion condensed{condensed_offset, condensed_count};
  const WorkspaceRegion temporary{temporary_offset, temporary_scalars};
  std::size_t scalars_per_batch = 0;
  std::size_t scalar_count = 0;
  if (!add(temporary.offset, temporary.size, scalars_per_batch) ||
      !multiply(scalars_per_batch, batch_count, scalar_count))
  {
    return false;
  }
  (void)scalar_count;
  layout = {residual,
            jacobian,
            factor,
            condensed,
            temporary,
            scalars_per_batch,
            batch_count};
  return true;
}

template <typename Scalar>
struct WorkspaceView
{
  Scalar* data = nullptr;
  WorkspaceLayout layout{};

  DG_IGR_HDG_HOST_DEVICE constexpr Scalar* BatchData(
      const std::size_t batch) const noexcept
  {
    return data + batch * layout.scalars_per_batch;
  }

  DG_IGR_HDG_HOST_DEVICE constexpr Scalar* RegionData(
      const std::size_t batch,
      const WorkspaceRegion region) const noexcept
  {
    return BatchData(batch) + region.offset;
  }
};

/** Host reference owner. Device backends instead construct WorkspaceView from
 * their persistent MFEM/Umpire/raw allocation without changing the contract.
 */
template <typename Scalar>
class HostWorkspaceStorage
{
public:
  void Resize(const WorkspaceLayout layout)
  {
    layout_ = layout;
    storage_.resize(layout.ScalarCount());
  }

  WorkspaceView<Scalar> View() noexcept
  {
    return {storage_.data(), layout_};
  }

  WorkspaceView<const Scalar> View() const noexcept
  {
    return {storage_.data(), layout_};
  }

  const WorkspaceLayout& Layout() const noexcept { return layout_; }

private:
  WorkspaceLayout layout_{};
  std::vector<Scalar> storage_;
};

/** Backend composition points are arbitrary tags; no external backend is a
 * dependency of this header and no execution route is selected here.
 */
template <typename ExecutionTag,
          typename MemoryTag,
          typename CommunicationTag>
struct BackendPolicy
{
  using execution_tag = ExecutionTag;
  using memory_tag = MemoryTag;
  using communication_tag = CommunicationTag;
};

template <typename T>
struct IsBackendPolicy : std::false_type
{};

template <typename ExecutionTag, typename MemoryTag, typename CommunicationTag>
struct IsBackendPolicy<BackendPolicy<ExecutionTag, MemoryTag, CommunicationTag>>
    : std::true_type
{};

/** Explicit opt-in markers keep model/stabilization/time composition closed
 * without imposing a formulation or backend implementation. */
struct ModelTagMarker
{};
struct StabilizationTagMarker
{};
struct TimeTreatmentTagMarker
{};

template <typename T>
struct IsModelTag : std::is_base_of<ModelTagMarker, T>
{};

template <typename T>
struct IsStabilizationTag : std::is_base_of<StabilizationTagMarker, T>
{};

template <typename T>
struct IsTimeTreatmentTag : std::is_base_of<TimeTreatmentTagMarker, T>
{};

/** Precision roles remain distinct so later FP64 or mixed-precision research
 * does not change the logical HDG stage interfaces.
 */
template <typename StateScalar,
          typename ResidualScalar,
          typename JacobianScalar,
          typename FactorScalar,
          typename AccumulatorScalar>
struct PrecisionPolicy
{
  static_assert(std::is_floating_point<StateScalar>::value &&
                    std::is_floating_point<ResidualScalar>::value &&
                    std::is_floating_point<JacobianScalar>::value &&
                    std::is_floating_point<FactorScalar>::value &&
                    std::is_floating_point<AccumulatorScalar>::value,
                "HDG precision roles must use floating-point scalar types");
  static_assert(std::is_same<StateScalar, ResidualScalar>::value &&
                    std::is_same<StateScalar, JacobianScalar>::value &&
                    std::is_same<StateScalar, FactorScalar>::value &&
                    std::is_same<StateScalar, AccumulatorScalar>::value,
                "the static HDG core currently requires homogeneous precision across all five roles");
  using state_scalar = StateScalar;
  using residual_scalar = ResidualScalar;
  using jacobian_scalar = JacobianScalar;
  using factor_scalar = FactorScalar;
  using accumulator_scalar = AccumulatorScalar;
};

template <typename T>
struct IsPrecisionPolicy : std::false_type
{};

template <typename StateScalar,
          typename ResidualScalar,
          typename JacobianScalar,
          typename FactorScalar,
          typename AccumulatorScalar>
struct IsPrecisionPolicy<PrecisionPolicy<StateScalar,
                                         ResidualScalar,
                                         JacobianScalar,
                                         FactorScalar,
                                         AccumulatorScalar>> : std::true_type
{};

/** Closed compile-time dispatch description. DevicePolicy is owned by the
 * model adapter and must be POD; topology/condensation are not model hooks.
 */
template <typename ModelTag,
          typename TracePolicy,
          typename StabilizationTag,
          typename TimeTreatmentTag,
          typename Backend,
          typename Precision,
          typename DevicePolicy>
struct KernelSpec
{
  static_assert(IsModelTag<ModelTag>::value,
                "HDG model tags must explicitly opt into ModelTagMarker");
  static_assert(IsTraceTuple<TracePolicy>::value,
                "HDG kernel trace policy must be a TraceTuple");
  static_assert(IsStabilizationTag<StabilizationTag>::value,
                "HDG stabilization tags must explicitly opt into StabilizationTagMarker");
  static_assert(IsTimeTreatmentTag<TimeTreatmentTag>::value,
                "HDG time-treatment tags must explicitly opt into TimeTreatmentTagMarker");
  static_assert(IsBackendPolicy<Backend>::value,
                "HDG kernel backend must be a BackendPolicy");
  static_assert(IsPrecisionPolicy<Precision>::value,
                "HDG kernel precision must be a PrecisionPolicy");
  static_assert(TracePolicy::component_count <=
                    static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max()),
                "HDG trace component count exceeds LocalIndex");
  static_assert(std::is_trivially_copyable<DevicePolicy>::value,
                "per-model HDG device policy must be trivially copyable");
  static_assert(std::is_standard_layout<DevicePolicy>::value,
                "per-model HDG device policy must have standard layout");
  static_assert(!std::is_polymorphic<DevicePolicy>::value,
                "per-model HDG device policy cannot use virtual dispatch");

  using model_tag = ModelTag;
  using trace_policy = TracePolicy;
  using stabilization_tag = StabilizationTag;
  using time_treatment_tag = TimeTreatmentTag;
  using backend_policy = Backend;
  using precision_policy = Precision;
  using device_policy = DevicePolicy;
  using scalar_type = typename precision_policy::state_scalar;
  static constexpr LocalIndex trace_component_count =
      static_cast<LocalIndex>(TracePolicy::component_count);
};

template <typename T>
struct IsKernelSpec : std::false_type
{};

template <typename ModelTag,
          typename TracePolicy,
          typename StabilizationTag,
          typename TimeTreatmentTag,
          typename Backend,
          typename Precision,
          typename DevicePolicy>
struct IsKernelSpec<KernelSpec<ModelTag,
                               TracePolicy,
                               StabilizationTag,
                               TimeTreatmentTag,
                               Backend,
                               Precision,
                               DevicePolicy>> : std::true_type
{};

/** Element Schur contribution handed to an assembly policy. */
template <typename Scalar>
struct TraceAssemblyContribution
{
  LocalIndex element = invalid_local_index;
  const LocalIndex* trace_dofs = nullptr;
  LocalIndex trace_dof_count = 0;
  StridedBlockView<const Scalar> schur;
  StridedBlockView<const Scalar> right_hand_side;
};

/** The static core deliberately uses one scalar storage type for all five
 * named precision roles.  The aliases preserve the role boundary while making
 * the initial no-packing contract unambiguous. */
template <typename Spec>
struct KernelStorageRoles
{
  static_assert(IsKernelSpec<Spec>::value,
                "HDG storage roles require a validated KernelSpec");
  using precision = typename Spec::precision_policy;
  using state_scalar = typename precision::state_scalar;
  using residual_scalar = typename precision::residual_scalar;
  using jacobian_scalar = typename precision::jacobian_scalar;
  using factor_scalar = typename precision::factor_scalar;
  using accumulator_scalar = typename precision::accumulator_scalar;
};

template <typename Spec>
using KernelScalar = typename KernelStorageRoles<Spec>::state_scalar;

/** Separate logical role views retain allocation ownership with the backend.
 * Homogeneous precision makes all five scalar aliases equal in this core. */
template <typename Spec>
struct KernelWorkspaceView
{
  WorkspaceView<typename KernelStorageRoles<Spec>::residual_scalar> residual{};
  WorkspaceView<typename KernelStorageRoles<Spec>::jacobian_scalar> jacobian{};
  WorkspaceView<typename KernelStorageRoles<Spec>::factor_scalar> factorization{};
  WorkspaceView<typename KernelStorageRoles<Spec>::accumulator_scalar>
      condensed_system{};
  WorkspaceView<typename KernelStorageRoles<Spec>::accumulator_scalar> temporary{};
};

template <typename Spec>
struct ElementPhaseContext
{
  static_assert(IsKernelSpec<Spec>::value,
                "HDG phase contexts require a validated KernelSpec");
  HdgTopologyView topology{};
  LocalIndex element = invalid_local_index;
  StridedBlockView<const typename KernelStorageRoles<Spec>::state_scalar>
      interior_state{};
  StridedBlockView<const typename KernelStorageRoles<Spec>::state_scalar>
      trace_state{};
  KernelWorkspaceView<Spec> workspace{};
};

template <typename Spec>
struct LocalResidualRequest
{
  ElementPhaseContext<Spec> context{};
};

template <typename Spec>
struct LocalResidualResult
{
  StridedBlockView<typename KernelStorageRoles<Spec>::residual_scalar>
      interior_residual{};
  StridedBlockView<typename KernelStorageRoles<Spec>::residual_scalar>
      trace_residual{};
};

template <typename Spec>
struct LocalJacobianRequest
{
  ElementPhaseContext<Spec> context{};
};

template <typename Spec>
struct LocalJacobianResult
{
  LocalLinearizationView<typename KernelStorageRoles<Spec>::residual_scalar,
                         typename KernelStorageRoles<Spec>::jacobian_scalar>
      linearization{};
};

template <typename Spec>
struct CondenseRequest
{
  ElementPhaseContext<Spec> context{};
  LocalLinearizationView<typename KernelStorageRoles<Spec>::residual_scalar,
                         typename KernelStorageRoles<Spec>::jacobian_scalar>
      linearization{};
};

template <typename Spec>
struct CondenseResult
{
  TraceAssemblyContribution<typename KernelStorageRoles<Spec>::accumulator_scalar>
      contribution{};
};

template <typename Spec>
struct ApplySchurRequest
{
  ElementPhaseContext<Spec> context{};
  TraceAssemblyContribution<typename KernelStorageRoles<Spec>::accumulator_scalar>
      contribution{};
};

template <typename Spec>
struct ApplySchurResult
{
  StridedBlockView<typename KernelStorageRoles<Spec>::accumulator_scalar>
      trace_residual{};
};

template <typename Spec>
struct TraceAssemblyRequest
{
  TraceAssemblyContribution<typename KernelStorageRoles<Spec>::accumulator_scalar>
      contribution{};
};

template <typename Spec>
struct TraceAssemblyResult
{
  StridedBlockView<typename KernelStorageRoles<Spec>::accumulator_scalar>
      trace_system{};
  StridedBlockView<typename KernelStorageRoles<Spec>::accumulator_scalar>
      right_hand_side{};
};

template <typename Spec>
struct TraceSolveRequest
{
  StridedBlockView<const typename KernelStorageRoles<Spec>::accumulator_scalar>
      trace_system{};
  StridedBlockView<const typename KernelStorageRoles<Spec>::accumulator_scalar>
      right_hand_side{};
};

template <typename Spec>
struct TraceSolveResult
{
  StridedBlockView<typename KernelStorageRoles<Spec>::state_scalar>
      trace_solution{};
};

template <typename Spec>
struct RecoveryRequest
{
  ElementPhaseContext<Spec> context{};
  StridedBlockView<const typename KernelStorageRoles<Spec>::state_scalar>
      trace_solution{};
};

template <typename Spec>
struct RecoveryResult
{
  StridedBlockView<typename KernelStorageRoles<Spec>::state_scalar>
      interior_solution{};
};

/** Validates dynamic topology against the compile-time trace tuple before its
 * unchecked execution view is captured by a backend kernel. */
template <typename Spec>
bool HasConsistentRangesForSpec(const HdgTopologyStorage& topology) noexcept
{
  static_assert(IsKernelSpec<Spec>::value,
                "HDG topology/spec validation requires a KernelSpec");
  if (!topology.HasConsistentRanges())
  {
    return false;
  }
  const LocalIndex components = Spec::trace_component_count;
  for (const auto& element : topology.elements)
  {
    if (element.blocks.trace_components != components)
    {
      return false;
    }
  }
  for (const auto& face : topology.faces)
  {
    if (face.trace_count % components != 0)
    {
      return false;
    }
  }
  for (const auto& dof : topology.trace_dofs)
  {
    if (dof.component >= components)
    {
      return false;
    }
  }
  return true;
}

namespace detail
{

#define DG_IGR_HDG_DECLARE_EXACT_PHASE(TRAIT, METHOD)                         \
  template <typename Operator, typename Request, typename Result, typename = void> \
  struct TRAIT : std::false_type                                                \
  {};                                                                            \
  template <typename Operator, typename Request, typename Result>              \
  struct TRAIT<Operator,                                                         \
               Request,                                                          \
               Result,                                                           \
               std::void_t<decltype(std::declval<Operator&>().METHOD(          \
                   std::declval<const Request&>(), std::declval<Result&>()))>> \
      : std::is_same<decltype(std::declval<Operator&>().METHOD(                 \
                         std::declval<const Request&>(),                        \
                         std::declval<Result&>())),                             \
                     void>                                                       \
  {}

DG_IGR_HDG_DECLARE_EXACT_PHASE(HasEvaluateResidual, EvaluateResidual);
DG_IGR_HDG_DECLARE_EXACT_PHASE(HasEvaluateJacobian, EvaluateJacobian);
DG_IGR_HDG_DECLARE_EXACT_PHASE(HasCondense, Condense);
DG_IGR_HDG_DECLARE_EXACT_PHASE(HasApplySchur, ApplySchur);
DG_IGR_HDG_DECLARE_EXACT_PHASE(HasAssemble, Assemble);
DG_IGR_HDG_DECLARE_EXACT_PHASE(HasSolve, Solve);
DG_IGR_HDG_DECLARE_EXACT_PHASE(HasRecover, Recover);

#undef DG_IGR_HDG_DECLARE_EXACT_PHASE

} // namespace detail

/** Statically-dispatched, Spec-bound HDG phase boundary. */
template <typename Spec,
          typename LocalOperator,
          typename CondensationOperator,
          typename TraceAssembler,
          typename GlobalTraceSolver,
          typename RecoveryOperator>
class Pipeline
{
  static_assert(IsKernelSpec<Spec>::value,
                "HDG pipelines require a validated KernelSpec");
  static_assert(!std::is_polymorphic<LocalOperator>::value,
                "HDG local operator cannot use virtual dispatch");
  static_assert(!std::is_polymorphic<CondensationOperator>::value,
                "HDG condensation operator cannot use virtual dispatch");
  static_assert(!std::is_polymorphic<TraceAssembler>::value,
                "HDG trace assembler cannot use virtual dispatch");
  static_assert(!std::is_polymorphic<GlobalTraceSolver>::value,
                "HDG trace solver cannot use virtual dispatch");
  static_assert(!std::is_polymorphic<RecoveryOperator>::value,
                "HDG recovery operator cannot use virtual dispatch");
  static_assert(detail::HasEvaluateResidual<LocalOperator,
                                            LocalResidualRequest<Spec>,
                                            LocalResidualResult<Spec>>::value,
                "HDG local residual phase must use its exact Spec-bound request/result signature");
  static_assert(detail::HasEvaluateJacobian<LocalOperator,
                                            LocalJacobianRequest<Spec>,
                                            LocalJacobianResult<Spec>>::value,
                "HDG local Jacobian phase must use its exact Spec-bound request/result signature");
  static_assert(detail::HasCondense<CondensationOperator,
                                    CondenseRequest<Spec>,
                                    CondenseResult<Spec>>::value,
                "HDG condensation phase must use its exact Spec-bound request/result signature");
  static_assert(detail::HasApplySchur<CondensationOperator,
                                      ApplySchurRequest<Spec>,
                                      ApplySchurResult<Spec>>::value,
                "HDG Schur phase must use its exact Spec-bound request/result signature");
  static_assert(detail::HasAssemble<TraceAssembler,
                                    TraceAssemblyRequest<Spec>,
                                    TraceAssemblyResult<Spec>>::value,
                "HDG trace assembly phase must use its exact Spec-bound request/result signature");
  static_assert(detail::HasSolve<GlobalTraceSolver,
                                 TraceSolveRequest<Spec>,
                                 TraceSolveResult<Spec>>::value,
                "HDG trace solve phase must use its exact Spec-bound request/result signature");
  static_assert(detail::HasRecover<RecoveryOperator,
                                   RecoveryRequest<Spec>,
                                   RecoveryResult<Spec>>::value,
                "HDG recovery phase must use its exact Spec-bound request/result signature");

public:
  using spec_type = Spec;

  Pipeline(LocalOperator local,
           CondensationOperator condensation,
           TraceAssembler assembler,
           GlobalTraceSolver solver,
           RecoveryOperator recovery)
      : local_(std::move(local)),
        condensation_(std::move(condensation)),
        assembler_(std::move(assembler)),
        solver_(std::move(solver)),
        recovery_(std::move(recovery))
  {}

  void EvaluateLocalResidual(const LocalResidualRequest<Spec>& request,
                             LocalResidualResult<Spec>& result)
  {
    local_.EvaluateResidual(request, result);
  }

  void EvaluateLocalJacobian(const LocalJacobianRequest<Spec>& request,
                             LocalJacobianResult<Spec>& result)
  {
    local_.EvaluateJacobian(request, result);
  }

  void CondenseLocalSystem(const CondenseRequest<Spec>& request,
                           CondenseResult<Spec>& result)
  {
    condensation_.Condense(request, result);
  }

  void ApplySchur(const ApplySchurRequest<Spec>& request,
                  ApplySchurResult<Spec>& result)
  {
    condensation_.ApplySchur(request, result);
  }

  void AssembleTraceSystem(const TraceAssemblyRequest<Spec>& request,
                           TraceAssemblyResult<Spec>& result)
  {
    assembler_.Assemble(request, result);
  }

  void SolveGlobalTraceSystem(const TraceSolveRequest<Spec>& request,
                              TraceSolveResult<Spec>& result)
  {
    solver_.Solve(request, result);
  }

  void RecoverInterior(const RecoveryRequest<Spec>& request,
                       RecoveryResult<Spec>& result)
  {
    recovery_.Recover(request, result);
  }

private:
  LocalOperator local_;
  CondensationOperator condensation_;
  TraceAssembler assembler_;
  GlobalTraceSolver solver_;
  RecoveryOperator recovery_;
};

static_assert(std::is_trivially_copyable<StridedBlockView<double>>::value,
              "HDG strided blocks must be trivially copyable");
static_assert(std::is_standard_layout<StridedBlockView<double>>::value,
              "HDG strided blocks must have standard layout");
static_assert(std::is_trivially_copyable<WorkspaceLayout>::value,
              "HDG workspace layouts must be trivially copyable");
static_assert(std::is_trivially_copyable<WorkspaceView<double>>::value,
              "HDG workspace views must be trivially copyable");
static_assert(
    std::is_trivially_copyable<TraceAssemblyContribution<double>>::value,
    "HDG trace assembly contributions must be trivially copyable");
static_assert(
    std::is_trivially_copyable<LocalLinearizationView<double, double>>::value,
    "HDG local linearization views must be trivially copyable");

} // namespace hdg
} // namespace dg
