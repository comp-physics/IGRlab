#pragma once

/**
 * @file numerics/hdg/hdg_contract.hpp
 * @brief Static, model-neutral HDG operation and storage contracts.
 */

#include "numerics/hdg/hdg_topology.hpp"
#include "numerics/hdg/hdg_trace.hpp"

#include <cstddef>
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
template <typename Scalar>
struct LocalLinearizationView
{
  StridedBlockView<Scalar> interior_residual;
  StridedBlockView<Scalar> trace_residual;
  StridedBlockView<Scalar> A;
  StridedBlockView<Scalar> B;
  StridedBlockView<Scalar> C;
  StridedBlockView<Scalar> D;
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

inline constexpr WorkspaceLayout MakeWorkspaceLayout(
    const LocalBlockShape shape,
    const std::size_t batch_count,
    const std::size_t temporary_scalars = 0) noexcept
{
  const std::size_t ni = static_cast<std::size_t>(shape.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape.TraceUnknowns());
  const std::size_t residual_count = ni + nt;
  const std::size_t jacobian_count = ni * ni + ni * nt + nt * ni + nt * nt;
  const std::size_t factor_count = ni * ni;
  const std::size_t condensed_count = nt * nt + nt;
  const WorkspaceRegion residual{0, residual_count};
  const WorkspaceRegion jacobian{residual.offset + residual.size,
                                 jacobian_count};
  const WorkspaceRegion factor{jacobian.offset + jacobian.size, factor_count};
  const WorkspaceRegion condensed{factor.offset + factor.size,
                                  condensed_count};
  const WorkspaceRegion temporary{condensed.offset + condensed.size,
                                  temporary_scalars};
  return {residual,
          jacobian,
          factor,
          condensed,
          temporary,
          temporary.offset + temporary.size,
          batch_count};
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
  using state_scalar = StateScalar;
  using residual_scalar = ResidualScalar;
  using jacobian_scalar = JacobianScalar;
  using factor_scalar = FactorScalar;
  using accumulator_scalar = AccumulatorScalar;
};

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
  static_assert(IsTraceTuple<TracePolicy>::value,
                "HDG kernel trace policy must be a TraceTuple");
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
};

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

/**
 * Statically-dispatched HDG phase boundary.  Each policy is a concrete type;
 * calls are direct member calls and polymorphic policies are rejected.  The
 * argument types remain generic so MFEM reference objects and future packed
 * backend views can share the same phase ordering without sharing storage.
 */
template <typename Spec,
          typename LocalOperator,
          typename CondensationOperator,
          typename TraceAssembler,
          typename GlobalTraceSolver,
          typename RecoveryOperator>
class Pipeline
{
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

  template <typename... Args>
  decltype(auto) EvaluateLocalResidual(Args&&... args)
  {
    return local_.EvaluateResidual(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) EvaluateLocalJacobian(Args&&... args)
  {
    return local_.EvaluateJacobian(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) CondenseLocalSystem(Args&&... args)
  {
    return condensation_.Condense(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) ApplySchur(Args&&... args)
  {
    return condensation_.ApplySchur(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) AssembleTraceSystem(Args&&... args)
  {
    return assembler_.Assemble(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) SolveGlobalTraceSystem(Args&&... args)
  {
    return solver_.Solve(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) RecoverInterior(Args&&... args)
  {
    return recovery_.Recover(std::forward<Args>(args)...);
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
    std::is_trivially_copyable<LocalLinearizationView<double>>::value,
    "HDG local linearization views must be trivially copyable");

} // namespace hdg
} // namespace dg
