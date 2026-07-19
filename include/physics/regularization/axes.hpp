#pragma once

#include <cstdint>

namespace dg_igr
{
namespace regularization
{

/** The continuum closure/model axis. Only neutral fixture models are
 * registered by CONTRACT-01. */
enum class ContinuumModelKind : std::uint8_t
{
  ZeroFixture,
  ConstantFixture
};

/** The finite-element representation used for auxiliary unknowns. */
enum class AuxiliaryDiscretizationKind : std::uint8_t
{
  ElementL2,
  TraceL2,
  HybridizedTrace
};

/** The strategy used to obtain the auxiliary solution. */
enum class AuxiliarySolutionKind : std::uint8_t
{
  PointwiseEvaluation,
  LocalElimination,
  GlobalLinearSolve
};

/** Spatial operator family within the fourth independent axis. */
enum class SpatialBackendKind : std::uint8_t
{
  DG,
  DGSEM,
  HDG
};

/** MFEM execution target within the fourth independent axis. */
enum class ExecutionTargetKind : std::uint8_t
{
  MFEMCpu,
  MFEMDevice
};

/** The fourth axis is the Cartesian product of spatial operator and execution
 * target. Neither member implies the other. */
struct SpatialExecutionBackend
{
  SpatialBackendKind spatial = SpatialBackendKind::DG;
  ExecutionTargetKind execution = ExecutionTargetKind::MFEMCpu;
};

/** Time compatibility is validated separately from the four model axes. */
enum class TimeTreatmentKind : std::uint8_t
{
  FrozenRKStage,
  LaggedStep,
  FullyCoupled
};

/** Entropy compatibility is validated separately from the four model axes. */
enum class EntropyTreatmentKind : std::uint8_t
{
  Neutral,
  EntropyConservative,
  EntropyStable
};

struct RegularizationAxes
{
  ContinuumModelKind continuum_model = ContinuumModelKind::ZeroFixture;
  AuxiliaryDiscretizationKind auxiliary_discretization =
    AuxiliaryDiscretizationKind::ElementL2;
  AuxiliarySolutionKind auxiliary_solution =
    AuxiliarySolutionKind::PointwiseEvaluation;
  SpatialExecutionBackend backend{};
};

struct RegularizationCompatibility
{
  TimeTreatmentKind time = TimeTreatmentKind::FrozenRKStage;
  EntropyTreatmentKind entropy = EntropyTreatmentKind::Neutral;
};

struct RegularizationSelection
{
  RegularizationAxes axes{};
  RegularizationCompatibility compatibility{};
};

// Type-level representatives used after the one-time host selection.
struct ZeroFixtureModelTag
{};
struct ConstantFixtureModelTag
{};
struct UnregisteredFixtureModelTag
{};

struct ElementL2Tag
{};
struct TraceL2Tag
{};
struct HybridizedTraceTag
{};

struct PointwiseEvaluationTag
{};
struct LocalEliminationTag
{};
struct GlobalLinearSolveTag
{};

struct DGSpatialBackendTag
{};
struct DGSEMSpatialBackendTag
{};
struct HDGSpatialBackendTag
{};

struct MFEMCpuTargetTag
{
  static constexpr bool use_device = false;
};
struct MFEMDeviceTargetTag
{
  static constexpr bool use_device = true;
};

template <typename SpatialTag, typename ExecutionTag>
struct SpatialExecutionBackendTag
{
  using spatial_backend_tag = SpatialTag;
  using execution_target_tag = ExecutionTag;
  static constexpr bool use_device = ExecutionTag::use_device;
};

using DGCPUBackendTag =
  SpatialExecutionBackendTag<DGSpatialBackendTag, MFEMCpuTargetTag>;
using DGDeviceBackendTag =
  SpatialExecutionBackendTag<DGSpatialBackendTag, MFEMDeviceTargetTag>;
using DGSEMCPUBackendTag =
  SpatialExecutionBackendTag<DGSEMSpatialBackendTag, MFEMCpuTargetTag>;
using DGSEMDeviceBackendTag =
  SpatialExecutionBackendTag<DGSEMSpatialBackendTag, MFEMDeviceTargetTag>;
using HDGCPUBackendTag =
  SpatialExecutionBackendTag<HDGSpatialBackendTag, MFEMCpuTargetTag>;
using HDGDeviceBackendTag =
  SpatialExecutionBackendTag<HDGSpatialBackendTag, MFEMDeviceTargetTag>;

struct FrozenRKStageTag
{};
struct LaggedStepTag
{};
struct FullyCoupledTag
{};

struct EntropyNeutralTag
{};
struct EntropyConservativeTag
{};
struct EntropyStableTag
{};

} // namespace regularization
} // namespace dg_igr
