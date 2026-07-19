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

/** The MFEM execution policy selected for a launch. */
enum class ExecutionBackendKind : std::uint8_t
{
  MFEMCpu,
  MFEMDevice
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
  ExecutionBackendKind execution_backend = ExecutionBackendKind::MFEMCpu;
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

struct MFEMCpuBackendTag
{
  static constexpr bool use_device = false;
};
struct MFEMDeviceBackendTag
{
  static constexpr bool use_device = true;
};

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
