#include "physics/regularization/regularization.hpp"

#include "mfem.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

namespace regularization = dg_igr::regularization;

namespace
{

template <typename Storage, typename = void>
struct CanBindEvaluatedPointField : std::false_type
{
};

template <typename Storage>
struct CanBindEvaluatedPointField<
  Storage,
  std::void_t<decltype(regularization::BindEvaluatedPointField(
    std::declval<const Storage&>(),
    std::declval<regularization::EvaluatedPointFieldMetadata>()))>>
  : std::true_type
{
};

static_assert(CanBindEvaluatedPointField<mfem::Vector>::value,
              "explicit MFEM vectors must bind as evaluated point arrays");
static_assert(!CanBindEvaluatedPointField<mfem::GridFunction>::value,
              "raw GridFunction DOFs must not bind as evaluated points");
#ifdef MFEM_USE_MPI
static_assert(!CanBindEvaluatedPointField<mfem::ParGridFunction>::value,
              "raw ParGridFunction DOFs must not bind as evaluated points");
#endif
static_assert(
  std::is_trivially_copyable<regularization::EvaluatedPointFieldBinding>::value,
  "evaluated field binding must remain trivially copyable");
static_assert(
  std::is_trivially_copyable<regularization::RegularizationPointInput>::value,
  "point input must remain trivially copyable");
static_assert(regularization::HasRegularizationTraits<
                regularization::BrokenDGContractProbeModelTag>::value,
              "optional-hook probe must satisfy trait discovery");
static_assert(
  !regularization::RegularizationTraits<
     regularization::BrokenDGContractProbeModelTag>::capabilities.wall_law,
  "probe wall capability must be false");
static_assert(
  !regularization::RegularizationTraits<
     regularization::BrokenDGContractProbeModelTag>::capabilities.diagnostics,
  "probe diagnostic capability must be false");
static_assert(regularization::SupportedCpuFixtureCombination::value,
              "existing CPU fixture combination must remain supported");
static_assert(regularization::SupportedDeviceFixtureCombination::value,
              "existing device fixture combination must remain supported");
static_assert(regularization::SupportedHDGDeviceFixtureCombination::value,
              "existing HDG device fixture combination must remain supported");
static_assert(regularization::SupportedHDGHybridizedCombination::value,
              "existing HDG hybridized fixture must remain supported");
static_assert(regularization::SupportedBrokenDGProbeCombination::value,
              "exact BrokenDG probe combination must be supported");

using ProbeDevice = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGDeviceBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeDGSEM = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGSEMCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeHDG = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::HDGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbePointwise = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::PointwiseEvaluationTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeLocal = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::LocalEliminationTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeTrace = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::TraceL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeHybrid = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::HybridizedTraceTag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeNoGradient = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::NoneGradientTreatmentTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyNeutralTag>;
using ProbeLagged = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::LaggedStepTag,
  regularization::EntropyNeutralTag>;
using ProbeCoupled = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FullyCoupledTag,
  regularization::EntropyNeutralTag>;
using ProbeEntropyConservative = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyConservativeTag>;
using ProbeEntropyStable = regularization::StaticCombinationSupport<
  regularization::BrokenDGContractProbeModelTag,
  regularization::ElementL2Tag,
  regularization::GlobalLinearSolveTag,
  regularization::DGCPUBackendTag,
  regularization::ConservativeStateBrokenDGTag,
  regularization::FrozenRKStageTag,
  regularization::EntropyStableTag>;

static_assert(!ProbeDevice::value && !ProbeDGSEM::value && !ProbeHDG::value,
              "probe backend support must be exactly DG/MFEM CPU");
static_assert(!ProbePointwise::value && !ProbeLocal::value,
              "probe solve support must be exactly global linear solve");
static_assert(!ProbeTrace::value && !ProbeHybrid::value,
              "probe discretization support must be exactly ElementL2");
static_assert(!ProbeNoGradient::value,
              "probe must require ConservativeStateBrokenDG");
static_assert(!ProbeLagged::value && !ProbeCoupled::value,
              "probe must require frozen-stage time treatment");
static_assert(!ProbeEntropyConservative::value && !ProbeEntropyStable::value,
              "probe must require entropy-neutral treatment");

class TestContext
{
public:
  void Check(bool condition, const char* message)
  {
    if (!condition)
    {
      ++failures_;
      std::cerr << "[FAIL] " << message << '\n';
    }
  }

  void Near(mfem::real_t actual,
            mfem::real_t expected,
            mfem::real_t tolerance,
            const char* message)
  {
    if (std::abs(actual - expected) > tolerance)
    {
      ++failures_;
      std::cerr << "[FAIL] " << message << ": actual=" << actual
                << " expected=" << expected << '\n';
    }
  }

  int Failures() const { return failures_; }

private:
  int failures_ = 0;
};

constexpr mfem::real_t PrecisionTolerance()
{
  return static_cast<mfem::real_t>(64) *
         std::numeric_limits<mfem::real_t>::epsilon();
}

regularization::RegularizationValidationContext ValidationContext(
  int expected_rk_stage = 2,
  std::uint64_t expected_state_generation = 19u,
  regularization::ExpectedEvaluatedPointProducerGenerations
    expected_producer_generations = {11u, 17u, 23u, 29u, 31u})
{
  return {false,
          expected_rk_stage,
          expected_state_generation,
          expected_producer_generations};
}

regularization::AuxiliaryFieldCapabilities VolumeCapabilities()
{
  using regularization::AuxiliaryCapability;
  return {regularization::CapabilityBit(AuxiliaryCapability::DeviceAccessible) |
          regularization::CapabilityBit(AuxiliaryCapability::OutputVisible) |
          regularization::CapabilityBit(AuxiliaryCapability::StageLocal)};
}

regularization::RegularizationSelection
ConstantCpuSelection(regularization::SpatialBackendKind spatial =
                       regularization::SpatialBackendKind::DG)
{
  regularization::RegularizationSelection selection;
  selection.axes.continuum_model =
    regularization::ContinuumModelKind::ConstantFixture;
  selection.axes.gradient_treatment =
    regularization::GradientTreatmentKind::None;
  selection.axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::ElementL2;
  selection.axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::PointwiseEvaluation;
  selection.axes.backend.spatial = spatial;
  selection.axes.backend.execution =
    regularization::ExecutionTargetKind::MFEMCpu;
  selection.compatibility.time =
    regularization::TimeTreatmentKind::FrozenRKStage;
  selection.compatibility.entropy =
    regularization::EntropyTreatmentKind::Neutral;
  return selection;
}

regularization::RegularizationSelection ProbeSelection()
{
  regularization::RegularizationSelection selection;
  selection.axes.continuum_model =
    regularization::ContinuumModelKind::BrokenDGContractProbe;
  selection.axes.gradient_treatment =
    regularization::GradientTreatmentKind::ConservativeStateBrokenDG;
  selection.axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::ElementL2;
  selection.axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::GlobalLinearSolve;
  selection.axes.backend.spatial = regularization::SpatialBackendKind::DG;
  selection.axes.backend.execution =
    regularization::ExecutionTargetKind::MFEMCpu;
  selection.compatibility.time =
    regularization::TimeTreatmentKind::FrozenRKStage;
  selection.compatibility.entropy =
    regularization::EntropyTreatmentKind::Neutral;
  return selection;
}

struct PointFixture
{
  int points;
  int dimension;
  int state_components;
  int auxiliary_components;

  mfem::Vector state;
  mfem::Vector gradient;
  mfem::Vector auxiliary;
  mfem::Vector trace;
  mfem::Vector normal_input;
  mfem::Vector coefficient;
  mfem::Vector source;
  mfem::Vector volume;
  mfem::Vector normal_output;
  mfem::Vector wall;
  mfem::Vector bounds;
  mfem::Vector diagnostics;

  PointFixture(int point_count,
               int dimension_in,
               int state_components_in,
               int auxiliary_components_in = 1)
    : points(point_count),
      dimension(dimension_in),
      state_components(state_components_in),
      auxiliary_components(auxiliary_components_in),
      state(state_components * points),
      gradient(state_components * dimension * points),
      auxiliary(auxiliary_components * points),
      trace(points),
      normal_input(dimension * points),
      coefficient(auxiliary_components * points),
      source(auxiliary_components * points),
      volume(state_components * dimension * points),
      normal_output(state_components * points),
      wall(state_components * points),
      bounds(2 * points),
      diagnostics(3 * points)
  {
    auto state_view =
      mfem::Reshape(state.HostWrite(), state_components, points);
    auto normal_view =
      mfem::Reshape(normal_input.HostWrite(), dimension, points);
    mfem::real_t* trace_values = trace.HostWrite();
    for (int point = 0; point < points; ++point)
    {
      for (int component = 0; component < state_components; ++component)
      {
        state_view(component, point) =
          static_cast<mfem::real_t>(10 * component + point + 1);
      }
      trace_values[point] = state_view(0, point) + 0.5;
      for (int direction = 0; direction < dimension; ++direction)
      {
        normal_view(direction, point) =
          direction == 0 ? static_cast<mfem::real_t>(1.0)
                         : static_cast<mfem::real_t>(0.25 * (point + 1));
      }
    }
    gradient = static_cast<mfem::real_t>(0.25);
    auxiliary = static_cast<mfem::real_t>(2.0);
    ResetOutputs(0.0);
  }

  regularization::EvaluatedPointFieldMetadata
  Metadata(int components,
           regularization::EvaluationSiteKind site,
           std::uint64_t point_set_id,
           std::uint64_t producer_generation,
           int stage = 2,
           std::uint64_t state_generation = 19u,
           int point_count = -1) const
  {
    return {point_count < 0 ? points : point_count,
            components,
            dimension,
            site,
            point_set_id,
            stage,
            state_generation,
            producer_generation};
  }

  regularization::EvaluatedPointBatch
  Batch(regularization::EvaluationSiteKind site,
        std::uint64_t point_set_id,
        int stage = 2,
        std::uint64_t state_generation = 19u) const
  {
    regularization::EvaluatedPointBatch batch;
    batch.point_count = points;
    batch.dimension = dimension;
    batch.site = site;
    batch.point_set_id = point_set_id;
    batch.rk_stage = stage;
    batch.state_generation = state_generation;
    batch.state = regularization::BindEvaluatedPointField(
      state,
      Metadata(state_components,
               site,
               point_set_id,
               11u,
               stage,
               state_generation));
    batch.conservative_state_gradient = regularization::BindEvaluatedPointField(
      gradient,
      Metadata(state_components * dimension,
               site,
               point_set_id,
               17u,
               stage,
               state_generation));
    batch.auxiliary = regularization::BindEvaluatedPointField(
      auxiliary,
      Metadata(auxiliary_components,
               site,
               point_set_id,
               23u,
               stage,
               state_generation));
    batch.trace = regularization::BindEvaluatedPointField(
      trace,
      Metadata(1,
               site,
               point_set_id,
               29u,
               stage,
               state_generation));
    batch.normal = regularization::BindEvaluatedPointField(
      normal_input,
      Metadata(dimension,
               site,
               point_set_id,
               31u,
               stage,
               state_generation));
    return batch;
  }

  regularization::RegularizationOperationOutputs
  Outputs(regularization::EvaluationSiteKind site, std::uint64_t point_set_id)
  {
    return {
      regularization::BindEvaluatedPointOutput(coefficient,
                                               points,
                                               auxiliary_components,
                                               dimension,
                                               site,
                                               point_set_id),
      regularization::BindEvaluatedPointOutput(source,
                                               points,
                                               auxiliary_components,
                                               dimension,
                                               site,
                                               point_set_id),
      regularization::BindEvaluatedPointOutput(volume,
                                               points,
                                               state_components * dimension,
                                               dimension,
                                               site,
                                               point_set_id),
      regularization::BindEvaluatedPointOutput(normal_output,
                                               points,
                                               state_components,
                                               dimension,
                                               site,
                                               point_set_id),
      regularization::BindEvaluatedPointOutput(wall,
                                               points,
                                               state_components,
                                               dimension,
                                               site,
                                               point_set_id),
      regularization::BindEvaluatedPointOutput(bounds,
                                               points,
                                               2,
                                               dimension,
                                               site,
                                               point_set_id),
      regularization::BindEvaluatedPointOutput(diagnostics,
                                               points,
                                               3,
                                               dimension,
                                               site,
                                               point_set_id)};
  }

  void ResetOutputs(mfem::real_t value)
  {
    coefficient = value;
    source = value;
    volume = value;
    normal_output = value;
    wall = value;
    bounds = value;
    diagnostics = value;
  }
};

bool AllValuesEqual(const mfem::Vector& values, mfem::real_t expected)
{
  const mfem::real_t* read = values.HostRead();
  for (int index = 0; index < values.Size(); ++index)
  {
    if (read[index] != expected)
    {
      return false;
    }
  }
  return true;
}
bool AllOutputsEqual(const PointFixture& fixture, mfem::real_t expected)
{
  const mfem::Vector* const outputs[] = {&fixture.coefficient,
                                         &fixture.source,
                                         &fixture.volume,
                                         &fixture.normal_output,
                                         &fixture.wall,
                                         &fixture.bounds,
                                         &fixture.diagnostics};
  for (const mfem::Vector* output : outputs)
  {
    if (!AllValuesEqual(*output, expected))
    {
      return false;
    }
  }
  return true;
}

void TestAuxiliaryRestartTruthfulness(TestContext& test)
{
  constexpr regularization::AuxiliaryFieldIdentity serialized_identity{0x1001u,
                                                                       0x2001u,
                                                                       0x3001u};
  constexpr regularization::AuxiliaryFieldIdentity cold_identity{0x1002u,
                                                                 0u,
                                                                 0x3002u};
  const regularization::AuxiliaryFieldProvenance provenance{0, 19u, 7u};
  mfem::Vector serialized_storage(3);
  mfem::Vector cold_storage(3);

  regularization::ScalarAuxiliaryFieldDescriptor serialized(
    serialized_storage,
    3,
    2,
    regularization::AuxiliaryFieldOwnership::StateOwned,
    provenance,
    serialized_identity,
    regularization::AuxiliaryRestartKind::Serialized,
    VolumeCapabilities());
  regularization::ScalarAuxiliaryFieldDescriptor cold(
    cold_storage,
    3,
    2,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    cold_identity,
    regularization::AuxiliaryRestartKind::ColdRecomputed,
    VolumeCapabilities());
  mfem::Vector none_storage(3);
  regularization::ScalarAuxiliaryFieldDescriptor none(
    none_storage,
    3,
    2,
    regularization::AuxiliaryFieldOwnership::BorrowedReadOnly,
    provenance,
    {0x1003u, 0u, 0x3003u},
    regularization::AuxiliaryRestartKind::None,
    VolumeCapabilities());

  test.Check(regularization::ValidateAuxiliaryFieldBinding(
               regularization::BindAuxiliaryField(none))
               .Ok(),
             "None with restart identity zero and RK stage zero is accepted");

  test.Check(regularization::ValidateAuxiliaryFieldBinding(
               regularization::BindAuxiliaryField(serialized))
               .Ok(),
             "Serialized with nonzero restart identity is accepted");
  test.Check(regularization::ValidateAuxiliaryFieldBinding(
               regularization::BindAuxiliaryField(cold))
               .Ok(),
             "ColdRecomputed with restart identity zero is accepted");
  test.Check(cold.RestartKind() ==
                 regularization::AuxiliaryRestartKind::ColdRecomputed &&
               cold.Provenance().producer_generation != 0u,
             "cold recomputation retains a fresh producer epoch");

  regularization::AuxiliaryFieldBinding invalid =
    regularization::BindAuxiliaryField(serialized);
  invalid.identity.restart_id = 0u;
  test.Check(regularization::ValidateAuxiliaryFieldBinding(invalid).code ==
               regularization::ValidationCode::InvalidFieldIdentity,
             "Serialized with restart identity zero is rejected");

  invalid = regularization::BindAuxiliaryField(cold);
  invalid.identity.restart_id = 0x9999u;
  test.Check(regularization::ValidateAuxiliaryFieldBinding(invalid).code ==
               regularization::ValidationCode::InvalidFieldIdentity,
             "ColdRecomputed with nonzero restart identity is rejected");

  invalid = regularization::BindAuxiliaryField(serialized);
  invalid.provenance.rk_stage = -1;
  test.Check(regularization::ValidateAuxiliaryFieldBinding(invalid).code ==
               regularization::ValidationCode::InvalidProvenanceStamp,
             "auxiliary RK stage -1 is an invalid unbound sentinel");
}

void TestExistingFixtureSelections(TestContext& test)
{
  const regularization::ContinuumModelKind models[] = {
    regularization::ContinuumModelKind::ZeroFixture,
    regularization::ContinuumModelKind::ConstantFixture};
  const regularization::SpatialBackendKind spatial_backends[] = {
    regularization::SpatialBackendKind::DG,
    regularization::SpatialBackendKind::DGSEM,
    regularization::SpatialBackendKind::HDG};
  const regularization::ExecutionTargetKind targets[] = {
    regularization::ExecutionTargetKind::MFEMCpu,
    regularization::ExecutionTargetKind::MFEMDevice};

  for (regularization::ContinuumModelKind model : models)
  {
    for (regularization::SpatialBackendKind spatial : spatial_backends)
    {
      for (regularization::ExecutionTargetKind target : targets)
      {
        regularization::RegularizationSelection selection =
          ConstantCpuSelection(spatial);
        selection.axes.continuum_model = model;
        selection.axes.backend.execution = target;
        test.Check(
          regularization::IsRuntimeCombinationSupported(selection) &&
            regularization::ConfigureRegularization(selection).IsConfigured(),
          "all existing ElementL2/pointwise fixture selections remain "
          "accepted");
      }
    }
  }

  for (regularization::SpatialBackendKind spatial :
       {regularization::SpatialBackendKind::DG,
        regularization::SpatialBackendKind::DGSEM})
  {
    for (regularization::ExecutionTargetKind target : targets)
    {
      regularization::RegularizationSelection selection =
        ConstantCpuSelection(spatial);
      selection.axes.auxiliary_discretization =
        regularization::AuxiliaryDiscretizationKind::TraceL2;
      selection.axes.auxiliary_solution =
        regularization::AuxiliarySolutionKind::LocalElimination;
      selection.axes.backend.execution = target;
      test.Check(
        regularization::IsRuntimeCombinationSupported(selection) &&
          regularization::ConfigureRegularization(selection).IsConfigured(),
        "existing trace/local fixture selections remain accepted");
    }
  }

  for (regularization::ExecutionTargetKind target : targets)
  {
    regularization::RegularizationSelection selection =
      ConstantCpuSelection(regularization::SpatialBackendKind::HDG);
    selection.axes.auxiliary_discretization =
      regularization::AuxiliaryDiscretizationKind::HybridizedTrace;
    selection.axes.auxiliary_solution =
      regularization::AuxiliarySolutionKind::GlobalLinearSolve;
    selection.axes.backend.execution = target;
    test.Check(
      regularization::IsRuntimeCombinationSupported(selection) &&
        regularization::ConfigureRegularization(selection).IsConfigured(),
      "existing HDG hybridized/global fixture selections remain accepted");
  }
}

void TestExistingFixtureNumerics(TestContext& test)
{
  PointFixture fixture(3, 2, 4);
  const regularization::RegularizationDevicePolicy policy{2.5,
                                                          -3.0,
                                                          0.5,
                                                          2.0,
                                                          -4.0};
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  test.Check(launcher.IsConfigured() && launcher.HostSelectionCount() == 1,
             "constant fixture configures one static operation table");

  regularization::EvaluatedPointBatch volume_batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x101u);
  regularization::RegularizationOperationOutputs volume_outputs =
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, 0x101u);
  const regularization::RegularizationOperationKind volume_operations[] = {
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    regularization::RegularizationOperationKind::AuxiliarySource,
    regularization::RegularizationOperationKind::VolumeCorrection,
    regularization::RegularizationOperationKind::Diagnostics};
  for (regularization::RegularizationOperationKind operation :
       volume_operations)
  {
    const regularization::RegularizationLaunchResult result =
      launcher.Launch(operation,
                      ValidationContext(),
                      policy,
                      volume_batch,
                      volume_outputs);
    test.Check(result.validation.Ok() && result.launched &&
                 result.host_selection_count == 1,
               "existing volume operation launches statically");
  }

  regularization::EvaluatedPointBatch face_batch =
    fixture.Batch(regularization::EvaluationSiteKind::Face, 0x201u);
  regularization::RegularizationOperationOutputs face_outputs =
    fixture.Outputs(regularization::EvaluationSiteKind::Face, 0x201u);
  for (regularization::RegularizationOperationKind operation :
       {regularization::RegularizationOperationKind::NormalCorrection,
        regularization::RegularizationOperationKind::CharacteristicBounds})
  {
    const regularization::RegularizationLaunchResult result =
      launcher.Launch(operation,
                      ValidationContext(),
                      policy,
                      face_batch,
                      face_outputs);
    test.Check(result.validation.Ok() && result.launched,
               "existing face operation launches statically");
  }

  regularization::EvaluatedPointBatch wall_batch =
    fixture.Batch(regularization::EvaluationSiteKind::Wall, 0x301u);
  regularization::RegularizationOperationOutputs wall_outputs =
    fixture.Outputs(regularization::EvaluationSiteKind::Wall, 0x301u);
  const regularization::RegularizationLaunchResult wall_result =
    launcher.Launch(regularization::RegularizationOperationKind::WallLaw,
                    ValidationContext(),
                    policy,
                    wall_batch,
                    wall_outputs);
  test.Check(wall_result.validation.Ok() && wall_result.launched &&
               launcher.LaunchCount() == 7,
             "existing wall operation and all seven hooks launch");

  const mfem::real_t* coefficient = fixture.coefficient.HostRead();
  const mfem::real_t* source = fixture.source.HostRead();
  const auto volume = mfem::Reshape(fixture.volume.HostRead(),
                                    fixture.dimension,
                                    fixture.state_components,
                                    fixture.points);
  const auto normal = mfem::Reshape(fixture.normal_output.HostRead(),
                                    fixture.state_components,
                                    fixture.points);
  const auto normals = mfem::Reshape(fixture.normal_input.HostRead(),
                                     fixture.dimension,
                                     fixture.points);
  const auto wall = mfem::Reshape(fixture.wall.HostRead(),
                                  fixture.state_components,
                                  fixture.points);
  const auto bounds =
    mfem::Reshape(fixture.bounds.HostRead(), 2, fixture.points);
  const auto diagnostics =
    mfem::Reshape(fixture.diagnostics.HostRead(), 3, fixture.points);
  for (int point = 0; point < fixture.points; ++point)
  {
    test.Near(coefficient[point],
              2.5,
              0.0,
              "existing coefficient numerical output");
    test.Near(source[point], -3.0, 0.0, "existing source numerical output");
    for (int component = 0; component < fixture.state_components; ++component)
    {
      const mfem::real_t scale = static_cast<mfem::real_t>(component + 1);
      mfem::real_t contraction = 0.0;
      for (int direction = 0; direction < fixture.dimension; ++direction)
      {
        test.Near(volume(direction, component, point),
                  scale,
                  0.0,
                  "existing volume numerical output");
        contraction +=
          volume(direction, component, point) * normals(direction, point);
      }
      test.Near(normal(component, point),
                contraction,
                PrecisionTolerance(),
                "existing normal numerical output");
      test.Near(wall(component, point),
                scale,
                PrecisionTolerance(),
                "existing wall numerical output");
    }
    test.Near(bounds(0, point),
              -4.0,
              0.0,
              "existing lower characteristic bound");
    test.Near(bounds(1, point),
              4.0,
              0.0,
              "existing upper characteristic bound");
    test.Near(diagnostics(0, point), 3.0, 0.0, "existing source diagnostic");
    test.Near(diagnostics(1, point),
              20.0,
              0.0,
              "existing correction diagnostic");
    test.Near(diagnostics(2, point),
              10.0,
              PrecisionTolerance(),
              "existing wall diagnostic");
  }

  PointFixture vector_fixture(3, 2, 4, 2);
  regularization::EvaluatedPointBatch vector_batch =
    vector_fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x401u);
  regularization::RegularizationOperationOutputs vector_outputs =
    vector_fixture.Outputs(regularization::EvaluationSiteKind::Volume, 0x401u);
  test.Check(
    launcher
        .Launch(
          regularization::RegularizationOperationKind::AuxiliaryCoefficient,
          ValidationContext(),
          policy,
          vector_batch,
          vector_outputs)
        .launched &&
      launcher
        .Launch(regularization::RegularizationOperationKind::AuxiliarySource,
                ValidationContext(),
                policy,
                vector_batch,
                vector_outputs)
        .launched,
    "existing vector auxiliary coefficient/source extents remain accepted");
}

void TestProbeOperations(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ProbeSelection());
  test.Check(launcher.IsConfigured() && launcher.HostSelectionCount() == 1 &&
               launcher.LaunchCount() == 0,
             "exact probe configures once");

  const regularization::RegularizationDevicePolicy policy{2.5,
                                                          -3.0,
                                                          0.5,
                                                          0.0,
                                                          -4.0};
  PointFixture coefficient_fixture(2, 2, 4);
  PointFixture source_fixture(3, 2, 4);
  PointFixture volume_fixture(4, 2, 4);
  PointFixture normal_fixture(5, 2, 4);
  PointFixture bounds_fixture(6, 2, 4);

  struct ProbeLaunch
  {
    regularization::RegularizationOperationKind operation;
    PointFixture* fixture;
    regularization::EvaluationSiteKind site;
    std::uint64_t point_set_id;
  };
  ProbeLaunch launches[] = {
    {regularization::RegularizationOperationKind::AuxiliaryCoefficient,
     &coefficient_fixture,
     regularization::EvaluationSiteKind::Volume,
     0x1101u},
    {regularization::RegularizationOperationKind::AuxiliarySource,
     &source_fixture,
     regularization::EvaluationSiteKind::Volume,
     0x1102u},
    {regularization::RegularizationOperationKind::VolumeCorrection,
     &volume_fixture,
     regularization::EvaluationSiteKind::Volume,
     0x1103u},
    {regularization::RegularizationOperationKind::NormalCorrection,
     &normal_fixture,
     regularization::EvaluationSiteKind::Face,
     0x2101u},
    {regularization::RegularizationOperationKind::CharacteristicBounds,
     &bounds_fixture,
     regularization::EvaluationSiteKind::Face,
     0x2102u}};

  for (const ProbeLaunch& launch : launches)
  {
    regularization::EvaluatedPointBatch batch =
      launch.fixture->Batch(launch.site, launch.point_set_id, 0);
    batch.trace = {};
    regularization::RegularizationOperationOutputs operation_outputs =
      launch.fixture->Outputs(launch.site, launch.point_set_id);
    operation_outputs.wall_flux = {};
    operation_outputs.diagnostics = {};
    const regularization::RegularizationLaunchResult result =
      launcher.Launch(launch.operation,
                      ValidationContext(0),
                      policy,
                      batch,
                      operation_outputs);
    test.Check(result.validation.Ok() && result.launched &&
                 result.host_selection_count == 1,
               "supported probe operation launches through its static kernel");
  }
  test.Check(
    launcher.LaunchCount() == 5,
    "one probe configuration launches exactly five supported operations");
  test.Check(
    coefficient_fixture.points != source_fixture.points &&
      source_fixture.points != volume_fixture.points &&
      normal_fixture.points != bounds_fixture.points,
    "coefficient/source/volume/face launches use distinct point counts");
  test.Check(
    coefficient_fixture
        .Batch(regularization::EvaluationSiteKind::Volume, 0x1101u)
        .point_set_id !=
      source_fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x1102u)
        .point_set_id,
    "operation batches use distinct point-set identities");

  test.Near(coefficient_fixture.coefficient.HostRead()[0],
            2.5,
            0.0,
            "probe coefficient output");
  test.Near(source_fixture.source.HostRead()[0],
            -2.75,
            PrecisionTolerance(),
            "probe source consumes conservative gradient");
  test.Near(volume_fixture.volume.HostRead()[0],
            1.0,
            0.0,
            "probe volume output");
  test.Near(normal_fixture.normal_output.HostRead()[0],
            1.25,
            PrecisionTolerance(),
            "probe normal output");
  test.Near(bounds_fixture.bounds.HostRead()[0],
            -4.0,
            0.0,
            "probe bounds output");
}

void TestProbeUnsupportedSelections(TestContext& test)
{
  regularization::RegularizationSelection cases[12];
  for (auto& selection : cases)
  {
    selection = ProbeSelection();
  }
  cases[0].axes.backend.execution =
    regularization::ExecutionTargetKind::MFEMDevice;
  cases[1].axes.backend.spatial = regularization::SpatialBackendKind::DGSEM;
  cases[2].axes.backend.spatial = regularization::SpatialBackendKind::HDG;
  cases[3].axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::PointwiseEvaluation;
  cases[4].axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::LocalElimination;
  cases[5].axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::TraceL2;
  cases[6].axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::HybridizedTrace;
  cases[7].axes.gradient_treatment =
    regularization::GradientTreatmentKind::None;
  cases[8].compatibility.time = regularization::TimeTreatmentKind::LaggedStep;
  cases[9].compatibility.time = regularization::TimeTreatmentKind::FullyCoupled;
  cases[10].compatibility.entropy =
    regularization::EntropyTreatmentKind::EntropyConservative;
  cases[11].compatibility.entropy =
    regularization::EntropyTreatmentKind::EntropyStable;

  for (const auto& selection : cases)
  {
    const regularization::ConfiguredRegularizationLauncher launcher =
      regularization::ConfigureRegularization(selection);
    test.Check(!launcher.IsConfigured() && launcher.HostSelectionCount() == 0 &&
                 launcher.ConfigurationValidation().code ==
                   regularization::ValidationCode::UnsupportedCombination &&
                 !regularization::IsRuntimeCombinationSupported(selection),
               "nearby unsupported probe combination fails before launch");
  }

  regularization::RegularizationSelection unknown = ProbeSelection();
  unknown.axes.gradient_treatment =
    static_cast<regularization::GradientTreatmentKind>(255);
  const regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(unknown);
  test.Check(!launcher.IsConfigured() &&
               launcher.ConfigurationValidation().code ==
                 regularization::ValidationCode::UnknownAxisValue,
             "unknown gradient enum yields UnknownAxisValue");
}

void TestUnsupportedCapabilitiesBeforeBuffers(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ProbeSelection());
  PointFixture sentinel(1, 2, 4);
  sentinel.ResetOutputs(73.0);
  const regularization::EvaluatedPointBatch no_buffers{};
  const regularization::RegularizationOperationOutputs no_outputs{};

  regularization::RegularizationLaunchResult result =
    launcher.Launch(regularization::RegularizationOperationKind::WallLaw,
                    {},
                    {},
                    no_buffers,
                    no_outputs);
  test.Check(result.validation.code ==
                 regularization::ValidationCode::UnsupportedCapability &&
               !result.launched && launcher.LaunchCount() == 0,
             "disabled probe wall law fails before buffer validation");

  result =
    launcher.Launch(regularization::RegularizationOperationKind::Diagnostics,
                    {},
                    {},
                    no_buffers,
                    no_outputs);
  test.Check(result.validation.code ==
                 regularization::ValidationCode::UnsupportedCapability &&
               !result.launched && launcher.LaunchCount() == 0 &&
               AllValuesEqual(sentinel.wall, 73.0) &&
               AllValuesEqual(sentinel.diagnostics, 73.0),
             "disabled probe diagnostics fails without touching buffers");
}

void TestPreflightBeforeAccess(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ProbeSelection());
  PointFixture fixture(3, 2, 4);
  fixture.ResetOutputs(81.0);
  regularization::EvaluatedPointBatch batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x5101u);
  regularization::EvaluatedPointFieldMetadata wrong_state =
    fixture.Metadata(3,
                     regularization::EvaluationSiteKind::Volume,
                     0x5101u,
                     11u);
  batch.state =
    regularization::EvaluatedPointFieldBinding::Missing(wrong_state);
  regularization::RegularizationLaunchResult result = launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    ValidationContext(),
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, 0x5101u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidModelShape &&
               !result.launched && AllOutputsEqual(fixture, 81.0),
             "wrong Euler extent beats missing storage and preserves output");

  batch = fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x5102u);
  batch.conservative_state_gradient = regularization::BindEvaluatedPointField(
    fixture.gradient,
    fixture.Metadata(7,
                     regularization::EvaluationSiteKind::Volume,
                     0x5102u,
                     17u));
  result = launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    ValidationContext(),
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, 0x5102u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidModelShape &&
               !result.launched && AllOutputsEqual(fixture, 81.0),
             "wrong conservative-gradient extent fails model preflight");
  batch = fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x5103u);
  batch.dimension = 0;
  result = launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    ValidationContext(),
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, 0x5103u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidModelShape &&
               !result.launched && AllOutputsEqual(fixture, 81.0),
             "probe dimension rule is enforced by model preflight");

  PointFixture vector_auxiliary(2, 2, 4, 2);
  vector_auxiliary.ResetOutputs(82.0);
  batch =
    vector_auxiliary.Batch(regularization::EvaluationSiteKind::Volume, 0x5104u);
  result = launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    ValidationContext(),
    {},
    batch,
    vector_auxiliary.Outputs(regularization::EvaluationSiteKind::Volume,
                             0x5104u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidModelShape &&
               !result.launched && AllOutputsEqual(vector_auxiliary, 82.0),
             "probe scalar-auxiliary rule is enforced by model preflight");

  batch = fixture.Batch(regularization::EvaluationSiteKind::Face, 0x5105u);
  result = launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    ValidationContext(),
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Face, 0x5105u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidModelShape &&
               !result.launched && AllOutputsEqual(fixture, 81.0),
             "probe source rejects a face evaluation site in preflight");

  PointFixture wrong_normal(2, 2, 4);
  wrong_normal.ResetOutputs(83.0);
  batch = wrong_normal.Batch(regularization::EvaluationSiteKind::Face, 0x5106u);
  batch.normal = regularization::BindEvaluatedPointField(
    wrong_normal.normal_input,
    wrong_normal.Metadata(1,
                          regularization::EvaluationSiteKind::Face,
                          0x5106u,
                          31u));
  result = launcher.Launch(
    regularization::RegularizationOperationKind::NormalCorrection,
    ValidationContext(),
    {},
    batch,
    wrong_normal.Outputs(regularization::EvaluationSiteKind::Face, 0x5106u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidModelShape &&
               !result.launched && AllOutputsEqual(wrong_normal, 83.0),
             "probe face-normal extent is enforced by model preflight");
}

void TestZeroBasedStageProvenance(TestContext& test)
{
  test.Check(
    regularization::EvaluatedPointFieldMetadata{}.rk_stage == -1 &&
      regularization::EvaluatedPointBatchMetadata{}.rk_stage == -1 &&
      regularization::EvaluatedPointBatch{}.rk_stage == -1 &&
      regularization::AuxiliaryFieldProvenance{}.rk_stage == -1 &&
      regularization::RegularizationValidationContext{}.expected_rk_stage ==
        -1,
    "every RK-stage default uses the explicit -1 unbound sentinel");

  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ProbeSelection());
  PointFixture fixture(2, 2, 4);
  const std::uint64_t point_set_id = 0x6001u;
  const regularization::RegularizationDevicePolicy policy{
    2.5, -3.0, 0.5, 0.0, -4.0};

  regularization::EvaluatedPointBatch batch = fixture.Batch(
    regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  fixture.ResetOutputs(101.0);
  regularization::RegularizationLaunchResult result = launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    ValidationContext(0),
    policy,
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Volume,
                    point_set_id));
  test.Check(result.validation.Ok() && result.launched &&
               !AllValuesEqual(fixture.source, 101.0),
             "probe launches directly at valid RK stage zero");

  auto expect_source_failure =
    [&](regularization::EvaluatedPointBatch candidate_batch,
        const regularization::RegularizationValidationContext& context,
        regularization::ValidationCode expected,
        const char* message)
  {
    fixture.ResetOutputs(102.0);
    const regularization::RegularizationLaunchResult failed = launcher.Launch(
      regularization::RegularizationOperationKind::AuxiliarySource,
      context,
      policy,
      candidate_batch,
      fixture.Outputs(regularization::EvaluationSiteKind::Volume,
                      point_set_id));
    test.Check(failed.validation.code == expected && !failed.launched &&
                 AllOutputsEqual(fixture, 102.0),
               message);
  };

  batch = fixture.Batch(
    regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  expect_source_failure(batch,
                        ValidationContext(-1),
                        regularization::ValidationCode::InvalidProvenanceStamp,
                        "unbound expected RK stage -1 fails without writes");

  batch = fixture.Batch(
    regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  batch.rk_stage = -1;
  expect_source_failure(batch,
                        ValidationContext(0),
                        regularization::ValidationCode::InvalidProvenanceStamp,
                        "unbound batch RK stage -1 fails without writes");

  batch = fixture.Batch(
    regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     11u,
                     -1));
  expect_source_failure(
    batch,
    ValidationContext(0),
    regularization::ValidationCode::InvalidProvenanceStamp,
    "unbound required-field RK stage -1 fails without writes");

  batch = fixture.Batch(
    regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  expect_source_failure(batch,
                        ValidationContext(1),
                        regularization::ValidationCode::StaleProvenance,
                        "batch stage zero versus expected stage one is stale");

  batch = fixture.Batch(
    regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     11u,
                     1));
  expect_source_failure(
    batch,
    ValidationContext(0),
    regularization::ValidationCode::StaleProvenance,
    "field stage one versus batch stage zero is stale without writes");
}

void TestProducerGenerationAuthority(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher probe =
    regularization::ConfigureRegularization(ProbeSelection());
  PointFixture fixture(2, 2, 4);

  auto expect_probe_failure =
    [&](regularization::RegularizationOperationKind operation,
        regularization::EvaluationSiteKind site,
        std::uint64_t point_set_id,
        const regularization::EvaluatedPointBatch& batch,
        const regularization::RegularizationValidationContext& context,
        regularization::ValidationCode expected,
        const char* message)
  {
    fixture.ResetOutputs(131.0);
    const regularization::RegularizationLaunchResult result = probe.Launch(
      operation,
      context,
      {},
      batch,
      fixture.Outputs(site, point_set_id));
    test.Check(result.validation.code == expected && !result.launched &&
                 AllOutputsEqual(fixture, 131.0),
               message);
  };

  std::uint64_t point_set_id = 0x6201u;
  regularization::EvaluatedPointBatch batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  regularization::RegularizationValidationContext context =
    ValidationContext();
  context.expected_producer_generations.state = 0u;
  expect_probe_failure(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    regularization::EvaluationSiteKind::Volume,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "required state needs a nonzero authoritative producer generation");

  context = ValidationContext();
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     12u));
  expect_probe_failure(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    regularization::EvaluationSiteKind::Volume,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::StaleProvenance,
    "wrong nonzero state producer generation is stale without writes");

  point_set_id = 0x6202u;
  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  context = ValidationContext();
  context.expected_producer_generations.conservative_state_gradient = 0u;
  expect_probe_failure(
    regularization::RegularizationOperationKind::AuxiliarySource,
    regularization::EvaluationSiteKind::Volume,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "required gradient needs a nonzero authoritative producer generation");

  context = ValidationContext();
  batch.conservative_state_gradient =
    regularization::BindEvaluatedPointField(
      fixture.gradient,
      fixture.Metadata(8,
                       regularization::EvaluationSiteKind::Volume,
                       point_set_id,
                       18u));
  expect_probe_failure(
    regularization::RegularizationOperationKind::AuxiliarySource,
    regularization::EvaluationSiteKind::Volume,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::StaleProvenance,
    "wrong nonzero gradient producer generation is stale without writes");

  point_set_id = 0x6203u;
  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  context = ValidationContext();
  context.expected_producer_generations.auxiliary = 0u;
  expect_probe_failure(
    regularization::RegularizationOperationKind::VolumeCorrection,
    regularization::EvaluationSiteKind::Volume,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "required auxiliary needs a nonzero authoritative producer generation");

  context = ValidationContext();
  batch.auxiliary = regularization::BindEvaluatedPointField(
    fixture.auxiliary,
    fixture.Metadata(1,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     24u));
  expect_probe_failure(
    regularization::RegularizationOperationKind::VolumeCorrection,
    regularization::EvaluationSiteKind::Volume,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::StaleProvenance,
    "wrong nonzero auxiliary producer generation is stale without writes");

  point_set_id = 0x6204u;
  batch = fixture.Batch(regularization::EvaluationSiteKind::Face, point_set_id);
  context = ValidationContext();
  context.expected_producer_generations.normal = 0u;
  expect_probe_failure(
    regularization::RegularizationOperationKind::CharacteristicBounds,
    regularization::EvaluationSiteKind::Face,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::InvalidProvenanceStamp,
    "required normal needs a nonzero authoritative producer generation");

  context = ValidationContext();
  batch.normal = regularization::BindEvaluatedPointField(
    fixture.normal_input,
    fixture.Metadata(2,
                     regularization::EvaluationSiteKind::Face,
                     point_set_id,
                     32u));
  expect_probe_failure(
    regularization::RegularizationOperationKind::CharacteristicBounds,
    regularization::EvaluationSiteKind::Face,
    point_set_id,
    batch,
    context,
    regularization::ValidationCode::StaleProvenance,
    "wrong nonzero normal producer generation is stale without writes");

  point_set_id = 0x6205u;
  regularization::ConfiguredRegularizationLauncher constant =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  batch = fixture.Batch(regularization::EvaluationSiteKind::Wall, point_set_id);
  context = ValidationContext();
  context.expected_producer_generations.trace = 0u;
  fixture.ResetOutputs(132.0);
  regularization::RegularizationLaunchResult result = constant.Launch(
    regularization::RegularizationOperationKind::WallLaw,
    context,
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Wall, point_set_id));
  test.Check(
    result.validation.code ==
        regularization::ValidationCode::InvalidProvenanceStamp &&
      !result.launched && AllOutputsEqual(fixture, 132.0),
    "required trace needs a nonzero authoritative producer generation");

  context = ValidationContext();
  batch.trace = regularization::BindEvaluatedPointField(
    fixture.trace,
    fixture.Metadata(1,
                     regularization::EvaluationSiteKind::Wall,
                     point_set_id,
                     30u));
  fixture.ResetOutputs(133.0);
  result = constant.Launch(
    regularization::RegularizationOperationKind::WallLaw,
    context,
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Wall, point_set_id));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::StaleProvenance &&
               !result.launched && AllOutputsEqual(fixture, 133.0),
             "wrong nonzero trace producer generation is stale without writes");

  point_set_id = 0x6206u;
  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  test.Check(batch.state.ProducerGeneration() == 11u &&
               batch.conservative_state_gradient.ProducerGeneration() == 17u &&
               batch.auxiliary.ProducerGeneration() == 23u &&
               batch.state.ProducerGeneration() !=
                 batch.conservative_state_gradient.ProducerGeneration() &&
               batch.conservative_state_gradient.ProducerGeneration() !=
                 batch.auxiliary.ProducerGeneration(),
             "required field producers retain unequal independent counters");
  fixture.ResetOutputs(134.0);
  result = probe.Launch(
    regularization::RegularizationOperationKind::AuxiliarySource,
    ValidationContext(0),
    {},
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, point_set_id));
  test.Check(result.validation.Ok() && result.launched &&
               !AllValuesEqual(fixture.source, 134.0),
             "valid unequal producer generations launch at stage zero");

  point_set_id = 0x6207u;
  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id, 0);
  batch.conservative_state_gradient = {};
  batch.trace = {};
  batch.normal = {};
  context = ValidationContext(0);
  context.expected_producer_generations.conservative_state_gradient = 0u;
  context.expected_producer_generations.trace = 0u;
  context.expected_producer_generations.normal = 0u;
  fixture.ResetOutputs(135.0);
  const regularization::RegularizationDevicePolicy policy{
    2.5, -3.0, 0.5, 0.0, -4.0};
  result = probe.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    context,
    policy,
    batch,
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, point_set_id));
  test.Check(result.validation.Ok() && result.launched &&
               fixture.coefficient.HostRead()[0] == 2.5,
             "absent unrequired fields need no expected producer generation");
}

void TestStampAndBatchAgreementFailures(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ProbeSelection());
  PointFixture fixture(3, 2, 4);
  const std::uint64_t point_set_id = 0x6101u;

  auto run_source = [&](regularization::EvaluatedPointBatch batch,
                        regularization::ValidationCode expected,
                        const char* message)
  {
    fixture.ResetOutputs(91.0);
    const regularization::RegularizationLaunchResult result = launcher.Launch(
      regularization::RegularizationOperationKind::AuxiliarySource,
      ValidationContext(),
      {},
      batch,
      fixture.Outputs(regularization::EvaluationSiteKind::Volume,
                      point_set_id));
    test.Check(result.validation.code == expected && !result.launched &&
                 AllOutputsEqual(fixture, 91.0),
               message);
  };

  regularization::EvaluatedPointBatch batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.rk_stage = -1;
  run_source(batch,
             regularization::ValidationCode::InvalidProvenanceStamp,
             "unbound batch RK stage -1 fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.state_generation = 0u;
  run_source(batch,
             regularization::ValidationCode::InvalidProvenanceStamp,
             "zero batch state generation fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     0u));
  run_source(batch,
             regularization::ValidationCode::InvalidProvenanceStamp,
             "zero required producer generation fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     11u,
                     3));
  run_source(batch,
             regularization::ValidationCode::StaleProvenance,
             "stale required field stage fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     11u,
                     2,
                     20u));
  run_source(batch,
             regularization::ValidationCode::StaleProvenance,
             "stale required field state generation fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.auxiliary = regularization::BindEvaluatedPointField(
    fixture.auxiliary,
    fixture.Metadata(1,
                     regularization::EvaluationSiteKind::Face,
                     point_set_id,
                     23u));
  run_source(batch,
             regularization::ValidationCode::InvalidEvaluationSite,
             "mixed required-field site fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.auxiliary = regularization::BindEvaluatedPointField(
    fixture.auxiliary,
    fixture.Metadata(1,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     23u,
                     2,
                     19u,
                     fixture.points + 1));
  run_source(batch,
             regularization::ValidationCode::InvalidPointCount,
             "mixed required-field point count fails without writes");

  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.auxiliary = regularization::BindEvaluatedPointField(
    fixture.auxiliary,
    fixture.Metadata(1,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id + 1u,
                     23u));
  run_source(batch,
             regularization::ValidationCode::InvalidPointSetIdentity,
             "mixed required-field point-set identity fails without writes");
}

void TestAliasAndOverflowProtection(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher probe =
    regularization::ConfigureRegularization(ProbeSelection());
  PointFixture fixture(1, 2, 4);
  const std::uint64_t point_set_id = 0x7101u;
  regularization::EvaluatedPointBatch batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  regularization::RegularizationOperationOutputs outputs =
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, point_set_id);
  outputs.auxiliary_coefficient = regularization::BindEvaluatedPointOutput(
    fixture.auxiliary,
    fixture.points,
    fixture.auxiliary_components,
    fixture.dimension,
    regularization::EvaluationSiteKind::Volume,
    point_set_id);
  regularization::RegularizationLaunchResult result = probe.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    ValidationContext(),
    {},
    batch,
    outputs);
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidBufferAliasing &&
               !result.launched && AllValuesEqual(fixture.auxiliary, 2.0),
             "per-operation writable/input alias is rejected");

  const int huge = std::numeric_limits<int>::max();
  batch =
    fixture.Batch(regularization::EvaluationSiteKind::Volume, point_set_id);
  batch.point_count = huge;
  batch.state = regularization::BindEvaluatedPointField(
    fixture.state,
    fixture.Metadata(4,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     11u,
                     2,
                     19u,
                     huge));
  batch.auxiliary = regularization::BindEvaluatedPointField(
    fixture.auxiliary,
    fixture.Metadata(1,
                     regularization::EvaluationSiteKind::Volume,
                     point_set_id,
                     23u,
                     2,
                     19u,
                     huge));
  outputs =
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, point_set_id);
  outputs.auxiliary_coefficient = regularization::BindEvaluatedPointOutput(
    fixture.coefficient,
    huge,
    1,
    fixture.dimension,
    regularization::EvaluationSiteKind::Volume,
    point_set_id);
  result = probe.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    ValidationContext(),
    {},
    batch,
    outputs);
  test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidComponentCount &&
               !result.launched,
             "per-operation evaluated input overflow is rejected");

  regularization::ConfiguredRegularizationLauncher constant =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  PointFixture shape_fixture(1, 3, 1);
  batch =
    shape_fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x7102u);
  batch.state = regularization::BindEvaluatedPointField(
    shape_fixture.state,
    shape_fixture.Metadata(huge,
                           regularization::EvaluationSiteKind::Volume,
                           0x7102u,
                           11u));
  outputs = {};
  result = constant.Launch(
    regularization::RegularizationOperationKind::VolumeCorrection,
    ValidationContext(),
    {},
    batch,
    outputs);
  test.Check(
    result.validation.code ==
        regularization::ValidationCode::InvalidComponentCount &&
      !result.launched,
    "per-operation output-component overflow is rejected before access");
}

void TestUnknownOperationAndDeviceAvailability(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  const regularization::RegularizationLaunchResult unknown = launcher.Launch(
    static_cast<regularization::RegularizationOperationKind>(255),
    {},
    {},
    {},
    {});
  test.Check(unknown.validation.code ==
                 regularization::ValidationCode::UnknownAxisValue &&
               !unknown.launched,
             "unknown operation fails before launch");

  regularization::RegularizationSelection device_selection =
    ConstantCpuSelection();
  device_selection.axes.backend.execution =
    regularization::ExecutionTargetKind::MFEMDevice;
  regularization::ConfiguredRegularizationLauncher device_launcher =
    regularization::ConfigureRegularization(device_selection);
  PointFixture fixture(1, 2, 4);
  const auto result = device_launcher.Launch(
    regularization::RegularizationOperationKind::AuxiliaryCoefficient,
    ValidationContext(),
    {},
    fixture.Batch(regularization::EvaluationSiteKind::Volume, 0x8101u),
    fixture.Outputs(regularization::EvaluationSiteKind::Volume, 0x8101u));
  test.Check(result.validation.code ==
                 regularization::ValidationCode::BackendUnavailable &&
               !result.launched,
             "unavailable MFEM device fails before storage acquisition");
}

} // namespace

int main()
{
  mfem::Device device("cpu");
  TestContext test;
  TestAuxiliaryRestartTruthfulness(test);
  TestExistingFixtureSelections(test);
  TestExistingFixtureNumerics(test);
  TestProbeOperations(test);
  TestProbeUnsupportedSelections(test);
  TestUnsupportedCapabilitiesBeforeBuffers(test);
  TestPreflightBeforeAccess(test);
  TestZeroBasedStageProvenance(test);
  TestProducerGenerationAuthority(test);
  TestStampAndBatchAgreementFailures(test);
  TestAliasAndOverflowProtection(test);
  TestUnknownOperationAndDeviceAvailability(test);

  if (test.Failures() == 0)
  {
    std::cout << "[PASS] evaluated-point static regularization contract\n";
  }
  return test.Failures() == 0 ? 0 : 1;
}
