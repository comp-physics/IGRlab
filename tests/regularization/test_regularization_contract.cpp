#include "physics/regularization/regularization.hpp"

#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

namespace regularization = dg_igr::regularization;

namespace
{

constexpr regularization::AuxiliaryFieldIdentity kScalarIdentity{
  0x1001u, 0x2001u, 0x3001u};
constexpr regularization::AuxiliaryFieldIdentity kScalarIdentityCopy{
  0x1001u, 0x2001u, 0x3001u};
constexpr regularization::AuxiliaryFieldIdentity kTraceIdentity{
  0x1002u, 0x2002u, 0x3002u};

static_assert(kScalarIdentity == kScalarIdentityCopy,
              "field identity equality must be compile-time stable");
static_assert(kScalarIdentity != kTraceIdentity,
              "distinct restart/output identities must remain distinct");
static_assert(
  std::is_trivially_copyable<regularization::RegularizationDevicePolicy>::value,
  "fixture policy must remain trivially copyable");
static_assert(
  std::is_trivially_copyable<regularization::RegularizationPointInput>::value,
  "multi-component point view must remain trivially copyable");
static_assert(
  std::is_trivially_copyable<regularization::AuxiliaryFieldBinding>::value,
  "typed field binding must remain trivially copyable");
static_assert(
  regularization::HasAllRegularizationCapabilities(
    regularization::RegularizationTraits<
      regularization::ZeroFixtureModelTag>::capabilities),
  "registered traits must enable all six hooks");
static_assert(
  !regularization::HasAllRegularizationCapabilities(
    {true, true, true, true, true, false}),
  "an incomplete hook declaration must fail the capability contract");
static_assert(
  std::is_same<decltype(regularization::RegularizationAxes{}.backend),
               regularization::SpatialExecutionBackend>::value,
  "the fourth axis must be a spatial/execution backend product");
static_assert(
  regularization::HasRegularizationTraits<
    regularization::ZeroFixtureModelTag>::value,
  "zero fixture trait must be available");
static_assert(
  regularization::HasRegularizationTraits<
    regularization::ConstantFixtureModelTag>::value,
  "constant fixture trait must be available");
static_assert(
  !regularization::HasRegularizationTraits<
    regularization::UnregisteredFixtureModelTag>::value,
  "unknown model trait must be unavailable");
static_assert(regularization::SupportedCpuFixtureCombination::value,
              "DG CPU fixture combination must remain available");
static_assert(regularization::SupportedDeviceFixtureCombination::value,
              "DGSEM device fixture combination must remain available");
static_assert(regularization::SupportedHDGDeviceFixtureCombination::value,
              "HDG device fixture combination must remain available");
static_assert(regularization::SupportedHDGHybridizedCombination::value,
              "HDG hybridized fixture combination must remain available");
static_assert(!regularization::UnsupportedDGHybridizedCombination::value,
              "DG hybridized mismatch must remain unavailable");
static_assert(!regularization::UnsupportedSolveCombination::value,
              "unsupported solve combination must remain unavailable");
static_assert(!regularization::UnsupportedTimeCombination::value,
              "unsupported time combination must remain unavailable");
static_assert(!regularization::UnsupportedEntropyCombination::value,
              "unsupported entropy combination must remain unavailable");

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

regularization::AuxiliaryFieldCapabilities VolumeCapabilities()
{
  using regularization::AuxiliaryCapability;
  return {regularization::CapabilityBit(AuxiliaryCapability::DeviceAccessible) |
          regularization::CapabilityBit(AuxiliaryCapability::Restartable) |
          regularization::CapabilityBit(AuxiliaryCapability::OutputVisible) |
          regularization::CapabilityBit(AuxiliaryCapability::StageLocal)};
}

regularization::AuxiliaryFieldCapabilities TraceCapabilities()
{
  using regularization::AuxiliaryCapability;
  return {VolumeCapabilities().bits |
          regularization::CapabilityBit(AuxiliaryCapability::TraceCoupled)};
}

void TestTypedDescriptors(TestContext& test)
{
  constexpr int points = 3;
  constexpr int dimension = 2;
  const regularization::AuxiliaryFieldProvenance provenance{2, 19u, 7u};

  mfem::Vector scalar_storage(points);
  mfem::Vector vector_storage(dimension * points);
  mfem::Vector tensor_storage(dimension * dimension * points);
  mfem::Vector trace_storage(points);

  regularization::ScalarAuxiliaryFieldDescriptor scalar(
    scalar_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::StateOwned,
    provenance,
    kScalarIdentity,
    VolumeCapabilities());
  regularization::VectorAuxiliaryFieldDescriptor vector(
    vector_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    {0x1010u, 0x2010u, 0x3010u},
    VolumeCapabilities());
  regularization::TensorAuxiliaryFieldDescriptor tensor(
    tensor_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    {0x1020u, 0x2020u, 0x3020u},
    VolumeCapabilities());
  regularization::TraceAuxiliaryFieldDescriptor trace(
    trace_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::BorrowedReadOnly,
    provenance,
    kTraceIdentity,
    TraceCapabilities());

  scalar_storage = 2.0;
  const mfem::real_t* scalar_read = scalar.Read(false);
  test.Near(scalar_read[1], 2.0, 0.0, "scalar descriptor MFEM Memory read");

  auto vector_view = mfem::Reshape(vector.Write(false), dimension, points);
  for (int point = 0; point < points; ++point)
  {
    for (int d = 0; d < dimension; ++d)
    {
      vector_view(d, point) = static_cast<mfem::real_t>(10 * point + d);
    }
  }

  tensor_storage = 0.0;
  auto tensor_view =
    mfem::Reshape(tensor.ReadWrite(false), dimension, dimension, points);
  tensor_view(1, 0, 2) = 9.0;
  trace_storage = 4.0;
  const mfem::real_t* trace_read = trace.Read(false);

  const regularization::AuxiliaryFieldBinding scalar_binding =
    regularization::BindAuxiliaryField(scalar);
  const regularization::AuxiliaryFieldBinding vector_binding =
    regularization::BindAuxiliaryField(vector);
  const regularization::AuxiliaryFieldBinding tensor_binding =
    regularization::BindAuxiliaryField(tensor);
  const regularization::AuxiliaryFieldBinding trace_binding =
    regularization::BindAuxiliaryField(trace);

  test.Check(scalar.Components() == 1 && scalar.HasExpectedSize(),
             "typed scalar descriptor shape");
  test.Check(vector.Components() == dimension && vector.HasExpectedSize(),
             "typed vector descriptor shape");
  test.Check(tensor.Components() == dimension * dimension &&
               tensor.HasExpectedSize(),
             "typed tensor descriptor shape");
  test.Check(trace.Topology() == regularization::AuxiliaryFieldTopology::Trace &&
               trace.HasExpectedSize(),
             "typed trace descriptor topology");
  test.Check(scalar.Ownership() ==
               regularization::AuxiliaryFieldOwnership::StateOwned &&
               trace.Ownership() ==
                 regularization::AuxiliaryFieldOwnership::BorrowedReadOnly,
             "descriptor ownership metadata");
  test.Check(regularization::SameStageState(scalar.Provenance(),
                                             trace.Provenance()),
             "descriptor RK-stage/state-generation provenance");
  test.Check(scalar.Identity() == kScalarIdentity &&
               trace.Identity() == kTraceIdentity,
             "descriptor restart/output identity");
  test.Check(scalar.Capabilities().Has(
               regularization::AuxiliaryCapability::Restartable) &&
               trace.Capabilities().Has(
                 regularization::AuxiliaryCapability::TraceCoupled),
             "descriptor capability metadata");
  test.Check(scalar_binding.components == 1 &&
               scalar_binding.value_kind ==
                 regularization::AuxiliaryValueKind::Scalar &&
               vector_binding.components == dimension &&
               vector_binding.value_kind ==
                 regularization::AuxiliaryValueKind::Vector &&
               tensor_binding.components == dimension * dimension &&
               tensor_binding.value_kind ==
                 regularization::AuxiliaryValueKind::Tensor &&
               trace_binding.topology ==
                 regularization::AuxiliaryFieldTopology::Trace,
             "typed descriptors retain shape through launch bindings");
  test.Near(vector_storage.HostRead()[5], 21.0, 0.0,
            "vector descriptor MFEM Reshape write");
  test.Near(tensor_storage.HostRead()[1 + dimension * dimension * 2],
            9.0,
            0.0,
            "tensor descriptor MFEM ReadWrite/Reshape");
  test.Near(trace_read[0], 4.0, 0.0, "trace descriptor MFEM Memory read");
}

struct FixtureBuffers
{
  static constexpr int points = 3;
  static constexpr int dimension = 2;
  static constexpr int state_components = 4;
  static constexpr int correction_components = 4;

  mfem::Vector state = mfem::Vector(state_components * points);
  mfem::Vector auxiliary_storage = mfem::Vector(points);
  mfem::Vector trace_storage = mfem::Vector(points);
  mfem::Vector state_gradient =
    mfem::Vector(state_components * dimension * points);
  mfem::Vector normals = mfem::Vector(dimension * points);
  mfem::Vector coefficient = mfem::Vector(points);
  mfem::Vector source = mfem::Vector(points);
  mfem::Vector volume =
    mfem::Vector(correction_components * dimension * points);
  mfem::Vector normal = mfem::Vector(correction_components * points);
  mfem::Vector wall = mfem::Vector(correction_components * points);
  mfem::Vector bounds = mfem::Vector(2 * points);
  mfem::Vector diagnostics = mfem::Vector(3 * points);

  regularization::AuxiliaryFieldProvenance provenance{2, 19u, 7u};
  regularization::ScalarAuxiliaryFieldDescriptor auxiliary{
    auxiliary_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::StateOwned,
    provenance,
    kScalarIdentity,
    VolumeCapabilities()};
  regularization::TraceAuxiliaryFieldDescriptor trace{
    trace_storage,
    points,
    dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    kTraceIdentity,
    TraceCapabilities()};

  FixtureBuffers()
  {
    auto state_view =
      mfem::Reshape(state.HostWrite(), state_components, points);
    mfem::real_t* trace_values = trace_storage.HostWrite();
    for (int point = 0; point < points; ++point)
    {
      state_view(0, point) = static_cast<mfem::real_t>(point + 1);
      for (int component = 1; component < state_components; ++component)
      {
        state_view(component, point) =
          static_cast<mfem::real_t>(10 * component + point);
      }
      trace_values[point] = state_view(0, point) + 0.5;
    }
    auxiliary_storage = 2.0;
    state_gradient = 0.0;

    auto normal_view =
      mfem::Reshape(normals.HostWrite(), dimension, points);
    normal_view(0, 0) = 1.0;
    normal_view(1, 0) = 0.0;
    normal_view(0, 1) = 0.0;
    normal_view(1, 1) = 1.0;
    normal_view(0, 2) = 0.6;
    normal_view(1, 2) = 0.8;
    ResetOutputs(0.0);
  }

  void ResetOutputs(mfem::real_t value)
  {
    coefficient = value;
    source = value;
    volume = value;
    normal = value;
    wall = value;
    bounds = value;
    diagnostics = value;
  }

  regularization::RegularizationLaunchBuffers LaunchBuffers()
  {
    return {points,
            dimension,
            state_components,
            &state,
            regularization::BindAuxiliaryField(auxiliary),
            regularization::BindAuxiliaryField(trace),
            &state_gradient,
            &normals,
            &coefficient,
            &source,
            regularization::BindVolumeFluxCorrection(
              volume, correction_components, dimension, points),
            regularization::BindStateVectorCorrection(
              normal, correction_components, points),
            regularization::BindStateVectorCorrection(
              wall, correction_components, points),
            &bounds,
            &diagnostics};
  }
};

regularization::RegularizationSelection ConstantCpuSelection(
  regularization::SpatialBackendKind spatial =
    regularization::SpatialBackendKind::DG)
{
  regularization::RegularizationSelection selection;
  selection.axes.continuum_model =
    regularization::ContinuumModelKind::ConstantFixture;
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

regularization::RegularizationValidationContext ValidationContext()
{
  return {false, 2, 19u};
}

void TestSystemShapedConstantCorrection(TestContext& test)
{
  FixtureBuffers fixture;
  const regularization::RegularizationDevicePolicy policy{
    2.5, -3.0, 0.5, 2.0, -4.0};
  const regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(ConstantCpuSelection(),
                                          ValidationContext(),
                                          policy,
                                          fixture.LaunchBuffers());

  test.Check(result.validation.Ok() && result.launched,
             "multi-component constant fixture host dispatch launches");
  test.Check(result.host_selection_count == 1,
             "host selection occurs exactly once before static execution");

  const mfem::real_t* coefficient = fixture.coefficient.HostRead();
  const mfem::real_t* source = fixture.source.HostRead();
  const auto volume = mfem::Reshape(fixture.volume.HostRead(),
                                     FixtureBuffers::dimension,
                                     FixtureBuffers::correction_components,
                                     FixtureBuffers::points);
  const auto normals = mfem::Reshape(fixture.normals.HostRead(),
                                      FixtureBuffers::dimension,
                                      FixtureBuffers::points);
  const auto normal = mfem::Reshape(fixture.normal.HostRead(),
                                     FixtureBuffers::correction_components,
                                     FixtureBuffers::points);
  const auto wall = mfem::Reshape(fixture.wall.HostRead(),
                                   FixtureBuffers::correction_components,
                                   FixtureBuffers::points);
  const auto bounds =
    mfem::Reshape(fixture.bounds.HostRead(), 2, FixtureBuffers::points);
  const auto diagnostics =
    mfem::Reshape(fixture.diagnostics.HostRead(), 3, FixtureBuffers::points);

  for (int point = 0; point < FixtureBuffers::points; ++point)
  {
    test.Near(coefficient[point], 2.5, 0.0,
              "constant auxiliary coefficient behavior");
    test.Near(source[point], -3.0, 0.0,
              "constant auxiliary source behavior");
    for (int component = 0;
         component < FixtureBuffers::correction_components;
         ++component)
    {
      const mfem::real_t expected_component =
        static_cast<mfem::real_t>(component + 1);
      mfem::real_t tensor_dot_normal = 0.0;
      for (int direction = 0; direction < FixtureBuffers::dimension;
           ++direction)
      {
        test.Near(volume(direction, component, point),
                  expected_component,
                  0.0,
                  "equation-by-direction volume correction behavior");
        tensor_dot_normal += volume(direction, component, point) *
                             normals(direction, point);
      }
      test.Near(normal(component, point),
                tensor_dot_normal,
                PrecisionTolerance(),
                "normal vector equals volume-tensor contraction");
      test.Near(wall(component, point),
                expected_component,
                0.0,
                "equation-vector wall-law behavior");
    }
    test.Near(bounds(0, point), -4.0, 0.0,
              "lower characteristic bound");
    test.Near(bounds(1, point), 4.0, 0.0,
              "upper characteristic bound");
    test.Near(diagnostics(0, point), 3.0, 0.0,
              "source diagnostic behavior");
    test.Near(diagnostics(1, point), 20.0, 0.0,
              "system volume diagnostic behavior");
    test.Near(diagnostics(2, point), 10.0, 0.0,
              "system wall diagnostic behavior");
  }
}

void TestConfiguredLauncherReuse(TestContext& test)
{
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  test.Check(launcher.IsConfigured() && launcher.HostSelectionCount() == 1 &&
               launcher.LaunchCount() == 0,
             "configuration traverses the host selection exactly once");

  FixtureBuffers first_fixture;
  FixtureBuffers second_fixture;
  regularization::RegularizationLaunchBuffers first_buffers =
    first_fixture.LaunchBuffers();
  regularization::RegularizationLaunchBuffers second_buffers =
    second_fixture.LaunchBuffers();
  second_buffers.auxiliary.provenance = {3, 20u, 8u};
  second_buffers.trace.provenance = {3, 20u, 9u};
  const regularization::RegularizationDevicePolicy policy{
    1.5, -2.0, 0.25, 1.0, 3.0};

  const regularization::RegularizationLaunchResult first =
    launcher.Launch(ValidationContext(), policy, first_buffers);
  const regularization::RegularizationValidationContext second_context{
    false, 3, 20u};
  const regularization::RegularizationLaunchResult second =
    launcher.Launch(second_context, policy, second_buffers);

  test.Check(first.validation.Ok() && first.launched &&
               second.validation.Ok() && second.launched,
             "one configured launcher accepts two valid buffer/provenance sets");
  test.Check(first.host_selection_count == 1 &&
               second.host_selection_count == 1 &&
               launcher.HostSelectionCount() == 1 &&
               launcher.LaunchCount() == 2,
             "two launches do not retraverse model or axis selection");
  test.Near(first_fixture.coefficient.HostRead()[0],
            static_cast<mfem::real_t>(1.5),
            0.0,
            "first configured launch writes its distinct buffers");
  test.Near(second_fixture.coefficient.HostRead()[0],
            static_cast<mfem::real_t>(1.5),
            0.0,
            "second configured launch writes its distinct buffers");
}

void TestAllSpatialBackends(TestContext& test)
{
  const regularization::SpatialBackendKind backends[] = {
    regularization::SpatialBackendKind::DG,
    regularization::SpatialBackendKind::DGSEM,
    regularization::SpatialBackendKind::HDG};
  for (regularization::SpatialBackendKind backend : backends)
  {
    FixtureBuffers fixture;
    const regularization::RegularizationLaunchResult result =
      regularization::ExecuteRegularization(ConstantCpuSelection(backend),
                                            ValidationContext(),
                                            {},
                                            fixture.LaunchBuffers());
    test.Check(result.validation.Ok() && result.launched &&
                 result.host_selection_count == 1,
               "DG/DGSEM/HDG CPU host selection reaches static execution");
    regularization::RegularizationSelection device_matrix =
      ConstantCpuSelection(backend);
    device_matrix.axes.backend.execution =
      regularization::ExecutionTargetKind::MFEMDevice;
    test.Check(regularization::IsRuntimeCombinationSupported(device_matrix),
               "DG/DGSEM/HDG device combination is represented");
  }

  regularization::RegularizationSelection device =
    ConstantCpuSelection(regularization::SpatialBackendKind::DGSEM);
  device.axes.backend.execution =
    regularization::ExecutionTargetKind::MFEMDevice;
  test.Check(regularization::IsRuntimeCombinationSupported(device),
             "DGSEM device combination is represented in runtime matrix");
}

void TestVectorAuxiliaryBinding(TestContext& test)
{
  FixtureBuffers fixture;
  mfem::Vector vector_auxiliary_storage(
    FixtureBuffers::dimension * FixtureBuffers::points);
  vector_auxiliary_storage = 2.0;
  regularization::VectorAuxiliaryFieldDescriptor vector_auxiliary(
    vector_auxiliary_storage,
    FixtureBuffers::points,
    FixtureBuffers::dimension,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    fixture.provenance,
    {0x1100u, 0x2100u, 0x3100u},
    VolumeCapabilities());
  mfem::Vector vector_coefficient(
    FixtureBuffers::dimension * FixtureBuffers::points);
  mfem::Vector vector_source(
    FixtureBuffers::dimension * FixtureBuffers::points);
  regularization::RegularizationLaunchBuffers buffers = fixture.LaunchBuffers();
  buffers.auxiliary = regularization::BindAuxiliaryField(vector_auxiliary);
  buffers.coefficient = &vector_coefficient;
  buffers.source = &vector_source;

  const regularization::RegularizationDevicePolicy policy{
    1.25, -0.75, 0.5, 0.0, 0.0};
  const regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(
      ConstantCpuSelection(regularization::SpatialBackendKind::DGSEM),
      ValidationContext(),
      policy,
      buffers);
  test.Check(result.validation.Ok() && result.launched,
             "vector auxiliary descriptor binds without public API change");
  const auto coefficient =
    mfem::Reshape(vector_coefficient.HostRead(),
                  FixtureBuffers::dimension,
                  FixtureBuffers::points);
  const auto source = mfem::Reshape(vector_source.HostRead(),
                                     FixtureBuffers::dimension,
                                     FixtureBuffers::points);
  for (int point = 0; point < FixtureBuffers::points; ++point)
  {
    for (int component = 0; component < FixtureBuffers::dimension; ++component)
    {
      test.Near(coefficient(component, point),
                1.25,
                0.0,
                "vector auxiliary coefficient component");
      test.Near(source(component, point),
                -0.75,
                0.0,
                "vector auxiliary source component");
    }
  }
}

void TestZeroBehavior(TestContext& test)
{
  FixtureBuffers fixture;
  fixture.ResetOutputs(0.0);
  regularization::RegularizationSelection selection = ConstantCpuSelection();
  selection.axes.continuum_model =
    regularization::ContinuumModelKind::ZeroFixture;
  const regularization::RegularizationDevicePolicy nonzero_policy{
    9.0, 8.0, 7.0, 6.0, 5.0};
  const regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(selection,
                                          ValidationContext(),
                                          nonzero_policy,
                                          fixture.LaunchBuffers());
  test.Check(result.validation.Ok() && result.launched,
             "multi-component zero fixture launches through static dispatch");

  const mfem::Vector* outputs[] = {&fixture.coefficient,
                                   &fixture.source,
                                   &fixture.volume,
                                   &fixture.normal,
                                   &fixture.wall,
                                   &fixture.bounds,
                                   &fixture.diagnostics};
  for (const mfem::Vector* output : outputs)
  {
    const mfem::real_t* values = output->HostRead();
    for (int i = 0; i < output->Size(); ++i)
    {
      test.Near(values[i], 0.0, 0.0, "zero fixture output behavior");
    }
  }
}

void TestSupportMatrixAndFailures(TestContext& test)
{
  FixtureBuffers fixture;
  fixture.ResetOutputs(17.0);
  const regularization::RegularizationDevicePolicy policy{};

  regularization::RegularizationSelection hybrid = ConstantCpuSelection(
    regularization::SpatialBackendKind::HDG);
  hybrid.axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::HybridizedTrace;
  hybrid.axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::GlobalLinearSolve;
  test.Check(regularization::IsRuntimeCombinationSupported(hybrid),
             "HDG hybridized/global combination is supported");
  regularization::RegularizationLaunchResult result =
    regularization::ExecuteRegularization(hybrid,
                                          ValidationContext(),
                                          policy,
                                          fixture.LaunchBuffers());
  test.Check(result.validation.Ok() && result.launched,
             "HDG hybridized/global combination reaches static execution");
  fixture.ResetOutputs(17.0);

  regularization::RegularizationSelection unsupported = hybrid;
  unsupported.axes.backend.spatial = regularization::SpatialBackendKind::DG;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(!result.launched && result.host_selection_count == 0 &&
               result.validation.code ==
                 regularization::ValidationCode::UnsupportedCombination,
             "DG/HDG-solve mismatch fails before dispatch");

  unsupported = ConstantCpuSelection();
  unsupported.compatibility.entropy =
    regularization::EntropyTreatmentKind::EntropyStable;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(!result.launched && result.host_selection_count == 0 &&
               result.validation.code ==
                 regularization::ValidationCode::UnsupportedCombination,
             "entropy-incompatible selection fails before dispatch");
  test.Near(fixture.source.HostRead()[0],
            17.0,
            0.0,
            "validation failure leaves output untouched");

  unsupported = ConstantCpuSelection();
  unsupported.compatibility.time =
    regularization::TimeTreatmentKind::LaggedStep;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnsupportedCombination &&
               !result.launched,
             "time-incompatible selection fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.auxiliary_discretization =
    regularization::AuxiliaryDiscretizationKind::TraceL2;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnsupportedCombination &&
               !result.launched,
             "discretization/solve mismatch fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.continuum_model =
    regularization::ContinuumModelKind::ZeroFixture;
  unsupported.axes.auxiliary_solution =
    regularization::AuxiliarySolutionKind::GlobalLinearSolve;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnsupportedCombination &&
               !result.launched,
             "model/solve mismatch fails before launch");

  unsupported = ConstantCpuSelection(
    regularization::SpatialBackendKind::DGSEM);
  unsupported.axes.backend.execution =
    regularization::ExecutionTargetKind::MFEMDevice;
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::BackendUnavailable &&
               !result.launched,
             "unavailable execution target fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.backend.spatial =
    static_cast<regularization::SpatialBackendKind>(255);
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnknownAxisValue &&
               !result.launched,
             "unknown spatial backend fails before launch");

  unsupported = ConstantCpuSelection();
  unsupported.axes.backend.execution =
    static_cast<regularization::ExecutionTargetKind>(255);
  result = regularization::ExecuteRegularization(unsupported,
                                                 ValidationContext(),
                                                 policy,
                                                 fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::UnknownAxisValue &&
               !result.launched,
             "unknown execution target fails before launch");
}

void TestOnePointBufferAliasing(TestContext& test)
{
  constexpr int points = 1;
  constexpr int dimension = 1;
  constexpr int components = 1;
  mfem::Vector state(points);
  mfem::Vector auxiliary_storage(points);
  mfem::Vector trace_storage(points);
  mfem::Vector state_gradient(points);
  mfem::Vector normals(points);
  mfem::Vector coefficient(points);
  mfem::Vector source(points);
  mfem::Vector volume(points);
  mfem::Vector normal(points);
  mfem::Vector wall(points);
  mfem::Vector bounds(2 * points);
  mfem::Vector diagnostics(3 * points);
  state = 4.0;
  auxiliary_storage = 2.0;
  trace_storage = 4.5;
  state_gradient = 0.0;
  normals = 1.0;
  coefficient = 0.0;
  source = 0.0;
  volume = 0.0;
  normal = 0.0;
  wall = 0.0;
  bounds = 0.0;
  diagnostics = 0.0;

  const regularization::AuxiliaryFieldProvenance provenance{2, 19u, 7u};
  const regularization::AuxiliaryFieldBinding auxiliary{
    &auxiliary_storage,
    points,
    dimension,
    components,
    regularization::AuxiliaryValueKind::Scalar,
    regularization::AuxiliaryFieldTopology::Volume,
    regularization::AuxiliaryFieldOwnership::StateOwned,
    provenance,
    kScalarIdentity,
    VolumeCapabilities()};
  const regularization::AuxiliaryFieldBinding trace{
    &trace_storage,
    points,
    dimension,
    components,
    regularization::AuxiliaryValueKind::Scalar,
    regularization::AuxiliaryFieldTopology::Trace,
    regularization::AuxiliaryFieldOwnership::OperatorOwned,
    provenance,
    kTraceIdentity,
    TraceCapabilities()};
  regularization::RegularizationLaunchBuffers buffers{
    points,
    dimension,
    components,
    &state,
    auxiliary,
    trace,
    &state_gradient,
    &normals,
    &coefficient,
    &source,
    regularization::BindVolumeFluxCorrection(
      volume, components, dimension, points),
    regularization::BindStateVectorCorrection(normal, components, points),
    regularization::BindStateVectorCorrection(wall, components, points),
    &bounds,
    &diagnostics};

  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  regularization::RegularizationLaunchBuffers input_alias = buffers;
  input_alias.coefficient = &state;
  regularization::RegularizationLaunchResult result =
    launcher.Launch(ValidationContext(), {}, input_alias);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidBufferAliasing &&
               !result.launched && launcher.LaunchCount() == 0,
             "one-point writable-output/input alias fails before launch");

  regularization::RegularizationLaunchBuffers output_alias = buffers;
  output_alias.source = &coefficient;
  result = launcher.Launch(ValidationContext(), {}, output_alias);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidBufferAliasing &&
               !result.launched && launcher.LaunchCount() == 0,
             "one-point writable-output/output alias fails before launch");
  test.Near(state.HostRead()[0],
            static_cast<mfem::real_t>(4.0),
            0.0,
            "alias rejection leaves the input object untouched");
}

void TestValueKindValidation(TestContext& test)
{
  FixtureBuffers fixture;
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ConstantCpuSelection());

  regularization::RegularizationLaunchBuffers malformed =
    fixture.LaunchBuffers();
  malformed.auxiliary.value_kind =
    static_cast<regularization::AuxiliaryValueKind>(255);
  regularization::RegularizationLaunchResult result =
    launcher.Launch(ValidationContext(), {}, malformed);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidFieldValueKind &&
               !result.launched,
             "unknown auxiliary value kind fails before launch");

  malformed = fixture.LaunchBuffers();
  malformed.trace.value_kind = regularization::AuxiliaryValueKind::Vector;
  result = launcher.Launch(ValidationContext(), {}, malformed);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidComponentCount &&
               !result.launched,
             "trace component extent must match its vector value kind");

  malformed = fixture.LaunchBuffers();
  malformed.auxiliary.value_kind = regularization::AuxiliaryValueKind::Tensor;
  malformed.auxiliary.components = 3;
  result = launcher.Launch(ValidationContext(), {}, malformed);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidComponentCount &&
               !result.launched && launcher.LaunchCount() == 0,
             "tensor component extent must equal checked dimension square");
}

void TestIdentityCollisions(TestContext& test)
{
  FixtureBuffers fixture;
  fixture.ResetOutputs(29.0);
  const regularization::AuxiliaryFieldIdentity collisions[] = {
    {kScalarIdentity.semantic_id,
     kTraceIdentity.restart_id,
     kTraceIdentity.output_id},
    {kTraceIdentity.semantic_id,
     kScalarIdentity.restart_id,
     kTraceIdentity.output_id},
    {kTraceIdentity.semantic_id,
     kTraceIdentity.restart_id,
     kScalarIdentity.output_id}};
  const char* messages[] = {"semantic identity collision fails before launch",
                            "restart identity collision fails before launch",
                            "output identity collision fails before launch"};

  for (int collision = 0; collision < 3; ++collision)
  {
    regularization::RegularizationLaunchBuffers buffers =
      fixture.LaunchBuffers();
    buffers.trace.identity = collisions[collision];
    const regularization::RegularizationLaunchResult result =
      regularization::ExecuteRegularization(ConstantCpuSelection(),
                                            ValidationContext(),
                                            {},
                                            buffers);
    test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidFieldIdentity &&
                 !result.launched && result.host_selection_count == 1,
               messages[collision]);
    test.Near(fixture.volume.HostRead()[0],
              29.0,
              0.0,
              "identity collision leaves outputs untouched");
  }
}

void TestValidationProductOverflow(TestContext& test)
{
  FixtureBuffers fixture;
  const int point_counts[] = {std::numeric_limits<int>::max(),
                              std::numeric_limits<int>::max() / 2};
  const char* messages[] = {
    "characteristic-bound size overflow fails before buffer access",
    "diagnostic size overflow fails before buffer access"};

  for (int scenario = 0; scenario < 2; ++scenario)
  {
    regularization::RegularizationLaunchBuffers buffers =
      fixture.LaunchBuffers();
    buffers.point_count = point_counts[scenario];
    buffers.dimension = 1;
    buffers.state_components = 1;
    buffers.auxiliary.components = 1;
    buffers.trace.components = 1;
    buffers.volume_correction.components = 1;
    buffers.volume_correction.dimension = 1;
    buffers.volume_correction.point_count = point_counts[scenario];
    buffers.normal_correction.components = 1;
    buffers.normal_correction.point_count = point_counts[scenario];
    buffers.wall_flux.components = 1;
    buffers.wall_flux.point_count = point_counts[scenario];

    const regularization::RegularizationLaunchResult result =
      regularization::ExecuteRegularization(ConstantCpuSelection(),
                                            ValidationContext(),
                                            {},
                                            buffers);
    test.Check(result.validation.code ==
                 regularization::ValidationCode::InvalidComponentCount &&
                 !result.launched && result.host_selection_count == 1,
               messages[scenario]);
  }
}

void TestStaleProvenanceAndShape(TestContext& test)
{
  FixtureBuffers fixture;
  fixture.ResetOutputs(37.0);
  regularization::ConfiguredRegularizationLauncher launcher =
    regularization::ConfigureRegularization(ConstantCpuSelection());
  regularization::RegularizationLaunchResult result = launcher.Launch(
    regularization::RegularizationValidationContext{},
    {},
    fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidProvenanceStamp &&
               !result.launched && launcher.LaunchCount() == 0,
             "default provenance context fails before launch");
  test.Near(fixture.source.HostRead()[0],
            static_cast<mfem::real_t>(37.0),
            0.0,
            "unstamped provenance leaves outputs untouched");

  regularization::RegularizationValidationContext stale = ValidationContext();
  stale.expected_state_generation = 20u;
  result = launcher.Launch(stale, {}, fixture.LaunchBuffers());
  test.Check(result.validation.code ==
               regularization::ValidationCode::StaleProvenance &&
               !result.launched && launcher.LaunchCount() == 0,
             "stale state-generation provenance fails before launch");

  regularization::RegularizationLaunchBuffers wrong_extent =
    fixture.LaunchBuffers();
  wrong_extent.volume_correction.components = 3;
  wrong_extent.normal_correction.components = 3;
  wrong_extent.wall_flux.components = 3;
  wrong_extent.volume_correction.storage = nullptr;
  wrong_extent.normal_correction.storage = nullptr;
  wrong_extent.wall_flux.storage = nullptr;
  result = regularization::ExecuteRegularization(ConstantCpuSelection(),
                                                 ValidationContext(),
                                                 {},
                                                 wrong_extent);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidBufferSize &&
               !result.launched && result.host_selection_count == 1,
             "matching correction extents that differ from state fail before buffer access");

  regularization::RegularizationLaunchBuffers wrong_shape =
    fixture.LaunchBuffers();
  wrong_shape.normal_correction.components = 3;
  result = regularization::ExecuteRegularization(ConstantCpuSelection(),
                                                 ValidationContext(),
                                                 {},
                                                 wrong_shape);
  test.Check(result.validation.code ==
               regularization::ValidationCode::InvalidBufferSize &&
               !result.launched,
             "mismatched system correction shape fails before launch");
}

} // namespace

int main()
{
  mfem::Device device("cpu");
  TestContext test;
  TestTypedDescriptors(test);
  TestSystemShapedConstantCorrection(test);
  TestConfiguredLauncherReuse(test);
  TestAllSpatialBackends(test);
  TestVectorAuxiliaryBinding(test);
  TestZeroBehavior(test);
  TestSupportMatrixAndFailures(test);
  TestOnePointBufferAliasing(test);
  TestValueKindValidation(test);
  TestIdentityCollisions(test);
  TestValidationProductOverflow(test);
  TestStaleProvenanceAndShape(test);

  if (test.Failures() == 0)
  {
    std::cout << "[PASS] static MFEM regularization contract fixtures\n";
  }
  return test.Failures() == 0 ? 0 : 1;
}
